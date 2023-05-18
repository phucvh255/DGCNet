import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy import misc
from mymodel import DualSeg_res50
from mydataloader import test_dataset
from utils import compute_iou,compute_dice
import cv2
from utils import *
parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--pth_path', type=str, default='../best_teachermodel.pth')

for _data_name in ['CVC-ClinicDB','CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:
    data_path = 'GALD-DGCNet/TestDataset/{}'.format(_data_name)
    # save_path = './results/PraNet/{}/'.format(_data_name)
    opt = parser.parse_args()
    model = DualSeg_res50(2)
    model.load_state_dict(torch.load(opt.pth_path))
    model_teacher = EMA(model, 0.99)
    model_teacher.model.cuda()
    model_teacher.model.eval()

    # os.makedirs(save_path, exist_ok=True)
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    test_loader = test_dataset(image_root, gt_root, opt.testsize)
    all_iou = []
    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data(i)
        gt = gt.squeeze(0)
        # gt = np.asarray(gt, np.float32)
        # gt /= (gt.max() + 1e-8)
        image = image.cuda()
        image = image.unsqueeze(0)

        pred = model_teacher.model(image)
        pred = pred.squeeze(0)
        pred = torch.softmax(pred, dim = 0)
        pred = torch.argmax(pred, dim = 0).cpu().numpy().astype(np.float32)
        pred *= 255.
        cv2.imwrite("GALD-DGCNet/results_test/"+name, pred)
        # break
    break
        