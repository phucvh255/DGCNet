import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy import misc
from mymodel import DualSeg_res50, DualSeg_res101
from mydataloader import test_dataset
from utils import compute_iou,compute_dice
from utils import *
parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--pth_path', type=str, default='GALD-DGCNet/best_teachermodel.pth')

for _data_name in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:
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

    conf_mat = ConfMatrix(2)
    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data(i)
        # gt = gt.squeeze(0)
        # gt = np.asarray(gt, np.float32)
        # gt /= (gt.max() + 1e-8)
        image = image.cuda()
        gt = gt.cuda().squeeze(0)
        image = image.unsqueeze(0)


        pred = model_teacher.model(image)
        # pred = F.interpolate(pred, size=gt.shape, mode='bilinear', align_corners=True)
        # conf_mat.update(pred.argmax(1).flatten(), gt.flatten())
        pred = pred.squeeze(0)
        pred = torch.softmax(pred, dim = 0)
        pred = torch.argmax(pred, dim = 0)
        # iou = compute_iou(pred.cpu().detach().numpy(),gt.cpu().detach().numpy().astype(np.int64))
        iou = compute_iou(pred.cpu(),gt.int().cpu())
        # dice = compute_dice(pred.cpu(),gt.int().cpu())
        all_iou.append(iou)
    # print(conf_mat.get_metrics())
    print('{} : {}'.format(_data_name,np.mean(all_iou)))
        