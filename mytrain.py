from ast import arg
from cProfile import label
import torch
import torchvision
from transformers import PegasusPreTrainedModel
from mydataloader import get_loader, test_dataset
from libs.models.DualGCNNet import DualGCNHead
from mymodel import DualSeg_res50, DualSeg_res101
import cv2
import random
import numpy as np
import torch.nn as nn
from libs.core.loss import CriterionDSN
import argparse
import os
import os.path as osp
import timeit
import numpy as np
from torch.optim.lr_scheduler import StepLR
from utils import compute_iou
import torch
from torch.utils import data
import torch.optim as optim
import torch.backends.cudnn as cudnn

from libs.utils.logger import Logger as Log
from libs.utils.tools import adjust_learning_rate, all_reduce_tensor
from libs.datasets.cityscapes import Cityscapes
import torch.nn.functional as F
from libs.core.loss import CriterionOhemDSN, CriterionDSN
from utils import *
from tqdm import tqdm
img_TrainPath = "GALD-DGCNet/TrainDataset/image/"
label_TrainPath = "GALD-DGCNet/TrainDataset/mask/"
img_cvc300TestPath = "GALD-DGCNet/TestDataset/CVC-300/images/"
label_cvc300TestPath = "GALD-DGCNet/TestDataset/CVC-300/masks/"
EPOCH = 50
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def main(args):
    # Log.init(
    #     log_level=args.log_level,
    #     log_file=osp.join(args.save_dir, args.log_file),
    #     log_format=args.log_format,
    #     rewrite=args.rewrite,
    #     stdout_level=args.stdout_level
    # )
    #dataloader 
    device = "cuda:0"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    train_l_loader, train_u_loader, val_loader = get_loader(img_TrainPath, label_TrainPath, args.batch_size, args.input_size)
    # val_loader = test_dataset(img_cvc300TestPath, label_cvc300TestPath, args.input_size)
    #model
    model = DualSeg_res50(2).cuda()
    model_teacher = EMA(model, 0.99)
    model_teacher.model.cuda()
    model.float()
    model.cuda()
    #loss function
    criterion = CriterionDSN()
    #set optimizer
    optimizer = optim.SGD(
        [{'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': args.learning_rate}],
        lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer.zero_grad()
    start = timeit.default_timer()
    best_loss = 1e+10
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    
    total_epoch = 200
    train_epoch = len(train_l_loader)
    test_epoch = len(val_loader)
    avg_cost = np.zeros((total_epoch, 10))
    iteration = 0
    for index in range(total_epoch):
        model_teacher.model.train()
        model.train()
        l_conf_mat = ConfMatrix(2)
        u_conf_mat = ConfMatrix(2)
        cost = np.zeros(3)
        train_l_dataset = iter(train_l_loader)
        train_u_dataset = iter(train_u_loader)
        for i in tqdm(range(train_epoch)):
            # break
            train_l_data, train_l_label = train_l_dataset.next()
            train_l_data, train_l_label = train_l_data.to(device), train_l_label.to(device)

            train_u_data, train_u_label = train_u_dataset.next()
            train_u_data, train_u_label = train_u_data.to(device), train_u_label.to(device)

            optimizer.zero_grad()
            # labels = labels.squeeze(1).long().cuda()
            with torch.no_grad():
                pred_u, _ = model_teacher.model(train_u_data)
                pred_u_large_raw = F.interpolate(pred_u, size=train_u_label.shape[1:], mode='bilinear', align_corners=True)
              
                pseudo_logits, pseudo_labels = torch.max(torch.softmax(pred_u_large_raw, dim=1), dim=1)

                # random scale images first
                train_u_aug_data, train_u_aug_label, train_u_aug_logits = \
                    batch_transform(train_u_data, pseudo_labels, pseudo_logits,
                                    (args.input_size, args.input_size), (1.0,1.0), apply_augmentation=False)

                # apply mixing strategy: cutout, cutmix or classmix
                train_u_aug_data, train_u_aug_label, train_u_aug_logits = \
                    generate_unsup_data(train_u_aug_data, train_u_aug_label, train_u_aug_logits, mode=args.apply_aug)

                # apply augmentation: color jitter + flip + gaussian blur
                train_u_aug_data, train_u_aug_label, train_u_aug_logits = \
                    batch_transform(train_u_aug_data, train_u_aug_label, train_u_aug_logits,
                                    (args.input_size, args.input_size), (1.0, 1.0), apply_augmentation=True)
            # generate labelled and unlabelled data loss
            pred_l, rep_l = model(train_l_data)
            pred_l_large = F.interpolate(pred_l, size=train_l_label.shape[1:], mode='bilinear', align_corners=True)

            pred_u, rep_u = model(train_u_aug_data)
            pred_u_large = F.interpolate(pred_u, size=train_l_label.shape[1:], mode='bilinear', align_corners=True)

            rep_all = torch.cat((rep_l, rep_u))
            pred_all = torch.cat((pred_l, pred_u))

            # supervised-learning loss
            sup_loss = compute_supervised_loss(pred_l_large, train_l_label)
            # unsupervised-learning loss

            unsup_loss = compute_unsupervised_loss(pred_u_large, train_u_aug_label, train_u_aug_logits, args.strong_threshold)
        
            # apply regional contrastive loss
            if args.apply_reco:
                with torch.no_grad():
                    train_u_aug_mask = train_u_aug_logits.ge(args.weak_threshold).float()
                    mask_all = torch.cat(((train_l_label.unsqueeze(1) >= 0).float(), train_u_aug_mask.unsqueeze(1)))
                    mask_all = F.interpolate(mask_all, size=pred_all.shape[2:], mode='nearest')

                    label_l = F.interpolate(label_onehot(train_l_label, 2), size=pred_all.shape[2:], mode='nearest')
                    label_u = F.interpolate(label_onehot(train_u_aug_label, 2), size=pred_all.shape[2:], mode='nearest')
                    label_all = torch.cat((label_l, label_u))

                    prob_l = torch.softmax(pred_l, dim=1)
                    prob_u = torch.softmax(pred_u, dim=1)
                    prob_all = torch.cat((prob_l, prob_u))

                reco_loss = compute_reco_loss(rep_all, label_all, mask_all, prob_all, args.strong_threshold,
                                            args.temp, args.num_queries, args.num_negatives)
            else:
                reco_loss = torch.tensor(0.0)
            loss = sup_loss + unsup_loss + reco_loss
            loss.backward()
            optimizer.step()
            model_teacher.update(model)

            l_conf_mat.update(pred_l_large.argmax(1).flatten(), train_l_label.flatten())
            u_conf_mat.update(pred_u_large_raw.argmax(1).flatten(), train_u_label.flatten())

            cost[0] = sup_loss.item()
            cost[1] = unsup_loss.item()
            cost[2] = reco_loss.item()
            avg_cost[index, :3] += cost / train_epoch
            iteration += 1
        avg_cost[index, 3:5] = l_conf_mat.get_metrics()
        avg_cost[index, 5:7] = u_conf_mat.get_metrics()
        with torch.no_grad():
            model_teacher.model.eval()
            test_dataset = iter(val_loader)
            conf_mat = ConfMatrix(2)
            for i in tqdm(range(test_epoch)):
                # test_data, test_label, name = val_loader.load_data(i)
                # test_label = test_label.squeeze(0)
                # test_label = test_label.long()
                # test_data, test_label = test_dataset.next()
                test_data, test_label = test_dataset.next()
                test_data, test_label = test_data.to(device), test_label.to(device)

                pred = model_teacher.model(test_data)
                pred = F.interpolate(pred, size=test_label.shape[1:], mode='bilinear', align_corners=True)
                loss = compute_supervised_loss(pred, test_label)

                conf_mat.update(pred.argmax(1).flatten(), test_label.flatten())
                avg_cost[index, 7] += loss.item() / test_epoch

            avg_cost[index, 8:] = conf_mat.get_metrics()
        scheduler.step()
        print('EPOCH: {:04d} ITER: {:04d} | TRAIN [Loss | mIoU | Acc.]: {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} || Test [Loss | mIoU | Acc.]: {:.4f} {:.4f} {:.4f}'
            .format(index, iteration, avg_cost[index][0], avg_cost[index][1], avg_cost[index][2],
                    avg_cost[index][3], avg_cost[index][4], avg_cost[index][5], avg_cost[index][6], avg_cost[index][7], avg_cost[index][8],
                    avg_cost[index][9]))
        print('Top: mIoU {:.4f} Acc {:.4f}'.format(avg_cost[:, 8].max(), avg_cost[:, 9].max()))

        if avg_cost[index][8] >= avg_cost[:, 8].max():
            torch.save(model_teacher.model.state_dict(),"best_teachermodel.pth")
            # if args.apply_reco:
            #     torch.save(model_teacher.model.state_dict(), 'model_weights/{}_label{}_semi_{}_reco_{}.pth'.format(args.dataset, args.num_labels, args.apply_aug, args.seed))
            # else:
            #     torch.save(model_teacher.model.state_dict(), 'model_weights/{}_label{}_semi_{}_{}.pth'.format(args.dataset, args.num_labels, args.apply_aug, args.seed))

        # if args.apply_reco:
        #     np.save('logging/{}_label{}_semi_{}_reco_{}.npy'.format(args.dataset, args.num_labels, args.apply_aug, args.seed), avg_cost)
        # else:
        #     np.save('logging/{}_label{}_semi_{}_{}.npy'.format(args.dataset, args.num_labels, args.apply_aug, args.seed), avg_cost)


    #         loss = criterion(preds, labels)
    #         loss.backward()
    #         optimizer.step()
    #         loss_epoch.append(loss.item())
    #         if iter % 50 == 0:
    #             print('epoch = {} iter = {} of {} completed, loss = {}'.format(epoch, iter,
    #                                         len(trainloader), loss.data.cpu().numpy()))
    #     scheduler.step()
    #     total_loss = np.mean(loss_epoch)
    #     print('epoch = {} lr={}, loss = {}'.format(epoch, scheduler.get_last_lr(), total_loss))
    #     if  total_loss <= best_loss:
    #         best_loss = total_loss                                  
    #         print('save models ...')
    #         torch.save(model.state_dict(), os.path.join(args.save_dir, str(args.arch)+'_bestmodel.pth'))                                
    # end = timeit.default_timer()

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Number of images sent to the network in one step.")
    parser.add_argument('--gpu_num', type=int, default=8)
    parser.add_argument("--input_size", type=int, default=352 ,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--learning_rate", type=float, default=1e-2,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--num_classes", type=int, default=19,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num_steps", type=int, default=50000,
                        help="Number of training steps.")
    parser.add_argument("--power", type=float, default=0.9,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--random_mirror", action="store_true", default=True,
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random_scale", action="store_true", default=True,
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random_seed", type=int, default=1234,
                        help="Random seed to have reproducible results.")

    # ***** Params for save and load ******
    parser.add_argument("--restore_from", type=str, default="./pretrained",
                        help="Where restore models parameters from.")
    parser.add_argument("--save_pred_every", type=int, default=5000,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Where to save snapshots of the models.")
    parser.add_argument("--arch", type=str, default="fcn_dualgcn",
                        help="Where to save snapshots of the models.")
    parser.add_argument("--save_start",type=int, default=40000)
    parser.add_argument("--gpu", type=str, default=None,
                        help="choose gpu device.")
    parser.add_argument("--ft", type=bool, default=False,
                        help="fine-tune the models with large input size.")
    # ***** Params for logging ***** #
    parser.add_argument('--log_level', default="info", type=str,
                        dest='log_level', help='To set the log level to files.')
    parser.add_argument('--log_file', default="./log/train.log", type=str,
                        dest='log_file', help='The path of log files.')
    parser.add_argument("--log_format", default="%(asctime)s %(levelname)-7s %(message)s", type=str,
                        dest="log_format", help="format of log files"
                        )
    parser.add_argument('--stdout_level', default="info", type=str,
                        dest='stdout_level', help='To set the level to print to screen.')
    parser.add_argument("--rewrite", default=False, type=bool,
                        dest="rewrite", help="whether write the file when using log"
                        )
    parser.add_argument("--local_rank", default=0, type=int, help="parameter used by apex library")
    parser.add_argument('--mode', default=None, type=str)
    parser.add_argument('--port', default=None, type=int)

    parser.add_argument('--num_labels', default=2, type=int, help='number of labelled training data, set 0 to use all training data')
    parser.add_argument('--lr', default=2.5e-3, type=float)

    parser.add_argument('--dataset', default='cityscapes', type=str, help='pascal, cityscapes, sun')
    parser.add_argument('--apply_aug', default='cutout', type=str, help='apply semi-supervised method: cutout cutmix classmix')
    parser.add_argument('--id', default=1, type=int, help='number of repeated samples')
    parser.add_argument('--weak_threshold', default=0.7, type=float)
    parser.add_argument('--strong_threshold', default=0.97, type=float)
    parser.add_argument('--apply_reco', action='store_true')
    parser.add_argument('--num_negatives', default=512, type=int, help='number of negative keys')
    parser.add_argument('--num_queries', default=256, type=int, help='number of queries per segment per image')
    parser.add_argument('--temp', default=0.5, type=float)
    parser.add_argument('--output_dim', default=256, type=int, help='output dimension from representation head')
    parser.add_argument('--backbone', default='deeplabv3p', type=str, help='choose backbone: deeplabv3p, deeplabv2')
    parser.add_argument('--seed', default=0, type=int)
    args = parser.parse_args()
    main(args)