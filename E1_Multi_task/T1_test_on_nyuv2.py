import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
import torch.utils.data.sampler as sampler

from E1_Multi_task.create_dataset import *
from E1_Multi_task.DR_MGFutils import *
from E1_Multi_task.models.model_segnet_mt_DRMGF import *
parser = argparse.ArgumentParser(description='Multi-task: Split')
parser.add_argument('--type', default='standard', type=str, help='split type: standard, wide, deep')
parser.add_argument('--weight', default='equal', type=str, help='multi-task weighting: equal, uncert, dwa')
parser.add_argument('--dataroot', default='/home/user/nyuv2', type=str, help='dataset root')
parser.add_argument('--method', default='mgd', type=str, help='optimization method')
parser.add_argument('--temp', default=2.0, type=float, help='temperature for DWA (must be positive)')
parser.add_argument('--alpha', default=0.5, type=float, help='the alpha')
parser.add_argument('--lr', default=1e-4, type=float, help='the learning rate')
parser.add_argument('--seed', default=0, type=int, help='the seed')
parser.add_argument('--apply_augmentation', action='store_true', help='toggle to apply data augmentation on NYUv2')
parser.add_argument('--comparedbaseline',default='losslist.npy')
parser.add_argument('--save',default='./tmpsave/',type=str)
opt = parser.parse_args()
color=['cyan','green','blue','olive','black','yellow','red']



def eval(train_loader,test_loader,multi_task_model, device,total_epoch=1):
    start_time = time.time()
    train_batch = len(train_loader)
    test_batch = len(test_loader)
    T = opt.temp
    avg_cost = np.zeros([total_epoch, 24], dtype=np.float32)
    lambda_weight = np.ones([3, total_epoch])
    for index in range(total_epoch):
        epoch_start_time = time.time()
        cost = np.zeros(24, dtype=np.float32)

        # evaluating test data
        multi_task_model.eval()
        conf_mat = ConfMatrix(multi_task_model.class_nb)
        with torch.no_grad():  # operations inside don't track history
            test_dataset = iter(test_loader)
            for k in range(test_batch):
                print("{}/{}".format(k,test_batch))
                test_data, test_label, test_depth, test_normal = test_dataset.next()
                test_data, test_label = test_data.to(device), test_label.long().to(device)
                test_depth, test_normal = test_depth.to(device), test_normal.to(device)

                test_pred, _ = multi_task_model(test_data)
                test_loss = [model_fit(test_pred[0], test_label, 'semantic'),
                             model_fit(test_pred[1], test_depth, 'depth'),
                             model_fit(test_pred[2], test_normal, 'normal')]

                conf_mat.update(test_pred[0].argmax(1).flatten(), test_label.flatten())
                # avg_cost[index, 13:15] = conf_mat.get_metrics()
                cost[12] = test_loss[0].item()
                cost[15] = test_loss[1].item()
                cost[16], cost[17] = depth_error(test_pred[1], test_depth)
                cost[18] = test_loss[2].item()
                cost[19], cost[20], cost[21], cost[22], cost[23] = normal_error(test_pred[2], test_normal)
                avg_cost[index, 12:] += cost[12:] / test_batch

            # compute mIoU and acc
            avg_cost[index, 13:15] = conf_mat.get_metrics()

        epoch_end_time = time.time()
        print(
            'Epoch: {:04d} | TRAIN: {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} ||'
            'TEST: {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} | {:.4f}'
            .format(index,avg_cost[index, 0], avg_cost[index, 1], avg_cost[index, 2], avg_cost[index, 3],
                    avg_cost[index, 4], avg_cost[index, 5], avg_cost[index, 6], avg_cost[index, 7], avg_cost[index, 8],
                    avg_cost[index, 9], avg_cost[index, 10], avg_cost[index, 11], avg_cost[index, 12],
                    avg_cost[index, 13],
                    avg_cost[index, 14], avg_cost[index, 15], avg_cost[index, 16], avg_cost[index, 17],
                    avg_cost[index, 18],
                    avg_cost[index, 19], avg_cost[index, 20], avg_cost[index, 21], avg_cost[index, 22],
                    avg_cost[index, 23], epoch_end_time - epoch_start_time))
    end_time = time.time()
    print("Training time: ", end_time - start_time)

    return Avg_loss,Avg_loss_train


TASKNUM=3
AUXILIARY=0.1
testmodelpath="/home/user/DRMGF_1115/MetaGF_TPAMI/E1_Multi_task/models/tmpsave/Metalr0.1_ema_0.5_aux_0.42022-11-17-2-4-51/model.pth"
if __name__ == '__main__':
    # control seed
    torch.backends.cudnn.enabled = False
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    torch.cuda.set_device(0)
    # define model, optimiser and scheduler
    device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
    SegNet_MTAN = SegNet().to(device)

    tmpmodel=torch.load(testmodelpath,map_location=device)
    SegNet_MTAN.load_state_dict(tmpmodel)
    print('Parameter Space: ABS: {:.1f}, REL: {:.4f}'.format(count_parameters(SegNet_MTAN),
                                                             count_parameters(SegNet_MTAN) / 24981069))
    print('LOSS FORMAT: SEMANTIC_LOSS MEAN_IOU PIX_ACC | DEPTH_LOSS ABS_ERR REL_ERR | NORMAL_LOSS MEAN MED <11.25 <22.5 <30')

    # define dataset
    dataset_path = opt.dataroot
    if opt.apply_augmentation:
        nyuv2_train_set = NYUv2(root=dataset_path, train=True, augmentation=True)
        print('Applying data augmentation on NYUv2.')
    else:
        nyuv2_train_set = NYUv2(root=dataset_path, train=True)
        print('Standard training strategy without data augmentation.')

    nyuv2_test_set = NYUv2(root=dataset_path, train=False)

    batch_size = 2
    nyuv2_train_loader = torch.utils.data.DataLoader(
        dataset=nyuv2_train_set,
        batch_size=batch_size,
        shuffle=True)

    nyuv2_test_loader = torch.utils.data.DataLoader(
        dataset=nyuv2_test_set,
        batch_size=batch_size,
        shuffle=False)

    eval(nyuv2_train_loader, nyuv2_test_loader, SegNet_MTAN, device, total_epoch=1)



