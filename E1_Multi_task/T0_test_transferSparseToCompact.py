#!/usr/bin/env python3
import torch.utils.data.sampler as sampler
from E1_Multi_task.create_dataset import *
from E1_Multi_task.utils import draw_color
from E1_Multi_task.models.SparseSegnet import SegNet
from E1_Multi_task.models.Segnet import SegNet_compact
import argparse
from E1_Multi_task.Train_utils import *
from E1_Multi_task.DR_MGFutils import *
parser = argparse.ArgumentParser(description='Multi-task: Split')
parser.add_argument('--type', default='standard', type=str, help='split type: standard, wide, deep')
parser.add_argument('--weight', default='equal', type=str, help='multi-task weighting: equal, uncert, dwa')
# parser.add_argument('--dataroot', default='/media/sunyi/E/nyuv2', type=str, help='dataset root')
parser.add_argument('--dataroot', default='/home/user/nyuv2', type=str, help='dataset root')
parser.add_argument('--method', default='mgd', type=str, help='optimization method')
parser.add_argument('--temp', default=2.0, type=float, help='temperature for DWA (must be positive)')
parser.add_argument('--alpha', default=0.5, type=float, help='the alpha')
parser.add_argument('--lr', default=1e-4, type=float, help='the learning rate')
parser.add_argument('--seed', default=0, type=int, help='the seed')
parser.add_argument('--apply_augmentation', action='store_true', help='toggle to apply data augmentation on NYUv2')
parser.add_argument('--comparedbaseline',default='losslist.npy')
parser.add_argument('--save',default='./tmpsave/',type=str)
parser.add_argument('--device',default='cuda:4',type=str)
parser.add_argument('--metalr',default=0.1,type=float)
parser.add_argument('--ema',default=0.9,type=float)
parser.add_argument('--auxlr',default=0.1,type=float)
parser.add_argument('--MetaGFstartEpoch',default=50,type=int)
parser.add_argument('--usinglossscale',default=0,type=int)
opt = parser.parse_args()
color = ['cyan', 'green', 'blue', 'olive', 'black', 'yellow', 'red']



TASKNUM=3
device="cuda:0"
def main():
    global args
    # ['vgg16', 'resnet56', 'wideresnet32_4', 'mobilenet']
    savepath="PATH/model.pth"
    SegNet_MTAN = SegNet().cuda()
    SegNet_MTANcompact=SegNet_compact().cuda()

    #
    # for n,p in SegNet_MTAN.named_parameters():
    #     print("n:{} p:{}".format(n,p.shape))
    #
    # for n, p in SegNet_MTANcompact.named_parameters():
    #     print("n:{} p:{}".format(n, p.shape))

    try:
        state_dict = torch.load(savepath,map_location="cuda:0")['state_dict']
    except:
        state_dict = torch.load(savepath,map_location="cuda:0",encoding='ascii')
        print("here")
    SegNet_MTAN.load_state_dict(state_dict)
    if opt.apply_augmentation:
        nyuv2_train_set = NYUv2(root=opt.dataroot, train=True, augmentation=True)
        print('Applying data augmentation on NYUv2.')
    else:
        nyuv2_train_set = NYUv2(root=opt.dataroot, train=True)
        print('Standard training strategy without data augmentation.')
    nyuv2_test_set = NYUv2(root= opt.dataroot, train=False)

    batch_size = 2
    nyuv2_train_loader = torch.utils.data.DataLoader(
        dataset=nyuv2_train_set,
        batch_size=batch_size,
        shuffle=True)
    nyuv2_test_loader = torch.utils.data.DataLoader(
        dataset=nyuv2_test_set,
        batch_size=batch_size,
        shuffle=False)

    # Avg_loss, Avg_loss_train = eval(nyuv2_test_loader, nyuv2_train_loader, SegNet_MTAN, device, 0)
    # print(Avg_loss)


    namelist=[]
    for name in SegNet_MTAN.state_dict():
        namelist.append(name)

    compactnamelist=[]
    for name in SegNet_MTANcompact.state_dict():
        compactnamelist.append(name)
    assert (len(compactnamelist)==len(namelist))
    ####transfering
    from collections import OrderedDict
    state_dict_compact=OrderedDict()
    for i in range(0,len(namelist)):
        state_dict_compact[compactnamelist[i]]=SegNet_MTAN.state_dict()[namelist[i]]

    SegNet_MTANcompact.load_state_dict(state_dict_compact)
    Avg_loss, Avg_loss_train = eval(nyuv2_test_loader, nyuv2_train_loader, SegNet_MTANcompact, device, 0)
    print(Avg_loss)


def eval(test_loader, train_loader, multi_task_model, device, globalepoch):
    # evaluating test data
    multi_task_model.eval()
    conf_mat = ConfMatrix(multi_task_model.class_nb)
    Avg_cost = AverageMeter()
    Avg_cost_train = AverageMeter()
    Avg_loss = []
    Avg_loss_train = []
    for i in range(0, TASKNUM):
        Avg_loss.append(AverageMeter())
        Avg_loss_train.append(AverageMeter())
    starttime = time.time()
    with torch.no_grad():  # operations inside don't track history
        test_batch = len(test_loader)
        test_dataset = iter(test_loader)
        for k in range(test_batch):
            cost = np.zeros(24, dtype=np.float32)
            test_data, test_label, test_depth, test_normal = test_dataset.next()
            test_data, test_label = test_data.to(device), test_label.long().to(device)
            test_depth, test_normal = test_depth.to(device), test_normal.to(device)

            test_pred, _ = multi_task_model(test_data)
            test_loss = [model_fit(test_pred[0], test_label, 'semantic'),
                         model_fit(test_pred[1], test_depth, 'depth'),
                         model_fit(test_pred[2], test_normal, 'normal')]

            for i in range(0, TASKNUM):
                Avg_loss[i].update(test_loss[i].item())
            conf_mat.update(test_pred[0].argmax(1).flatten(), test_label.flatten())

            cost[0] = test_loss[0].item()
            cost[3] = test_loss[1].item()
            cost[4], cost[5] = depth_error(test_pred[1], test_depth)
            cost[6] = test_loss[2].item()
            cost[7], cost[8], cost[9], cost[10], cost[11] = normal_error(test_pred[2], test_normal)
            cost[:12] += cost[:12] / test_batch
            Avg_cost.update(cost)

        # compute mIoU and acc
        Avg_cost.avg[1:3] = conf_mat.get_metrics()

    ##################displaying the prediction#############
    with torch.no_grad():
        plt.figure(2)
        plt.subplot(1, 4, 1)
        plt.imshow(test_data[0, :, :, :].cpu().numpy().transpose(1, 2, 0))
        plt.subplot(1, 4, 2)
        plt.imshow(test_pred[1][0, :, :, :].cpu().numpy().transpose(1, 2, 0))
        plt.title("depth")
        plt.subplot(1, 4, 3)
        plt.imshow(test_pred[2][0, :, :, :].cpu().numpy().transpose(1, 2, 0))
        plt.title("normal")
        plt.subplot(1, 4, 4)

        semanticmap = test_pred[0].argmax(1)
        semanticmap = semanticmap[0, :, :].cpu().numpy()
        semanticmap = draw_color(semanticmap)
        plt.imshow(semanticmap)
        # plt.savefig(opt.save + '/{}.png'.format(globalepoch))
        plt.pause(0.01)

    conf_mat = ConfMatrix(multi_task_model.class_nb)
    with torch.no_grad():  # operations inside don't track history
        test_dataset = iter(train_loader)
        test_batch = len(train_loader)
        for k in range(test_batch):
            cost = np.zeros(24, dtype=np.float32)
            test_data, test_label, test_depth, test_normal = test_dataset.next()
            test_data, test_label = test_data.to(device), test_label.long().to(device)
            test_depth, test_normal = test_depth.to(device), test_normal.to(device)

            test_pred, _ = multi_task_model(test_data)
            test_loss = [model_fit(test_pred[0], test_label, 'semantic'),
                         model_fit(test_pred[1], test_depth, 'depth'),
                         model_fit(test_pred[2], test_normal, 'normal')]

            for i in range(0, TASKNUM):
                Avg_loss_train[i].update(test_loss[i].item())
            conf_mat.update(test_pred[0].argmax(1).flatten(), test_label.flatten())

            cost[0] = test_loss[0].item()
            cost[3] = test_loss[1].item()
            cost[4], cost[5] = depth_error(test_pred[1], test_depth)
            cost[6] = test_loss[2].item()
            cost[7], cost[8], cost[9], cost[10], cost[11] = normal_error(test_pred[2], test_normal)
            cost[:12] += cost[:12] / test_batch
            Avg_cost_train.update(cost)

        # compute mIoU and acc
        Avg_cost_train.avg[1:3] = conf_mat.get_metrics()

    end_time = time.time()
    print(
        'Epoch: {:04d} | TRAIN: {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} ||'
        'TEST: {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} | {:.4f}'
            .format(globalepoch, Avg_cost_train.avg[0], Avg_cost_train.avg[1], Avg_cost_train.avg[2],
                    Avg_cost_train.avg[3],
                    Avg_cost_train.avg[4], Avg_cost_train.avg[5], Avg_cost_train.avg[6], Avg_cost_train.avg[7],
                    Avg_cost_train.avg[8],
                    Avg_cost_train.avg[9], Avg_cost_train.avg[10], Avg_cost_train.avg[11], Avg_cost.avg[0],
                    Avg_cost.avg[1], Avg_cost.avg[2], Avg_cost.avg[3],
                    Avg_cost.avg[4], Avg_cost.avg[5], Avg_cost.avg[6], Avg_cost.avg[7], Avg_cost.avg[8],
                    Avg_cost.avg[9], Avg_cost.avg[10], Avg_cost.avg[11], end_time - starttime))

    return Avg_loss, Avg_loss_train


if __name__ == '__main__':
    main()
