import sys

import numpy as np
import random
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
sys.path.insert(1,"../../")
sys.path.insert(1,"../../sytools/")
import argparse
import torch.utils.data.sampler as sampler
from E1_Multi_task.create_dataset import *
from E1_Multi_task.DR_MGFutils import *
from sytools.Sparse_conv_v2 import *
from G0_Gradient_tools.MetaGFgrad_ECCV_multitask import *
from sytools.Train_utils import *
parser = argparse.ArgumentParser(description='Multi-task: Split')
parser.add_argument('--type', default='standard', type=str, help='split type: standard, wide, deep')
parser.add_argument('--weight', default='equal', type=str, help='multi-task weighting: equal, uncert, dwa')
parser.add_argument('--dataroot', default='../../data/nyuv2', type=str, help='dataset root')
parser.add_argument('--method', default='mgd', type=str, help='optimization method')
parser.add_argument('--temp', default=2.0, type=float, help='temperature for DWA (must be positive)')
parser.add_argument('--alpha', default=0.5, type=float, help='the alpha')
parser.add_argument('--lr', default=1e-4, type=float, help='the learning rate')
parser.add_argument('--seed', default=0, type=int, help='the seed')
parser.add_argument('--apply_augmentation', action='store_true', help='toggle to apply data augmentation on NYUv2')
parser.add_argument('--comparedbaseline',default='losslist.npy')
parser.add_argument('--save',default='./tmpsave/',type=str)
parser.add_argument('--device',default='cuda:0',type=str)
parser.add_argument('--metalr',default=0.1,type=float)
parser.add_argument('--ema',default=0.5,type=float)
parser.add_argument('--auxlr',default=0.1,type=float)

expgroup=parser.add_argument_group()

opt = parser.parse_args()
color=['cyan','green','blue','olive','black','yellow','red']
class SegNet(nn.Module):
    def __init__(self):
        super(SegNet, self).__init__()
        # initialise network parameters
        filter = [64, 128, 256, 512, 512]
        self.class_nb = 13

        # define encoder decoder layers
        self.encoder_block = nn.ModuleList([self.conv_layer([3, filter[0]])])
        self.decoder_block = nn.ModuleList([self.conv_layer([filter[0], filter[0]])])
        for i in range(4):
            self.encoder_block.append(self.conv_layer([filter[i], filter[i + 1]]))
            self.decoder_block.append(self.conv_layer([filter[i + 1], filter[i]]))

        # define convolution layer
        self.conv_block_enc = nn.ModuleList([self.conv_layer([filter[0], filter[0]])])
        self.conv_block_dec = nn.ModuleList([self.conv_layer([filter[0], filter[0]])])
        for i in range(4):
            if i == 0:
                self.conv_block_enc.append(self.conv_layer([filter[i + 1], filter[i + 1]]))
                self.conv_block_dec.append(self.conv_layer([filter[i], filter[i]]))
            else:
                self.conv_block_enc.append(nn.Sequential(self.conv_layer([filter[i + 1], filter[i + 1]]),
                                                         self.conv_layer([filter[i + 1], filter[i + 1]])))
                self.conv_block_dec.append(nn.Sequential(self.conv_layer([filter[i], filter[i]]),
                                                         self.conv_layer([filter[i], filter[i]])))

        # define task attention layers
        self.encoder_att = nn.ModuleList([nn.ModuleList([self.att_layer([filter[0], filter[0], filter[0]])])])
        self.decoder_att = nn.ModuleList([nn.ModuleList([self.att_layer([2 * filter[0], filter[0], filter[0]])])])
        self.encoder_block_att = nn.ModuleList([self.conv_layer([filter[0], filter[1]])])
        self.decoder_block_att = nn.ModuleList([self.conv_layer([filter[0], filter[0]])])

        for j in range(3):
            if j < 2:
                self.encoder_att.append(nn.ModuleList([self.att_layer([filter[0], filter[0], filter[0]])]))
                self.decoder_att.append(nn.ModuleList([self.att_layer([2 * filter[0], filter[0], filter[0]])]))
            for i in range(4):
                self.encoder_att[j].append(self.att_layer([2 * filter[i + 1], filter[i + 1], filter[i + 1]]))
                self.decoder_att[j].append(self.att_layer([filter[i + 1] + filter[i], filter[i], filter[i]]))

        for i in range(4):
            if i < 3:
                self.encoder_block_att.append(self.conv_layer([filter[i + 1], filter[i + 2]]))
                self.decoder_block_att.append(self.conv_layer([filter[i + 1], filter[i]]))
            else:
                self.encoder_block_att.append(self.conv_layer([filter[i + 1], filter[i + 1]]))
                self.decoder_block_att.append(self.conv_layer([filter[i + 1], filter[i + 1]]))

        self.pred_task1 = self.conv_layer([filter[0], self.class_nb], pred=True)
        self.pred_task2 = self.conv_layer([filter[0], 1], pred=True)
        self.pred_task3 = self.conv_layer([filter[0], 3], pred=True)

        # define pooling and unpooling functions
        self.down_sampling = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.up_sampling = nn.MaxUnpool2d(kernel_size=2, stride=2)

        self.logsigma = nn.Parameter(torch.FloatTensor([-0.5, -0.5, -0.5]))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def shared_modules(self):
        return [self.encoder_block, self.decoder_block,
                self.conv_block_enc, self.conv_block_dec,
                #self.encoder_att, self.decoder_att,
                self.encoder_block_att, self.decoder_block_att,
                self.down_sampling, self.up_sampling]

    def zero_grad_shared_modules(self):
        for mm in self.shared_modules():
            mm.zero_grad()

    def conv_layer(self, channel, pred=False):
        if not pred:
            conv_block = nn.Sequential(
                nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=3, padding=1),
                nn.BatchNorm2d(num_features=channel[1]),
                nn.ReLU(inplace=True),
            )
        else:
            conv_block = nn.Sequential(
                nn.Conv2d(in_channels=channel[0], out_channels=channel[0], kernel_size=3, padding=1),
                nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=1, padding=0),
            )
        return conv_block

    def att_layer(self, channel):
        att_block = nn.Sequential(
            nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=1, padding=0),
            nn.BatchNorm2d(channel[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channel[1], out_channels=channel[2], kernel_size=1, padding=0),
            nn.BatchNorm2d(channel[2]),
            nn.Sigmoid(),
        )
        return att_block

    def forward(self, x):
        g_encoder, g_decoder, g_maxpool, g_upsampl, indices = ([0] * 5 for _ in range(5))
        for i in range(5):
            g_encoder[i], g_decoder[-i - 1] = ([0] * 2 for _ in range(2))

        # define attention list for tasks
        atten_encoder, atten_decoder = ([0] * 3 for _ in range(2))
        for i in range(3):
            atten_encoder[i], atten_decoder[i] = ([0] * 5 for _ in range(2))
        for i in range(3):
            for j in range(5):
                atten_encoder[i][j], atten_decoder[i][j] = ([0] * 3 for _ in range(2))

        # define global shared network
        for i in range(5):
            if i == 0:
                g_encoder[i][0] = self.encoder_block[i](x)
                g_encoder[i][1] = self.conv_block_enc[i](g_encoder[i][0])
                g_maxpool[i], indices[i] = self.down_sampling(g_encoder[i][1])
            else:
                g_encoder[i][0] = self.encoder_block[i](g_maxpool[i - 1])
                g_encoder[i][1] = self.conv_block_enc[i](g_encoder[i][0])
                g_maxpool[i], indices[i] = self.down_sampling(g_encoder[i][1])

        for i in range(5):
            if i == 0:
                g_upsampl[i] = self.up_sampling(g_maxpool[-1], indices[-i - 1])
                g_decoder[i][0] = self.decoder_block[-i - 1](g_upsampl[i])
                g_decoder[i][1] = self.conv_block_dec[-i - 1](g_decoder[i][0])
            else:
                g_upsampl[i] = self.up_sampling(g_decoder[i - 1][-1], indices[-i - 1])
                g_decoder[i][0] = self.decoder_block[-i - 1](g_upsampl[i])
                g_decoder[i][1] = self.conv_block_dec[-i - 1](g_decoder[i][0])

        # define task dependent attention module
        for i in range(3):
            for j in range(5):
                if j == 0:
                    atten_encoder[i][j][0] = self.encoder_att[i][j](g_encoder[j][0])
                    atten_encoder[i][j][1] = (atten_encoder[i][j][0]) * g_encoder[j][1]
                    atten_encoder[i][j][2] = self.encoder_block_att[j](atten_encoder[i][j][1])
                    atten_encoder[i][j][2] = F.max_pool2d(atten_encoder[i][j][2], kernel_size=2, stride=2)
                else:
                    atten_encoder[i][j][0] = self.encoder_att[i][j](torch.cat((g_encoder[j][0], atten_encoder[i][j - 1][2]), dim=1))
                    atten_encoder[i][j][1] = (atten_encoder[i][j][0]) * g_encoder[j][1]
                    atten_encoder[i][j][2] = self.encoder_block_att[j](atten_encoder[i][j][1])
                    atten_encoder[i][j][2] = F.max_pool2d(atten_encoder[i][j][2], kernel_size=2, stride=2)

            for j in range(5):
                if j == 0:
                    atten_decoder[i][j][0] = F.interpolate(atten_encoder[i][-1][-1], scale_factor=2, mode='bilinear', align_corners=True)
                    atten_decoder[i][j][0] = self.decoder_block_att[-j - 1](atten_decoder[i][j][0])
                    atten_decoder[i][j][1] = self.decoder_att[i][-j - 1](torch.cat((g_upsampl[j], atten_decoder[i][j][0]), dim=1))
                    atten_decoder[i][j][2] = (atten_decoder[i][j][1]) * g_decoder[j][-1]
                else:
                    atten_decoder[i][j][0] = F.interpolate(atten_decoder[i][j - 1][2], scale_factor=2, mode='bilinear', align_corners=True)
                    atten_decoder[i][j][0] = self.decoder_block_att[-j - 1](atten_decoder[i][j][0])
                    atten_decoder[i][j][1] = self.decoder_att[i][-j - 1](torch.cat((g_upsampl[j], atten_decoder[i][j][0]), dim=1))
                    atten_decoder[i][j][2] = (atten_decoder[i][j][1]) * g_decoder[j][-1]

        # define task prediction layers
        t1_pred = F.log_softmax(self.pred_task1(atten_decoder[0][-1][-1]), dim=1)
        t2_pred = self.pred_task2(atten_decoder[1][-1][-1])
        t3_pred = self.pred_task3(atten_decoder[2][-1][-1])
        t3_pred = t3_pred / torch.norm(t3_pred, p=2, dim=1, keepdim=True)

        return [t1_pred, t2_pred, t3_pred], self.logsigma


def get_rgbforlabels():
    return np.array([
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32]])

def draw_color(img):
    h=img.shape[0]
    w=img.shape[1]

    colorlist=get_rgbforlabels()
    colorimg=np.zeros([h,w,3],dtype=np.float)
    for label in range(0,14):
        mask=img==label
        colorimg[mask,:]=colorlist[label]/255.0

    return colorimg

''' ===== multi task MGD trainer ==== '''
def train_minibatch(train_loader, multi_task_model, device,
                           optimizer,weightmodel,epoch,task,lambda_weight,taskOptimizer):
    losses = AverageMeter()
    train_batch = len(train_loader)
    Avg_cost=AverageMeter()


    multi_task_model.train()
    train_dataset = iter(train_loader)
    conf_mat = ConfMatrix(multi_task_model.class_nb)
    ###############Disable the

    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    for k in range(train_batch):
        cost = np.zeros(24, dtype=np.float32)
        train_data, train_label, train_depth, train_normal = train_dataset.next()
        train_data, train_label = train_data.to(device), train_label.long().to(device)
        train_depth, train_normal = train_depth.to(device), train_normal.to(device)

        train_pred, logsigma = multi_task_model(train_data)

        train_loss = [model_fit(train_pred[0], train_label, 'semantic'),
                      model_fit(train_pred[1], train_depth, 'depth'),
                      model_fit(train_pred[2], train_normal, 'normal')]

        loss=0
        for i in range(TASKNUM):
            if i==task:
                loss=loss+train_loss[i] * lambda_weight[i, epoch]
            else:
                loss = loss + train_loss[i] * lambda_weight[i, epoch]*AUXILIARY
        losses.update(loss.item(), train_data.size(0))

        taskOptimizer.zero_grad()
        loss.backward()
        taskOptimizer.step()

        # accumulate label prediction for every pixel in training images
        conf_mat.update(train_pred[0].argmax(1).flatten(), train_label.flatten())

        cost[0] = train_loss[0].item()
        cost[3] = train_loss[1].item()
        cost[4], cost[5] = depth_error(train_pred[1], train_depth)
        cost[6] = train_loss[2].item()
        cost[7], cost[8], cost[9], cost[10], cost[11] = normal_error(train_pred[2], train_normal)
        cost[:12] += cost[:12] / train_batch
        Avg_cost.update(cost)
        # compute mIoU and acc


        batch_time.update(time.time() - end)
        end = time.time()
        if k % 10 == 0:
            print('SparseEpoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.avg:.3f}\t'
                  'Data {data_time.avg:.3f}\t'
                  'Loss {loss.val:.4f}\t'.format(
                epoch, k + 1, len(train_loader),
                batch_time=batch_time, data_time=data_time,
                loss=losses))

    Avg_cost.avg[1:3] = conf_mat.get_metrics()


def eval(test_loader, train_loader,multi_task_model, device,globalepoch):
    # evaluating test data

    multi_task_model.eval()
    conf_mat = ConfMatrix(multi_task_model.class_nb)
    Avg_cost=AverageMeter()
    Avg_cost_train=AverageMeter()
    Avg_loss=[]
    Avg_loss_train=[]
    for i in range(0,TASKNUM):
        Avg_loss.append(AverageMeter())
        Avg_loss_train.append(AverageMeter())
    starttime=time.time()
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

            for i in range(0,TASKNUM):
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
        plt.subplot(1,4,1)
        plt.imshow(test_data[0,:,:,:].cpu().numpy().transpose(1,2,0))
        plt.subplot(1, 4, 2)
        plt.imshow(test_pred[1][0, :, :, :].cpu().numpy().transpose(1, 2, 0))
        plt.title("depth")
        plt.subplot(1, 4, 3)
        plt.imshow(test_pred[2][0, :, :, :].cpu().numpy().transpose(1, 2, 0))
        plt.title("normal")
        plt.subplot(1, 4, 4)

        semanticmap=test_pred[0].argmax(1)
        semanticmap=semanticmap[0,:,:].cpu().numpy()
        semanticmap=draw_color(semanticmap)
        plt.imshow(semanticmap)
        plt.savefig(opt.save+'/{}.png'.format(globalepoch))
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
        .format(globalepoch, Avg_cost_train.avg[0], Avg_cost_train.avg[1], Avg_cost_train.avg[2], Avg_cost_train.avg[3],
                Avg_cost_train.avg[4], Avg_cost_train.avg[5], Avg_cost_train.avg[6], Avg_cost_train.avg[7], Avg_cost_train.avg[8],
                Avg_cost_train.avg[9], Avg_cost_train.avg[10], Avg_cost_train.avg[11], Avg_cost.avg[0], Avg_cost.avg[1], Avg_cost.avg[2], Avg_cost.avg[3],
                Avg_cost.avg[4], Avg_cost.avg[5], Avg_cost.avg[6], Avg_cost.avg[7], Avg_cost.avg[8],
                Avg_cost.avg[9], Avg_cost.avg[10], Avg_cost.avg[11], end_time - starttime))

    return Avg_loss,Avg_loss_train


def deepcopy_parameter(model, oldmodel):
    with torch.no_grad():
        for n, p in model.named_parameters():
            p.data=deepcopy(oldmodel.state_dict()[n])


TASKNUM=3
AUXILIARY=opt.auxlr
EMAoldmomentum=opt.ema
if __name__ == '__main__':
    # control seed
    torch.backends.cudnn.enabled = False
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    devicenum=int(opt.device[-1])
    torch.cuda.set_device(devicenum)
    # define model, optimiser and scheduler
    device = torch.device(opt.device if torch.cuda.is_available() else "cpu")
    SegNet_MTAN = SegNet().to(device)

    # baselineFile = opt.comparedbaseline
    # baseline = np.load(baselineFile)
    # baseline_acclist = []
    # baseline_trainacclist = []
    # for i in range(0, baseline.shape[0]):
    #     baseline_acclist.append(baseline[i, 5])
    #     baseline_trainacclist.append(baseline[i, 4])

    if not os.path.exists(opt.save):
        os.makedirs(opt.save)
    t = time.localtime()
    opt.save = opt.save + "/" + "Metalr{0}_ema_{1}_aux_{2}".format(opt.metalr,opt.ema,opt.auxlr) + str(
        t.tm_year) + '-' + str(t.tm_mon) + '-' + str(
        t.tm_mday) + '-' + str(t.tm_hour) + '-' + str(t.tm_min) + '-' + str(t.tm_sec) + "/"
    if not os.path.exists(opt.save):
        os.makedirs(opt.save)

    # plt.figure(1)
    #
    # plt.plot(np.array(range(0, len(baseline_acclist))), np.stack(baseline_acclist), '--', color='black')
    # plt.plot(np.array(range(0, len(baseline_trainacclist))), np.stack(baseline_trainacclist), '-', color='black')
    # plt.grid(axis='x', color='0.95')
    # plt.legend(title='Parameter where:')
    # plt.pause(0.01)

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

    batch_size = 1
    nyuv2_train_loader = torch.utils.data.DataLoader(
        dataset=nyuv2_train_set,
        batch_size=batch_size,
        shuffle=True)

    nyuv2_test_loader = torch.utils.data.DataLoader(
        dataset=nyuv2_test_set,
        batch_size=batch_size,
        shuffle=False)

    ####Defining the fusion model###
    weight_model = Meta_fusion_weights_list(SegNet_MTAN,tasknum=3)
    weight_model = torch.nn.DataParallel(weight_model).cuda()
    '''3. Defining the optimizer'''

    optimization_paramslist = []
    for i in range(0, TASKNUM):
        optimization_paramslist.append(
            {"params": weight_model.module.weightlist[i].parameters(), "initial_lr": opt.metalr, "lr": opt.metalr})


    FusionOptimizer = torch.optim.Adam(optimization_paramslist, lr= opt.lr)  # the weight decay matters
    taskmodel_optimizer=optim.Adam(SegNet_MTAN.parameters(), lr=opt.lr)
    scheduler1 = optim.lr_scheduler.StepLR(taskmodel_optimizer, step_size=100, gamma=0.5)
    optimizer = MetaGrad(optimizer=taskmodel_optimizer, temperature=1,
                         tasknum=TASKNUM,
                         meta_Optimizer=FusionOptimizer, inneriteration=1,device=device)
    lambda_weight = np.ones([3, 200])

    losslist = []
    trainlosslist = []

    for i in range(0, TASKNUM):
        losslist.append([])
        trainlosslist.append([])


    for epoch in range(0,200):
        starttime=time.time()
        weightlist = []
        for i in range(0, TASKNUM):
            weightlist.append([])

        oldmodel = deepcopy(SegNet_MTAN)
        for taskid in tqdm(range(0,TASKNUM)):
            train_minibatch(nyuv2_train_loader, SegNet_MTAN, device,
                        optimizer,  weight_model, epoch, taskid, lambda_weight,taskmodel_optimizer)

            adaptmodel = deepcopy(SegNet_MTAN)
            weightlist[taskid] = (deepcopy(adaptmodel.state_dict()))
            del adaptmodel
            deepcopy_parameter(SegNet_MTAN, oldmodel)
            # SegNet_MTAN.load_state_dict(oldmodel.state_dict())
        oldweightmodel = deepcopy(weight_model)
        print(">>>>>meta updating")
        adaptionmodel = deepcopy(SegNet_MTAN)
        tmp = optimizer.pc_backward(weightlist, adaptionmodel, weight_model, nyuv2_train_loader, epoch)
        innner_loop_state = deepcopy(tmp)
        print('------------------the total time cost:{}'.format(time.time() - starttime))
        del tmp
        with torch.no_grad():
            for n, p in weight_model.named_parameters():
                p.data = deepcopy(
                    (1 - EMAoldmomentum) * p.data + EMAoldmomentum * oldweightmodel.state_dict()[n])
        SegNet_MTAN.load_state_dict(innner_loop_state)
        del adaptionmodel
        print(">>>>>meta updating")
        scheduler1.step()
        # stepSheduler.step()

        Avg_loss,Avg_loss_train=eval(nyuv2_test_loader, nyuv2_train_loader, SegNet_MTAN, device, epoch)
        for i in range(0, TASKNUM):
            trainlosslist[i].append(Avg_loss_train[i].avg)
            losslist[i].append(Avg_loss[i].avg)

        plt.figure(1)
        plt.clf()
        # plt.plot(np.array(range(0, len(baseline_acclist))), np.stack(baseline_acclist), '--', color='black', alpha=0.3)
        # plt.plot(np.array(range(0, len(baseline_trainacclist))), np.stack(baseline_trainacclist), '-', color='black',
        #          alpha=0.3)
        for i in range(0, TASKNUM):
            plt.plot(np.array(range(0, len(losslist[i]))), np.stack(losslist[i]), '--', color=color[i],
                     label="exit" + str(i))
            plt.plot(np.array(range(0, len(trainlosslist[i]))), np.stack(trainlosslist[i]), '-', color=color[i])
        plt.legend()
        np.save(opt.save + "trainacclist.npy", trainlosslist)
        np.save(opt.save + "valacclist.npy", losslist)
        plt.pause(0.01)
        plt.savefig(opt.save + "/acc")

        torch.save(SegNet_MTAN.state_dict(), opt.save+'/model.pth')

