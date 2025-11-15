import sys

import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

sys.path.insert(1, "../../")
import argparse
import torch.utils.data.sampler as sampler
from E1_Multi_task.create_dataset import *
from E1_Multi_task.DR_MGFutils import *
from tools.Sparse_conv_v2 import *
from G0_Gradient_tools.DRMGF_multi_task_0202_fusionextra import *
from tools.Train_utils import *
from E1_Multi_task.utils import get_rgbforlabels,draw_color
from E1_Multi_task.models.SparseSegnet import SegNet
from time import sleep
parser = argparse.ArgumentParser(description='Multi-task: Split')
parser.add_argument('--type', default='standard', type=str, help='split type: standard, wide, deep')
parser.add_argument('--weight', default='equal', type=str, help='multi-task weighting: equal, uncert, dwa')
parser.add_argument('--dataroot', default='/home/sunyi/nyuv2_10', type=str, help='dataset root')
# parser.add_argument('--dataroot', default='/media/sunyi/E/nyuv2', type=str, help='dataset root')
parser.add_argument('--method', default='mgd', type=str, help='optimization method')
parser.add_argument('--temp', default=2.0, type=float, help='temperature for DWA (must be positive)')
parser.add_argument('--alpha', default=0.5, type=float, help='the alpha')
parser.add_argument('--lr', default=1e-4, type=float, help='the learning rate')
parser.add_argument('--seed', default=0, type=int, help='the seed')
parser.add_argument('--apply_augmentation', action='store_true', help='toggle to apply data augmentation on NYUv2')
parser.add_argument('--comparedbaseline',default='losslist.npy')
parser.add_argument('--save',default='./tmpsave/',type=str)
parser.add_argument('--device',default=0,type=int)
parser.add_argument('--metalr',default=0.1,type=float)
parser.add_argument('--ema',default=0.9,type=float)
parser.add_argument('--auxlr',default=0.4,type=float)
parser.add_argument('--MetaGFstartEpoch',default=50,type=int)
parser.add_argument('--usinglossscale',default=0,type=int)
parser.add_argument('--inverse',default=1,type=int)
parser.add_argument('--weightdecay',default=0,type=float)
parser.add_argument('--usinglosslandscape',default=1,type=int)
parser.add_argument('--CONSTANTSHARELAYER',default=0,type=int)
parser.add_argument('--SCALINGvalue',default=1e-3,type=float)
opt = parser.parse_args()
color = ['cyan', 'green', 'blue', 'olive', 'black', 'yellow', 'red']


opt.device="cuda:{}".format(opt.device)
CONSTANT_SHARELAYER=opt.CONSTANTSHARELAYER
''' ===== multi task MGD trainer ==== '''

def train_minibatch(train_loader, multi_task_model, device,
                    optimizer, weightmodel, epoch, task, lambda_weight,
                    METAGF_START_EPOCH,taskOptimizer,global_ratiolist,args):
    # print(lambda_weight)
    losses = AverageMeter()
    train_batch = len(train_loader)
    Avg_cost = AverageMeter()

    multi_task_model.train()
    conf_mat = ConfMatrix(multi_task_model.class_nb)
    currentlearningrate=999999999
    for param_group in taskOptimizer.param_groups:
        currentlearningrate=param_group['lr']
    print("lr:{}".format(currentlearningrate))
    model_static=deepcopy(multi_task_model.state_dict())
    tic = time.time()

    '''-----------------1. constant auxilr---------------------'''
    ratio =global_ratiolist[task]
    print("-----------end of analyzing the loss ratio:{}".format(time.time()-tic))

    multi_task_model.load_state_dict(model_static)
    multi_task_model.train()
    print("-----------loading_static".format(time.time()-tic))

    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    train_dataset = iter(train_loader)

    print("---------------------------------")
    progress_bar = tqdm(range(train_batch), desc=f"Epoch {epoch + 1} Task {task}")
    for k in progress_bar:
        cost = np.zeros(24, dtype=np.float32)
        train_data, train_label, train_depth, train_normal = next(train_dataset)
        train_data, train_label = train_data.to(device), train_label.long().to(device)
        train_depth, train_normal = train_depth.to(device), train_normal.to(device)

        if epoch < METAGF_START_EPOCH:
            '''Dynamically calculating the task-specific inference routes: refer to Eq.(7)'''
            meta_connection_weight = meta_fusingModel(multi_task_model, weightmodel, TASKNUM,task)
            train_pred, logsigma = multi_task_model(train_data, meta_connection_weight)
        else:
            train_pred, logsigma = multi_task_model(train_data)

        train_loss = [model_fit(train_pred[0], train_label, 'semantic'),
                      model_fit(train_pred[1], train_depth, 'depth'),
                      model_fit(train_pred[2], train_normal, 'normal')]

        loss = 0
        for i in range(TASKNUM):
            if i == task:
                loss = loss + train_loss[i] * lambda_weight[i, epoch]
            else:
                tmp = ratio[i]  * lambda_weight[i, epoch] * train_loss[i]
                loss = loss + tmp

        losses.update(loss.item(), train_data.size(0))

        taskOptimizer.zero_grad()
        if epoch < METAGF_START_EPOCH:
            optimizer.meta_weightOPtmizer.zero_grad()

        weightmodel.zero_grad()
        loss.backward()

        taskOptimizer.step()
        if epoch < METAGF_START_EPOCH:
            optimizer.meta_weightOPtmizer.step()

        optimizer.meta_weightOPtmizer.zero_grad()
        taskOptimizer.zero_grad()
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
        if k % 100 == 0:
            print('SparseEpoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.avg:.3f}\t'
                  'Data {data_time.avg:.3f}\t'
                  'Loss {loss.val:.4f}\t'.format(
                epoch, k + 1, len(train_loader),
                batch_time=batch_time, data_time=data_time,
                loss=losses))
        progress_bar.set_postfix({
            'avg_loss': f'{losses.val:.4f}'
        })

    Avg_cost.avg[1:3] = conf_mat.get_metrics()


def eval(test_loader, train_loader, multi_task_model, device, globalepoch):
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
        progress_bar = tqdm(range(test_batch),desc=f'Test (epoch {globalepoch})')
        for _ in progress_bar:
            cost = np.zeros(24, dtype=np.float32)
            test_data, test_label, test_depth, test_normal = next(test_dataset)
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
        plt.savefig(opt.save + '/{}.png'.format(globalepoch))
        plt.pause(0.01)

    conf_mat = ConfMatrix(multi_task_model.class_nb)
    with torch.no_grad():  # operations inside don't track history
        test_dataset = iter(train_loader)
        test_batch = len(train_loader)
        for k in range(test_batch):
            cost = np.zeros(24, dtype=np.float32)
            test_data, test_label, test_depth, test_normal = next(test_dataset)
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


def meta_fusingModel(model, weightmodel, tasknum,taskid):
    '''updating the forward params'''
    from collections import OrderedDict
    meta_connection_weight = OrderedDict()

    layeridx=0
    for name, p in model.named_parameters():
        weight_sum = 0
        if p.dim() == 4: #conv layers
            layeridx+=1
            weightslist = []
            for idx in range(0, tasknum):
                tmpweight = getattr(weightmodel.module.weightlist[idx], name.replace(".", "#")).w
                if tmpweight is None:
                    continue
                '''calculating gradient'''
                # #pruned-structure
                tmpconnection = torch.abs(tmpweight)
                ### for multi-exit networks

                p = tmpconnection / (1e-4 * torch.sum(tmpconnection, dim=1, keepdim=True) + 1).pow(0.75)
                if idx == taskid:
                    weights=10
                else:
                    weights = 1
                weight_sum += weights
                weightslist.append(weights * p)

            if len(weightslist) and name not in model.fixedsharedlayer:
                meta_connection_weight[name] = torch.sum(torch.stack(weightslist), dim=0) / (weight_sum + 1e-20)
            elif len(weightlist) and name in model.fixedsharedlayer:
                meta_connection_weight[name] = None

        elif p.dim() == 1:
            weightslist = []
            for idx in range(0, tasknum):
                # module.layers.0.layers.1.weight
                if not weightmodel.module.weightlist[idx].forwardlist.__contains__(name):
                    continue
                tmpweight = getattr(weightmodel.module.weightlist[idx], name.replace(".", "#")).w
                if tmpweight is None:
                    continue
                '''calculating gradient'''
                # #pruned-structure
                weightslist.append(tmpweight)
            if len(weightslist):
                meta_connection_weight[name] = None

    return meta_connection_weight

def weighting_themodel(model, weightmodel):
    ##soft weighting the routes
    # dynamically soft-pruning the network
    model_dict = deepcopy(model.state_dict())
    for name in model_dict:
        if weightmodel.__contains__(name):
            w = weightmodel[name]
            if model_dict[name].dim() == 4:
                if w is not None:
                    tmpconnection = w
                    ##1-3
                    gate = tmpconnection
                    outc, inc, k, k = model_dict[name].size()
                    w = model_dict[name].view(outc, inc, -1)
                    w = F.normalize(w, dim=-1)
                    w = w.view(outc, inc, k, k)

                    # debug
                    nanmask = torch.isnan(w * expand(gate, model_dict[name]))
                    if nanmask.sum():
                        raise ("nan detected")
                    model_dict[name] = w * expand(gate, model_dict[name])
                else:
                    pass
            elif model_dict[name].dim() == 1:
                if w is not None:
                    if name.__contains__('weight'):
                        tmpconnection = torch.abs(w).squeeze()
                    else:
                        tmpconnection = torch.abs(w).squeeze()

                    nanmask = torch.isnan(tmpconnection)
                    if nanmask.sum():
                        raise ("nan detected")
                    model_dict[name] = tmpconnection* torch.sign(model_dict[name] )
                else:
                    pass

    model.load_state_dict(model_dict)


def deepcopy_parameter(model, oldmodel):
    with torch.no_grad():
        for n, p in model.named_parameters():
            p.data = deepcopy(oldmodel.state_dict()[n])


TASKNUM = 3
AUXILIARY=opt.auxlr
GRID_SIZE=20
SCALE=AUXILIARY
if __name__ == '__main__':
    print("--------0117  loss landscape--------")
    # control seed
    torch.backends.cudnn.enabled = False
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    devicenum = int(opt.device[-1])
    torch.cuda.set_device(devicenum)

    device = torch.device(opt.device if torch.cuda.is_available() else "cpu")
    SegNet_MTAN = SegNet().to(device)
    SegNet_MTAN.Obtaining_shallowSharedLayer(CONSTANT_SHARELAYER)


    if not os.path.exists(opt.save):
        os.makedirs(opt.save)
    t = time.localtime()
    opt.save = opt.save + "/" + "cuda{10}_SCALINGvalue_{9}_CONSTANT_SHARELAYER_{8}_losslandscape{7}_Metalr{0}_ema_{1}_aux_{2}_MetaGFstartEpoch{3}_UsingScale{4}_extraScaling_inverse{5}_weightdecay{6}".format(opt.metalr, opt.ema, opt.auxlr,
                                                                                                                  opt.MetaGFstartEpoch,
                                                                                                                    opt.usinglossscale,
                                                                                                                 opt.inverse,opt.weightdecay,
                                                                                                                 opt.usinglosslandscape,
                                                                                                                opt.CONSTANTSHARELAYER,opt.SCALINGvalue,opt.device) + str(
        t.tm_year) + '-' + str(t.tm_mon) + '-' + str(
        t.tm_mday) + '-' + str(t.tm_hour) + '-' + str(t.tm_min) + '-' + str(t.tm_sec) + "/"


    if not os.path.exists(opt.save):
        os.makedirs(opt.save)


    print('Parameter Space: ABS: {:.1f}, REL: {:.4f}'.format(count_parameters(SegNet_MTAN),
                                                             count_parameters(SegNet_MTAN) / 24981069))
    print(
        'LOSS FORMAT: SEMANTIC_LOSS MEAN_IOU PIX_ACC | DEPTH_LOSS ABS_ERR REL_ERR | NORMAL_LOSS MEAN MED <11.25 <22.5 <30')

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

    ####Defining the fusion model###
    weight_model = Meta_fusion_weights_list(SegNet_MTAN, tasknum=3,SCALINGvalue=opt.SCALINGvalue)
    weight_model = torch.nn.DataParallel(weight_model).cuda()
    '''3. Defining the optimizer'''

    ##################This part must use weight decay#######################
    optimization_paramslist = []
    for i in range(0, TASKNUM):
        optimization_paramslist.append(
            {"params": weight_model.module.weightlist[i].parameters(), "initial_lr": opt.lr, "lr": opt.lr,'weight_decay':opt.weightdecay})

    optimization_paramslist_fusion = []
    for i in range(0, TASKNUM):
        optimization_paramslist_fusion.append(
            {"params": weight_model.module.scalinglist[i].parameters(), "initial_lr": opt.metalr, "lr": opt.metalr})
    ##################This part must use weight decay#######################
    print('metalr:{}'.format(opt.metalr))

    ##################Meta Optimizer########################################S
    stepsizeOptimizer = torch.optim.Adam(optimization_paramslist, lr=opt.lr)  # the weight decay matters
    FusionOptimizer = torch.optim.Adam(optimization_paramslist_fusion, lr=opt.metalr)  # the weight decay matters
    stepSheduler = optim.lr_scheduler.StepLR(stepsizeOptimizer, step_size=100, gamma=0.5)
    stepSheduler_fusion = optim.lr_scheduler.StepLR(FusionOptimizer, step_size=100, gamma=0.5)
    ##################Meta Optimizer########################################S


    taskmodel_optimizer = optim.Adam(SegNet_MTAN.parameters(), lr=opt.lr)
    optimizer = MetaGrad(optimizer=optim.Adam(SegNet_MTAN.parameters(), lr=opt.lr), temperature=1,
                         tasknum=TASKNUM,
                         meta_Optimizer=stepsizeOptimizer, inneriteration=1, device=device,model=SegNet_MTAN,weightmodel=weight_model,
                         fusionoptimizer=FusionOptimizer)
    scheduler = optim.lr_scheduler.StepLR(taskmodel_optimizer, step_size=100, gamma=0.5)
    lambda_weight = np.ones([3, 200])
    lambda_weight[2,:]=10
    losslist = []
    trainlosslist = []


    global_ratiolist=[]
    for i in range(0,TASKNUM):
        global_ratio=np.ones([TASKNUM])*AUXILIARY
        global_ratio[i]=0
        global_ratiolist.append(global_ratio)


    for i in range(0, TASKNUM):
        losslist.append([])
        trainlosslist.append([])

    for epoch in range(0, 200):
        starttime = time.time()
        weightlist = []
        for i in range(0, TASKNUM):
            weightlist.append([])

        oldmodel = deepcopy(SegNet_MTAN)
        oldweightmodel = deepcopy(weight_model)
        if opt.inverse:
            tasklist = list(range(TASKNUM - 1, -1, -1))
        else:
            tasklist = list(range(0, TASKNUM))
        for taskid in tqdm(tasklist,desc=f'Disentanglement training stage. Totoal stage is {TASKNUM} (Epoch {epoch})'):
            train_minibatch(nyuv2_train_loader, SegNet_MTAN, device,
                            optimizer, weight_model, epoch, taskid, lambda_weight
                            ,opt.MetaGFstartEpoch,taskmodel_optimizer,global_ratiolist,opt)

            adaptmodel = deepcopy(SegNet_MTAN)
            if epoch < opt.MetaGFstartEpoch:
                meta_connection_weight = meta_fusingModel(SegNet_MTAN, weight_model, TASKNUM,taskid)
                weighting_themodel(adaptmodel, meta_connection_weight)
                weightlist[taskid] = (deepcopy(adaptmodel.state_dict()))
            else:
                weightlist[taskid] = (deepcopy(SegNet_MTAN.state_dict()))
            del adaptmodel
            deepcopy_parameter(SegNet_MTAN, oldmodel)
            # SegNet_MTAN.load_state_dict(oldmodel.state_dict())


        print(">>>>>meta updating")
        adaptionmodel = deepcopy(SegNet_MTAN)
        tmp = optimizer.pc_backward(weightlist, adaptionmodel, weight_model, nyuv2_train_loader, epoch)
        innner_loop_state = deepcopy(tmp)
        print('------------------the total time cost:{}'.format(time.time() - starttime))
        del tmp

        SegNet_MTAN.load_state_dict(innner_loop_state)
        del adaptionmodel
        print(">>>>>meta updating")
        stepSheduler.step()
        stepSheduler_fusion.step()
        scheduler.step()

        if epoch >=  opt.MetaGFstartEpoch:
            with torch.no_grad():
                for n, p in weight_model.named_parameters():
                    p.data = deepcopy(
                        (1 - opt.ema) * p.data + opt.ema * oldweightmodel.state_dict()[n])

        Avg_loss, Avg_loss_train = eval(nyuv2_test_loader, nyuv2_train_loader, SegNet_MTAN, device, epoch)
        for i in range(0, TASKNUM):
            trainlosslist[i].append(Avg_loss_train[i].avg)
            losslist[i].append(Avg_loss[i].avg)

        plt.figure(1)
        plt.clf()
        for i in range(0, TASKNUM):
            plt.plot(np.array(range(0, len(losslist[i]))), np.stack(losslist[i]), '--', color=color[i],
                     label="exit" + str(i))
            plt.plot(np.array(range(0, len(trainlosslist[i]))), np.stack(trainlosslist[i]), '-', color=color[i])
        plt.legend()
        np.save(opt.save + "trainacclist.npy", trainlosslist)
        np.save(opt.save + "valacclist.npy", losslist)
        plt.pause(0.01)
        plt.savefig(opt.save + "/acc")

        torch.save(SegNet_MTAN.state_dict(), opt.save + '/model.pth')
        torch.save(weight_model.state_dict(),opt.save+'/weightmodel.pth')