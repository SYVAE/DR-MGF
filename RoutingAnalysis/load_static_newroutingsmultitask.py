#!/usr/bin/env python3

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from D0_dataset.dataloader import get_dataloaders
from tools.utils import *
from E1_Multi_task.models.SparseSegnet import SegNet

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from tools.Train_utils import *
from G0_Gradient_tools.DRMGF_multi_task_0202_fusionextra import Meta_fusion_weights_list
from G0_Gradient_tools.Binary_step import Differentiable_step
from models.SDN_Constructing import SDN
import argparse
from E1_Multi_task.create_dataset import *
from E1_Multi_task.DR_MGFutils import *
parser = argparse.ArgumentParser(description='Multi-task: Split')
parser.add_argument('--type', default='standard', type=str, help='split type: standard, wide, deep')
parser.add_argument('--weight', default='equal', type=str, help='multi-task weighting: equal, uncert, dwa')
# parser.add_argument('--dataroot', default='/home/user/nyuv2', type=str, help='dataset root')
parser.add_argument('--dataroot', default='/media/sunyi/E/nyuv2', type=str, help='dataset root')
parser.add_argument('--method', default='mgd', type=str, help='optimization method')
parser.add_argument('--temp', default=2.0, type=float, help='temperature for DWA (must be positive)')
parser.add_argument('--alpha', default=0.5, type=float, help='the alpha')
parser.add_argument('--lr', default=1e-4, type=float, help='the learning rate')
parser.add_argument('--seed', default=0, type=int, help='the seed')
parser.add_argument('--apply_augmentation', action='store_true', help='toggle to apply data augmentation on NYUv2')
parser.add_argument('--comparedbaseline',default='losslist.npy')
parser.add_argument('--save',default='./tmpsave/',type=str)
parser.add_argument('--device',default=1,type=int)
parser.add_argument('--metalr',default=0.1,type=float)
parser.add_argument('--ema',default=0.9,type=float)
parser.add_argument('--auxlr',default=0.1,type=float)
parser.add_argument('--MetaGFstartEpoch',default=50,type=int)
parser.add_argument('--usinglossscale',default=0,type=int)
parser.add_argument('--inverse',default=0,type=int)
parser.add_argument('--weightdecay',default=1e-6,type=float)
parser.add_argument('--usinglosslandscape',default=0,type=int)
parser.add_argument('--CONSTANTSHARELAYER',default=0,type=int)
parser.add_argument('--evaluate-from',default=' ',type=str)
parser.add_argument('--nBlocks',default=3,type=int)
opt = parser.parse_args()
color = ['cyan', 'green', 'blue', 'olive', 'black', 'yellow', 'red']

'''Test the performance of the proposed approach when pruning the networks with the learned fusion weight'''
EPS=1e-30
# global args
opt.save="/home/user/DRMGF_0222/MetaGF_TPAMI/E1_Multi_task/models/tmpsave/CONSTANT_SHARELAYER_0_losslandscape0_Metalr0.1_ema_0.5_aux_0.5_MetaGFstartEpoch100_UsingScale0_extraScaling_inverse0_weightdecay0.02023-2-27-11-3-46/"




'''Loading and analyzing the routing 2022/09/05: the ith layer, d channel , n exits share the d channel'''
def sy_load_static():
    torch.cuda.set_device(7)
    # ['vgg16','resnet56','wideresnet32_4','mobilenet']
    opt.evaluate_from = opt.save + "/model.pth"
    # args.evaluate_from = args.save + "save_models/best_model.pth.tar"
    print(opt.evaluate_from)
    time.sleep(2)

    SegNet_MTAN = SegNet().cuda()

    weight_model = Meta_fusion_weights_list(SegNet_MTAN, tasknum=3)
    weight_model = torch.nn.DataParallel(weight_model).cuda()

    cudnn.benchmark = True
    tmpmodel = torch.load(opt.evaluate_from, map_location="cuda:0")
    routing_dict = torch.load(opt.save + 'weightmodel.pth', map_location="cuda:0")

    SegNet_MTAN.load_state_dict(tmpmodel)
    print('Parameter Space: ABS: {:.1f}, REL: {:.4f}'.format(count_parameters(SegNet_MTAN),
                                                             count_parameters(SegNet_MTAN) / 24981069))
    print(
        'LOSS FORMAT: SEMANTIC_LOSS MEAN_IOU PIX_ACC | DEPTH_LOSS ABS_ERR REL_ERR | NORMAL_LOSS MEAN MED <11.25 <22.5 <30')
    '''dataloader...'''
    SegNet_MTAN.requires_grad_(False)
    weight_model.load_state_dict(routing_dict)

    weight_model.cpu()
    plt.figure("weight distribution")
    plt.clf()

    tmplist = []

    with torch.no_grad():

        for name,p in SegNet_MTAN.named_parameters():
            # if p.dim()==1:
            #     continue
            '''The shared layer:'''
            featurename = name.split('.')[-1]
            if featurename in ["running_mean", "running_var", "num_batches_tracked"]:
                continue
            shared_parameter=SegNet_MTAN.state_dict()[name]
            print(name)
            print(p.shape)
            if not (p.dim()==4 or p.dim()==1):
                continue
            channel_number=shared_parameter.shape[0]
            tmp_sum=0
            for i in range(0,len(weight_model.module.weightlist)):
                if weight_model.module.weightlist[i].gradlist.__contains__(name):
                    # w_ = (torch.exp(getattr(weight_model.module.weightlist[i], name.replace(".", "#")).w)).view(-1)
                    # tmpconnection = (getattr(weight_model.module.weightlist[i], name.replace(".", "#")).w ** 2)
                    # fusion_mag = tmpconnection - getattr(weight_model.module.thresholdlist[i],name.replace(".", "#")).threshold
                    # mask = (fusion_mag) > 0  # channel-wise pruning
                    #
                    # ratio = 1 - torch.sum(mask) / mask.numel()
                    # print("sparsification:{}".format(ratio))
                    # w_ = tmpconnection * mask
                    if p.dim()==4:
                        tmpconnection = torch.abs(
                                getattr(weight_model.module.weightlist[i], name.replace(".", "#")).w)
                        # w_norm_relative=tmpconnection / (1e-1 * torch.sum(tmpconnection, dim=1, keepdim=True) + 1e-10).pow(0.5)
                        w_norm_relative = tmpconnection / torch.sum(tmpconnection, dim=1, keepdim=True)

                        w_norm_relative=w_norm_relative*torch.abs(
                            getattr(weight_model.module.scalinglist[i], name.replace(".", "#")).w)

                        # print(w_)
                        # tmp_sum+=torch.exp(args.temperature*w_norm_relative)
                        tmp_sum += w_norm_relative
                        # w_expand = expand(w, shared_parameter)
                    elif p.dim()==1:
                        if name.__contains__('weight'):
                            tmpconnection = torch.abs(
                                getattr(weight_model.module.weightlist[i], name.replace(".", "#")).w).squeeze()
                            tmpconnection = tmpconnection / (torch.sum(tmpconnection, dim=0, keepdim=True) + 1e-10).pow(
                                1)
                            tmpconnection = tmpconnection * torch.abs(
                                getattr(weight_model.module.scalinglist[i], name.replace(".", "#")).w).squeeze()

                        else:
                            tmpconnection = torch.abs(
                                getattr(weight_model.module.scalinglist[i], name.replace(".", "#")).w).squeeze()
                        # w_= w_ / torch.sum(w_, dim=1, keepdim=True)
                        # w_ = (getattr(weight_model.module.weightlist[i], name.replace(".", "#")).w)
                        tmp_sum += tmpconnection
            '''normalizing'''
            tmp_sumlist=[]
            for i in range(0,len(weight_model.module.weightlist)):
                if weight_model.module.weightlist[i].gradlist.__contains__(name):
                    if p.dim()==4:

                        tmpconnection = torch.abs(
                            getattr(weight_model.module.weightlist[i], name.replace(".", "#")).w)
                        # w_norm_relative=tmpconnection / (1e-1 * torch.sum(tmpconnection, dim=1, keepdim=True) + 1e-10).pow(0.5)
                        w_norm_relative = tmpconnection / torch.sum(tmpconnection, dim=1, keepdim=True)

                        w_norm_relative = w_norm_relative * torch.abs(
                            getattr(weight_model.module.scalinglist[i], name.replace(".", "#")).w)
                        # tmp_sumlist.append((torch.exp(args.temperature*w_norm_relative)/(tmp_sum+1e-20)).view(-1).cpu().detach().numpy())
                        tmp_sumlist.append((w_norm_relative / (tmp_sum + 1e-20)).view(
                            -1).cpu().detach().numpy())
                    elif p.dim()==1:
                        if name.__contains__('weight'):
                            tmpconnection = torch.abs(
                                getattr(weight_model.module.weightlist[i], name.replace(".", "#")).w).squeeze()
                            tmpconnection = tmpconnection / (torch.sum(tmpconnection, dim=0, keepdim=True) + 1e-10).pow(
                                1)
                            tmpconnection = tmpconnection * torch.abs(
                                getattr(weight_model.module.scalinglist[i], name.replace(".", "#")).w).squeeze()

                        else:
                            tmpconnection = torch.abs(
                                getattr(weight_model.module.scalinglist[i], name.replace(".", "#")).w).squeeze()
                        # w_ = w_ / torch.sum(w_, dim=1, keepdim=True)
                        tmp_sumlist.append((tmpconnection / (tmp_sum + 1e-20)).view(
                            -1).cpu().detach().numpy())


            if len(tmp_sumlist)==0:
                continue
            '''NxM ,N is the number of task and M means the number of the shared channel'''
            shared_matrix=np.stack(tmp_sumlist)
            entropy=np.sum(-shared_matrix*np.log(shared_matrix),axis=0)
            avg_entropy=np.mean(entropy)



            fig = plt.figure(1)
            plt.clf()
            plt.title(name+" entropy:{}".format(avg_entropy))
            ax = fig.add_subplot(1, 1, 1, projection='3d',facecolor='white')
            ax.set_xlabel("Exits")
            ax.set_ylabel("Shared Paramters")
            ax.set_zlabel("Fusion Weight")
            # # plt.xticks(np.linspace(0, 50, 5, endpoint=True))
            # # # 修改纵坐标的刻度
            # # plt.yticks(np.linspace(0, 100, 10, endpoint=True))
            # # ax.set_zlim3d(0, 100)
            # # ax.axis('off')
            # ax.set_frame_on(False)

            ##meshgrid
            # x, y = np.meshgrid(np.linspace(1, shared_matrix.shape[0], shared_matrix.shape[0]),
            #                    np.linspace(1, shared_matrix.shape[1], shared_matrix.shape[1]))
            # ax.plot_surface(x, y, shared_matrix.transpose(1,0), cmap=plt.cm.gist_rainbow)
            # plt.show()
            downsample_ratio = shared_matrix.shape[1] // 32
            if downsample_ratio == 0:
                downsample_ratio = 1
            x, y = np.meshgrid(np.linspace(1, shared_matrix.shape[0], shared_matrix.shape[0]),
                               np.linspace(1, shared_matrix.shape[1] // downsample_ratio,
                                           shared_matrix.shape[1] // downsample_ratio))
            # bar
            # x, y = np.meshgrid(np.linspace(1, shared_matrix.shape[0], shared_matrix.shape[0]),
            #                    np.linspace(1, shared_matrix.shape[1]//downsample_ratio, shared_matrix.shape[1]//downsample_ratio))
            X=x.ravel()
            Y=y.ravel()


            N=shared_matrix.shape[0]
            D=shared_matrix.shape[1]

            # entropy_order=np.argsort(entropy)
            # shared_matrix=shared_matrix[:,entropy_order]
            shared_matrix=shared_matrix[:,::downsample_ratio]
            Z=shared_matrix.transpose(1, 0).reshape(-1)
            height= np.zeros_like(Z)
            width = depth = 1
            cmap_color=plt.cm.get_cmap('winter')
            level_list = np.linspace(0, 1, 65)
            color_list=cmap_color(level_list)

            # tmpZ=(N*2*((shared_matrix-np.min(shared_matrix,axis=0,keepdims=True))/(np.max(shared_matrix,axis=0,keepdims=True)-np.min(shared_matrix,axis=0,keepdims=True)))).astype(np.int64)-1
            # tmpZ=tmpZ.transpose(1, 0).reshape(-1)
            try:
                tmpZ = (64* ((Z - np.min(Z, axis=0, keepdims=True)) / (
                            np.max(Z, axis=0, keepdims=True) - np.min(Z, axis=0,keepdims=True)+1e-5))).astype(np.int64)
                # tmpZ = tmpZ.transpose(1, 0).reshape(-1)
                c = color_list[tmpZ,0:4]
                # im4 = ax.plot(x, y, shared_matrix.transpose(1,0), rstride=2, cstride=2, alpha=0.6, facecolor='white',
                #                       cmap="jet")
                ax.bar3d(X, Y, height, width, depth, Z, color=c, shade=False,edgecolor="black", alpha=1)
                plt.pause(1)
                # plt.show()

                ##imshow
                # plt.imshow(shared_matrix)
                plt.show()
                plt.pause(0.1)
                plt.savefig("base.png")
            except:
                continue

            # if(len(tmp_sumlist)):
            #     # print("ratio_sum:{0} maxnum:{1} max_value:{2} name:{3}".format(np.sum(np.stack(tmp_sumlist)),np.argmax(np.stack(tmp_sumlist)),
            #     #                                                   np.max(np.stack(tmp_sumlist)),name))
            #     if(np.max(np.stack(tmp_sumlist))>2*1/len(weight_model.module.weightlist)):
            #         print("{0} is no shared_parameter:{1}".format(name,np.max(np.stack(tmp_sumlist))))
            # else:
            #     print(name)

        # plt.pause(0.01)

def expand(w,grad):
    if grad.dim() == 2:
        w=w.view(-1,1)
    elif grad.dim()==4:
        w=w.view(-1,1,1,1)
    elif grad.dim()==1:
        w=w
    w_expand=w.expand_as(grad)
    return w_expand

def checking_sampleindex():
    index=torch.load(os.path.join('/home/sunyi/sy/meta_fusion/Meta_Fusion/results/setting1/GE100/1/index.pth'))
    print(index)
    index = torch.load(os.path.join('/home/sunyi/sy/meta_fusion/Meta_Fusion/results/reptile100_meta/1678536634327/index.pth'))
    print(index)
    index = torch.load(os.path.join('/home/sunyi/sy/meta_fusion/Meta_Fusion/results/setting1/GE100_pcgrad/1/index.pth'))
    print(index)
    index = torch.load(os.path.join('/home/sunyi/sy/meta_fusion/Meta_Fusion/results/setting1/MSDnet/1/index.pth'))
    print(index)
    index = torch.load(os.path.join('/home/sunyi/sy/meta_fusion/Meta_Fusion/results/setting1/reptile100_meta_fineclassifier/1/index.pth'))
    print(index)


if __name__ == '__main__':




    sy_load_static()
    # checking_sampleindex()