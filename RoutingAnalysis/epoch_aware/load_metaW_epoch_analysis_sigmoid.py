import torch
import numpy as np
import  torch.nn as nn
import matplotlib.pyplot as plt
import models
from config.args import arg_parser, arch_resume_names
from tools.opcounter import measure_model
from models.adaptive_inference import dynamic_evaluate
import models
import matplotlib.pyplot as plt
import os
from copy import deepcopy
from models.SDN_Constructing import SDN
from tools.Train_utils import *
# from main_metaWeights511_entropy import Meta_fusion_weights_list
from Efficient_MetaGF.MetaGFgrad import Meta_fusion_weights_list
args = arg_parser.parse_args()
args.nBlocks =7
args.stepmode = "even"
args.step = 2
args.base = 4
args.grFactor = "1-2-4"
args.bnFactor = "1-2-4"
args.growthRate = 16
args.nChannels = 16
args.data = 'cifar100'
args.arch="msdnet_ge"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

args.grFactor = list(map(int, args.grFactor.split('-')))
args.bnFactor = list(map(int, args.bnFactor.split('-')))
args.nScales = len(args.grFactor)

if args.use_valid:
    args.splits = ['train', 'val', 'test']
else:
    args.splits = ['train', 'val']



temperature=50
'''Loading and analyzing the routing 2022/05/07: the ith layer, d channel , n exits share the d channel'''
def sy_load_static():
    # ['vgg16','resnet56','wideresnet32_4','mobilenet']
    args.sdnarch='vgg16'
    args.data='cifar10'
    args.task=args.data
    folder="/home/sunyi/Shallow-Deep-Networks_MetaGF/E0_verify_gradient_conflict/results_cifar10/vgg16_sdn/meta_gradient_importance_2022-7-3-21-38-35/save_models/"
    static_name=folder+"/best_model.pth.tar"
    epoch=50
    if args.data == 'cifar10':
        args.num_classes = 10
    elif args.data == 'cifar100':
        args.num_classes = 100
    else:
        args.num_classes = 1000

    dict_model=torch.load(static_name,map_location="cuda:0")
    model = SDN(args)
    model = torch.nn.DataParallel(model).cuda()
    model.train()
    weight_model = Meta_fusion_weights_list(model, args)
    weight_model = torch.nn.DataParallel(weight_model).cuda()

    with torch.no_grad():
        for name,p in model.named_parameters():
            for epochidx in range(5,6,1):
                static_name=folder+'model_onlyrouting_{0}.pth.tar'.format(epochidx)
                dict_model = torch.load(static_name, map_location="cuda:0")
                weight_model.load_state_dict(dict_model['routing_index'])

                tmplist = []

                '''The shared layer:'''
                featurename = name.split('.')[-1]
                if featurename in ["running_mean", "running_var", "num_batches_tracked"]:
                    continue
                count=0
                for i in range(0, len(weight_model.module.weightlist)):
                    if weight_model.module.weightlist[i].gradlist.__contains__(name):
                        count+=1
                if count==0:
                    continue

                '''layernorm'''
                sum=torch.zeros(p.shape[0]).cuda()
                for i in range(0, len(weight_model.module.weightlist)):
                    if weight_model.module.weightlist[i].gradlist.__contains__(name):
                        w = abs(getattr(weight_model.module.weightlist[i], name.replace(".", "#")).w)
                        sum+= torch.sigmoid(temperature*w)

                for i in range(0, len(weight_model.module.weightlist)):
                    if weight_model.module.weightlist[i].gradlist.__contains__(name):
                        w = abs(getattr(weight_model.module.weightlist[i], name.replace(".", "#")).w)
                        w_normlize = torch.sigmoid(temperature*w)/sum
                        # print(w_normlize)
                        getattr(weight_model.module.weightlist[i], name.replace(".", "#")).w.data = w_normlize

                '''normalizing'''
                tmp_sumlist=[]
                for i in range(0,len(weight_model.module.weightlist)):
                    if weight_model.module.weightlist[i].gradlist.__contains__(name):
                        # w=torch.exp(temperature*getattr(weight_model.module.weightlist[i], name.replace(".", "#")).w)/tmp_sum
                        w=getattr(weight_model.module.weightlist[i], name.replace(".", "#")).w
                        # w = getattr(weight_model.module.weightlist[i], name.replace(".", "#")).w
                        getattr(weight_model.module.weightlist[i], name.replace(".", "#")).w.data=w
                        tmp_sumlist.append(w.cpu().detach().numpy())

                '''NxM ,N is the number of task and M means the number of the shared channel'''
                shared_matrix=np.stack(tmp_sumlist)
                entropy=np.sum(-shared_matrix*np.log(shared_matrix),axis=0)
                avg_entropy=np.mean(entropy)


                fig = plt.figure(1)
                plt.clf()
                plt.title(name)
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

                # bar
                x, y = np.meshgrid(np.linspace(1, shared_matrix.shape[0], shared_matrix.shape[0]),
                                   np.linspace(1, shared_matrix.shape[1], shared_matrix.shape[1]))
                X=x.ravel()
                Y=y.ravel()


                N=shared_matrix.shape[0]
                D=shared_matrix.shape[1]

                # entropy_order=np.argsort(entropy)
                # shared_matrix=shared_matrix[:,entropy_order]
                Z=shared_matrix.transpose(1, 0).reshape(-1)
                height= np.zeros_like(Z)
                width = depth = 1
                cmap_color=plt.cm.get_cmap('winter')
                level_list = np.linspace(0, 1, 65)
                color_list=cmap_color(level_list)

                # tmpZ=(N*2*((shared_matrix-np.min(shared_matrix,axis=0,keepdims=True))/(np.max(shared_matrix,axis=0,keepdims=True)-np.min(shared_matrix,axis=0,keepdims=True)))).astype(np.int64)-1
                # tmpZ=tmpZ.transpose(1, 0).reshape(-1)
                tmpZ = (64* ((Z - np.min(Z, axis=0, keepdims=True)) / (
                            np.max(Z, axis=0, keepdims=True) - np.min(Z, axis=0,keepdims=True)+1e-5))).astype(np.int64)
                # tmpZ = tmpZ.transpose(1, 0).reshape(-1)
                c = color_list[tmpZ,0:4]
                # im4 = ax.plot(x, y, shared_matrix.transpose(1,0), rstride=2, cstride=2, alpha=0.6, facecolor='white',
                #                       cmap="jet")
                ax.bar3d(X, Y, height, width, depth, Z, color=c, shade=False,edgecolor="black", alpha=1)
                plt.pause(0.1)
                plt.show()

                ##imshow
                # plt.imshow(shared_matrix)
                # plt.show()
                plt.savefig("base.png")

                if(len(tmp_sumlist)):
                    # print("ratio_sum:{0} maxnum:{1} max_value:{2} name:{3}".format(np.sum(np.stack(tmp_sumlist)),np.argmax(np.stack(tmp_sumlist)),
                    #                                                   np.max(np.stack(tmp_sumlist)),name))
                    if(np.max(np.stack(tmp_sumlist))>2*1/len(weight_model.module.weightlist)):
                        print("{0} is no shared_parameter:{1}".format(name,np.max(np.stack(tmp_sumlist))))
                else:
                    print(name)

            # for i in range(0,len(weight_model.module.weightlist)):
            #     tmplist = []
            #     for name in model.state_dict():
            #         if  weight_model.module.weightlist[i].gradlist.__contains__(name):
            #             tmplist.append(getattr(weight_model.module.weightlist[i], name.replace(".", "#")).w.cpu().detach().numpy())
            #         else:
            #             tmplist.append(0)
            #     plt.subplot(3, len(weight_model.module.weightlist) // 3 + 1, i + 1)
            #     plt.plot(np.array(range(0, len(tmplist))), tmplist)
            #     # plt.bar(np.array(range(0, len(tmplist))), tmplist)
            #     print(np.mean(np.stack(tmplist)))
            # plt.show()
            # # plt.pause(0.01)

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