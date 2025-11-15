import copy

import cv2

from models.SDN_Constructing import SDN
from tools.Train_utils import *
from G0_Gradient_tools.bk2.DRMGF_exp3 import Meta_fusion_weights_list
from config.args import arg_parser
from tqdm import tqdm
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

color=['red','coral','limegreen','c','dodgerblue','stategray',
       'royalblue','deeppink']

##VGG plain

WIDTH=4000
HEIGHT=2000
MAPNAME="Networks visualization"
STARTY = 500
STARTX = 50
LAYERNUMBER=5
INTERVAL = int((HEIGHT-2*STARTY)/LAYERNUMBER)
PRUNED=0.8
node_interval=5
def showConnection():
    Global_map=np.zeros([HEIGHT,WIDTH,3])
    cv2.namedWindow(MAPNAME,cv2.WINDOW_NORMAL)
    cv2.putText(Global_map,MAPNAME,(80,80),fontScale=4,fontFace=1,color=(0,0,255),thickness=4,lineType=1)
    cv2.imshow(MAPNAME,Global_map)
    # cv2.waitKey(0)
    # ['vgg16','resnet56','wideresnet32_4','mobilenet']
    ####1. Defining the model
    args.sdnarch = 'resnet56'
    args.data = 'cifar10'
    args.task = args.data
    static_name = "/home/sunyi/MetaGF_TPAMI/Baseline_res/20221129/results/resnet/drmgf/1/save_models/final.pth.tar"
    args.sparsification = 1
    args.temperature = 10
    if args.data == 'cifar10':
        args.num_classes = 10
    elif args.data == 'cifar100':
        args.num_classes = 100
    else:
        args.num_classes = 1000

    dict_model = torch.load(static_name, map_location="cuda:0")
    model = SDN(args)
    model = torch.nn.DataParallel(model).cuda()
    model.train()
    weight_model = Meta_fusion_weights_list(model, args)
    weight_model = torch.nn.DataParallel(weight_model).cuda()
    weight_model.load_state_dict(dict_model['routing_index'])

    similaritmatrix = np.zeros([args.nBlocks, args.nBlocks])
    tasklist = list(range(args.nBlocks - 1, -1, -1))
    print(tasklist)
    for taskidx in tqdm(tasklist):
        for taskidy in tqdm(tasklist):
            if taskidy < taskidx:
                continue
            Structurelist1 = draw_map_(model, weight_model, taskidx)
            Structurelist2 = draw_map_(model, weight_model, taskidy)
            similarity = AverageMeter()
            for i in range(0, len(Structurelist1)):
                x1 = Structurelist1[i]
                x2 = Structurelist2[i]
                x1 = x1 / torch.norm(x1)
                x2 = x2 / torch.norm(x2)
                similarity.update(torch.sum(x1 * x2))

            similaritmatrix[taskidx, taskidy] = similarity.avg.numpy()
            similaritmatrix[taskidy, taskidx] = similarity.avg.numpy()
    print(similaritmatrix)
    plt.figure(7)
    plt.clf()
    plt.imshow(similaritmatrix)
    plt.title("structure similarity")
    plt.pause(0.01)
    plt.savefig("structure_sim")


    _,Structurelist1=draw_map(model, Global_map,weight_model,2)
    _,Structurelist2=draw_map(model, Global_map,weight_model, 6,700)

    for i in range(0,len(Structurelist1)):
        x1=Structurelist1[i]
        x2=Structurelist2[i]
        x1=x1/torch.norm(x1)
        x2=x2/torch.norm(x2)
        similarity=torch.sum(x1*x2)
        print(similarity)

    #5-6 0.988
    #0-6 0.9034
    #0-1 0.95
    #1-6 0.96
    #2-6 0.97
    tmp=copy.copy(Global_map)
    Global_map[:,:,0]=tmp[:,:,2]
    Global_map[:,:,2]=tmp[:,:,0]
    fig=plt.figure(1)
    plt.axis("off")
    height, width, channels = Global_map.shape
    # 如果dpi=300，那么图像大小=height*width
    fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.imshow(Global_map)
    # dpi是设置清晰度的，大于300就很清晰了，但是保存下来的图片很大
    plt.savefig('connection.png')
    plt.show()
    cv2.imshow(MAPNAME,Global_map)
    # cv2.imwrite("connection.png",Global_map)
    cv2.waitKey(0)

def draw_map_(model,weightmodel,task):
    convlist=[]
    weightlist=[]
    for name,p in model.named_parameters():
        tmp_sum = 0
        for i in range(0, len(weightmodel.module.weightlist)):
            if weightmodel.module.weightlist[i].gradlist.__contains__(name):
               tmp_sum+=1

        if p.dim()==4 and tmp_sum>=7:#shared by seven tasks
            convlist.append(p)
            if not weightmodel.module.weightlist[task].forwardlist.__contains__(name):
                weightlist.append(torch.zeros(p.shape[0],p.shape[1]))
                continue
            if getattr(weightmodel.module.weightlist[task], name.replace(".", "#")).w is None:
                weightlist.append(torch.zeros(p.shape[0], p.shape[1]))
                continue
            weightlist.append(abs(getattr(weightmodel.module.weightlist[task], name.replace(".", "#")).w)) #[outc,inc] the dimension of the inc is related to the connection

    tmpStructurelist=[]
    for i in range(0,len(convlist)):
        #from current layer--to--next layer
        if i < len(convlist)-1:
            weights_norm = weightlist[i+1].cpu().detach()
            tmpStructurelist.append(weights_norm.flatten())
    Structurelist = []
    Structurelist.append(torch.cat(tmpStructurelist))
    return Structurelist

def draw_map(model,map,weightmodel,task,Hoffset=0):
    cv2.line(map,(0,STARTY+Hoffset+5),(map.shape[1],STARTY+Hoffset+5),color=(0,0,255),thickness=4)
    convlist=[]
    weightlist=[]
    for name,p in model.named_parameters():
        tmp_sum = 0
        for i in range(0, len(weightmodel.module.weightlist)):
            if weightmodel.module.weightlist[i].gradlist.__contains__(name):
               tmp_sum+=1

        if p.dim()==4 and tmp_sum>=7:#shared by seven tasks
            convlist.append(p)
            weightlist.append((getattr(weightmodel.module.weightlist[task], name.replace(".", "#")).w)**2) #[outc,inc] the dimension of the inc is related to the connection

    total_layerpos = []
    for i in range(0, len(convlist)):
        wdistance = ((WIDTH - 2 * STARTX) / (convlist[i].shape[0] + 1))
        layerpos = []
        # print(wdistance)
        for j in range(0, convlist[i].shape[0]):
            plot_y = STARTY + i * INTERVAL+Hoffset
            plot_x = STARTX + int((j + 1) * wdistance)
            layerpos.append(tuple([plot_x, plot_y]))
            # cv2.circle(Global_map,(plot_x,plot_y),color=(0,255,255),radius=5,thickness=-1)
        total_layerpos.append(layerpos)


    Structurelist=[]

    for i in range(0,len(convlist)):
        toplayer=total_layerpos[i]
        if i<len(convlist)-1:
            downlayer=total_layerpos[i+1]


        #previous layer to current layer
        # weights1 = convlist[i]
        # weights1 = weights1.view(weights1.shape[0], weights1.shape[1], -1)
        # weightsnorm1 = torch.norm(weights1, dim=2)

        weightsnorm1=weightlist[i].cpu().detach()

        Nodenorm1 = torch.sum(weightsnorm1, dim=1,keepdim=True).cpu().detach()

        maxv = torch.max(Nodenorm1)
        Nodenorm1 = Nodenorm1 / (1e-1*(Nodenorm1)+1e-20).pow(0.5)
        Nodenorm1 = torch.clamp(Nodenorm1, max=1).numpy().astype(np.float)
        # Nodenorm1 = np.uint8(255 * Nodenorm1)
        # heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        # heatmap = heatmap.squeeze()
        # Nodenorm = heatmap.astype(np.float) / 255

        #from current layer--to--next layer
        if i < len(convlist)-1:
            # weights=convlist[i+1]
            # weights=weights.view(weights.shape[0],weights.shape[1],-1)
            # weights_norm = torch.norm(weights,dim=2).cpu().detach()
            weights_norm = weightlist[i+1].cpu().detach()
            # weights=torch.mean(weights,dim=1,keepdim=True)
            maxv,_=torch.max(weights_norm,dim=1,keepdim=True)
            # normalized_norm = weights_norm / maxv
            normalized_norm = weights_norm / maxv
            # normalized_norm = weights_norm / (1e-1 * torch.sum(weights_norm, dim=1, keepdim=True) + 1e-10).pow(0.5)
            to_downpweights = normalized_norm.numpy().astype(np.float)
            Structurelist.append(normalized_norm.flatten())
            # heatmap = np.uint8(255 * to_downpweights)
            # heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            # heatmap = heatmap.squeeze()
            # heatmap = heatmap.astype(np.float) / 255
        if i < len(convlist) - 1:
            for m in range(0,len(downlayer),node_interval): ##reducing the number of node
                ux = downlayer[m][0]
                uy = downlayer[m][1]
                connection=to_downpweights[m,:]
                idx=np.argsort(connection)
                idx=idx[::-1]
                selected=idx[0:int((1-PRUNED)*to_downpweights.shape[1])] #pruned connection

                for n in range(0, len(toplayer),node_interval):
                    if n not in selected:
                        continue
                    dx = toplayer[n][0]
                    dy = toplayer[n][1]
                    cv2.line(map, (ux, uy), (dx, dy),
                             color=(int((to_downpweights[m, n])*255), int((to_downpweights[m, n])*255), int((to_downpweights[m, n])*255)), thickness=1)
                # cv2.circle(map, (dx, dy), color=((Nodenorm1[n,0]), (Nodenorm1[n,0]),(Nodenorm1[n,0])),
                #          thickness=-1,radius=20)

    for i in range(0, len(convlist)):
        toplayer = total_layerpos[i]
        weightsnorm1=weightlist[i].cpu().detach()
        Nodenorm1 = torch.mean(weightsnorm1, dim=1, keepdim=True).cpu().detach()

        maxv = torch.max(Nodenorm1)
        Nodenorm1 = Nodenorm1 / maxv
        Nodenorm1 = torch.clamp(Nodenorm1, max=1).numpy().astype(np.float)
        for n in range(0, len(toplayer),node_interval):
            dx = toplayer[n][0]
            dy = toplayer[n][1]
            cv2.circle(map, (dx, dy), color=((Nodenorm1[n,0]), (Nodenorm1[n,0]),(Nodenorm1[n,0])),
                       thickness=-1, radius=20)
        # cv2.imshow("map",map)
        # cv2.waitKey(0)
    for i in range(0, len(convlist)):
        toplayer = total_layerpos[i]
        weightsnorm1=weightlist[i].cpu().detach()
        Nodenorm1 = torch.mean(weightsnorm1, dim=1, keepdim=True).cpu().detach()

        maxv = torch.max(Nodenorm1)
        Nodenorm1 = Nodenorm1 / maxv
        Nodenorm1 = torch.clamp(Nodenorm1, max=1).numpy().astype(np.float)
        for n in range(0, len(toplayer),node_interval):
            dx = toplayer[n][0]
            dy = toplayer[n][1]
            cv2.circle(map, (dx, dy), color=(0, 0, 255), radius=10,thickness=-1)

    for i in range(0, len(convlist)):
        toplayer = total_layerpos[i]
        weightsnorm1=weightlist[i].cpu().detach()
        Nodenorm1 = torch.mean(weightsnorm1, dim=1, keepdim=True).cpu().detach()

        maxv = torch.max(Nodenorm1)
        Nodenorm1 = Nodenorm1 / maxv
        Nodenorm1 = torch.clamp(Nodenorm1, max=1).numpy().astype(np.float)
        for n in range(0, len(toplayer),node_interval):
            dx = toplayer[n][0]
            dy = toplayer[n][1]
            cv2.circle(map, (dx, dy), color=(0, 255, 255), radius=5,thickness=-1)

    return map,Structurelist



def showNetActivation_withPruned():
    ##statistics of activation level ---four conv layers
    total_weights = np.load("total_weights.npy",allow_pickle=True)


    with torch.no_grad():
        Global_map=np.zeros([HEIGHT,WIDTH,3])
        cv2.namedWindow(MAPNAME,cv2.WINDOW_NORMAL)
        cv2.putText(Global_map,MAPNAME,(20,20),fontScale=1,fontFace=1,color=(0,0,255),thickness=1,lineType=1)
        cv2.imshow(MAPNAME,Global_map)

        hidden_nodes=[1,16,32,32,10]
        total_layerpos=[]
        for i in range(0,5):
            wdistance = int((width - 2*startx)/(hidden_nodes[i]+1))
            layerpos=[]
            # print(wdistance)
            for j in range(0,hidden_nodes[i]):
                plot_y=starty+i*hdistance
                plot_x=startx+(j+1)*wdistance
                layerpos.append(tuple([plot_x,plot_y]))
                # cv2.circle(Global_map,(plot_x,plot_y),color=(0,255,255),radius=5,thickness=-1)
            total_layerpos.append(layerpos)

        raw_globalmap = Global_map.copy()

        ###pruned
        unitskeep_index=[]
        for i in range(0,4):
            weights=total_weights[i]
            weights = weights.view(weights.shape[0], weights.shape[1], -1)
            weights = torch.norm(weights, dim=2)
            weights = torch.mean(weights, dim=0).unsqueeze(0)
            tmp_value,tmp_unitskeep_index=torch.topk(weights,dim=1,k=int(pruned_ration*weights.shape[1]))
            unitskeep_index.append(tmp_unitskeep_index.cpu())
        np.save("kept_unitindex.npy",unitskeep_index)

        for i in range(0,4):
            toplayer=total_layerpos[i+1]
            weights=total_weights[i]
            weights=weights.view(weights.shape[0],weights.shape[1],-1)
            weights = torch.norm(weights,dim=2)
            weights=torch.mean(weights,dim=0).unsqueeze(0)

            transfer_w = weights.cpu()
            maxv = torch.max(transfer_w)
            transfer_w = transfer_w / maxv
            to_topweights = torch.clamp(transfer_w, max=1).numpy().astype(np.float)
            heatmap = np.uint8(255 * to_topweights)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            heatmap = heatmap.squeeze()
            heatmap = heatmap.astype(np.float) / 255

            for n in range(0, len(toplayer)):

                dx = toplayer[n][0]
                dy = toplayer[n][1]
                # cv2.line(Global_map, (ux, uy), (dx, dy), color=(heatmap[n][0], heatmap[n][1],heatmap[n][2]), thickness=1)
                cv2.circle(Global_map, (dx, dy), color=(to_topweights[0,n], to_topweights[0,n], to_topweights[0,n]),
                         thickness=-1,radius=20)

        for i in range(0, 5):
            bottomlayer = total_layerpos[i]
            for m in range(0, len(bottomlayer)):
                ux = bottomlayer[m][0]
                uy = bottomlayer[m][1]
                cv2.circle(Global_map, (ux, uy), color=(0, 0, 255), radius=10, thickness=-1)


       ####################################show connection###################################################
        raw_globalmap = Global_map.copy()

        savepath = './tempmodel/'
        model = CNN_lenet.Lenet()
        if os.path.exists(savepath + 'tempmodel.pkl'):
            model.load_state_dict(torch.load(savepath + 'tempmodel.pkl', map_location="cuda:0"))

        weightname = ["layers1.layers.0.weight", "layers3.layers.0.weight", "layers4.layers.0.weight",
                      "layers6.layers.weight"]
        total_weights = []
        for i in range(0, 4):
            for name, p in model.named_parameters():
                print(name)
                if name == weightname[i]:
                    # print(name)
                    total_weights.append(p)

        for i in range(0,4):
            bottomlayer=total_layerpos[i]


            toplayer=total_layerpos[i+1]
            weights=total_weights[i]

            weights=weights.view(weights.shape[0],weights.shape[1],-1)
            weights = torch.norm(weights,dim=2)
            # weights = torch.mean(weights, dim=0).unsqueeze(0)
            print(len(bottomlayer))
            for m in range(0,len(bottomlayer)):

                if i < 4 and i>0:
                    kept_index = unitskeep_index[i-1]
                    tmp = torch.where(kept_index == m)
                    print(tmp[0].size())
                    if tmp[0].size() == torch.Size([0]):
                        print("here")
                        continue

                transfer_w=weights[:,m]
                maxv=torch.max(transfer_w)
                transfer_w=transfer_w/maxv
                to_topweights=torch.clamp(transfer_w,max=1).cpu().numpy().astype(np.float)


                Global_map=raw_globalmap.copy()
                for n in range(0,len(toplayer)):

                    if i < 3:
                        kept_index=unitskeep_index[i]
                        tmp=torch.where(kept_index==n)
                        print(tmp[0].size())
                        if tmp[0].size()==torch.Size([0]):
                            print("here")
                            continue

                    ux=bottomlayer[m][0]
                    uy=bottomlayer[m][1]

                    dx=toplayer[n][0]
                    dy=toplayer[n][1]
                    # cv2.line(Global_map, (ux, uy), (dx, dy), color=(heatmap[n][0], heatmap[n][1],heatmap[n][2]), thickness=1)
                    cv2.line(Global_map, (ux, uy), (dx, dy), color=(to_topweights[n], to_topweights[n], to_topweights[n]),
                             thickness=2)
                if 1:
                    for ii in range(0, 5):
                        bottom = total_layerpos[ii]
                        for mm in range(0, len(bottom)):
                            ux = bottom[mm][0]
                            uy = bottom[mm][1]
                            cv2.circle(Global_map, (ux, uy), color=(0, 255, 255), radius=5, thickness=-1)
                    cv2.imshow(mapname, Global_map)
                    cv2.waitKey(0)
       ####################################show connection###################################################

        torch.cuda.empty_cache()
        cv2.imshow(mapname, Global_map)
        cv2.waitKey(0)

if __name__ == '__main__':
    showConnection()