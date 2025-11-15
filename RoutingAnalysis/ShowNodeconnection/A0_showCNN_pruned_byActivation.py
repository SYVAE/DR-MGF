import torch.nn as nn
import torch

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import network.classificationNet as networks
from tqdm import tqdm
import os
import cv2
import copy
from torchvision import transforms
from Config import config as Cfg
from tqdm import tqdm
import network.classificationNet as Model
from Data.CaltechDataloader import CaltechDataSet
color=['red','coral','limegreen','c','dodgerblue','stategray',
       'royalblue','deeppink']
classname=['back_pack','bike','calculator','headphones',
'keyboard','laptop_computer','monitor',
'mouse','mug','projector']


def showNet_weights():
    '''This function is used to show the weight-connection and calculate the norm of each channel'''
    savepath = './tempmodel/'
    width = 4000
    height = 2000
    weightname = ["layers1.layers.0.weight", "layers3.layers.0.weight", "layers4.layers.0.weight",
                  "layers6.layers.0.weight", "layers8.layers.weight", "fc.layers.weight"]
    layers = len(weightname)
    mapname = "Networks visualization1"
    starty = 30
    startx = 50
    show = True
    hdistance = int((height - 2 * starty) / layers)

    ###########################1. first model#######################
    model = networks.plainCNN()
    if os.path.exists(savepath + '1tempmodel.pkl'):
        try:
            tempmodel = torch.load(Cfg.training_cfg.savepath + str(1) + 'tempmodel.pkl', map_location="cuda:0")
            from collections import OrderedDict
            new_dict = OrderedDict()
            for k, v in tempmodel.items():
                name = k[7:]
                new_dict[name] = v
            model.load_state_dict(new_dict)
        except:
            model.load_state_dict(
                torch.load(Cfg.training_cfg.savepath + str(1) + 'tempmodel.pkl', map_location="cuda:0"))

    total_weights = []
    for i in range(0, len(weightname)):
        for name, p in model.named_parameters():
            print(name)
            if name == weightname[i]:
                # print(name)
                total_weights.append(p)
    hidden_nodes = []
    for i in range(0, len(weightname)):
        if i == 0:
            hidden_nodes.append(total_weights[i].shape[1])
        hidden_nodes.append(total_weights[i].shape[0])

    total_layerpos = []
    for i in range(0, len(weightname) + 1):
        wdistance = int((width - 2 * startx) / (hidden_nodes[i] + 1))
        layerpos = []
        # print(wdistance)
        for j in range(0, hidden_nodes[i]):
            plot_y = starty + i * hdistance
            plot_x = startx + (j + 1) * wdistance
            layerpos.append(tuple([plot_x, plot_y]))
            # cv2.circle(Global_map,(plot_x,plot_y),color=(0,255,255),radius=5,thickness=-1)
        total_layerpos.append(layerpos)

    with torch.no_grad():
        Global_map=np.zeros([height,width,3])
        Global_map,weigts_norm=draw_weight_connection(Global_map, weightname, total_layerpos, total_weights, mapname, step_show=False,
                               show=False)



        np.save("weights_norm.npy",weigts_norm)
        cv2.namedWindow(mapname, cv2.WINDOW_NORMAL)
        cv2.putText(Global_map, mapname, (20, 20), fontScale=1, fontFace=1, color=(0, 0, 255), thickness=1, lineType=1)
        cv2.imshow(mapname, Global_map)
        cv2.waitKey(0)


def showStoredNorm():
    '''used for showing the statistics results of weights_norm'''
    weights_norm = np.load("weights_norm.npy",allow_pickle=True)
    savepath = './tempmodel/'
    width = 4000
    height = 2000
    weightname = ["layers1.layers.0.weight", "layers3.layers.0.weight", "layers4.layers.0.weight",
                  "layers6.layers.0.weight", "layers8.layers.weight", "fc.layers.weight"]
    layers = len(weightname)
    mapname = "Networks visualization1"
    starty = 30
    startx = 50
    show = True
    hdistance = int((height - 2 * starty) / layers)

    ###########################1. first model#######################
    model = networks.plainCNN()
    if os.path.exists(savepath + '1tempmodel.pkl'):
        try:
            tempmodel = torch.load(Cfg.training_cfg.savepath + str(1) + 'tempmodel.pkl', map_location="cuda:0")
            from collections import OrderedDict
            new_dict = OrderedDict()
            for k, v in tempmodel.items():
                name = k[7:]
                new_dict[name] = v
            model.load_state_dict(new_dict)
        except:
            model.load_state_dict(
                torch.load(Cfg.training_cfg.savepath + str(1) + 'tempmodel.pkl', map_location="cuda:0"))

    model.cuda()
    model.eval()
    total_weights = []
    for i in range(0, len(weightname)):
        for name, p in model.named_parameters():
            print(name)
            if name == weightname[i]:
                # print(name)
                total_weights.append(p)
    hidden_nodes = []
    for i in range(0, len(weightname)):
        if i == 0:
            hidden_nodes.append(total_weights[i].shape[1])
        hidden_nodes.append(total_weights[i].shape[0])

    total_layerpos = []
    for i in range(0, len(weightname) + 1):
        wdistance = int((width - 2 * startx) / (hidden_nodes[i] + 1))
        layerpos = []
        # print(wdistance)
        for j in range(0, hidden_nodes[i]):
            plot_y = starty + i * hdistance
            plot_x = startx + (j + 1) * wdistance
            layerpos.append(tuple([plot_x, plot_y]))
            # cv2.circle(Global_map,(plot_x,plot_y),color=(0,255,255),radius=5,thickness=-1)
        total_layerpos.append(layerpos)

    with torch.no_grad():
            Global_map = np.zeros([height, width, 3])
            if show:
                Global_map =  draw_weight_connection_stored(Global_map,weightname,total_layerpos,total_weights,mapname,weights_norm,show=False)

            cv2.namedWindow(mapname, cv2.WINDOW_NORMAL)
            cv2.putText(Global_map, mapname, (20, 20), fontScale=1, fontFace=1, color=(0, 0, 255), thickness=1,
                        lineType=1)
            cv2.imshow(mapname, Global_map)
            cv2.waitKey(0)


def draw_after_pruned(Global_map,weightname,total_layerpos,unitskeep_index,total_weights,raw_globalmap,mapname,step_show=False,show=False):
    '''draw connections after pruned'''
    for i in range(0, len(weightname)):
        bottomlayer = total_layerpos[i]
        toplayer = total_layerpos[i + 1]
        weights = total_weights[i]
        weights = weights.view(weights.shape[0], weights.shape[1], -1)
        weights = torch.norm(weights, dim=2)
        # weights = torch.mean(weights, dim=0).unsqueeze(0)
        # print(len(bottomlayer))
        for m in range(0, len(bottomlayer)):
            if i < len(weightname) and i > 0:
                kept_index = torch.from_numpy(unitskeep_index[i - 1])
                tmp = torch.where(kept_index == m)
                # print(tmp[0].size())
                if tmp[0].size() == torch.Size([0]):
                    # print("here")
                    continue

            transfer_w = weights[:, m]
            maxv = torch.max(transfer_w)
            transfer_w = transfer_w / maxv
            to_topweights = torch.clamp(transfer_w, max=1).cpu().numpy().astype(np.float)
            if step_show:
                Global_map = raw_globalmap.copy()
            for n in range(0, len(toplayer)):

                if i < len(weightname) - 1:
                    kept_index = torch.from_numpy(unitskeep_index[i])
                    tmp = torch.where(kept_index == n)
                    # print(tmp[0].size())
                    if tmp[0].size() == torch.Size([0]):
                        # print("here")
                        continue

                ux = bottomlayer[m][0]
                uy = bottomlayer[m][1]

                dx = toplayer[n][0]
                dy = toplayer[n][1]
                # cv2.line(Global_map, (ux, uy), (dx, dy), color=(heatmap[n][0], heatmap[n][1],heatmap[n][2]), thickness=1)
                cv2.line(Global_map, (ux, uy), (dx, dy), color=(to_topweights[n], to_topweights[n], to_topweights[n]),
                         thickness=2)
            if 1:
                for ii in range(0, len(weightname) + 1):
                    bottom = total_layerpos[ii]
                    for mm in range(0, len(bottom)):
                        ux = bottom[mm][0]
                        uy = bottom[mm][1]
                        cv2.circle(Global_map, (ux, uy), color=(0, 255, 255), radius=5, thickness=-1)
                if show:
                    cv2.imshow(mapname, Global_map)
                    cv2.waitKey(1)
    return Global_map

def draw_weight_connection(Global_map,weightname,total_layerpos,total_weights,mapname,step_show=False,show=False):
    raw_globalmap = Global_map.copy()

    weights_norm=[]
    for i in range(0, len(weightname)):
        bottomlayer = total_layerpos[i]
        toplayer = total_layerpos[i + 1]
        weights = total_weights[i]

        weights = weights.view(weights.shape[0], weights.shape[1], -1)
        weights = torch.norm(weights, dim=2)

        print(len(bottomlayer))
        for m in range(0, len(bottomlayer)):
            transfer_w = weights[:, m]
            maxv = torch.max(transfer_w)
            transfer_w = transfer_w / maxv
            to_topweights = torch.clamp(transfer_w, max=1).numpy().astype(np.float)

            if step_show:
                Global_map=raw_globalmap.copy()
            for n in range(0, len(toplayer)):
                ux = bottomlayer[m][0]
                uy = bottomlayer[m][1]

                dx = toplayer[n][0]
                dy = toplayer[n][1]
                # cv2.line(Global_map, (ux, uy), (dx, dy), color=(heatmap[n][0], heatmap[n][1],heatmap[n][2]), thickness=1)
                cv2.line(Global_map, (ux, uy), (dx, dy), color=(to_topweights[n], to_topweights[n], to_topweights[n]),
                         thickness=2)



        weights = total_weights[i]
        weights = weights.view(weights.shape[0], -1)#64 1
        weights = torch.norm(weights, dim=1)
        ##########storing weights norm################
        weights_norm.append(weights.numpy())

        transfer_w = weights
        maxv = torch.max(transfer_w)
        transfer_w = transfer_w / maxv
        to_topweights = torch.clamp(transfer_w, max=1).numpy().astype(np.float)

        for n in range(0, len(toplayer)):
            dx = toplayer[n][0]
            dy = toplayer[n][1]
            # cv2.line(Global_map, (ux, uy), (dx, dy), color=(heatmap[n][0], heatmap[n][1],heatmap[n][2]), thickness=1)
            cv2.circle(Global_map, (dx, dy),
                       color=(to_topweights[n], to_topweights[n], to_topweights[n]),
                       thickness=-1, radius=10)
            if 1:
                for ii in range(0, len(weightname) + 1):
                    bottom = total_layerpos[ii]
                    for mm in range(0, len(bottom)):
                        ux = bottom[mm][0]
                        uy = bottom[mm][1]
                        cv2.circle(Global_map, (ux, uy), color=(0, 255, 255), radius=5, thickness=-1)
                if show:
                    cv2.imshow(mapname, Global_map)
                    cv2.waitKey(1)

    for i in range(0, len(weightname) + 1):
        bottomlayer = total_layerpos[i]
        for m in range(0, len(bottomlayer)):
            ux = bottomlayer[m][0]
            uy = bottomlayer[m][1]
            cv2.circle(Global_map, (ux, uy), color=(0, 0, 255), radius=5, thickness=-1)
    return Global_map,weights_norm


def draw_weight_connection_stored(Global_map,weightname,total_layerpos,mapname,weights_norm,show=False):
    for i in range(0, len(weights_norm)):
        toplayer = total_layerpos[i + 1]

        transfer_w = torch.from_numpy(weights_norm[i])
        maxv = torch.max(transfer_w)
        transfer_w = transfer_w / maxv
        to_topweights = torch.clamp(transfer_w, max=1).numpy().astype(np.float)

        for n in range(0, len(toplayer)):
            dx = toplayer[n][0]
            dy = toplayer[n][1]
            # cv2.line(Global_map, (ux, uy), (dx, dy), color=(heatmap[n][0], heatmap[n][1],heatmap[n][2]), thickness=1)
            cv2.circle(Global_map, (dx, dy),
                       color=(to_topweights[n], to_topweights[n], to_topweights[n]),
                       thickness=-1, radius=10)
            if 1:
                for ii in range(0, len(weightname) + 1):
                    bottom = total_layerpos[ii]
                    for mm in range(0, len(bottom)):
                        ux = bottom[mm][0]
                        uy = bottom[mm][1]
                        cv2.circle(Global_map, (ux, uy), color=(0, 255, 255), radius=5, thickness=-1)
                if show:
                    cv2.imshow(mapname, Global_map)
                    cv2.waitKey(1)

    for i in range(0, len(weightname) + 1):
        bottomlayer = total_layerpos[i]
        for m in range(0, len(bottomlayer)):
            ux = bottomlayer[m][0]
            uy = bottomlayer[m][1]
            cv2.circle(Global_map, (ux, uy), color=(0, 0, 255), radius=5, thickness=-1)
    return Global_map

def showNetActivation_withPruning(pruned_ratio):
    '''pruning and show networks after pruned'''
    weights_norm = np.load("weights_norm.npy", allow_pickle=True)

    savepath = './tempmodel/'
    width = 4000
    height = 2000
    weightname = ["layers1.layers.0.weight", "layers3.layers.0.weight", "layers4.layers.0.weight",
                  "layers6.layers.0.weight", "layers8.layers.weight", "fc.layers.weight"]
    layers = len(weightname)
    mapname = "Networks visualization1"
    starty = 30
    startx = 50
    show = True
    hdistance = int((height - 2 * starty) / layers)
    pruned_ration=pruned_ratio
    ###########################1. first model#######################
    model = networks.plainCNN()
    if os.path.exists(savepath + '1tempmodel.pkl'):
        try:
            tempmodel = torch.load(Cfg.training_cfg.savepath + str(1) + 'tempmodel.pkl', map_location="cuda:0")
            from collections import OrderedDict
            new_dict = OrderedDict()
            for k, v in tempmodel.items():
                name = k[7:]
                new_dict[name] = v
            model.load_state_dict(new_dict)
        except:
            model.load_state_dict(
                torch.load(Cfg.training_cfg.savepath + str(1) + 'tempmodel.pkl', map_location="cuda:0"))

    model.cuda()
    model.eval()
    total_weights = []
    for i in range(0, len(weightname)):
        for name, p in model.named_parameters():
            # print(name)
            if name == weightname[i]:
                total_weights.append(p)
    hidden_nodes = []
    for i in range(0, len(weightname)):
        if i == 0:
            hidden_nodes.append(total_weights[i].shape[1])
        hidden_nodes.append(total_weights[i].shape[0])

    total_layerpos = []
    for i in range(0, len(weightname) + 1):
        wdistance = int((width - 2 * startx) / (hidden_nodes[i] + 1))
        layerpos = []
        # print(wdistance)
        for j in range(0, hidden_nodes[i]):
            plot_y = starty + i * hdistance
            plot_x = startx + (j + 1) * wdistance
            layerpos.append(tuple([plot_x, plot_y]))
            # cv2.circle(Global_map,(plot_x,plot_y),color=(0,255,255),radius=5,thickness=-1)
        total_layerpos.append(layerpos)

    with torch.no_grad():
        Global_map=np.zeros([height,width,3])
        cv2.namedWindow(mapname, cv2.WINDOW_NORMAL)
        raw_globalmap = Global_map.copy()
        ###pruned
        unitskeep_index=[]
        for i in range(0,len(weightname)):
            weights=torch.from_numpy(weights_norm[i])
            tmp_value,tmp_unitskeep_index=torch.topk(weights,dim=0,k=int((1-pruned_ration)*weights.shape[0]))
            unitskeep_index.append(tmp_unitskeep_index.numpy())
        np.save("kept_unitindex.npy",unitskeep_index)

        Global_map = draw_weight_connection_stored(Global_map,weightname,total_layerpos,mapname,weights_norm,show=False)

####################################show connection###################################################
        Global_map=draw_after_pruned(Global_map,weightname, total_layerpos, unitskeep_index, total_weights, raw_globalmap, mapname,
                          step_show=False,show=True)
####################################show connection###################################################

        torch.cuda.empty_cache()
        cv2.putText(Global_map, mapname, (20, 20), fontScale=1, fontFace=1, color=(0, 0, 255), thickness=1, lineType=1)
        cv2.imshow(mapname, Global_map)
        cv2.waitKey(1)


def test(modelid):
    model=Model.plainCNN_forpruned()
    if os.path.exists(Cfg.training_cfg.savepath+str(modelid)+'tempmodel.pkl'):
        try:
            tempmodel = torch.load(Cfg.training_cfg.savepath+str(modelid)+'tempmodel.pkl',map_location="cuda:0")
            from collections import OrderedDict
            new_dict = OrderedDict()
            for k, v in tempmodel.items():
                name = k[7:]
                new_dict[name] = v
            model.load_state_dict(new_dict)
        except:
            model.load_state_dict(torch.load(Cfg.training_cfg.savepath +str(modelid)+ 'tempmodel.pkl', map_location="cuda:0"))

    else:
        raise("error")
        # model.load_state_dict(torch.load(Cfg.training_cfg.savepath+'tempmodel.pkl',map_location="cuda:0"))


    model.cuda()
    model.eval()

    ##data_loader
    DataSet=CaltechDataSet("/home/sunyi/sy/data/office_caltech_10/caltech/",'list.txt',transform_flag=False)
    Loader=DataLoader(DataSet,batch_size=Cfg.training_cfg.batchsize,shuffle=True,num_workers=1,pin_memory=True)

    count=0
    correct_pred=[]

    for data in tqdm(Loader):
        count=count+(data[0].shape[0])
        input=data[0].float().cuda()
        label=data[1].long().cuda()
        label=label.unsqueeze(1).unsqueeze(2)


        #########################scale=1#################################
        output=model.forward(input)
        prediction=torch.argmax(output[0],dim=1)
        mask=(prediction-label)==0
        correct=mask.sum()
        correct_pred.append(float(correct))

    return (np.stack(correct_pred).sum()/len(DataSet))

if __name__ == '__main__':
    # showNet_weights()
    # showStoredNorm()
    # showNetActivation_withPruning(0.9)
    prundedlist=np.arange(start=0,stop=0.9,step=0.2)
    for i in (prundedlist):
        showNetActivation_withPruning(i)
        acc=test(1)
        print("pruned ratio:{0} acc:{1}".format(i,acc))