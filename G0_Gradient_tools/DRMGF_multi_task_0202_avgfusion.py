import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pdb
import numpy as np
import copy
import random
import time
from copy import deepcopy
from tools.DifUpdate import *
import matplotlib.pyplot as plt
from einops import rearrange
from tools.Train_utils import AverageMeter
from scipy.optimize import minimize, Bounds, minimize_scalar
from tqdm import tqdm
from tools.Train_utils import accuracy
from tools.Train_utils import clip_grad
from tools.FixBN import *
from E1_Multi_task.DR_MGFutils import *
'''metaGrad'''
'''2022.10.15'''
EPS=1e-20
# SCALINGvalue=1e-3
class one_weigth_module(nn.Module):
    def __init__(self, Dim1,Dim2, require_grad,magnitude=None):
        super(one_weigth_module, self).__init__()
        if magnitude is not None:
            self.w = nn.Parameter(data=torch.ones([Dim1,Dim2], dtype=torch.float)*(magnitude), requires_grad=require_grad)
        else:
            self.w = nn.Parameter(data=torch.ones([Dim1, Dim2], dtype=torch.float),requires_grad=require_grad)

class zero_weigth_module(nn.Module):
    def __init__(self, Dim1,Dim2, require_grad):
        super(zero_weigth_module, self).__init__()

        self.w = nn.Parameter(data=torch.zeros([Dim1, Dim2], dtype=torch.float),requires_grad=require_grad)


class one_threshold_module(nn.Module):
    def __init__(self, Dim1,Dim2, require_grad,magnitude=None):
        super(one_threshold_module, self).__init__()
        self.threshold=nn.Parameter(data=torch.zeros([Dim1,Dim2], dtype=torch.float), requires_grad=require_grad)



class no_weigth_module(nn.Module):
    def __init__(self, Dim1,Dim2, require_grad):
        super(no_weigth_module, self).__init__()
        # self.w = nn.Parameter(data=torch.zeros([Dim1,Dim2], dtype=torch.float), requires_grad=require_grad)
        # self.threshold=nn.Parameter(data=torch.zeros([Dim1,1], dtype=torch.float), requires_grad=require_grad)
        self.w = None
        self.threshold=None
        self.K = None
        self.alpha = None
        self.beta = None

class Meta_fusion_weights(nn.Module):
    def __init__(self, oldmodel, depth,SCALINGvalue,SCALING=False):
        global max_channel
        '''The output depth'''
        super(Meta_fusion_weights, self).__init__()
        self.Nodes_weights = []
        # print("----capturing nodes--------")
        IMAGE_SIZEH=288
        IMAGE_SIZEW=384
        from collections import OrderedDict
        self.forwardlist = OrderedDict() # recording the forward connected weights
        self.gradlist=OrderedDict() # record

        oldmodel = deepcopy(oldmodel)

        model = deepcopy(oldmodel)
        model.train()
        data = torch.autograd.Variable(torch.zeros(1, 3, IMAGE_SIZEH, IMAGE_SIZEW)).cuda()
        output, _ = model(data)
        if not isinstance(output, list):
            output = [output]

        l = output[depth].sum()
        l.backward()
        namelist=[]
        self.onedimlist=[]
        for n, p in model.named_parameters():
            if p.dim()==1:
                namelist.append(n)
                self.onedimlist.append(n)
                # if n=="module.layers.0.layers.1.weight":
                #     pass

        for n, p in model.named_parameters():
            if p.grad is not None:
                name = n.replace('.', '#')
                if p.dim() == 4:  # convweight [Output channel, Input channel,K,K]
                    tmp=p.view(p.shape[0],p.shape[1],-1)
                    norm=torch.norm(tmp,dim=-1).cpu()
                    if SCALING:
                        self.add_module(name,
                                        one_weigth_module(p.shape[0], p.shape[1], require_grad=True, magnitude=SCALINGvalue))
                    else:
                        self.add_module(name, one_weigth_module(p.shape[0],p.shape[1], require_grad=True,magnitude=norm)) #outputchannel*input_channel==the total number of connections
                    self.gradlist[n] = 1
                    if not ((n not in model.encoder_blocknamelist) and (n not in model.decoder_blocknamelist) and (
                            n not in model.conv_block_encnamelist) and (n not in model.conv_block_decnamelist)
                            and (n not in model.encoder_block_attnamelist) and (n not in model.decoder_block_attnamelist)):
                    # if not ((n not in model.encoder_blocknamelist) and (n not in model.decoder_blocknamelist) and (
                    #         n not in model.conv_block_encnamelist) and (n not in model.conv_block_decnamelist)):
                        self.forwardlist[n]=1
                elif p.dim() == 2:  # Linear weights->classification layer [output channel, inputchannel ]
                    tmp = p.view(p.shape[0], p.shape[1]).cpu()
                    self.add_module(name, one_weigth_module(p.shape[0],p.shape[1], require_grad=False))
                    self.gradlist[n] = 1
                elif p.dim() == 1:  # bias
                    self.gradlist[n] = 1
                    if n.__contains__('bias'):
                        # checking if bn parameters
                        newname = n.replace('bias', 'weight')
                        if namelist.__contains__(newname):
                            if not ((n not in model.encoder_blocknamelist) and (
                                    n not in model.decoder_blocknamelist) and (
                                            n not in model.conv_block_encnamelist) and (
                                            n not in model.conv_block_decnamelist)
                                    and (n not in model.encoder_block_attnamelist) and (
                                            n not in model.decoder_block_attnamelist)):
                            # if not ((n not in model.encoder_blocknamelist) and (
                            #         n not in model.decoder_blocknamelist) and (
                            #                 n not in model.conv_block_encnamelist) and (
                            #                 n not in model.conv_block_decnamelist)):
                                self.forwardlist[n] = 1
                                if SCALING:
                                    self.add_module(name, one_weigth_module(p.shape[0], 1, require_grad=True,magnitude=SCALINGvalue))
                                else:
                                    self.add_module(name, zero_weigth_module(p.shape[0], 1, require_grad=True))
                            else:
                                if SCALING:
                                    self.add_module(name, one_weigth_module(p.shape[0], 1, require_grad=True,magnitude=SCALINGvalue))
                                else:
                                    self.add_module(name, zero_weigth_module(p.shape[0], 1, require_grad=True))
                        else:
                            if SCALING:
                                self.add_module(name, one_weigth_module(p.shape[0], 1, require_grad=True,magnitude=SCALINGvalue))
                            else:
                                self.add_module(name, zero_weigth_module(p.shape[0], 1, require_grad=True))
                    elif n.__contains__('weight'):
                        # checking if bn parameters
                        newname = n.replace('weight', 'bias')
                        if SCALING:
                            self.add_module(name, one_weigth_module(p.shape[0], 1, require_grad=True,magnitude=SCALINGvalue))
                        else:
                            self.add_module(name, one_weigth_module(p.shape[0], 1, require_grad=True))
                        if namelist.__contains__(newname):
                            if not ((n not in model.encoder_blocknamelist) and (
                                    n not in model.decoder_blocknamelist) and (
                                            n not in model.conv_block_encnamelist) and (
                                            n not in model.conv_block_decnamelist)
                                    and (n not in model.encoder_block_attnamelist) and (
                                            n not in model.decoder_block_attnamelist)):
                            # if not ((n not in model.encoder_blocknamelist) and (
                            #         n not in model.decoder_blocknamelist) and (
                            #                 n not in model.conv_block_encnamelist) and (
                            #                 n not in model.conv_block_decnamelist)):
                                self.forwardlist[n] = 1
                    else:

                        if SCALING:
                            self.add_module(name, one_weigth_module(p.shape[0], 1, require_grad=True,magnitude=SCALINGvalue))
                        else:
                            self.add_module(name, zero_weigth_module(p.shape[0], 1, require_grad=True))

                    # print(n)
            else:
                name = n.replace('.', '#')
                if p.dim() == 4:  # convweight [Output channel, Input channel,K,K]
                    self.add_module(name, no_weigth_module(p.shape[0], p.shape[1],require_grad=False))  # outputchannel*input_channel==the total number of connections
                    if not ((n not in model.encoder_blocknamelist) and (n not in model.decoder_blocknamelist) and (
                            n not in model.conv_block_encnamelist) and (n not in model.conv_block_decnamelist)
                            and (n not in model.encoder_block_attnamelist) and (
                                    n not in model.decoder_block_attnamelist)):
                    # if not ((n not in model.encoder_blocknamelist) and (
                    #         n not in model.decoder_blocknamelist) and (
                    #                 n not in model.conv_block_encnamelist) and (
                    #                 n not in model.conv_block_decnamelist)):
                        self.forwardlist[n] = 1

                elif p.dim() == 1:  # bias
                    # checking bn parameters
                    if n.__contains__('linear'):
                        pass
                    else:
                        if n.__contains__('bias'):
                            # checking if bn parameters
                            newname = n.replace('bias', 'weight')
                            if namelist.__contains__(newname):
                                if not ((n not in model.encoder_blocknamelist) and (
                                        n not in model.decoder_blocknamelist) and (
                                                n not in model.conv_block_encnamelist) and (
                                                n not in model.conv_block_decnamelist)
                                        and (n not in model.encoder_block_attnamelist) and (
                                                n not in model.decoder_block_attnamelist)):
                                # if not ((n not in model.encoder_blocknamelist) and (
                                #         n not in model.decoder_blocknamelist) and (
                                #                 n not in model.conv_block_encnamelist) and (
                                #                 n not in model.conv_block_decnamelist)):
                                    self.forwardlist[n] = 1
                                self.add_module(name, no_weigth_module(p.shape[0], 1, require_grad=False))

                        elif n.__contains__('weight'):
                            # checking if bn parameters
                            newname = n.replace('weight', 'bias')
                            if namelist.__contains__(newname):
                                if not ((n not in model.encoder_blocknamelist) and (
                                        n not in model.decoder_blocknamelist) and (
                                                n not in model.conv_block_encnamelist) and (
                                                n not in model.conv_block_decnamelist)
                                        and (n not in model.encoder_block_attnamelist) and (
                                                n not in model.decoder_block_attnamelist)):
                                # if not ((n not in model.encoder_blocknamelist) and (
                                #         n not in model.decoder_blocknamelist) and (
                                #                 n not in model.conv_block_encnamelist) and (
                                #                 n not in model.conv_block_decnamelist)):
                                    self.forwardlist[n] = 1
                                self.add_module(name, no_weigth_module(p.shape[0], 1, require_grad=False))

class Meta_fusion_weights_list(nn.Module):
    def __init__(self, model, tasknum,SCALINGvalue):
        super(Meta_fusion_weights_list, self).__init__()
        self.weightlist = nn.ModuleList()
        from collections import OrderedDict
        self.totalsharedlist=OrderedDict()
        self.scalinglist = nn.ModuleList()
        self.lrnlist=nn.ModuleList()
        self.onedimlist = []
        for n, p in model.named_parameters():
            if p.dim() == 1:
                self.onedimlist.append(n)

        for i in range(0, tasknum):
            self.weightlist.append(Meta_fusion_weights(model, i,SCALINGvalue=0))
            self.scalinglist.append(Meta_fusion_weights(model,i,SCALINGvalue,SCALING=True))
        for name,p in model.named_parameters():
            count = 0
            for i in range(0,tasknum):
                if self.weightlist[i].forwardlist.__contains__(name) and getattr(self.weightlist[i], name.replace(".", "#")).w is not None:
                    count+=1

            if count==tasknum:
                self.totalsharedlist[name]=1


class MetaGrad():
    def __init__(self, optimizer, temperature, tasknum, meta_Optimizer, inneriteration,device,model,weightmodel,fusionoptimizer=None,
                 reduction='mean'):
        print('------------------------DR-AVG-------------------------------')
        time.sleep(2)
        self._optim, self._reduction = optimizer, reduction
        print("metaGF: modified by sy...")
        time.sleep(1)
        self.temperature = temperature
        self.tasknum = tasknum
        self.meta_weightOPtmizer = meta_Optimizer
        self.tau = 0.01
        self.inner_iteration = inneriteration
        self.eps=1e-30
        self.device=device
        assert (fusionoptimizer is not None)
        self.FusionOptimizer = fusionoptimizer


        self.sharednamelist = []
        self.Conbiasnamelist = []
        for name, p in model.named_parameters():
            shared = 0
            forward_shared = 0
            for idx in range(0, self.tasknum):
                if not weightmodel.module.weightlist[idx].gradlist.__contains__(name):
                    continue
                shared += 1

            if shared > 1:
                self.sharednamelist.append(name) #finding all the shared parameters

            if p.dim() == 1: #finding the conv bias or linear bias and exclude the BN weights,bias
                if name.__contains__('bias'):
                    # checking if bn parameters
                    newname = name.replace('bias', 'weight')
                    if not weightmodel.module.onedimlist.__contains__(newname):
                        self.Conbiasnamelist.append(name)

        print(self.Conbiasnamelist)
        return

    @property
    def optimizer(self):
        return self._optim

    def zero_grad(self):
        '''
        clear the gradient of the parameters
        '''
        self._optim.zero_grad(set_to_none=True)
        return self._optim.zero_grad()

    def step(self):
        '''
        update the parameters with the gradient
        '''

        return self._optim.step()

    def pc_backward(self, modelist, oldmodel, weightmodel, dataloader, globalepoch):
        '''
        calculate the gradient of the parameters

        input:
        - objectives: a list of objectives
        '''

        Grad_Dictlist = self._pack_grad(modelist, oldmodel, weightmodel)
        ##fusing BN
        self.fusing_bn_new(oldmodel, weightmodel, modelist)
        print('>>>>adapting')
        # innner_loop_state = deepcopy(oldmodel.state_dict())
        final_state = self.adapt_model(oldmodel, weightmodel, Grad_Dictlist, dataloader, globalepoch)

        return final_state

    def _pack_grad(self, modelist, oldmodel, weightmodel):
        '''
        pack the gradient of the parameters of the network for each objective

        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''
        Grad_Dictlist = []
        for idx in range(0, len(modelist)):
            tmpmodel = modelist[idx]
            Grad_Dict = self._retrieve_grad(tmpmodel, oldmodel, idx, weightmodel)
            Grad_Dictlist.append(Grad_Dict)

        ###-------Debug------######
        # with torch.no_grad():
        #     self.visualize_conflictmap(oldmodel,Grad_Dictlist)

        return Grad_Dictlist


    def expand(self, w, grad):
        if grad.dim()>=2:
            outc,inc=w.size()
            if grad.dim() == 2:
                pass
            elif grad.dim() == 4:
                w = w.view(outc, inc, 1, 1)
            w_expand = w.expand_as(grad)
        else:
            w_expand=w.squeeze()
        return w_expand

    def adapt_model(self, model, weightmodel, Grad_Dictlist, train_loader, epoch):
        # switch to train mode
        self.FusionOptimizer.zero_grad(set_to_none=True)
        train_batch = len(train_loader)
        train_dataset = iter(train_loader)
        losses = AverageMeter()
        # self.updating_noshared_parameters(model, weightmodel, Grad_Dictlist)
        with torch.no_grad():
            self.updating_noshared_parameters(model, weightmodel, Grad_Dictlist)
        with torch.no_grad():
            for k in range(train_batch):
                train_data, train_label, train_depth, train_normal = train_dataset.next()
                train_data, train_label = train_data.to(self.device), train_label.long().to(self.device)
                train_depth, train_normal = train_depth.to(self.device), train_normal.to(self.device)

                metaGF_dict=self.meta_fusingGradigent(model, weightmodel, Grad_Dictlist, ifdifupdating=True)
                train_pred, logsigma = model(train_data,metaGF_dict)
            # train_loss = [model_fit(train_pred[0], train_label, 'semantic'),
            #               model_fit(train_pred[1], train_depth, 'depth'),
            #               model_fit(train_pred[2], train_normal, 'normal')]
            #
            # tmploss = 0
            # for i in range(3):
            #     tmploss = tmploss + train_loss[i]
            # self.FusionOptimizer.zero_grad(set_to_none=True)
            # tmploss.backward()

            # for n,p in weightmodel.named_parameters():
            #     if p.grad is not None:
            #         if p.grad.norm()>0:
            #             print(n,p.grad.norm())

            # old_model=deepcopy(weightmodel)
            # self.FusionOptimizer.step()
            # for n,p in weightmodel.named_parameters():
            #     if p.grad is not None:
            #         if p.grad.norm() > 0:
            #             print(n,(old_model.state_dict()[n]-p).sum())

            # self.FusionOptimizer.zero_grad(set_to_none=True)
            # weightmodel.zero_grad(set_to_none=True)
            # model.zero_grad(set_to_none=True)
            # losses.update(tmploss.detach().cpu())
            # if k % 100 == 0:
            #     print('Epoch(adapt):{0}',
            #           'Loss {loss.val:.4f}\t'.format(epoch, loss=losses))

        with torch.no_grad():
            self.meta_fusingGradigent(model, weightmodel, Grad_Dictlist, ifdifupdating=False)
            final_state = deepcopy(model.state_dict())
        return final_state

    def fusing_bn(self, model, weightmodel,modellist):
        with torch.no_grad():

            for name in model.state_dict():
                if not name.__contains__("running_var"):
                    continue
                running_varname = name
                running_meanname = name.replace("running_var", "running_mean")

                varlist = []
                meanlist = []
                for idx in range(0, self.tasknum):
                    if not weightmodel.module.weightlist[idx].gradlist.__contains__(name):
                        continue

                    var = modellist[idx][running_varname]
                    mean = modellist[idx][running_meanname]
                    varlist.append(var)
                    meanlist.append(mean)

                mean_var = torch.mean(torch.stack(varlist), dim=0)
                mean_mean = torch.mean(torch.stack(meanlist), dim=0)
                model.state_dict()[running_varname].copy_(mean_var)
                model.state_dict()[running_meanname].copy_(mean_mean)

    def fusing_bn_new(self, model, weightmodel, modellist):
        with torch.no_grad():

            for name in model.state_dict():
                if not name.__contains__("running_var"):
                    continue
                running_varname = name
                running_meanname = name.replace("running_var", "running_mean")
                weightname = running_varname.replace("running_var", "weight")

                varlist = []
                meanlist = []
                weightsum = 0

                for idx in range(0, self.tasknum):
                    if not weightmodel.module.weightlist[idx].gradlist.__contains__(weightname):
                        continue

                    tmpconnection = torch.abs(
                        getattr(weightmodel.module.weightlist[idx], weightname.replace(".", "#")).w)
                    tmpconnection = tmpconnection / (torch.sum(tmpconnection, dim=1, keepdim=True) + 1e-10).pow(1)
                    weightsum += tmpconnection.squeeze()

                for idx in range(0, self.tasknum):
                    if not weightmodel.module.weightlist[idx].gradlist.__contains__(weightname):
                        continue
                    tmpconnection = torch.abs(
                        getattr(weightmodel.module.weightlist[idx], weightname.replace(".", "#")).w)
                    tmpconnection = tmpconnection / (torch.sum(tmpconnection, dim=1, keepdim=True) + 1e-10).pow(1)
                    tmpconnection = tmpconnection.squeeze()

                    var = tmpconnection * (
                                modellist[idx][running_varname] + modellist[idx][running_meanname] ** 2) / weightsum
                    mean = tmpconnection * modellist[idx][running_meanname] / weightsum
                    varlist.append(var)
                    meanlist.append(mean)

                mean_mean = torch.sum(torch.stack(meanlist), dim=0)
                mean_var = torch.sum(torch.stack(varlist), dim=0) - mean_mean ** 2  ##E(var)=E(var+u**2)-(E(u))**2
                model.state_dict()[running_varname].copy_(mean_var)
                model.state_dict()[running_meanname].copy_(mean_mean)
    def updating_noshared_parameters(self, model, weightmodel, Grad_Dictlist):
        for name, p in model.named_parameters():
            if name not in self.sharednamelist:  # task specific
                addidx=0
                with torch.no_grad():
                    for idx in range(0, self.tasknum):
                        if not weightmodel.module.weightlist[idx].gradlist.__contains__(name):

                            continue
                        addidx+=1
                        assert (addidx<=1)
                        p.data = p.data + Grad_Dictlist[idx][name]
            else:
                if p.dim() == 4:
                    pass
                elif p.dim() == 1:
                    #
                    if name in self.Conbiasnamelist:  # shared bias
                        with torch.no_grad():
                            biaslist = []
                            for idx in range(0, self.tasknum):
                                if not weightmodel.module.weightlist[idx].gradlist.__contains__(name):
                                    continue
                                biaslist.append(Grad_Dictlist[idx][name])

                            p.data = p.data + torch.mean(torch.stack(biaslist), dim=0)
                    else:
                        pass
    def meta_fusingGradigent(self, model, weightmodel, Grad_Dictlist, ifdifupdating=True):
        '''metaupdating now'''
        from collections import OrderedDict
        metaGF_dict = OrderedDict()
        for name, p in model.named_parameters():
            weight_sum=0
            if name not in self.sharednamelist: #task specific
               pass
            else:
                if p.dim() == 4:
                    for idx in range(0, self.tasknum):
                        if not weightmodel.module.weightlist[idx].gradlist.__contains__(name):
                            continue
                        tmpweight = getattr(weightmodel.module.weightlist[idx], name.replace(".", "#")).w
                        if tmpweight is None:
                            continue
                        tmpconnection = torch.ones_like(tmpweight).cuda()
                        w_ = tmpconnection
                        # w_expand = self.expand(w_ / (torch.sum(w_, dim=1, keepdim=True) + 1e-10).pow(1), p)
                        w_expand = self.expand((w_ / (torch.sum(w_, dim=1, keepdim=True) + 1e-10).pow(1)) * torch.abs(
                            getattr(weightmodel.module.scalinglist[idx], name.replace(".", "#")).w), p)

                        adjusted_p = w_expand * Grad_Dictlist[idx][
                            name]  # reallocating the imortance channel for each exit

                        if metaGF_dict.__contains__(name):
                            metaGF_dict[name] = metaGF_dict[name] + adjusted_p
                            weight_sum = weight_sum + w_expand
                        else:
                            metaGF_dict[name] = adjusted_p
                            weight_sum = w_expand

                    '''calculating average'''
                    metaGF_dict[name] = p.data + metaGF_dict[name] / (weight_sum + EPS)
                elif p.dim() == 1:
                    #
                    if name in self.Conbiasnamelist: #shared bias
                        with torch.no_grad():
                           pass
                    else:
                        for idx in range(0, self.tasknum):
                            if not weightmodel.module.weightlist[idx].gradlist.__contains__(name):
                                continue
                            if getattr(weightmodel.module.weightlist[idx], name.replace(".", "#")).w is None:
                                continue
                            tmpweight = getattr(weightmodel.module.weightlist[idx], name.replace(".", "#")).w

                            if tmpweight is None:
                                continue

                            tmpconnection = torch.ones_like(Grad_Dictlist[idx][name]).cuda()
                            w_ = tmpconnection.squeeze()
                            adjusted_p = (w_ * Grad_Dictlist[idx][name]) # reallocating the imortance channel for each exit
                            if metaGF_dict.__contains__(name):
                                metaGF_dict[name] = metaGF_dict[name] + adjusted_p
                                weight_sum = weight_sum + w_
                            else:
                                metaGF_dict[name] = adjusted_p
                                weight_sum = w_

                        if metaGF_dict.__contains__(name):
                            metaGF_dict[name] =p.data+ metaGF_dict[name] / (weight_sum + EPS)

        if ifdifupdating:
            return metaGF_dict
        else:
            for name, p in model.named_parameters():
                if metaGF_dict.__contains__(name):
                    p.data =((metaGF_dict[name])).detach()
            return None

    def _unflatten_grad(self, grads, shapes):
        unflatten_grad, idx = [], 0
        for shape in shapes:
            length = np.prod(shape)
            unflatten_grad.append(grads[idx:idx + length].view(shape).clone())
            idx += length
        return unflatten_grad

    def _flatten_grad(self, grads, shapes):
        flatten_grad = torch.cat([g.flatten() for g in grads])
        return flatten_grad

    def _retrieve_grad(self, model, oldmodel, idx, weightmodel):
        '''
        get the gradient of the parameters of the network with specific
        objective

        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''
        from collections import OrderedDict
        Grad_Dict = OrderedDict()
        for n, p in oldmodel.named_parameters():
            # if p.grad is None: continue
            # tackle the multi-head scenario
            if not weightmodel.module.weightlist[idx].gradlist.__contains__(n):
                # Grad_Dict[n] = torch.zeros_like(p).to(p.device)
                continue
            Grad_Dict[n] = (model[n]-p.data).detach() #-g
        return Grad_Dict
