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
from E1_Multi_task.DR_MGFutils import *
'''metaGrad'''
EPS=1e-10
class one_weigth_module(nn.Module):
    def __init__(self, Dim1,Dim2, require_grad):
        super(one_weigth_module, self).__init__()
        self.w = nn.Parameter(data=torch.zeros([Dim1, Dim2], dtype=torch.float),requires_grad=require_grad)

class Meta_fusion_weights(nn.Module):
    def __init__(self, oldmodel, depth):
        global max_channel
        '''The output depth'''
        super(Meta_fusion_weights, self).__init__()
        self.Nodes_weights = []
        # print("----capturing nodes--------")
        IMAGE_SIZEH = 288
        IMAGE_SIZEW = 384
        from collections import OrderedDict
        self.forwardlist = OrderedDict() # recording the forward connected weights
        self.gradlist=OrderedDict() # record

        oldmodel = deepcopy(oldmodel)

        model = deepcopy(oldmodel)
        data = torch.autograd.Variable(torch.zeros(1, 3, IMAGE_SIZEH, IMAGE_SIZEW)).cuda()
        output, _ = model(data)
        if not isinstance(output, list):
            output = [output]

        l = output[depth].sum()
        l.backward()
        for n, p in model.named_parameters():
            # print(n)
            if p.grad is not None:
                name = n.replace('.', '#')
                # if p.dim() == 1:  # bias
                #     self.add_module(name, one_weigth_module(p.shape[0], require_grad=True))
                # elif p.dim() == 2:  # Linear weights->classification layer [output channel, inputchannel ]
                #     self.add_module(name, one_weigth_module(p.shape[0], require_grad=True))
                # elif p.dim() == 4:  # convweight [Output channel, Input channel,K,K]
                #     self.add_module(name, one_weigth_module(p.shape[0], require_grad=True))
                if p.dim() == 4:  # convweight [Output channel, Input channel,K,K]
                    tmp=p.view(p.shape[0],p.shape[1],-1)
                    norm=torch.norm(tmp,dim=-1).cpu()
                    self.add_module(name, one_weigth_module(p.shape[0],p.shape[1], require_grad=True)) #outputchannel*input_channel==the total number of connections
                    self.gradlist[n] = 1
                    self.forwardlist[n]=1
                elif p.dim() == 2:  # Linear weights->classification layer [output channel, inputchannel ]
                    tmp = p.view(p.shape[0], p.shape[1]).cpu()
                    self.add_module(name, one_weigth_module(p.shape[0],p.shape[1], require_grad=False))
                    self.gradlist[n] = 1
                elif p.dim() == 1:  # bias
                    self.add_module(name, one_weigth_module(p.shape[0],1, require_grad=True))
                    self.gradlist[n] = 1


class Meta_fusion_weights_list(nn.Module):
    def __init__(self, model, tasknum):
        super(Meta_fusion_weights_list, self).__init__()
        self.weightlist = nn.ModuleList()
        from collections import OrderedDict
        self.totalsharedlist=OrderedDict()
        for i in range(0, tasknum):
            self.weightlist.append(Meta_fusion_weights(model, i))

        for name,p in model.named_parameters():
            count = 0
            for i in range(0, tasknum):
                if self.weightlist[i].forwardlist.__contains__(name) and getattr(self.weightlist[i], name.replace(".", "#")).w is not None:
                    count+=1

            if count==tasknum:
                self.totalsharedlist[name]=1


class MetaGrad():
    def __init__(self, optimizer, temperature, tasknum, meta_Optimizer, inneriteration,device,
                 reduction='mean'):
        self._optim, self._reduction = optimizer, reduction
        print("metaGF: modified by sy...")
        time.sleep(1)
        self.temperature = temperature
        self.tasknum = tasknum
        self.meta_weightOPtmizer = meta_Optimizer
        self.tau = 0.01
        self.inner_iteration = inneriteration
        self.eps=1e-30
        self.device = device
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
        print('>>>>adapting')
        innner_loop_state = deepcopy(oldmodel.state_dict())
        final_state = self.adapt_model(oldmodel, weightmodel, Grad_Dictlist, dataloader, globalepoch,innner_loop_state)

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

    def adapt_model(self, model, weightmodel, Grad_Dictlist, train_loader, epoch,innner_loop_state):
        # switch to train mode
        loss = AverageMeter()
        self.meta_weightOPtmizer.zero_grad()

        train_batch = len(train_loader)
        train_dataset = iter(train_loader)
        losses = AverageMeter()
        # self.updating_noshared_parameters(model, weightmodel, Grad_Dictlist)
        final_state=None
        for k in range(train_batch):
            tmpmodel = deepcopy(model)
            train_data, train_label, train_depth, train_normal = train_dataset.next()
            train_data, train_label = train_data.to(self.device), train_label.long().to(self.device)
            train_depth, train_normal = train_depth.to(self.device), train_normal.to(self.device)

            self.meta_fusingGradigent(tmpmodel, weightmodel, Grad_Dictlist, ifdifupdating=True)
            train_pred, logsigma = tmpmodel(train_data)
            train_loss = [model_fit(train_pred[0], train_label, 'semantic'),
                          model_fit(train_pred[1], train_depth, 'depth'),
                          model_fit(train_pred[2], train_normal, 'normal')]

            tmploss = 0
            for i in range(3):
                tmploss = tmploss + train_loss[i]
            self.meta_weightOPtmizer.zero_grad()
            tmploss.backward()

            self.meta_weightOPtmizer.step()
            self.meta_weightOPtmizer.zero_grad()
            # del tmpmodel
            '''UPDATING THE BN norm'''
            from collections import OrderedDict
            average_dict = OrderedDict()
            for name in innner_loop_state:
                featurename = name.split('.')[-1]
                if featurename in ["running_mean", "running_var", "num_batches_tracked"]:
                    average_dict[name] = deepcopy(tmpmodel.state_dict()[name])
                else:
                    average_dict[name] = deepcopy(innner_loop_state[name])

            model.load_state_dict(average_dict)
            innner_loop_state = deepcopy(average_dict)
            final_state = deepcopy(tmpmodel.state_dict())
            loss.update(tmploss.detach())

            losses.update(tmploss.detach().cpu())
            del tmpmodel
            if k % 10 == 0:
                print('Epoch(adapt):{0}',
                      'Loss {loss.val:.4f}\t'.format(epoch, loss=losses))
        # self.meta_weightOPtmizer.step()

        return final_state

    def meta_fusingGradigent(self, model, weightmodel, Grad_Dictlist, ifdifupdating=True):
        '''metaupdating now'''
        from collections import OrderedDict
        metaGF_dict = OrderedDict()
        Conflict_LOSS = 0.0
        for name, p in model.named_parameters():
            weight_sum = 0
            for idx in range(0, self.tasknum):
                if not weightmodel.module.weightlist[idx].gradlist.__contains__(name):
                    continue
                '''calculating gradient'''
                w_ = torch.sigmoid(getattr(weightmodel.module.weightlist[idx], name.replace(".", "#")).w)

                w_expand = self.expand(w_, p)
                adjusted_p = w_expand * Grad_Dictlist[idx][name]  # reallocating the imortance channel for each exit
                if metaGF_dict.__contains__(name):
                    metaGF_dict[name] = metaGF_dict[name] + adjusted_p
                    weight_sum = weight_sum + w_expand
                else:
                    metaGF_dict[name] = adjusted_p
                    weight_sum = w_expand

            '''calculating average'''

            if metaGF_dict.__contains__(name):
                metaGF_dict[name] = metaGF_dict[name] / (weight_sum + EPS)
            else:
                metaGF_dict[name] = torch.zeros_like(p, device=self.device)
            p.update = (metaGF_dict[name])


        if ifdifupdating:
            update_module(model)
            return Conflict_LOSS
        else:
            for name, p in model.named_parameters():
                p.data = (metaGF_dict[name])

            return Conflict_LOSS

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
            Grad_Dict[n] = model[n]-p.data #-g
        return Grad_Dict


    def calculatingconflict(self, model,weightmodel, Grad_Dictlist):
        '''global '''
        #normalizaing the weighting factor
        tmpGrad_Dictlist=deepcopy(Grad_Dictlist)
        for name, p in model.named_parameters():
            weight_sum = 0
            for idx in range(0, self.tasknum):
                if not weightmodel.module.weightlist[idx].gradlist.__contains__(name):
                    continue
                '''calculating gradient'''
                # layernorm
                # w_ = torch.exp(getattr(weightmodel.module.weightlist[idx], name.replace(".", "#")).w)
                w_ = (getattr(weightmodel.module.weightlist[idx], name.replace(".", "#")).w**2)
                # w_expand = self.expand(w_ / (torch.max(w_) + EPS), p)
                w_expand = self.expand(w_ / (torch.sum(w_) + EPS), p)
                weight_sum = weight_sum + w_expand

            for idx in range(0, self.tasknum):
                if not weightmodel.module.weightlist[idx].gradlist.__contains__(name):
                    continue
                '''calculating gradient'''
                # layernorm
                # w_ = torch.exp(getattr(weightmodel.module.weightlist[idx], name.replace(".", "#")).w)
                w_ = (getattr(weightmodel.module.weightlist[idx], name.replace(".", "#")).w ** 2)
                # w_expand = self.expand(w_ / (torch.max(w_) + EPS), p)
                w_expand = self.expand(w_ / (torch.sum(w_) + EPS), p)
                tmpGrad_Dictlist[idx][name] = w_expand * Grad_Dictlist[idx][name]/(weight_sum+EPS)  # reallocating the imortance channel for each exit

        glist = []
        for idx in range(0, self.tasknum):
            tmp_list = []
            for name, oldp in model.named_parameters():
                adjusted_p =(tmpGrad_Dictlist[idx][name])
                tmp_list.append(adjusted_p.flatten())
            glist.append(torch.cat(tmp_list))
        '''NxMxD ,N is the number of task and M means the number of the shared channel,D means the dimension of the gradients'''
        shared_matrix = torch.stack(glist)
        # normalize_direction = shared_matrix / (gradient_norm + self.eps)
        '''calculating the conflict map----cosine-similarity*scale_ratio'''
        conflictmap = shared_matrix @ shared_matrix.transpose(-2, -1)
        normfactor=torch.diag(conflictmap)+self.eps

        conflictmap = -conflictmap/normfactor
        # print(conflictmap)

        conflictloss = torch.nn.functional.relu(conflictmap+0.2).mean()
        # if conflictloss>0:
        #     print(">>>>>>>conflict happen>>>>>>>>>>>>>>>>\n\n")
        return conflictloss