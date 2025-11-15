import torch.nn as nn
import torch
from einops import rearrange

def _retrieve_grad(model):
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
    for n, p in model.named_parameters():
        # if p.grad is None: continue
        # tackle the multi-head scenario
        if p.grad is None:
            Grad_Dict[n] = torch.zeros_like(p).to(p.device)
            continue
        Grad_Dict[n] = p.grad.detach()
    return Grad_Dict

def recording_gradient_importance(model, weightmodel,Grad_Dictlist,ifupdating=True):
    '''metaupdating now'''
    from collections import OrderedDict
    for name,p in model.named_parameters():
        for idx in range(0, len(Grad_Dictlist)):

            if not weightmodel.module.weightlist[idx].gradlist.__contains__(name):
                continue
            # getattr(weightmodel.module.weightlist[idx], name.replace(".", "#")).w.data=0.9*getattr(weightmodel.module.weightlist[idx], name.replace(".", "#")).w.data+0.1*torch.norm((Grad_Dictlist[idx][name]).view(Grad_Dictlist[idx][name].shape[0],-1),dim=-1)
            getattr(weightmodel.module.weightlist[idx], name.replace(".", "#")).w.data = getattr(
                weightmodel.module.weightlist[idx], name.replace(".", "#")).w.data +  torch.norm(
                (Grad_Dictlist[idx][name]).view(Grad_Dictlist[idx][name].shape[0], -1), dim=-1)
            # getattr(weightmodel.module.weightlist[idx], name.replace(".", "#")).w.data = getattr(
            #     weightmodel.module.weightlist[idx], name.replace(".", "#")).w.data + torch.sum(
            #     (Grad_Dictlist[idx][name]).view(Grad_Dictlist[idx][name].shape[0], -1), dim=-1)
    return 0


def recording_gradient_importance_(model, weightmodel_norm,weightmodel_sum,Grad_Dictlist,ifupdating=True):
    '''metaupdating now'''
    from collections import OrderedDict
    for name,p in model.named_parameters():
        for idx in range(0, len(Grad_Dictlist)):

            if not weightmodel_norm.module.weightlist[idx].gradlist.__contains__(name):
                continue
            # getattr(weightmodel.module.weightlist[idx], name.replace(".", "#")).w.data=0.9*getattr(weightmodel.module.weightlist[idx], name.replace(".", "#")).w.data+0.1*torch.norm((Grad_Dictlist[idx][name]).view(Grad_Dictlist[idx][name].shape[0],-1),dim=-1)
            getattr(weightmodel_norm.module.weightlist[idx], name.replace(".", "#")).w.data = getattr(
                weightmodel_norm.module.weightlist[idx], name.replace(".", "#")).w.data +  torch.norm(
                (Grad_Dictlist[idx][name]).view(Grad_Dictlist[idx][name].shape[0], -1), dim=-1)
            getattr(weightmodel_sum.module.weightlist[idx], name.replace(".", "#")).w.data = getattr(
                weightmodel_sum.module.weightlist[idx], name.replace(".", "#")).w.data + torch.sum(
                (Grad_Dictlist[idx][name]).view(Grad_Dictlist[idx][name].shape[0], -1), dim=-1)
    return 0


def calculating_conflict(Grad_Dictlist,model,tasknum):
    eps=1e-30
    with torch.no_grad():
        '''calculating the conflict per minibatch'''
        Conflict_LOSS = []
        #
        glist = []
        for idx in range(0, tasknum):
            tmp_list = []
            for name, oldp in model.named_parameters():
                with torch.no_grad():
                    p = Grad_Dictlist[idx][name]  #
                    tmp_list.append(p.flatten())
            glist.append(torch.cat(tmp_list))
        '''NxMxD ,N is the number of task and M means the number of the shared channel,D means the dimension of the gradients'''
        shared_matrix = torch.stack(glist)
        # normalize_direction = shared_matrix / (gradient_norm + self.eps)
        '''calculating the conflict map----cosine-similarity*scale_ratio'''
        conflictmap = shared_matrix @ shared_matrix.transpose(-2, -1)
        normfactor = (torch.diag(conflictmap) + eps).detach()
        gradient_norm = (1 / normfactor).unsqueeze(0).unsqueeze(2)
        horizonal = gradient_norm.repeat([1, 1, shared_matrix.shape[0]])
        gradient_norm_tranpose = gradient_norm.transpose(-2, -1)
        vertical = gradient_norm_tranpose.repeat([1, shared_matrix.shape[0], 1])

        normalizing_factor = horizonal + vertical

        for name, oldp in model.named_parameters():
            weight_sum = 0
            tmp_sumlist = []


            for idx in range(0, tasknum):
                adjusted_p = Grad_Dictlist[idx][name]  #
                tmp = adjusted_p.view(adjusted_p.shape[0], -1)
                tmp_sumlist.append(tmp)

            '''NxMxD ,N is the number of task and M means the number of the shared channel,D means the dimension of the gradients'''
            shared_matrix = torch.stack(tmp_sumlist)
            shared_matrix = rearrange(shared_matrix, "N M D->M N D")

            '''calculating the conflict map----cosine-similarity*scale_ratio'''
            conflictmap = shared_matrix @ shared_matrix.transpose(-2, -1)
            conflictmap = -conflictmap * normalizing_factor.repeat([conflictmap.shape[0], 1, 1])
            # print("conflict:{0}".format(conflictmap.sum()))

            conflictloss = torch.relu(conflictmap)
            conflictloss = conflictloss.view(conflictloss.shape[0], -1)
            conflictloss = torch.sum(conflictloss, dim=-1)
            Conflict_LOSS.append(conflictloss)

        conflict_matrix = torch.cat(Conflict_LOSS)
        return conflict_matrix
