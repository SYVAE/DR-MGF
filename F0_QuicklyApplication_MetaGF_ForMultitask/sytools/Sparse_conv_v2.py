# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# import torch
# from G0_Gradient_tools.Binary_step import Differentiable_step
# '''Using the fusion-weight to dynamically pruning the weight'''
# '''The shared model parameters are just responsible for learning patterns'''
# '''10.10'''
#
# IFsparsity=False
# global mode
#
# #'weighting',"differentialupdate"
# def expand(w, grad):
#     # outc, inc = w.size()
#     w = w.unsqueeze(2).unsqueeze(3)
#     w_expand = w.expand_as(grad)
#     return w_expand
#
#
# class SparseConv(nn.Module):
#     def __init__(self,in_channels, channels, kernel_size=3, stride=1, padding=0, bias=True):
#         super(SparseConv,self).__init__()
#         self.in_channels=in_channels
#         self.out_channels=channels
#         self.kernel_size=[kernel_size,kernel_size]
#         self.stride=[stride,stride]
#         self.padding=[padding,padding]
#         self.weight= nn.Parameter(nn.init.kaiming_normal_(torch.zeros(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1], requires_grad=True),mode='fan_out', nonlinearity='relu'))
#         # self.norm=nn.Parameter(nn.init.kaiming_normal_(torch.zeros(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1], requires_grad=True),mode='fan_out', nonlinearity='relu'))
#         #
#         if bias:
#             self.bias = nn.Parameter(torch.empty(channels))
#         else:
#             self.bias = None
#         self.groups=1
#         self.stepfunction=Differentiable_step
#     def forward(self,x,fusionweight=None,threshold=None):
#         # global mode
#         if fusionweight is None:
#             if self.bias is not None:
#                 y=torch.nn.functional.conv2d(x,weight=self.weight,padding=self.padding,stride=self.stride,groups=self.groups,bias=self.bias)
#             else:
#                 y = torch.nn.functional.conv2d(x, weight=self.weight, padding=self.padding, stride=self.stride,
#                                                groups=self.groups)
#
#         elif fusionweight.dim()==2:
#             w=F.normalize(self.weight.view(self.out_channels,self.in_channels,-1),dim=1)
#             w=w.view(self.out_channels,self.in_channels,self.kernel_size[0],self.kernel_size[1])
#             ##4
#             p = fusionweight.unsqueeze(2).unsqueeze(3)
#             connection_level = p.expand_as(self.weight)
#
#             convweight=w*connection_level
#             if self.bias is not None:
#                 y = torch.nn.functional.conv2d(x, weight=convweight, padding=self.padding, stride=self.stride,
#                                            groups=self.groups,bias=self.bias)
#             else:
#                 y = torch.nn.functional.conv2d(x, weight=convweight, padding=self.padding,
#                                                stride=self.stride,
#                                                groups=self.groups)
#         else:
#             if self.bias is not None:
#                 y=torch.nn.functional.conv2d(x,weight=fusionweight,padding=self.padding,stride=self.stride,groups=self.groups,bias=self.bias)
#             else:
#                 y = torch.nn.functional.conv2d(x, weight=fusionweight, padding=self.padding, stride=self.stride,
#                                                groups=self.groups)
#         # if fusionweight is None:
#         #     y=torch.nn.functional.conv2d(x,weight=self.weights,padding=self.padding,stride=self.stride,groups=self.groups)
#         # else:
#         #     y = torch.nn.functional.conv2d(x, weight=self.weights*fusionweight, padding=self.padding, stride=self.stride,
#         #                                    groups=self.groups)
#         return y
#
#
#
# # class SparseConv(nn.Module):
# #     def __init__(self,in_channels, channels, kernel_size=3, stride=1, padding=0, bias=False):
# #         super(SparseConv,self).__init__()
# #         self.in_channels=in_channels
# #         self.out_channels=channels
# #         self.kernel_size=[kernel_size,kernel_size]
# #         self.stride=[stride,stride]
# #         self.padding=[padding,padding]
# #         self.bias=bias
# #         self.weights= nn.Parameter(
# #                 nn.init.kaiming_normal_(torch.zeros(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1], requires_grad=True),mode='fan_out', nonlinearity='relu'))
# #         self.groups=1
# #     def forward(self,x,fusionweight=None):
# #         if fusionweight is None:
# #             y=torch.nn.functional.conv2d(x,weight=self.weights,padding=self.padding,stride=self.stride,groups=self.groups)
# #         else:
# #             y = torch.nn.functional.conv2d(x, weight=self.weights*fusionweight, padding=self.padding, stride=self.stride,groups=self.groups)
# #         return y
#
# from torch.nn.modules.batchnorm import _NormBase
# # from torch.tensor import Tensor
# # nn.BatchNorm2d
# class Sparse_BN(_NormBase):
#     '''The dynamic will cause negative influence to the running mean ? because different routes will have different means'''
#     def __init__(self,num_features, eps=1e-5, momentum=0.1, affine=True,
#                  track_running_stats=True):
#         super(Sparse_BN,self).__init__( num_features, eps, momentum, affine, track_running_stats)
#         self.stepfunction = Differentiable_step
#     def forward(self, input,weight=None,bias=None,threshold=None):
#
#         # exponential_average_factor is set to self.momentum
#         # (when it is available) only so that it gets updated
#         # in ONNX graph when this node is exported to ONNX.
#         if self.momentum is None:
#             exponential_average_factor = 0.0
#         else:
#             exponential_average_factor = self.momentum
#
#         if self.training and self.track_running_stats:
#             # TODO: if statement only here to tell the jit to skip emitting this when it is None
#             if self.num_batches_tracked is not None:  # type: ignore
#                 self.num_batches_tracked = self.num_batches_tracked + 1  # type: ignore
#                 if self.momentum is None:  # use cumulative moving average
#                     exponential_average_factor = 1.0 / float(self.num_batches_tracked)
#                 else:  # use exponential moving average
#                     exponential_average_factor = self.momentum
#
#         r"""
#         Decide whether the mini-batch stats should be used for normalization rather than the buffers.
#         Mini-batch stats are used in training mode, and in eval mode when buffers are None.
#         """
#         if self.training:
#             bn_training = True
#         else:
#             bn_training = (self.running_mean is None) and (self.running_var is None)
#
#         r"""
#         Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
#         passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
#         used for normalization (i.e. in eval mode when buffers are not None).
#         """
#         assert self.running_mean is None or isinstance(self.running_mean, torch.Tensor)
#         assert self.running_var is None or isinstance(self.running_var, torch.Tensor)
#
#         # if weight is not None and bias is not None:
#         #     # mask = self.stepfunction(weight)
#         #     # mask=mask.squeeze()
#         #     tmpweight=weight.squeeze()
#         #     tmpbias=bias.squeeze()
#         #     # print(tmpweight.shape)
#         #     return F.batch_norm(
#         #         input,
#         #         # If buffers are not to be tracked, ensure that they won't be updated
#         #         self.running_mean if not self.training or self.track_running_stats else None,
#         #         self.running_var if not self.training or self.track_running_stats else None,
#         #         tmpweight, tmpbias, bn_training, exponential_average_factor, self.eps)
#         # else:
#         return F.batch_norm(
#             input,
#             # If buffers are not to be tracked, ensure that they won't be updated
#             self.running_mean if not self.training or self.track_running_stats else None,
#             self.running_var if not self.training or self.track_running_stats else None,
#             self.weight, self.bias, bn_training, exponential_average_factor, self.eps)

#
#
# class Sparse_BN(_NormBase):
#     '''The dynamic will cause negative influence to the running mean ? because different routes will have different means'''
#     def __init__(self,num_features, eps=1e-5, momentum=0.1, affine=True,
#                  track_running_stats=True):
#         super(Sparse_BN,self).__init__( num_features, eps, momentum, affine, track_running_stats)
#
#         self.context_batch_mean = torch.zeros((1, num_features, 1, 1),requires_grad=True)
#         self.context_batch_var = torch.ones((1, num_features, 1, 1),requires_grad=True)
#     @staticmethod
#     def _compute_batch_moments(x):
#         """
#         Compute conventional batch mean and variance.
#         :param x: input activations
#         :return: batch mean, batch variance
#         """
#         return torch.mean(x, dim=(0, 2, 3), keepdim=True), torch.var(x, dim=(0, 2, 3), keepdim=True)
#
#
#     def _normalize(self, x, mean, var,weight,bias):
#         """
#         Normalize activations.
#         :param x: input activations
#         :param mean: mean used to normalize
#         :param var: var used to normalize
#         :return: normalized activations
#         """
#         return (weight.view(1, -1, 1, 1) * (x - mean) / torch.sqrt(var + self.eps)) + bias.view(1, -1, 1, 1)
#
#     @staticmethod
#     def _compute_layer_moments(x):
#         """
#         Compute layer mean and variance.
#         :param x: input activations
#         :return: layer mean, layer variance
#         """
#         return torch.mean(x, dim=(1, 2, 3), keepdim=True), torch.var(x, dim=(1, 2, 3), keepdim=True)
#
#     def forward(self, input,weight=None,bias=None):
#
#         batch_mean, batch_var = self._compute_batch_moments(input)
#         # pooled_mean, pooled_var = self._compute_pooled_moments(x, alpha, batch_mean, batch_var,
#         #                                                        self._get_augment_moment_fn())
#         if weight is not None and bias is not None:
#             return self._normalize(input, batch_mean, batch_var,torch.abs(weight),bias)
#         else:
#             return self._normalize(input, batch_mean,  batch_var,self.weight,self.bias)
#
#


import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from G0_Gradient_tools.Binary_step import Differentiable_step
import math
'''Using the fusion-weight to dynamically pruning the weight'''
'''The shared model parameters are just responsible for learning patterns'''
'''10.10'''
import torch.jit as jit
IFsparsity=False
global mode

#'weighting',"differentialupdate"
def expand(w, grad):
    # outc, inc = w.size()
    w = w.unsqueeze(2).unsqueeze(3)
    w_expand = w.expand_as(grad)
    return w_expand


class SparseConv(nn.Module):
    def __init__(self,in_channels, channels, kernel_size=3, stride=1, padding=0, bias=True):
        super(SparseConv,self).__init__()
        self.in_channels=in_channels
        self.out_channels=channels
        self.kernel_size=[kernel_size,kernel_size]
        self.stride=[stride,stride]
        self.padding=[padding,padding]
        # nn.Conv2d
        self.weight= nn.Parameter(nn.init.kaiming_normal_(torch.zeros(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1], requires_grad=True),mode='fan_out', nonlinearity='relu'))
        self.weight= nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # if bias:
        #     self.bias = nn.Parameter(torch.empty(channels))
        # else:
        #     self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.empty(channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.bias = None
        self.groups=1
        self.stepfunction=Differentiable_step
    def forward(self,x,fusionweight=None,threshold=None):
        # global mode
        The_other_condition=False #for compatible with cooperation
        if fusionweight is not None and fusionweight.dim()==1:
            if fusionweight[0]==-99999:
                The_other_condition=True

        if fusionweight is None or The_other_condition:
            if self.bias is not None:
                y=torch.nn.functional.conv2d(x,weight=self.weight,padding=self.padding,stride=self.stride,groups=self.groups,bias=self.bias)
            else:
                y = torch.nn.functional.conv2d(x, weight=self.weight, padding=self.padding, stride=self.stride,
                                               groups=self.groups)

        elif fusionweight.dim()==2:
            w=F.normalize(self.weight.view(self.out_channels,self.in_channels,-1),dim=2)
            w=w.view(self.out_channels,self.in_channels,self.kernel_size[0],self.kernel_size[1])
            w_ = torch.abs(fusionweight)
            p = w_.unsqueeze(2).unsqueeze(3)
            connection_level = p.expand_as(self.weight)

            if self.bias is not None:
                y = torch.nn.functional.conv2d(x, weight=w*connection_level, padding=self.padding, stride=self.stride,
                                           groups=self.groups,bias=self.bias)
            else:
                y = torch.nn.functional.conv2d(x, weight=w * connection_level, padding=self.padding,
                                               stride=self.stride,
                                               groups=self.groups)
        else:
            if self.bias is not None:
                y=torch.nn.functional.conv2d(x,weight=fusionweight,padding=self.padding,stride=self.stride,groups=self.groups,bias=self.bias)
            else:
                y = torch.nn.functional.conv2d(x, weight=fusionweight, padding=self.padding, stride=self.stride,
                                               groups=self.groups)
        # if fusionweight is None:
        #     y=torch.nn.functional.conv2d(x,weight=self.weights,padding=self.padding,stride=self.stride,groups=self.groups)
        # else:
        #     y = torch.nn.functional.conv2d(x, weight=self.weights*fusionweight, padding=self.padding, stride=self.stride,
        #                                    groups=self.groups)
        return y



class SparseLinear(nn.Module):
    def __init__(self,in_channels, channels, kernel_size=3, stride=1, padding=0, bias=True):
        super(SparseLinear,self).__init__()
        self.in_channels=in_channels
        self.out_channels=channels
        self.kernel_size=[kernel_size,kernel_size]
        self.stride=[stride,stride]
        self.padding=[padding,padding]
        # nn.Conv2d
        self.weight= nn.Parameter(nn.init.kaiming_normal_(torch.zeros(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1], requires_grad=True),mode='fan_out', nonlinearity='relu'))
        self.weight= nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # if bias:
        #     self.bias = nn.Parameter(torch.empty(channels))
        # else:
        #     self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.empty(channels))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.bias = None
        self.groups=1
        self.stepfunction=Differentiable_step
    def forward(self,x,fusionweight=None,threshold=None):
        # global mode
        The_other_condition=False #for compatible with cooperation
        if fusionweight is not None and fusionweight.dim()==1:
            if fusionweight[0]==-99999:
                The_other_condition=True

        if fusionweight is None or The_other_condition:
            if self.bias is not None:
                y=torch.nn.functional.conv2d(x,weight=self.weight,padding=self.padding,stride=self.stride,groups=self.groups,bias=self.bias)
            else:
                y = torch.nn.functional.conv2d(x, weight=self.weight, padding=self.padding, stride=self.stride,
                                               groups=self.groups)

        elif fusionweight.dim()==2:
            w=F.normalize(self.weight.view(self.out_channels,self.in_channels,-1),dim=2)
            w=w.view(self.out_channels,self.in_channels,self.kernel_size[0],self.kernel_size[1])
            w_ = torch.abs(fusionweight)
            p = w_.unsqueeze(2).unsqueeze(3)
            connection_level = p.expand_as(self.weight)

            if self.bias is not None:
                y = torch.nn.functional.conv2d(x, weight=w*connection_level, padding=self.padding, stride=self.stride,
                                           groups=self.groups,bias=self.bias)
            else:
                y = torch.nn.functional.conv2d(x, weight=w * connection_level, padding=self.padding,
                                               stride=self.stride,
                                               groups=self.groups)
        else:
            if self.bias is not None:
                y=torch.nn.functional.conv2d(x,weight=fusionweight,padding=self.padding,stride=self.stride,groups=self.groups,bias=self.bias)
            else:
                y = torch.nn.functional.conv2d(x, weight=fusionweight, padding=self.padding, stride=self.stride,
                                               groups=self.groups)
        # if fusionweight is None:
        #     y=torch.nn.functional.conv2d(x,weight=self.weights,padding=self.padding,stride=self.stride,groups=self.groups)
        # else:
        #     y = torch.nn.functional.conv2d(x, weight=self.weights*fusionweight, padding=self.padding, stride=self.stride,
        #                                    groups=self.groups)
        return y


# class SparseConv(jit.ScriptModule):
#     def __init__(self,in_channels, channels, kernel_size=3, stride=1, padding=0, bias=True):
#         super(SparseConv,self).__init__()
#         self.in_channels=in_channels
#         self.out_channels=channels
#         self.kernel_size=[kernel_size,kernel_size]
#         self.stride=[stride,stride]
#         self.padding=[padding,padding]
#         self.weight= nn.Parameter(nn.init.kaiming_normal_(torch.zeros(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1], requires_grad=True),mode='fan_out', nonlinearity='relu'))
#         if bias:
#             self.bias = nn.Parameter(torch.empty(channels))
#         else:
#             self.bias = None
#         self.groups=1
#         self.stepfunction=Differentiable_step
#     @torch.jit.script_method
#     def forward(self,x,fusionweight=torch.tensor(0,dtype=torch.int)):
#         # global mode
#         # print(fusionweight)
#         if fusionweight==0:
#             # print(
#             #     "here"
#             # )
#             if self.bias is not None:
#                 y=torch.nn.functional.conv2d(x,weight=self.weight,padding=self.padding,stride=self.stride,groups=self.groups,bias=self.bias)
#             else:
#                 y = torch.nn.functional.conv2d(x, weight=self.weight, padding=self.padding, stride=self.stride,
#                                                groups=self.groups)
#
#         elif fusionweight.dim()==2:
#             w=F.normalize(self.weight.view(self.out_channels,self.in_channels,-1),dim=2)
#             w=w.view(self.out_channels,self.in_channels,self.kernel_size[0],self.kernel_size[1])
#             w_ = torch.abs(fusionweight)
#             p = w_.unsqueeze(2).unsqueeze(3)
#             connection_level = p.expand_as(self.weight)
#
#             if self.bias is not None:
#                 y = torch.nn.functional.conv2d(x, weight=w*connection_level, padding=self.padding, stride=self.stride,
#                                            groups=self.groups,bias=self.bias)
#             else:
#                 y = torch.nn.functional.conv2d(x, weight=w * connection_level, padding=self.padding,
#                                                stride=self.stride,
#                                                groups=self.groups)
#         else:
#             if self.bias is not None:
#                 y=torch.nn.functional.conv2d(x,weight=fusionweight,padding=self.padding,stride=self.stride,groups=self.groups,bias=self.bias)
#             else:
#                 y = torch.nn.functional.conv2d(x, weight=fusionweight, padding=self.padding, stride=self.stride,
#                                                groups=self.groups)
#         # if fusionweight is None:
#         #     y=torch.nn.functional.conv2d(x,weight=self.weights,padding=self.padding,stride=self.stride,groups=self.groups)
#         # else:
#         #     y = torch.nn.functional.conv2d(x, weight=self.weights*fusionweight, padding=self.padding, stride=self.stride,
#         #                                    groups=self.groups)
#         return y
#



from torch.nn.modules.batchnorm import _NormBase
# from torch.tensor import Tensor
# nn.BatchNorm2d
class Sparse_BN(_NormBase):
    '''The dynamic will cause negative influence to the running mean ? because different routes will have different means'''
    def __init__(self,num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(Sparse_BN,self).__init__( num_features, eps, momentum, affine, track_running_stats)
        self.stepfunction = Differentiable_step
    def forward(self, input,weight=None,bias=None,threshold=None):

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore
                self.num_batches_tracked = self.num_batches_tracked + 1  # type: ignore
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        assert self.running_mean is None or isinstance(self.running_mean, torch.Tensor)
        assert self.running_var is None or isinstance(self.running_var, torch.Tensor)

        if weight is not None and bias is not None:

            # tmpweight=torch.abs(weight.squeeze())+0*self.weight
            # tmpbias=bias.squeeze()+0*self.bias
            if weight.dim()==2:
                tmpweight = torch.abs(weight.squeeze()) * torch.sign(self.weight)
                tmpbias = torch.abs(bias.squeeze()) * torch.sign(self.bias)
                # tmpweight=torch.abs(weight.squeeze())+0*self.weight
                # tmpbias=bias.squeeze()+0*self.bias
            else:
                tmpweight = (weight)
                tmpbias = bias

            # print(tmpweight.shape)
            # print("here")
            # raise("...")
            return F.batch_norm(
                input,
                # If buffers are not to be tracked, ensure that they won't be updated
                self.running_mean if not self.training or self.track_running_stats else None,
                self.running_var if not self.training or self.track_running_stats else None,
                tmpweight, tmpbias, bn_training, exponential_average_factor, self.eps)
        else:
            return F.batch_norm(
                input,
                # If buffers are not to be tracked, ensure that they won't be updated
                self.running_mean if not self.training or self.track_running_stats else None,
                self.running_var if not self.training or self.track_running_stats else None,
                self.weight, self.bias, bn_training, exponential_average_factor, self.eps)



# class Sparse_BN(_NormBase):
#     '''The dynamic will cause negative influence to the running mean ? because different routes will have different means'''
#     def __init__(self,num_features, eps=1e-5, momentum=0.1, affine=True,
#                  track_running_stats=True):
#         super(Sparse_BN,self).__init__( num_features, eps, momentum, affine, track_running_stats)
#
#         self.context_batch_mean = torch.zeros((1, num_features, 1, 1),requires_grad=True)
#         self.context_batch_var = torch.ones((1, num_features, 1, 1),requires_grad=True)
#     @staticmethod
#     def _compute_batch_moments(x):
#         """
#         Compute conventional batch mean and variance.
#         :param x: input activations
#         :return: batch mean, batch variance
#         """
#         return torch.mean(x, dim=(0, 2, 3), keepdim=True), torch.var(x, dim=(0, 2, 3), keepdim=True)
#
#
#     def _normalize(self, x, mean, var,weight,bias):
#         """
#         Normalize activations.
#         :param x: input activations
#         :param mean: mean used to normalize
#         :param var: var used to normalize
#         :return: normalized activations
#         """
#         return (weight.view(1, -1, 1, 1) * (x - mean) / torch.sqrt(var + self.eps)) + bias.view(1, -1, 1, 1)
#
#     @staticmethod
#     def _compute_layer_moments(x):
#         """
#         Compute layer mean and variance.
#         :param x: input activations
#         :return: layer mean, layer variance
#         """
#         return torch.mean(x, dim=(1, 2, 3), keepdim=True), torch.var(x, dim=(1, 2, 3), keepdim=True)
#
#     def forward(self, input,weight=None,bias=None):
#
#         batch_mean, batch_var = self._compute_batch_moments(input)
#         # pooled_mean, pooled_var = self._compute_pooled_moments(x, alpha, batch_mean, batch_var,
#         #                                                        self._get_augment_moment_fn())
#         if weight is not None and bias is not None:
#             return self._normalize(input, batch_mean, batch_var,torch.abs(weight),bias)
#         else:
#             return self._normalize(input, batch_mean,  batch_var,self.weight,self.bias)




