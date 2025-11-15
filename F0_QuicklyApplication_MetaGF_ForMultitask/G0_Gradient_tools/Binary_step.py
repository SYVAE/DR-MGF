import math
import torch
import torch.nn as nn

"""
Function for activation binarization: https://github.com/junjieliu2910/DynamicSparseTraining
"""
class BinaryStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input>0.).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        zero_index = torch.abs(input) > 1
        middle_index = (torch.abs(input) <= 1) * (torch.abs(input) > 0.4)
        additional = 2-4*torch.abs(input)
        additional[zero_index] = 0.
        additional[middle_index] = 0.4
        return grad_input*additional

Differentiable_step=BinaryStep.apply


def SaturatingSigmoid(x):
    x=1.2*torch.sigmoid(x)-0.1
    y=torch.clamp(x,max=1)
    y=torch.clamp(y,min=0)
    return y