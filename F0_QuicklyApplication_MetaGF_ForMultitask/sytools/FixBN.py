import torch.nn as nn

def fix_bn(m):
    classname=m.__class__.__name__
    if classname.find('BatchNorm')!=-1 or classname.find('Sparse_BN')!=-1:
        m.eval()


def unfix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1 or classname.find('Sparse_BN')!=-1:
        m.train()