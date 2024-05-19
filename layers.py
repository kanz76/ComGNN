import math 
from torch import nn 
import torch 
import torch.nn.functional as F

def MyLinear(in_feats, out_feats):
    return nn.Linear(in_feats, out_feats, bias=False)

