"""Tracking-only temporal neural model. No game/team/player identity inputs."""
import numpy as np
import torch
from torch import nn


class Residual(nn.Module):
    def __init__(self,channels,dilation,dropout):
        super().__init__();pad=2*dilation//2
        self.net=nn.Sequential(nn.Conv1d(channels,channels,3,padding=pad,dilation=dilation),nn.ReLU(),nn.Dropout(dropout),
                               nn.Conv1d(channels,channels,3,padding=pad,dilation=dilation),nn.ReLU(),nn.Dropout(dropout))
    def forward(self,x):return x+self.net(x)


class Temporal(nn.Module):
    def __init__(self,features=108,channels=48,dropout=.15):
        super().__init__();self.net=nn.Sequential(nn.Conv1d(features*2,channels,1),nn.ReLU(),
            *[Residual(channels,d,dropout) for d in [1,2,4,8,16]],nn.Conv1d(channels,2,1))
    def forward(self,x):return self.net(x)


def normalize(x,mean,scale):
    valid=np.isfinite(x)
    z=np.clip((np.where(valid,x,mean)-mean)/scale,-8,8).astype(np.float32)
    return np.concatenate([z,valid.astype(np.float32)],axis=-1)


def loss(logits,gt,dead_weight):
    """Targets influence loss only; unknowns never become dead supervision."""
    import torch.nn.functional as F
    valid=gt<3;live=gt<2
    wd=dead_weight*valid
    d=F.binary_cross_entropy_with_logits(logits[:,0],(gt==2).float(),reduction='none')
    h=F.binary_cross_entropy_with_logits(logits[:,1],(gt==0).float(),reduction='none')
    return (d*wd).sum()/wd.sum().clamp_min(1)+(h*live).sum()/live.sum().clamp_min(1)
