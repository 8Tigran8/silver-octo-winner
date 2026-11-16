"""Player-motion-only dead/live classifiers; ball absence is never an input."""
import numpy as np
import torch
from torch import nn
from mtah.temporal import Residual,normalize

class PointResidual(nn.Module):
    def __init__(self,c):
        super().__init__();self.net=nn.Sequential(nn.Conv1d(c,c,1),nn.ReLU(),nn.Dropout(.15),nn.Conv1d(c,c,1),nn.ReLU(),nn.Dropout(.15))
    def forward(self,x):return x+self.net(x)

class ActivityModel(nn.Module):
    def __init__(self,kind):
        super().__init__();self.context=128 if kind=='temporal' else 0;c=32 if kind=='temporal' else 56
        blocks=[Residual(c,d,.15) for d in [1,2,4,8,16,32]] if kind=='temporal' else [PointResidual(c) for _ in range(6)]
        self.net=nn.Sequential(nn.Conv1d(128,c,1),nn.ReLU(),*blocks,nn.Conv1d(c,1,1))
    def forward(self,x):return self.net(x)[:,0]

def loss(logits,gt,missing):
    valid=(gt<3)&missing;raw=nn.functional.binary_cross_entropy_with_logits(logits,(gt==2).float(),reduction='none')
    return (raw*valid).sum()/valid.sum().clamp_min(1)

def probabilities(model,x,segments,mean,scale):
    model.eval();model.cpu();p=np.full(len(x),.5,np.float32);radius=model.context
    with torch.inference_mode():
        for a,b in segments:
            for first in range(a,b,512):
                last=min(first+512,b);ix=np.clip(np.arange(first-radius,last+radius),a,b-1);t=torch.from_numpy(normalize(x[ix],mean,scale).T.copy()).unsqueeze(0)
                p[first:last]=torch.sigmoid(model(t))[0,radius:radius+last-first].numpy()
    return p
