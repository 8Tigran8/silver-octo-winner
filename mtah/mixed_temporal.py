"""Registered multiscale and boundary-aware tracking models; geometry only."""
import numpy as np
import torch
from torch import nn
from . import temporal


class MixedTemporal(nn.Module):
    def __init__(self,spec,features=108,channels=48,dropout=.15):
        super().__init__();self.context=int(spec['context_steps'])
        self.net=nn.Sequential(nn.Conv1d(features*2,channels,1),nn.ReLU(),
            *[temporal.Residual(channels,d,dropout) for d in spec['dilations']],nn.Conv1d(channels,2,1))
    def forward(self,x):return self.net(x)


def boundary_loss(logits,gt,weights,positive_weight=8):
    """Auxiliary change loss; unknown/context pairs cannot create transitions."""
    pd=torch.sigmoid(logits[:,0]);ph=torch.sigmoid(logits[:,1])
    q=torch.stack([(1-pd)*ph,(1-pd)*(1-ph),pd],dim=1)
    same=(q[:,:,1:]*q[:,:,:-1]).sum(dim=1).clamp(1e-6,1-1e-6)
    valid=(gt[:,1:]<3)&(gt[:,:-1]<3);change=gt[:,1:]!=gt[:,:-1]
    w=torch.minimum(weights[:,1:],weights[:,:-1])*valid
    w=w*torch.where(change,positive_weight,1)
    loss=torch.where(change,-torch.log1p(-same),-torch.log(same))
    return (loss*w).sum()/w.sum().clamp_min(1)


def probabilities(model,x,xs,segments,mean,scale,device='cpu',core=512):
    pd=np.full(len(x),.5,np.float32);ph=pd.copy();model.eval();model.to(device);radius=model.context
    with torch.inference_mode():
        for a,b in segments:
            for first in range(a,b,core):
                last=min(first+core,b);ix=np.clip(np.arange(first-radius,last+radius),a,b-1)
                direct=torch.from_numpy(temporal.normalize(x[ix],mean,scale).T.copy()).unsqueeze(0).to(device)
                swapped=torch.from_numpy(temporal.normalize(xs[ix],mean,scale).T.copy()).unsqueeze(0).to(device)
                out=torch.sigmoid(model(torch.cat([direct,swapped],0)))[:,:,radius:radius+last-first].cpu().numpy()
                pd[first:last]=.5*(out[0,0]+out[1,0]);ph[first:last]=.5*(out[0,1]+1-out[1,1])
    return pd,ph
