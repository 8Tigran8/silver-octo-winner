"""Training-only duration weights; no inference dependency on possession truth."""
import numpy as np
import torch
import torch.nn.functional as F


def live_spell_weights(gt,segments,fps=5):
    weight=np.ones(len(gt),np.float32)
    for a,b in segments:
        edges=np.r_[a,np.flatnonzero(gt[a+1:b]!=gt[a:b-1])+a+1,b]
        for start,end in zip(edges[:-1],edges[1:]):
            if gt[start]<2:weight[start:end]=np.clip(np.sqrt(10/((end-start)/fps)),1,4)
    return weight


def loss(logits,gt,dead_weight,live_weight,missing,weighted):
    valid=gt<3;live=gt<2
    wd=dead_weight*valid*(1+missing.float() if weighted else 1)
    wh=live*(live_weight if weighted else 1)
    d=F.binary_cross_entropy_with_logits(logits[:,0],(gt==2).float(),reduction='none')
    h=F.binary_cross_entropy_with_logits(logits[:,1],(gt==0).float(),reduction='none')
    return (d*wd).sum()/wd.sum().clamp_min(1)+(h*wh).sum()/wh.sum().clamp_min(1)
