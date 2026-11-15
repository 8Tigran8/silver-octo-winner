"""Training-only spell and transition weighting; no inference reference input."""
import numpy as np
import torch
import torch.nn.functional as F


def training_weights(gt,segments):
    weights=np.zeros(len(gt),np.float32)
    for a,b in segments:
        y=np.asarray(gt[a:b]);w=np.where(y<3,1.,0.).astype(np.float32)
        edges=np.flatnonzero(np.r_[True,y[1:]!=y[:-1],True])
        for lo,hi in zip(edges[:-1],edges[1:]):
            if y[lo]<2:w[lo:hi]=np.clip(np.sqrt(25./(hi-lo)),.5,3.)
        boundaries=np.flatnonzero((y[1:]!=y[:-1])&(y[1:]<3)&(y[:-1]<3))+1
        mask=np.zeros(len(y),bool)
        for pos in boundaries:mask[max(0,pos-5):min(len(y),pos+5)]=True
        w[mask]*=2;weights[a:b]=np.minimum(w,4)
    return weights


def weighted(logits,gt,weights):
    valid=gt<3;live=gt<2
    dead=F.binary_cross_entropy_with_logits(logits[:,0],(gt==2).float(),reduction='none')
    home=F.binary_cross_entropy_with_logits(logits[:,1],(gt==0).float(),reduction='none')
    wd=weights*valid;wh=weights*live
    return (dead*wd).sum()/wd.sum().clamp_min(1)+(home*wh).sum()/wh.sum().clamp_min(1)
