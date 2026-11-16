"""Training-only conditional ownership objective; D is served by frozen parent."""
import numpy as np
import torch
import torch.nn.functional as F

def short_centres(gt,segments,fps=5):
    result=[]
    for a,b in segments:
        edges=np.r_[a,np.flatnonzero(gt[a+1:b]!=gt[a:b-1])+a+1,b]
        result.append(np.array([(x+y-1)//2 for x,y in zip(edges[:-1],edges[1:]) if gt[x]<2 and y-x<3*fps],np.int64))
    return result

def ownership_loss(logits,gt,duration_weight,missing,focused):
    live=gt<2;target=(gt==0).float();raw=F.binary_cross_entropy_with_logits(logits[:,1],target,reduction='none')
    weight=live*duration_weight
    if focused:
        weight=weight*(1+2*missing.float())
        p=torch.sigmoid(logits[:,1]);pt=torch.where(gt==0,p,1-p)
        raw=raw*(1-pt) # gamma1 focal term; affects difficult team decisions only
    return (raw*weight).sum()/weight.sum().clamp_min(1)
