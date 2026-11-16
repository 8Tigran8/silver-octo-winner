"""Train-only hard missing-live sampling; no reference inputs at inference."""
import numpy as np
import torch

def hard_centres(gt,seen,segments):
    result=[]
    for a,b in segments:
        missing=~seen[a:b];edges=np.r_[0,np.flatnonzero(missing[1:]!=missing[:-1])+1,b-a];indices=[]
        for x,y in zip(edges[:-1],edges[1:]):
            if missing[x] and y-x>=50:
                ix=np.arange(a+x,a+y,5);indices.extend(ix[gt[ix]<2].tolist())
        result.append(np.asarray(indices,np.int64))
    return result

def dead_loss(logits,gt,missing,focused,reliability=None):
    valid=gt<3;w=valid.float()*(1+missing.float())
    if reliability is not None:w=w*reliability
    if focused:w=w*(1+((gt<2)&missing).float())
    raw=torch.nn.functional.binary_cross_entropy_with_logits(logits[:,0],(gt==2).float(),reduction='none')
    return (raw*w).sum()/w.sum().clamp_min(1)
