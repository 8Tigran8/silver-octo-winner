"""Binary gamma-one focal posterior link, without fitted calibration parameters."""
import torch

def focal_posterior(p):
    """Invert stationary equation q*Lpos'(p)+(1-q)*Lneg'(p)=0."""
    p=p.clamp(1e-6,1-1e-6)
    a=p/(1-p)-torch.log1p(-p)
    b=(1-p)/p-torch.log(p)
    return a/(a+b)

class FocalOwnershipLink(torch.nn.Module):
    def __init__(self,base):
        super().__init__();self.base=base;self.context=base.context
    def forward(self,x):
        logits=self.base(x);out=logits.clone()
        out[:,1]=torch.logit(focal_posterior(torch.sigmoid(logits[:,1])).clamp(1e-6,1-1e-6))
        return out

def weighted_ownership_bce(logits,gt,weights,missing):
    live=gt<2
    raw=torch.nn.functional.binary_cross_entropy_with_logits(logits[:,1],(gt==0).float(),reduction='none')
    w=live*weights*(1+2*missing.float())
    return (raw*w).sum()/w.sum().clamp_min(1)
