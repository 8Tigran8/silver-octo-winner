"""Reference-blind old/new temporal blending for a bounded local followup."""
import numpy as np


def ood_alpha(x,mean,scale,segments,spec):
    valid=np.isfinite(x);z=np.minimum(np.abs((np.where(valid,x,mean)-mean)/scale),12.)
    z=np.where(valid,z,-np.inf);top=np.partition(z,-spec['top_features'],axis=1)[:,-spec['top_features']:]
    ok=np.isfinite(top);count=ok.sum(1);score=np.divide(np.where(ok,top,0).sum(1),count,out=np.full(len(x),np.inf),where=count>0)
    alpha=np.clip((spec['high']-score)/(spec['high']-spec['low']),0,1)
    result=np.zeros(len(x))
    for a,b in segments:
        width=min(spec['smooth_steps'],b-a);kernel=np.ones(width);values=alpha[a:b]
        result[a:b]=np.convolve(values,kernel,mode='same')/np.convolve(np.ones(len(values)),kernel,mode='same')
    return result


def upsample(values,meta):
    result=np.full(meta['full_length'],.5)
    for (a,b),(lo,hi) in zip(meta['full_segments'],meta['segments']):result[a:b]=np.interp(np.arange(a,b),np.arange(lo,hi)*5,values[lo:hi])
    return result


def combine(spec,old_dead,old_home,new_dead,new_home,x,mean,scale,meta):
    kind=spec['kind']
    if kind=='temperature':
        def soften(p):
            q=np.clip(p,1e-6,1-1e-6);return 1/(1+np.exp(-(np.log(q)-np.log1p(-q))/spec['temperature']))
        return soften(new_dead),soften(new_home)
    if kind=='alpha':wd,wh=spec['dead'],spec['home']
    elif kind=='ood':wd=wh=upsample(ood_alpha(x,mean,scale,meta['segments'],spec),meta)
    else:raise ValueError('Unregistered temporal blend')
    return (1-wd)*old_dead+wd*new_dead,(1-wh)*old_home+wh*new_home
