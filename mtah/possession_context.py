"""Observed ball-contact context for a team-only classifier.

All inputs derive from tracking features, never possession/dead labels. Carries
and rolling windows reset at independently supplied period boundaries.
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from .methods import boost

CARRY = ("margin", "dmin", "h_d1", "a_d1", "h_n3", "a_n3", "bspeed")


def endpoints(values, observed):
    n = len(values)
    at = np.arange(n)
    valid = observed & np.isfinite(values)
    previous = np.maximum.accumulate(np.where(valid, at, -1))
    following = np.minimum.accumulate(np.where(valid, at, n)[::-1])[::-1]
    before = np.where(previous >= 0, values[np.maximum(previous,0)], np.nan)
    after = np.where(following < n, values[np.minimum(following,n-1)], np.nan)
    return before, after, previous, following


def extend(x, names, segments):
    columns = {name:i for i,name in enumerate(names)}
    extra_names = [f"observed_{key}_{side}" for key in CARRY for side in ("previous","following")]
    extra_names += ["observed_margin_previous_decay", "observed_margin_following_decay",
                    "observed_endpoint_agreement", "observed_endpoint_margin_change"]
    extra_names += [f"observed_margin_mean_{w}" for w in (25,75,250)]
    result = np.full((len(x),len(extra_names)),np.nan,np.float32)
    for a,b in segments:
        seen=x[a:b,columns["seen"]]>.5
        values=[]
        for key in CARRY:
            before,after,previous,following=endpoints(x[a:b,columns[key]],seen)
            values.extend([before,after])
            if key=="margin":
                margin_before,margin_after=before,after
                times=np.arange(b-a)
                decay_before=before*np.exp(-(times-previous)/100.)
                decay_after=after*np.exp(-(following-times)/100.)
        values += [decay_before,decay_after,
                   np.where(np.isfinite(margin_before)&np.isfinite(margin_after),
                            (np.sign(margin_before)==np.sign(margin_after)).astype(float),np.nan),
                   margin_after-margin_before]
        visible=seen & np.isfinite(x[a:b,columns["margin"]])
        margin=np.where(visible,x[a:b,columns["margin"]],0.)
        for window in (25,75,250):
            width=min(window,b-a)
            kernel=np.ones(width)
            numerator=np.convolve(margin,kernel,mode="same")
            denominator=np.convolve(visible.astype(float),kernel,mode="same")
            values.append(np.divide(numerator,denominator,out=np.full(b-a,np.nan),where=denominator>0))
        result[a:b]=np.stack(values,axis=1)
    return np.concatenate([x,result],axis=1),list(names)+extra_names


def rows(x,xs,gt,stride=4):
    idx=np.arange(0,len(gt),stride)
    idx=idx[gt[idx]<2]
    return np.concatenate([x[idx],xs[idx]]),np.concatenate([gt[idx]==0,gt[idx]==1]).astype(np.int8)


def fit(parts):
    x=np.concatenate([r[0] for r in parts]);y=np.concatenate([r[1] for r in parts])
    model=HistGradientBoostingClassifier(random_state=0,early_stopping=False,**boost.MODEL)
    return model.fit(x,y)


def predict(model,x,xs):
    return .5*(model.predict_proba(x)[:,1]+1-model.predict_proba(xs)[:,1])
