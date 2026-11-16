"""Conservative measured-contact cue; no imputed ball or reference inputs."""
import numpy as np


def contact_cue(data, segments):
    if 'gt' in data:raise ValueError('Labels are not contact features')
    result=np.zeros(len(data['ball']),np.int8);team=np.asarray(data['slot_team'])
    pitch=np.asarray(data['pitch']);lag=10
    for start,end in segments:
        ball=data['ball'][start:end,:2]*pitch
        players=data['players'][start:end,:,:2]*pitch
        seen=np.isfinite(ball).all(axis=1);visible=np.isfinite(players).all(axis=2)
        distance=np.linalg.norm(players-ball[:,None,:],axis=2)
        distance=np.where(np.isfinite(distance),distance,np.inf)
        nearest=distance.argmin(axis=1);side=team[nearest];t=np.arange(end-start)
        close=distance[t,nearest]
        opponent=np.where(team[None,:]!=side[:,None],distance,np.inf).min(axis=1)
        margin=np.full(len(close),np.nan)
        np.subtract(opponent,close,out=margin,where=np.isfinite(opponent)&np.isfinite(close))
        support=np.minimum((visible&(team[None,:]==0)).sum(axis=1),(visible&(team[None,:]==1)).sum(axis=1))
        bv=np.full_like(ball,np.nan);pv=np.full_like(players,np.nan)
        bv[lag:]=(ball[lag:]-ball[:-lag])*(25/lag)
        pv[lag:]=(players[lag:]-players[:-lag])*(25/lag)
        speed=np.linalg.norm(bv,axis=1)
        relative=np.linalg.norm(bv-pv[t,nearest],axis=1)
        continuous=np.convolve(seen.astype(int),np.ones(lag+1,dtype=int),mode='full')[:len(seen)]==lag+1
        usable=(continuous&(support>=3)&(close<=2)&(margin>=2)&
                (speed>=.5)&(speed<=10)&(relative<=3))
        result[start:end]=np.where(usable,np.where(side==0,1,-1),0)
    return result


def contact_probabilities(ph,pd,cue):
    result=ph.copy();mask=(cue!=0)&(pd<.5)
    p=np.clip(ph[mask],1e-6,1-1e-6)
    result[mask]=1/(1+np.exp(-(np.log(p)-np.log1p(-p)+np.log(9)*cue[mask])))
    return result
