"""Model-independent toy tracking, end-to-end recognition diagnostic, never tuning."""
import json
from pathlib import Path
import sys
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
from mtah import mixed_candidate, v4
from mtah.methods import boost
from pff_mixed_score import decoder
from decoder_development import metric


def controlled_tracking(owner, n=1500):
    """22 players, ball 0.5 m from a moving controller, other team >= 10 m away.

    Constant team ownership is constructed before inference, not inferred from it.
    This tests obvious geometric control; no referee/deflection truth is implied.
    """
    t = np.arange(n) / 25
    home = np.array([[.06,.5],[.20,.15],[.20,.38],[.20,.62],[.20,.85],
                     [.35,.2],[.35,.4],[.35,.6],[.35,.8],[.43,.3],[.43,.7]])
    away = home.copy(); away[:,0] = 1 - away[:,0]
    xy = np.concatenate([home, away])
    players = np.zeros((n,22,3))
    players[:,:,:2] = xy
    # Smooth bounded 2D motion; speed stored in physical m/s from the same path.
    motion = np.c_[.025*np.sin(t/3), .035*np.sin(t/2)]
    players[:,:,:2] += motion[:,None,:]
    velocity = np.c_[.025/3*np.cos(t/3)*105, .035/2*np.cos(t/2)*68]
    players[:,:,2] = np.linalg.norm(velocity,axis=1)[:,None]
    controller = 6 if owner == 0 else 17
    ball = np.zeros((n,3));ball[:,:2] = players[:,controller,:2] + [0.5/105,0]
    return dict(ball=ball, players=players, slot_team=np.repeat([0,1],11), pitch=(105.,68.))


def main():
    assets=v4.load();_,state=mixed_candidate.load_bundle(ROOT/'assets/pff-mixed-candidate/bundle.json')
    rows=[]
    for owner in (0,1):
        for observation in ('complete','missing_ball_10s','partial_y_10s'):
            data=controlled_tracking(owner)
            if observation=='missing_ball_10s':data['ball'][500:750,:2]=np.nan
            if observation=='partial_y_10s':data['ball'][500:750,1]=np.nan
            pd,ph=mixed_candidate.probabilities(data,[(0,1500)],assets,state)
            pred=boost.label(pd,ph,[(0,1500)],1500,decoder())
            gt=np.full(1500,owner,np.int8)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',RuntimeWarning)
                score=metric(gt,pred)
            rows.append(dict(owner='HA'[owner],observation=observation,**score,
                             correct_frame_fraction=float(np.mean(pred==gt)),
                             predicted_counts={label:int(np.sum(pred==i)) for i,label in enumerate('HADX')}))
    result=dict(scope='Synthetic geometric recognition diagnostic only; not real-match quality or model selection',
                construction='22 players; smooth bounded motion; ball at 0.5m from known H/A controller; no possession transitions',records=rows)
    path=ROOT/'reports/independent_audit/synthetic_geometry.json'
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
