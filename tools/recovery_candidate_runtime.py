"""Frozen research-candidate inference, before production adoption."""
from pathlib import Path
import json
import numpy as np
import torch,cloudpickle
from threadpoolctl import threadpool_limits
from mtah import mixed_candidate,mixed_temporal,v4,features2,possession_context
from tools.short_gap_methods import activity_features
ROOT=Path(__file__).resolve().parents[1]


def load():
    from tools.pff_reader import sha
    selected=json.loads((ROOT/'docs/experiments/spell-weighted-finetune/selection.json').read_text())
    neural=ROOT/'reports/spell_weighted_finetune/weighted/model.pt';activity=ROOT/'reports/short_gap_recovery/gap_expert.pkl'
    if sha(neural)!=selected['neural_sha256'] or sha(activity)!=selected['activity_sha256']:raise ValueError('Selected weights changed')
    bundle,parent=mixed_candidate.load_bundle(ROOT/'assets/pff-mixed-candidate/bundle.json')
    meta=json.loads((neural.parent/'meta.json').read_text())
    expert_meta=json.loads((activity.parent/'expert_meta.json').read_text())
    if meta['training_games']!=bundle['training_games']:raise ValueError('Neural training registry differs')
    expected={g.split('/')[1] for g in bundle['training_games'] if g.startswith('pff/')}
    if set(expert_meta['training_games'])!=expected:raise ValueError('Expert training registry differs')
    state=torch.load(neural,map_location='cpu',weights_only=True)
    if not np.array_equal(state['mean'].numpy(),parent[1]) or not np.array_equal(state['scale'].numpy(),parent[2]):raise ValueError('Normalizer changed')
    model=mixed_temporal.MixedTemporal(bundle['spec']);model.load_state_dict(state['state_dict']);model.eval()
    with activity.open('rb') as f:expert=cloudpickle.load(f)
    return v4.load(),(model,parent[1],parent[2]),expert,parent


def probabilities(data,segments,assets,state,expert,parent=None):
    if 'gt' in data:raise ValueError('Labels are not inference inputs')
    bd,bh,_,_=v4.components(data,segments,assets)
    with threadpool_limits(1):
        x,columns=features2.build(data,segments);xs,_=features2.build(data,segments,swap=True)
        x,_=possession_context.extend(x,columns,segments);xs,_=possession_context.extend(xs,columns,segments)
        activity,_=activity_features(data,segments)
        small=expert.predict_proba(activity[::5])[:,1]
    n=len(x);down=[((a+4)//5,(b+4)//5) for a,b in segments]
    model,mean,scale=state
    nd,nh=mixed_temporal.probabilities(model,x[::5],xs[::5],down,mean,scale)
    nd,nh=v4.neural_upsample(nd,nh,segments,n)
    activity=v4.neural_upsample(small,small,segments,n)[0]
    pd,ph=.5*bd+.5*nd,.5*bh+.5*nh
    missing=~np.isfinite(data['ball'][:,:2]).all(axis=1)
    pd[missing]=.75*nd[missing]+.25*activity[missing]
    if parent is None:return pd,ph
    old,old_mean,old_scale=parent
    od,oh=mixed_temporal.probabilities(old,x[::5],xs[::5],down,old_mean,old_scale)
    od,oh=v4.neural_upsample(od,oh,segments,n)
    return (pd,ph),(.5*bd+.5*od,.5*bh+.5*oh)
