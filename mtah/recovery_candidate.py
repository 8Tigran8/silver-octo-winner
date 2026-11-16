"""Weighted short-possession model with player-activity support during ball gaps.

Frozen external-trained weights only. No event, target or reference-label input.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch,cloudpickle
from threadpoolctl import threadpool_limits
from . import mixed_temporal,v4,features2,possession_context
from .recovery_features import activity_features
from .config import ROOT


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_bundle(path):
    path=Path(path);bundle=json.loads(path.read_text());root=Path(ROOT)
    for name,digest in bundle['files_sha256'].items():
        if sha(path.parent/name)!=digest:raise ValueError('Recovery model/metadata changed: '+name)
    parent=root/'assets/pff-mixed-candidate/bundle.json'
    if sha(parent)!=bundle['parent_bundle_sha256'] or sha(root/'assets/v4/manifest.json')!=bundle['base_manifest_sha256']:
        raise ValueError('Inherited model registry changed')
    original=json.loads(parent.read_text());meta=json.loads((path.parent/'neural_meta.json').read_text())
    expert_meta=json.loads((path.parent/'activity_meta.json').read_text())
    if meta['training_games']!=original['training_games'] or bundle['training_games']!=original['training_games']:
        raise ValueError('Neural training registry differs')
    pff={g.split('/')[1] for g in original['training_games'] if g.startswith('pff/')}
    if set(expert_meta['training_games'])!=pff:raise ValueError('Activity training registry differs')
    protocol=json.loads((root/'docs/experiments/more-data/pff_protocol.json').read_text())
    forbidden={g['game_id'] for split in ['development','final_test','country_bridge_excluded_from_training_and_selection'] for g in protocol[split]}
    if pff & forbidden:raise ValueError('Training/validation overlap')
    state=torch.load(path.parent/'neural_state.pt',map_location='cpu',weights_only=True)
    mean,scale=state['mean'].numpy(),state['scale'].numpy()
    if mean.shape!=(108,) or scale.shape!=(108,) or not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale<=0).any():
        raise ValueError('Invalid normalization')
    model=mixed_temporal.MixedTemporal(bundle['spec']);model.load_state_dict(state['state_dict']);model.eval()
    with (path.parent/'activity.pkl').open('rb') as f:expert=cloudpickle.load(f)
    if expert.n_features_in_!=64 or not np.array_equal(expert.classes_,[0,1]):raise ValueError('Invalid activity classifier')
    return bundle,(v4.load(),(model,mean,scale),expert,expert_meta['columns'])


def probabilities(data,segments,state):
    if 'gt' in data:raise ValueError('Reference labels are not inference inputs')
    assets,neural,expert,expected_columns=state
    bd,bh,_,_=v4.components(data,segments,assets)
    with threadpool_limits(1):
        x,columns=features2.build(data,segments);xs,_=features2.build(data,segments,swap=True)
        x,_=possession_context.extend(x,columns,segments);xs,_=possession_context.extend(xs,columns,segments)
        activity,names=activity_features(data,segments)
        if names!=expected_columns:raise ValueError('Activity feature schema differs')
        small=expert.predict_proba(activity[::5])[:,1]
    n=len(x);down=[((a+4)//5,(b+4)//5) for a,b in segments]
    model,mean,scale=neural
    nd,nh=mixed_temporal.probabilities(model,x[::5],xs[::5],down,mean,scale)
    nd,nh=v4.neural_upsample(nd,nh,segments,n)
    activity=v4.neural_upsample(small,small,segments,n)[0]
    pd,ph=.5*bd+.5*nd,.5*bh+.5*nh
    missing=~np.isfinite(data['ball'][:,:2]).all(axis=1)
    pd[missing]=.75*nd[missing]+.25*activity[missing]
    return pd,ph
