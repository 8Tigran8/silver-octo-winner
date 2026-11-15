"""Frozen65-game research inference; no training cache or labels required."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from . import v4,features2,possession_context as context,mixed_temporal
from .config import ROOT


def load_bundle(path):
    path=Path(path);bundle=json.loads(path.read_text())
    baseline=Path(ROOT)/'assets/v4/manifest.json'
    if hashlib.sha256(baseline.read_bytes()).hexdigest()!=bundle['baseline_asset_manifest_sha256']:raise ValueError('Frozen baseline components changed')
    old=json.loads(baseline.read_text())['model_training_games']
    protocol=json.loads((Path(ROOT)/'docs/experiments/more-data/pff_protocol.json').read_text())
    allowed=set(old)|{'pff/'+g['game_id'] for g in protocol['training']}
    forbidden={'pff/'+g['game_id'] for part in ['development','final_test','country_bridge_excluded_from_training_and_selection'] for g in protocol.get(part,[])}
    if set(bundle['training_games'])!=allowed or allowed&forbidden:raise ValueError('Unapproved/leaking full training registry')
    model_path=path.parent/bundle['model_path'];metadata=path.parent/bundle['metadata_path']
    for file,key in [(model_path,'model_sha256'),(metadata,'metadata_sha256')]:
        if hashlib.sha256(file.read_bytes()).hexdigest()!=bundle[key]:raise ValueError('Frozen mixed model/metadata changed')
    info=json.loads(metadata.read_text())
    if set(info['train'])!=allowed or info['excluded'] or info['spec']!=bundle['spec'] or info['candidate']!=bundle['candidate'] or info['mixed_plan_sha256']!=bundle['training_plan_sha256'] or not bundle['normalizer_checked_train_only']:raise ValueError('Checkpoint is not the approved full fit')
    state=torch.load(model_path,map_location='cpu',weights_only=True)
    model=mixed_temporal.MixedTemporal(bundle['spec']);model.load_state_dict(state['state_dict']);model.eval()
    return bundle,(model,state['mean'].numpy(),state['scale'].numpy())


def paired_probabilities(data,segments,assets,state):
    if 'gt' in data:raise ValueError('Reference labels are not model inputs')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    bd,bh,od,oh=v4.components(data,segments,assets)
    with threadpool_limits(1):
        x,cols=features2.build(data,segments);xs,_=features2.build(data,segments,swap=True)
        x,_=context.extend(x,cols,segments);xs,_=context.extend(xs,cols,segments)
        parts=[((a+4)//5,(b+4)//5) for a,b in segments];model,mean,scale=state
        nd,nh=mixed_temporal.probabilities(model,x[::5],xs[::5],parts,mean,scale)
        nd,nh=v4.neural_upsample(nd,nh,segments,len(x))
    return (.5*bd+.5*od,.5*bh+.5*oh),(.5*bd+.5*nd,.5*bh+.5*nh)


def probabilities(data,segments,assets,state):
    return paired_probabilities(data,segments,assets,state)[1]
