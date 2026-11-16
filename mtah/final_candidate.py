"""Frozen release of uniform D, focused ownership and player-motion experts."""
import json
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from . import recovery_candidate,mixed_temporal,activity_temporal,v4,expanded_team
from .recovery_features import activity_features
from .config import ROOT
sha=recovery_candidate.sha

def load_bundle(path):
    path=Path(path);bundle=json.loads(path.read_text());parent=Path(ROOT)/'assets/recovery-candidate/bundle.json'
    if sha(parent)!=bundle['parent_bundle_sha256']:raise ValueError('Parent bundle changed')
    for name,h in bundle['files_sha256'].items():
        if sha(path.parent/name)!=h:raise ValueError('Final artifact changed: '+name)
    pb,ps=recovery_candidate.load_bundle(parent)
    if bundle['training_games']!=pb['training_games'] or bundle['spec']!=pb['spec'] or bundle['decoder']!=pb['decoder']:raise ValueError('Registry/spec/decoder changed')
    models=[]
    for kind in ['dead','owner']:
        meta=json.loads((path.parent/(kind+'_meta.json')).read_text())
        if meta['training_games']!=pb['training_games'] or meta['model_sha256']!=sha(path.parent/(kind+'.pt')):raise ValueError('Specialist provenance differs')
        st=torch.load(path.parent/(kind+'.pt'),map_location='cpu',weights_only=True)
        if not np.array_equal(st['mean'].numpy(),ps[1][1]) or not np.array_equal(st['scale'].numpy(),ps[1][2]):raise ValueError('Specialist normalizer differs')
        m=mixed_temporal.MixedTemporal(bundle['spec']);m.load_state_dict(st['state_dict']);m.eval();models.append(m)
    meta=json.loads((path.parent/'activity_meta.json').read_text())
    if set(meta['training_games'])!={g.split('/')[1] for g in pb['training_games'] if g.startswith('pff/')}:raise ValueError('Activity training registry differs')
    if meta['kind']!='temporal' or meta['model_sha256']!=sha(path.parent/'activity.pt'):raise ValueError('Activity provenance differs')
    st=torch.load(path.parent/'activity.pt',map_location='cpu',weights_only=True);mean,scale=st['mean'].numpy(),st['scale'].numpy()
    if mean.shape!=(64,) or scale.shape!=(64,) or not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale<=0).any():raise ValueError('Invalid activity normalization')
    model=activity_temporal.ActivityModel('temporal');model.load_state_dict(st['state_dict']);model.eval()
    return bundle,(ps,*models,(model,mean,scale))

def probabilities(data,segments,state):
    if 'gt' in data:raise ValueError('Reference labels are not inference inputs')
    parent,dead,owner,activity=state;assets,neural,_,expected_columns=parent
    bd,bh,_,_=v4.components(data,segments,assets)
    with threadpool_limits(1):
        x,xs,_=expanded_team.feature_pair(data,segments);a,names=activity_features(data,segments)
        if names!=expected_columns:raise ValueError('Activity schema differs')
    n=len(x);small=[((a+4)//5,(b+4)//5) for a,b in segments];model,mean,scale=neural
    nd,nh=mixed_temporal.probabilities(model,x[::5],xs[::5],small,mean,scale);nd,nh=v4.neural_upsample(nd,nh,segments,n)
    dd,_=mixed_temporal.probabilities(dead,x[::5],xs[::5],small,mean,scale);dd=v4.neural_upsample(dd,dd,segments,n)[0]
    _,hh=mixed_temporal.probabilities(owner,x[::5],xs[::5],small,mean,scale);hh=v4.neural_upsample(hh,hh,segments,n)[0]
    am,mu,sd=activity;ad=activity_temporal.probabilities(am,a[::5],small,mu,sd);ad=v4.neural_upsample(ad,ad,segments,n)[0]
    pd,ph=.5*bd+.5*nd,.5*bh+.5*nh;missing=~np.isfinite(data['ball'][:,:2]).all(axis=1)
    pd[missing]=.75*dd[missing]+.25*ad[missing];ph[missing]=hh[missing]
    return pd,ph
