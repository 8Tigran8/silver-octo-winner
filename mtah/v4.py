"""Frozen external-trained temporal fusion. CPU inference, no GT reads."""
from pathlib import Path
import hashlib,json
import numpy as np
from threadpoolctl import threadpool_limits
from . import features2,stack,possession_context as context,temporal
from .methods import boost
from .config import ROOT,HALVES,N_FRAMES,D,LABELS


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load():
    import torch,cloudpickle
    folder=Path(ROOT)/'assets/v4';manifest=json.loads((folder/'manifest.json').read_text())
    for item in manifest['files']:
        p=folder/item['path']
        if p.stat().st_size!=item['size'] or digest(p)!=item['sha256']:raise ValueError('Frozen model asset verification failed')
    with (folder/'base_external.pkl').open('rb') as f:base=cloudpickle.load(f)
    with (folder/'team_external.pkl').open('rb') as f:team=cloudpickle.load(f)
    state=torch.load(folder/'temporal_state.pt',map_location='cpu',weights_only=True);model=temporal.Temporal();model.load_state_dict(state['state_dict']);model.eval()
    return base,team,model,state['mean'].numpy(),state['scale'].numpy(),manifest


def neural_upsample(pd,ph,segments,n):
    nd=np.full(n,.5);nh=nd.copy()
    for a,b in segments:
        lo,hi=(a+4)//5,(b+4)//5;times=np.arange(lo,hi)*5;full=np.arange(a,b)
        nd[a:b]=np.interp(full,times,pd[lo:hi]);nh[a:b]=np.interp(full,times,ph[lo:hi])
    return nd,nh


def components(data,segments,assets=None):
    import torch
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    base,team,model,mean,scale,manifest=assets or load()
    with threadpool_limits(1):
        x,cols=features2.build(data,segments);xs,_=features2.build(data,segments,swap=True)
        pd,ph=boost.predict(base['stage1'],x,xs);pd,ph=stack.predict(base['stage2'],pd,ph,segments)
        x,columns=context.extend(x,cols,segments);xs,_=context.extend(xs,cols,segments)
        if hashlib.sha256(json.dumps(columns,separators=(',',':')).encode()).hexdigest()!=manifest['feature_columns_sha256']:raise ValueError('Frozen feature schema changed')
        ph=.5*ph+.5*context.predict(team,x,xs)
        spans=[((a+4)//5,(b+4)//5) for a,b in segments]
        nd,nh=temporal.probabilities(model,x[::5],xs[::5],spans,mean,scale,device='cpu')
        nd,nh=neural_upsample(nd,nh,segments,len(x))
    return pd,ph,nd,nh


def probabilities(data,segments,assets=None):
    pd,ph,nd,nh=components(data,segments,assets)
    return .5*pd+.5*nd,.5*ph+.5*nh


def run(data):
    # Inherit the verified v3 duration/decoder defaults, including min lengths.
    from .methods import boost_train
    params=dict(boost_train.decoder_params(),**json.loads((Path(ROOT)/'mtah/methods/v2_params.json').read_text())['decoder'])
    segments=[(a,b+1) for _,a,b in HALVES];pd,ph=probabilities(data,segments)
    play=[(max(a,boost.KICKOFF.get(i,a)),b) for i,(a,b) in enumerate(segments)]
    codes=boost.label(pd,ph,play,N_FRAMES,params,boost.kick_frames(data,params))
    for i,(a,b) in enumerate(segments):codes[a:max(a,boost.KICKOFF.get(i,a))]=D
    return np.array(LABELS)[codes]
