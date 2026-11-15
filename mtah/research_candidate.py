"""Local candidate inference; uses geometry and frozen public-trained weights only."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from . import v4,features2,possession_context as context,temporal,reliability


def load_bundle(path):
    path=Path(path);bundle=json.loads(path.read_text());states=[]
    for item in bundle['models']:
        file=path.parent/item['path']
        if hashlib.sha256(file.read_bytes()).hexdigest()!=item['sha256']:raise ValueError('Research model hash changed')
        state=torch.load(file,map_location='cpu',weights_only=True);model=temporal.Temporal();model.load_state_dict(state['state_dict']);model.eval()
        states.append((model,state['mean'].numpy(),state['scale'].numpy()))
    return bundle,states


def probabilities(data,segments,assets,states,spec):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    bd,bh,_,_=v4.components(data,segments,assets)
    with threadpool_limits(1):
        x,columns=features2.build(data,segments);xs,_=features2.build(data,segments,swap=True)
        gap=x[:,columns.index('gap')];seen=x[:,columns.index('seen_w151')]
        x,_=context.extend(x,columns,segments);xs,_=context.extend(xs,columns,segments)
        spans=[((a+4)//5,(b+4)//5) for a,b in segments];pairs=[]
        for model,mean,scale in states:
            pd,ph=temporal.probabilities(model,x[::5],xs[::5],spans,mean,scale,device='cpu')
            pairs.append(v4.neural_upsample(pd,ph,segments,len(x)))
        nd=np.mean([p[0] for p in pairs],axis=0);nh=np.mean([p[1] for p in pairs],axis=0)
    return reliability.mix(spec,bd,bh,nd,nh,gap,seen)
