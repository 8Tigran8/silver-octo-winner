"""Train two prospectively registered65-game models; final/dev excluded."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context
import json
import os
from pathlib import Path
import sys
import time
import subprocess
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from threadpoolctl import threadpool_limits
import temporal_trial as trial
from pff_reader import sha
from pff_preflight import convert
from pff_development import geometry,spans
from mtah import mixed_temporal,temporal,expanded_team,degrade

PLAN=ROOT/'docs/experiments/pff-mixed/plan.json';OUT=ROOT/'reports/pff_mixed'
SOURCES=['tools/pff_mixed_train.py','mtah/mixed_temporal.py','mtah/temporal.py','mtah/features2.py','mtah/possession_context.py','mtah/degrade.py','tools/pff_reader.py']


def plan():return json.loads(PLAN.read_text())


def spec(name):
    matches=[s for s in plan()['candidates'] if s['name']==name]
    if len(matches)!=1:raise ValueError('Unregistered model')
    return matches[0]


def prepare(game):
    p=plan()
    if game not in p['pff_train_ids']:raise ValueError('Only original36PFFtraining games; dev/final/bridges prohibited')
    receipt=ROOT/'reports/pff_access'/('kloppy_'+game+'_preflight.json')
    if not receipt.exists():convert(game)
    data,info=geometry(game);parts=spans(data['period']);dummy=np.where(data['period']>0,0,3).astype(np.int8)
    noisy=degrade.degrade(dict(data,gt=dummy),seed=7,params=expanded_team.BLIND);noisy.pop('gt')
    ref=ROOT/'reports/pff_access/references'/('kloppy_'+game+'.npz')
    if sha(ref)!=info['reference_cache_sha256']:raise ValueError('Reference changed after source preflight')
    # Geometry-only features are written before the supervision array is read.
    for scenario,observed in [('natural_detected',data),('label_blind_seed7',noisy)]:
        dest=OUT/'sequences'/('pff__'+game)/scenario;dest.mkdir(parents=True,exist_ok=True)
        if (dest/'meta.json').exists():
            if json.loads((dest/'meta.json').read_text())['mixed_plan_sha256']!=sha(PLAN):raise ValueError('Stale completed cache')
            continue
        with threadpool_limits(1):x,xs,cols=expanded_team.feature_pair(observed,parts)
        if len(cols)!=108:raise ValueError('Feature schema changed')
        np.save(dest/'x.npy',x[::5].astype(np.float32));np.save(dest/'xs.npy',xs[::5].astype(np.float32));del x,xs
        with np.load(ref,allow_pickle=False) as z:gt=z['gt'][::5]
        np.save(dest/'gt.npy',gt)
        meta={'game':'pff/'+game,'scenario':scenario,'segments':[((a+4)//5,(b+4)//5) for a,b in parts],
            'full_segments':parts,'full_length':len(data['ball']),'features':108,'mixed_plan_sha256':sha(PLAN),
            'reader_sha256':sha(ROOT/'tools/pff_reader.py'),'geometry_sha256':info['geometry_cache_sha256'],
            'reference_sha256':info['reference_cache_sha256'],'feature_sha256':{f:sha(dest/f) for f in ['x.npy','xs.npy','gt.npy']}}
        (dest/'meta.json').write_text(json.dumps(meta,indent=2)+'\n')
    return game


def split(fold):
    old,test=trial.game_split('full' if fold=='full' else 'fold'+str(fold));p=plan()
    if sha(trial.PLAN)!=p['old_fold_plan_sha256']:raise ValueError('Old folds changed')
    pff=['pff/'+g for g in p['pff_train_ids']];train=old+pff
    forbidden={'pff/'+g for g in p['pff_dev_ids']+p['final_ids']}
    if set(train)&(set(test)|forbidden) or set(train)-set(p['training_games']):raise ValueError('Leak/unapproved training')
    return train,test


def sequences(train,verify=True):
    seq=trial.sequences([g for g in train if not g.startswith('pff/')])
    for game in train:
        if not game.startswith('pff/'):continue
        seq[game]=[]
        for scenario in ['natural_detected','label_blind_seed7']:
            folder=OUT/'sequences'/game.replace('/','__')/scenario;meta=json.loads((folder/'meta.json').read_text())
            if meta['mixed_plan_sha256']!=sha(PLAN) or meta['reader_sha256']!=sha(ROOT/'tools/pff_reader.py'):raise ValueError('Stale training source/reader')
            if verify and any(sha(folder/f)!=d for f,d in meta['feature_sha256'].items()):raise ValueError('Training feature/reference bytes changed')
            seq[game].append(dict(meta=meta,**{k:np.load(folder/(k+'.npy'),mmap_mode='r') for k in ['x','xs','gt']}))
    return seq


def fit(name,fold,device):
    cfg=plan()['architecture'];s=spec(name);dest=OUT/'models'/(name+'_fold'+str(fold));dest.mkdir(parents=True,exist_ok=True)
    if (dest/'meta.json').exists():raise ValueError('Do not overwrite completed fit')
    source_hashes={f:sha(ROOT/f) for f in SOURCES};source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    train,test=split(fold);seq=sequences(train);mean,scale=trial.normalization(seq)
    torch.set_num_threads(2);torch.manual_seed(cfg['seed']);torch.use_deterministic_algorithms(True);rng=np.random.default_rng(cfg['seed'])
    model=mixed_temporal.MixedTemporal(s,channels=cfg['channels'],dropout=cfg['dropout']).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'],foreach=False)
    groups=[[g for g in train if '/idsse_' in g],[g for g in train if '/metrica_' in g],[g for g in train if g.startswith('skillcorner/')],[g for g in train if g.startswith('pff/')]]
    if any(not g for g in groups):raise ValueError('Missing training provider')
    radius=s['context_steps'];corelen=cfg['core_steps'];started=time.perf_counter();losses=[]
    for step in range(cfg['steps']):
        bx=[];by=[];bw=[]
        for _ in range(cfg['batch']):
            group=groups[int(rng.integers(4))];game=group[int(rng.integers(len(group)))];rows=seq[game];row=rows[int(rng.integers(len(rows)))];parts=row['meta']['segments']
            a,b=parts[int(rng.integers(len(parts)))];core=int(rng.integers(a,max(a+1,b-corelen+1)));ix=np.clip(np.arange(core-radius,core+corelen+radius),a,b-1)
            swap=bool(rng.integers(2));x=row['xs' if swap else 'x'][ix];y=np.array(row['gt'][ix],copy=True)
            if swap:y=np.where(y<2,1-y,y)
            w=np.ones(len(ix),np.float32)
            if game.startswith('skillcorner/'):w[y==2]=.25
            y[:radius]=3;y[radius+corelen:]=3
            bx.append(temporal.normalize(x,mean,scale).T);by.append(y);bw.append(w)
        x=torch.from_numpy(np.stack(bx)).to(device);y=torch.from_numpy(np.stack(by).astype(np.int64)).to(device);w=torch.from_numpy(np.stack(bw)).to(device)
        opt.zero_grad(set_to_none=True);logits=model(x);loss=temporal.loss(logits,y,w)
        if s['boundary_loss_weight']:loss=loss+s['boundary_loss_weight']*mixed_temporal.boundary_loss(logits,y,w,s['boundary_positive_weight'])
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();losses.append(float(loss.detach().cpu()))
        if (step+1)%1000==0:print(name,fold,step+1,'loss',round(float(np.mean(losses[-100:])),5),'seconds',round(time.perf_counter()-started,1),flush=True)
    if any(sha(ROOT/f)!=digest for f,digest in source_hashes.items()):raise ValueError('Training source changed during fit')
    torch.save({'state_dict':model.cpu().state_dict(),'mean':torch.from_numpy(mean),'scale':torch.from_numpy(scale)},dest/'model.pt')
    meta={'candidate':name,'spec':s,'train':train,'excluded':test,'plan_sha256':sha(trial.PLAN),'mixed_plan_sha256':sha(PLAN),
        'device':device,'seed':cfg['seed'],'steps':cfg['steps'],'seconds':time.perf_counter()-started,'model_sha256':sha(dest/'model.pt'),
        'training_loss_last100':float(np.mean(losses[-100:])),
        'source_commit':source_commit,'source_sha256':source_hashes,
        'training_sequences':{g:[sha((OUT/'sequences'/g.replace('/','__')/r['meta']['scenario']/'meta.json')) if g.startswith('pff/') else sha(trial.folder(g,r['meta']['scenario'])/'meta.json') for r in rows] for g,rows in seq.items()}}
    (dest/'meta.json').write_text(json.dumps(meta,indent=2)+'\n');print(name,fold,'complete',flush=True)


def load(name,fold):
    folder=OUT/'models'/(name+'_fold'+str(fold));meta=json.loads((folder/'meta.json').read_text());train,test=split(fold)
    if meta['train']!=train or meta['excluded']!=test or meta['mixed_plan_sha256']!=sha(PLAN) or meta['model_sha256']!=sha(folder/'model.pt'):raise ValueError('Mixed model lineage/hash mismatch')
    if any(sha(ROOT/f)!=digest for f,digest in meta['source_sha256'].items()):raise ValueError('Training/feature source changed after fit')
    state=torch.load(folder/'model.pt',map_location='cpu',weights_only=True);mean,scale=trial.normalization(sequences(train,verify=False))
    if not np.array_equal(mean,state['mean'].numpy()) or not np.array_equal(scale,state['scale'].numpy()):raise ValueError('Not actual train-only normalization')
    model=mixed_temporal.MixedTemporal(spec(name));model.load_state_dict(state['state_dict']);model.eval()
    return (model,mean,scale),dict(meta,normalizer_recomputed_train_only=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['prepare','fit']);p.add_argument('--jobs',type=int,choices=[1,2],default=2);p.add_argument('--device',default='mps');p.add_argument('--full',action='store_true');p.add_argument('--all-fits',action='store_true');p.add_argument('--games',nargs='+');a=p.parse_args();OUT.mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=a.jobs,mp_context=get_context('spawn')) as pool:
        folds=['full']+list(range(5)) if a.all_fits else ['full'] if a.full else list(range(5))
        jobs=[pool.submit(prepare,g) for g in (a.games or plan()['pff_train_ids'])] if a.mode=='prepare' else [pool.submit(fit,c['name'],f,a.device) for f in folds for c in plan()['candidates']]
        for f in as_completed(jobs):result=f.result();print(result or 'fit complete',flush=True)
