"""Whole-game TCN research; no competition input or learned base features."""
import argparse,json,os,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context
import sys
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from mtah import temporal,expanded_team as expanded,possession_context as context,metrics
from quality_audit import sha,read_feature
from skillcorner_reader import read_match
from legacy_skillcorner_reader import period_spans
from mtah import degrade
PLAN=ROOT/'docs/experiments/temporal/plan.json';OUT=Path(os.environ.get('MTAH_TEMPORAL_CACHE',str(ROOT/'reports/temporal')));AUDIT=ROOT/'reports/quality_audit/380987b3029c'


def plan():return json.loads(PLAN.read_text())
def stem(game):return game.replace('/','__')
def scenarios(game):return ['matched_seed0','label_blind_seed1','clean_source'] if game.startswith('original/') else ['natural_detected','label_blind_seed7']
def folder(game,scenario):return OUT/'sequences'/stem(game)/scenario


def observed_periods(data):
    # Original converters insert an all-missing10s separator. Infer it from
    # positions only; no H/A/D/reference field is needed to recover halves.
    observed=np.isfinite(data['players'][:,:,:2]).any(axis=(1,2))|np.isfinite(data['ball'][:,:2]).any(axis=1)
    missing=~observed;e=np.diff(np.r_[0,missing.astype(np.int8),0]);gaps=[(a,b) for a,b in zip(np.flatnonzero(e==1),np.flatnonzero(e==-1)) if b-a>=250 and a>0 and b<len(observed)]
    if len(gaps)!=1:raise ValueError('Cannot identify independent original half separator')
    a,b=gaps[0];return [(0,int(a)),(int(b),len(observed))]


def prepare(game):
    if game not in plan()['training_games']:raise ValueError('Not whitelisted')
    family,name=game.split('/')
    if family=='original':
        with np.load(ROOT/'external/conv'/(name+'.npz')) as z:clean={k:z[k] for k in z.files}
        segments=observed_periods(clean)
    else:clean=read_match(ROOT/'external/skillcorner'/name,name);segments=period_spans(clean['period'])
    for scenario in scenarios(game):
        dest=folder(game,scenario);dest.mkdir(parents=True,exist_ok=True)
        if (dest/'meta.json').exists():continue
        if family=='original':
            seed,params,_=expanded.ORIGINAL_SCENARIOS[scenario]
            data=clean if seed is None else degrade.degrade(clean,seed=seed,params=params)
            x,xs,cols=expanded.feature_pair(data,segments);gt=data['gt']
        else:
            data=clean if scenario=='natural_detected' else degrade.degrade(clean,seed=7,params=expanded.BLIND)
            x,xs,cols=expanded.feature_pair(data,segments);gt=data['gt']
        np.save(dest/'x.npy',x[::5].astype(np.float32));np.save(dest/'xs.npy',xs[::5].astype(np.float32));np.save(dest/'gt.npy',gt[::5])
        spans=[((a+4)//5,(b+4)//5) for a,b in segments]
        (dest/'meta.json').write_text(json.dumps({'game':game,'scenario':scenario,'segments':spans,'full_segments':segments,'full_length':len(gt),'features':len(cols),'plan_sha256':sha(PLAN)},indent=2)+'\n')
    return game


def game_split(key):
    p=plan();names=p['training_games']
    if key.startswith('fold'):
        k=int(key[4:]);test=[g for g in names if p['folds'][g]==k];train=[g for g in names if g not in test]
    elif key=='provider_dfl':train=[g for g in names if '/idsse_' in g];test=[g for g in names if g not in train]
    elif key=='provider_skill':train=[g for g in names if g.startswith('skillcorner/')];test=[g for g in names if g not in train]
    elif key=='full':train=names;test=[]
    else:raise ValueError('Unregistered split')
    if set(train)&set(test) or set(train)|set(test)!=set(names):raise ValueError('Split leak')
    return train,test


def sequences(train):
    result={}
    for g in train:
        result[g]=[]
        for s in scenarios(g):
            f=folder(g,s);m=json.loads((f/'meta.json').read_text())
            if m['plan_sha256']!=sha(PLAN):raise ValueError('Stale cache')
            result[g].append({'x':np.load(f/'x.npy',mmap_mode='r'),'xs':np.load(f/'xs.npy',mmap_mode='r'),'gt':np.load(f/'gt.npy',mmap_mode='r'),'meta':m})
    return result


def normalization(seq):
    total=np.zeros(108);squares=np.zeros(108);count=np.zeros(108)
    for rows in seq.values():
        for row in rows:
            for field in ['x','xs']:
                x=row[field][::5];valid=np.isfinite(x);value=np.where(valid,x,0).astype(np.float64)
                total+=value.sum(0);squares+=(value*value).sum(0);count+=valid.sum(0)
    mean=np.divide(total,count,out=np.zeros(108),where=count>0);var=np.divide(squares,count,out=np.ones(108),where=count>0)-mean**2
    return mean.astype(np.float32),np.sqrt(np.maximum(var,1e-4)).astype(np.float32)


def fit(key,device):
    dest=OUT/'models'/key;dest.mkdir(parents=True,exist_ok=True)
    if (dest/'meta.json').exists():return
    train,test=game_split(key);seq=sequences(train);mean,scale=normalization(seq)
    cfg=plan()['architecture'];seed=cfg['seed'];torch.manual_seed(seed);rng=np.random.default_rng(seed);torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    model=temporal.Temporal().to(device);opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'],foreach=False)
    groups=[[g for g in train if g.startswith('original/')],[g for g in train if g.startswith('skillcorner/')]];groups=[g for g in groups if g]
    started=time.perf_counter();losses=[]
    for step in range(cfg['training_steps']):
        bx=[];by=[];bw=[]
        for _ in range(cfg['batch']):
            group=groups[int(rng.integers(len(groups)))];game=group[int(rng.integers(len(group)))];rows=seq[game];r=rows[int(rng.integers(len(rows)))];spans=r['meta']['segments']
            a,b=spans[int(rng.integers(len(spans)))];core=int(rng.integers(a,max(a+1,b-cfg['core_steps']+1)));ix=np.clip(np.arange(core-64,core+320),a,b-1)
            swap=bool(rng.integers(2));x=r['xs' if swap else 'x'][ix];y=np.array(r['gt'][ix],copy=True)
            if swap:y=np.where(y<2,1-y,y)
            w=np.ones(len(ix),np.float32)
            if game.startswith('skillcorner/'):w[y==2]=.25
            y[:64]=3;y[320:]=3
            bx.append(temporal.normalize(x,mean,scale).T);by.append(y);bw.append(w)
        x=torch.from_numpy(np.stack(bx)).to(device);y=torch.from_numpy(np.stack(by).astype(np.int64)).to(device);w=torch.from_numpy(np.stack(bw)).to(device)
        opt.zero_grad(set_to_none=True);loss=temporal.loss(model(x),y,w);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();losses.append(float(loss.detach().cpu()))
        if (step+1)%100==0:print(key,step+1,'loss',round(float(np.mean(losses[-100:])),5),'seconds',round(time.perf_counter()-started,1),flush=True)
    model=model.cpu();torch.save({'state_dict':model.state_dict(),'mean':torch.from_numpy(mean),'scale':torch.from_numpy(scale)},dest/'model.pt')
    (dest/'meta.json').write_text(json.dumps({'train':train,'excluded':test,'plan_sha256':sha(PLAN),'device':device,'seconds':time.perf_counter()-started,'model_sha256':sha(dest/'model.pt'),'training_loss_last100':float(np.mean(losses[-100:]))},indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['prepare','fit']);parser.add_argument('--jobs',type=int,default=4);parser.add_argument('--split',default='all');parser.add_argument('--device',default='mps');args=parser.parse_args();OUT.mkdir(exist_ok=True)
    if args.mode=='prepare':
        with ProcessPoolExecutor(max_workers=args.jobs,mp_context=get_context('spawn')) as pool:
            for f in as_completed([pool.submit(prepare,g) for g in plan()['training_games']]):print(f.result(),'prepared',flush=True)
    else:
        for key in ([f'fold{i}' for i in range(5)]+['provider_dfl','provider_skill','full'] if args.split=='all' else [args.split]):fit(key,args.device)
