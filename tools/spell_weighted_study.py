"""Matched warm-start training with corrected features and training-only weights."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[key]='1'
import argparse,json,time,sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context
import numpy as np
import torch
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
from mtah import mixed_candidate,mixed_temporal,temporal,expanded_team,degrade,v4
from mtah.methods import boost
from pff_development import geometry,spans
from pff_reader import sha
from temporal_trial import observed_periods
from legacy_skillcorner_reader import period_spans
from pff_mixed_score import decoder
from decoder_development import metric
from short_gap_study import diagnostics
from spell_weighted_loss import live_spell_weights,loss
DOC=ROOT/'docs/experiments/spell-weighted-finetune';OUT=ROOT/'reports/spell_weighted_finetune'


def plan():return json.loads((DOC/'plan.json').read_text())


def check():
    for file,digest in json.loads((DOC/'freeze.json').read_text())['files_sha256'].items():
        if sha(ROOT/file)!=digest:raise ValueError('Frozen input/source changed: '+file)


def training_data(game):
    if game not in plan()['training_games']:raise ValueError('Not an allowed training fixture')
    family,name=game.split('/')
    if family=='pff':
        data,receipt=geometry(name);parts=spans(data['period'])
        ref=ROOT/'reports/pff_access/references'/f'kloppy_{name}.npz'
        if sha(ref)!=receipt['reference_cache_sha256']:raise ValueError('Changed reference')
        with np.load(ref) as z:gt=z['gt']
        data=dict(data,gt=gt);sources={str(ref.relative_to(ROOT)):sha(ref),str(Path('external/pff/converted')/(name+'_geometry.npz')):receipt['geometry_cache_sha256']}
    else:
        path=ROOT/('external/conv' if family=='original' else 'reports/v2_search/skillcorner')/(name+'.npz')
        with np.load(path) as z:data=dict(z)
        parts=observed_periods(data) if family=='original' else period_spans(data['period'])
        sources={str(path.relative_to(ROOT)):sha(path)}
    specs=expanded_team.ORIGINAL_SCENARIOS if family=='original' else expanded_team.SKILL_SCENARIOS
    result=[]
    for scenario,(seed,params,_) in specs.items():
        observed=data if seed is None else degrade.degrade(data,seed=seed,params=params)
        gt=observed['gt'];observed={k:v for k,v in observed.items() if k!='gt'}
        result.append((scenario,observed,gt))
    return parts,result,sources


def prepare_game(game):
    started=time.monotonic();parts,rows,sources=training_data(game);records=[]
    for scenario,data,gt in rows:
        dest=OUT/'sequences'/game.replace('/','__')/scenario;dest.mkdir(parents=True,exist_ok=False)
        with threadpool_limits(1):x,xs,cols=expanded_team.feature_pair(data,parts)
        if len(cols)!=108:raise ValueError('Feature schema changed')
        for key,value in [('x',x[::5]),('xs',xs[::5]),('gt',gt[::5])]:np.save(dest/(key+'.npy'),value)
        meta=dict(game=game,scenario=scenario,segments=[((a+4)//5,(b+4)//5) for a,b in parts],seen_index=cols.index('seen'),
                  source_sha256=sources,feature_sha256={k:sha(dest/(k+'.npy')) for k in ['x','xs','gt']})
        (dest/'meta.json').write_text(json.dumps(meta,indent=2)+'\n');records.append(str(dest.relative_to(ROOT)))
    return game,records,time.monotonic()-started


def prepare():
    OUT.mkdir(exist_ok=False);receipts={}
    with ProcessPoolExecutor(max_workers=2,mp_context=get_context('spawn')) as pool:
        for future in as_completed([pool.submit(prepare_game,g) for g in plan()['training_games']]):
            game,rows,seconds=future.result();receipts[game]=rows;print('Prepared',game,round(seconds,1),flush=True)
    (OUT/'prepared.json').write_text(json.dumps(receipts,indent=2)+'\n')


def fit(name):
    cfg=plan();dest=OUT/name;dest.mkdir(exist_ok=False)
    bundle,state=mixed_candidate.load_bundle(ROOT/'assets/pff-mixed-candidate/bundle.json')
    torch.set_num_threads(2);torch.manual_seed(cfg['seed']);torch.use_deterministic_algorithms(True);rng=np.random.default_rng(cfg['seed'])
    # Same checkpoint, normalizer, architecture, sampled windows and dropout seed.
    model,mean,scale=state;model.to('mps');model.train()
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=.001,foreach=False)
    prepared=json.loads((OUT/'prepared.json').read_text());seq={};hashes={}
    if set(prepared)!=set(cfg['training_games']):raise ValueError('Incomplete training registry')
    for game in cfg['training_games']:
        seq[game]=[]
        for relative in prepared[game]:
            folder=ROOT/relative;meta=json.loads((folder/'meta.json').read_text());hashes[relative]=sha(folder/'meta.json')
            if any(sha(folder/(k+'.npy'))!=h for k,h in meta['feature_sha256'].items()):raise ValueError('Changed training sequence')
            row={k:np.load(folder/(k+'.npy'),mmap_mode='r') for k in ['x','xs','gt']}
            row.update(meta=meta,spell_weight=live_spell_weights(row['gt'],meta['segments']));seq[game].append(row)
    groups=[[g for g in seq if '/idsse_' in g],[g for g in seq if '/metrica_' in g],
            [g for g in seq if g.startswith('skillcorner/')],[g for g in seq if g.startswith('pff/')]]
    radius=256;corelen=256;started=time.monotonic();losses=[]
    for step in range(cfg['steps']):
        bx=[];by=[];bd=[];bh=[];bm=[]
        for _ in range(16):
            group=groups[int(rng.integers(4))];game=group[int(rng.integers(len(group)))];rows=seq[game];row=rows[int(rng.integers(len(rows)))]
            parts=row['meta']['segments'];a,b=parts[int(rng.integers(len(parts)))];core=int(rng.integers(a,max(a+1,b-corelen+1)))
            ix=np.clip(np.arange(core-radius,core+corelen+radius),a,b-1);swap=bool(rng.integers(2))
            x=row['xs' if swap else 'x'][ix];y=np.array(row['gt'][ix],copy=True)
            if swap:y=np.where(y<2,1-y,y)
            wd=np.ones(len(ix),np.float32)
            if game.startswith('skillcorner/'):wd[y==2]=.25
            y[:radius]=3;y[radius+corelen:]=3
            bx.append(temporal.normalize(x,mean,scale).T);by.append(y);bd.append(wd)
            bh.append(row['spell_weight'][ix]);bm.append(x[:,row['meta']['seen_index']]<.5)
        x=torch.from_numpy(np.stack(bx)).to('mps');y=torch.from_numpy(np.stack(by).astype(np.int64)).to('mps')
        wd=torch.from_numpy(np.stack(bd)).to('mps');wh=torch.from_numpy(np.stack(bh)).to('mps');missing=torch.from_numpy(np.stack(bm)).to('mps')
        opt.zero_grad(set_to_none=True);logits=model(x)
        objective=loss(logits,y,wd,wh,missing,name=='weighted')+.15*mixed_temporal.boundary_loss(logits,y,wd,8)
        objective.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();losses.append(float(objective.detach().cpu()))
        if (step+1)%500==0:print(name,step+1,'loss',float(np.mean(losses[-100:])),'seconds',round(time.monotonic()-started,1),flush=True)
    torch.save(dict(state_dict=model.cpu().state_dict(),mean=torch.from_numpy(mean),scale=torch.from_numpy(scale)),dest/'model.pt')
    meta=dict(candidate=name,training_games=cfg['training_games'],training_sequences_sha256=hashes,plan_sha256=sha(DOC/'plan.json'),
        model_sha256=sha(dest/'model.pt'),parent_bundle_sha256=sha(ROOT/'assets/pff-mixed-candidate/bundle.json'),
        frozen_train_only_normalizer=True,steps=cfg['steps'],seconds=time.monotonic()-started,loss_last100=float(np.mean(losses[-100:])))
    (dest/'meta.json').write_text(json.dumps(meta,indent=2)+'\n');print(name,'fit complete',flush=True)


def initialize_score():
    global ASSETS,STATES
    ASSETS=v4.load();STATES={}
    bundle,_=mixed_candidate.load_bundle(ROOT/'assets/pff-mixed-candidate/bundle.json')
    for name in ['uniform','weighted']:
        meta=json.loads((OUT/name/'meta.json').read_text())
        if meta['model_sha256']!=sha(OUT/name/'model.pt'):raise ValueError('Model changed')
        state=torch.load(OUT/name/'model.pt',map_location='cpu',weights_only=True)
        model=mixed_temporal.MixedTemporal(bundle['spec']);model.load_state_dict(state['state_dict']);model.eval()
        STATES[name]=(model,state['mean'].numpy(),state['scale'].numpy())


def score_game(game):
    if game not in plan()['development_games']:raise ValueError('Not development')
    data,receipt=geometry(game);parts=spans(data['period']);start=time.monotonic();rows=[]
    dummy=np.where(data['period']>0,0,3).astype(np.int8)
    noisy=degrade.degrade(dict(data,gt=dummy),seed=7,params=expanded_team.BLIND);noisy.pop('gt')
    ref=ROOT/'reports/pff_access/references'/f'kloppy_{game}.npz'
    if sha(ref)!=receipt['reference_cache_sha256']:raise ValueError('Reference changed')
    baseline=json.loads((ROOT/'validation/observation_regression.json').read_text())['records']
    baseline={(r['game'],r['scenario'],r['reference']):r for r in baseline}
    for scenario,observed in [('natural_detected',data),('label_blind_seed7',noisy)]:
        bd,bh,_,_=v4.components(observed,parts,ASSETS)
        with threadpool_limits(1):x,xs,_=expanded_team.feature_pair(observed,parts)
        down=[((a+4)//5,(b+4)//5) for a,b in parts];predictions={}
        for name,(model,mean,scale) in STATES.items():
            nd,nh=mixed_temporal.probabilities(model,x[::5],xs[::5],down,mean,scale)
            nd,nh=v4.neural_upsample(nd,nh,parts,len(x))
            pd,ph=.5*bd+.5*nd,.5*bh+.5*nh
            for short in [False,True]:
                params=decoder()
                if short:params.update(min_team_s=.4,cost_ha=4.5)
                key=name+('_short' if short else '')
                predictions[key]=boost.label(pd,ph,parts,len(pd),params)
        np.savez_compressed(OUT/f'{game}_{scenario}_predictions.npz',**predictions)
        with np.load(ref) as archive:
            for reference,key in [('canonical','gt'),('sdk_shadow','alternate')]:
                gt=archive[key];base=baseline[game,scenario,reference]
                for name,pred in predictions.items():
                    score=metric(gt,pred)
                    rows.append(dict(game=game,scenario=scenario,reference=reference,candidate=name,**score,
                        spell_gain=score['spell_f1_mean']-base['spell_f1_mean'],macro_gain=score['macro_f1']-base['macro_f1'],
                        diagnostics=diagnostics(gt,pred,np.isfinite(observed['ball'][:,:2]).all(axis=1))))
    (OUT/f'{game}_scores.json').write_text(json.dumps(rows,indent=2)+'\n')
    return game,rows,time.monotonic()-start


def score():
    if (OUT/'results.json').exists():raise ValueError('Completed results must not be overwritten')
    rows=[]
    with ProcessPoolExecutor(max_workers=2,mp_context=get_context('spawn'),initializer=initialize_score) as pool:
        for future in as_completed([pool.submit(score_game,g) for g in plan()['development_games']]):
            game,rr,seconds=future.result();rows.extend(rr);print('Scored',game,round(seconds,1),flush=True)
    (OUT/'results.json').write_text(json.dumps(dict(plan_sha256=sha(DOC/'plan.json'),records=rows),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','fit','score']);p.add_argument('--candidate',choices=['uniform','weighted']);a=p.parse_args()
    check()
    if a.action=='prepare':prepare()
    elif a.action=='fit':fit(a.candidate)
    else:score()
    check()
