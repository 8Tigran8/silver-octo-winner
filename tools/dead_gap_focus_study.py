"""Registered long missing-live D specialist; unchanged train/dev split."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
import argparse,json,sys,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
import team_gap_specialist as study
from dead_gap_objective import hard_centres,dead_loss
from activity_temporal_model import ActivityModel,probabilities as activity_probabilities
from report_team_gap_specialist import summarize
from pff_reader import sha
from mtah import mixed_temporal,temporal,v4
from mtah.methods import boost
from short_gap_study import diagnostics
from decoder_development import metric
DOC=ROOT/'docs/experiments/dead-gap-focus';OUT=ROOT/'reports/dead_gap_focus'

def cfg():return json.loads((DOC/'plan.json').read_text())
def check():
    study.install_read_guard();study.check()
    for p,h in json.loads((DOC/'freeze.json').read_text())['files_sha256'].items():assert sha(ROOT/p)==h,p

def fit(kind):
    config=study.plan();dest=OUT/kind;dest.mkdir(parents=True,exist_ok=False);study.initialize();model,mean,scale=study.STATE[1];model.to('mps');model.train()
    torch.manual_seed(config['seed']);torch.use_deterministic_algorithms(True);rng=np.random.default_rng(config['seed']);focus_rng=np.random.default_rng(config['seed']+1)
    opt=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=.001,foreach=False)
    prepared=json.loads((study.SEQ/'prepared.json').read_text());assert set(prepared)==set(config['training_games']);seq={};hashes={}
    for g in config['training_games']:
        study.guard(g,'training');seq[g]=[]
        for relative in prepared[g]:
            folder=ROOT/relative;m=json.loads((folder/'meta.json').read_text());hashes[relative]=sha(folder/'meta.json');assert all(sha(folder/(k+'.npy'))==h for k,h in m['feature_sha256'].items())
            row={k:np.load(folder/(k+'.npy'),mmap_mode='r') for k in ['x','xs','gt']};row.update(meta=m,centres=hard_centres(row['gt'],row['x'][:,m['seen_index']]>.5,m['segments']));seq[g].append(row)
    groups=[[g for g in seq if '/idsse_' in g],[g for g in seq if '/metrica_' in g],[g for g in seq if g.startswith('skillcorner/')],[g for g in seq if g.startswith('pff/')]]
    radius=256;corelen=256;started=time.monotonic();losses=[];focused_windows=0
    for step in range(config['steps']):
        bx=[];by=[];bw=[];bm=[]
        for _ in range(16):
            group=groups[int(rng.integers(4))];game=group[int(rng.integers(len(group)))];rr=seq[game];row=rr[int(rng.integers(len(rr)))];parts=row['meta']['segments'];pi=int(rng.integers(len(parts)));a,b=parts[pi]
            first=int(rng.integers(a,max(a+1,b-corelen+1)));swap=bool(rng.integers(2))
            if kind=='focus' and focus_rng.random()<.5 and len(row['centres'][pi]):
                centre=int(focus_rng.choice(row['centres'][pi]));first=int(np.clip(centre-focus_rng.integers(corelen),a,max(a,b-corelen)));focused_windows+=1
            ix=np.clip(np.arange(first-radius,first+corelen+radius),a,b-1);x=row['xs' if swap else 'x'][ix];y=np.array(row['gt'][ix],copy=True)
            if swap:y=np.where(y<2,1-y,y)
            y[:radius]=3;y[radius+corelen:]=3
            bx.append(temporal.normalize(x,mean,scale).T);by.append(y);bw.append(np.where((y==2)&game.startswith('skillcorner/'),.25,1).astype(np.float32));bm.append(x[:,row['meta']['seen_index']]<.5)
        batch=torch.from_numpy(np.stack(bx)).to('mps');target=torch.from_numpy(np.stack(by).astype(np.int64)).to('mps');weight=torch.from_numpy(np.stack(bw)).to('mps');missing=torch.from_numpy(np.stack(bm)).to('mps')
        opt.zero_grad(set_to_none=True);loss=dead_loss(model(batch),target,missing,kind=='focus',weight);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();losses.append(float(loss.detach().cpu()))
        if (step+1)%1000==0:print(kind,step+1,'loss',float(np.mean(losses[-100:])),round(time.monotonic()-started,1),'seconds',flush=True)
    torch.save(dict(state_dict=model.cpu().state_dict(),mean=torch.from_numpy(mean),scale=torch.from_numpy(scale)),dest/'model.pt')
    meta=dict(candidate=kind,training_games=config['training_games'],steps=config['steps'],seed=config['seed'],focused_windows=focused_windows,training_sequences_sha256=hashes,model_sha256=sha(dest/'model.pt'),parent_sha256=sha(ROOT/'assets/recovery-candidate/neural_state.pt'),normalizer_unchanged=True,seconds=time.monotonic()-started,plan_sha256=sha(DOC/'plan.json'))
    (dest/'meta.json').write_text(json.dumps(meta,indent=2)+'\n');(DOC/(kind+'_meta.json')).write_text(json.dumps(meta,indent=2)+'\n');print('Fit complete',kind,flush=True)

def freeze():
    from datetime import datetime,timezone
    import subprocess
    study.install_read_guard();study.check();assert not (DOC/'freeze.json').exists() and not OUT.exists()
    files=[ROOT/'tools/dead_gap_focus_study.py',ROOT/'tools/dead_gap_objective.py',ROOT/'tests/test_dead_gap_objective.py',ROOT/'tools/activity_temporal_model.py',ROOT/'tools/report_team_gap_specialist.py',DOC/'plan.json',study.OUT/'focus/model.pt',study.OUT/'focus/meta.json',ROOT/'reports/activity_temporal/temporal.pt',ROOT/'docs/experiments/activity-temporal/temporal_meta.json']
    for g in study.plan()['development_games']:
        for scenario in study.SCENARIOS:files.extend([ROOT/f'reports/short_gap_recovery/development/{g}_{scenario}.npz',study.OUT/'development'/f'{g}_{scenario}.npz'])
    (DOC/'freeze.json').write_text(json.dumps(dict(frozen_utc=datetime.now(timezone.utc).isoformat(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),files_sha256={str(p.relative_to(ROOT)):sha(p) for p in files}),indent=2)+'\n')

def initialize_score():
    global MODELS,ACTIVITY
    study.initialize_score(True);MODELS={}
    for kind in ['uniform','focus']:
        dest=OUT/kind;st=torch.load(dest/'model.pt',map_location='cpu',weights_only=True);assert sha(dest/'model.pt')==json.loads((dest/'meta.json').read_text())['model_sha256']
        assert np.array_equal(st['mean'].numpy(),study.STATE[1][1]) and np.array_equal(st['scale'].numpy(),study.STATE[1][2]);m=mixed_temporal.MixedTemporal(study.BUNDLE['spec']);m.load_state_dict(st['state_dict']);m.eval();MODELS[kind]=m
    p=ROOT/'reports/activity_temporal/temporal.pt';assert sha(p)==json.loads((ROOT/'docs/experiments/activity-temporal/temporal_meta.json').read_text())['model_sha256'];st=torch.load(p,map_location='cpu',weights_only=True);m=ActivityModel('temporal');m.load_state_dict(st['state_dict']);m.eval();ACTIVITY=(m,st['mean'].numpy(),st['scale'].numpy())

def worker(game):
    study.guard(game,'development');rows=[];start=time.monotonic();ref=ROOT/f'reports/pff_access/references/kloppy_{game}.npz';receipt=json.loads((ROOT/f'reports/pff_access/kloppy_{game}_preflight.json').read_text());assert sha(ref)==receipt['reference_cache_sha256']
    for scenario in study.SCENARIOS:
        with np.load(study.OUT/'development'/f'{game}_{scenario}.npz') as z:c=dict(z)
        parts=c['segments'].tolist();small=c['small_segments'].tolist();missing=~c['seen']
        with np.load(ROOT/f'reports/short_gap_recovery/development/{game}_{scenario}.npz') as z:
            assert np.array_equal(z['seen'],c['seen']) and np.array_equal(z['segments'],c['segments']);x=z['activity'];assert z['columns'].tolist()==study.STATE[3]
        from threadpoolctl import threadpool_limits
        with threadpool_limits(1):old=study.STATE[2].predict_proba(x)[:,1]
        m,mean,scale=ACTIVITY;act=activity_probabilities(m,x,small,mean,scale);old,act=v4.neural_upsample(old,act,parts,len(c['pd']))
        new={}
        for kind,m in MODELS.items():
            d,_=mixed_temporal.probabilities(m,c['x'],c['xs'],small,study.STATE[1][1],study.STATE[1][2]);new[kind]=v4.neural_upsample(d,d,parts,len(c['pd']))[0]
        _,h=mixed_temporal.probabilities(study.MODELS['focus'],c['x'],c['xs'],small,study.STATE[1][1],study.STATE[1][2]);h=v4.neural_upsample(h,h,parts,len(c['pd']))[0];predictions={}
        for name in cfg()['candidates']:
            pd=c['pd'].copy();ph=c['ph'].copy()
            if name=='prior_temporal':pd[missing]=c['pd'][missing]-.25*old[missing]+.25*act[missing]
            elif name not in ['control','ownership_only']:
                kind=name.split('_')[0];a=act if 'temporal' in name else old;pd[missing]=.75*new[kind][missing]+.25*a[missing]
            if name in ['ownership_only','prior_temporal'] or name.endswith('_focus'):ph[missing]=h[missing]
            predictions[name]=boost.label(pd,ph,parts,len(pd),study.BUNDLE['decoder'])
        with np.load(ROOT/f'reports/activity_temporal/{game}_{scenario}_predictions.npz') as z:
            assert np.array_equal(predictions['control'],z['control']);assert np.array_equal(predictions['ownership_only'],z['ownership_only']);assert np.array_equal(predictions['prior_temporal'],z['temporal_25_focus'])
        np.savez_compressed(OUT/f'{game}_{scenario}_predictions.npz',**predictions)
        with np.load(ref) as z:
            for reference,key in [('canonical','gt'),('sdk_shadow','alternate')]:
                gt=z[key]
                for name,pred in predictions.items():
                    d=diagnostics(gt,pred,c['seen']);mask=(gt<2)&missing;d['missing_wrong_team']=int((mask&(pred<2)&(pred!=gt)).sum());rows.append(dict(game=game,scenario=scenario,reference=reference,candidate=name,**metric(gt,pred),diagnostics=d))
    return game,rows,time.monotonic()-start

def score():
    dest=DOC/'scores.json';assert not dest.exists();rows=[]
    with ProcessPoolExecutor(max_workers=2,mp_context=get_context('spawn'),initializer=initialize_score) as pool:
        for f in as_completed([pool.submit(worker,g) for g in study.plan()['development_games']]):
            g,r,t=f.result();rows+=r;print('Scored',g,round(t,1),flush=True)
    dest.write_text(json.dumps(dict(records=rows),indent=2)+'\n');result=summarize(rows,cfg()['candidates'],study.plan());ix={(r['game'],r['scenario'],r['reference'],r['candidate']):r for r in rows}
    for r in result['summary']:
        bases=[ix[g,r['scenario'],r['reference'],'control'] for g in study.plan()['development_games']];items=[ix[g,r['scenario'],r['reference'],r['candidate']] for g in study.plan()['development_games']]
        r['dead_f1_gain']=float(np.mean([x['per_class']['D']['f1']-b['per_class']['D']['f1'] for x,b in zip(items,bases)]));bc={k:sum(x['diagnostics'][k] for x in bases) for k in bases[0]['diagnostics']};ct=r['counts'];r['missing_false_live_gain']=ct['missing_false_live']/ct['missing_dead']-bc['missing_false_live']/bc['missing_dead']
        r['gates'].update(dead_f1=r['dead_f1_gain']>=-.002,missing_false_live=r['missing_false_live_gain']<=.01);r['pass_all']=all(r['gates'].values())
    eligible=[n for n in cfg()['candidates'] if n not in ['control','ownership_only','prior_temporal'] and all(r['pass_all'] for r in result['summary'] if r['candidate']==n)];result['eligible']=eligible;result['selected']=max(eligible,key=lambda c:min(r['gain'] for r in result['summary'] if r['candidate']==c)) if eligible else None;result['decision']='Eligible for technical validation and extra-seed regression only' if eligible else 'No candidate qualifies; production unchanged'
    (DOC/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    for r in result['summary']:
        if r['reference']=='canonical':print(r['candidate'],r['scenario'],round(r['mean_spell'],6),round(r['gain'],6),r['wins'],r['gates'],flush=True)
    print('SELECTED',result['selected'],flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','fit','score']);p.add_argument('--kind',choices=['uniform','focus']);a=p.parse_args()
    if a.action=='freeze':freeze()
    else:
        check()
        if a.action=='fit':fit(a.kind)
        else:score()
        check()
