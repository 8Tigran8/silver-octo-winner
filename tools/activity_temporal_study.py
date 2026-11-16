"""Frozen36train/10development temporal activity study; excludes18audit fixtures."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
import argparse,json,sys,time,subprocess,hashlib
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context
import numpy as np
import torch
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
import team_gap_specialist as ownership
from activity_temporal_model import ActivityModel,probabilities,loss
from mtah import temporal,v4,mixed_temporal
from mtah.methods import boost
from mtah.recovery_features import activity_features
from pff_development import geometry,spans
from pff_reader import sha
from short_gap_study import scenarios,diagnostics
from decoder_development import metric
from report_team_gap_specialist import summarize
DOC=ROOT/'docs/experiments/activity-temporal';OUT=ROOT/'reports/activity_temporal';OLD=ROOT/'reports/short_gap_recovery'

def plan():return json.loads((DOC/'plan.json').read_text())
def check():
    ownership.install_read_guard();ownership.check()
    for p,h in json.loads((DOC/'freeze.json').read_text())['files_sha256'].items():assert sha(ROOT/p)==h,p

def freeze():
    ownership.install_read_guard();ownership.check();assert not (DOC/'freeze.json').exists() and not OUT.exists()
    files=[ROOT/'tools/activity_temporal_model.py',ROOT/'tools/activity_temporal_study.py',ROOT/'tests/test_activity_temporal_model.py',DOC/'plan.json',ROOT/'tools/report_team_gap_specialist.py',ownership.OUT/'focus/model.pt',ownership.OUT/'focus/meta.json']
    for split in ['training','development']:
        for g in plan()[split+'_games']:
            ownership.guard('pff/'+g if split=='training' else g,split)
            files.extend(ROOT/p for p in [f'external/pff/converted/{g}_geometry.npz',f'reports/pff_access/kloppy_{g}_preflight.json',f'reports/pff_access/references/kloppy_{g}.npz'])
            if split=='development':
                for s in ownership.SCENARIOS:files.extend([OLD/'development'/f'{g}_{s}.npz',ownership.OUT/'development'/f'{g}_{s}.npz'])
    (DOC/'freeze.json').write_text(json.dumps(dict(frozen_utc=datetime.now(timezone.utc).isoformat(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),files_sha256={str(p.relative_to(ROOT)):sha(p) for p in files}),indent=2)+'\n')

def initialize_prepare():ownership.install_read_guard();torch.set_num_threads(1)

def prepare_game(game):
    ownership.guard('pff/'+game,'training');started=time.monotonic();data,receipt=geometry(game);parts=spans(data['period']);small=[((a+4)//5,(b+4)//5) for a,b in parts];rows=[]
    ref=ROOT/f'reports/pff_access/references/kloppy_{game}.npz';assert sha(ref)==receipt['reference_cache_sha256']
    for scenario,d in scenarios(data):
        with threadpool_limits(1):x,names=activity_features(d,parts)
        missing=~np.isfinite(d['ball'][:,:2]).all(axis=1)
        # Labels opened only after geometry-only feature construction.
        with np.load(ref) as z:gt=z['gt']
        dest=OUT/'training'/f'{game}_{scenario}.npz';np.savez_compressed(dest,x=x[::5],gt=gt[::5],missing=missing[::5],segments=np.array(small),columns=np.array(names))
        rows.append(dict(game=game,scenario=scenario,path=str(dest.relative_to(ROOT)),sha256=sha(dest)))
    return game,rows,time.monotonic()-started

def prepare():
    (OUT/'training').mkdir(parents=True,exist_ok=False);rows=[]
    with ProcessPoolExecutor(max_workers=2,mp_context=get_context('spawn'),initializer=initialize_prepare) as pool:
        for f in as_completed([pool.submit(prepare_game,g) for g in plan()['training_games']]):
            g,r,t=f.result();rows+=r;print('Prepared',g,round(t,1),flush=True)
    (DOC/'prepared.json').write_text(json.dumps(rows,indent=2)+'\n')

def fit(kind):
    assert not (OUT/(kind+'.pt')).exists()
    torch.set_num_threads(1);torch.manual_seed(plan()['seed']);torch.use_deterministic_algorithms(True);rng=np.random.default_rng(plan()['seed']);seq={};sample=[];hashes={}
    for row in json.loads((DOC/'prepared.json').read_text()):
        ownership.guard('pff/'+row['game'],'training');path=ROOT/row['path'];assert sha(path)==row['sha256'];hashes[row['path']]=row['sha256']
        with np.load(path) as z:r=dict(z)
        seq.setdefault(row['game'],[]).append(r)
        for a,b in r['segments']:sample.append(r['x'][a:b:5])
    x=np.concatenate(sample);mean=np.nanmean(x,axis=0);scale=np.nanstd(x,axis=0);mean=np.nan_to_num(mean).astype(np.float32);scale=np.where(np.isfinite(scale)&(scale>.01),scale,1).astype(np.float32)
    # Provider/game/period sampling; no development data participates in normalization.
    model=ActivityModel(kind).to('mps');model.train();opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.001,foreach=False);radius=model.context;started=time.monotonic();losses=[];games=plan()['training_games']
    for step in range(plan()['steps']):
        bx=[];by=[];bm=[]
        for _ in range(16):
            game=games[int(rng.integers(len(games)))];rr=seq[game];r=rr[int(rng.integers(len(rr)))];parts=r['segments'];a,b=parts[int(rng.integers(len(parts)))];first=int(rng.integers(a,max(a+1,b-128+1)));ix=np.clip(np.arange(first-radius,first+128+radius),a,b-1)
            y=r['gt'][ix].copy();y[:radius]=3;y[radius+128:]=3;bx.append(temporal.normalize(r['x'][ix],mean,scale).T);by.append(y);bm.append(r['missing'][ix])
        tx=torch.from_numpy(np.stack(bx)).to('mps');ty=torch.from_numpy(np.stack(by).astype(np.int64)).to('mps');tm=torch.from_numpy(np.stack(bm)).to('mps')
        opt.zero_grad(set_to_none=True);l=loss(model(tx),ty,tm);l.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();losses.append(float(l.detach().cpu()))
        if (step+1)%1000==0:print(kind,step+1,float(np.mean(losses[-100:])),round(time.monotonic()-started,1),flush=True)
    dest=OUT/(kind+'.pt');assert not dest.exists();torch.save(dict(state_dict=model.cpu().state_dict(),mean=torch.from_numpy(mean),scale=torch.from_numpy(scale)),dest)
    meta=dict(kind=kind,training_games=games,training_cache_sha256=hashes,steps=plan()['steps'],seed=plan()['seed'],model_sha256=sha(dest),normalizer_sha256=hashlib.sha256(mean.tobytes()+scale.tobytes()).hexdigest(),normalization_rows=len(x),normalization_scope='Training periods only,1Hz,including all activity states',parameter_count=sum(p.numel() for p in model.parameters()),seconds=time.monotonic()-started)
    (DOC/(kind+'_meta.json')).write_text(json.dumps(meta,indent=2)+'\n');print('Fit complete',kind,flush=True)

def initialize_score():
    global MODELS
    ownership.initialize_score(True);MODELS={}
    for kind in ['pointwise','temporal']:
        p=OUT/(kind+'.pt');assert sha(p)==json.loads((DOC/(kind+'_meta.json')).read_text())['model_sha256'];st=torch.load(p,map_location='cpu',weights_only=True);m=ActivityModel(kind);m.load_state_dict(st['state_dict']);m.eval();MODELS[kind]=(m,st['mean'].numpy(),st['scale'].numpy())
    assert np.array_equal(MODELS['pointwise'][1],MODELS['temporal'][1]) and np.array_equal(MODELS['pointwise'][2],MODELS['temporal'][2])

def worker(game):
    ownership.guard(game,'development');rows=[];started=time.monotonic();ref=ROOT/f'reports/pff_access/references/kloppy_{game}.npz';receipt=json.loads((ROOT/f'reports/pff_access/kloppy_{game}_preflight.json').read_text());assert sha(ref)==receipt['reference_cache_sha256']
    for scenario in ownership.SCENARIOS:
        with np.load(ownership.OUT/'development'/f'{game}_{scenario}.npz') as z:c=dict(z)
        parts=c['segments'].tolist();small=c['small_segments'].tolist()
        with np.load(OLD/'development'/f'{game}_{scenario}.npz') as z:
            assert np.array_equal(z['seen'],c['seen']) and np.array_equal(z['segments'],c['segments']);x=z['activity'];assert z['columns'].tolist()==ownership.STATE[3]
        with threadpool_limits(1):old=ownership.STATE[2].predict_proba(x)[:,1]
        old=v4.neural_upsample(old,old,parts,len(c['pd']))[0];experts={}
        for kind,(model,mean,scale) in MODELS.items():
            e=probabilities(model,x,small,mean,scale);experts[kind]=v4.neural_upsample(e,e,parts,len(c['pd']))[0]
        _,h=mixed_temporal.probabilities(ownership.MODELS['focus'],c['x'],c['xs'],small,ownership.STATE[1][1],ownership.STATE[1][2]);h=v4.neural_upsample(h,h,parts,len(c['pd']))[0];missing=~c['seen'];nd=(c['pd'][missing]-.25*old[missing])/.75;predictions={}
        for name in plan()['candidates']:
            pd=c['pd'].copy();ph=c['ph'].copy()
            if name not in ['control','ownership_only']:
                kind,w=name.split('_')[:2];w=int(w)/100;pd[missing]=(1-w)*nd+w*experts[kind][missing]
            if name=='ownership_only' or name.endswith('_focus'):ph[missing]=h[missing]
            predictions[name]=boost.label(pd,ph,parts,len(pd),ownership.BUNDLE['decoder'])
        with np.load(ownership.OUT/f'followup_{game}_{scenario}_predictions.npz') as z:
            assert np.array_equal(predictions['control'],z['control']);assert np.array_equal(predictions['ownership_only'],z['focus_missing_only'])
        np.savez_compressed(OUT/f'{game}_{scenario}_predictions.npz',**predictions)
        with np.load(ref) as z:
            for reference,key in [('canonical','gt'),('sdk_shadow','alternate')]:
                gt=z[key]
                for name,pred in predictions.items():
                    d=diagnostics(gt,pred,c['seen']);mask=(gt<2)&missing;d['missing_wrong_team']=int((mask&(pred<2)&(pred!=gt)).sum());rows.append(dict(game=game,scenario=scenario,reference=reference,candidate=name,**metric(gt,pred),diagnostics=d))
    return game,rows,time.monotonic()-started

def score():
    assert not (DOC/'scores.json').exists();rows=[]
    with ProcessPoolExecutor(max_workers=2,mp_context=get_context('spawn'),initializer=initialize_score) as pool:
        for f in as_completed([pool.submit(worker,g) for g in plan()['development_games']]):
            g,r,t=f.result();rows+=r;print('Scored',g,round(t,1),flush=True)
    (DOC/'scores.json').write_text(json.dumps(dict(records=rows),indent=2)+'\n')
    result=summarize(rows,plan()['candidates'],ownership.plan());ix={(r['game'],r['scenario'],r['reference'],r['candidate']):r for r in rows}
    for r in result['summary']:
        bases=[ix[g,r['scenario'],r['reference'],'control'] for g in plan()['development_games']];items=[ix[g,r['scenario'],r['reference'],r['candidate']] for g in plan()['development_games']]
        r['dead_f1_gain']=float(np.mean([x['per_class']['D']['f1']-b['per_class']['D']['f1'] for x,b in zip(items,bases)]));bc={k:sum(x['diagnostics'][k] for x in bases) for k in bases[0]['diagnostics']};ct=r['counts'];r['missing_false_live_gain']=ct['missing_false_live']/ct['missing_dead']-bc['missing_false_live']/bc['missing_dead']
        r['gates'].update(dead_f1=r['dead_f1_gain']>=-.002,missing_false_live=r['missing_false_live_gain']<=.01);r['pass_all']=all(r['gates'].values())
    eligible=[n for n in plan()['candidates'] if n not in ['control','ownership_only'] and all(r['pass_all'] for r in result['summary'] if r['candidate']==n)];result['eligible']=eligible;result['selected']=max(eligible,key=lambda c:min(r['gain'] for r in result['summary'] if r['candidate']==c)) if eligible else None;result['decision']='Eligible for independent implementation/regression only' if eligible else 'No candidate qualifies; production unchanged'
    (DOC/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    for r in result['summary']:
        if r['reference']=='canonical':print(r['candidate'],r['scenario'],round(r['mean_spell'],6),round(r['gain'],6),r['wins'],r['gates'],flush=True)
    print('SELECTED',result['selected'],flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','prepare','fit','score']);p.add_argument('--kind',choices=['pointwise','temporal']);a=p.parse_args()
    if a.action=='freeze':freeze()
    else:
        check()
        if a.action=='prepare':prepare()
        elif a.action=='fit':fit(a.kind)
        else:score()
        check()
