"""Bounded train-only loss/link follow-up; original study source stays frozen."""
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
from team_gap_calibration import FocalOwnershipLink,weighted_ownership_bce
from report_team_gap_specialist import summarize
from pff_reader import sha
from mtah import mixed_temporal,temporal,v4
from mtah.methods import boost
from short_gap_study import diagnostics
from decoder_development import metric
from team_gap_objective import short_centres
from spell_weighted_loss import live_spell_weights
DOC=study.DOC;OUT=study.OUT/'calibration'

def cfg():return json.loads((DOC/'calibration_plan.json').read_text())
def check():
    study.install_read_guard();study.check()
    for p,h in json.loads((DOC/'calibration_freeze.json').read_text())['files_sha256'].items():assert sha(ROOT/p)==h,p

def fit():
    config=study.plan();OUT.mkdir(exist_ok=False);study.initialize();model,mean,scale=study.STATE[1];model.to('mps');model.train()
    torch.manual_seed(config['seed']);torch.use_deterministic_algorithms(True);rng=np.random.default_rng(config['seed']);focus_rng=np.random.default_rng(config['seed']+1)
    opt=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=.001,foreach=False)
    prepared=json.loads((study.SEQ/'prepared.json').read_text());assert set(prepared)==set(config['training_games']);seq={};hashes={}
    for g in config['training_games']:
        study.guard(g,'training');seq[g]=[]
        for relative in prepared[g]:
            folder=ROOT/relative;m=json.loads((folder/'meta.json').read_text());hashes[relative]=sha(folder/'meta.json');assert all(sha(folder/(k+'.npy'))==h for k,h in m['feature_sha256'].items())
            row={k:np.load(folder/(k+'.npy'),mmap_mode='r') for k in ['x','xs','gt']};row.update(meta=m,weights=live_spell_weights(row['gt'],m['segments']),centres=short_centres(row['gt'],m['segments']));seq[g].append(row)
    groups=[[g for g in seq if '/idsse_' in g],[g for g in seq if '/metrica_' in g],[g for g in seq if g.startswith('skillcorner/')],[g for g in seq if g.startswith('pff/')]]
    radius=256;corelen=256;started=time.monotonic();losses=[];focused_windows=0
    for step in range(config['steps']):
        bx=[];by=[];bw=[];bm=[]
        for _ in range(16):
            group=groups[int(rng.integers(4))];game=group[int(rng.integers(len(group)))];rr=seq[game];row=rr[int(rng.integers(len(rr)))];parts=row['meta']['segments'];pi=int(rng.integers(len(parts)));a,b=parts[pi]
            first=int(rng.integers(a,max(a+1,b-corelen+1)));swap=bool(rng.integers(2))
            if focus_rng.random()<.5 and len(row['centres'][pi]):
                centre=int(focus_rng.choice(row['centres'][pi]));first=int(np.clip(centre-focus_rng.integers(corelen),a,max(a,b-corelen)));focused_windows+=1
            ix=np.clip(np.arange(first-radius,first+corelen+radius),a,b-1);x=row['xs' if swap else 'x'][ix];y=np.array(row['gt'][ix],copy=True)
            if swap:y=np.where(y<2,1-y,y)
            y[:radius]=3;y[radius+corelen:]=3
            bx.append(temporal.normalize(x,mean,scale).T);by.append(y);bw.append(row['weights'][ix]);bm.append(x[:,row['meta']['seen_index']]<.5)
        batch=torch.from_numpy(np.stack(bx)).to('mps');target=torch.from_numpy(np.stack(by).astype(np.int64)).to('mps');weight=torch.from_numpy(np.stack(bw)).to('mps');missing=torch.from_numpy(np.stack(bm)).to('mps')
        opt.zero_grad(set_to_none=True);loss=weighted_ownership_bce(model(batch),target,weight,missing);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();losses.append(float(loss.detach().cpu()))
        if (step+1)%1000==0:print('focus_bce',step+1,'loss',float(np.mean(losses[-100:])),round(time.monotonic()-started,1),'seconds',flush=True)
    torch.save(dict(state_dict=model.cpu().state_dict(),mean=torch.from_numpy(mean),scale=torch.from_numpy(scale)),OUT/'model.pt')
    meta=dict(candidate='focus_bce',training_games=config['training_games'],steps=config['steps'],seed=config['seed'],focused_windows=focused_windows,training_sequences_sha256=hashes,model_sha256=sha(OUT/'model.pt'),parent_sha256=sha(ROOT/'assets/recovery-candidate/neural_state.pt'),normalizer_unchanged=True,seconds=time.monotonic()-started,plan_sha256=sha(DOC/'calibration_plan.json'))
    matched=json.loads((DOC/'focus_meta.json').read_text());assert focused_windows==matched['focused_windows'] and hashes==matched['training_sequences_sha256']
    (OUT/'meta.json').write_text(json.dumps(meta,indent=2)+'\n');(DOC/'focus_bce_meta.json').write_text(json.dumps(meta,indent=2)+'\n');print('Fit complete focus_bce',flush=True)

def initialize_score():
    global MODELS
    study.initialize_score(True);st=torch.load(OUT/'model.pt',map_location='cpu',weights_only=True);assert sha(OUT/'model.pt')==json.loads((OUT/'meta.json').read_text())['model_sha256']
    assert np.array_equal(st['mean'].numpy(),study.STATE[1][1]) and np.array_equal(st['scale'].numpy(),study.STATE[1][2]);m=mixed_temporal.MixedTemporal(study.BUNDLE['spec']);m.load_state_dict(st['state_dict']);m.eval()
    MODELS=dict(focus_link=FocalOwnershipLink(study.MODELS['focus']),focus_bce=m)

def worker(game):
    study.guard(game,'development');started=time.monotonic();rows=[];manifest={(r['game'],r['scenario']):r for r in json.loads((study.OUT/'development_manifest.json').read_text())}
    ref=ROOT/'reports/pff_access/references'/f'kloppy_{game}.npz';receipt=json.loads((ROOT/'reports/pff_access'/f'kloppy_{game}_preflight.json').read_text());assert sha(ref)==receipt['reference_cache_sha256']
    for scenario in study.SCENARIOS:
        path=study.OUT/'development'/f'{game}_{scenario}.npz';assert sha(path)==manifest[game,scenario]['cache_sha256']
        with np.load(path) as z:c=dict(z)
        parts=c['segments'].tolist();neural={}
        for name,m in MODELS.items():
            _,h=mixed_temporal.probabilities(m,c['x'],c['xs'],c['small_segments'].tolist(),study.STATE[1][1],study.STATE[1][2]);neural[name]=v4.neural_upsample(h,h,parts,len(c['pd']))[0]
        predictions={}
        for name in cfg()['candidates']:
            ph=c['ph'].copy()
            if name!='control':
                model='focus_link' if name.startswith('focus_link') else 'focus_bce'
                if name.endswith('gap100'):ph=.5*c['bh']+.5*neural[model]
                ph[~c['seen']]=neural[model][~c['seen']]
            predictions[name]=boost.label(c['pd'],ph,parts,len(c['pd']),study.BUNDLE['decoder'])
        with np.load(study.OUT/f'learned_{game}_{scenario}_predictions.npz') as z:assert np.array_equal(z['control'],predictions['control'])
        np.savez_compressed(OUT/f'{game}_{scenario}_predictions.npz',**predictions)
        with np.load(ref) as z:
            for reference,key in [('canonical','gt'),('sdk_shadow','alternate')]:
                gt=z[key]
                for name,pred in predictions.items():
                    d=diagnostics(gt,pred,c['seen']);mask=(gt<2)&~c['seen'];d['missing_wrong_team']=int(((pred<2)&(pred!=gt)&mask).sum());rows.append(dict(game=game,scenario=scenario,reference=reference,candidate=name,**metric(gt,pred),diagnostics=d))
    return game,rows,time.monotonic()-started

def score():
    dest=DOC/'calibration_scores.json';assert not dest.exists();rows=[]
    with ProcessPoolExecutor(max_workers=2,mp_context=get_context('spawn'),initializer=initialize_score) as pool:
        for f in as_completed([pool.submit(worker,g) for g in study.plan()['development_games']]):
            g,r,t=f.result();rows.extend(r);print('Scored calibration',g,round(t,1),flush=True)
    dest.write_text(json.dumps(dict(records=rows),indent=2)+'\n');result=summarize(rows,cfg()['candidates'],study.plan());result.update(score_sha256=sha(dest),plan_sha256=sha(DOC/'calibration_plan.json'),scope='Adaptive loss/link follow-up on original10development only; no18match reads; frozen gates')
    (DOC/'calibration_result.json').write_text(json.dumps(result,indent=2)+'\n')
    for r in result['summary']:
        if r['reference']=='canonical':print(r['candidate'],r['scenario'],r['mean_spell'],r['gain'],r['wins'],r['short_recall'],r['missing_wrong_team_rate'],r['gates'],flush=True)
    print('Selected',result['selected'],flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['fit','score']);a=p.parse_args();check()
    if a.action=='fit':fit()
    else:score()
    check()
