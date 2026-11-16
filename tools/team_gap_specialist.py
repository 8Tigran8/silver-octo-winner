"""Bounded next-round ownership study: original train/dev only, no18-match audit."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
import argparse,json,sys,time,subprocess
from datetime import datetime,timezone
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context
import numpy as np
import torch
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
from pff_reader import sha
from pff_development import geometry,spans
from short_gap_study import scenarios,diagnostics
from decoder_development import metric
from team_gap_objective import short_centres,ownership_loss
from spell_weighted_loss import live_spell_weights
from mtah import recovery_candidate,mixed_temporal,temporal,expanded_team,v4
from mtah.methods import boost
DOC=ROOT/'docs/experiments/team-gap-specialist';OUT=ROOT/'reports/team_gap_specialist'
SEQ=ROOT/'reports/spell_weighted_finetune'
SCENARIOS=['natural_detected','label_blind_seed7']

def plan():return json.loads((DOC/'plan.json').read_text())
def guard(game,split):
    if game not in plan()[split+'_games']:raise ValueError('Fixture prohibited for '+split)

def install_read_guard():
    protocol=json.loads((ROOT/'docs/experiments/more-data/pff_protocol.json').read_text());forbidden={g['game_id'] for k in ['final_test','country_bridge_excluded_from_training_and_selection'] for g in protocol[k]}
    import re
    def audit(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        path=os.fsdecode(args[0])
        if any(t in path for t in ['/reports/recovery_opened_audit/','/reports/unseen_bridge_audit/','/reports/pff_mixed_final/','/reports/pff_final/','/docs/experiments/recovery-opened-audit/','/docs/experiments/unseen-bridge-audit/','/evaluation/gt/']):raise PermissionError('Blocked heldout/target evidence read')
        if ('/external/pff/' in path or '/reports/pff_access/' in path) and any(g in re.findall(r'(?<!\d)\d{4,5}(?!\d)',path) for g in forbidden):raise PermissionError('Blocked heldout fixture read')
    sys.addaudithook(audit)

def freeze():
    if (DOC/'freeze.json').exists() or OUT.exists():raise ValueError('Already frozen')
    if subprocess.check_output(['git','status','--porcelain'],text=True):raise ValueError('Commit plan/source first')
    cfg=plan();protocol=json.loads((ROOT/'docs/experiments/more-data/pff_protocol.json').read_text());assert set(cfg['development_games'])=={g['game_id'] for g in protocol['development']}
    files=set((ROOT/'mtah').rglob('*.py'))|set((ROOT/'mtah').rglob('*.json'))
    for folder in ['assets/v4','assets/pff-mixed-candidate','assets/recovery-candidate']:files.update(p for p in (ROOT/folder).rglob('*') if p.is_file())
    files.update(ROOT/n for n in ['tools/team_gap_specialist.py','tools/team_gap_objective.py','tools/spell_weighted_loss.py','tools/pff_reader.py','tools/pff_development.py','tools/short_gap_study.py','tools/decoder_development.py','tools/quality_audit.py','docs/experiments/team-gap-specialist/plan.json','docs/experiments/more-data/pff_protocol.json','ref/official/evaluation/possession_metrics.py','docs/experiments/spell-weighted-finetune/combined_scores.json','reports/spell_weighted_finetune/prepared.json'])
    prepared=json.loads((SEQ/'prepared.json').read_text());assert set(prepared)==set(cfg['training_games'])
    for g,folders in prepared.items():
        guard(g,'training')
        for folder in folders:
            p=ROOT/folder/'meta.json';m=json.loads(p.read_text());files.add(p)
            for name,h in m['feature_sha256'].items():
                if sha(ROOT/folder/(name+'.npy'))!=h:raise ValueError('Changed training arrays')
    for g in cfg['development_games']:
        files.update(ROOT/n for n in [f'external/pff/converted/{g}_geometry.npz',f'reports/pff_access/kloppy_{g}_preflight.json',f'reports/pff_access/references/kloppy_{g}.npz'])
        for s in SCENARIOS:files.add(SEQ/f'combined_{g}_{s}_predictions.npz')
    record=dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),frozen_utc=datetime.now(timezone.utc).isoformat(),files_sha256={str(p.relative_to(ROOT)):sha(p) for p in sorted(files)},training_arrays_verified_from_metadata=True,heldout_opened=False)
    (DOC/'freeze.json').write_text(json.dumps(record,indent=2)+'\n');print('Frozen train/dev-only ownership study',flush=True)

def check():
    for name,h in json.loads((DOC/'freeze.json').read_text())['files_sha256'].items():
        if sha(ROOT/name)!=h:raise ValueError('Frozen source/input changed: '+name)

def initialize():
    global BUNDLE,STATE
    install_read_guard();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    BUNDLE,STATE=recovery_candidate.load_bundle(ROOT/'assets/recovery-candidate/bundle.json')

def prepare_game(game):
    guard(game,'development');started=time.monotonic();data,receipt=geometry(game);parts=spans(data['period']);rows=[]
    for scenario,d in scenarios(data):
        bd,bh,_,_=v4.components(d,parts,STATE[0])
        with threadpool_limits(1):
            x,xs,_=expanded_team.feature_pair(d,parts)
            activity,names=recovery_candidate.activity_features(d,parts)
            assert names==STATE[3];sm=STATE[2].predict_proba(activity[::5])[:,1]
        smallparts=[((a+4)//5,(b+4)//5) for a,b in parts];model,mean,scale=STATE[1]
        nd,nh=mixed_temporal.probabilities(model,x[::5],xs[::5],smallparts,mean,scale);nd,nh=v4.neural_upsample(nd,nh,parts,len(x));ad=v4.neural_upsample(sm,sm,parts,len(x))[0]
        seen=np.isfinite(d['ball'][:,:2]).all(axis=1);pd=.5*bd+.5*nd;ph=.5*bh+.5*nh;pd[~seen]=.75*nd[~seen]+.25*ad[~seen]
        if game==plan()['development_games'][0] and scenario=='natural_detected':
            actual=recovery_candidate.probabilities(d,parts,STATE)
            assert all(np.array_equal(a,b) for a,b in zip(actual,(pd,ph)))
        pred=boost.label(pd,ph,parts,len(pd),BUNDLE['decoder'])
        prior=SEQ/f'combined_{game}_{scenario}_predictions.npz'
        with np.load(prior) as z:assert np.array_equal(pred,z['weighted_short_gap'])
        dest=OUT/'development'/f'{game}_{scenario}.npz'
        np.savez_compressed(dest,pd=pd,ph=ph,bh=bh,nh=nh,seen=seen,x=x[::5],xs=xs[::5],segments=np.array(parts),small_segments=np.array(smallparts))
        rows.append(dict(game=game,scenario=scenario,cache_sha256=sha(dest),geometry_sha256=receipt['geometry_cache_sha256'],parent_prediction_equal=True))
    return game,rows,time.monotonic()-started

def prepare():
    (OUT/'development').mkdir(parents=True,exist_ok=False);rows=[]
    with ProcessPoolExecutor(max_workers=2,mp_context=get_context('spawn'),initializer=initialize) as pool:
        for f in as_completed([pool.submit(prepare_game,g) for g in plan()['development_games']]):
            g,r,seconds=f.result();rows.extend(r);print('Prepared',g,round(seconds,1),flush=True)
    (OUT/'development_manifest.json').write_text(json.dumps(rows,indent=2)+'\n');(DOC/'control_parity.json').write_text(json.dumps(dict(all20_exact_parent_predictions=True,records=rows),indent=2)+'\n')

def fit(name):
    cfg=plan();dest=OUT/name;dest.mkdir(exist_ok=False);initialize();model,mean,scale=STATE[1];model.to('mps');model.train()
    torch.manual_seed(cfg['seed']);torch.use_deterministic_algorithms(True);rng=np.random.default_rng(cfg['seed']);focus_rng=np.random.default_rng(cfg['seed']+1)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=.001,foreach=False)
    prepared=json.loads((SEQ/'prepared.json').read_text());assert set(prepared)==set(cfg['training_games']);seq={};hashes={}
    for g in cfg['training_games']:
        guard(g,'training');seq[g]=[]
        for relative in prepared[g]:
            folder=ROOT/relative;m=json.loads((folder/'meta.json').read_text());hashes[relative]=sha(folder/'meta.json')
            assert all(sha(folder/(k+'.npy'))==h for k,h in m['feature_sha256'].items())
            row={k:np.load(folder/(k+'.npy'),mmap_mode='r') for k in ['x','xs','gt']};row.update(meta=m,weights=live_spell_weights(row['gt'],m['segments']),centres=short_centres(row['gt'],m['segments']));seq[g].append(row)
    groups=[[g for g in seq if '/idsse_' in g],[g for g in seq if '/metrica_' in g],[g for g in seq if g.startswith('skillcorner/')],[g for g in seq if g.startswith('pff/')]]
    radius=256;corelen=256;started=time.monotonic();losses=[];focused_windows=0
    for step in range(cfg['steps']):
        bx=[];by=[];bw=[];bm=[]
        for _ in range(16):
            group=groups[int(rng.integers(4))];game=group[int(rng.integers(len(group)))];rr=seq[game];row=rr[int(rng.integers(len(rr)))];parts=row['meta']['segments'];pi=int(rng.integers(len(parts)));a,b=parts[pi]
            first=int(rng.integers(a,max(a+1,b-corelen+1)));swap=bool(rng.integers(2))
            if name=='focus' and focus_rng.random()<.5 and len(row['centres'][pi]):
                centre=int(focus_rng.choice(row['centres'][pi]));first=int(np.clip(centre-focus_rng.integers(corelen),a,max(a,b-corelen)));focused_windows+=1
            ix=np.clip(np.arange(first-radius,first+corelen+radius),a,b-1);x=row['xs' if swap else 'x'][ix];y=np.array(row['gt'][ix],copy=True)
            if swap:y=np.where(y<2,1-y,y)
            y[:radius]=3;y[radius+corelen:]=3
            bx.append(temporal.normalize(x,mean,scale).T);by.append(y);bw.append(row['weights'][ix]);bm.append(x[:,row['meta']['seen_index']]<.5)
        batch=torch.from_numpy(np.stack(bx)).to('mps');target=torch.from_numpy(np.stack(by).astype(np.int64)).to('mps');weight=torch.from_numpy(np.stack(bw)).to('mps');missing=torch.from_numpy(np.stack(bm)).to('mps')
        opt.zero_grad(set_to_none=True);loss=ownership_loss(model(batch),target,weight,missing,name=='focus');loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();losses.append(float(loss.detach().cpu()))
        if (step+1)%1000==0:print(name,step+1,'loss',float(np.mean(losses[-100:])),round(time.monotonic()-started,1),'seconds',flush=True)
    torch.save(dict(state_dict=model.cpu().state_dict(),mean=torch.from_numpy(mean),scale=torch.from_numpy(scale)),dest/'model.pt')
    meta=dict(candidate=name,training_games=cfg['training_games'],steps=cfg['steps'],seed=cfg['seed'],focused_windows=focused_windows,training_sequences_sha256=hashes,model_sha256=sha(dest/'model.pt'),parent_sha256=sha(ROOT/'assets/recovery-candidate/neural_state.pt'),normalizer_unchanged=True,seconds=time.monotonic()-started,plan_sha256=sha(DOC/'plan.json'))
    (dest/'meta.json').write_text(json.dumps(meta,indent=2)+'\n');(DOC/(name+'_meta.json')).write_text(json.dumps(meta,indent=2)+'\n');print('Fit complete',name,flush=True)

def initialize_score(learned):
    global MODELS
    initialize();MODELS={}
    if learned:
        for name in ['uniform','focus']:
            path=OUT/name/'model.pt';assert sha(path)==json.loads((OUT/name/'meta.json').read_text())['model_sha256'];st=torch.load(path,map_location='cpu',weights_only=True)
            assert np.array_equal(st['mean'].numpy(),STATE[1][1]) and np.array_equal(st['scale'].numpy(),STATE[1][2]);m=mixed_temporal.MixedTemporal(BUNDLE['spec']);m.load_state_dict(st['state_dict']);m.eval();MODELS[name]=m

def worker(item):
    game,phase=item;guard(game,'development');rows=[];started=time.monotonic();cfg=plan();allowed=cfg[phase+'_candidates'];manifest={(r['game'],r['scenario']):r for r in json.loads((OUT/'development_manifest.json').read_text())}
    ref=ROOT/'reports/pff_access/references'/f'kloppy_{game}.npz';receipt=json.loads((ROOT/'reports/pff_access'/f'kloppy_{game}_preflight.json').read_text());assert sha(ref)==receipt['reference_cache_sha256']
    for scenario in SCENARIOS:
        path=OUT/'development'/f'{game}_{scenario}.npz';assert sha(path)==manifest[game,scenario]['cache_sha256']
        with np.load(path) as z:c=dict(z)
        parts=c['segments'].tolist();nh={}
        for name,m in MODELS.items():
            _,h=mixed_temporal.probabilities(m,c['x'],c['xs'],c['small_segments'].tolist(),STATE[1][1],STATE[1][2]);nh[name]=v4.neural_upsample(h,h,parts,len(c['pd']))[0]
        predictions={}
        for name in allowed:
            pd=c['pd'];ph=c['ph'].copy();params=dict(BUNDLE['decoder'])
            if name in ['neural_gap75','neural_gap100']:
                w=.75 if name.endswith('75') else 1.;ph[~c['seen']]=(1-w)*c['bh'][~c['seen']]+w*c['nh'][~c['seen']]
            if name.startswith('uniform'):ph=.5*c['bh']+.5*nh['uniform']
            if name.startswith('focus'):
                ph=.5*c['bh']+.5*nh['focus']
                if 'gap100' in name:ph[~c['seen']]=nh['focus'][~c['seen']]
            if 'short02' in name:params.update(min_team_s=.2,cost_ha=3.)
            predictions[name]=boost.label(pd,ph,parts,len(pd),params)
        np.savez_compressed(OUT/f'{phase}_{game}_{scenario}_predictions.npz',**predictions)
        with np.load(ref) as z:
            for reference,key in [('canonical','gt'),('sdk_shadow','alternate')]:
                gt=z[key];base=metric(gt,predictions['control'])
                for name,pred in predictions.items():
                    score=metric(gt,pred);diag=diagnostics(gt,pred,c['seen']);mask=(gt<2)&~c['seen'];diag['missing_wrong_team']=int(((pred<2)&(pred!=gt)&mask).sum())
                    rows.append(dict(game=game,scenario=scenario,reference=reference,candidate=name,**score,spell_gain=score['spell_f1_mean']-base['spell_f1_mean'],macro_gain=score['macro_f1']-base['macro_f1'],diagnostics=diag))
    (OUT/f'{phase}_{game}_scores.json').write_text(json.dumps(rows,indent=2)+'\n');return game,rows,time.monotonic()-started

def score(phase):
    dest=DOC/(phase+'_scores.json')
    if dest.exists():raise ValueError('Preserve completed results')
    rows=[]
    with ProcessPoolExecutor(max_workers=2,mp_context=get_context('spawn'),initializer=initialize_score,initargs=(phase=='learned',)) as pool:
        for f in as_completed([pool.submit(worker,(g,phase)) for g in plan()['development_games']]):
            g,r,seconds=f.result();rows.extend(r);print('Scored',phase,g,round(seconds,1),flush=True)
    dest.write_text(json.dumps(dict(records=rows),indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','prepare','fit','score']);p.add_argument('--name',choices=['uniform','focus','mechanical','learned']);a=p.parse_args();install_read_guard()
    if a.action=='freeze':freeze()
    else:
        check()
        if a.action=='prepare':prepare()
        elif a.action=='fit':fit(a.name)
        else:score(a.name)
        check()
