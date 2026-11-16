"""Preregistered player-activity dead/live balance study; original train/dev only."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
import argparse,json,sys,time,subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context
from datetime import datetime,timezone
import numpy as np
import cloudpickle
from threadpoolctl import threadpool_limits
from sklearn.ensemble import HistGradientBoostingClassifier
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
import team_gap_specialist as ownership
from pff_reader import sha
from report_team_gap_specialist import summarize
from mtah import v4,mixed_temporal
from mtah.methods import boost
from short_gap_study import diagnostics
from decoder_development import metric
DOC=ROOT/'docs/experiments/dead-gap-balance';OUT=ROOT/'reports/dead_gap_balance';OLD=ROOT/'reports/short_gap_recovery'

def plan():return json.loads((DOC/'plan.json').read_text())
def check():
    ownership.install_read_guard();ownership.check()
    for p,h in json.loads((DOC/'freeze.json').read_text())['files_sha256'].items():
        if sha(ROOT/p)!=h:raise ValueError('Frozen input changed: '+p)

def balanced_weights(labels,weights):
    totals=np.bincount(labels,weights=weights,minlength=2)
    if (totals<=0).any():raise ValueError('Both live and dead classes required')
    result=weights/(2*totals[labels]);return result*len(result)/result.sum()

def routed_probabilities(name,cache,old_activity,new_activity,focus):
    pd=cache['pd'].copy();ph=cache['ph'].copy();missing=~cache['seen']
    if name=='control':return pd,ph
    if name=='ownership_only':ph[missing]=focus[missing];return pd,ph
    family=name.split('_')[0];weight=int(name.split('_')[1])/100
    # Parent missing D is exactly .75*weightedNN + .25*old activity.
    nd=(cache['pd'][missing]-.25*old_activity[missing])/.75
    if (nd< -1e-5).any() or (nd>1+1e-5).any():raise ValueError('Parent fusion does not reconstruct probability')
    activity=old_activity if family=='activity' else new_activity
    pd[missing]=(1-weight)*nd+weight*activity[missing]
    if name.endswith('_focus'):ph[missing]=focus[missing]
    return pd,ph

def freeze():
    ownership.install_read_guard();ownership.check();assert not (DOC/'freeze.json').exists() and not OUT.exists()
    files=[ROOT/'tools/dead_gap_balance.py',ROOT/'tests/test_dead_gap_balance.py',DOC/'plan.json',ROOT/'tools/report_team_gap_specialist.py']
    meta=json.loads((ROOT/'assets/recovery-candidate/activity_meta.json').read_text())
    assert set(plan()['training_games'])==set(meta['training_games'])
    for g in plan()['training_games']:
        ownership.guard('pff/'+g,'training');p=OLD/'training'/f'{g}.npz';assert sha(p)==meta['training_cache_sha256'][g];files.append(p)
    files.extend([ownership.OUT/'focus/model.pt',ownership.OUT/'focus/meta.json',ownership.OUT/'development_manifest.json'])
    manifest={(r['game'],r['scenario']):r for r in json.loads((ownership.OUT/'development_manifest.json').read_text())}
    for g in plan()['development_games']:
        ownership.guard(g,'development')
        for scenario in ownership.SCENARIOS:
            p=ownership.OUT/'development'/f'{g}_{scenario}.npz';assert sha(p)==manifest[g,scenario]['cache_sha256'];files.extend([p,OLD/'development'/f'{g}_{scenario}.npz'])
    (DOC/'freeze.json').write_text(json.dumps(dict(frozen_utc=datetime.now(timezone.utc).isoformat(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),files_sha256={str(p.relative_to(ROOT)):sha(p) for p in files}),indent=2)+'\n');print('Frozen',flush=True)

def fit():
    OUT.mkdir(exist_ok=False);xs=[];ys=[];weights=[];columns=None
    for g in plan()['training_games']:
        ownership.guard('pff/'+g,'training')
        with np.load(OLD/'training'/f'{g}.npz') as z:
            if columns is not None:assert np.array_equal(columns,z['columns'])
            columns=z['columns'];xs.append(z['x']);ys.append(z['y']);weights.append(np.full(len(z['y']),1/len(z['y'])))
    x=np.concatenate(xs);y=np.concatenate(ys);w=np.concatenate(weights);w=balanced_weights(y,w)
    model=HistGradientBoostingClassifier(**plan()['expert_parameters']);start=time.monotonic()
    with threadpool_limits(2):model.fit(x,y,sample_weight=w)
    with (OUT/'balanced.pkl').open('wb') as f:cloudpickle.dump(model,f)
    meta=dict(training_games=plan()['training_games'],rows=len(y),columns=columns.tolist(),model_sha256=sha(OUT/'balanced.pkl'),weighted_class_mass=np.bincount(y,weights=w).tolist(),seconds=time.monotonic()-start,plan_sha256=sha(DOC/'plan.json'))
    (DOC/'model_meta.json').write_text(json.dumps(meta,indent=2)+'\n');print('Fit complete',len(y),round(meta['seconds'],1),flush=True)

def initialize():
    global NEW
    ownership.initialize_score(True)
    assert sha(OUT/'balanced.pkl')==json.loads((DOC/'model_meta.json').read_text())['model_sha256']
    with (OUT/'balanced.pkl').open('rb') as f:NEW=cloudpickle.load(f)

def worker(game):
    ownership.guard(game,'development');rows=[];start=time.monotonic();receipt=json.loads((ROOT/f'reports/pff_access/kloppy_{game}_preflight.json').read_text());ref=ROOT/f'reports/pff_access/references/kloppy_{game}.npz';assert sha(ref)==receipt['reference_cache_sha256']
    for scenario in ownership.SCENARIOS:
        with np.load(ownership.OUT/'development'/f'{game}_{scenario}.npz') as z:c=dict(z)
        parts=c['segments'].tolist()
        with np.load(OLD/'development'/f'{game}_{scenario}.npz') as z:
            assert np.array_equal(z['segments'],c['segments']) and np.array_equal(z['seen'],c['seen']);assert z['columns'].tolist()==ownership.STATE[3];activity=z['activity']
        with threadpool_limits(1):old=ownership.STATE[2].predict_proba(activity)[:,1];new=NEW.predict_proba(activity)[:,1]
        old,new=v4.neural_upsample(old,new,parts,len(c['pd']))
        # Prove arithmetic reconstruction matches current NN before policy comparison.
        nd,nh=mixed_temporal.probabilities(ownership.STATE[1][0],c['x'],c['xs'],c['small_segments'].tolist(),ownership.STATE[1][1],ownership.STATE[1][2]);nd=v4.neural_upsample(nd,nd,parts,len(c['pd']))[0]
        assert np.allclose(c['pd'][~c['seen']],.75*nd[~c['seen']]+.25*old[~c['seen']],atol=1e-7,rtol=0)
        _,focus=mixed_temporal.probabilities(ownership.MODELS['focus'],c['x'],c['xs'],c['small_segments'].tolist(),ownership.STATE[1][1],ownership.STATE[1][2]);focus=v4.neural_upsample(focus,focus,parts,len(c['pd']))[0]
        predictions={}
        for name in plan()['candidates']:
            pd,ph=routed_probabilities(name,c,old,new,focus);predictions[name]=boost.label(pd,ph,parts,len(pd),ownership.BUNDLE['decoder'])
        with np.load(ownership.OUT/f'followup_{game}_{scenario}_predictions.npz') as z:
            assert np.array_equal(predictions['control'],z['control']);assert np.array_equal(predictions['ownership_only'],z['focus_missing_only'])
        np.savez_compressed(OUT/f'{game}_{scenario}_predictions.npz',**predictions)
        for reference,key in [('canonical','gt'),('sdk_shadow','alternate')]:
            with np.load(ref) as z:gt=z[key]
            for name,pred in predictions.items():
                d=diagnostics(gt,pred,c['seen']);mask=(gt<2)&~c['seen'];d['missing_wrong_team']=int((mask&(pred<2)&(pred!=gt)).sum());rows.append(dict(game=game,scenario=scenario,reference=reference,candidate=name,**metric(gt,pred),diagnostics=d))
    return game,rows,time.monotonic()-start

def report(rows):
    result=summarize(rows,plan()['candidates'],dict(ownership.plan(),development_games=plan()['development_games']));ix={(r['game'],r['scenario'],r['reference'],r['candidate']):r for r in rows}
    for r in result['summary']:
        bases=[ix[g,r['scenario'],r['reference'],'control'] for g in plan()['development_games']];items=[ix[g,r['scenario'],r['reference'],r['candidate']] for g in plan()['development_games']]
        # Official per-class keys are strings; fail closed if format changes.
        r['dead_f1_gain']=float(np.mean([x['per_class']['D']['f1']-b['per_class']['D']['f1'] for x,b in zip(items,bases)]))
        bc={k:sum(x['diagnostics'][k] for x in bases) for k in bases[0]['diagnostics']};ct=r['counts'];r['missing_false_live_gain']=ct['missing_false_live']/ct['missing_dead']-bc['missing_false_live']/bc['missing_dead']
        r['gates'].update(dead_f1=r['dead_f1_gain']>=-.002,missing_false_live=r['missing_false_live_gain']<=.01);r['pass_all']=all(r['gates'].values())
    eligible=[n for n in plan()['candidates'] if n not in ['control','ownership_only'] and all(r['pass_all'] for r in result['summary'] if r['candidate']==n)];result['eligible']=eligible
    result['selected']=max(eligible,key=lambda c:min(r['gain'] for r in result['summary'] if r['candidate']==c)) if eligible else None;result['decision']='Eligible for independent implementation/regression only' if eligible else 'No candidate qualifies; production remains unchanged'
    result['scope']='Original10development games only;36PFFtrain activity reweighting; inherited65train models. No18audit or targetGT.'
    (DOC/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    for r in result['summary']:
        if r['reference']=='canonical':print(r['candidate'],r['scenario'],round(r['mean_spell'],6),round(r['gain'],6),r['wins'],r['gates'],flush=True)
    print('SELECTED',result['selected'],flush=True)

def score():
    assert not (DOC/'scores.json').exists();rows=[]
    with ProcessPoolExecutor(max_workers=2,mp_context=get_context('spawn'),initializer=initialize) as pool:
        for f in as_completed([pool.submit(worker,g) for g in plan()['development_games']]):
            g,r,t=f.result();rows+=r;print('Scored',g,round(t,1),flush=True)
    (DOC/'scores.json').write_text(json.dumps(dict(records=rows),indent=2)+'\n');report(rows)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','fit','score','report']);a=p.parse_args()
    if a.action=='freeze':freeze()
    else:
        check()
        if a.action=='fit':fit()
        elif a.action=='score':score()
        else:report(json.loads((DOC/'scores.json').read_text())['records'])
        check()
