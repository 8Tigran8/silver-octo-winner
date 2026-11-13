"""Fixed 19-to-29-game capacity-controlled experiment; release stays v3.

All scored-game augmentations are excluded. Previously scored games are reused
research data, never advertised as a fresh holdout. No competition data is read.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from multiprocessing import get_context
import os
from pathlib import Path
import sys
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cloudpickle
import numpy as np
from threadpoolctl import threadpool_limits
from mtah import expanded_team as expanded, possession_context as context, degrade
from mtah.methods import boost, boost_train
from quality_audit import official, read_feature, sha
from decoder_development import metric
from skillcorner_reader import read_match
from expanded_team_trial import load_model as old_model

FOLDER=ROOT/'reports/v5_search'
PRIOR=ROOT/'reports/v4_search'
OLDER=ROOT/'reports/v2_search'
AUDIT=ROOT/'reports/quality_audit/380987b3029c'
PLAN=ROOT/'docs/experiments/more-data/plan.json'


def plan():
    p=json.loads(PLAN.read_text())
    if len(p['training_games'])!=29 or len(set(p['training_games']))!=29:
        raise ValueError('Invalid training registry')
    if not set(p['evaluation_games']) < set(p['training_games']):
        raise ValueError('Invalid evaluation registry')
    if sha(ROOT/'mtah/possession_context.py')!=p['context_source_sha256']:
        raise ValueError('Feature capacity changed')
    for path,expected in p['source_manifests_sha256'].items():
        if sha(ROOT/path)!=expected:raise ValueError('Source identity changed')
    return p


def key(game):return game.replace('/','__')


def split(excluded):
    p=plan();names=p['training_games']
    if excluded is not None and excluded not in p['evaluation_games']:
        raise ValueError('Unregistered evaluation game')
    train=[n for n in names if n!=excluded]
    expanded.assert_training_split(names,train,excluded)
    return train


def row_path(game):
    if game not in plan()['training_games']:raise ValueError('Not a training game')
    if game in expanded.games():return PRIOR/'rows'/(key(game)+'.npz')
    return FOLDER/'rows'/(key(game)+'.npz')


def prepare(game):
    p=plan()
    if game not in p['new_training_games']:raise ValueError('Only added games need conversion')
    name=game.split('/')[1];destination=row_path(game);metadata=destination.with_suffix('.json')
    if metadata.exists():
        m=json.loads(metadata.read_text())
        if m['plan_sha256']!=sha(PLAN) or m['rows_sha256']!=sha(destination):raise ValueError('Stale rows')
        return game
    clean=read_match(ROOT/'external/skillcorner'/name,name);parts=[];columns=None
    for scenario,(seed,params,stride) in expanded.SKILL_SCENARIOS.items():
        data=clean if seed is None else degrade.degrade(clean,seed=seed,params=params)
        x,xs,cols=expanded.feature_pair(data)
        if columns is not None and columns!=cols:raise ValueError('Changed schema')
        columns=cols;parts.append(context.rows(x,xs,data['gt'],stride=stride))
    x=np.concatenate([r[0] for r in parts]);y=np.concatenate([r[1] for r in parts])
    destination.parent.mkdir(parents=True,exist_ok=True);np.savez(destination,X=x,y=y,cols=np.array(columns))
    metadata.write_text(json.dumps({'game':game,'rows':len(y),'columns':len(columns),
        'rows_sha256':sha(destination),'plan_sha256':sha(PLAN),'reference_live_frames':int((clean['gt']<2).sum())},indent=2)+'\n')
    return game


def diagnostic(model,x,y):
    # Fixed evenly spaced augmented/symmetric rows; descriptive, not selection.
    idx=np.linspace(0,len(y)-1,min(2048,len(y)),dtype=int)
    p=model.predict_proba(x[idx])[:,1];target=y[idx];pred=p>=.5
    clipped=np.clip(p,1e-7,1-1e-7)
    return {'sampled_rows':len(idx),'log_loss':float(-(target*np.log(clipped)+(1-target)*np.log(1-clipped)).mean()),
            'accuracy':float((pred==target).mean())}


def fit_job(excluded):
    train=split(excluded);destination=FOLDER/'models'/('exclude__'+key(excluded) if excluded else 'all_twenty_nine')
    destination.parent.mkdir(parents=True,exist_ok=True);model_path=destination.with_suffix('.pkl');meta_path=destination.with_suffix('.json')
    meta={'excluded':excluded,'train':train,'plan_sha256':sha(PLAN),
          'rows_sha256':{n:sha(row_path(n)) for n in train}}
    if meta_path.exists():
        prior=json.loads(meta_path.read_text())
        if any(prior.get(k)!=v for k,v in meta.items()) or sha(model_path)!=prior['model_sha256']:raise ValueError('Stale fit')
        return excluded or 'all_twenty_nine'
    with threadpool_limits(1):
        parts=[]
        for game in train:
            with np.load(row_path(game),allow_pickle=False) as z:parts.append((z['X'],z['y']))
        model=context.fit(parts)
        meta['training_diagnostic']={n:diagnostic(model,*part) for n,part in zip(train,parts)}
        meta['sampled_training_rows']=sum(len(part[1]) for part in parts)
        if excluded is not None:
            with np.load(row_path(excluded),allow_pickle=False) as z:
                meta['heldout_diagnostic']=diagnostic(model,z['X'],z['y'])
                meta['v3_heldout_diagnostic']=diagnostic(old_model(excluded),z['X'],z['y'])
    with model_path.open('wb') as stream:cloudpickle.dump(model,stream)
    meta.update(model_sha256=sha(model_path),fitted_utc=datetime.now(timezone.utc).isoformat())
    meta_path.write_text(json.dumps(meta,indent=2)+'\n');return excluded or 'all_twenty_nine'


def load(excluded):
    path=FOLDER/'models'/('exclude__'+key(excluded)+'.pkl' if excluded else 'all_twenty_nine.pkl')
    m=json.loads(path.with_suffix('.json').read_text())
    if m['train']!=split(excluded) or m['plan_sha256']!=sha(PLAN) or m['model_sha256']!=sha(path):raise ValueError('Fit integrity failed')
    with path.open('rb') as stream:return cloudpickle.load(stream)


def score_job(game):
    family,name=game.split('/');cohort='known' if family=='original' else 'development'
    scenarios=['matched_seed0','label_blind_seed1','stress_seed1'] if family=='original' else ['natural_detected','label_blind_seed7']
    model=load(game);records=[]
    params=dict(boost_train.decoder_params(),**json.loads((ROOT/'mtah/methods/v2_params.json').read_text())['decoder'])
    with threadpool_limits(1):
        for scenario in scenarios:
            with np.load(OLDER/('known' if family=='original' else 'skillcorner')/f'{name}__{scenario}__probs.npz') as z:
                pd,ph,gt=z['pd'],z['ph'],z['gt']
            if family=='original':
                f=read_feature(AUDIT,name,scenario)
                if not np.array_equal(gt,f['gt']):raise ValueError('Reference mismatch')
                segments=official.valid_runs(gt);x,_=context.extend(f['X'],f['cols'].tolist(),segments);xs,_=context.extend(f['Xs'],f['cols'].tolist(),segments)
            else:
                with np.load(PRIOR/'evaluation'/f'{name}__{scenario}.npz') as z:
                    if not np.array_equal(gt,z['gt']):raise ValueError('Reference mismatch')
                    x,xs=z['X'],z['Xs']
                segments=official.valid_runs(gt)
            new=context.predict(model,x,xs)
            with np.load(PRIOR/'probabilities'/f'{name}__{scenario}.npz') as z:old=z['ph']
            for candidate,team in [('v3',old),('data29',new)]:
                pred=boost.label(pd,.5*ph+.5*team,segments,len(gt),params)
                records.append({'match':name,'game':game,'cohort':cohort,'scenario':scenario,'candidate':candidate,**metric(gt,pred)})
    return game,records


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['prepare','fit','score']);parser.add_argument('--jobs',type=int,choices=range(1,5),default=2);args=parser.parse_args();p=plan();FOLDER.mkdir(exist_ok=True)
    if args.mode=='prepare':jobs=p['new_training_games'];function=prepare
    elif args.mode=='fit':jobs=[None]+p['evaluation_games'];function=fit_job
    else:
        if (FOLDER/'development_results.json').exists():raise ValueError('Never overwrite completed scores')
        jobs=p['evaluation_games'];function=score_job
    records=[]
    with ProcessPoolExecutor(max_workers=args.jobs,mp_context=get_context('spawn')) as pool:
        for future in as_completed([pool.submit(function,j) for j in jobs]):
            result=future.result()
            if args.mode=='score':game,rows=result;records.extend(rows)
            else:game=result
            print(game,args.mode,'PASS',flush=True)
    if args.mode=='score':(FOLDER/'development_results.json').write_text(json.dumps({'plan_sha256':sha(PLAN),'records':records},indent=2)+'\n')


if __name__=='__main__':main()
