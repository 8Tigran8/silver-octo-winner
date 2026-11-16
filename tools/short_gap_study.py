"""Bounded recovery study: original 36 training / 10 development fixtures only."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[key]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
from pff_reader import sha
from pff_development import geometry,spans
from pff_mixed_score import decoder
from decoder_development import metric
from quality_audit import official
from short_gap_methods import activity_features,policy
DOC=ROOT/'docs/experiments/short-gap-recovery'
OUT=ROOT/'reports/short_gap_recovery'


def plan(): return json.loads((DOC/'plan.json').read_text())


def guard(game,split):
    if game not in plan()[split]: raise ValueError('Fixture not allowed for this operation')


def check():
    frozen=json.loads((DOC/'freeze.json').read_text())
    for file,digest in frozen['files_sha256'].items():
        if sha(ROOT/file)!=digest:raise ValueError('Frozen study changed: '+file)


def scenarios(data):
    from mtah import degrade,expanded_team
    dummy=np.where(data['period']>0,0,3).astype(np.int8)
    blind=degrade.degrade(dict(data,gt=dummy),seed=7,params=expanded_team.BLIND);blind.pop('gt')
    return [('natural_detected',data),('label_blind_seed7',blind)]


def references(game,receipt):
    path=ROOT/'reports/pff_access/references'/f'kloppy_{game}.npz'
    if sha(path)!=receipt['reference_cache_sha256']:raise ValueError('Reference changed')
    with np.load(path) as z:return dict(canonical=z['gt'],sdk_shadow=z['alternate'])


def initialize():
    global ASSETS,STATE
    from mtah import v4,mixed_candidate
    ASSETS=v4.load();_,STATE=mixed_candidate.load_bundle(ROOT/'assets/pff-mixed-candidate/bundle.json')


def prepare_dev(game):
    guard(game,'development_games')
    from mtah import mixed_candidate,v4
    data,receipt=geometry(game);parts=spans(data['period']);start=time.monotonic()
    for scenario,d in scenarios(data):
        path=OUT/'development'/f'{game}_{scenario}.npz'
        if path.exists():raise ValueError('Do not overwrite completed preparation')
        _,current=mixed_candidate.paired_probabilities(d,parts,ASSETS,STATE)
        bd,bh,_,_=v4.components(d,parts,ASSETS)
        pd,ph=current;nd=2*pd-bd
        x,cols=activity_features(d,parts)
        np.savez_compressed(path,pd=pd,ph=ph,nd=nd,seen=np.isfinite(d['ball'][:,:2]).all(axis=1),
                            ball=d['ball'],segments=np.array(parts),activity=x[::5],columns=np.array(cols))
    return game,dict(seconds=time.monotonic()-start,geometry_sha256=receipt['geometry_cache_sha256'])


def prepare_train(game):
    guard(game,'training_games')
    data,receipt=geometry(game);parts=spans(data['period']);gt=references(game,receipt)['canonical']
    start=time.monotonic();rows=[];labels=[]
    for scenario,d in scenarios(data):
        x,cols=activity_features(d,parts)
        ids=np.arange(0,len(gt),10)
        ids=ids[(gt[ids]<3)&(~np.isfinite(d['ball'][ids,:2]).all(axis=1))]
        rows.append(x[ids]);labels.append((gt[ids]==2).astype(np.int8))
    path=OUT/'training'/f'{game}.npz'
    if path.exists():raise ValueError('Do not overwrite training cache')
    np.savez_compressed(path,x=np.concatenate(rows),y=np.concatenate(labels),columns=np.array(cols))
    return game,dict(seconds=time.monotonic()-start,rows=sum(map(len,labels)),geometry_sha256=receipt['geometry_cache_sha256'],
                     reference_sha256=receipt['reference_cache_sha256'])


def prepare(split):
    folder=OUT/('development' if split=='dev' else 'training');folder.mkdir(parents=True,exist_ok=False)
    ids=plan()['development_games' if split=='dev' else 'training_games'];receipts={}
    with ProcessPoolExecutor(max_workers=2,mp_context=get_context('spawn'),initializer=initialize if split=='dev' else None) as pool:
        work=prepare_dev if split=='dev' else prepare_train
        for future in as_completed([pool.submit(work,g) for g in ids]):
            game,info=future.result();receipts[game]=info;print(split,game,round(info['seconds'],1),'seconds',flush=True)
    (OUT/(split+'_preparation.json')).write_text(json.dumps(receipts,indent=2)+'\n')


def fit():
    import cloudpickle
    from sklearn.ensemble import HistGradientBoostingClassifier
    from threadpoolctl import threadpool_limits
    xs=[];ys=[];weights=[];hashes={};columns=None
    for game in plan()['training_games']:
        guard(game,'training_games');path=OUT/'training'/f'{game}.npz'
        with np.load(path) as z:
            if columns is not None and not np.array_equal(columns,z['columns']):raise ValueError('Feature schema differs')
            columns=z['columns'];xs.append(z['x']);ys.append(z['y']);weights.append(np.full(len(z['y']),1/len(z['y'])))
        hashes[game]=sha(path)
    x=np.concatenate(xs);y=np.concatenate(ys);w=np.concatenate(weights);w*=len(w)/w.sum()
    with threadpool_limits(2):model=HistGradientBoostingClassifier(**plan()['expert_parameters']).fit(x,y,sample_weight=w)
    if not np.array_equal(model.classes_,[0,1]):raise ValueError('Missing binary class')
    path=OUT/'gap_expert.pkl'
    if path.exists():raise ValueError('Do not overwrite model')
    with path.open('wb') as f:cloudpickle.dump(model,f)
    meta=dict(training_games=plan()['training_games'],training_cache_sha256=hashes,rows=len(y),columns=columns.tolist(),
              model_sha256=sha(path),plan_sha256=sha(DOC/'plan.json'),class_fraction_dead=float(y.mean()))
    (OUT/'expert_meta.json').write_text(json.dumps(meta,indent=2)+'\n');print('Expert fit complete',len(y),'rows',flush=True)


def diagnostics(gt,pred,seen):
    runs=official.valid_runs(gt);truth=official.possession_spells(gt,runs)
    predicted=official.possession_spells(official.fill_abstentions(pred,runs),runs)
    matched=official.matched_gt_flags(official.spell_overlaps(truth,predicted),len(truth),.5)
    duration=np.array([(b-a)/25 for a,b,_ in truth]);short=duration<3
    missing_live=(gt<2)&~seen;missing_dead=(gt==2)&~seen
    return dict(short_reference=int(short.sum()),short_matched=int(matched[short].sum()),
                missing_live=int(missing_live.sum()),missing_false_dead=int(((pred==2)&missing_live).sum()),
                missing_dead=int(missing_dead.sum()),missing_false_live=int(((pred<2)&missing_dead).sum()))


def score_game(item):
    game,phase=item;guard(game,'development_games')
    from mtah import v4
    from mtah.methods import boost
    from threadpoolctl import threadpool_limits
    model=None
    if phase=='expert':
        import cloudpickle
        meta=json.loads((OUT/'expert_meta.json').read_text())
        if sha(OUT/'gap_expert.pkl')!=meta['model_sha256']:raise ValueError('Expert changed')
        with (OUT/'gap_expert.pkl').open('rb') as f:model=cloudpickle.load(f)
    receipt=json.loads((ROOT/'reports/pff_access'/f'kloppy_{game}_preflight.json').read_text());refs=references(game,receipt)
    baseline=json.loads((ROOT/'validation/observation_regression.json').read_text())['records']
    baseline={(r['game'],r['scenario'],r['reference']):r for r in baseline}
    rows=[];start=time.monotonic()
    for scenario in plan()['scenarios']:
        path=OUT/'development'/f'{game}_{scenario}.npz'
        with np.load(path) as z:cache=dict(z)
        expert=None
        if model is not None:
            with threadpool_limits(1):small=model.predict_proba(cache['activity'])[:,1]
            expert=v4.neural_upsample(small,small,cache['segments'],len(cache['pd']))[0]
        predictions={}
        for name in plan()[phase+'_candidates']:
            pd,ph,params=policy(name,cache,decoder(),expert)
            predictions[name]=boost.label(pd,ph,cache['segments'],len(pd),params)
        for name,pred in predictions.items():
            for reference,gt in refs.items():
                score=metric(gt,pred);base=baseline[game,scenario,reference]
                if name=='control' and any(score[k]!=base[k] for k in ['spell_f1_mean','macro_f1','per_class']):
                    raise ValueError('Current control does not reproduce audited baseline')
                rows.append(dict(game=game,scenario=scenario,candidate=name,reference=reference,**score,
                                 spell_gain=score['spell_f1_mean']-base['spell_f1_mean'],macro_gain=score['macro_f1']-base['macro_f1'],
                                 diagnostics=diagnostics(gt,pred,cache['seen'])))
        np.savez_compressed(OUT/f'{phase}_{game}_{scenario}_predictions.npz',**predictions)
    (OUT/f'{phase}_{game}_scores.json').write_text(json.dumps(rows,indent=2)+'\n')
    return game,rows,time.monotonic()-start


def score(phase):
    destination=OUT/(phase+'_results.json')
    if destination.exists():raise ValueError('Never overwrite completed evaluation')
    rows=[]
    with ProcessPoolExecutor(max_workers=2,mp_context=get_context('spawn')) as pool:
        for future in as_completed([pool.submit(score_game,(g,phase)) for g in plan()['development_games']]):
            game,rr,seconds=future.result();rows.extend(rr);print(phase,game,round(seconds,1),'seconds',flush=True)
    destination.write_text(json.dumps(dict(plan_sha256=sha(DOC/'plan.json'),records=rows),indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare-dev','prepare-train','fit','mechanical','expert']);a=parser.parse_args()
    check()
    if a.action.startswith('prepare'):prepare(a.action.split('-')[1])
    elif a.action=='fit':fit()
    else:score(a.action)
    check()
