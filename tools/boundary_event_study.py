"""Preregistered boundary classifier, isolated from every scored old/dev game."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[key]='2'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cloudpickle
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from mtah import boundary_events as event,features2,possession_context as context,stack,v4,mixed_candidate,degrade,expanded_team
from mtah.methods import boost
from pff_development import geometry,spans
from pff_mixed_score import decoder,neural
from decoder_development import metric
import pff_mixed_train as previous

DOC=ROOT/'docs/experiments/boundary-events';PLAN=DOC/'plan.json';OUT=ROOT/'reports/boundary_events'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def plan():return json.loads(PLAN.read_text())
def write(path,obj):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj,indent=2)+'\n')


def names():
    fake=dict(players=np.ones((500,4,3))*.5,ball=np.ones((500,3))*.5,
              slot_team=np.array([0,0,1,1]),pitch=np.array([105.,68.]))
    x,cols=features2.build(fake,[(0,500)]);_,cols=context.extend(x,cols,[(0,500)])
    manifest=json.loads((ROOT/'assets/v4/manifest.json').read_text())
    if hashlib.sha256(json.dumps(cols,separators=(',',':')).encode()).hexdigest()!=manifest['feature_columns_sha256']:
        raise ValueError('Inherited feature schema differs')
    return cols


def source_hashes():
    return {name:sha(ROOT/name) for name in ['mtah/boundary_events.py','tools/boundary_event_study.py',
        'mtah/features2.py','mtah/possession_context.py','mtah/methods/boost.py']}


def fit():
    cfg=plan();dest=OUT/'models';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'complete.json').exists():raise ValueError('Preserve completed fit')
    if sha(ROOT/'assets/pff-mixed-candidate/bundle.json')!=cfg['base_bundle_sha256']:raise ValueError('Base candidate changed')
    allowed=set(previous.plan()['pff_train_ids']);rows=[];labels=[];weights=[];provenance=[];cols=names()
    for number,game in enumerate(cfg['training_games']):
        family,match=game.split('/')
        if family!='pff' or match not in allowed:raise ValueError('Unapproved boundary training game')
        for scenario in ['natural_detected','label_blind_seed7']:
            folder=previous.OUT/'sequences'/('pff__'+match)/scenario
            meta=json.loads((folder/'meta.json').read_text())
            if meta['mixed_plan_sha256']!=sha(previous.PLAN):raise ValueError('Stale training cache')
            for file,digest in meta['feature_sha256'].items():
                if sha(folder/file)!=digest:raise ValueError('Training cache bytes changed')
            gt=np.load(folder/'gt.npy');target=event.targets(gt,meta['segments']);valid=target>=0
            select=valid&((target>0)|(np.arange(len(gt))%10==0));ix=np.flatnonzero(select)
            weight=np.where(target[ix]>0,1.,10.)/max(1,int(valid.sum()))
            for field in ['x','xs']:
                x=np.load(folder/(field+'.npy'),mmap_mode='r')
                extended=event.temporal_features(x,cols,meta['segments'],cfg['delta_features'])
                rows.append(extended[ix]);labels.append(target[ix]);weights.append(weight)
                del extended,x
            provenance.append({'game':game,'scenario':scenario,'valid_pair_count':int(valid.sum()),'selected_rows_before_swap':len(ix),'target_counts':np.bincount(target[valid],minlength=3).tolist(),'sequence_hashes':meta['feature_sha256']})
        if number%6==5:print('Boundary training table',number+1,'of36games loaded',flush=True)
    x=np.concatenate(rows);y=np.concatenate(labels);w=np.concatenate(weights);del rows,labels,weights
    w=w/w.mean();prior=np.bincount(y,weights=w,minlength=3)/w.sum()
    write(OUT/'training_table.json',{'plan_sha256':sha(PLAN),'features':x.shape[1],'rows':len(x),'sampled_counts':np.bincount(y,minlength=3).tolist(),'class_prior':prior.tolist(),'train':cfg['training_games'],'provenance':provenance,'source_sha256':source_hashes()})
    print('Boundary table ready',x.shape,'priors',prior.tolist(),flush=True)
    for seed in cfg['seeds']:
        started=time.perf_counter();model=HistGradientBoostingClassifier(random_state=seed,verbose=1,**cfg['classifier'])
        with threadpool_limits(2):model.fit(x,y,sample_weight=w)
        file=dest/f'seed{seed}.pkl'
        with file.open('wb') as stream:cloudpickle.dump(model,stream)
        write(file.with_suffix('.json'),{'plan_sha256':sha(PLAN),'seed':seed,'train':cfg['training_games'],'class_prior':prior.tolist(),'source_sha256':source_hashes(),'model_sha256':sha(file),'seconds':time.perf_counter()-started,'training_table_sha256':sha(OUT/'training_table.json')})
        print('Boundary seed',seed,'complete',round(time.perf_counter()-started,1),'seconds',flush=True)
    write(dest/'complete.json',{'plan_sha256':sha(PLAN),'seeds':cfg['seeds'],'class_prior':prior.tolist(),'training_table_sha256':sha(OUT/'training_table.json')})


def load():
    cfg=plan();dest=OUT/'models';complete=json.loads((dest/'complete.json').read_text())
    if complete['plan_sha256']!=sha(PLAN):raise ValueError('Registered plan changed')
    models=[]
    for seed in cfg['seeds']:
        file=dest/f'seed{seed}.pkl';meta=json.loads(file.with_suffix('.json').read_text())
        if meta['model_sha256']!=sha(file) or meta['train']!=cfg['training_games'] or meta['source_sha256']!=source_hashes():raise ValueError('Boundary fit lineage/source changed')
        with file.open('rb') as stream:models.append(cloudpickle.load(stream))
    return models,np.array(complete['class_prior'])


def components(data,parts,assets,state,models):
    if 'gt' in data:raise ValueError('Reference is not a predictor input')
    base,team,*_=assets
    x,cols=features2.build(data,parts);xs,_=features2.build(data,parts,swap=True)
    bd,bh=boost.predict(base['stage1'],x,xs);bd,bh=stack.predict(base['stage2'],bd,bh)
    x,columns=context.extend(x,cols,parts);xs,_=context.extend(xs,cols,parts)
    if columns!=names():raise ValueError('Feature schema mismatch')
    bh=.5*bh+.5*context.predict(team,x,xs)
    nd,nh=neural(state,x,xs,parts)
    down=[((a+4)//5,(b+4)//5) for a,b in parts]
    direct=event.temporal_features(x[::5],columns,down,plan()['delta_features'])
    swapped=event.temporal_features(xs[::5],columns,down,plan()['delta_features'])
    mean,individual=event.probabilities(models,direct,swapped)
    def expand(q):
        result=np.zeros((len(x),3));result[:,0]=1
        for a,b in parts:
            lo,hi=(a+4)//5,(b+4)//5
            for k in range(3):result[a:b,k]=np.interp(np.arange(a,b),np.arange(lo,hi)*5,q[lo:hi,k])
        return result
    return .5*bd+.5*nd,.5*bh+.5*nh,expand(mean),[expand(q) for q in individual]


def summarize(records):
    result={}
    for policy in plan()['policies']:
        name=policy['name'];checks={};comparisons={}
        for ref in ['canonical','sdk_shadow']:
            for scenario in ['natural_detected','label_blind_seed7']:
                base={r['game']:r for r in records if r['candidate']=='unchanged' and r['scenario']==scenario and r['reference']==ref}
                current={r['game']:r for r in records if r['candidate']==name and r['scenario']==scenario and r['reference']==ref}
                if set(base)!=set(plan()['pff_dev_ids']) or set(current)!=set(base):raise ValueError('Incomplete comparison')
                gains=[current[g]['spell_f1_mean']-base[g]['spell_f1_mean'] for g in base]
                macro=float(np.mean([current[g]['macro_f1']-base[g]['macro_f1'] for g in base]))
                classes={k:float(np.mean([current[g]['per_class'][k]['f1']-base[g]['per_class'][k]['f1'] for g in base])) for k in ['H','A','D']}
                comp={'baseline':float(np.mean([r['spell_f1_mean'] for r in base.values()])),'candidate':float(np.mean([r['spell_f1_mean'] for r in current.values()])),'gain':float(np.mean(gains)),'wins':int(np.sum(np.array(gains)>0)),'macro_gain':macro,'class_f1_gains':classes,'per_game_gain':dict(zip(base,gains))}
                comparisons[ref+'/'+scenario]=comp
                if ref=='canonical' and scenario=='natural_detected':
                    gate=plan()['pff_dev_gate'];checks.update(gain=comp['gain']>=gate['min_natural_spell_gain_vs_boundary102'],wins=comp['wins']>=gate['min_natural_game_wins'],macro=macro>=-gate['max_macro_loss'],classes=min(classes.values())>=-gate['max_perclass_loss'])
                else:
                    checks[ref+'/'+scenario]=comp['gain']>=-.01
                    checks[ref+'/'+scenario+'_macro']=macro>=-.01
                    checks[ref+'/'+scenario+'_classes']=min(classes.values())>=-.02
        result[name]={'checks':checks,'comparisons':comparisons,'eligible_for_old_audit':policy['kind']=='learned' and all(checks.values())}
    return result


def pffdev():
    dest=OUT/'pff_dev_results.json'
    if dest.exists():raise ValueError('Do not overwrite completed scores')
    cfg=plan();models,prior=load();assets=v4.load();_,state=mixed_candidate.load_bundle(ROOT/'assets/pff-mixed-candidate/bundle.json');records=[]
    historical=json.loads((ROOT/'docs/experiments/pff-mixed/pff_dev_results.json').read_text())['records']
    indexed={(r['game'],r['scenario'],r['reference']):r for r in historical if r['candidate']=='boundary102'}
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    for game in cfg['pff_dev_ids']:
        started=time.perf_counter();data,receipt=geometry(game);parts=spans(data['period']);dummy=np.where(data['period']>0,0,3).astype(np.int8)
        noisy=degrade.degrade(dict(data,gt=dummy),seed=7,params=expanded_team.BLIND);noisy.pop('gt');predictions={}
        for scenario,observed in [('natural_detected',data),('label_blind_seed7',noisy)]:
            with threadpool_limits(1):
                pd,ph,q,individual=components(observed,parts,assets,state,models)
                cache=OUT/'probabilities'/f'{game}__{scenario}.npz';cache.parent.mkdir(exist_ok=True)
                np.savez_compressed(cache,pd=pd,ph=ph,q=q,**{f'seed{seed}':prob for seed,prob in zip(cfg['seeds'],individual)})
                for policy in cfg['policies']:
                    params=decoder()
                    if policy['kind']=='constant_control':
                        params.update({k:policy['cost'] for k in ['cost_ha','cost_in','cost_same','cost_other']})
                    predictions[(scenario,policy['name'])]=event.label(pd,ph,q,prior,parts,len(dummy),params,policy.get('strength',0))
                for seed,prob in zip(cfg['seeds'],individual):
                    predictions[(scenario,f'event1_seed{seed}')]=event.label(pd,ph,prob,prior,parts,len(dummy),decoder(),1)
        reference_file=ROOT/'reports/pff_access/references'/f'kloppy_{game}.npz'
        if sha(reference_file)!=receipt['reference_cache_sha256']:raise ValueError('Reference bytes changed')
        with np.load(reference_file,allow_pickle=False) as z:
            for ref,key in [('canonical','gt'),('sdk_shadow','alternate')]:
                for (scenario,name),pred in predictions.items():
                    r=dict(game=game,scenario=scenario,candidate=name,reference=ref,**metric(z[key],pred))
                    if name=='unchanged' and any(r[k]!=indexed[game,scenario,ref][k] for k in ['spell_f1_mean','macro_f1','per_class']):raise ValueError('Frozen boundary102 control did not reproduce')
                    records.append(r)
        np.savez_compressed(OUT/f'{game}_predictions.npz',**{s+'__'+n:v for (s,n),v in predictions.items()})
        print('Boundary PFFdev',game,'complete',round(time.perf_counter()-started,1),'seconds',flush=True)
    decisions=summarize(records)
    write(dest,{'plan_sha256':sha(PLAN),'records':records,'decisions':decisions,'baseline_control_exactly_reproduced':True,'original_final_used':False})
    for name,r in decisions.items():print(name,json.dumps(r['comparisons']['canonical/natural_detected']),r['eligible_for_old_audit'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['fit','pffdev']);a=p.parse_args();OUT.mkdir(exist_ok=True)
    fit() if a.mode=='fit' else pffdev()
