"""Strict old-game folds and PFFdev for the two registered mixed-source models."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cloudpickle
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from mtah import mixed_temporal,temporal,v4,features2,possession_context as context,stack,degrade,expanded_team
from mtah.methods import boost,boost_train
import pff_mixed_train as train
import temporal_trial as trial
import strict_fullfold as strict
from pff_development import geometry,spans,summarize
from pff_reader import sha
from decoder_development import metric
from check_evaluation import evaluate,evaluate_paired_audit

OUT=train.OUT;PLAN=train.PLAN


def decoder():return dict(boost_train.decoder_params(),**json.loads((ROOT/'mtah/methods/v2_params.json').read_text())['decoder'])


def inherited_probabilities(models,x,xs,cols,parts):
    bd,bh=boost.predict(models['stage1'],x,xs);bd,bh=stack.predict(models['stage2'],bd,bh)
    x,columns=context.extend(x,cols,parts);xs,_=context.extend(xs,cols,parts)
    bh=.5*bh+.5*context.predict(models['team'],x,xs)
    return bd,bh,x,xs


def neural(state,x,xs,parts):
    model,mean,scale=state;down=[((a+4)//5,(b+4)//5) for a,b in parts]
    nd,nh=mixed_temporal.probabilities(model,x[::5],xs[::5],down,mean,scale)
    return v4.neural_upsample(nd,nh,parts,len(x))


def old_games():
    dest=OUT/'old_results.json'
    if dest.exists():raise ValueError('Do not overwrite completed old-game scores')
    strict.attest_periods();cfg=train.plan();names=[s['name'] for s in cfg['candidates']]
    control_payload=json.loads((ROOT/'reports/strict_fullfold/results.json').read_text())
    records=[r for r in control_payload['records'] if r['candidate']=='strict_v4'];proofs={name:{} for name in names}
    old_proof=json.loads((ROOT/'reports/strict_fullfold/provenance.json').read_text())
    params=decoder();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    for fold in range(5):
        # Local dependency wait lets scoring overlap later GPU fits. No model
        # is selected or skipped, and root continues monitoring both jobs.
        for name in names:
            pending=OUT/'models'/(name+'_fold'+str(fold))/'meta.json';waiting=time.monotonic()
            while True:
                try:
                    if pending.exists():json.loads(pending.read_text());break
                except json.JSONDecodeError:pass
                if time.monotonic()-waiting>1800:raise TimeoutError('Registered fit did not finish: '+str(pending))
                time.sleep(2)
        folder=ROOT/'reports/strict_fullfold/folds'/('fold'+str(fold));inherited=json.loads((folder/'meta.json').read_text());_,test=trial.game_split('fold'+str(fold))
        prior_proof=old_proof['fold'+str(fold)]
        if sha(folder/'models.pkl')!=inherited['model_sha256'] or inherited['plan_sha256']!=prior_proof['audit_plan_sha256'] or any(v!=prior_proof.get(k) for k,v in inherited.items() if k!='plan_sha256'):
            raise ValueError('Audited inherited stage artifact/provenance changed')
        strict.assert_component_isolation(inherited,test)
        with (folder/'models.pkl').open('rb') as stream:models=cloudpickle.load(stream)
        states={}
        for name in names:
            states[name],meta=train.load(name,fold);proofs[name]['fold'+str(fold)]=dict(meta,inherited=inherited,complete_transitive_outer_isolation=True)
        for game in test:
            started=time.perf_counter();family,match=game.split('/');scenarios=['matched_seed0','label_blind_seed1','stress_seed1'] if family=='original' else ['natural_detected','label_blind_seed7']
            for scenario in scenarios:
                with threadpool_limits(1):
                    x,xs,cols,gt,parts=strict.game_features(game,scenario)
                    bd,bh,x,xs=inherited_probabilities(models,x,xs,cols,parts)
                    for name,state in states.items():
                        nd,nh=neural(state,x,xs,parts);pred=boost.label(.5*bd+.5*nd,.5*bh+.5*nh,parts,len(gt),params)
                        records.append({'candidate':name,'game':game,'match':match,'family':family,'fold':fold,'scenario':scenario,**metric(gt,pred)})
            print('old-fold',game,'complete',round(time.perf_counter()-started,1),'seconds',flush=True)
    payload={'plan_sha256':sha(trial.PLAN),'mixed_plan_sha256':sha(PLAN),'strict_control_result_sha256':sha(ROOT/'reports/strict_fullfold/results.json'),'records':records};decisions={}
    for name in names:
        historical=evaluate(payload,name,proofs[name]);paired=evaluate_paired_audit(payload,name,'strict_v4',proofs[name])
        decisions[name]={'historical':historical,'paired':paired};print(name,'old historical',historical['status'],'paired',paired['status'],flush=True)
    (OUT/'old_provenance.json').write_text(json.dumps(proofs,indent=2)+'\n');payload['decisions']=decisions;dest.write_text(json.dumps(payload,indent=2)+'\n')


def pff_development():
    dest=OUT/'pff_dev_results.json'
    if dest.exists():raise ValueError('Do not overwrite completed PFFdev research')
    cfg=train.plan();names=[s['name'] for s in cfg['candidates']];states={name:train.load(name,'full')[0] for name in names};assets=v4.load();params=decoder();records=[]
    frozen=json.loads((ROOT/'reports/pff_development/results.json').read_text());controls=[r for r in frozen['records'] if r['candidate']=='v4']
    baseline={(r['game'],r['scenario'],r['reference']):r for r in controls}
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    for game in cfg['pff_dev_ids']:
        started=time.perf_counter();data,receipt=geometry(game);parts=spans(data['period']);dummy=np.where(data['period']>0,0,3).astype(np.int8)
        noisy=degrade.degrade(dict(data,gt=dummy),seed=7,params=expanded_team.BLIND);noisy.pop('gt');predictions={}
        for scenario,observed in [('natural_detected',data),('label_blind_seed7',noisy)]:
            with threadpool_limits(1):
                bd,bh,od,oh=v4.components(observed,parts,assets)
                # Recompute the complete frozen control on every new-source cell.
                predictions[scenario+'__v4']=boost.label(.5*bd+.5*od,.5*bh+.5*oh,parts,len(dummy),params)
                x,cols=features2.build(observed,parts);xs,_=features2.build(observed,parts,swap=True)
                x,_=context.extend(x,cols,parts);xs,_=context.extend(xs,cols,parts)
                for name,state in states.items():
                    nd,nh=neural(state,x,xs,parts);predictions[scenario+'__'+name]=boost.label(.5*bd+.5*nd,.5*bh+.5*nh,parts,len(dummy),params)
        pred_file=OUT/('pffdev_'+game+'_predictions.npz');np.savez_compressed(pred_file,**predictions)
        refs=ROOT/'reports/pff_access/references'/('kloppy_'+game+'.npz')
        if sha(refs)!=receipt['reference_cache_sha256']:raise ValueError('Source/reference changed')
        with np.load(refs,allow_pickle=False) as z:
            for reference,key in [('canonical','gt'),('sdk_shadow','alternate')]:
                for value,pred in predictions.items():
                    scenario,name=value.split('__');r={'game':game,'scenario':scenario,'candidate':name,'reference':reference,**metric(z[key],pred)}
                    if name=='v4' and any(r[k]!=baseline[game,scenario,reference][k] for k in ['spell_f1_mean','macro_f1','per_class']):raise ValueError('Immutable v4 PFFcontrol changed')
                    records.append(r)
        print('PFFdev',game,'complete',round(time.perf_counter()-started,1),'seconds',flush=True)
    decisions={}
    for name in names:
        # summarize uses the registered former candidate's name solely as a
        # neutral table key; no prior checkpoint is used for new predictions.
        rows=[dict(r,candidate='source_seed20261113' if r['candidate']==name else r['candidate']) for r in records if r['candidate'] in ['v4',name]]
        decisions[name]=summarize(rows,cfg['pff_dev_ids']);print(name,json.dumps(decisions[name]),flush=True)
    dest.write_text(json.dumps({'mixed_plan_sha256':sha(PLAN),'original_control_sha256':sha(ROOT/'reports/pff_development/results.json'),'records':records,'decisions':decisions},indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('cohort',choices=['old','pffdev']);a=p.parse_args();OUT.mkdir(exist_ok=True)
    old_games() if a.cohort=='old' else pff_development()
