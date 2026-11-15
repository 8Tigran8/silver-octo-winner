"""One frozen candidate versus v4 on all ten registered PFF development games.

This tool cannot download, convert or score final/bridge games. Model inputs
contain geometry only; references are opened only after predictions are saved.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from mtah import v4,research_candidate,features2,possession_context as context,temporal,degrade,expanded_team
from mtah.methods import boost,boost_train
from decoder_development import metric
from pff_reader import sha

OUT=ROOT/'reports/pff_development'
BUNDLE=ROOT/'assets/research-balanced/bundle.json'


def spans(period):
    """All independently recorded periods, including extra time."""
    result=[]
    for value in (1,2,3,4):
        edges=np.diff(np.r_[0,(period==value).astype(np.int8),0])
        result.extend(zip(np.flatnonzero(edges==1).tolist(),np.flatnonzero(edges==-1).tolist()))
    return sorted(result)


def paired(data,segments,assets,states):
    if 'gt' in data:raise ValueError('Reference labels are prohibited from inference')
    bd,bh,od,oh=v4.components(data,segments,assets)
    with threadpool_limits(1):
        x,cols=features2.build(data,segments);xs,_=features2.build(data,segments,swap=True)
        x,_=context.extend(x,cols,segments);xs,_=context.extend(xs,cols,segments)
        parts=[((a+4)//5,(b+4)//5) for a,b in segments];pairs=[]
        for model,mean,scale in states:
            nd,nh=temporal.probabilities(model,x[::5],xs[::5],parts,mean,scale,device='cpu')
            pairs.append(v4.neural_upsample(nd,nh,segments,len(x)))
        nd=np.mean([p[0] for p in pairs],axis=0);nh=np.mean([p[1] for p in pairs],axis=0)
    return (.5*bd+.5*od,.5*bh+.5*oh),(.5*bd+.5*nd,.5*bh+.5*nh)


def geometry(game):
    path=ROOT/'external/pff/converted'/(game+'_geometry.npz')
    receipt=json.loads((ROOT/'reports/pff_access'/('kloppy_'+game+'_preflight.json')).read_text())
    if receipt['status']!='PASS' or sha(path)!=receipt['geometry_cache_sha256']:
        raise ValueError('Source preflight or geometry hash failed')
    if receipt['reader_sha256']!=sha(ROOT/'tools/pff_reader.py'):raise ValueError('Reader changed after source preflight')
    with np.load(path,allow_pickle=False) as z:data={k:z[k] for k in z.files}
    if 'gt' in data:raise ValueError('Label-contaminated geometry cache')
    return data,receipt


def summarize(rows,games):
    indexed={(r['game'],r['scenario'],r['candidate'],r['reference']):r for r in rows}
    if len(indexed)!=len(rows) or len(rows)!=len(games)*8:raise ValueError('Incomplete/duplicate score grid')
    result={};checks={}
    for reference in ['canonical','sdk_shadow']:
        for scenario in ['natural_detected','label_blind_seed7']:
            base=[indexed[g,scenario,'v4',reference] for g in games]
            current=[indexed[g,scenario,'source_seed20261113',reference] for g in games]
            gains=np.array([c['spell_f1_mean']-b['spell_f1_mean'] for b,c in zip(base,current)])
            delta=float(gains.mean());macro=float(np.mean([c['macro_f1']-b['macro_f1'] for b,c in zip(base,current)]))
            classes={k:float(np.mean([c['per_class'][k]['f1']-b['per_class'][k]['f1'] for b,c in zip(base,current)])) for k in ['H','A','D']}
            result[reference+'/'+scenario]={'baseline':float(np.mean([r['spell_f1_mean'] for r in base])),
                'candidate':float(np.mean([r['spell_f1_mean'] for r in current])),
                'gain':delta,'wins':int((gains>0).sum()),'macro_gain':macro,'class_f1_gains':classes,
                'per_game_gain':dict(zip(games,map(float,gains)))}
            if reference=='canonical':
                if scenario=='natural_detected':
                    checks.update(natural_gain=delta>=.02,game_wins=int((gains>0).sum())>=7,
                                  macro_guard=macro>=-.01,**{k+'_guard':v>=-.02 for k,v in classes.items()})
                else:checks['blind_guard']=delta>=-.01
            else:checks['shadow_'+scenario]=delta>=-.01
    return {'status':'PASS: PFF development only; final unopened' if all(checks.values()) else 'FAIL: do not open final reserve',
            'checks':checks,'comparisons':result,'final_reserve_opened':False}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--equivalence',action='store_true');a=p.parse_args()
    OUT.mkdir(exist_ok=True);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    bundle,states=research_candidate.load_bundle(BUNDLE);assets=v4.load()
    if bundle['candidate']!='source_seed20261113' or bundle['rule']!={'name':'uniform50','kind':'fixed','dead':.5,'home':.5} or bundle.get('temporal_blend'):
        raise ValueError('Only the frozen registered candidate is allowed')
    if a.equivalence:
        dest=OUT/'equivalence_train3816.json'
        if dest.exists():raise ValueError('Preserve completed equivalence check')
        data,_=geometry('3816');segments=spans(data['period']);b,c=paired(data,segments,assets,states)
        rb=v4.probabilities(data,segments,assets);rc=research_candidate.probabilities(data,segments,assets,states,bundle['rule'])
        errors={'v4':max(float(np.max(np.abs(x-y))) for x,y in zip(b,rb)),
                'candidate':max(float(np.max(np.abs(x-y))) for x,y in zip(c,rc))}
        if max(errors.values())!=0:raise ValueError('Paired inference differs from frozen entrypoints')
        dest.write_text(json.dumps({'status':'PASS','max_absolute_probability_error':errors,'game':'training3816','no_quality_scores':True},indent=2)+'\n');print(errors,flush=True);return
    equivalence=json.loads((OUT/'equivalence_train3816.json').read_text())
    if equivalence['status']!='PASS':raise ValueError('Unverified paired inference')
    protocol=json.loads((ROOT/'docs/experiments/more-data/pff_protocol.json').read_text());games=[g['game_id'] for g in protocol['development']]
    if len(games)!=10 or set(games)&{g['game_id'] for g in protocol['final_test']}:raise ValueError('Original split invalid')
    dest=OUT/'results.json'
    if dest.exists():raise ValueError('Do not overwrite completed PFF development scores')
    params=dict(boost_train.decoder_params(),**json.loads((ROOT/'mtah/methods/v2_params.json').read_text())['decoder']);records=[]
    for game in games:
        start=time.perf_counter();file=OUT/(game+'_predictions.npz');data,receipt=geometry(game);segments=spans(data['period'])
        if file.exists():raise ValueError('Completed predictions exist; preserve before diagnosing interruption')
        dummy=np.where(data['period']>0,0,3).astype(np.int8);corrupt=degrade.degrade(dict(data,gt=dummy),seed=7,params=expanded_team.BLIND);corrupt.pop('gt')
        predictions={}
        for scenario,observed in [('natural_detected',data),('label_blind_seed7',corrupt)]:
            control,candidate=paired(observed,segments,assets,states)
            for name,probs in [('v4',control),('source_seed20261113',candidate)]:
                predictions[scenario+'__'+name]=boost.label(*probs,segments,len(dummy),params)
        np.savez_compressed(file,**predictions)
        refs=ROOT/'reports/pff_access/references'/('kloppy_'+game+'.npz')
        if sha(refs)!=receipt['reference_cache_sha256']:raise ValueError('Reference hash changed')
        with np.load(refs,allow_pickle=False) as z:
            for reference,key in [('canonical','gt'),('sdk_shadow','alternate')]:
                gt=z[key]
                for name,pred in predictions.items():
                    scenario,candidate=name.split('__');records.append({'game':game,'scenario':scenario,'candidate':candidate,'reference':reference,**metric(gt,pred)})
        (OUT/(game+'_scores.json')).write_text(json.dumps({'records':records[-8:],'prediction_sha256':sha(file),'preflight':receipt},indent=2)+'\n')
        print(game,'complete',round(time.perf_counter()-start,1),'seconds',flush=True)
    result=summarize(records,games)
    dest.write_text(json.dumps({'scope':'All10 registered development games; no PFF fitting or selection; final unopened','bundle_sha256':sha(BUNDLE),'plan_sha256':sha(ROOT/'docs/experiments/pff/plan_v2.json'),'tool_sha256':sha(__file__),'records':records,'decision':result},indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
