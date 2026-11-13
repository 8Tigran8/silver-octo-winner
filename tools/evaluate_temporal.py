"""Expanded whole-game/provider evaluation of the fixed temporal model."""
import argparse,json,sys,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from temporal_trial import OUT,PLAN,AUDIT,plan,folder,stem,game_split,temporal,context,expanded,read_feature,read_match,degrade,sha
from quality_audit import official
from decoder_development import metric
from mtah import features2
from mtah.methods import boost,boost_train
from expanded_team_trial import load_model as v3_team
from threadpoolctl import threadpool_limits
import numpy as np
import torch
OLD=ROOT/'reports/v2_search';PRIOR=ROOT/'reports/v4_search'


def model(key,device):
    dest=OUT/'models'/key;meta=json.loads((dest/'meta.json').read_text());train,test=game_split(key)
    if meta['train']!=train or meta['excluded']!=test or meta['plan_sha256']!=sha(PLAN) or meta['model_sha256']!=sha(dest/'model.pt'):raise ValueError('Model split/hash mismatch')
    checkpoint=torch.load(dest/'model.pt',map_location='cpu',weights_only=True);m=temporal.Temporal();m.load_state_dict(checkpoint['state_dict']);m.eval().to(device)
    return m,checkpoint['mean'].numpy(),checkpoint['scale'].numpy()


def inputs(game,scenario):
    f=folder(game,scenario)
    if not (f/'meta.json').exists():
        family,name=game.split('/')
        if family!='original' or scenario!='stress_seed1':raise ValueError('Unexpected evaluation augmentation')
        base=json.loads((folder(game,'matched_seed0')/'meta.json').read_text());features=read_feature(AUDIT,name,scenario)
        x,cols=context.extend(features['X'],features['cols'].tolist(),base['full_segments']);xs,_=context.extend(features['Xs'],features['cols'].tolist(),base['full_segments'])
        f.mkdir(parents=True,exist_ok=True);np.save(f/'x.npy',x[::5].astype(np.float32));np.save(f/'xs.npy',xs[::5].astype(np.float32));np.save(f/'gt.npy',features['gt'][::5]);(f/'meta.json').write_text(json.dumps(dict(base,scenario=scenario),indent=2)+'\n')
    meta=json.loads((f/'meta.json').read_text())
    return np.load(f/'x.npy',mmap_mode='r'),np.load(f/'xs.npy',mmap_mode='r'),meta


def baseline(game,scenario,oldhead):
    family,name=game.split('/')
    with np.load(OLD/('known' if family=='original' else 'skillcorner')/f'{name}__{scenario}__probs.npz') as z:pd,ph,gt=z['pd'],z['ph'],z['gt']
    oldpath=PRIOR/'probabilities'/f'{name}__{scenario}.npz'
    if oldpath.exists():
        with np.load(oldpath) as z:team=z['ph']
    else:
        path=OUT/'v3_probabilities'/f'{name}__{scenario}.npz';path.parent.mkdir(exist_ok=True)
        if path.exists():
            with np.load(path) as z:team=z['ph']
        else:
            clean=read_match(ROOT/'external/skillcorner'/name,name);data=clean if scenario=='natural_detected' else degrade.degrade(clean,seed=7,params=expanded.BLIND)
            from legacy_skillcorner_reader import period_spans
            x,xs,cols=expanded.feature_pair(data,period_spans(clean['period']));team=context.predict(oldhead,x,xs);np.savez(path,ph=team)
    return pd,.5*ph+.5*team,gt


def upsample(pd,ph,meta):
    nd=np.full(meta['full_length'],.5);nh=nd.copy()
    for (a,b),(lo,hi) in zip(meta['full_segments'],meta['segments']):
        times=np.arange(lo,hi)*5;full=np.arange(a,b);nd[a:b]=np.interp(full,times,pd[lo:hi]);nh[a:b]=np.interp(full,times,ph[lo:hi])
    return nd,nh


def neural(m,x,xs,meta,mean,scale,device,key,game,scenario):
    dest=OUT/'predictions'/key/f'{stem(game)}__{scenario}.npz';dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():
        with np.load(dest) as z:return z['pd'],z['ph']
    pd,ph=temporal.probabilities(m,x,xs,meta['segments'],mean,scale,device=device);pd,ph=upsample(pd,ph,meta);np.savez(dest,pd=pd,ph=ph);return pd,ph


def evaluate(device):
    destination=OUT/'cv_results.json'
    if destination.exists():raise ValueError('Do not overwrite completed CV scores')
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);oldhead=v3_team();params=dict(boost_train.decoder_params(),**json.loads((ROOT/'mtah/methods/v2_params.json').read_text())['decoder']);records=[]
    expected={}
    for file in ['development_results.json','secondary_results.json']:
        for r in json.loads((PRIOR/file).read_text())['records']:
            if r['candidate']=='expanded50':expected[(r.get('game','skillcorner/'+r['match']),r['scenario'])]=r['spell_f1_mean']
    with threadpool_limits(1):
        for k in range(5):
            key='fold'+str(k);m,mean,scale=model(key,device);_,test=game_split(key)
            for game in test:
                family,name=game.split('/');scenarios=['matched_seed0','label_blind_seed1','stress_seed1'] if family=='original' else ['natural_detected','label_blind_seed7']
                for scenario in scenarios:
                    x,xs,meta=inputs(game,scenario);vd,vh,gt=baseline(game,scenario,oldhead);nd,nh=neural(m,x,xs,meta,mean,scale,device,key,game,scenario)
                    for candidate,pd,ph in [('v3',vd,vh),('tcn',nd,nh),('fusion50',.5*vd+.5*nd,.5*vh+.5*nh)]:
                        pred=boost.label(pd,ph,meta['full_segments'],len(gt),params);result=metric(gt,pred)
                        if candidate=='v3' and abs(result['spell_f1_mean']-expected[game,scenario])>1e-12:raise ValueError('V3 control changed')
                        records.append({'game':game,'match':name,'fold':k,'family':family,'scenario':scenario,'candidate':candidate,**result})
                print(game,'fold',k,'OOF PASS',flush=True)
            del m
    destination.write_text(json.dumps({'plan_sha256':sha(PLAN),'v3_exact_control':True,'records':records},indent=2)+'\n')


def diagnostic(device):
    dest=OUT/'generalization_results.json'
    if dest.exists():raise ValueError('Do not overwrite completed diagnostics')
    torch.set_num_threads(2);params=dict(boost_train.decoder_params(),**json.loads((ROOT/'mtah/methods/v2_params.json').read_text())['decoder']);oldhead=v3_team();rows=[]
    with threadpool_limits(1):
        for key in ['full','provider_dfl','provider_skill']:
            m,mean,scale=model(key,device);train,test=game_split(key);games=train if key=='full' else test
            for game in games:
                family,name=game.split('/');scenario='matched_seed0' if family=='original' else 'natural_detected';x,xs,meta=inputs(game,scenario);vd,vh,gt=baseline(game,scenario,oldhead);nd,nh=neural(m,x,xs,meta,mean,scale,device,key,game,scenario)
                pred=boost.label(nd,nh,meta['full_segments'],len(gt),params);rows.append({'game':game,'source_holdout':key,'in_sample':key=='full','scenario':scenario,**metric(gt,pred)})
                print(key,game,'diagnostic PASS',flush=True)
            del m
    dest.write_text(json.dumps({'plan_sha256':sha(PLAN),'records':rows},indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['cv','diagnostic']);parser.add_argument('--device',default='mps');args=parser.parse_args()
    if args.mode=='cv':evaluate(args.device)
    else:diagnostic(args.device)
