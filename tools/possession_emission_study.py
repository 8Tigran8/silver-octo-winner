"""Registered training-only duration-weighted geometry ownership experiment."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='2'
import json,time,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import cloudpickle
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from boundary_event_study import names,sha,write
from mtah import boundary_events as event,features2,possession_context as context,degrade,expanded_team
from mtah.methods import boost
from pff_development import geometry,spans
from pff_mixed_score import decoder
from decoder_development import metric
PLAN=ROOT/'docs/experiments/possession-emissions/plan.json';OUT=ROOT/'reports/possession_emissions'

def weights(gt,parts):
    w=np.zeros(len(gt))
    for a,b in parts:
        edges=np.r_[a,a+np.flatnonzero(gt[a+1:b]!=gt[a:b-1])+1,b]
        for lo,hi in zip(edges[:-1],edges[1:]):
            if gt[lo]<3:w[lo:hi]=np.clip(50/(hi-lo),1,10)
    return w

def swapped(y):return np.where(y==0,1,np.where(y==1,0,y))

def blend(pd,ph,q,alpha):
    base=np.column_stack(((1-pd)*ph,(1-pd)*(1-ph),pd));p=(1-alpha)*base+alpha*q
    return p[:,2],p[:,0]/np.maximum(p[:,:2].sum(axis=1),1e-9)

def main():
    cfg=json.loads(PLAN.read_text());OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'model.pkl').exists() or (OUT/'results.json').exists():raise ValueError('Preserve completed study')
    cols=names();delta=json.loads((ROOT/'docs/experiments/boundary-events/plan.json').read_text())['delta_features'];xx=[];yy=[];ww=[];proof=[]
    for game in cfg['training_games']:
        for scenario in ['natural_detected','label_blind_seed7']:
            folder=ROOT/'reports/pff_mixed/sequences'/game.replace('/','__')/scenario;meta=json.loads((folder/'meta.json').read_text())
            for file,digest in meta['feature_sha256'].items():
                if sha(folder/file)!=digest:raise ValueError('Training cache changed')
            gt=np.load(folder/'gt.npy');w=weights(gt,meta['segments']);ix=np.arange(0,len(gt),5);ix=ix[gt[ix]<3];w=w[ix];w=w/w.sum()
            for field,lab in [('x',gt),('xs',swapped(gt))]:
                x=np.load(folder/(field+'.npy'));x=event.temporal_features(x,cols,meta['segments'],delta);xx.append(x[ix]);yy.append(lab[ix]);ww.append(w)
            proof.append({'game':game,'scenario':scenario,'hashes':meta['feature_sha256']})
    x=np.concatenate(xx);y=np.concatenate(yy);w=np.concatenate(ww);w/=w.mean();del xx,yy,ww
    write(OUT/'fit.json',{'plan_sha256':sha(PLAN),'source_sha256':sha(Path(__file__)),'rows':len(y),'features':x.shape[1],'training':proof})
    print('Training',x.shape,flush=True);model=HistGradientBoostingClassifier(random_state=cfg['seed'],**cfg['classifier']);start=time.perf_counter()
    with threadpool_limits(2):model.fit(x,y,sample_weight=w)
    with (OUT/'model.pkl').open('wb') as f:cloudpickle.dump(model,f)
    write(OUT/'model_receipt.json',{'sha256':sha(OUT/'model.pkl'),'seconds':time.perf_counter()-start,'fit_sha256':sha(OUT/'fit.json')});del x,y,w
    print('Fit complete',round(time.perf_counter()-start,1),flush=True)
    old=json.loads((ROOT/'docs/experiments/boundary-events/pff_dev_results.json').read_text())['records'];indexed={(r['game'],r['scenario'],r['reference']):r for r in old if r['candidate']=='unchanged'};records=[]
    for game in cfg['dev']:
        start=time.perf_counter();data,receipt=geometry(game);parts=spans(data['period']);n=len(data['period']);dummy=np.where(data['period']>0,0,3).astype(np.int8)
        noisy=degrade.degrade(dict(data,gt=dummy),seed=7,params=expanded_team.BLIND);noisy.pop('gt');predictions={}
        for scenario,observed in [('natural_detected',data),('label_blind_seed7',noisy)]:
            assert 'gt' not in observed
            qs=[]
            for swap in [False,True]:
                x,c=features2.build(observed,parts,swap=swap);x,c=context.extend(x,c,parts);assert c==cols
                down=[((a+4)//5,(b+4)//5) for a,b in parts];x=event.temporal_features(x[::5],cols,down,delta)
                with threadpool_limits(2):q=model.predict_proba(x)
                qs.append(q[:,[1,0,2]] if swap else q)
            q5=np.mean(qs,axis=0);q=np.zeros((n,3));q[:,2]=1
            for a,b in parts:
                lo,hi=(a+4)//5,(b+4)//5
                for k in range(3):q[a:b,k]=np.interp(np.arange(a,b),np.arange(lo,hi)*5,q5[lo:hi,k])
            with np.load(ROOT/'reports/boundary_events/probabilities'/f'{game}__{scenario}.npz') as z:pd=z['pd'];ph=z['ph']
            for name,alpha in cfg['policies'].items():
                d,h=(pd,ph) if alpha==0 else blend(pd,ph,q,alpha)
                predictions[scenario+'__'+name]=boost.label(d,h,parts,n,decoder())
        ref=ROOT/'reports/pff_access/references'/f'kloppy_{game}.npz';assert sha(ref)==receipt['reference_cache_sha256']
        with np.load(ref) as z:
            for reference,key in [('canonical','gt'),('sdk_shadow','alternate')]:
                for case,pred in predictions.items():
                    scenario,name=case.split('__');r=dict(game=game,scenario=scenario,candidate=name,reference=reference,**metric(z[key],pred))
                    if name=='unchanged':assert all(r[k]==indexed[game,scenario,reference][k] for k in ['spell_f1_mean','macro_f1','per_class'])
                    records.append(r)
        np.savez_compressed(OUT/f'{game}_predictions.npz',**predictions);print('Scored',game,round(time.perf_counter()-start,1),flush=True)
    decisions={}
    for name in cfg['policies']:
        if name=='unchanged':continue
        comparisons={};checks=[]
        for ref in ['canonical','sdk_shadow']:
            for scenario in ['natural_detected','label_blind_seed7']:
                a={r['game']:r for r in records if r['candidate']==name and r['reference']==ref and r['scenario']==scenario};b={r['game']:r for r in records if r['candidate']=='unchanged' and r['reference']==ref and r['scenario']==scenario};assert set(a)==set(b)==set(cfg['dev'])
                gain=np.array([a[g]['spell_f1_mean']-b[g]['spell_f1_mean'] for g in a]);macro=np.mean([a[g]['macro_f1']-b[g]['macro_f1'] for g in a]);cl={k:float(np.mean([a[g]['per_class'][k]['f1']-b[g]['per_class'][k]['f1'] for g in a])) for k in ['H','A','D']}
                comparisons[ref+'/'+scenario]={'score':float(np.mean([r['spell_f1_mean'] for r in a.values()])),'gain':float(gain.mean()),'wins':int((gain>0).sum()),'macro_gain':float(macro),'class_gains':cl,'per_game_gain':dict(zip(a,gain.tolist()))}
                checks.extend([macro>=-.01,min(cl.values())>=-.02,gain.mean()>=(.02 if scenario=='natural_detected' else -.01)])
                if scenario=='natural_detected':checks.append((gain>0).sum()>=8)
        decisions[name]={'eligible':bool(all(checks)),'comparisons':comparisons}
    write(OUT/'results.json',{'plan_sha256':sha(PLAN),'records':records,'decisions':decisions,'baseline_exactly_reproduced':True,'final_bridges_organizer_used':False});print(json.dumps(decisions,indent=2),flush=True)

if __name__=='__main__':main()
