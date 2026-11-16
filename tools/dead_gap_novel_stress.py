"""Frozen diagnostic on two new label-blind corruptions; cannot override original gate."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
import json,sys,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
import dead_gap_focus_study as fit
from pff_development import geometry,spans
from pff_reader import sha
from mtah import v4,mixed_temporal,expanded_team,degrade
from mtah.recovery_features import activity_features
from mtah.methods import boost
from activity_temporal_model import probabilities as activity_probabilities
from decoder_development import metric
from short_gap_study import diagnostics
DOC=fit.DOC;OUT=fit.OUT/'novel_stress'

def plan():return json.loads((DOC/'stress_plan.json').read_text())
def check():
    fit.check()
    for p,h in json.loads((DOC/'stress_freeze.json').read_text())['files_sha256'].items():assert sha(ROOT/p)==h,p

def worker(game):
    fit.study.guard(game,'development');data,receipt=geometry(game);parts=spans(data['period']);small=[((a+4)//5,(b+4)//5) for a,b in parts];started=time.monotonic();rows=[]
    ref=ROOT/f'reports/pff_access/references/kloppy_{game}.npz';assert sha(ref)==receipt['reference_cache_sha256']
    for base_seed in plan()['base_seeds']:
        seed=base_seed^int(game);scenario='novel_'+str(base_seed);dummy=np.where(data['period']>0,0,3).astype(np.int8);d=degrade.degrade(dict(data,gt=dummy),seed=seed,params=expanded_team.BLIND);d.pop('gt')
        bd,bh,_,_=v4.components(d,parts,fit.study.STATE[0])
        with threadpool_limits(1):
            x,xs,_=expanded_team.feature_pair(d,parts);activity,names=activity_features(d,parts);assert names==fit.study.STATE[3];old_activity=fit.study.STATE[2].predict_proba(activity[::5])[:,1]
        model,mean,scale=fit.study.STATE[1];nd,nh=mixed_temporal.probabilities(model,x[::5],xs[::5],small,mean,scale);nd,nh=v4.neural_upsample(nd,nh,parts,len(x));old_activity=v4.neural_upsample(old_activity,old_activity,parts,len(x))[0]
        newd,_=mixed_temporal.probabilities(fit.MODELS['uniform'],x[::5],xs[::5],small,mean,scale);newd=v4.neural_upsample(newd,newd,parts,len(x))[0]
        _,newh=mixed_temporal.probabilities(fit.study.MODELS['focus'],x[::5],xs[::5],small,mean,scale);newh=v4.neural_upsample(newh,newh,parts,len(x))[0]
        m,am,asc=fit.ACTIVITY;act=activity_probabilities(m,activity[::5],small,am,asc);act=v4.neural_upsample(act,act,parts,len(x))[0]
        seen=np.isfinite(d['ball'][:,:2]).all(axis=1);pd=.5*bd+.5*nd;ph=.5*bh+.5*nh;pd[~seen]=.75*nd[~seen]+.25*old_activity[~seen];cd=pd.copy();ch=ph.copy();cd[~seen]=.75*newd[~seen]+.25*act[~seen];ch[~seen]=newh[~seen]
        predictions={'control':boost.label(pd,ph,parts,len(x),fit.study.BUNDLE['decoder']),'uniform_temporal_focus':boost.label(cd,ch,parts,len(x),fit.study.BUNDLE['decoder'])};np.savez_compressed(OUT/f'{game}_{scenario}.npz',seen=seen,**predictions)
        with np.load(ref) as z:
            for reference,key in [('canonical','gt'),('sdk_shadow','alternate')]:
                gt=z[key]
                for name,pred in predictions.items():
                    diag=diagnostics(gt,pred,seen);mask=(gt<2)&~seen;diag['missing_wrong_team']=int((mask&(pred<2)&(pred!=gt)).sum());rows.append(dict(game=game,scenario=scenario,reference=reference,seed=seed,candidate=name,**metric(gt,pred),diagnostics=diag))
    return game,rows,time.monotonic()-started

def main():
    check();OUT.mkdir(exist_ok=False);rows=[]
    with ProcessPoolExecutor(max_workers=2,mp_context=get_context('spawn'),initializer=fit.initialize_score) as pool:
        for f in as_completed([pool.submit(worker,g) for g in fit.study.plan()['development_games']]):
            g,r,t=f.result();rows+=r;print('New stress',g,round(t,1),flush=True)
    path=DOC/'stress_scores.json';assert not path.exists();path.write_text(json.dumps(dict(records=rows),indent=2)+'\n');summary=[]
    for scenario in ['novel_23','novel_101']:
        for ref in ['canonical','sdk_shadow']:
            rr=[r for r in rows if r['scenario']==scenario and r['reference']==ref];idx={(r['game'],r['candidate']):r for r in rr};assert len(rr)==40//2
            base=[idx[g,'control'] for g in fit.study.plan()['development_games']];cand=[idx[g,'uniform_temporal_focus'] for g in fit.study.plan()['development_games']];gain=np.array([c['spell_f1_mean']-b['spell_f1_mean'] for c,b in zip(cand,base)])
            counts={k:sum(c['diagnostics'][k] for c in cand) for k in cand[0]['diagnostics']};bc={k:sum(c['diagnostics'][k] for c in base) for k in base[0]['diagnostics']}
            macro=float(np.mean([c['macro_f1']-b['macro_f1'] for c,b in zip(cand,base)]));df=float(np.mean([c['per_class']['D']['f1']-b['per_class']['D']['f1'] for c,b in zip(cand,base)]));fl=counts['missing_false_live']/counts['missing_dead']-bc['missing_false_live']/bc['missing_dead']
            gates=dict(mean_gain=float(gain.mean())>=.02,wins=int((gain>0).sum())>=8,worst=float(gain.min())>=-.02,macro=macro>=-.002,dead_f1=df>=-.002,missing_false_live=fl<=.01,short_recall=counts['short_matched']/counts['short_reference']>=bc['short_matched']/bc['short_reference'],missing_wrong_team=counts['missing_wrong_team']/counts['missing_live']<=bc['missing_wrong_team']/bc['missing_live'])
            summary.append(dict(scenario=scenario,reference=ref,baseline_spell=float(np.mean([b['spell_f1_mean'] for b in base])),candidate_spell=float(np.mean([c['spell_f1_mean'] for c in cand])),gain=float(gain.mean()),wins=int((gain>0).sum()),worst=float(gain.min()),macro_gain=macro,dead_f1_gain=df,counts=counts,control_counts=bc,gates=gates,pass_all=all(gates.values()),per_game_gain={g:float(d) for g,d in zip(fit.study.plan()['development_games'],gain)}))
    (DOC/'stress_result.json').write_text(json.dumps(dict(summary=summary,decision='Diagnostic only; original registration failed, so no automatic adoption regardless of this outcome',independent_matches=False,original18read=False),indent=2)+'\n');check()
    for r in summary:
        if r['reference']=='canonical':print(r,flush=True)
if __name__=='__main__':main()
