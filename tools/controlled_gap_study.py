"""Preregistered player-relative missing-ball experiment; development only."""
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
from mtah import features2,mixed_candidate,v4,degrade,expanded_team
from mtah.methods import boost
from pff_mixed_score import decoder
from pff_development import geometry,spans
from decoder_development import metric


def fill_controlled_gaps(data, periods, max_seconds):
    ball=data['ball'].copy();players=data['players'];pitch=np.asarray(data.get('pitch',(105.3,68.)))
    filled=0;gaps=0
    for start,end in periods:
        seen=np.isfinite(ball[start:end,:2]).all(axis=1)
        for aa,bb in zip(*features2._runs(~seen)):
            a,b=start+aa,start+bb
            if a==start or b==end or b-a>int(25*max_seconds):continue
            endpoints=[a-1,b]
            distance=np.linalg.norm((players[endpoints,:,:2]-ball[endpoints,None,:2])*pitch,axis=2)
            distance=np.where(np.isfinite(distance),distance,np.inf)
            near=np.argmin(distance,axis=1)
            if near[0]!=near[1] or (distance[np.arange(2),near]>2).any():continue
            xy=players[a-1:b+1,near[0],:2]
            if not np.isfinite(xy).all():continue
            velocity=np.linalg.norm(np.diff(xy,axis=0)*pitch,axis=1)*25
            if (velocity>12).any():continue
            offsets=ball[endpoints,:2]-xy[[0,-1]]
            alpha=np.arange(1,b-a+1)/(b-a+1)
            ball[a:b,:2]=xy[1:-1]+(1-alpha[:,None])*offsets[0]+alpha[:,None]*offsets[1]
            filled+=b-a;gaps+=1
    return dict(data,ball=ball),dict(filled_frames=int(filled),filled_gaps=int(gaps))


def main():
    plan_path=ROOT/'docs/experiments/controlled-gap/plan.json';plan=json.loads(plan_path.read_text())
    baseline_path=ROOT/'reports/independent_audit/observation_regression.json'
    baseline=json.loads(baseline_path.read_text())
    base={(r['game'],r['scenario'],r['reference']):r for r in baseline['records']}
    if set(plan['development_games'])!={k[0] for k in base}:raise ValueError('Development registry mismatch')
    out=ROOT/'reports/controlled_gap';out.mkdir(exist_ok=False)
    assets=v4.load();_,state=mixed_candidate.load_bundle(ROOT/'assets/pff-mixed-candidate/bundle.json')
    rows=[]
    for game in plan['development_games']:
        start=time.monotonic();data,_=geometry(game);parts=spans(data['period'])
        dummy=np.where(data['period']>0,0,3).astype(np.int8)
        blind=degrade.degrade(dict(data,gt=dummy),seed=7,params=expanded_team.BLIND);blind.pop('gt')
        with np.load(ROOT/'reports/pff_access/references'/f'kloppy_{game}.npz') as archive:
            refs={name:archive[key] for name,key in [('canonical','gt'),('sdk_shadow','alternate')]}
        for scenario,d in [('natural_detected',data),('label_blind_seed7',blind)]:
            for seconds in plan['candidates_seconds']:
                repaired,counts=fill_controlled_gaps(d,parts,seconds)
                pd,ph=mixed_candidate.probabilities(repaired,parts,assets,state)
                pred=boost.label(pd,ph,parts,len(dummy),decoder())
                for reference,gt in refs.items():
                    score=metric(gt,pred);previous=base[game,scenario,reference]
                    rows.append(dict(game=game,scenario=scenario,reference=reference,candidate_seconds=seconds,**counts,**score,
                                     spell_gain=score['spell_f1_mean']-previous['spell_f1_mean'],macro_gain=score['macro_f1']-previous['macro_f1']))
        (out/'progress.json').write_text(json.dumps(rows,indent=2)+'\n')
        print(game,round(time.monotonic()-start,1),flush=True)
    summaries=[];gate=plan['promotion_gate_each_scenario']
    for seconds in plan['candidates_seconds']:
        for scenario in plan['scenarios']:
            rr=[r for r in rows if r['candidate_seconds']==seconds and r['scenario']==scenario and r['reference']=='canonical']
            s=dict(candidate_seconds=seconds,scenario=scenario,mean_spell_gain=float(np.mean([r['spell_gain'] for r in rr])),
                   mean_macro_gain=float(np.mean([r['macro_gain'] for r in rr])),positive_games=sum(r['spell_gain']>0 for r in rr),
                   worst_spell_gain=min(r['spell_gain'] for r in rr),filled_frames=sum(r['filled_frames'] for r in rr),
                   spell_f1_mean=float(np.mean([r['spell_f1_mean'] for r in rr])))
            s['gate_pass']=s['mean_spell_gain']>=gate['mean_spell_gain_at_least'] and s['mean_macro_gain']>=gate['mean_macro_gain_at_least'] and s['positive_games']>=gate['positive_games_at_least'] and s['worst_spell_gain']>=gate['worst_game_spell_gain_at_least']
            summaries.append(s)
    result=dict(plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),baseline_sha256=hashlib.sha256(baseline_path.read_bytes()).hexdigest(),records=rows,summary=summaries)
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(summaries,indent=2),flush=True)


if __name__=='__main__':main()
