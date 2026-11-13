"""Bounded, registered decoding study on reused game-excluded predictions."""
import argparse,json,statistics
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from more_data_trial import FOLDER,PRIOR,OLDER,AUDIT,plan,load,sha,context,read_feature,official,metric,boost,boost_train
import numpy as np
from threadpoolctl import threadpool_limits
PROTOCOL=ROOT/'docs/experiments/more-data/decoder_plan.json'


def temperature(p,t):
    if t==1.:return p
    if t<=0 or not np.isfinite(p).all():raise ValueError('Invalid probability temperature input')
    p=np.clip(p,1e-7,1-1e-7);logit=np.log(p)-np.log1p(-p)
    return 1/(1+np.exp(-logit/t))


def job(game):
    family,name=game.split('/');cohort='known' if family=='original' else 'development'
    scenarios=['matched_seed0','label_blind_seed1','stress_seed1'] if family=='original' else ['natural_detected','label_blind_seed7']
    specs=json.loads(PROTOCOL.read_text())['candidates'];model=load(game);records=[]
    params=dict(boost_train.decoder_params(),**json.loads((ROOT/'mtah/methods/v2_params.json').read_text())['decoder'])
    with threadpool_limits(1):
        for scenario in scenarios:
            with np.load(OLDER/('known' if family=='original' else 'skillcorner')/f'{name}__{scenario}__probs.npz') as z:pd,ph,gt=z['pd'],z['ph'],z['gt']
            segments=official.valid_runs(gt);path=FOLDER/'probabilities'/f'{name}__{scenario}.npz';path.parent.mkdir(exist_ok=True)
            if path.exists():
                with np.load(path) as z:new=z['ph']
            else:
                if family=='original':
                    f=read_feature(AUDIT,name,scenario);x,_=context.extend(f['X'],f['cols'].tolist(),segments);xs,_=context.extend(f['Xs'],f['cols'].tolist(),segments)
                else:
                    with np.load(PRIOR/'evaluation'/f'{name}__{scenario}.npz') as z:x,xs=z['X'],z['Xs']
                new=context.predict(model,x,xs);np.savez(path,ph=new);del x,xs
            blended=.5*ph+.5*new
            for candidate,spec in specs.items():
                decoder=dict(params,**{k:float(spec['cost']) for k in ['cost_ha','cost_in','cost_same','cost_other']})
                pred=boost.label(pd,temperature(blended,spec['temperature']),segments,len(gt),decoder)
                records.append({'match':name,'game':game,'cohort':cohort,'scenario':scenario,'candidate':candidate,**metric(gt,pred)})
    return game,records


def report():
    data=json.loads((FOLDER/'decoder_results.json').read_text());rows=data['records'];previous=json.loads((FOLDER/'development_results.json').read_text())['records']
    expected={(r['game'],r['scenario']):r['spell_f1_mean'] for r in previous if r['candidate']=='data29'}
    actual={(r['game'],r['scenario']):r['spell_f1_mean'] for r in rows if r['candidate']=='c9_t1'}
    if expected.keys()!=actual.keys() or any(abs(expected[k]-actual[k])>1e-12 for k in expected):raise ValueError('Unchanged settings did not reproduce data29')
    groups=[('known','matched_seed0'),('known','label_blind_seed1'),('known','stress_seed1'),('development','natural_detected'),('development','label_blind_seed7')]
    base=json.loads((FOLDER/'decision.json').read_text())['scores']['v3'];scores={}
    for candidate in json.loads(PROTOCOL.read_text())['candidates']:
        means=[]
        for cohort,scenario in groups:
            records=[r for r in rows if (r['candidate'],r['cohort'],r['scenario'])==(candidate,cohort,scenario)]
            if len(records)!=(9 if cohort=='known' else 10):raise ValueError('Incomplete sweep')
            means.append(statistics.mean(r['spell_f1_mean'] for r in records))
        weighted=.4*means[0]+.2*means[1]+.2*means[3]+.2*means[4];gain=weighted-base['weighted']
        scores[candidate]={'means':means,'weighted':weighted,'gain':gain,'pass':gain>=.005 and all(a>=b-.005 for a,b in zip(means,base['means']))}
    eligible=[n for n,s in scores.items() if s['pass']];specs=json.loads(PROTOCOL.read_text())['candidates']
    selected=max(eligible,key=lambda n:(scores[n]['weighted'],-abs(specs[n]['temperature']-1),-specs[n]['cost'])) if eligible else None
    result={'registered_protocol_sha256':sha(PROTOCOL),'unchanged_control_reproduced':True,'scores':scores,'selected_research_candidate':selected,'fresh_confirmation':False,'promoted':False,'release':'v3 unchanged'}
    (FOLDER/'decoder_decision.json').write_text(json.dumps(result,indent=2)+'\n');out=PROTOCOL.parent
    (out/'decoder_decision.json').write_text(json.dumps(result,indent=2)+'\n');(out/'decoder_results.json').write_text(json.dumps(data,indent=2)+'\n')
    lines=['\n## Bounded follow-up: temporal costs and conditional calibration','',
       'After the data-only result, nine decoding policies were registered before scoring. All reuse fixed29-game weights, the same whole-game exclusions and the same known development games. This is further adaptive development, not additional independent validation. The c9_t1 control reproduces the initial data29 scores exactly.','',
       '| Policy | Known matched | Known label-blind | Known stress | Skill natural | Skill label-blind | Weighted gain vs v3 | Pass |','|---|---:|---:|---:|---:|---:|---:|---|']
    for name,s in sorted(scores.items(),key=lambda kv:-kv[1]['weighted']):lines.append('| '+name+' | '+' | '.join(f'{v:.4f}' for v in s['means'])+f" | {s['gain']:+.4f} | {s['pass']} |")
    lines += ['',f'Selected research candidate: **{selected or "none"}**. V3 remains the verified release. No European9 or PFF confirmation scores are opened for this sweep.','']
    p=out/'REPORT.md';s=p.read_text().split('\n## Bounded follow-up:')[0];p.write_text(s+'\n'.join(lines));print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['score','report']);parser.add_argument('--jobs',type=int,default=4);args=parser.parse_args()
    if args.mode=='report':report()
    else:
        destination=FOLDER/'decoder_results.json'
        if destination.exists():raise ValueError('Never overwrite completed sweep')
        records=[]
        with ProcessPoolExecutor(max_workers=args.jobs,mp_context=get_context('spawn')) as pool:
            for future in as_completed([pool.submit(job,g) for g in plan()['evaluation_games']]):game,rows=future.result();records.extend(rows);print(game,'PASS',flush=True)
        destination.write_text(json.dumps({'protocol_sha256':sha(PROTOCOL),'records':records},indent=2)+'\n')
