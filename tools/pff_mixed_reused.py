"""Both registered models on all eleven previously reused external games."""
import json
import os
from pathlib import Path
import sys
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from mtah import v4,degrade,expanded_team,features2,possession_context as context
from mtah.methods import boost
from legacy_skillcorner_reader import period_spans
from fresh_metrica_holdout import parse_tracking,event_labels
from wcsf_reader import labels
from confirm_expanded_team import team_metrics
from decoder_development import metric
import pff_mixed_train as train
from pff_mixed_score import neural,decoder
from pff_reader import sha


def main():
    dest=train.OUT/'reused_results.json'
    if dest.exists():raise ValueError('Preserve completed reused-source results')
    names=[s['name'] for s in train.plan()['candidates']];states={name:train.load(name,'full')[0] for name in names};assets=v4.load();params=decoder();records=[]
    original=json.loads((ROOT/'docs/experiments/temporal/deployment_cpu_results.json').read_text())['records']
    baseline={(r['cohort'],r['match'],r['scenario']):r for r in original if r['candidate']=='fusion50'}
    games=[]
    with np.load(ROOT/'reports/temporal/external/wcsf_geometry.npz') as z:games.append(('wcsf','wcsf',{k:z[k] for k in z.files}))
    source=ROOT/'external/holdout_metrica3';data,teams,_=parse_tracking(source/'Sample_Game_3_metadata.xml',source/'Sample_Game_3_tracking.txt');games.append(('metrica3','metrica_3',data))
    for game in ['2068','2269','2417','2440','2841','3442','3518','3749','4039']:
        with np.load(ROOT/'reports/v3_search/legacy'/(game+'.npz')) as z:games.append(('european',game,{k:z[k] for k in z.files}))
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    for cohort,game,clean in games:
        parts=[(0,len(clean['ball']))] if cohort=='metrica3' else period_spans(clean['period']);dummy=np.full(len(clean['ball']),3,np.int8)
        for a,b in parts:dummy[a:b]=0
        data={k:v for k,v in clean.items() if k!='gt'};cases=[('natural_detected',data)]
        if cohort!='european':
            noisy=degrade.degrade(dict(data,gt=dummy),seed=7,params=expanded_team.BLIND);noisy.pop('gt');cases.append(('label_blind_seed7',noisy))
        predictions=[]
        for scenario,observed in cases:
            with threadpool_limits(1):
                bd,bh,_,_=v4.components(observed,parts,assets)
                x,cols=features2.build(observed,parts);xs,_=features2.build(observed,parts,swap=True)
                x,_=context.extend(x,cols,parts);xs,_=context.extend(xs,cols,parts)
                for name,state in states.items():
                    nd,nh=neural(state,x,xs,parts);pred=boost.label(.5*bd+.5*nd,.5*bh+.5*nh,parts,len(dummy),params)
                    predictions.append((scenario,name,pred,np.isfinite(observed['ball'][:,0])))
        gt=labels(ROOT/'external/wcsf',ROOT/'reports/temporal/selection.json') if cohort=='wcsf' else event_labels(source/'Sample_Game_3_events.json',len(dummy),teams)[0] if cohort=='metrica3' else clean['gt']
        for scenario,name,pred,seen in predictions:
            result=team_metrics(gt,pred,seen) if cohort=='european' else metric(gt,pred);key='all_conditional_team_macroF1' if cohort=='european' else 'spell_f1_mean';prior=baseline[cohort,game,scenario]
            records.append({'cohort':cohort,'match':game,'scenario':scenario,'candidate':name,'primary_metric':key,'v4_primary':prior[key],'candidate_primary':result[key],'gain':result[key]-prior[key],**result})
        print('reused',game,'complete',flush=True)
    decisions={}
    for name in names:
        rows=[r for r in records if r['candidate']==name];gains={cohort+'/'+scenario:float(np.mean([r['gain'] for r in rows if r['cohort']==cohort and r['scenario']==scenario])) for cohort,scenario in sorted({(r['cohort'],r['scenario']) for r in rows})}
        decisions[name]={'checks':{key:gain>=-.01 for key,gain in gains.items()},'primary_gains':gains,'scope':'Eleven reused research games; European9conditionalH/A only, not fresh final confirmation'}
    dest.write_text(json.dumps({'mixed_plan_sha256':sha(train.PLAN),'records':records,'decisions':decisions},indent=2)+'\n');print(json.dumps(decisions),flush=True)


if __name__=='__main__':main()
