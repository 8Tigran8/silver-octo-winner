"""Validate exact deployment assets/CPU implementation on external sources."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from temporal_trial import OUT
from mtah import v4,degrade,expanded_team as expanded
from mtah.methods import boost,boost_train
from legacy_skillcorner_reader import period_spans
from fresh_metrica_holdout import parse_tracking,event_labels
from wcsf_reader import labels
from confirm_expanded_team import team_metrics
from decoder_development import metric
import numpy as np


if __name__=='__main__':
    dest=OUT/'deployment_cpu_results.json'
    if dest.exists():raise ValueError('Do not overwrite completed CPU audit')
    assets=v4.load();params=dict(boost_train.decoder_params(),**json.loads((ROOT/'mtah/methods/v2_params.json').read_text())['decoder']);records=[];differences=[]
    games=[]
    with np.load(OUT/'external/wcsf_geometry.npz') as z:games.append(('wcsf','wcsf',{k:z[k] for k in z.files}))
    source=ROOT/'external/holdout_metrica3';data,teams,_=parse_tracking(source/'Sample_Game_3_metadata.xml',source/'Sample_Game_3_tracking.txt');games.append(('metrica3','metrica_3',data))
    for name in ['2068','2269','2417','2440','2841','3442','3518','3749','4039']:
        with np.load(ROOT/'reports/v3_search/legacy'/(name+'.npz')) as z:games.append(('european',name,{k:z[k] for k in z.files}))
    for cohort,name,clean in games:
        spans=[(0,len(clean['ball']))] if cohort=='metrica3' else period_spans(clean['period']);dummy=np.full(len(clean['ball']),3,np.int8)
        for a,b in spans:dummy[a:b]=0
        data={k:v for k,v in clean.items() if k!='gt'};data['gt']=dummy
        scenarios=[('natural_detected',data)]
        if cohort!='european':scenarios.append(('label_blind_seed7',degrade.degrade(data,seed=7,params=expanded.BLIND)))
        gt=labels(ROOT/'external/wcsf',OUT/'selection.json') if cohort=='wcsf' else event_labels(source/'Sample_Game_3_events.json',len(dummy),teams)[0] if cohort=='metrica3' else clean['gt']
        original=json.loads((OUT/(cohort+'_results.json')).read_text())['records']
        for scenario,data in scenarios:
            pd,ph,nd,nh=v4.components(data,spans,assets)
            for candidate,d,h in [('v3',pd,ph),('fusion50',.5*pd+.5*nd,.5*ph+.5*nh)]:
                pred=boost.label(d,h,spans,len(gt),params);result=team_metrics(gt,pred,np.isfinite(data['ball'][:,0])) if cohort=='european' else metric(gt,pred);key='all_conditional_team_macroF1' if cohort=='european' else 'spell_f1_mean'
                prior=next(r for r in original if (r['match'],r['candidate'],r['scenario'])==(name,candidate,scenario));differences.append(abs(result[key]-prior[key]));records.append({'cohort':cohort,'match':name,'candidate':candidate,'scenario':scenario,**result})
        print(name,'CPU release-asset comparison PASS',flush=True)
    dest.write_text(json.dumps({'deployment':'Frozen assets/v4 and single-thread CPU inference','max_primary_metric_difference_vs_MPS':max(differences),'records':records},indent=2)+'\n')
