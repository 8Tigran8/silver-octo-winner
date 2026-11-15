"""Frozen local candidate check on already-used external cohorts; not fresh tests."""
from pathlib import Path
import json,sys,hashlib,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from mtah import v4,research_candidate,degrade,expanded_team as expanded
from mtah.methods import boost,boost_train
from legacy_skillcorner_reader import period_spans
from fresh_metrica_holdout import parse_tracking,event_labels
from wcsf_reader import labels
from confirm_expanded_team import team_metrics
from decoder_development import metric
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--bundle',type=Path,default=ROOT/'reports/reliability/full_bundle.json');parser.add_argument('--output',type=Path,default=ROOT/'reports/reliability/reused_external.json');args=parser.parse_args()
    out=args.output
    if out.exists():raise ValueError('Do not overwrite external research checks')
    bundle_path=args.bundle;bundle,states=research_candidate.load_bundle(bundle_path);assets=v4.load()
    original=json.loads((ROOT/'docs/experiments/temporal/deployment_cpu_results.json').read_text())['records']
    baseline={(r['cohort'],r['match'],r['scenario']):r for r in original if r['candidate']=='fusion50'}
    params=dict(boost_train.decoder_params(),**json.loads((ROOT/'mtah/methods/v2_params.json').read_text())['decoder']);games=[];records=[]
    with np.load(ROOT/'reports/temporal/external/wcsf_geometry.npz') as z:games.append(('wcsf','wcsf',{k:z[k] for k in z.files}))
    source=ROOT/'external/holdout_metrica3';data,metrica_teams,_=parse_tracking(source/'Sample_Game_3_metadata.xml',source/'Sample_Game_3_tracking.txt');games.append(('metrica3','metrica_3',data))
    for name in ['2068','2269','2417','2440','2841','3442','3518','3749','4039']:
        with np.load(ROOT/'reports/v3_search/legacy'/(name+'.npz')) as z:games.append(('european',name,{k:z[k] for k in z.files}))
    for cohort,name,clean in games:
        spans=[(0,len(clean['ball']))] if cohort=='metrica3' else period_spans(clean['period']);dummy=np.full(len(clean['ball']),3,np.int8)
        for a,b in spans:dummy[a:b]=0
        data={k:v for k,v in clean.items() if k!='gt'};data['gt']=dummy
        cases=[('natural_detected',data)]
        if cohort!='european':cases.append(('label_blind_seed7',degrade.degrade(data,seed=7,params=expanded.BLIND)))
        predictions=[]
        for scenario,data in cases:
            pd,ph=research_candidate.probabilities(data,spans,assets,states,bundle['rule'],bundle.get('temporal_blend'))
            pred=boost.label(pd,ph,spans,len(dummy),params);predictions.append((scenario,pred,np.isfinite(data['ball'][:,0])))
        gt=labels(ROOT/'external/wcsf',ROOT/'reports/temporal/selection.json') if cohort=='wcsf' else event_labels(source/'Sample_Game_3_events.json',len(dummy),metrica_teams)[0] if cohort=='metrica3' else clean['gt']
        for scenario,pred,seen in predictions:
            result=team_metrics(gt,pred,seen) if cohort=='european' else metric(gt,pred);key='all_conditional_team_macroF1' if cohort=='european' else 'spell_f1_mean';prior=baseline[cohort,name,scenario]
            records.append({'cohort':cohort,'match':name,'scenario':scenario,'candidate':bundle['candidate'],'primary_metric':key,'v4_primary':prior[key],'candidate_primary':result[key],'gain':result[key]-prior[key],**result})
        print(name,'reused-source research check complete',flush=True)
    out.write_text(json.dumps({'candidate':bundle['candidate'],'bundle_sha256':hashlib.sha256(bundle_path.read_bytes()).hexdigest(),'fresh_independent_test':False,'scope':'Same11previously used external games; European9 conditional-only. WCSF ambiguity/identity caveats unchanged. No competition quality score. No candidate retuning/runner-up selection in this check.','records':records},indent=2)+'\n')


if __name__=='__main__':main()
