"""Score one frozen temporal candidate outside neural training; no retuning."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from temporal_trial import OUT,PLAN,plan,sha,context,temporal
from evaluate_temporal import model,upsample
from expanded_team_trial import load_model as v3_team
from legacy_skillcorner_reader import period_spans
from confirm_expanded_team import team_metrics
from wcsf_reader import labels as wcsf_labels
from fresh_metrica_holdout import event_labels,parse_tracking
from mtah import features2,stack,expanded_team as expanded,degrade
from mtah.methods import boost,boost_train
from quality_audit import official
from decoder_development import metric
from threadpoolctl import threadpool_limits
import cloudpickle,numpy as np,torch


def frozen():
    selection=json.loads((OUT/'selection.json').read_text())
    if selection['plan_sha256']!=sha(PLAN) or selection['full_checkpoint_sha256']!=sha(OUT/'models/full/model.pt') or selection['baseline_checkpoint_sha256']!=sha(ROOT/'out/training_preflight.pkl') or selection['baseline_team_sha256']!=sha(ROOT/'reports/v4_search/models/all_nineteen.pkl'):raise ValueError('Frozen models/protocol changed')
    return selection


def probabilities(data,segments,m,mean,scale,base,head,device):
    x,cols=features2.build(data,segments);xs,_=features2.build(data,segments,swap=True)
    vd,vh=boost.predict(base['stage1'],x,xs);vd,vh=stack.predict(base['stage2'],vd,vh)
    x,_=context.extend(x,cols,segments);xs,_=context.extend(xs,cols,segments);vh=.5*vh+.5*context.predict(head,x,xs)
    spans=[((a+4)//5,(b+4)//5) for a,b in segments]
    nd,nh=temporal.probabilities(m,x[::5],xs[::5],spans,mean,scale,device)
    nd,nh=upsample(nd,nh,{'full_segments':segments,'segments':spans,'full_length':len(x)})
    return vd,vh,nd,nh


def whole_span_from_geometry(data):
    # No period metadata in Metrica3 export: predict the entire tracking stream
    # rather than borrowing annotation-derived period boundaries.
    return [(0,len(data['ball']))]


def run(cohort,device):
    selected=frozen();path=OUT/(cohort+'_results.json')
    if path.exists():raise ValueError('Never overwrite completed external evaluation')
    m,mean,scale=model('full',device);head=v3_team()
    with (ROOT/'out/training_preflight.pkl').open('rb') as f:base=cloudpickle.load(f)
    params=dict(boost_train.decoder_params(),**json.loads((ROOT/'mtah/methods/v2_params.json').read_text())['decoder']);records=[]
    if cohort=='wcsf':
        eligibility=json.loads((ROOT/'docs/experiments/temporal/wcsf_eligibility.json').read_text())
        if not eligibility['eligible_for_anonymous_research_diagnostic']:raise ValueError('Potential duplicate source')
        with np.load(OUT/'external/wcsf_geometry.npz') as z:clean={k:z[k] for k in z.files}
        games=[('wcsf',clean)]
    elif cohort=='metrica3':
        source=ROOT/'external/holdout_metrica3';clean,teams,_=parse_tracking(source/'Sample_Game_3_metadata.xml',source/'Sample_Game_3_tracking.txt');games=[('metrica_3',clean)]
    else:
        games=[]
        for name in ['2068','2269','2417','2440','2841','3442','3518','3749','4039']:
            with np.load(ROOT/'reports/v3_search/legacy'/(name+'.npz')) as z:games.append((name,{k:z[k] for k in z.files}))
    torch.set_num_threads(2)
    with threadpool_limits(1):
        for name,clean in games:
            segments=whole_span_from_geometry(clean) if cohort=='metrica3' else period_spans(clean['period'])
            # Only clock/period-derived masks drive input degradation. Reference
            # ownership/alive labels are never passed to predictor/degrader.
            dummy=np.full(len(clean['ball']),3,np.int8)
            for a,b in segments:dummy[a:b]=0
            predictor={k:v for k,v in clean.items() if k!='gt'};predictor['gt']=dummy
            cases=[('natural_detected',predictor)]
            if cohort!='european':cases.append(('label_blind_seed7',degrade.degrade(predictor,seed=7,params=expanded.BLIND)))
            predictions=[]
            for scenario,data in cases:
                vd,vh,nd,nh=probabilities(data,segments,m,mean,scale,base,head,device)
                methods=[('v3',vd,vh)]
                methods.append((selected['candidate'],nd,nh) if selected['candidate']=='tcn' else ('fusion50',.5*vd+.5*nd,.5*vh+.5*nh))
                for method,pd,ph in methods:
                    pred=boost.label(pd,ph,segments,len(dummy),params);predictions.append((scenario,method,pred,np.isfinite(data['ball'][:,0])))
            # Reference read happens only after fixed candidate predictions.
            if cohort=='wcsf':gt=wcsf_labels(ROOT/'external/wcsf',OUT/'selection.json')
            elif cohort=='metrica3':gt,_=event_labels(ROOT/'external/holdout_metrica3/Sample_Game_3_events.json',len(dummy),teams)
            else:gt=clean['gt']
            for scenario,method,pred,seen in predictions:
                result=team_metrics(gt,pred,seen) if cohort=='european' else metric(gt,pred)
                records.append({'match':name,'scenario':scenario,'candidate':method,**result})
            print(name,'frozen external comparison PASS',flush=True)
    path.write_text(json.dumps({'cohort':cohort,'selection_sha256':sha(OUT/'selection.json'),'records':records},indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('cohort',choices=['wcsf','european','metrica3']);p.add_argument('--device',default='mps');a=p.parse_args();run(a.cohort,a.device)
