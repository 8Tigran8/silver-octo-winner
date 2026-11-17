"""One-shot frozen release comparison. No fitting or target-label access."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
import argparse,json,sys,time,subprocess
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context
import numpy as np
import torch
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
from mtah import final_candidate,recovery_candidate,v4,mixed_temporal,activity_temporal,expanded_team,degrade
from mtah.recovery_features import activity_features
from mtah.methods import boost
from pff_reader import sha,load_geometry,tracking_reference
from pff_development import spans,geometry
from decoder_development import metric
from quality_audit import official
from provider_stress import corrupt_geometry
DOC=ROOT/'docs/experiments/final-opened-comparison';OUT=ROOT/'reports/final_opened_comparison'
OLD='weighted_short_gap';NEW='uniform_temporal_focus'
def now():return datetime.now(timezone.utc).isoformat()
def plan():return json.loads((DOC/'plan.json').read_text())
def dump(path,obj):Path(path).write_text(json.dumps(obj,indent=2)+'\n')
def raw_paths(g):return {k:ROOT/'external/pff'/d/(g+s) for k,d,s in [('tracking','Tracking Data','.jsonl.bz2'),('metadata','Metadata','.json'),('rosters','Rosters','.json')]}
def install_target_guard():
 def audit(event,args):
  if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
  path=str(Path(os.fsdecode(args[0])).resolve())
  if '/evaluation/gt/' in path or 'BAR-ATH' in path or '/hackathons/more-than-a-hack-2027/data/' in path:raise PermissionError('Organizer data/labels blocked in external audit')
 sys.addaudithook(audit)
def initialize():
 install_target_guard()
 global BUNDLE,STATE
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
 BUNDLE,STATE=final_candidate.load_bundle(ROOT/'assets/final-candidate/bundle.json')
def paired(data,parts,state):
 if 'gt' in data:raise ValueError('Reference labels forbidden in inference')
 parent,dead,owner,activity=state;assets,neural,expert,columns=parent
 bd,bh,_,_=v4.components(data,parts,assets)
 with threadpool_limits(1):
  x,xs,_=expanded_team.feature_pair(data,parts);a,names=activity_features(data,parts)
  if names!=columns:raise ValueError('Activity schema changed')
  oa=expert.predict_proba(a[::5])[:,1]
 n=len(x);down=[((s+4)//5,(e+4)//5) for s,e in parts];model,mean,scale=neural
 nd,nh=mixed_temporal.probabilities(model,x[::5],xs[::5],down,mean,scale);nd,nh=v4.neural_upsample(nd,nh,parts,n)
 dd,_=mixed_temporal.probabilities(dead,x[::5],xs[::5],down,mean,scale);dd=v4.neural_upsample(dd,dd,parts,n)[0]
 _,hh=mixed_temporal.probabilities(owner,x[::5],xs[::5],down,mean,scale);hh=v4.neural_upsample(hh,hh,parts,n)[0]
 am,mu,sd=activity;ad=activity_temporal.probabilities(am,a[::5],down,mu,sd);ad=v4.neural_upsample(ad,ad,parts,n)[0]
 oa=v4.neural_upsample(oa,oa,parts,n)[0];missing=~np.isfinite(data['ball'][:,:2]).all(axis=1)
 pd,ph=.5*bd+.5*nd,.5*bh+.5*nh;cd,ch=pd.copy(),ph.copy()
 pd[missing]=.75*nd[missing]+.25*oa[missing];cd[missing]=.75*dd[missing]+.25*ad[missing];ch[missing]=hh[missing]
 return {OLD:(pd,ph),NEW:(cd,ch)}
def scenario(data,name):
 if 'gt' in data:raise ValueError('Reference labels forbidden in corruption')
 if name=='natural_detected':return data,{'kind':'unchanged geometry'}
 if name=='label_blind_seed7':
  dummy=np.where(data['period']>0,0,3).astype(np.int8)
  d=degrade.degrade(dict(data,gt=dummy),seed=7,params=expanded_team.BLIND);d.pop('gt');d['period']=data['period']
  return d,dict(kind='original blind',seed=7)
 p=plan();preset=p['vv_presets'][name.removeprefix('vv_')];n=len(data['ball'])
 native={k:data[k] for k in ['players','ball','period']};native.update(timestamps=np.arange(n)/25,source_frame=np.arange(n,dtype=np.int64))
 damaged,receipt=corrupt_geometry(native,data['players'],data['ball'],data['period'],data['pitch'],25,preset,p['vv_seed'])
 d=dict(data,players=damaged['players'],ball=damaged['ball']);receipt['clock_scope']='25Hz canonical input, not original native PFF fps';return d,receipt

def diagnostics(gt,pred,seen):
 runs=official.valid_runs(gt);filled=official.fill_abstentions(pred,runs)
 truth=official.possession_spells(gt,runs);other=official.possession_spells(filled,runs)
 flags=official.matched_gt_flags(official.spell_overlaps(truth,other),len(truth),.5)
 lengths=np.array([(b-a)/25 for a,b,_ in truth]);short=lengths<3;sub=lengths<1
 live=(gt<2)&~seen;dead=(gt==2)&~seen
 return dict(short_reference=int(short.sum()),short_matched=int(flags[short].sum()),subsecond_reference=int(sub.sum()),subsecond_matched=int(flags[sub].sum()),missing_live=int(live.sum()),missing_dead=int(dead.sum()),missing_false_dead=int(((pred==2)&live).sum()),missing_false_live=int(((pred<2)&dead).sum()),missing_wrong_team=int(((pred<2)&(pred!=gt)&live).sum()))

def summary(rows):
 p=plan();expected={(g['game_id'],s,c,r) for g in p['fixtures'] for s in p['scenarios'] for c in [OLD,NEW] for r in p['references']}
 index={(r['game'],r['scenario'],r['candidate'],r['reference']):r for r in rows}
 if len(index)!=len(rows) or set(index)!=expected:raise ValueError('Incomplete/duplicate comparison grid')
 output=[];guards=p['selection']['comparison_guards'];dg=p['selection']['diagnostic_guards']
 for cohort in p['cohorts']:
  games=[g['game_id'] for g in p['fixtures'] if cohort=='all18' or g['cohort']==cohort]
  for s in p['scenarios']:
   for ref in p['references']:
    old=[index[g,s,OLD,ref] for g in games];new=[index[g,s,NEW,ref] for g in games]
    delta=np.array([b['spell_f1_mean']-a['spell_f1_mean'] for a,b in zip(old,new)])
    macro=float(np.mean([b['macro_f1']-a['macro_f1'] for a,b in zip(old,new)]))
    classes={k:float(np.mean([b['per_class'][k]['f1']-a['per_class'][k]['f1'] for a,b in zip(old,new)])) for k in ['H','A','D']}
    rng=np.random.default_rng(20261117);boot=delta[rng.integers(0,len(games),size=(10000,len(games)))].mean(axis=1)
    checks=dict(mean_spell=bool(delta.mean()>=guards['mean_spell_gain_min']),macro=macro>=guards['mean_macro_gain_min'],worst=bool(delta.min()>=guards['worst_match_spell_gain_min']),**{k+'_f1':v>=guards['mean_each_class_f1_gain_min'] for k,v in classes.items()})
    counts={label:{k:sum(x['diagnostics'][k] for x in group) for k in group[0]['diagnostics']} for label,group in [('old',old),('new',new)]}
    rates={label:{'short_recall':d['short_matched']/d['short_reference'] if d['short_reference'] else None,'false_dead':d['missing_false_dead']/d['missing_live'] if d['missing_live'] else None,'false_live':d['missing_false_live']/d['missing_dead'] if d['missing_dead'] else None,'wrong_team':d['missing_wrong_team']/d['missing_live'] if d['missing_live'] else None} for label,d in counts.items()}
    if any(v is None for d in rates.values() for v in d.values()):raise ValueError('Undefined registered diagnostic rate')
    changes={k:rates['new'][k]-rates['old'][k] for k in rates['old']}
    if cohort=='all18' and s in ['natural_detected','label_blind_seed7']:
     checks.update(short_recall=changes['short_recall']>=dg['short_under3_recall_gain_min'],false_dead=changes['false_dead']<=dg['missing_false_dead_rate_increase_max'],false_live=changes['false_live']<=dg['missing_false_live_rate_increase_max'],wrong_team=changes['wrong_team']<=dg['missing_wrong_team_rate_increase_max'])
    output.append(dict(cohort=cohort,scenario=s,reference=ref,games=len(games),old_spell=float(np.mean([x['spell_f1_mean'] for x in old])),new_spell=float(np.mean([x['spell_f1_mean'] for x in new])),gain=float(delta.mean()),wins=int((delta>0).sum()),ties=int((delta==0).sum()),worst_gain=float(delta.min()),bootstrap95=list(map(float,np.quantile(boot,[.025,.975]))),macro_gain=macro,class_f1_gain=classes,counts=counts,rates=rates,diagnostic_rate_changes=changes,checks=checks,pass_all=all(checks.values()),per_match_gain=dict(zip(games,map(float,delta)))))
 passed=all(x['pass_all'] for x in output)
 return dict(selected=NEW if passed else OLD,new_passes=passed,comparisons=output,rule=p['selection'],fresh_holdout=False)

def check():
 frozen=json.loads((DOC/'freeze.json').read_text())
 for name,h in frozen['files_sha256'].items():
  if sha(ROOT/name)!=h:raise ValueError('Frozen input/source changed: '+name)
 return frozen

def freeze():
 if (DOC/'freeze.json').exists() or OUT.exists():raise ValueError('Already frozen')
 if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True):raise ValueError('Commit source/plan before freeze')
 p=plan();fixtures=p['fixtures'];protocol=json.loads((ROOT/'docs/experiments/more-data/pff_protocol.json').read_text());excluded={g['game_id'] for k in ['training','development'] for g in protocol[k]}
 ids={g['game_id'] for g in fixtures}
 if len(ids)!=18 or ids&excluded:raise ValueError('Invalid audit split')
 for folder in ['recovery-candidate','final-candidate']:
  training=set(json.loads((ROOT/f'assets/{folder}/bundle.json').read_text())['training_games'])
  if any('pff/'+g in training for g in ids):raise ValueError('Training overlap')
 files=set((ROOT/'mtah').rglob('*.py'))|set((ROOT/'mtah').rglob('*.json'))
 for folder in ['assets/v4','assets/pff-mixed-candidate','assets/recovery-candidate','assets/final-candidate']:files.update(x for x in (ROOT/folder).rglob('*') if x.is_file())
 files.update(ROOT/x for x in ['tools/final_opened_comparison.py','tools/provider_stress.py','tools/provider_stress_presets.json','tools/pff_reader.py','tools/skillcorner_reader.py','tools/pff_development.py','tools/decoder_development.py','tools/quality_audit.py','docs/experiments/final-opened-comparison/plan.json','docs/experiments/more-data/pff_protocol.json','ref/official/evaluation/possession_metrics.py','submission_manifest.json','tests/test_final_opened_comparison.py','docs/experiments/final-opened-comparison/equivalence.json','docs/experiments/final-opened-comparison/tests.log'])
 for g in fixtures:
  files.update(raw_paths(g['game_id']).values());files.update(ROOT/'reports/recovery_opened_audit'/g['game_id']/x for x in ['predictions.npz','diagnostic_inputs.npz'])
 dump(DOC/'freeze.json',dict(frozen_utc=now(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),files_sha256={str(f.relative_to(ROOT)):sha(f) for f in sorted(files)}));print('Frozen',len(files),'files',flush=True)

def equivalence():
 initialize();data,_=geometry('3816');parts=spans(data['period']);out=[]
 for s in ['natural_detected','vv_hz10_noise0p5']:
  d,_=scenario(data,s);pairs=paired(d,parts,STATE)
  direct={OLD:recovery_candidate.probabilities(d,parts,STATE[0]),NEW:final_candidate.probabilities(d,parts,STATE)}
  for c in [OLD,NEW]:
   error=max(float(np.max(np.abs(x-y))) for x,y in zip(pairs[c],direct[c]));assert error==0,(s,c,error)
   assert np.array_equal(boost.label(*pairs[c],parts,len(d['ball']),BUNDLE['decoder']),boost.label(*direct[c],parts,len(d['ball']),BUNDLE['decoder']))
   out.append(dict(scenario=s,candidate=c,max_probability_error=error))
  print('Training-only runtime parity',s,'PASS',flush=True)
 dump(DOC/'equivalence.json',dict(status='PASS',game='3816',quality_scored=False,records=out,tool_sha256=sha(__file__)))

def worker(fixture):
 g=fixture['game_id'];folder=OUT/g;folder.mkdir(exist_ok=True);started=time.monotonic();paths=raw_paths(g)
 meta=json.loads(paths['metadata'].read_text());meta=meta[0] if isinstance(meta,list) else meta
 if str(meta['id'])!=g or any(meta[k+'Team']['name']!=fixture[k] for k in ['home','away']):raise ValueError('Fixture identity mismatch')
 data,geo=load_geometry(paths['tracking'],paths['metadata'],paths['rosters']);parts=spans(data['period']);predictions={};seen={};receipts={}
 print('Geometry',g,round(time.monotonic()-started,1),flush=True)
 for s in plan()['scenarios']:
  d,receipts[s]=scenario(data,s);pairs=paired(d,parts,STATE);seen[s]=np.isfinite(d['ball'][:,:2]).all(axis=1)
  for c in [OLD,NEW]:predictions[s+'__'+c]=boost.label(*pairs[c],parts,len(d['ball']),BUNDLE['decoder'])
  print('Predicted',g,s,round(time.monotonic()-started,1),flush=True)
 path=folder/'predictions.npz';np.savez_compressed(path,**predictions);prediction_time=now()
 # Historical output comparison is a reproducibility check, not a source for new inference.
 with np.load(ROOT/'reports/recovery_opened_audit'/g/'predictions.npz',allow_pickle=False) as z:
  for s in ['natural_detected','label_blind_seed7']:
   if not np.array_equal(predictions[s+'__'+OLD],z[s+'__'+OLD]):raise ValueError('Old output changed: '+g+' '+s)
 gt,alt,ref=tracking_reference(paths['tracking'],data['time'],data['period'],geo['fps'])
 if ref['known_fraction']<.99 or any(ref['class_frames'][k]<=0 for k in ['H','A','D']):raise ValueError('Invalid reference; no fixture dropping')
 with np.load(ROOT/'reports/recovery_opened_audit'/g/'diagnostic_inputs.npz',allow_pickle=False) as z:
  if not np.array_equal(gt,z['gt']) or not np.array_equal(alt,z['alternate']):raise ValueError('Reference changed')
 rows=[]
 for r,truth in [('canonical',gt),('sdk_shadow',alt)]:
  for s in plan()['scenarios']:
   for c in [OLD,NEW]:
    pred=predictions[s+'__'+c];rows.append(dict(game=g,cohort=fixture['cohort'],scenario=s,candidate=c,reference=r,**metric(truth,pred),diagnostics=diagnostics(truth,pred,seen[s])))
 result=dict(records=rows,source=dict(fixture=fixture,geometry=geo,reference=ref,scenario_receipts=receipts,predictions_sha256=sha(path),prediction_saved_utc=prediction_time,references_loaded_after_predictions=True,historical_old_parity=True,historical_reference_parity=True,seconds=time.monotonic()-started),freeze_sha256=sha(DOC/'freeze.json'))
 dump(folder/'result.json',result);print('Completed',g,round(time.monotonic()-started,1),flush=True);return result

def run():
 check();e=json.loads((DOC/'equivalence.json').read_text());assert e['status']=='PASS' and e['tool_sha256']==sha(__file__)
 OUT.mkdir(exist_ok=False);dump(OUT/'OPENED.json',dict(started_utc=now(),freeze_sha256=sha(DOC/'freeze.json')));rows=[];sources=[]
 with ProcessPoolExecutor(max_workers=2,mp_context=get_context('spawn'),initializer=initialize) as pool:
  for f in as_completed([pool.submit(worker,g) for g in plan()['fixtures']]):
   r=f.result();rows.extend(r['records']);sources.append(r['source'])
 check();dump(DOC/'results.json',dict(records=rows,sources=sources,freeze_sha256=sha(DOC/'freeze.json')))
 decision=summary(rows);dump(DOC/'decision.json',decision);print('Decision',decision['selected'],flush=True)
if __name__=='__main__':
 install_target_guard()
 p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','equivalence','run']);a=p.parse_args();globals()[a.action]()
