"""Correctness-fix regression on familiar PFF development only; no selection sweep."""
import sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from mtah import mixed_candidate,v4,degrade,expanded_team
from mtah.methods import boost
from pff_mixed_score import decoder
from pff_development import geometry,spans
from decoder_development import metric
old=json.loads((ROOT/'docs/experiments/pff-mixed/pff_dev_results.json').read_text())['records'];base={(r['game'],r['scenario'],r['reference']):r for r in old if r['candidate']=='boundary102'};games=list(dict.fromkeys(k[0] for k in base));assets=v4.load();_,state=mixed_candidate.load_bundle(ROOT/'assets/pff-mixed-candidate/bundle.json');rows=[]
for game in games:
 start=time.monotonic();data,receipt=geometry(game);parts=spans(data['period']);dummy=np.where(data['period']>0,0,3).astype(np.int8);bad=degrade.degrade(dict(data,gt=dummy),seed=7,params=expanded_team.BLIND);bad.pop('gt');preds={}
 for case,d in [('natural_detected',data),('label_blind_seed7',bad)]:
  pd,ph=mixed_candidate.probabilities(d,parts,assets,state);preds[case]=boost.label(pd,ph,parts,len(dummy),decoder())
 with np.load(ROOT/'reports/pff_access/references'/f'kloppy_{game}.npz') as z:
  for reference,key in [('canonical','gt'),('sdk_shadow','alternate')]:
   for case,pred in preds.items():
    r=metric(z[key],pred);b=base[game,case,reference];rows.append(dict(game=game,scenario=case,reference=reference,**r,spell_gain=r['spell_f1_mean']-b['spell_f1_mean'],macro_gain=r['macro_f1']-b['macro_f1']))
 print(game,round(time.monotonic()-start,1),flush=True)
summary={}
for ref in ['canonical','sdk_shadow']:
 for case in ['natural_detected','label_blind_seed7']:
  rr=[r for r in rows if r['reference']==ref and r['scenario']==case];summary[ref+'/'+case]={k:float(np.mean([r[k] for r in rr])) for k in ['spell_f1_mean','macro_f1','spell_gain','macro_gain']};summary[ref+'/'+case]['worst_spell_gain']=min(r['spell_gain'] for r in rr)
(ROOT/'reports/independent_audit/external_regression.json').write_text(json.dumps({'scope':'Familiar development regression for correctness repair; not new final validation','records':rows,'summary':summary},indent=2)+'\n');print(summary)
