"""Descriptive development-only evidence audit; never changes predictions."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
import numpy as np
from quality_audit import official
from boundary_event_study import sha,write
cfg=json.loads((ROOT/'docs/experiments/possession-emissions/plan.json').read_text());rows=[]
for game in cfg['dev']:
 with np.load(ROOT/'reports/pff_access/references'/f'kloppy_{game}.npz') as z:gt=z['gt']
 with np.load(ROOT/'reports/boundary_events/probabilities'/f'{game}__natural_detected.npz') as z:pd=z['pd'];ph=z['ph']
 with np.load(ROOT/'reports/boundary_events'/f'{game}_predictions.npz') as z:pred=z['natural_detected__unchanged']
 q=np.column_stack(((1-pd)*ph,(1-pd)*(1-ph),pd));runs=official.valid_runs(gt);spells=official.possession_spells(gt,runs);ps=official.possession_spells(official.fill_abstentions(pred,runs),runs);flags=official.matched_gt_flags(official.spell_overlaps(spells,ps),len(spells),.5)
 for i,(a,b,k) in enumerate(spells):
  length=(b-a)/25;duration='under1' if length<1 else '1to3' if length<3 else '3to10' if length<10 else '10plus';window=q[a:b];best=window.argmax(axis=1)
  rows.append({'game':game,'duration':duration,'matched_iou50':bool(flags[i]),'fraction_correct_argmax':float(np.mean(best==k)),'correct_mean_probability_wins':bool(window.mean(axis=0).argmax()==k),'correct_log_evidence_wins':bool(np.log(np.maximum(window,1e-7)).sum(axis=0).argmax()==k)})
summary={}
for duration in ['under1','1to3','3to10','10plus']:
 summary[duration]={}
 for category in ['all','missed_iou50']:
  rr=[r for r in rows if r['duration']==duration and (category=='all' or not r['matched_iou50'])]
  summary[duration][category]={'spells':len(rr),**{k:float(np.mean([r[k] for r in rr])) for k in ['fraction_correct_argmax','correct_mean_probability_wins','correct_log_evidence_wins']}}
write(ROOT/'reports/possession_emissions/evidence_audit.json',{'scope':'Post-hoc canonical natural development diagnostic. Ground-truth interval endpoints used ONLY to measure evidence; this is oracle-conditioned analysis, not a deployable score or independent test. No final/bridge/organizer data.','summary':summary})
print(json.dumps(summary,indent=2))
