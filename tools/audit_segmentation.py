"""Label-free output structure; no target reference is opened."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
import numpy as np
from quality_audit import official
from mtah.config import HALVES
p=ROOT/'reports/independent_audit/audited-run-1.csv'
c=official.label_codes(official.read_label_csv(p));parts=[(a,b+1) for _,a,b in HALVES];runs=official.segments(c,parts);trans=official.transitions(c,parts)
def stats(values):
 return {'count':len(values),'mean_s':float(np.mean(values)) if values else None,'quantiles_s':np.quantile(values,[0,.1,.25,.5,.75,.9,1]).tolist() if values else [],'fraction_under1s':float(np.mean(np.array(values)<1)) if values else None}
r={'scope':'Prediction-only aggregate morphology. No target truth, tuning rule or hidden quality estimate.','by_label':{k:stats([(b-a)/25 for a,b,x in runs if x==i]) for i,k in enumerate('HAD')},'transitions':{f'{a}->{b}':sum(x==i and y==j for t,x,y in trans) for i,a in enumerate('HAD') for j,b in enumerate('HAD') if i!=j},'switches_per_minute':len(trans)/(sum(b-a for a,b in parts)/25/60),'X_fraction_all':float(np.mean(c==3)),'X_fraction_graded':float(np.mean(np.concatenate([c[a:b] for a,b in parts])==3)),'islands_under1s':sum((b-a)<25 for a,b,k in runs),'patterns':{}}
for labels in [(0,1,0),(2,0,2)]:
 r['patterns'][''.join('HADX'[x] for x in labels)]=sum(tuple(s[2] for s in runs[i:i+3])==labels and runs[i][1]==runs[i+1][0] and runs[i+1][1]==runs[i+2][0] for i in range(len(runs)-2))
near=np.zeros(len(c),bool)
for t,a,b in trans:near[max(0,t-25):min(len(c),t+26)]=True
r['X_fraction_within_1s_transition']=float(np.mean(c[near]==3)) if near.any() else None
(ROOT/'reports/independent_audit/segmentation.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
