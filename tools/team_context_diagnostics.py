import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
import numpy as np
from mtah.methods import boost,boost_train
from quality_audit import official
from threadpoolctl import threadpool_limits
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context
FOLDER=ROOT/'reports/v2_search'
def job(item):
 name,cohort,scenario=item
 with np.load(FOLDER/cohort/f'{name}__{scenario}__probs.npz') as z: pd,ph,gt=z['pd'],z['ph'],z['gt']
 p=dict(boost_train.decoder_params(),**json.loads((ROOT/'mtah/methods/v2_params.json').read_text())['decoder'])
 segs=official.valid_runs(gt);rows=[]
 with threadpool_limits(1):
  for tag,qd,qh in [('v2',pd,ph),('ORACLE_dead_probability',np.where(gt==2,.9999,.0001),ph),('ORACLE_team_probability',pd,np.where(gt==0,.9999,.0001)),('ORACLE_both',np.where(gt==2,.9999,.0001),np.where(gt==0,.9999,.0001))]:
   pred=boost.label(qd,qh,segs,len(gt),p);r=official.evaluate(gt,pred,bootstrap_n=0)
   rows.append({'match':name,'cohort':cohort,'scenario':scenario,'method':tag,'spell':r['spell_f1_mean'],'macro':r['macro_f1'],'n_gt':r['n_gt_spells'],'n_pred':r['n_pred_spells'],'per_class':r['per_class'],'false_dead_fraction':float(((pred==2)&(gt<2)).sum()/max(1,(gt<2).sum())),'missed_dead_fraction':float(((pred<2)&(gt==2)).sum()/max(1,(gt==2).sum()))})
 return rows
if __name__=='__main__':
 plan=json.loads((FOLDER/'plan.json').read_text());known=json.loads((ROOT/'reports/quality_audit/380987b3029c/protocol.json').read_text())['matches']
 jobs=[(n,'known',s) for n in known for s in ['matched_seed0','label_blind_seed1']]+[(n,'skillcorner','natural_detected') for n in plan['development']]
 rows=[]
 with ProcessPoolExecutor(max_workers=4,mp_context=get_context('spawn')) as pool:
  for future in as_completed([pool.submit(job,j) for j in jobs]): rows.extend(future.result())
 (ROOT/'reports/v3_search/diagnostic.json').write_text(json.dumps({'kind':'Oracle probabilities on external development data only; not attainable model scores','records':rows},indent=2)+'\n')
 for cohort,scenario in [('known','matched_seed0'),('known','label_blind_seed1'),('skillcorner','natural_detected')]:
  for method in ['v2','ORACLE_dead_probability','ORACLE_team_probability','ORACLE_both']:
   vals=[r for r in rows if r['cohort']==cohort and r['scenario']==scenario and r['method']==method]
   print(cohort,scenario,method,round(float(np.mean([r['spell'] for r in vals])),4),'false_dead',round(float(np.mean([r['false_dead_fraction'] for r in vals])),3),'miss_dead',round(float(np.mean([r['missed_dead_fraction'] for r in vals])),3),flush=True)
