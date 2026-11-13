from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import numpy as np
from legacy_skillcorner_reader import read_match,period_spans
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context

def job(name):
 d=read_match(ROOT/'external/skillcorner_legacy'/name)
 np.savez_compressed(ROOT/'reports/v3_search/legacy'/(name+'.npz'),**d)
 valid=d['gt']<2;seen=np.isfinite(d['ball'][:,0])
 return {'match':name,'frames':len(valid),'reference_live_frames':int(valid.sum()),'reference_live_without_ball':int((valid&~seen).sum()),'period_spans':period_spans(d['period']),'dead_reference_available':False}
if __name__=='__main__':
 manifest=json.loads((ROOT/'tools/legacy_skillcorner_sources.json').read_text())
 from fetch_external import fetch
 for item in manifest['files']:fetch(item,ROOT/'external')
 (ROOT/'reports/v3_search/legacy').mkdir(parents=True,exist_ok=True)
 names=sorted({x['path'].split('/')[1] for x in manifest['files']})
 rows=[]
 with ProcessPoolExecutor(max_workers=2,mp_context=get_context('spawn')) as pool:
  for f in as_completed([pool.submit(job,n) for n in names]):r=f.result();rows.append(r);print(r['match'],'converted, no model scoring',flush=True)
 (ROOT/'reports/v3_search/legacy_eligibility.json').write_text(json.dumps({'reference':'Automatic known-team possession only, unknown ownership X, no D labels','records':rows},indent=2)+'\n')
