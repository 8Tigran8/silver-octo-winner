"""Fresh committed-source inference in separately installed environment; no GT."""
import argparse,json,hashlib,subprocess,tarfile,os,time,platform
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--python',required=True);p.add_argument('--data-dir',required=True);p.add_argument('--out-dir',type=Path,default=ROOT/'reports/independent_audit');a=p.parse_args();out=a.out_dir.resolve();out.mkdir(parents=True,exist_ok=True);commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();dest=out/'clean-source';dest.mkdir(exist_ok=False);archive=out/'audited_source.tar';subprocess.run(['git','archive','-o',str(archive),commit],cwd=ROOT,check=True)
 with tarfile.open(archive) as t:t.extractall(dest,filter='data')
 data=Path(a.data_dir).resolve();env={'PATH':os.environ.get('PATH','/usr/bin:/bin'),'PYTHONHASHSEED':'0','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1'}
 inputs={f'tactical/BAR-ATH_tactical_atd.{ext}':sha(data/'tactical'/f'BAR-ATH_tactical_atd.{ext}') for ext in ['xml','txt']};runs=[]
 for i in [1,2]:
  target=out/f'audited-run-{i}.csv';start=time.monotonic()
  with (out/f'clean-run-{i}.log').open('w') as log:
   subprocess.run([a.python,'tools/generate_mixed_candidate.py','--bundle','assets/pff-mixed-candidate/bundle.json','--data-dir',str(data),'--output',str(target)],cwd=dest,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
   subprocess.run([a.python,str(ROOT/'reports/independent_audit/official/evaluation/validate_submission.py'),str(target)],cwd=dest,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
  runs.append({'run':i,'seconds':time.monotonic()-start,'sha256':sha(target),'official_validator':'PASS'});print(runs[-1],flush=True)
 assert runs[0]['sha256']==runs[1]['sha256'];assert inputs=={k:sha(data/k) for k in inputs}
 packages=subprocess.check_output([a.python,'-m','pip','freeze'],text=True)
 original=ROOT/'reports/independent_audit/pre_audit.csv'
 receipt={'source_commit':commit,'model':'boundary102 weights; audited preprocessing/stack period reset and observation integrity','python':subprocess.check_output([a.python,'--version'],text=True).strip(),'platform':platform.platform(),'dependencies':packages.splitlines(),'input_sha256':inputs,'runs':runs,'original_csv_sha256':sha(original),'output_changed_from_original':runs[0]['sha256']!=sha(original),'command':'python tools/generate_mixed_candidate.py --bundle assets/pff-mixed-candidate/bundle.json --data-dir /path/to/data --output "Ruses Cules.csv"','environment':'Separately installed pinned venv, fresh git archive, minimal env, no copied cache. Second run exercises content-keyed cache.','public_diagnostic_used_for_selection':False}
 (out/'reproduction.json').write_text(json.dumps(receipt,indent=2)+'\n');print('Clean environment reproduction PASS',flush=True)
if __name__=='__main__':main()
