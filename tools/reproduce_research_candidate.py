"""Two clean local source generations with frozen external-trained research weights."""
import argparse,hashlib,json,os,subprocess,sys,tarfile,tempfile,time,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--bundle',type=Path,default=ROOT/'assets/research-v5/bundle.json');p.add_argument('--name',default='reliability');a=p.parse_args()
    if not a.name or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for c in a.name):p.error('Use a simple local candidate name')
    target=ROOT/'out/candidates'/a.name
    if (target/'Ruses Cules.csv').exists():p.error('Preserve existing candidate; choose another name')
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True):p.error('Commit local source before clean generation')
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();bundle=a.bundle.resolve();bundle_relative=bundle.relative_to(ROOT);b=json.loads(bundle.read_text());data=a.data_dir.resolve()
    files=[data/'tactical'/('BAR-ATH_tactical_atd.'+ext) for ext in ['xml','txt']]+[bundle]+[bundle.parent/m['path'] for m in b['models']]
    inputs={str(path):sha(path) for path in files};folder=Path(tempfile.mkdtemp(prefix='candidate-reproduction-',dir=ROOT/'reports/reliability'));archive=folder/'source.tar'
    subprocess.run(['git','archive','--format=tar','-o',str(archive),source],cwd=ROOT,check=True)
    receipt={'candidate':b['candidate'],'source_commit':source,'mode':'frozen_public_external_weights_CPU_inference','training_repeated':False,'target_quality_assessed':False,'input_sha256':inputs,'runs':[],'status':'IN_PROGRESS'}
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONHASHSEED='0')
    try:
        for number in [1,2]:
            start=time.perf_counter();output=folder/f'run-{number}.csv';log=folder/f'run-{number}.log'
            with tempfile.TemporaryDirectory(prefix=f'clean-{number}-',dir=folder) as temporary:
                clean=Path(temporary)
                with tarfile.open(archive) as source_tar:source_tar.extractall(clean,filter='data')
                with log.open('wb') as stream:
                    subprocess.run([sys.executable,'tools/generate_research_candidate.py','--bundle',str(clean/bundle_relative),'--data-dir',str(data),'--output',str(output)],cwd=clean,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
                    subprocess.run([sys.executable,str(ROOT/'ref/official/evaluation/validate_submission.py'),str(output)],cwd=clean,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
            receipt['runs'].append({'run':number,'seconds':round(time.perf_counter()-start,3),'csv_sha256':sha(output),'official_format':'PASS'});print('Research clean run',number,'format PASS',flush=True)
        if inputs!={str(path):sha(path) for path in files}:raise ValueError('Input hashes changed')
        if receipt['runs'][0]['csv_sha256']!=receipt['runs'][1]['csv_sha256']:raise ValueError('Clean candidate CSVs differ')
        receipt['status']='PASS: clean frozen-weight CPU generation and official format; not final quality confirmation';receipt['inputs_unchanged']=True
        target.mkdir(parents=True,exist_ok=True);shutil.copy2(folder/'run-2.csv',target/'Ruses Cules.csv');(target/'verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
        (target/'STATUS.txt').write_text('Local research candidate only. Not submitted. Verified v4 remains in out/submission. Independent confirmation and inherited-lineage audit pending. User hold active.\n')
    except Exception as error:
        receipt['status']='FAIL';receipt['reason']=str(error);raise
    finally:
        (folder/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print('Receipt',folder/'receipt.json',flush=True)


if __name__=='__main__':main()
