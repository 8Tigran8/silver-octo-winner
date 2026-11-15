"""Two clean-source CPU generations with frozen mixed-source weights, no GT."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from pff_reader import sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-dir',type=Path,required=True);a=p.parse_args()
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True):p.error('Commit exact candidate source/assets first')
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();bundle=ROOT/'assets/pff-mixed-candidate/bundle.json';b=json.loads(bundle.read_text());target=ROOT/'out/candidates/pff-mixed'
    if (target/'Ruses Cules.csv').exists():p.error('Do not overwrite a completed candidate reproduction')
    data=a.data_dir.resolve();files=[data/'tactical'/('BAR-ATH_tactical_atd.'+ext) for ext in ['xml','txt']]
    files+=[bundle,bundle.parent/b['model_path'],bundle.parent/b['metadata_path'],ROOT/'assets/v4/manifest.json']
    manifest=json.loads((ROOT/'assets/v4/manifest.json').read_text());files.extend(ROOT/'assets/v4'/f['path'] for f in manifest['files']);inputs={str(f):sha(f) for f in set(files)}
    folder=Path(tempfile.mkdtemp(prefix='mixed-clean-',dir=ROOT/'reports/pff_mixed'));archive=folder/'source.tar';subprocess.run(['git','archive','--format=tar','-o',str(archive),source],cwd=ROOT,check=True)
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONHASHSEED='0')
    inference_files=list((ROOT/'mtah').rglob('*.py'))+list((ROOT/'mtah').rglob('*.json'))+[ROOT/'tools/generate_mixed_candidate.py']
    inference_hashes={str(f.relative_to(ROOT)):sha(f) for f in inference_files}
    receipt={'status':'IN_PROGRESS','candidate':b['candidate'],'bundle_sha256':sha(bundle),'source_commit':source,
        'mode':'frozen_external_weights_CPU_inference','training_repeated':False,'target_quality_assessed':False,
        'platform':platform.platform(),'python':platform.python_version(),'input_sha256':inputs,'runs':[],
        'packages':{p:importlib.metadata.version(p) for p in ['numpy','scipy','scikit-learn','torch','cloudpickle','threadpoolctl']},
        'inference_source_sha256':inference_hashes}
    try:
        for number in [1,2]:
            output=folder/f'run-{number}.csv';started=time.perf_counter()
            with tempfile.TemporaryDirectory(prefix=f'run-{number}-',dir=folder) as temporary:
                clean=Path(temporary)
                with tarfile.open(archive) as tar:tar.extractall(clean,filter='data')
                with (folder/f'run-{number}.log').open('wb') as log:
                    subprocess.run([sys.executable,'tools/generate_mixed_candidate.py','--bundle',str(clean/'assets/pff-mixed-candidate/bundle.json'),'--data-dir',str(data),'--output',str(output)],cwd=clean,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                    subprocess.run([sys.executable,str(ROOT/'ref/official/evaluation/validate_submission.py'),str(output)],cwd=clean,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            receipt['runs'].append({'run':number,'seconds':round(time.perf_counter()-started,3),'csv_sha256':sha(output),'official_format':'PASS'});print('Mixed clean CPU run',number,'PASS',flush=True)
        if inputs!={str(f):sha(f) for f in set(files)}:raise ValueError('Input/model bytes changed during reproduction')
        if inference_hashes!={str(f.relative_to(ROOT)):sha(f) for f in inference_files}:raise ValueError('Inference source changed during reproduction')
        if receipt['runs'][0]['csv_sha256']!=receipt['runs'][1]['csv_sha256']:raise ValueError('Clean generations differ')
        peak=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss;receipt['peak_child_RSS_mb']=round(peak/(1024**2 if sys.platform=='darwin' else 1024),2)
        receipt.update(status='PASS: two identical frozen-weight CPU generations and official format',inputs_unchanged=True)
        target.mkdir(parents=True,exist_ok=True);shutil.copy2(folder/'run-2.csv',target/'Ruses Cules.csv');(target/'verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
    except Exception as error:receipt.update(status='FAIL',reason=str(error));raise
    finally:(folder/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print('Receipt',folder/'receipt.json',flush=True)


if __name__=='__main__':main()
