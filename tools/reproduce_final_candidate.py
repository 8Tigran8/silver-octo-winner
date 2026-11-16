"""Fresh-source reproduction on macOS and offline Linux, without target scoring."""
import argparse,hashlib,importlib.metadata,json,os,platform,shutil,subprocess,sys,tarfile,tempfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-dir',type=Path,required=True);a=p.parse_args()
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True):p.error('Commit exact source and packaged models first')
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    work=ROOT/'reports/final_release'/source;work.mkdir(parents=True,exist_ok=False)
    target=ROOT/'out/final-ready'
    if (target/'Ruses Cules.csv').exists():p.error('Do not overwrite a completed release')
    archive=work/'source.tar';subprocess.run(['git','archive','--format=tar','-o',str(archive),source],cwd=ROOT,check=True)
    data=a.data_dir.resolve();inputs={ext:data/'tactical'/('BAR-ATH_tactical_atd.'+ext) for ext in ['xml','txt']}
    input_hashes={str(f.relative_to(data)):sha(f) for f in inputs.values()}
    code=list((ROOT/'mtah').rglob('*.py'))+list((ROOT/'mtah').rglob('*.json'))+[ROOT/'tools/generate_final_candidate.py']
    models=[f for folder in ['assets/v4','assets/pff-mixed-candidate','assets/recovery-candidate','assets/final-candidate'] for f in (ROOT/folder).rglob('*') if f.is_file()]
    source_hashes={str(f.relative_to(ROOT)):sha(f) for f in code};model_hashes={str(f.relative_to(ROOT)):sha(f) for f in models}
    official=work/'official';(official/'evaluation').mkdir(parents=True)
    for name in ['evaluation/validate_submission.py','evaluation/possession_metrics.py','games.py']:
        shutil.copy2(ROOT/'ref/official'/name,official/name)
    receipt=dict(status='IN_PROGRESS',candidate='uniform_temporal_focus',source_commit=source,
        input_sha256=input_hashes,inference_source_sha256=source_hashes,model_file_sha256=model_hashes,
        official_sha256={name:sha(official/name) for name in ['evaluation/validate_submission.py','evaluation/possession_metrics.py','games.py']},
        host_platform=platform.platform(),python=platform.python_version(),
        packages={name:importlib.metadata.version(name) for name in ['numpy','scipy','scikit-learn','torch','cloudpickle','threadpoolctl']},
        target_quality_assessed=False,training_repeated=False,one_physical_host=True,runs=[])
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONHASHSEED='0')
    try:
        for number in [1,2]:
            started=time.monotonic();output=work/f'macos-{number}.csv'
            with tempfile.TemporaryDirectory(prefix='macos-source-',dir=work) as temporary:
                clean=Path(temporary)
                with tarfile.open(archive) as tar:tar.extractall(clean,filter='data')
                with (work/f'macos-{number}.log').open('wb') as log:
                    subprocess.run([sys.executable,'tools/generate_final_candidate.py','--bundle','assets/final-candidate/bundle.json','--data-dir',str(data),'--output',str(output)],cwd=clean,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                    subprocess.run([sys.executable,str(official/'evaluation/validate_submission.py'),str(output)],cwd=clean,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            receipt['runs'].append(dict(environment='macOS ARM',run=number,seconds=time.monotonic()-started,sha256=sha(output),official_validator='PASS'))
            print('macOS fresh source run',number,'PASS',flush=True)
        for arch,image in [('arm64','mtah-audit-runtime:py312'),('amd64','mtah-audit-runtime:amd64')]:
            # Resolve the verified local ID: this daemon lists tags correctly
            # but tag-based inspect intermittently reports "No such image".
            inventory=[json.loads(line) for line in subprocess.check_output(['docker','image','ls','--no-trunc','--format','{{json .}}'],text=True).splitlines()]
            matches=[r for r in inventory if r['Repository']+':'+r['Tag']==image]
            if len(matches)!=1:raise ValueError('Expected one installed runtime image: '+image)
            image_info=json.loads(subprocess.check_output(['docker','image','inspect',matches[0]['ID']],text=True))[0]
            if image_info['Architecture']!=arch or image not in image_info['RepoTags']:raise ValueError('Runtime identity/architecture differs')
            image_id=image_info['Id']
            clean=work/('linux-source-'+arch);clean.mkdir()
            with tarfile.open(archive) as tar:tar.extractall(clean,filter='data')
            output=work/('linux-output-'+arch);output.mkdir();started=time.monotonic()
            script="import shutil,subprocess,sys; shutil.copytree('/source','/tmp/code'); subprocess.run([sys.executable,'tools/generate_final_candidate.py','--bundle','assets/final-candidate/bundle.json','--data-dir','/data','--output','/output/Ruses Cules.csv'],cwd='/tmp/code',check=True); subprocess.run([sys.executable,'/official/evaluation/validate_submission.py','/output/Ruses Cules.csv'],check=True)"
            command=['docker','run','--rm','--pull','never','--network','none','--platform','linux/'+arch,
                     '-v',str(clean)+':/source:ro','-v',str(output)+':/output','-v',str(official)+':/official:ro']
            for ext,file in inputs.items():command+=['-v',str(file)+':/data/tactical/BAR-ATH_tactical_atd.'+ext+':ro']
            for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','PYTHONHASHSEED']:command+=['-e',key+'='+('0' if key=='PYTHONHASHSEED' else '1')]
            command += [image_id,'python','-c',script]
            with (work/('linux-'+arch+'.log')).open('wb') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
            receipt['runs'].append(dict(environment='Linux '+arch+(' emulated' if arch=='amd64' else ''),run=1,seconds=time.monotonic()-started,
                sha256=sha(output/'Ruses Cules.csv'),official_validator='PASS',network='none',image_id=image_id))
            print('Linux',arch,'fresh source PASS',flush=True)
        if len({r['sha256'] for r in receipt['runs']})!=1:raise ValueError('Cross-environment CSV bytes differ')
        if input_hashes!={str(f.relative_to(data)):sha(f) for f in inputs.values()}:raise ValueError('Original data changed')
        if source_hashes!={str(f.relative_to(ROOT)):sha(f) for f in code} or model_hashes!={str(f.relative_to(ROOT)):sha(f) for f in models}:raise ValueError('Source or model bytes changed')
        receipt.update(status='PASS: four byte-identical clean-source generations with official validators',inputs_unchanged=True)
        target.mkdir(parents=True,exist_ok=True);shutil.copy2(work/'macos-2.csv',target/'Ruses Cules.csv')
        (target/'reproduction.json').write_text(json.dumps(receipt,indent=2)+'\n')
    except Exception as error:
        receipt.update(status='FAIL',reason=str(error));raise
    finally:
        (work/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print('Receipt:',work/'receipt.json',flush=True)


if __name__=='__main__':main()
