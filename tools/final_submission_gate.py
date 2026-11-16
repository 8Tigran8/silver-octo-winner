"""Fail-closed exact artifact/source/official-validator gate; never scores target GT."""
import argparse,csv,hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--csv',type=Path,required=True);p.add_argument('--manifest',type=Path,default=ROOT/'submission_manifest.json');a=p.parse_args();m=json.loads(a.manifest.read_text());expected=m['runs'][0]['sha256']
 if sha(a.csv)!=expected:raise ValueError('CSV differs from audited output')
 for name,digest in m['inference_source_sha256'].items():
  if sha(ROOT/name)!=digest:raise ValueError('Inference source changed: '+name)
 for name,digest in m['model_file_sha256'].items():
  if sha(ROOT/name)!=digest:raise ValueError('Model/config changed: '+name)
 with a.csv.open(encoding='utf-8',newline='') as f:
  reader=csv.reader(f)
  if next(reader)!=['frame','label']:raise ValueError('Nonexact header')
  count=0
  for i,row in enumerate(reader):
   if len(row)!=2 or row[0]!=str(i) or row[1] not in ['H','A','D','X']:raise ValueError('Invalid row contract')
   count+=1
  if count!=199319:raise ValueError('Wrong row count')
 refs=json.loads((ROOT/'validation/official_sources.json').read_text())['sha256']
 for name in ['evaluation/possession_metrics.py','evaluation/validate_submission.py','games.py']:
  if sha(ROOT/'ref/official'/name)!=refs[name]:raise ValueError('Official source differs')
 subprocess.run([sys.executable,str(ROOT/'ref/official/evaluation/validate_submission.py'),str(a.csv.resolve())],check=True)
 print('PASS: exact audited CSV, source, weights/config and official validator; quality/eligibility/publication are separate.')
if __name__=='__main__':main()
