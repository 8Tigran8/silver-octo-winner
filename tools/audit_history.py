"""Bounded Git object/name audit; never print candidate secrets or raw match data."""
import subprocess,re,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
lines=subprocess.check_output(['git','rev-list','--objects','--all'],cwd=ROOT,text=True).splitlines();objects={};danger=[]
for line in lines:
 sha,_,name=line.partition(' ')
 if not name:continue
 objects[sha]=name
 if re.search(r'(_atd\.(xml|txt)|_homography\.csv|_smart_tagging\.xml|\.(mp4|mov|npz|npy|parquet)$|(^|/)(data|datasets|external|ref)/)',name):danger.append(name)
meta=subprocess.check_output(['git','cat-file','--batch-check=%(objectname) %(objecttype) %(objectsize)'],input='\n'.join(objects)+'\n',cwd=ROOT,text=True).splitlines();large=[];secrets=[];scanned=0
pattern=re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{30,}|AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_-]{30,}')
for line in meta:
 parts=line.split()
 if len(parts)!=3 or parts[1]!='blob':continue
 sha,typ,size=parts;size=int(size);name=objects[sha]
 if size>15*1024*1024:large.append({'path':name,'bytes':size})
 if size<2*1024*1024 and name.endswith(('.py','.md','.json','.txt','.yml','.yaml','.toml','.sh')):
  b=subprocess.check_output(['git','cat-file','blob',sha],cwd=ROOT);scanned+=1
  if pattern.search(b):secrets.append(name)
result={'scope':'All reachable refs: protected filename patterns, blobs >15MiB, recognized credential patterns in text blobs <2MiB. This is not proof against arbitrary disguised data or all secret formats.','objects':len(objects),'text_blobs_scanned':scanned,'protected_name_hits':sorted(set(danger)),'large_blobs':large,'recognized_secret_hits':sorted(set(secrets))}
(ROOT/'reports/independent_audit/history_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if danger or secrets:raise SystemExit(1)
