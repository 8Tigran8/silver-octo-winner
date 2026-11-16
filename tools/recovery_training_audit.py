"""Descriptive current-model training scores; never fit/select or claim fresh validation."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[key]='1'
import argparse,json,sys,time,subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
from recovery_opened_audit import DOC,check,now
from pff_reader import sha
OUT=ROOT/'reports/recovery_training_audit'

def ids():return [g['game_id'] for g in json.loads((ROOT/'docs/experiments/more-data/pff_protocol.json').read_text())['training']]
def freeze():
    check()
    if OUT.exists() or (DOC/'training_freeze.json').exists():raise ValueError('Already registered')
    files=[Path(__file__),DOC/'training_plan.json',DOC/'freeze.json',ROOT/'reports/training_gap_audit/results.json']
    for g in ids():
        for name in ['external/pff/converted/'+g+'_geometry.npz','reports/pff_access/kloppy_'+g+'_preflight.json','reports/pff_access/references/kloppy_'+g+'.npz']:files.append(ROOT/name)
    (DOC/'training_freeze.json').write_text(json.dumps(dict(frozen_utc=now(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),files_sha256={str(p.relative_to(ROOT)):sha(p) for p in files}),indent=2)+'\n')
    print('36-training-match natural audit frozen',flush=True)
def check_train():
    check()
    for name,h in json.loads((DOC/'training_freeze.json').read_text())['files_sha256'].items():
        if sha(ROOT/name)!=h:raise ValueError('Training audit input changed: '+name)
def initialize():
    global BUNDLE,STATE
    import torch
    from mtah import recovery_candidate
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    BUNDLE,STATE=recovery_candidate.load_bundle(ROOT/'assets/recovery-candidate/bundle.json')
def worker(game):
    from pff_development import geometry,spans
    from mtah import recovery_candidate
    from mtah.methods import boost
    from decoder_development import metric
    start=time.monotonic();data,receipt=geometry(game);parts=spans(data['period'])
    pd,ph=recovery_candidate.probabilities(data,parts,STATE);pred=boost.label(pd,ph,parts,len(pd),BUNDLE['decoder'])
    ref=ROOT/'reports/pff_access/references'/('kloppy_'+game+'.npz')
    if sha(ref)!=receipt['reference_cache_sha256']:raise ValueError('Reference changed')
    with np.load(ref) as z:gt=z['gt']
    result=dict(game=game,scope='PFF training in-sample, natural, current frozen model',**metric(gt,pred),geometry_sha256=receipt['geometry_cache_sha256'],reference_sha256=sha(ref),seconds=time.monotonic()-start)
    (OUT/(game+'.json')).write_text(json.dumps(result,indent=2)+'\n');print('Training',game,round(result['seconds'],1),'seconds',flush=True);return result
def run(resume=False):
    check_train();OUT.mkdir(exist_ok=resume);rows=[];pending=[]
    for g in ids():
        p=OUT/(g+'.json')
        if p.exists():rows.append(json.loads(p.read_text()))
        else:pending.append(g)
    with ProcessPoolExecutor(max_workers=2,mp_context=get_context('spawn'),initializer=initialize) as pool:
        for f in as_completed([pool.submit(worker,g) for g in pending]):rows.append(f.result())
    assert len(rows)==36 and {r['game'] for r in rows}==set(ids())
    check_train();previous=json.loads((ROOT/'reports/training_gap_audit/results.json').read_text())['records'];old={r['game']:r for r in previous}
    gains=[r['spell_f1_mean']-old[r['game']]['spell_f1_mean'] for r in rows]
    result=dict(scope='Descriptive in-sample train versus reused-development and reopened audit comparison; not a generalization bound',records=rows,
        mean_spell=float(np.mean([r['spell_f1_mean'] for r in rows])),mean_macro=float(np.mean([r['macro_f1'] for r in rows])),mean_gain_vs_parent=float(np.mean(gains)),wins_vs_parent=sum(g>0 for g in gains),
        training_freeze_sha256=sha(DOC/'training_freeze.json'))
    (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n');(DOC/'training_results.json').write_text(json.dumps(result,indent=2)+'\n');print({k:v for k,v in result.items() if k!='records'},flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run']);p.add_argument('--resume',action='store_true');a=p.parse_args();freeze() if a.action=='freeze' else run(a.resume)
