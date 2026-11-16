"""Exact portable-vs-research parity on40allowed development inputs; no GT scoring."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
import sys,json,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
from mtah import final_candidate,degrade,expanded_team
from mtah.methods import boost
from pff_development import geometry,spans
from pff_reader import sha
from team_gap_specialist import install_read_guard,guard,plan
DOC=ROOT/'docs/releases/uniform-temporal-focus'

def initialize():
    global BUNDLE,STATE
    install_read_guard();torch.set_num_threads(1);torch.use_deterministic_algorithms(True);BUNDLE,STATE=final_candidate.load_bundle(ROOT/'assets/final-candidate/bundle.json')

def worker(game):
    guard(game,'development');data,receipt=geometry(game);parts=spans(data['period']);rows=[];started=time.monotonic()
    for scenario,seed in [('natural_detected',None),('label_blind_seed7',7),('novel_23',23^int(game)),('novel_101',101^int(game))]:
        if seed is None:d=data
        else:
            dummy=np.where(data['period']>0,0,3).astype(np.int8);d=degrade.degrade(dict(data,gt=dummy),seed=seed,params=expanded_team.BLIND);d.pop('gt')
        pd,ph=final_candidate.probabilities(d,parts,STATE);pred=boost.label(pd,ph,parts,len(pd),BUNDLE['decoder'])
        path=ROOT/'reports/dead_gap_focus'/(f'{game}_{scenario}_predictions.npz' if not scenario.startswith('novel') else f'novel_stress/{game}_{scenario}.npz')
        with np.load(path) as z:expected=z['uniform_temporal_focus']
        mismatch=int(np.count_nonzero(pred!=expected));assert mismatch==0,(game,scenario,mismatch)
        rows.append(dict(game=game,scenario=scenario,frames=len(pred),mismatches=mismatch,geometry_sha256=receipt['geometry_cache_sha256'],research_predictions_sha256=sha(path)))
    return game,rows,time.monotonic()-started

if __name__=='__main__':
    install_read_guard();rows=[]
    with ProcessPoolExecutor(max_workers=2,mp_context=get_context('spawn'),initializer=initialize) as pool:
        for f in as_completed([pool.submit(worker,g) for g in plan()['development_games']]):
            g,r,t=f.result();rows+=r;print('Parity',g,round(t,1),'seconds',flush=True)
    assert len(rows)==40
    (DOC/'parity.json').write_text(json.dumps(dict(status='PASS',exact40=True,reference_labels_read=False,heldout18_read=False,records=rows),indent=2)+'\n')
