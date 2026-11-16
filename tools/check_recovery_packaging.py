"""Require portable inference to reproduce every registered development prediction."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import sys,json,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
from mtah import recovery_candidate
from mtah.methods import boost
from short_gap_study import scenarios
from pff_development import geometry,spans
from pff_reader import sha


def initialize():
    global BUNDLE,STATE
    BUNDLE,STATE=recovery_candidate.load_bundle(ROOT/'assets/recovery-candidate/bundle.json')


def worker(game):
    plan=json.loads((ROOT/'docs/experiments/spell-weighted-finetune/plan.json').read_text())
    if game not in plan['development_games']:raise ValueError('Forbidden fixture')
    data,_=geometry(game);parts=spans(data['period']);rows=[];start=time.monotonic()
    for scenario,d in scenarios(data):
        pd,ph=recovery_candidate.probabilities(d,parts,STATE)
        pred=boost.label(pd,ph,parts,len(pd),BUNDLE['decoder'])
        source=ROOT/'reports/spell_weighted_finetune'/f'combined_{game}_{scenario}_predictions.npz'
        with np.load(source) as z:expected=z['weighted_short_gap']
        changed=int((pred!=expected).sum())
        if changed:raise ValueError(f'Packaged candidate changed {changed} predictions on {game}/{scenario}')
        rows.append(dict(game=game,scenario=scenario,differing_frames=changed,reference_prediction_sha256=sha(source)))
    return game,rows,time.monotonic()-start


def main():
    rows=[];games=json.loads((ROOT/'docs/experiments/spell-weighted-finetune/plan.json').read_text())['development_games']
    with ProcessPoolExecutor(max_workers=2,mp_context=get_context('spawn'),initializer=initialize) as pool:
        for future in as_completed([pool.submit(worker,g) for g in games]):
            game,rr,seconds=future.result();rows.extend(rr);print('Portable parity',game,round(seconds,1),flush=True)
    if len(rows)!=20:raise ValueError('Incomplete equivalence check')
    result=dict(status='PASS',scope='All ten development games, two scenarios; exact predictions, not another model selection',records=rows,
                bundle_sha256=sha(ROOT/'assets/recovery-candidate/bundle.json'))
    (ROOT/'docs/experiments/spell-weighted-finetune/portable_parity.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
