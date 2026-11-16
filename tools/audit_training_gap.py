"""All 36 PFF training games, frozen current model, descriptive optimism audit."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[key]='1'
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
from multiprocessing import get_context
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
from pff_unseen_audit import check_freeze
from pff_reader import sha


def initialize():
    global ASSETS,STATE
    from mtah import v4,mixed_candidate
    ASSETS=v4.load();_,STATE=mixed_candidate.load_bundle(ROOT/'assets/pff-mixed-candidate/bundle.json')


def evaluate(game):
    import numpy as np
    from mtah import mixed_candidate
    from mtah.methods import boost
    from pff_development import geometry,spans
    from pff_mixed_score import decoder
    from decoder_development import metric
    start=time.monotonic();data,receipt=geometry(game)
    pd,ph=mixed_candidate.probabilities(data,spans(data['period']),ASSETS,STATE)
    pred=boost.label(pd,ph,spans(data['period']),len(pd),decoder())
    reference=ROOT/'reports/pff_access/references'/f'kloppy_{game}.npz'
    with np.load(reference) as archive:gt=archive['gt']
    return dict(game=game,scenario='natural_detected',scope='in-sample PFF training',**metric(gt,pred),seconds=time.monotonic()-start,
                geometry_sha256=receipt['geometry_cache_sha256'],reference_sha256=sha(reference))


def main():
    check_freeze()
    ids=[g['game_id'] for g in json.loads((ROOT/'docs/experiments/more-data/pff_protocol.json').read_text())['training']]
    if len(ids)!=36 or len(set(ids))!=36:raise ValueError('Unexpected training registry')
    out=ROOT/'reports/training_gap_audit';out.mkdir(exist_ok=False);rows=[]
    with ProcessPoolExecutor(max_workers=2,mp_context=get_context('spawn'),initializer=initialize) as pool:
        futures={pool.submit(evaluate,g):g for g in ids}
        for future in as_completed(futures):
            result=future.result();rows.append(result);(out/(result['game']+'.json')).write_text(json.dumps(result,indent=2)+'\n')
            print('Training audit',result['game'],len(rows),'/36',round(result['seconds'],1),'seconds',flush=True)
    check_freeze()
    rows.sort(key=lambda r:ids.index(r['game']))
    (out/'results.json').write_text(json.dumps(dict(plan_sha256=sha(ROOT/'docs/experiments/unseen-bridge-audit/training_gap_plan.json'),
        freeze_sha256=sha(ROOT/'docs/experiments/unseen-bridge-audit/freeze.json'),records=rows),indent=2)+'\n')


if __name__=='__main__':main()
