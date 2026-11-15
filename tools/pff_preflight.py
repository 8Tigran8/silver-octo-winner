"""Validate PFF train/dev adapter before any quality scores or reserve access."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
from multiprocessing import get_context
import os
from pathlib import Path
import sys
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from pff_reader import load_geometry,reference,tracking_reference,sha
OUT=ROOT/'reports/pff_access'


def convert(game,policy='kloppy'):
    protocol=json.loads((ROOT/'docs/experiments/more-data/pff_protocol.json').read_text())
    allowed={g['game_id'] for part in ['training','development'] for g in protocol[part]}
    if game not in allowed:raise ValueError('Reserve/bridge data are blocked from parser debugging')
    base=ROOT/'external/pff';dest=base/'converted';dest.mkdir(exist_ok=True)
    prefix='kloppy_' if policy=='kloppy' else ''
    info_path=OUT/(prefix+game+'_preflight.json')
    if info_path.exists():raise ValueError('Do not overwrite completed preflight')
    tracking=base/'Tracking Data'/(game+'.jsonl.bz2') if (base/'Tracking Data'/(game+'.jsonl.bz2')).exists() else base/'Tracking'/(game+'.jsonl.bz2')
    data,geometry=load_geometry(tracking,base/'Metadata'/(game+'.json'),base/'Rosters'/(game+'.json'))
    if policy=='kloppy':
        gt,alt,refs=tracking_reference(tracking,data['time'],data['period'],geometry['fps']);alternate={'policy':'Pinned kloppy last-row shadow','known_fraction':refs['sdk_shadow_known_fraction']}
    else:
        gt,refs=reference(base/'Event Data'/(game+'.json'),data['time'],data['period'])
        alt,alternate=reference(base/'Event Data'/(game+'.json'),data['time'],data['period'],mode='first_control')
    np.savez_compressed(dest/(game+'_geometry.npz'),**data)
    (OUT/'references').mkdir(exist_ok=True)
    np.savez(OUT/'references'/(prefix+game+'.npz'),gt=gt,alternate=alt)
    checks={'source_clock':geometry['source_clock_max_error_seconds']<=.002,
            'known_reference_coverage':refs['known_fraction']>=.99,
            'all_three_classes':all(refs['class_frames'][c]>0 for c in ['H','A','D']),
            'geometry_reference_separation':'gt' not in data,
            'independent_periods':np.any(data['period']==1) and np.any(data['period']==2)}
    result={'game':game,'scope':'Source/adapter train-dev preflight only; no model quality scores; final remains unopened.',
            'status':'PASS' if all(checks.values()) else 'FAIL: full-task source preflight',
            'checks':{k:bool(v) for k,v in checks.items()},'geometry':geometry,'reference':refs,'alternate_reference':alternate,
            'reference_variant_disagreement_fraction':float(np.mean(gt[data['period']>0]!=alt[data['period']>0])),
            'reader_sha256':sha(ROOT/'tools/pff_reader.py'),'geometry_cache_sha256':sha(dest/(game+'_geometry.npz')),
            'reference_cache_sha256':sha(OUT/'references'/(prefix+game+'.npz'))}
    info_path.write_text(json.dumps(result,indent=2)+'\n')
    return {'game':game,'status':result['status'],'coverage':refs['known_fraction'],'classes':refs['class_frames'],'observed_ball':geometry['visible_ball_fraction_in_periods']}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--games',nargs='+',default=['3816','3813','3821']);p.add_argument('--jobs',type=int,choices=[1,2],default=2);p.add_argument('--policy',choices=['kloppy','sequence_prototype'],default='kloppy');a=p.parse_args()
    with ProcessPoolExecutor(max_workers=a.jobs,mp_context=get_context('spawn')) as pool:
        for f in as_completed([pool.submit(convert,g,a.policy) for g in a.games]):print(json.dumps(f.result()),flush=True)
