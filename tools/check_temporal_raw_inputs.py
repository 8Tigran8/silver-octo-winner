"""Verify neural training caches exactly match generation from raw public data."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context
import json
from temporal_trial import OUT,ROOT,plan,folder,observed_periods,np,expanded,degrade


def job(game):
    family,name=game.split('/')
    if family!='original':return game,True
    with np.load(ROOT/'external/conv'/(name+'.npz')) as z:clean={k:z[k] for k in z.files}
    segments=observed_periods(clean)
    for scenario,(seed,params,_) in expanded.ORIGINAL_SCENARIOS.items():
        data=clean if seed is None else degrade.degrade(clean,seed=seed,params=params)
        x,xs,cols=expanded.feature_pair(data,segments);f=folder(game,scenario)
        for field,values in [('x',x),('xs',xs)]:
            saved=np.load(f/(field+'.npy'),mmap_mode='r')
            if not np.array_equal(saved,values[::5].astype(np.float32),equal_nan=True):raise ValueError('Raw preparation differs from learned cache')
    return game,True


if __name__=='__main__':
    names=[g for g in plan()['training_games'] if g.startswith('original/')];result={}
    with ProcessPoolExecutor(max_workers=3,mp_context=get_context('spawn')) as pool:
        for f in as_completed([pool.submit(job,g) for g in names]):g,ok=f.result();result[g]=ok;print(g,'raw input equality PASS',flush=True)
    (OUT/'raw_input_equality.json').write_text(json.dumps({'original_games':result,'scenarios_per_game':3,'exact_float32_features_and_swaps':True,'note':'SkillCorner preparation already reads pinned raw JSON/CSV directly'},indent=2)+'\n')
