"""Prospective composition of separately measured training and gap mechanisms."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[key]='1'
import sys,json,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context
import numpy as np
import torch,cloudpickle
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
from spell_weighted_study import DOC,OUT,plan,check
from short_gap_study import diagnostics,scenarios
from pff_development import geometry,spans
from pff_reader import sha
from pff_mixed_score import decoder
from decoder_development import metric
from report_short_gap import summarize
from mtah import mixed_candidate,mixed_temporal,expanded_team,v4
from mtah.methods import boost
GAP=ROOT/'reports/short_gap_recovery'


def initialize():
    global ASSETS,STATE,EXPERT
    ASSETS=v4.load();bundle,_=mixed_candidate.load_bundle(ROOT/'assets/pff-mixed-candidate/bundle.json')
    meta=json.loads((OUT/'weighted/meta.json').read_text())
    if meta['model_sha256']!=sha(OUT/'weighted/model.pt'):raise ValueError('Changed weighted checkpoint')
    state=torch.load(OUT/'weighted/model.pt',map_location='cpu',weights_only=True)
    model=mixed_temporal.MixedTemporal(bundle['spec']);model.load_state_dict(state['state_dict']);model.eval()
    STATE=model,state['mean'].numpy(),state['scale'].numpy()
    meta=json.loads((GAP/'expert_meta.json').read_text())
    if meta['model_sha256']!=sha(GAP/'gap_expert.pkl'):raise ValueError('Changed expert')
    with (GAP/'gap_expert.pkl').open('rb') as f:EXPERT=cloudpickle.load(f)


def worker(game):
    if game not in plan()['development_games']:raise ValueError('Not development')
    data,receipt=geometry(game);parts=spans(data['period']);start=time.monotonic();rows=[]
    reference=ROOT/'reports/pff_access/references'/f'kloppy_{game}.npz'
    if sha(reference)!=receipt['reference_cache_sha256']:raise ValueError('Reference changed')
    baseline=json.loads((ROOT/'validation/observation_regression.json').read_text())['records']
    baseline={(r['game'],r['scenario'],r['reference']):r for r in baseline}
    for scenario,observed in scenarios(data):
        with np.load(GAP/'development'/f'{game}_{scenario}.npz') as z:cache=dict(z)
        if not np.array_equal(cache['ball'],observed['ball'],equal_nan=True):raise ValueError('Cache input mismatch')
        bd,bh,_,_=v4.components(observed,parts,ASSETS)
        with threadpool_limits(1):
            x,xs,_=expanded_team.feature_pair(observed,parts)
            small=EXPERT.predict_proba(cache['activity'])[:,1]
        activity=v4.neural_upsample(small,small,parts,len(x))[0]
        model,mean,scale=STATE;down=[((a+4)//5,(b+4)//5) for a,b in parts]
        nd,nh=mixed_temporal.probabilities(model,x[::5],xs[::5],down,mean,scale)
        nd,nh=v4.neural_upsample(nd,nh,parts,len(x))
        pd,ph=.5*bd+.5*nd,.5*bh+.5*nh
        missing=~cache['seen'];pd[missing]=.75*nd[missing]+.25*activity[missing]
        predictions={}
        for name in ['weighted_gap','weighted_short_gap']:
            params=decoder()
            if 'short' in name:params.update(min_team_s=.4,cost_ha=4.5)
            predictions[name]=boost.label(pd,ph,parts,len(pd),params)
        np.savez_compressed(OUT/f'combined_{game}_{scenario}_predictions.npz',**predictions)
        with np.load(reference) as z:
            for reference_name,key in [('canonical','gt'),('sdk_shadow','alternate')]:
                gt=z[key];base=baseline[game,scenario,reference_name]
                for name,pred in predictions.items():
                    score=metric(gt,pred)
                    rows.append(dict(game=game,scenario=scenario,reference=reference_name,candidate=name,**score,
                        spell_gain=score['spell_f1_mean']-base['spell_f1_mean'],macro_gain=score['macro_f1']-base['macro_f1'],
                        diagnostics=diagnostics(gt,pred,cache['seen'])))
    (OUT/f'combined_{game}_scores.json').write_text(json.dumps(rows,indent=2)+'\n')
    return game,rows,time.monotonic()-start


def main():
    check()
    for file,digest in json.loads((DOC/'combined_freeze.json').read_text())['files_sha256'].items():
        if sha(ROOT/file)!=digest:raise ValueError('Composition source/input changed')
    destination=OUT/'combined_results.json'
    if destination.exists():raise ValueError('Preserve completed outcomes')
    rows=[]
    with ProcessPoolExecutor(max_workers=2,mp_context=get_context('spawn'),initializer=initialize) as pool:
        for future in as_completed([pool.submit(worker,g) for g in plan()['development_games']]):
            game,rr,seconds=future.result();rows.extend(rr);print('Combined',game,round(seconds,1),flush=True)
    destination.write_text(json.dumps(dict(plan_sha256=sha(DOC/'combined_plan.json'),records=rows),indent=2)+'\n')
    controls=[r for r in json.loads((ROOT/'docs/experiments/short-gap-recovery/scores.json').read_text())['records'] if r['candidate']=='control']
    cfg=plan();cfg.update(mechanical_candidates=['control'],expert_candidates=['control','weighted_gap','weighted_short_gap'],scenarios=['natural_detected','label_blind_seed7'])
    result=summarize(controls+rows,cfg);result['plan_sha256']=sha(DOC/'combined_plan.json')
    result['scope']='Separately registered composition, repeated development only; no new holdout claim'
    (DOC/'combined_result.json').write_text(json.dumps(result,indent=2)+'\n')
    (DOC/'combined_scores.json').write_text(json.dumps(dict(records=rows),indent=2)+'\n');check()
    for row in result['summary']:
        if row['reference']=='canonical':print(row['candidate'],row['scenario'],row['mean_spell_gain'],row['positive_games'],row['worst_spell_gain'],row['gate_pass'])
    print(result['decision'],result['selected'])


if __name__=='__main__':main()
