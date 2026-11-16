"""Registered complementary-error follow-up; existing weights and thresholds only."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[key]='1'
import json
from pathlib import Path
import sys
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context
import numpy as np
import cloudpickle
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
from short_gap_study import DOC,OUT,plan,check,guard,references,diagnostics
from short_gap_methods import policy
from pff_mixed_score import decoder
from pff_reader import sha
from decoder_development import metric
from mtah import v4
from mtah.methods import boost
from report_short_gap import summarize


def worker(game):
    guard(game,'development_games');cfg=json.loads((DOC/'complement_plan.json').read_text())
    meta=json.loads((OUT/'expert_meta.json').read_text())
    if sha(OUT/'gap_expert.pkl')!=meta['model_sha256']:raise ValueError('Expert changed')
    with (OUT/'gap_expert.pkl').open('rb') as f:model=cloudpickle.load(f)
    receipt=json.loads((ROOT/'reports/pff_access'/f'kloppy_{game}_preflight.json').read_text());refs=references(game,receipt)
    baseline=json.loads((ROOT/'validation/observation_regression.json').read_text())['records']
    baseline={(r['game'],r['scenario'],r['reference']):r for r in baseline};rows=[]
    for scenario in plan()['scenarios']:
        with np.load(OUT/'development'/f'{game}_{scenario}.npz') as z:cache=dict(z)
        with threadpool_limits(1):small=model.predict_proba(cache['activity'])[:,1]
        expert=v4.neural_upsample(small,small,cache['segments'],len(cache['pd']))[0]
        predictions={}
        for name in cfg['candidates']:
            pd,ph,params=policy(name,cache,decoder(),expert)
            pred=boost.label(pd,ph,cache['segments'],len(pd),params);predictions[name]=pred
            for reference,gt in refs.items():
                score=metric(gt,pred);base=baseline[game,scenario,reference]
                rows.append(dict(game=game,scenario=scenario,candidate=name,reference=reference,**score,
                    spell_gain=score['spell_f1_mean']-base['spell_f1_mean'],macro_gain=score['macro_f1']-base['macro_f1'],
                    diagnostics=diagnostics(gt,pred,cache['seen'])))
        np.savez_compressed(OUT/f'complement_{game}_{scenario}_predictions.npz',**predictions)
    return game,rows


def main():
    check();cfg=json.loads((DOC/'complement_plan.json').read_text())
    frozen=json.loads((DOC/'complement_freeze.json').read_text())
    for file,digest in frozen['files_sha256'].items():
        if sha(ROOT/file)!=digest:raise ValueError('Follow-up changed')
    dest=OUT/'complement_results.json'
    if dest.exists():raise ValueError('Never overwrite completed outcomes')
    rows=[]
    with ProcessPoolExecutor(max_workers=2,mp_context=get_context('spawn')) as pool:
        for future in as_completed([pool.submit(worker,g) for g in plan()['development_games']]):
            game,rr=future.result();rows.extend(rr);print('Complement',game,'complete',flush=True)
    dest.write_text(json.dumps(dict(plan_sha256=sha(DOC/'complement_plan.json'),records=rows),indent=2)+'\n')
    all_rows=json.loads((DOC/'scores.json').read_text())['records']+rows
    old=plan();old['expert_candidates']+=cfg['candidates']
    result=summarize(all_rows,old);result['complement_plan_sha256']=sha(DOC/'complement_plan.json')
    (DOC/'complement_result.json').write_text(json.dumps(result,indent=2)+'\n')
    (DOC/'complement_scores.json').write_text(json.dumps(dict(records=rows),indent=2)+'\n')
    check()
    for row in result['summary']:
        if row['candidate'] in cfg['candidates'] and row['reference']=='canonical':
            print(row['candidate'],row['scenario'],row['mean_spell_gain'],row['positive_games'],row['gate_pass'])
    print(result['decision'],result['selected'])


if __name__=='__main__':main()
