"""Separately registered contact-emission hypothesis on development only."""
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
from short_gap_study import DOC,OUT,plan,check,guard,references,diagnostics,scenarios
from short_gap_methods import policy
from visible_contact import contact_cue,contact_probabilities
from pff_development import geometry,spans
from pff_mixed_score import decoder
from pff_reader import sha
from decoder_development import metric
from mtah import v4
from mtah.methods import boost
from report_short_gap import summarize


def worker(game):
    guard(game,'development_games');cfg=json.loads((DOC/'contact_plan.json').read_text())
    meta=json.loads((OUT/'expert_meta.json').read_text())
    if sha(OUT/'gap_expert.pkl')!=meta['model_sha256']:raise ValueError('Expert changed')
    with (OUT/'gap_expert.pkl').open('rb') as f:model=cloudpickle.load(f)
    data,receipt=geometry(game);parts=spans(data['period']);refs=references(game,receipt)
    baseline=json.loads((ROOT/'validation/observation_regression.json').read_text())['records']
    baseline={(r['game'],r['scenario'],r['reference']):r for r in baseline};rows=[]
    for scenario,observed in scenarios(data):
        with np.load(OUT/'development'/f'{game}_{scenario}.npz') as z:cache=dict(z)
        if not np.array_equal(cache['ball'],observed['ball'],equal_nan=True):raise ValueError('Observed geometry mismatch')
        cue=contact_cue(observed,parts)
        corrected=contact_probabilities(cache['ph'],cache['pd'],cue)
        with threadpool_limits(1):small=model.predict_proba(cache['activity'])[:,1]
        expert=v4.neural_upsample(small,small,parts,len(cache['pd']))[0];predictions={}
        for name in cfg['candidates']:
            pd,_,params=policy(name,cache,decoder(),expert)
            pred=boost.label(pd,corrected,parts,len(pd),params);predictions[name]=pred
            for reference,gt in refs.items():
                score=metric(gt,pred);base=baseline[game,scenario,reference]
                rows.append(dict(game=game,scenario=scenario,candidate=name,reference=reference,**score,
                    spell_gain=score['spell_f1_mean']-base['spell_f1_mean'],macro_gain=score['macro_f1']-base['macro_f1'],
                    contact_frames=int((cue!=0).sum()),corrected_team_frames=int((corrected!=cache['ph']).sum()),
                    diagnostics=diagnostics(gt,pred,cache['seen'])))
        np.savez_compressed(OUT/f'contact_{game}_{scenario}_predictions.npz',**predictions)
    return game,rows


def main():
    check();cfg=json.loads((DOC/'contact_plan.json').read_text())
    for name in ['contact_freeze.json','complement_freeze.json']:
        for file,digest in json.loads((DOC/name).read_text())['files_sha256'].items():
            if sha(ROOT/file)!=digest:raise ValueError('Study source or input changed')
    dest=OUT/'contact_results.json'
    if dest.exists():raise ValueError('Never overwrite completed outcomes')
    rows=[]
    with ProcessPoolExecutor(max_workers=2,mp_context=get_context('spawn')) as pool:
        for future in as_completed([pool.submit(worker,g) for g in plan()['development_games']]):
            game,rr=future.result();rows.extend(rr);print('Contact',game,'complete',flush=True)
    dest.write_text(json.dumps(dict(plan_sha256=sha(DOC/'contact_plan.json'),records=rows),indent=2)+'\n')
    all_rows=json.loads((DOC/'scores.json').read_text())['records']+json.loads((DOC/'complement_scores.json').read_text())['records']+rows
    old=plan();old['expert_candidates']+=json.loads((DOC/'complement_plan.json').read_text())['candidates']+cfg['candidates']
    result=summarize(all_rows,old);result['contact_plan_sha256']=sha(DOC/'contact_plan.json')
    (DOC/'contact_result.json').write_text(json.dumps(result,indent=2)+'\n')
    (DOC/'contact_scores.json').write_text(json.dumps(dict(records=rows),indent=2)+'\n');check()
    for row in result['summary']:
        if row['candidate'] in cfg['candidates'] and row['reference']=='canonical':
            print(row['candidate'],row['scenario'],row['mean_spell_gain'],row['positive_games'],row['gate_pass'])
    print(result['decision'],result['selected'])


if __name__=='__main__':main()
