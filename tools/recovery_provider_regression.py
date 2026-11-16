"""Prespecified in-sample cross-provider guard; never an unseen benchmark."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[key]='1'
import json,sys,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
from spell_weighted_study import DOC,OUT,check,training_data
from recovery_candidate_runtime import load,probabilities
from pff_mixed_score import decoder
from decoder_development import metric
from pff_reader import sha
from mtah.methods import boost


def initialize():
    global ASSETS,STATE,EXPERT,PARENT
    ASSETS,STATE,EXPERT,PARENT=load()


def worker(game):
    allowed=json.loads((DOC/'combined_plan.json').read_text())['supplement_if_pass']['games']
    if game not in allowed:raise ValueError('Not a registered supplementary fixture')
    parts,scenarios,sources=training_data(game);rows=[];started=time.monotonic()
    for scenario,data,gt in scenarios:
        if scenario=='clean_source':continue
        folder=OUT/'sequences'/game.replace('/','__')/scenario
        if json.loads((folder/'meta.json').read_text())['source_sha256']!=sources:raise ValueError('Supplementary source differs from frozen fit inputs')
        current,control=probabilities(data,parts,ASSETS,STATE,EXPERT,PARENT)
        values={}
        for name,pair in [('control',control),('weighted_short_gap',current)]:
            params=decoder()
            if name!='control':params.update(min_team_s=.4,cost_ha=4.5)
            pred=boost.label(*pair,parts,len(gt),params);values[name]=metric(gt,pred)
        base=values['control'];new=values['weighted_short_gap']
        family='DFL' if '/idsse_' in game else 'Metrica' if '/metrica_' in game else 'SkillCorner'
        rows.append(dict(game=game,provider=family,scenario=scenario,scope='in-sample cross-provider regression',control=base,candidate=new,
                         spell_gain=new['spell_f1_mean']-base['spell_f1_mean'],macro_gain=new['macro_f1']-base['macro_f1'],source_sha256=sources))
    return game,rows,time.monotonic()-started


def main():
    check()
    for file,digest in json.loads((DOC/'provider_freeze.json').read_text())['files_sha256'].items():
        if sha(ROOT/file)!=digest:raise ValueError('Supplementary inputs/source changed')
    dest=OUT/'provider_regression.json'
    if dest.exists():raise ValueError('Never overwrite completed checks')
    games=json.loads((DOC/'combined_plan.json').read_text())['supplement_if_pass']['games'];rows=[]
    with ProcessPoolExecutor(max_workers=2,mp_context=get_context('spawn'),initializer=initialize) as pool:
        for future in as_completed([pool.submit(worker,g) for g in games]):
            game,rr,seconds=future.result();rows.extend(rr);print(game,round(seconds,1),flush=True)
    if len(rows)!=12 or len({(r['game'],r['scenario']) for r in rows})!=12:raise ValueError('Incomplete supplementary grid')
    groups=[]
    for provider,scenario in sorted({(r['provider'],r['scenario']) for r in rows}):
        rr=[r for r in rows if (r['provider'],r['scenario'])==(provider,scenario)]
        s=dict(provider=provider,scenario=scenario,mean_spell_gain=float(np.mean([r['spell_gain'] for r in rr])),
               mean_macro_gain=float(np.mean([r['macro_gain'] for r in rr])),worst_spell_gain=min(r['spell_gain'] for r in rr))
        s['pass']=s['mean_spell_gain']>=-.005 and s['mean_macro_gain']>=-.002 and s['worst_spell_gain']>=-.02
        groups.append(s)
    result=dict(scope='All six specified training matches from three providers; not fresh independent validation',
                selection_sha256=sha(DOC/'selection.json'),records=rows,summary=groups,passed=all(s['pass'] for s in groups))
    dest.write_text(json.dumps(result,indent=2)+'\n');(DOC/'provider_regression.json').write_text(json.dumps(result,indent=2)+'\n');check()
    print(json.dumps(groups,indent=2));print('Supplementary pass',result['passed'])


if __name__=='__main__':main()
