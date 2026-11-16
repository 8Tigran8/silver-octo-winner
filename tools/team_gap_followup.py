"""Registered routing follow-up, reusing immutable train/dev caches and fits."""
import argparse,json,sys,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
import team_gap_specialist as study
from report_team_gap_specialist import summarize
from pff_reader import sha
from mtah import mixed_temporal,v4
from mtah.methods import boost
from short_gap_study import diagnostics
from decoder_development import metric
DOC=study.DOC;OUT=study.OUT

def configuration():return json.loads((DOC/'followup_plan.json').read_text())
def check():
    study.install_read_guard();study.check()
    for p,h in json.loads((DOC/'followup_freeze.json').read_text())['files_sha256'].items():
        assert sha(ROOT/p)==h,p

def route(name,cache,neural):
    ph=cache['ph'].copy();missing=~cache['seen']
    if name=='uniform_gap100':
        ph=.5*cache['bh']+.5*neural['uniform'];ph[missing]=neural['uniform'][missing]
    elif name in ['uniform_missing_only','focus_missing_only']:
        model=name.split('_')[0];ph[missing]=neural[model][missing]
    elif name!='control':raise ValueError(name)
    return ph

def worker(game):
    study.guard(game,'development');started=time.monotonic();cfg=configuration();rows=[]
    manifest={(r['game'],r['scenario']):r for r in json.loads((OUT/'development_manifest.json').read_text())}
    ref=ROOT/'reports/pff_access/references'/f'kloppy_{game}.npz';receipt=json.loads((ROOT/'reports/pff_access'/f'kloppy_{game}_preflight.json').read_text());assert sha(ref)==receipt['reference_cache_sha256']
    for scenario in study.SCENARIOS:
        path=OUT/'development'/f'{game}_{scenario}.npz';assert sha(path)==manifest[game,scenario]['cache_sha256']
        with np.load(path) as z:c=dict(z)
        parts=c['segments'].tolist();neural={}
        for name,m in study.MODELS.items():
            _,h=mixed_temporal.probabilities(m,c['x'],c['xs'],c['small_segments'].tolist(),study.STATE[1][1],study.STATE[1][2]);neural[name]=v4.neural_upsample(h,h,parts,len(c['pd']))[0]
        predictions={name:boost.label(c['pd'],route(name,c,neural),parts,len(c['pd']),study.BUNDLE['decoder']) for name in cfg['candidates']}
        with np.load(OUT/f'learned_{game}_{scenario}_predictions.npz') as z:assert np.array_equal(z['control'],predictions['control'])
        np.savez_compressed(OUT/f'followup_{game}_{scenario}_predictions.npz',**predictions)
        with np.load(ref) as z:
            for reference,key in [('canonical','gt'),('sdk_shadow','alternate')]:
                gt=z[key]
                for name,pred in predictions.items():
                    d=diagnostics(gt,pred,c['seen']);mask=(gt<2)&~c['seen'];d['missing_wrong_team']=int(((pred<2)&(pred!=gt)&mask).sum())
                    rows.append(dict(game=game,scenario=scenario,reference=reference,candidate=name,**metric(gt,pred),diagnostics=d))
    return game,rows,time.monotonic()-started

def main():
    check();dest=DOC/'followup_scores.json'
    if dest.exists():raise ValueError('Preserve completed follow-up')
    rows=[]
    with ProcessPoolExecutor(max_workers=2,mp_context=get_context('spawn'),initializer=study.initialize_score,initargs=(True,)) as pool:
        for f in as_completed([pool.submit(worker,g) for g in study.plan()['development_games']]):
            g,r,t=f.result();rows.extend(r);print('Scored routing',g,round(t,1),flush=True)
    dest.write_text(json.dumps(dict(records=rows),indent=2)+'\n');result=summarize(rows,configuration()['candidates'],study.plan())
    result.update(score_sha256=sha(dest),plan_sha256=sha(DOC/'followup_plan.json'),scope='Adaptive follow-up on original10development only; no18match reads; frozen gates')
    (DOC/'followup_result.json').write_text(json.dumps(result,indent=2)+'\n');check()
    for r in result['summary']:
        if r['reference']=='canonical':print(r['candidate'],r['scenario'],r['mean_spell'],r['gain'],r['wins'],r['short_recall'],r['missing_wrong_team_rate'],r['gates'],flush=True)
    print('Selected',result['selected'],flush=True)
if __name__=='__main__':main()
