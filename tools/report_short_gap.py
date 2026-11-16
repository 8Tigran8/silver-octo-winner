"""All prespecified recovery outcomes, including failures and diagnostic tradeoffs."""
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
from short_gap_study import DOC,OUT,plan,check
from pff_reader import sha


def summarize(rows, cfg=None):
    cfg=plan() if cfg is None else cfg
    ids=cfg['development_games'];candidates=cfg['mechanical_candidates']+cfg['expert_candidates'][1:]
    index={(r['game'],r['scenario'],r['reference'],r['candidate']):r for r in rows}
    expected={(g,s,r,c) for g in ids for s in cfg['scenarios'] for r in ['canonical','sdk_shadow'] for c in candidates}
    if len(index)!=len(rows) or set(index)!=expected:raise ValueError('Incomplete or duplicate score grid')
    summaries=[];gate=cfg['gate_each_scenario'];rng=np.random.default_rng(20261116)
    draws=rng.integers(0,len(ids),(10000,len(ids)))
    for candidate in candidates:
        for scenario in cfg['scenarios']:
            for reference in ['canonical','sdk_shadow']:
                rr=[index[g,scenario,reference,candidate] for g in ids]
                base=[index[g,scenario,reference,'control'] for g in ids]
                gain=np.array([r['spell_gain'] for r in rr]);macro=np.mean([r['macro_gain'] for r in rr])
                totals={k:sum(r['diagnostics'][k] for r in rr) for k in rr[0]['diagnostics']}
                rates={k:totals[n]/totals[d] if totals[d] else None for k,n,d in [
                    ('short_recall_iou05','short_matched','short_reference'),
                    ('missing_false_dead_rate','missing_false_dead','missing_live'),
                    ('missing_false_live_rate','missing_false_live','missing_dead')]}
                stats=dict(candidate=candidate,scenario=scenario,reference=reference,
                    spell_f1_mean=float(np.mean([r['spell_f1_mean'] for r in rr])),mean_spell_gain=float(gain.mean()),
                    macro_f1_mean=float(np.mean([r['macro_f1'] for r in rr])),mean_macro_gain=float(macro),
                    positive_games=int((gain>0).sum()),worst_spell_gain=float(gain.min()),
                    per_game_gain=dict(zip(ids,map(float,gain))),
                    class_f1_gain={k:float(np.mean([r['per_class'][k]['f1']-b['per_class'][k]['f1'] for r,b in zip(rr,base)])) for k in ['H','A','D']},
                    diagnostics=dict(counts=totals,**rates),
                    descriptive_paired_bootstrap_gain_95=np.quantile(gain[draws].mean(axis=1),[.025,.975]).tolist())
                stats['gate_pass']=(stats['mean_spell_gain']>=gate['mean_spell_gain_at_least']
                    and stats['positive_games']>=gate['positive_games_at_least']
                    and stats['worst_spell_gain']>=gate['worst_game_spell_gain_at_least']
                    and stats['mean_macro_gain']>=gate['mean_macro_gain_at_least'])
                summaries.append(stats)
    passing=[c for c in candidates if c!='control' and all(r['gate_pass'] for r in summaries if r['candidate']==c)]
    selected=max(passing,key=lambda c:(min(r['mean_spell_gain'] for r in summaries if r['candidate']==c),-candidates.index(c))) if passing else None
    return dict(scope='Repeated ten-match development; no fresh confirmation and no six-bridge/old-final/target scoring',
                plan_sha256=sha(DOC/'plan.json'),summary=summaries,eligible=passing,selected=selected,
                decision='Eligible for final technical regression only' if selected else 'Reject all candidates; preserve current production',
                warning='Intervals are descriptive, post-selection and common-provider dependent, not no-overfitting guarantees.')


def main():
    check();rows=[];sources={}
    for phase in ['mechanical','expert']:
        path=OUT/(phase+'_results.json');payload=json.loads(path.read_text())
        if payload['plan_sha256']!=sha(DOC/'plan.json'):raise ValueError('Wrong plan')
        rows.extend(r for r in payload['records'] if phase=='mechanical' or r['candidate']!='control')
        sources[str(path.relative_to(ROOT))]=sha(path)
    result=summarize(rows);result['source_sha256']=sources
    (DOC/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    (DOC/'scores.json').write_text(json.dumps(dict(records=rows),indent=2)+'\n')
    for row in result['summary']:
        if row['reference']=='canonical':
            print(row['candidate'],row['scenario'],round(row['mean_spell_gain'],6),row['positive_games'],'wins',row['gate_pass'],flush=True)
    print(result['decision'],result['selected'],flush=True)


if __name__=='__main__':main()
