"""Compare both matched continuation fits with parent and with one another."""
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
from spell_weighted_study import DOC,OUT,plan,check
from report_short_gap import summarize
from pff_reader import sha


def main():
    check();cfg=plan();path=OUT/'results.json';payload=json.loads(path.read_text())
    if payload['plan_sha256']!=sha(DOC/'plan.json'):raise ValueError('Wrong registered study')
    source=ROOT/'docs/experiments/short-gap-recovery/scores.json'
    control=[r for r in json.loads(source.read_text())['records'] if r['candidate']=='control']
    report_cfg=dict(cfg,mechanical_candidates=['control'],expert_candidates=['control']+cfg['candidates'],scenarios=['natural_detected','label_blind_seed7'])
    result=summarize(control+payload['records'],report_cfg)
    result['plan_sha256']=sha(DOC/'plan.json');result['source_sha256']={str(path.relative_to(ROOT)):sha(path),str(source.relative_to(ROOT)):sha(source)}
    result['scope']='Matched additional-training experiment on reused development; no new final/bridge/target evaluation'
    rows=payload['records'];index={(r['game'],r['scenario'],r['reference'],r['candidate']):r for r in rows}
    paired=[]
    for suffix in ['', '_short']:
        for scenario in report_cfg['scenarios']:
            for ref in ['canonical','sdk_shadow']:
                values=np.array([index[g,scenario,ref,'weighted'+suffix]['spell_f1_mean']-index[g,scenario,ref,'uniform'+suffix]['spell_f1_mean'] for g in cfg['development_games']])
                paired.append(dict(decoder='short' if suffix else 'original',scenario=scenario,reference=ref,
                                   mean_weighted_minus_uniform=float(values.mean()),positive_games=int((values>0).sum()),
                                   worst_weighted_minus_uniform=float(values.min()),per_game=dict(zip(cfg['development_games'],map(float,values)))))
    result['matched_training_objective_comparison']=paired
    for name in ['uniform','weighted']:
        meta=OUT/name/'meta.json'
        (DOC/(name+'_meta.json')).write_text(meta.read_text())
    (DOC/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    (DOC/'scores.json').write_text(json.dumps(payload,indent=2)+'\n')
    for r in result['summary']:
        if r['reference']=='canonical':print(r['candidate'],r['scenario'],r['mean_spell_gain'],r['positive_games'],r['worst_spell_gain'],r['gate_pass'])
    print('Matched objective comparisons',json.dumps(paired));print(result['decision'],result['selected'])


if __name__=='__main__':main()
