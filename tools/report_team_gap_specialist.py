"""Complete preregistered ownership-study grid and unchanged quality gates."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
from team_gap_specialist import DOC,plan,check,install_read_guard,SCENARIOS
from pff_reader import sha

def summarize(rows,candidates,cfg):
    index={(r['game'],r['scenario'],r['reference'],r['candidate']):r for r in rows}
    expected={(g,s,ref,c) for g in cfg['development_games'] for s in SCENARIOS for ref in ['canonical','sdk_shadow'] for c in candidates}
    if len(index)!=len(rows) or set(index)!=expected:raise ValueError('Incomplete/duplicate outcome grid')
    results=[];gate=cfg['gate_each_scenario']
    for c in candidates:
        for s in SCENARIOS:
            for ref in ['canonical','sdk_shadow']:
                base=[index[g,s,ref,'control'] for g in cfg['development_games']];rr=[index[g,s,ref,c] for g in cfg['development_games']]
                gains=np.array([r['spell_f1_mean']-b['spell_f1_mean'] for r,b in zip(rr,base)]);macro=np.mean([r['macro_f1']-b['macro_f1'] for r,b in zip(rr,base)])
                counts={k:sum(r['diagnostics'][k] for r in rr) for k in rr[0]['diagnostics']};bc={k:sum(r['diagnostics'][k] for r in base) for k in base[0]['diagnostics']}
                gates=dict(mean_gain=float(gains.mean())>=gate['mean_spell_gain_at_least'],wins=int((gains>0).sum())>=gate['positive_games_at_least'],worst=float(gains.min())>=gate['worst_game_spell_gain_at_least'],macro=float(macro)>=gate['mean_macro_gain_at_least'],
                           short_recall=counts['short_matched']/counts['short_reference']>=bc['short_matched']/bc['short_reference'],missing_wrong_team=counts['missing_wrong_team']/counts['missing_live']<=bc['missing_wrong_team']/bc['missing_live'])
                results.append(dict(candidate=c,scenario=s,reference=ref,mean_spell=float(np.mean([r['spell_f1_mean'] for r in rr])),gain=float(gains.mean()),wins=int((gains>0).sum()),worst=float(gains.min()),macro_gain=float(macro),counts=counts,
                    short_recall=counts['short_matched']/counts['short_reference'],missing_wrong_team_rate=counts['missing_wrong_team']/counts['missing_live'],missing_total_live_error_rate=(counts['missing_wrong_team']+counts['missing_false_dead'])/counts['missing_live'],
                    per_game_gain={g:float(x) for g,x in zip(cfg['development_games'],gains)},gates=gates,pass_all=all(gates.values())))
    eligible=[c for c in candidates if c!='control' and all(r['pass_all'] for r in results if r['candidate']==c)]
    selected=max(eligible,key=lambda c:(min(r['gain'] for r in results if r['candidate']==c),-candidates.index(c))) if eligible else None
    return dict(summary=results,eligible=eligible,selected=selected,decision='Eligible for independent technical and frozen-development regression; no18-match evaluation authorized' if selected else 'Reject tested policies; keep current production model')

def main(phase):
    install_read_guard();check();cfg=plan();path=DOC/(phase+'_scores.json');rows=json.loads(path.read_text())['records'];result=summarize(rows,cfg[phase+'_candidates'],cfg)
    result.update(scope='Original ten development matches only; fixed fits on approved65traininggames; no18match or target reads',plan_sha256=sha(DOC/'plan.json'),score_sha256=sha(path),control='weighted_short_gap current adopted model')
    prior=json.loads((ROOT/'docs/experiments/spell-weighted-finetune/combined_scores.json').read_text())['records'];old={(r['game'],r['scenario'],r['reference']):r for r in prior if r['candidate']=='weighted_short_gap'}
    for r in rows:
        if r['candidate']=='control':assert all(r[k]==old[r['game'],r['scenario'],r['reference']][k] for k in ['spell_f1_mean','macro_f1','per_class','n_gt_spells','n_pred_spells'])
    if phase=='learned':
        ix={(r['game'],r['scenario'],r['reference'],r['candidate']):r for r in rows};result['focus_vs_matched_uniform']={s:dict(mean_spell_gain=float(np.mean([ix[g,s,'canonical','focus']['spell_f1_mean']-ix[g,s,'canonical','uniform']['spell_f1_mean'] for g in cfg['development_games']])),wins=sum(ix[g,s,'canonical','focus']['spell_f1_mean']>ix[g,s,'canonical','uniform']['spell_f1_mean'] for g in cfg['development_games'])) for s in SCENARIOS}
    (DOC/(phase+'_result.json')).write_text(json.dumps(result,indent=2)+'\n')
    for r in result['summary']:
        if r['reference']=='canonical':print(r['candidate'],r['scenario'],'spell',round(r['mean_spell'],6),'gain',round(r['gain'],6),'wins',r['wins'],'short',round(r['short_recall'],4),'wrong',round(r['missing_wrong_team_rate'],4),r['gates'],flush=True)
    print(result['decision'],result['selected'])
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['mechanical','learned']);main(p.parse_args().phase)
