"""Publish local aggregate experiment evidence without changing the release."""
import json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];DEST=ROOT/'docs/experiments/reliability'


def main():
    stages=[('adaptive',ROOT/'reports/reliability/adaptive_summary.json',ROOT/'reports/reliability/adaptive_results.json'),('long',ROOT/'reports/reliability/long_summary.json',ROOT/'reports/reliability/long_results.json'),('spell',ROOT/'reports/spell_trial/summary.json',ROOT/'reports/spell_trial/results.json')]
    combined={};counts={};provenance={}
    for stage,summary,records in stages:
        if not summary.exists() or not records.exists():raise ValueError('Study is incomplete: '+stage)
        values=json.loads(summary.read_text());combined.update({stage+'/'+k:v for k,v in values.items()});counts[stage]=len(json.loads(records.read_text())['records'])
        shutil.copy2(summary,DEST/f'{stage}_summary.json');shutil.copy2(records,DEST/f'{stage}_results.json')
    for family,base in [('long',ROOT/'reports/reliability'),('spell',ROOT/'reports/spell_trial')]:
        fits={p.parent.name:json.loads(p.read_text()) for p in (base/'models').glob('*/meta.json')}
        if len(fits)!=15:raise ValueError('Missing fits in '+family)
        provenance[family]=fits
    (DEST/'training_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    ranked=sorted(combined.items(),key=lambda item:item[1]['gain'],reverse=True);eligible=[name for name,value in ranked if value['status']=='ELIGIBLE_FOR_CONFIRMATION']
    decision={'status':'NUMERIC_DEVELOPMENT_GATES_PASS_MANUAL_LINEAGE_AND_CONFIRMATION_PENDING' if eligible else 'KEEP_VERIFIED_V4','best_development_candidate':ranked[0][0],'best_development':ranked[0][1],'automatic_gate_eligible':eligible,'release':'v4 unchanged','organizer_GT_used':False,'organizer_tracking_used':False,'user_hold':'No submission, messages, push or publication','new_candidate_fits':30,'scenario_cells_per_candidate':67,'record_counts':counts,'research_limit':'All29games are reused development; no fresh confirmation opened. Numeric checker validates declared neural exclusions only.','lineage_limit':'New neural fits exclude the entire scored fold. Inherited unchanged v3 probability paths retain scored-game exclusions as in frozen v4, not full neural-fold exclusions. This must be audited/resolved before claiming compliance with stricter all-stage fold-isolation language or promotion.'}
    (DEST/'decision.json').write_text(json.dumps(decision,indent=2)+'\n')
    lines=['# Local reliability and spell-aware training study','','**Release: verified v4, unchanged.** New work remains local under the user hold. No organizer reference or tracking input was used.','',
      '## Registered experiments','',
      'The [original plan](plan.json) was committed before new scores/fits. Eight rule-based probability mixtures use only gap length, observed visibility or symmetric model confidence. No learned gate was fitted from globally OOF component probabilities: those would require nested outer-game exclusions to avoid indirect label leakage. The uniform50 control exactly reproduced v4 spellF1 and macroF1 in all67cells.','',
      'The longer family keeps architecture, sources, sampling, loss and optimizer unchanged and increases fixed updates1500→4500. Allfive neural folds were trained with each of three predeclared seeds. Three individual uniform50 fusions and four equal-probability ensemble/rule mixtures were scored.','',
      'The [spell objective](spell_plan.json) was registered after adaptive-rule failures, while long training ran and before any long-family or spell-family score. It weights short H/A runs by clipped inverse-square-root duration and valid transition neighborhoods by2, capped4. It retains the .25SkillCorner dead-proxy multiplier. Weights affect training loss only; unknowns contribute no gradient. This family fits the samefive folds and three seeds for1500updates, scoring three individual uniform50 fusions and their equal-probability ensemble.','',
      'All hypotheses, seeds, failures and selection exposure are disclosed. These are reused development games. No new independent source was opened. Engineering thresholds are the previously frozen evaluation-1.0, not post-hoc adjusted.','',
      '## Complete candidate comparison','',
      '| Candidate | Composite spellF1 | Gain vs v4 | Wins /29 | Automatic status |','|---|---:|---:|---:|---|']
    for name,value in ranked:lines.append(f"| {name} | {value['candidate_composite']:.5f} | {value['gain']:+.5f} | {value['game_wins']} | {value['status']} |")
    lines += ['','The primary composite weights original matched/blind and SkillCorner natural/blind byone quarter each. Original stress and macro/per-class/lower-tail guards remain required. Bootstrap intervals in summaryJSONs are descriptive paired whole-game intervals stratified by source family; they do not correct repeated selection or shared opponents.','',
      '## Isolation and decision limits','',
      decision['lineage_limit'],'',
      'The gate checker covers metric completeness, numeric quality guards and declared neural training memberships. Its ELIGIBLE_FOR_CONFIRMATION status is not a release PASS and does not replace component-lineage review, provider-held-out checks, an unopened reserve and exact clean inference reproduction. A candidate satisfying only numeric checks cannot silently replace v4.','',
      'No public/private competition GT was scored or used to fit/choose these variants. The new predictors use public external feature caches whose raw preparation was verified in the earlier study. The original frozen PFF country split remains unopened; access/label eligibility are separate.','',
      '## Reproduce research','',
      '```bash','.venv/bin/python tools/reliability_train.py --device mps','.venv/bin/python tools/reliability_score.py adaptive','.venv/bin/python tools/reliability_score.py long --device cpu','.venv/bin/python tools/spell_train.py --device mps','.venv/bin/python tools/spell_score.py --device cpu','.venv/bin/python tools/report_reliability.py','```','',
      'The earlier pinned public-data preparation and original temporal outputs are prerequisites. Completed score files refuse overwrite; model metadata verifies folds and hashes. Research files remain under ignored reports/. See original source manifests and temporal reproduction instructions.','']
    (DEST/'REPORT.md').write_text('\n'.join(lines));print(json.dumps(decision,indent=2))


if __name__=='__main__':main()
