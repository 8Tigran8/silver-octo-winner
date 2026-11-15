"""Record bounded robustness followup, including failures and unchanged release."""
from pathlib import Path
import json,statistics,shutil
ROOT=Path(__file__).resolve().parents[1];IN=ROOT/'reports/robust_followup';OUT=ROOT/'docs/experiments/robust-followup'


def main():
    selection=json.loads((IN/'selection.json').read_text());summary=json.loads((IN/'development_summary.json').read_text());external=json.loads((IN/'reused_external.json').read_text());groups=[]
    for cohort in ['wcsf','metrica3','european']:
        for scenario in sorted({r['scenario'] for r in external['records'] if r['cohort']==cohort}):
            rows=[r for r in external['records'] if (r['cohort'],r['scenario'])==(cohort,scenario)]
            groups.append({'cohort':cohort,'scenario':scenario,'games':len(rows),'v4':statistics.mean(r['v4_primary'] for r in rows),'candidate':statistics.mean(r['candidate_primary'] for r in rows),'gain':statistics.mean(r['gain'] for r in rows),'wins':sum(r['gain']>0 for r in rows)})
    passed=all(r['gain']>=-.01-1e-12 for r in groups)
    other_seeds={str(seed):json.loads((IN/f'seed{seed}_summary.json').read_text()) for seed in [20261113,20261114]}
    decision={'candidate':selection['candidate'],'development':selection['development'],'other_seed_diagnostics':other_seeds,'reused_source_guard_pass':passed,'reused_sources':groups,'status':'RESEARCH_ONLY: numeric and reused-source guards pass; independent confirmation and strict inherited-fold review pending' if passed else 'NOT_ADOPTED: reused-source guard failed','independent_confirmation':'NOT ASSESSED; reserve unopened','fresh_games_this_cycle':0,'no_post_external_runnerup_selection':True,'production_release':'v4 unchanged','user_hold':'No submission/messages/push/publication'}
    (OUT/'decision.json').write_text(json.dumps(decision,indent=2)+'\n')
    for name in ['development_results.json','development_summary.json','selection.json','reused_external.json','seed20261113_summary.json','seed20261114_summary.json']:shutil.copy2(IN/name,OUT/name)
    lines=['# Reference-blind temporal robustness followup','',decision['status'],'',
      'The earlier long-TCN candidate gained+0.03019on29reused development games, but then regressed on Metrica3. This followup was registered after that observed transfer failure and before new rule scores. These are further development experiments; no source is relabelled fresh.','',
      'Fourteen configurations include old/long controls, channel-specific old/new neural mixtures, temperature softening and unsupervised outlier gates. The feature gate averages the eight largest absolute training-normalized observed features, then uses a period-reset five-second mean. Means/scales come from the excluded-fold training data only. No ownership/alive/event target influences inference weights.','',
      'Select the highest-scoring changed rule passing evaluation-1.0 on the fixed67cells; old/long controls are not new corrective candidates. The selected OOD4to8rule was frozen before checking other seeds and repeating old-source checks. No runner-up was chosen from those checks. Both control metrics reproduce their old scores exactly.','',
      '| Candidate | Composite | Gain | Wins /29 | Numeric dev status |','|---|---:|---:|---:|---|']
    for name,value in sorted(summary.items(),key=lambda item:item[1]['gain'],reverse=True):lines.append(f"| {name} | {value['candidate_composite']:.5f} | {value['gain']:+.5f} | {value['game_wins']} | {value['status']} |")
    lines += ['','## Frozen reused-source checks','','| Source | Scenario | V4 | Candidate | Gain |','|---|---|---:|---:|---:|']
    for row in groups:lines.append(f"| {row['cohort']} ({row['games']}) | {row['scenario']} | {row['v4']:.4f} | {row['candidate']:.4f} | {row['gain']:+.4f} |")
    lines += ['','All eleven games had previously been used, and Metrica3 failure motivated the hypothesis. These are development/regression evidence, not independent final confirmation. WCSF ambiguity/identity limits, event-derived Metrica labels and conditional-only European ownership labels remain unchanged. Other fixed seeds and all failed candidates are published as aggregate files.','',
      'The 29/29 game win count uses each game mean across its two primary scenarios. Extra corruptions are not independent matches. Bootstrap intervals are descriptive and do not correct repeated selection, opponent dependence or source effects.','',
      'Legacy component isolation still excludes the scored game rather than every other game in its neural fold. New TCN fitting excludes the whole fold. See ../reliability/lineage_audit.json; do not claim strict all-stage full-fold compliance. No competition GT was accessed. Only public external games trained the weights and supplied quality references; separate local file generation may use organizer automatic tracking without scoring its GT.','',
      'Verified v4 and its local submission package remain unchanged. The user hold bars submissions, messages, pushes and publication. No independent reserve was opened; final qualification and hidden-match quality are unknown.','']
    (OUT/'REPORT.md').write_text('\n'.join(lines));print(json.dumps(decision,indent=2))


if __name__=='__main__':main()
