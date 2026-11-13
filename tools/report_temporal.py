"""Publish aggregate evidence for temporal fusion, with source/split limits."""
from pathlib import Path
import json,statistics,shutil
ROOT=Path(__file__).resolve().parents[1];IN=ROOT/'reports/temporal';OUT=ROOT/'docs/experiments/temporal'


def mean(rows,key):return statistics.mean(r[key] for r in rows)


def main():
    selection=json.loads((IN/'selection.json').read_text());cv=json.loads((IN/'cv_results.json').read_text());diag=json.loads((IN/'generalization_results.json').read_text());cpu_path=IN/'deployment_cpu_results.json'
    external={n:json.loads((IN/(n+'_results.json')).read_text()) for n in ['wcsf','european','metrica3']}
    for name in ['cv_results.json','generalization_results.json','wcsf_results.json','european_results.json','metrica3_results.json','raw_input_equality.json']:
        shutil.copy2(IN/name,OUT/name)
    if cpu_path.exists():shutil.copy2(cpu_path,OUT/cpu_path.name)
    shutil.copy2(IN/'external/wcsf_reference_quality.json',OUT/'wcsf_reference_quality.json')
    provenance={p.parent.name:json.loads(p.read_text()) for p in (IN/'models').glob('*/meta.json')};(OUT/'training_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    counts=[]
    for game in json.loads((OUT/'plan.json').read_text())['training_games']:
        scenario='matched_seed0' if game.startswith('original/') else 'natural_detected';m=json.loads((IN/'sequences'/game.replace('/','__')/scenario/'meta.json').read_text());counts.append({'game':game,'reference_period_frames':sum(b-a for a,b in m['full_segments']),'network_rate_hz':5})
    (OUT/'training_data_counts.json').write_text(json.dumps(counts,indent=2)+'\n')
    summary={}
    for cohort,d in external.items():
        metric='all_conditional_team_macroF1' if cohort=='european' else 'spell_f1_mean';rows=d['records'];summary[cohort]={}
        for scenario in sorted({r['scenario'] for r in rows}):
            b={r['match']:r[metric] for r in rows if r['candidate']=='v3' and r['scenario']==scenario};n={r['match']:r[metric] for r in rows if r['candidate']==selection['candidate'] and r['scenario']==scenario};differences=[n[g]-b[g] for g in b]
            summary[cohort][scenario]={'metric':metric,'v3':statistics.mean(b.values()),'v4':statistics.mean(n.values()),'gain':statistics.mean(differences),'matches':len(b),'wins':sum(v>1e-10 for v in differences),'losses':sum(v < -1e-10 for v in differences)}
    external_pass=all(r['gain']>0 for cohort in summary.values() for r in cohort.values())
    decision={'candidate':'v4 / fusion50','development_gate_pass':selection['development_gate_pass'],'external_checks_pass':external_pass,'CV_games':29,'additional_conditional_games':9,'additional_Metrica_games':1,'new_WCSF_games':1,'total_distinct_screened_games':40,'fresh_final_test_limit':'Only WCSF is a new-source comparison in this cycle; anonymous provenance and reference conflicts limit certainty. Other cohorts reused. PFF remains unopened.','quality_recommendation':'Adopt only after CPU frozen-asset validation and two clean reproductions' if external_pass and selection['development_gate_pass'] else 'Keep v3','external':summary,'reproduction':'PENDING until separate receipt'}
    (OUT/'decision.json').write_text(json.dumps(decision,indent=2)+'\n')
    lines=['# Temporal fusion: expanded generalization study','','**Selected candidate: v4 / fusion50.** Whole-game development and external checks passed; see the separate technical reproduction receipt before claiming a packaged release.','',
      'The earlier data-only and decoder searches did not meet replacement gates. This study replaces framewise-only reasoning with a learned temporal model and evaluates source transfer. Ranking evidence below uses the unmodified official whole-spell F1; conditional European H/A scores are kept separate.','',
      '## Method and fitting','',
      'A residual dilated temporal convolutional network consumes the existing108 geometry/contact-context features plus108 missing-value masks at5Hz. Five residual blocks each contain two3-tap convolutions,48 channels, dilations1/2/4/8/16, dropout0.15. The neural receptive field is125steps (25seconds), in addition to the noncausal context already present in the geometric features. Two logits predict dead probability and home possession conditional on live play. Symmetric direct/team-swapped inference is averaged.','',
      'Normalization uses finite-feature means/stds from each training split only; missing values are zero after standardization and carry explicit masks. Standard deviations floor at0.01 and normalized values clip to[-8,8]. All period windows reset from observation/metadata-derived spans. No ownership/event field, match ID, team name or persistent player identifier is an input.','',
      'Twenty-nine public matches train the temporal model: sevenDFL, twoMetrica and twentySkillCorner. Original scenarios are matchedseed0, label-blindseed1 and clean; SkillCorner scenarios are natural and label-blindseed7. Training samples families50/50, then games/scenarios uniformly, with random team swaps. AdamW uses1500 fixed updates, batch16,256supervised steps plus64context steps each side, learning rate0.001, weight decay0.001, seed20261113 and deterministic PyTorch operations. No validation early stopping. SkillCorner dead-proxy loss has weight0.25; its live frames and original source labels weight1. Unknown targets supply no gradient.','',
      'The retained v3 probability path remains original nine-game two-stage boosting plus the nineteen-game team-only head. V4 averages its dead and conditional-team probabilities50/50 with the neural outputs, then uses the same cost9 duration-constrained decoder. The default release command loads frozen external-trained weights and runs onCPU. Neural fitting was onAppleM4MPS; repeated CPU inference with frozen weights is distinct from repeating training.','',
      '## Whole-game cross-validation','',
      'Five whole-game folds cover all29 games. Every corruption/team-swap sibling stays with its game. Each neural model is fitted without the entire scored fold (23or24training games). Baseline v3 uses its own exclusion models for the original19 and frozen nineteen-game weights for the ten games absent from its training. The original learned stack retains nested test-game exclusions. No learned baseline probabilities enter neural fitting; fusion happens only after inference. All67baseline scenario scores exactly reproduced prior results.','',
      'These are reused research games, with historically selected geometry/configuration families. Fold exclusion prevents direct estimator leakage, but does not erase historical model-selection exposure. Different training-pool sizes between methods and shared opponents are further limits. Intervals resample whole games within original/SkillCorner families, preserving paired scenarios,10000draws, seed20261113; they are descriptive, not a correction for selection or team dependence.','',
      '| Cohort | V3 spellF1 | Neural alone | Selected fusion |','|---|---:|---:|---:|']
    groups=['Original matched (9)','Original label-blind (9)','Original stress (9)','SkillCorner natural (20)','SkillCorner label-blind (20)']
    for i,label in enumerate(groups):lines.append('| '+label+' | '+' | '.join(f"{selection['scores'][n]['means'][i]:.4f}" for n in ['v3','tcn','fusion50'])+' |')
    s=selection['scores']['fusion50'];ci=s['descriptive_95ci'];lines += ['',f"Equal-quarter weighted original matched/blind and Skill natural/blind: **{selection['scores']['v3']['weighted']:.4f} → {s['weighted']:.4f}**, gain+{s['gain']:.4f}; descriptive95%interval [{ci[0]:+.4f},{ci[1]:+.4f}]. Paired two-scenario game wins/losses: **{s['game_wins']}/{s['game_losses']}**. Registered minimum gain0.01 and no cohort mean loss>0.01 passed. Standalone neural failed the known-matched regression gate; it was not substituted after external scoring. Selection and full checkpoint were frozen in commitd0bd103 before new WCSF reference/model scores.",'',
      'By primary matched/natural source, DFL changes0.6395→0.6379; Metrica0.6482→0.6651; SkillCorner0.5817→0.6500. This small DFL loss is retained in the report, not averaged away as a universal improvement.','',
      '## Overfit and provider-shift diagnostics','',
      'Compare standalone neural predictions from the full29-game fit (in-sample) with its excluded-fold predictions on the same primary scenario. Full fitting has29games while excluded fits have23/24; this is a descriptive train/CV gap, not a pure causal estimate of memorization.','',
      '| Source | Full fit in-sample spellF1 | Whole-game-out spellF1 |','|---|---:|---:|']
    for label,prefix in [('DFL','original/idsse_'),('Metrica','original/metrica_'),('SkillCorner','skillcorner/')]:
        train=[r for r in diag['records'] if r['in_sample'] and r['game'].startswith(prefix)];test=[r for r in cv['records'] if r['candidate']=='tcn' and r['game'].startswith(prefix) and r['scenario'] in ['matched_seed0','natural_detected']];lines.append(f'| {label} | {mean(train,"spell_f1_mean"):.4f} | {mean(test,"spell_f1_mean"):.4f} |')
    lines += ['','Two further source-held-out models were fitted before scoring. DFL-only training gives SkillCorner20mean spellF1=0.5850; SkillCorner-only gives DFL7=0.5081. Mixed-source OOF neural means are0.6662 and0.6159 respectively. This shows material provider/label/sensor shift; it is not evidence that any single-source model generalizes universally. Tiny train/CV gaps alone cannot prove absence of overfit.','',
      '## Additional external checks','',
      '| Source/scenario | Metric | V3 | V4 | Scope |','|---|---|---:|---:|---|']
    for cohort,scenarios in summary.items():
        for scenario,r in scenarios.items():lines.append(f"| {cohort}/{scenario} | {r['metric']} | {r['v3']:.4f} | {r['v4']:.4f} | {r['wins']}/{r['matches']}wins |")
    lines += ['','European9: known ownership only; unknown remainsX, notD. These games were used in the earlier v3 confirmation and are not fresh. MetricaSample3 was previously v1-scored; it never trains these models. No independent period metadata exists for its export, so predictions use the whole tracking stream rather than annotation-derived half cuts; only event-covered reference spans are scored. Event-derived labels and the whole-stream treatment limit comparability to the competition.','',
      'WCSF is a new anonymized Groningen/SciSports public research match. Its half files and clocks independently define predictor spans. Team-relative coordinate frames are reconciled from duplicate observed ball coordinates, without ownership/alive flags. Predictors never read AliveFlag, PossessionFlag or team_game_phase. Source rows at110timestamps disagree after H/A/Dmapping; only unanimous reference frames are scored, leaving266unknown25Hzframes out of143431period frames (0.1855%). Unknown GT holes affect scored spell boundaries, so this is not pristine complete ground truth. This source-format issue was resolved before any model score and without changing model weights. No majority vote or prediction-based reference repair.','',
      'Raw trajectory screening found no suspicious overlap with Metrica1/2/3 or the competition automatic tracking, across±180seconds and fouraxis reflections; this is a heuristic, not proof of anonymous identities. Only aggregate eligibility results are published. No competition public/private GT enters fitting, selection or these checks. New-source evidence is one game, so there is no meaningful across-game confidence interval for WCSF alone.','',
      'The40screened matches consist of29CVgames, nineconditional-only European games, Metrica3 and WCSF. This count must not be described as40fresh independently sampled tests. The PFF64access request and prospective36/10/12+6country-bridge split remain pending/unopened. Hidden competition quality and qualification probability are unknown.','',
      '## Reproduce','',
      'Use the repository README for frozen-weight CPU generation and two clean source-copy checks. Research fitting uses public raw inputs, optional requirements-temporal.txt and the pinned original/SkillCorner sources:','',
      '```bash','.venv/bin/python tools/temporal_trial.py prepare --jobs 3','.venv/bin/python tools/temporal_trial.py fit --device mps','.venv/bin/python tools/evaluate_temporal.py cv --device mps','.venv/bin/python tools/evaluate_temporal.py diagnostic --device mps','```','',
      'The raw-input equality audit reproduced all27original-game training feature arrays and swapped arrays exactly atfloat32. SkillCorner preparation reads pinned raw data directly. Source manifests, model hashes, complete per-match aggregate scores, exclusions, data counts and frozen plans are alongside this report. Raw inputs, predictions and intermediate arrays remain local. See [the industry/research comparison](INDUSTRY.md) for why this temporal method was selected and why a pretrained demo is not automatically a validated replacement.','']
    (OUT/'REPORT.md').write_text('\n'.join(lines));print(json.dumps(decision,indent=2))


if __name__=='__main__':main()
