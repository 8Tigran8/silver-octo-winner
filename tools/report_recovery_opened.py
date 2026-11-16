"""Report all reopened outcomes and explicitly separate evidence from no-overfit claims."""
import json,sys,re
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
from recovery_opened_audit import DOC,check
from pff_reader import sha


def readable_spacing(text):
    before=r"\b(The|All|all|the|on|of|for|current|original|approved|only|versus|below|contains|predicts|misses|separates|and|is|improves|declines|remains|frozen|with|against|supplies|has)(?=\d)"
    after=r"(?<=\d)(?=(?:matches|games|original|reference|possessions|spells|frames|records|training|changed|mostly|of|at|historical|sets|in|passing|tests|PFF|fresh|brute|source|fixed|s)\b)"
    text=re.sub(before,r"\1 ",text)
    text=re.sub(after," ",text)
    text=text.replace("macro F10.","macro F1 0.").replace("%ofgraded","% of graded").replace("%and","% and ").replace("%on","% on")
    return text


def main():
    check();d=json.loads((DOC/'results.json').read_text());training=json.loads((DOC/'training_results.json').read_text());p=json.loads((DOC/'plan.json').read_text());fixtures={g['game_id']:g for g in p['fixtures']}
    canonical=[r for r in d['records'] if r['reference']=='canonical'];index={(r['game'],r['scenario'],r['candidate']):r for r in canonical}
    selected=[r for r in d['summary'] if r['reference']=='canonical'];di={(r['game'],r['scenario'],r['candidate']):r for r in d['diagnostics']}
    report=['# Current model on all 18 previously opened matches','',
    '16 November 2026. **Completed fixed-model re-evaluation, not a fresh independent test.** The user explicitly requested reopening all existing test fixtures after the current model was adopted. The18matches stayed out of weight fitting and numerical candidate selection. Their earlier parent outcomes and weaknesses were already known and motivated the research direction; this is disclosed rather than called a pristine holdout. No organizer labels were read or scored.','',
    '## Protocol and result','',
    'One frozen current model (weighted_short_gap) was compared with boundary102, both using the same current audited preprocessing, original raw inputs, period boundaries and official scorer. All18matches, both natural/fixed blind scenarios and both canonical/SDK-shadow references were retained. No settings, weights, decoding thresholds, references or output labels were changed after outcomes. Plans, source/model/raw hashes and historical output hashes were frozen before prediction.','',
    '| Cohort | Scenario | Parent Spell F1 | Current Spell F1 | Gain | Wins | Worst gain | Review guards |','|---|---|---:|---:|---:|---:|---:|---|']
    for r in selected:report.append(f"| {r['cohort']} | {r['scenario']} | {r['parent_spell']:.6f} | {r['current_spell']:.6f} | {r['gain']:+.6f} | {r['wins']}/{r['games']} | {r['worst_gain']:+.6f} | {'PASS' if r['review_guard_pass'] else 'INVESTIGATE'} |")
    report+=['','Review guards were declared before this run: each cohort/scenario/reference must have nonnegative mean Spell change, mean macro change≥−0.002, and worst game Spell change≥−0.020. These are regression review rules, not a statistical proof of generalization.','',
    '## Every match','', '| Cohort / fixture (PFF ID) | Parent natural | Current natural | Natural gain | Parent blind | Current blind |','|---|---:|---:|---:|---:|---:|']
    for g in p['fixtures']:
        game=g['game_id'];o=index[game,'natural_detected','boundary102'];n=index[game,'natural_detected','weighted_short_gap'];ob=index[game,'label_blind_seed7','boundary102'];nb=index[game,'label_blind_seed7','weighted_short_gap']
        report.append(f"| {g['cohort']}: {g['home']}–{g['away']} ({game}) | {o['spell_f1_mean']:.6f} | {n['spell_f1_mean']:.6f} | {n['spell_f1_mean']-o['spell_f1_mean']:+.6f} | {ob['spell_f1_mean']:.6f} | {nb['spell_f1_mean']:.6f} |")
    report+=['','## Pooled error diagnostics across all18matches','', '| Scenario / model | Short spells<3s matched / reference (IoU0.5) | False D / missing-ball live frames | False live / missing-ball D frames | Wrong team / missing-ball live frames |','|---|---:|---:|---:|---:|']
    for scenario in ['natural_detected','label_blind_seed7']:
        for model in ['boundary102','weighted_short_gap']:
            rr=[r for r in d['diagnostics'] if r['scenario']==scenario and r['candidate']==model]
            total=sum(r['short_spells'][k]['reference'] for r in rr for k in ['under_1s','1_to_3s']);matched=sum(r['short_spells'][k]['matched'] for r in rr for k in ['under_1s','1_to_3s'])
            m={k:sum(r['observation']['missing'][k] for r in rr) for k in ['live','false_dead','dead','false_live','wrong_team']}
            cells=[f"{matched}/{total} ({100*matched/total:.2f}%)"]+[f"{m[k]}/{m[den]} ({100*m[k]/m[den]:.2f}%)" for k,den in [('false_dead','live'),('false_live','dead'),('wrong_team','live')]]
            report.append('| '+scenario+' / '+model+' | '+' | '.join(cells)+' |')
    report+=['','Counts pool frames/spells, unlike the equal-match headline average. Short recall has no precision term. Missing-ball wrong-team errors increase even while false stoppages and total error decrease; both sides are disclosed.']
    report+=['','## Current train / development / audit comparison','', '| Group | Games | Current natural Spell F1 | Interpretation |','|---|---:|---:|---|',
    f"| PFF training |36|{training['mean_spell']:.6f}|In-sample|",'| PFF development |10|0.680009|Reused model-selection set;22 changed policies|']
    for r in selected:
        if r['scenario']=='natural_detected':report.append(f"| {r['cohort']} |{r['games']}|{r['current_spell']:.6f}|Previously opened matches, no fitting on them|")
    report+=['','The training score was recomputed with the exact current portable model on all36original PFF training games, not copied from its parent. The model itself was not retrained. The group gap combines fitting optimism, match difficulty, teams and measurement conditions; small or negative train-minus-test differences cannot prove absence of overfitting. The current18-game comparison is positive or negative evidence for these matches only; repeated development selection and shared event-proxy semantics remain limitations.','',
    '## Why Netherlands–United States is difficult','']
    game='10502';old=di[game,'natural_detected','boundary102'];new=di[game,'natural_detected','weighted_short_gap'];o=index[game,'natural_detected','boundary102'];n=index[game,'natural_detected','weighted_short_gap']
    report.append(f"Spell F1 improves{o['spell_f1_mean']:.6f}→{n['spell_f1_mean']:.6f}; macro F1{o['macro_f1']:.6f}→{n['macro_f1']:.6f}. The reference contains{n['n_gt_spells']}possessions; the parent predicts{o['n_pred_spells']}, current{n['n_pred_spells']}. This is substantial under-segmentation, with missed brief switches and inaccurate extents, not evidence that the metric is capped near0.5.")
    report+=['','| Reference possession length | Reference count | Parent matched at IoU0.5 | Current matched at IoU0.5 |','|---|---:|---:|---:|']
    for k,v in new['short_spells'].items():report.append(f"| {k} | {v['reference']} | {old['short_spells'][k]['matched']} | {v['matched']} |")
    report+=['','| Error on Netherlands–USA | Parent | Current |','|---|---:|---:|']
    for label in ['visible','missing']:
        for k,den in [('false_dead','live'),('wrong_team','live'),('false_live','dead')]:
            a,b=old['observation'][label],new['observation'][label];report.append(f"| {label}: {k} / reference {den} frames | {a[k]}/{a[den]} ({100*a[k]/a[den]:.2f}%) | {b[k]}/{b[den]} ({100*b[k]/b[den]:.2f}%) |")
    report+=['','Some conditional errors worsen: wrong-team frames during ball gaps increase, and visible-ball false-D frames increase. These losses are not hidden by the aggregate improvement.','',
    'At IoU0.5 the current model misses161of309reference possessions. Descriptive majority-label attribution separates33mostly called D,62mostly assigned to the opponent, and66mostly assigned to the correct team but with unsuitable extent or one-to-one competition. These are measured error categories, not proven causal mechanisms. Of the missed possessions mostly called D, mean ball visibility is only5.94%.','',
    'Ball is missing on53.8%ofgraded frames, similar to England–Senegal53.9%and Brazil–South Korea53.6%. Therefore missingness rate alone does not explain the lower score. Current wrong-team error on reference-live visible frames is4.92%, versus3.21%and2.60%on those two matches. Several error sources coexist.','',
    'First-half Spell F1 slightly declines0.587013→0.583607; second-half improves0.437762→0.518310. The aggregate gain does not imply every period improves. At stricter IoU0.7 only102of309reference possessions match, versus148atIoU0.5. Boundary extent remains a material weakness.','',
    'The source clock check reports maximum error below0.000001s, unchanged raw hashes and full reference coverage. The matched typed-transition median latency is0frames for the current model (only among matches within±1s), so this does not suggest one simple global offset; that statistic cannot rule out large errors among unmatched transitions. No label-based time-shift tuning was performed. Automatic event-state references may differ from actual controlled possession; no video-based tactical explanation is invented.','',
    '## Leakage, implementation and reproducibility','',
    'The36/10/12/6 PFF groups have empty match-ID intersections. Both current training registries exactly match the approved65/36sets. Current normalization is byte-for-byte equal to the parent train-only normalization. Geometry excludes provider possession/event fields and estimated ball positions; unit tests perturb annotation fields without affecting geometry. Augmentation uses dummy labels for audit degradation. Inference receives independent source period spans, not reference-derived score windows. Actual references are loaded after prediction bytes are saved. These checks are bounded evidence against direct leakage, not proof against every possible leakage path or exhaustive cross-provider duplicate detection.','',
    'For Netherlands–USA, paired evaluation asserts zero probability difference against both standalone current and parent entrypoints in both scenarios before saving outputs. The parent is recomputed on all18matches; differences from historical parent arrays/scores are explicitly retained in source receipts. The original12historical numbers are not silently substituted for current-source controls. Canonical and SDK-shadow reconstruction agreement is checked in complete scores.','',
    'A diagnostic-only implementation error occurred: the blind augmentation helper omits period metadata, and the new reporting function initially expected it. Predictions had already used the independent original periods correctly and were saved before the error. reporting_amendment.json preserves original/new tool hashes; original freeze, failure log and all saved prediction/reference bytes are retained. Reporting now explicitly supplies original periods and resumes from saved arrays. A dedicated synthetic regression verifies exact saved prediction preservation and correct period diagnostics. No model, input geometry, reference policy or candidate choice was changed.','',
    'The full current test suite and exact submission gate are rerun after this repair; see verification.json and tests.txt. Previous clean-source generation evidence remains valid because inference files, model bytes and the target CSV are unchanged. Four macOS/Linux outputs were byte-identical and official-validator valid. The target CSV SHA-256 remains1f4602b5b8b04e4066c5478549a9b5e53f13852ca6513f01422906d1dcb1a1c0.','',
    '## Reproduction and decision','',
    'Original source and raw files must be available locally; raw/reference/prediction arrays remain ignored and are not redistributed. The evidence directory publishes aggregate metrics and hashes only. tools/recovery_opened_audit.py run --resume completes the frozen18-game audit using the reporting amendment; tools/recovery_training_audit.py run --resume completes the36-game training inference audit; tools/report_recovery_opened.py recreates this report. Completed evidence is not replaced with a different candidate.','',
    'All per-game regressions, failed guards and technical defects must be investigated before claiming readiness. All frozen review guards pass: retain the already selected model. These reopened outcomes do not authorize a new threshold search. Remaining weaknesses are short-possession recall, ambiguous ownership, long observation gaps, domain shift and proxy labels. Repeatedly searching these18matches would consume them as further development. A genuinely new provider-diverse labeled sample would still be needed for stronger prospective generalization evidence. No organizer submission or repository publication was performed.']
    (DOC/'REPORT.md').write_text(readable_spacing('\n'.join(report)+'\n'))
    compact=dict(scope=d['scope'],summary=selected,training_mean_spell=training['mean_spell'],all_guards_pass=all(r['review_guard_pass'] for r in d['summary']),
        source_results_sha256=sha(DOC/'results.json'),training_results_sha256=sha(DOC/'training_results.json'))
    (DOC/'summary.json').write_text(json.dumps(compact,indent=2)+'\n');print(json.dumps(compact,indent=2))
if __name__=='__main__':main()
