"""Publish aggregate-only diagnostics, retaining every stress loss and alias."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/experiments/provider-stress'


def comparisons(payload):
    pairs = {}
    for row in payload['records']:
        pairs.setdefault((row['game'], row['scenario']), {})[row['candidate']] = row
    for (game, scenario), models in pairs.items():
        a, b = models['strict_v4'], models['boundary102']
        yield game, scenario, a, b, b['spell_f1_mean'] - a['spell_f1_mean']


def main():
    files = {'results.json': ROOT/'reports/provider_stress_diagnostic/results.json',
             'decomposition_results.json': ROOT/'reports/provider_stress_decomposition/results.json'}
    for name, source in files.items():
        shutil.copy2(source, OUT/name)
    result = json.loads((OUT/'results.json').read_text())
    diagnostic = json.loads((OUT/'decomposition_results.json').read_text())
    by_case = {(x['game'], x['scenario']): x for x in result['exports']}
    unique = [(g,s,a,b,d) for g,s,a,b,d in comparisons(result)
              if s.startswith('hz') and by_case[g,s]['identical_to'] is None]
    assert len(unique) == 6 and len(result['records']) == 24
    # Canonical SkillCorner control must reproduce the prior scored natural cell.
    old = json.loads((ROOT/'docs/experiments/pff-mixed/old_results.json').read_text())
    for candidate in ['strict_v4', 'boundary102']:
        a = next(r for r in result['records'] if r['game']=='skillcorner/1925299' and r['scenario']=='baseline' and r['candidate']==candidate)
        b = next(r for r in old['records'] if r['game']=='skillcorner/1925299' and r['scenario']=='natural_detected' and r['candidate']==candidate)
        assert all(a[k] == b[k] for k in ['spell_f1_mean','macro_f1','n_gt_spells','n_pred_spells','per_class'])
    lines = ['# Supplemental provider/sensor stress diagnostic',
        '\n16 November 2026. **Robustness weakness found. No model changes.**',
        '\nVV\'s [PR #2](https://github.com/8Tigran8/silver-octo-winner/pull/2), commit `d69ee6b900efbb4dd45c626e5b2a3c2578e57d1c`, contributes the converter, four presets and twelve synthetic correctness tests. It is integrated into the local research branch; remote main and the organizer submission remain untouched.',
        '\n## Result and interpretation',
        f"\nBoundary102 improves both canonical/native controls, but loses all **six distinct perturbed conditions**, by {min(d for *_,d in unique):+.5f} to {max(d for *_,d in unique):+.5f} spellF1 against same-exclusion v4. Eight named perturbed copies reduce to six distinct inputs because native 10-Hz SkillCorner cannot produce a separate 15-Hz measurement stream. These are only **two familiar development games**, not six/eight new independent matches.",
        '\nThis is negative evidence about synthetic cadence/noise robustness. It does not overwrite the independently frozen 12-game PFF result, but it prevents a claim that the new model dominates v4 under every sensor condition. The main candidate, mixture, decoder and submitted-file candidate are unchanged; no seed, condition, game or losing row is dropped. Original development/final gates passed; this supplemental diagnostic was explicitly not registered as a new promotion gate.',
        '\n## Complete combined-pilot results',
        '\n| Source / fold | Condition | V4 spellF1 | Boundary102 | Gain | MacroF1 gain | Ball missing | Duplicate of |',
        '|---|---|---:|---:|---:|---:|---:|---|']
    for game, scenario, a, b, gain in comparisons(result):
        e=by_case[game,scenario]
        lines.append(f"| {game} / {2 if game.startswith('original/') else 4} | {scenario} | {a['spell_f1_mean']:.5f} | {b['spell_f1_mean']:.5f} | {gain:+.5f} | {b['macro_f1']-a['macro_f1']:+.5f} | {e['checks']['missing_ball_fraction']:.1%} | {e['identical_to'] or '-'} |")
    lines += ['\n`baseline` retains the exact existing reader format. `native_observed_only` uses the source-native frequency, zero noise and zero dropout bursts. For SkillCorner, baseline includes 10-to-25-Hz interpolation while all observed-only/stress inputs leave unmapped grid rows NaN. Therefore baseline-to-stress degradation combines representation, cadence, coordinate noise and bursts. The extra native control separates representation from added corruption.',
        '\nNoise is independent zero-mean Gaussian noise per x/y coordinate (0.5 m or 2 m), applied to ball and players. Ball burst starts follow a two-per-minute Poisson process, with uniform 2-6-second duration. No holding, interpolation, clipping or temporal correlation is introduced by the stress converter. Source gaps and half boundaries remain intact. Such framewise jitter is a synthetic failure probe, not a calibrated estimate of real tracking errors.',
        '\n## Post-hoc one-factor decomposition',
        '\nThese explanatory probes were registered after seeing combined-pilot losses. They are descriptive, use the same frozen models/folds, and do not constitute new confirmation or a tuning/adoption rule.',
        '\n| Source | One changed factor | V4 spellF1 | Boundary102 | Gain |',
        '|---|---|---:|---:|---:|']
    for game,scenario,a,b,gain in comparisons(diagnostic):
        lines.append(f"| {game} | {scenario} | {a['spell_f1_mean']:.5f} | {b['spell_f1_mean']:.5f} | {gain:+.5f} |")
    lines += ['\n## Isolation and reproducibility',
        '\nAll predictions use the original game\'s excluded outer-fold models (DFL fold 2; SkillCorner fold 4). Every inherited direct/nested stage excludes the whole scored fold. Both old and new neural normalizers are exactly recomputed from their training-only rows. Full-fit release weights are never used for these familiar games. The original natural SkillCorner scores exactly reproduce the previously recorded spell/macro/class/spell-count metrics for both models.',
        '\nOnly geometry, clocks, team-slot mapping, pitch size and independently supplied periods reach prediction. Reference arrays are attached after corruption and enter scoring after both predictions exist. Twelve converter tests verify reference independence, missingness, sampling clocks and period boundaries. The original 67 frozen model/source/evidence files pass before/after hash checks. No target-match input, organizer GT or original PFF final reference is scored in this diagnostic.',
        '\nThe runners stream one full-match geometry condition at a time and save only aggregate receipts and ignored local predictions, avoiding the standalone exporter\'s 20-GiB capacity requirement. Raw tracking and reference arrays are not redistributed.',
        '\n```bash\n.venv/bin/python -m unittest tests.test_provider_stress -v\n.venv/bin/python tools/evaluate_provider_stress.py\n.venv/bin/python tools/diagnose_provider_stress.py\npython3 tools/report_provider_stress.py\n```',
        '\nCompleted diagnostic endpoints refuse overwrite. Full research reproduction additionally requires the original public raw-data files, cached training-only normalizers and audited excluded-fold artifacts, as described in prior studies. Release inference remains the separate frozen-bundle command and requires none of these research caches.',
        '\n## Consequence for the current solution',
        '\nKeep boundary102 as the selected candidate on the strength of the untouched PFF final, while disclosing this additional weakness and the Metrica overfit risk. No model is switched using the consumed final. Improving jitter robustness is future research requiring development-only fitting and a new untouched reserve before replacement. No organizer submission or remote publication is performed by these scripts.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    identities={name:hashlib.sha256((OUT/name).read_bytes()).hexdigest()
                for name in ['results.json','decomposition_results.json','diagnostic_plan.json','decomposition_plan.json']}
    (OUT/'report_sources.json').write_text(json.dumps(identities,indent=2)+'\n')
    print('All combined/decomposition results retained:',OUT/'REPORT.md')


if __name__=='__main__':
    main()
