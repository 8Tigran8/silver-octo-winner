"""Check prospective development gains against frozen v4 aggregate scores.

This reads public external-match metrics only. It does not score/open a reserve,
fit models, access organizer data or certify a final release.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / 'docs/evaluation/policy.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_inputs():
    policy = json.loads(POLICY.read_text())
    base = policy['baseline']
    for key, hash_key in [('cv_records', 'cv_records_sha256'), ('plan', 'plan_sha256')]:
        if sha(ROOT / base[key]) != base[hash_key]:
            raise ValueError('Frozen baseline records or folds changed')
    if sha(ROOT / 'assets/v4/manifest.json') != base['asset_manifest_sha256']:
        raise ValueError('Frozen v4 asset manifest changed')
    reserve = policy['reserve']
    if sha(ROOT / reserve['legacy_protocol']) != reserve['legacy_protocol_sha256']:
        raise ValueError('Original reserve split/protocol changed')
    return policy, json.loads((ROOT / base['plan']).read_text()), json.loads((ROOT / base['cv_records']).read_text())


def index_records(payload, candidate, policy, plan):
    if payload.get('plan_sha256') != policy['baseline']['plan_sha256']:
        raise ValueError('Candidate must use the frozen complete-game fold plan')
    expected = {(game, group['scenario'])
                for game in plan['training_games']
                for group in policy['development']['groups']
                if game.startswith(group['family'] + '/')}
    result = {}
    for row in payload['records']:
        if row['candidate'] != candidate:
            continue
        key = row['game'], row['scenario']
        if key in result:
            raise ValueError('Duplicate game/scenario score; augmentation is not an independent match')
        if key not in expected or row.get('family') != key[0].split('/')[0] or row.get('fold') != plan['folds'][key[0]]:
            raise ValueError('Unknown game/scenario, wrong source family or wrong excluded fold')
        for value in [row['spell_f1_mean'], row['macro_f1']] + [row['per_class'][label]['f1'] for label in ['H', 'A', 'D']]:
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('All official metric scores must be finite and within [0,1]')
        result[key] = row
    if set(result) != expected:
        raise ValueError('Missing game/scenario scores; do not drop difficult matches')
    return result


def check_provenance(provenance, policy, plan):
    known = set(plan['training_games'])
    allowed_extra = set(policy['development']['allowed_additional_training_games'])
    for fold in sorted(set(plan['folds'].values())):
        meta = provenance[f'fold{fold}']
        excluded = {game for game, number in plan['folds'].items() if number == fold}
        train = set(meta['train'])
        if train & excluded or not excluded <= set(meta['excluded']):
            raise ValueError('Scored fold leaks into declared training or exclusions are incomplete')
        if train - (known | allowed_extra) or train & {f'pff/{g}' for g in policy['reserve']['final_ids'] + policy['reserve']['dev_ids']}:
            raise ValueError('Unapproved/reserved training game')
        if meta.get('plan_sha256') != policy['baseline']['plan_sha256']:
            raise ValueError('Training provenance uses another fold plan')
        digest = meta.get('model_sha256', '')
        if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('A checkpoint SHA-256 is required for every fold')
    # This validates declared membership, not truth of an artifact's training
    # history. Inspect every inherited learned stage, normalizer and stack too.


def evaluate(payload, candidate, provenance):
    policy, plan, baseline_payload = load_inputs()
    baseline = index_records(baseline_payload, policy['baseline']['cv_candidate'], policy, plan)
    current = index_records(payload, candidate, policy, plan)
    check_provenance(provenance, policy, plan)
    return compare_records(policy, candidate, baseline, current)


def evaluate_paired_audit(payload, candidate, control, provenance):
    """Additional matched isolation audit; does not replace frozen v4 gates."""
    policy, plan, _ = load_inputs()
    baseline = index_records(payload, control, policy, plan)
    current = index_records(payload, candidate, policy, plan)
    check_provenance(provenance, policy, plan)
    result = compare_records(policy, candidate, baseline, current)
    result.update(comparison_control=control,
                  scope='Additional paired reconstruction audit under identical exclusions; immutable historical-v4 gates and untouched confirmation remain separately required.')
    return result


def compare_records(policy, candidate, baseline, current):
    """Apply the exact frozen numeric thresholds to an explicitly indexed pair."""
    rules = policy['development']
    groups = []
    checks = {}
    for group in rules['groups']:
        keys = [key for key in baseline if key[0].startswith(group['family'] + '/') and key[1] == group['scenario']]
        name = group['family'] + '/' + group['scenario']
        b = float(np.mean([baseline[k]['spell_f1_mean'] for k in keys]))
        c = float(np.mean([current[k]['spell_f1_mean'] for k in keys]))
        macro_gain = float(np.mean([current[k]['macro_f1'] - baseline[k]['macro_f1'] for k in keys]))
        class_gains = {label: float(np.mean([current[k]['per_class'][label]['f1'] - baseline[k]['per_class'][label]['f1'] for k in keys])) for label in ['H', 'A', 'D']}
        groups.append(dict(group, baseline=b, candidate=c, spell_gain=c-b, macro_gain=macro_gain, class_f1_gains=class_gains))
        checks[name + '/spell_guard'] = c-b >= -rules['max_cohort_spell_loss'] - 1e-12
        checks[name + '/macro_guard'] = macro_gain >= -rules['max_cohort_macro_loss'] - 1e-12
        checks[name + '/class_guard'] = min(class_gains.values()) >= -rules['max_cohort_perclass_f1_loss'] - 1e-12
    base_score = sum(g['weight'] * g['baseline'] for g in groups)
    score = sum(g['weight'] * g['candidate'] for g in groups)
    vectors = []
    per_game = []
    for family in ['original', 'skillcorner']:
        scenarios = [g['scenario'] for g in rules['groups'] if g['family'] == family and g['weight'] > 0]
        games = sorted({game for game, _ in baseline if game.startswith(family + '/')})
        deltas = []
        for game in games:
            b = float(np.mean([baseline[game, s]['spell_f1_mean'] for s in scenarios]))
            c = float(np.mean([current[game, s]['spell_f1_mean'] for s in scenarios]))
            deltas.append(c-b)
            per_game.append({'game': game, 'baseline': b, 'candidate': c, 'gain': c-b})
        vectors.append(np.array(deltas))
    rng = np.random.default_rng(rules['descriptive_bootstrap']['seed'])
    draws = sum(.5 * v[rng.integers(0, len(v), (rules['descriptive_bootstrap']['draws'], len(v)))].mean(1) for v in vectors)
    ci = np.quantile(draws, [.025, .975]).tolist()
    wins = sum(g['gain'] > 1e-10 for g in per_game)
    large_losses = sum(g['gain'] < -rules['large_game_loss_threshold'] - 1e-12 for g in per_game)
    q25_gain = float(np.quantile([g['candidate'] for g in per_game], .25) - np.quantile([g['baseline'] for g in per_game], .25))
    checks.update(meaningful_gain=score-base_score >= rules['min_composite_spell_gain'] - 1e-12,
                  majority_games=wins >= math.ceil(len(per_game) * rules['min_game_win_fraction']),
                  positive_descriptive_lower_bound=ci[0] > 0,
                  lower_quartile_guard=q25_gain >= -rules['max_lower_quartile_spell_loss'] - 1e-12,
                  large_game_losses=large_losses <= math.floor(len(per_game) * rules['max_large_game_loss_fraction']))
    return {'policy': policy['version'], 'policy_sha256': sha(POLICY), 'candidate': candidate,
            'status': 'ELIGIBLE_FOR_CONFIRMATION' if all(checks.values()) else 'KEEP_V4',
            'scope': 'Development quality gates only; candidate history/seeds, provider diagnostics, unopened-source confirmation and technical release checks require separate evidence.',
            'final_confirmation': 'NOT ASSESSED', 'baseline_composite': base_score,
            'candidate_composite': score, 'gain': score-base_score, 'descriptive_95ci': ci,
            'game_wins': wins, 'game_losses': sum(g['gain'] < -1e-10 for g in per_game),
            'large_game_losses': large_losses, 'lower_quartile_score_gain': q25_gain,
            'checks': checks, 'cohorts': groups, 'games': per_game}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scores', type=Path, required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--provenance', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Do not overwrite an evaluation receipt; use a new output path')
    result = evaluate(json.loads(args.scores.read_text()), args.candidate, json.loads(args.provenance.read_text()))
    result.update(scores_sha256=sha(args.scores), provenance_sha256=sha(args.provenance))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ['status', 'gain', 'game_wins', 'game_losses', 'final_confirmation']}, indent=2))


if __name__ == '__main__':
    main()
