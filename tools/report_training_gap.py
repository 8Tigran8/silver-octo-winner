"""Describe frozen-model train/development/new-match scores without selection."""
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from pff_reader import sha
from pff_unseen_audit import check_freeze, DOC, IDS


def stats(rows):
    result = {'matches': len(rows)}
    for key in ['spell_f1_mean', 'macro_f1']:
        values = np.array([r[key] for r in rows])
        result[key] = dict(mean=float(values.mean()), median=float(np.median(values)),
                           minimum=float(values.min()), maximum=float(values.max()),
                           sample_std=float(values.std(ddof=1)))
    return result


def main():
    check_freeze()
    train_path = ROOT / 'reports/training_gap_audit/results.json'
    dev_path = ROOT / 'validation/observation_regression.json'
    new_path = DOC / 'results.json'
    train = json.loads(train_path.read_text())
    if train['freeze_sha256'] != sha(DOC / 'freeze.json'):
        raise ValueError('Training inference used a different freeze')
    if train['plan_sha256'] != sha(DOC / 'training_gap_plan.json'):
        raise ValueError('Training audit plan changed')
    protocol = json.loads((ROOT / 'docs/experiments/more-data/pff_protocol.json').read_text())
    training = train['records']
    development = [r for r in json.loads(dev_path.read_text())['records']
                   if r['reference'] == 'canonical' and r['scenario'] == 'natural_detected']
    unseen = [r for r in json.loads(new_path.read_text())['records']
              if r['reference'] == 'canonical' and r['scenario'] == 'natural_detected'
              and r['candidate'] == 'boundary102']
    expected = [set(g['game_id'] for g in protocol['training']),
                set(g['game_id'] for g in protocol['development']), set(IDS)]
    for rows, ids in zip([training, development, unseen], expected):
        if len(rows) != len(ids) or {r['game'] for r in rows} != ids:
            raise ValueError('Incomplete, duplicate or wrong match registry')
    if any(expected[i] & expected[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError('Overlapping comparison groups')
    # References must still match their original geometry/reference preflight.
    for row in training:
        game = row['game']
        receipt = json.loads((ROOT / 'reports/pff_access' / f'kloppy_{game}_preflight.json').read_text())
        reference = ROOT / 'reports/pff_access/references' / f'kloppy_{game}.npz'
        if row['reference_sha256'] != receipt['reference_cache_sha256'] or sha(reference) != row['reference_sha256']:
            raise ValueError('Reference identity mismatch: ' + game)
        if row['geometry_sha256'] != receipt['geometry_cache_sha256']:
            raise ValueError('Geometry identity mismatch: ' + game)
    groups = {name: stats(rows) for name, rows in zip(
        ['training_in_sample', 'development_reused', 'unseen_match_audit'],
        [training, development, unseen])}
    result = dict(
        interpretation='Descriptive equal-match comparison, not a fitted generalization bound. '
        'The training/test gap mixes fitting optimism, match difficulty, schedule and country mix; '
        'a small gap cannot establish absence of overfitting. All groups share provider proxies.',
        freeze_sha256=sha(DOC / 'freeze.json'),
        source_files_sha256={str(p.relative_to(ROOT)): sha(p) for p in [train_path, dev_path, new_path]},
        groups=groups,
        train_minus_unseen={k: groups['training_in_sample'][k]['mean'] - groups['unseen_match_audit'][k]['mean']
                            for k in ['spell_f1_mean', 'macro_f1']},
        development_minus_unseen={k: groups['development_reused'][k]['mean'] - groups['unseen_match_audit'][k]['mean']
                                  for k in ['spell_f1_mean', 'macro_f1']},
        training_records=training)
    (DOC / 'training_gap.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'training_records'}, indent=2))


if __name__ == '__main__':
    main()
