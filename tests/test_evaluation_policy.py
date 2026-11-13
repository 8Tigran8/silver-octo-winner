"""Reject evidence that would make a misleading development improvement claim."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('check_evaluation', ROOT / 'tools/check_evaluation.py')
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


class EvaluationPolicyTests(unittest.TestCase):
    def setUp(self):
        self.policy, self.plan, self.baseline = CHECK.load_inputs()
        self.provenance = json.loads((ROOT / 'docs/experiments/temporal/training_provenance.json').read_text())
        self.payload = copy.deepcopy(self.baseline)
        self.payload['records'] = [r for r in self.payload['records'] if r['candidate'] == 'fusion50']

    def test_identical_predictions_cannot_be_called_an_improvement(self):
        result = CHECK.evaluate(self.payload, 'fusion50', self.provenance)
        self.assertEqual(result['status'], 'KEEP_V4')
        self.assertEqual(result['gain'], 0)
        self.assertEqual(result['final_confirmation'], 'NOT ASSESSED')

    def test_missing_hard_match_and_duplicate_are_rejected(self):
        missing = copy.deepcopy(self.payload); missing['records'].pop()
        duplicate = copy.deepcopy(self.payload); duplicate['records'].append(duplicate['records'][0])
        for value in [missing, duplicate]:
            with self.assertRaises(ValueError):
                CHECK.evaluate(value, 'fusion50', self.provenance)

    def test_training_leak_and_reserved_country_game_are_rejected(self):
        for leaked in [self.provenance['fold0']['excluded'][0], 'pff/3812']:
            provenance = copy.deepcopy(self.provenance); provenance['fold0']['train'].append(leaked)
            with self.assertRaises(ValueError):
                CHECK.evaluate(self.payload, 'fusion50', provenance)

    def test_nan_and_wrong_fold_are_rejected(self):
        for field, value in [('spell_f1_mean', float('nan')), ('fold', 99)]:
            payload = copy.deepcopy(self.payload); payload['records'][0][field] = value
            with self.assertRaises(ValueError):
                CHECK.evaluate(payload, 'fusion50', self.provenance)

    def test_reserve_countries_and_all_64_match_ids_remain_isolated(self):
        reserve = json.loads((ROOT / self.policy['reserve']['legacy_protocol']).read_text())
        partitions = ['training', 'development', 'final_test', 'country_bridge_excluded_from_training_and_selection']
        ids = [game['game_id'] for part in partitions for game in reserve[part]]
        self.assertEqual(len(ids), 64)
        self.assertEqual(len(set(ids)), 64)
        final_countries = {game[side] for game in reserve['final_test'] for side in ['home', 'away']}
        used_countries = {game[side] for part in ['training', 'development'] for game in reserve[part] for side in ['home', 'away']}
        self.assertFalse(final_countries & used_countries)
        prospective = self.policy['reserve']
        for original, current in [('training', 'train_ids'), ('development', 'dev_ids'), ('final_test', 'final_ids'), ('country_bridge_excluded_from_training_and_selection', 'excluded_bridge_ids')]:
            self.assertEqual(set(prospective[current]), {g['game_id'] for g in reserve[original]})

    def test_average_gain_cannot_hide_failed_dead_ball_detection(self):
        payload = copy.deepcopy(self.payload)
        for row in payload['records']:
            row['spell_f1_mean'] = min(row['spell_f1_mean'] + .04, 1)
            row['per_class']['D']['f1'] -= .03
        result = CHECK.evaluate(payload, 'fusion50', self.provenance)
        self.assertTrue(result['checks']['meaningful_gain'])
        self.assertEqual(result['status'], 'KEEP_V4')
        self.assertFalse(result['checks']['original/matched_seed0/class_guard'])

    def test_positive_development_result_never_becomes_final_test_pass(self):
        payload = copy.deepcopy(self.payload)
        for row in payload['records']:
            row['spell_f1_mean'] = min(row['spell_f1_mean'] + .04, 1)
        result = CHECK.evaluate(payload, 'fusion50', self.provenance)
        self.assertEqual(result['status'], 'ELIGIBLE_FOR_CONFIRMATION')
        self.assertEqual(result['final_confirmation'], 'NOT ASSESSED')


if __name__ == '__main__':
    unittest.main()
