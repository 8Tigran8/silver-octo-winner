"""Whole-game isolation and a genuinely unchanged capacity/data comparison."""
import sys
import json
import numpy as np
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import more_data_trial as trial
from more_data_decoder import temperature


class MoreDataTests(unittest.TestCase):
    def test_calibration_preserves_team_swap_symmetry(self):
        p=np.linspace(.01,.99,101)
        np.testing.assert_array_equal(temperature(p,1.),p)
        for value in [.75,1.25]:
            q=temperature(p,value)
            np.testing.assert_allclose(temperature(1-p,value),1-q,atol=1e-12)
            self.assertTrue((np.diff(q)>0).all())
            self.assertAlmostEqual(float(temperature(np.array([.5]),value)[0]),.5)

    def test_complete_exclusion_for_every_scored_game(self):
        p=trial.plan()
        for game in p['evaluation_games']:
            names=trial.split(game)
            self.assertEqual(len(names),28)
            self.assertNotIn(game,names)
            self.assertEqual(set(names)|{game},set(p['training_games']))

    def test_new_training_games_are_never_claimed_as_holdout(self):
        p=trial.plan()
        self.assertEqual(len(p['new_training_games']),10)
        self.assertTrue(set(p['new_training_games']).isdisjoint(p['evaluation_games']))
        self.assertEqual(len(trial.split(None)),29)
        for game in ['original/BAR-ATH','skillcorner/2068','pff/10517']:
            with self.assertRaises(ValueError):trial.split(game)
            with self.assertRaises(ValueError):trial.row_path(game)

    def test_augmentation_siblings_keep_same_game_key(self):
        p=trial.plan()
        for excluded in p['evaluation_games']:
            parts=[(game,scenario) for game in trial.split(excluded)
                   for scenario in ('natural','corrupt','team_swapped')]
            self.assertFalse(any(game==excluded for game,_ in parts))

    def test_future_pff_holdout_excludes_entire_national_teams(self):
        p=json.loads((trial.PLAN.parent/'pff_protocol.json').read_text())
        countries=set(p['heldout_countries'])
        sets=[]
        for key,count in [('training',36),('development',10),('final_test',12),
                          ('country_bridge_excluded_from_training_and_selection',6)]:
            rows=p[key];self.assertEqual(len(rows),count)
            sets.append({r['game_id'] for r in rows})
            if key in ('training','development'):
                self.assertTrue(all(not countries.intersection([r['home'],r['away']]) for r in rows))
            if key=='final_test':
                self.assertTrue(all({r['home'],r['away']} <= countries for r in rows))
        self.assertEqual(len(set.union(*sets)),64)
        self.assertEqual(sum(len(s) for s in sets),64)
        self.assertNotIn('10517',sets[2])


if __name__=='__main__':unittest.main()
