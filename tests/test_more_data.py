"""Whole-game isolation and a genuinely unchanged capacity/data comparison."""
import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import more_data_trial as trial


class MoreDataTests(unittest.TestCase):
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


if __name__=='__main__':unittest.main()
