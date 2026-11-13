"""Expanded training must exclude whole games and preserve fixed channels."""
import unittest
import numpy as np
from mtah import expanded_team as expanded


class ExpandedTests(unittest.TestCase):
    def test_all_augmented_copies_share_game_exclusion(self):
        names=expanded.games()
        self.assertEqual(len(names),19)
        for test in names:
            train=[n for n in names if n!=test]
            expanded.assert_training_split(names,train,test)
            with self.assertRaisesRegex(ValueError,"leaked"):
                expanded.assert_training_split(names,train+[test],test)

    def test_competition_and_fresh_games_cannot_be_training_inputs(self):
        for name in ["original/BAR-ATH","skillcorner/2068","skillcorner/1874553"]:
            with self.assertRaisesRegex(ValueError,"whitelisted"):
                expanded.public_game(name)

    def test_taper_keeps_visible_predictions_and_bounds_weights(self):
        seen=np.ones(100,bool);seen[20:80]=False
        old=np.full(100,.2);new=np.full(100,.8)
        blended=expanded.blend_team(old,new,seen,[(0,100)],{"weight":1.,"gate":"taper_1s"})
        np.testing.assert_array_equal(blended[seen],old[seen])
        self.assertTrue(((blended>=old)&(blended<=new)).all())


if __name__=="__main__":unittest.main()
