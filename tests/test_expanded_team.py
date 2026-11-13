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

    def test_independent_periods_make_features_label_invariant(self):
        players=np.zeros((300,22,3),np.float32)
        players[:,:11,0]=.25;players[:,11:,0]=.75
        players[:,:,1]=np.linspace(.1,.9,22)
        data={"players":players,"ball":np.tile([.5,.5,0.],(300,1)),
              "slot_team":np.repeat([0,1],11),"pitch":(105.,68.),"gt":np.zeros(300,np.int8)}
        before,swapped,names=expanded.feature_pair(data,[(0,150),(150,300)])
        data["gt"][:]=3
        after,after_swapped,after_names=expanded.feature_pair(data,[(0,150),(150,300)])
        self.assertEqual(names,after_names)
        np.testing.assert_allclose(before,after,equal_nan=True)
        np.testing.assert_allclose(swapped,after_swapped,equal_nan=True)


if __name__=="__main__":unittest.main()
