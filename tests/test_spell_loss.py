import unittest
import numpy as np
import torch
from mtah.spell_loss import training_weights,weighted


class SpellLossTests(unittest.TestCase):
    def test_short_possessions_receive_more_weight_and_unknown_none(self):
        y=np.r_[np.zeros(100,dtype=int),np.full(5,3),np.ones(10,dtype=int),np.full(20,2)]
        w=training_weights(y,[(0,len(y))]);self.assertTrue((w[100:105]==0).all());self.assertGreater(w[108],w[50]);self.assertTrue(((w>=0)&(w<=4)).all())

    def test_swapping_home_and_away_preserves_weights(self):
        y=np.repeat([0,1,2,3,1,0],[12,30,10,4,7,25]);swapped=np.where(y<2,1-y,y)
        np.testing.assert_array_equal(training_weights(y,[(0,len(y))]),training_weights(swapped,[(0,len(y))]))

    def test_no_transition_weight_across_independent_periods(self):
        y=np.repeat([0,1],30);w=training_weights(y,[(0,30),(30,60)])
        self.assertEqual(w[29],w[10]);self.assertEqual(w[30],w[50])

    def test_unknown_targets_supply_no_gradient_even_with_positive_input_weights(self):
        x=torch.zeros((1,2,4),requires_grad=True);y=torch.tensor([[0,1,2,3]]);w=torch.ones((1,4))
        weighted(x,y,w).backward();self.assertTrue((x.grad[:,:,3]==0).all());self.assertNotEqual(float(x.grad[0,0,0]),0)


if __name__=='__main__':unittest.main()
