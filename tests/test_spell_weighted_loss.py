import unittest
import numpy as np
import torch
from mtah import temporal
from tools.spell_weighted_loss import live_spell_weights,loss


class WeightedObjectiveTests(unittest.TestCase):
    def test_uniform_control_matches_original_loss_exactly(self):
        torch.manual_seed(7);z=torch.randn(2,2,20);gt=torch.randint(0,4,(2,20));w=torch.rand(2,20)
        torch.testing.assert_close(loss(z,gt,w,torch.ones_like(w),gt==0,False),temporal.loss(z,gt,w),rtol=0,atol=0)

    def test_short_weighting_is_team_symmetric_and_period_local(self):
        y=np.r_[np.zeros(2,int),np.ones(50,int),np.full(10,2)]
        w=live_spell_weights(y,[(0,len(y))]);self.assertEqual(w[0],4);self.assertEqual(w[2],1)
        np.testing.assert_array_equal(w,live_spell_weights(np.where(y<2,1-y,y),[(0,len(y))]))
        same=np.zeros(100,int);parts=[(0,2),(2,100)]
        self.assertEqual(live_spell_weights(same,parts)[0],4)

    def test_unknown_targets_do_not_affect_weighted_loss(self):
        z=torch.zeros(1,2,4);gt=torch.tensor([[0,1,2,3]]);w=torch.ones(1,4)
        before=loss(z,gt,w,w,torch.ones_like(gt),True)
        z[:,:,3]=100
        torch.testing.assert_close(before,loss(z,gt,w,w,torch.ones_like(gt),True),rtol=0,atol=0)


if __name__=='__main__':unittest.main()
