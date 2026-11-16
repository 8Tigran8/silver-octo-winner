import unittest
import numpy as np
import torch
from tools.dead_gap_objective import hard_centres,dead_loss
class DeadGapObjectiveContracts(unittest.TestCase):
    def test_centres_require_long_gap_and_live_reference_with_period_reset(self):
        gt=np.r_[np.zeros(60),np.full(60,2)].astype(int);seen=np.zeros(120,bool)
        c=hard_centres(gt,seen,[(0,60),(60,120)]);self.assertEqual(c[0].tolist(),list(range(0,60,5)));self.assertEqual(c[1].tolist(),[])
        self.assertEqual([len(x) for x in hard_centres(gt[:60],seen[:60],[(0,30),(30,60)])],[0,0])
    def test_dead_objective_does_not_train_ownership_or_unknown(self):
        x=torch.zeros(1,2,4,requires_grad=True);gt=torch.tensor([[0,1,2,3]]);m=torch.ones(1,4,dtype=torch.bool);dead_loss(x,gt,m,True).backward()
        self.assertTrue(torch.all(x.grad[:,1]==0));self.assertTrue(torch.all(x.grad[:,:,3]==0));self.assertGreater(x.grad[0,0,0].item(),0);self.assertLess(x.grad[0,0,2].item(),0)
    def test_team_swap_preserves_dead_loss(self):
        x=torch.randn(1,2,4);gt=torch.tensor([[0,1,2,3]]);m=torch.tensor([[1,0,1,1]],dtype=torch.bool)
        for focus in [False,True]:self.assertEqual(dead_loss(x,gt,m,focus),dead_loss(x,torch.where(gt<2,1-gt,gt),m,focus))

    def test_uniform_loss_matches_parent_dead_component(self):
        x=torch.randn(1,2,4);gt=torch.tensor([[0,1,2,3]]);m=torch.tensor([[1,0,1,1]],dtype=torch.bool);reliability=torch.tensor([[1.,1.,.25,1.]])
        w=(gt<3)*reliability*(1+m.float());raw=torch.nn.functional.binary_cross_entropy_with_logits(x[:,0],(gt==2).float(),reduction='none')
        self.assertTrue(torch.equal(dead_loss(x,gt,m,False,reliability),(raw*w).sum()/w.sum()))
