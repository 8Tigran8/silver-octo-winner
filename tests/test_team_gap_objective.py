import unittest
import numpy as np
import torch
from tools.team_gap_objective import ownership_loss,short_centres
from tools.team_gap_specialist import guard
class OwnershipStudyContracts(unittest.TestCase):
    def test_dead_and_unknown_targets_have_no_ownership_gradient(self):
        x=torch.zeros((1,2,4),requires_grad=True);y=torch.tensor([[0,1,2,3]]);w=torch.ones((1,4));m=torch.ones((1,4),dtype=torch.bool)
        ownership_loss(x,y,w,m,True).backward()
        self.assertTrue(torch.equal(x.grad[:,0],torch.zeros((1,4))));self.assertTrue(torch.equal(x.grad[:,:,2:],torch.zeros((1,2,2))))
    def test_team_swap_preserves_loss_and_uniform_is_weighted_bce(self):
        x=torch.tensor([[[.2,.4,.1],[.5,-.3,.2]]]);y=torch.tensor([[0,1,0]]);w=torch.tensor([[1.,2.,3.]]);m=torch.tensor([[True,False,True]])
        swapped=x.clone();swapped[:,1]*=-1
        for focused in [False,True]:self.assertAlmostEqual(ownership_loss(x,y,w,m,focused).item(),ownership_loss(swapped,1-y,w,m,focused).item(),places=6)
        target=(y==0).float();expected=(torch.nn.functional.binary_cross_entropy_with_logits(x[:,1],target,reduction='none')*w).sum()/w.sum()
        self.assertEqual(ownership_loss(x,y,w,m,False),expected)
    def test_short_sampling_respects_periods_and_excludes_dead_unknown(self):
        gt=np.array([0]*10+[2]*5+[3]*5+[1]*20+[0]*10);centres=short_centres(gt,[(0,20),(20,50)])
        self.assertEqual([x.tolist() for x in centres],[[4],[44]])
    def test_audit_fixtures_are_blocked_from_fit_and_score(self):
        for g in ['10502','3812']:
            with self.assertRaises(ValueError):guard(g,'development')
            with self.assertRaises(ValueError):guard('pff/'+g,'training')
if __name__=='__main__':unittest.main()
