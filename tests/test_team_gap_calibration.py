import unittest
import torch
from tools.team_gap_calibration import focal_posterior,FocalOwnershipLink,weighted_ownership_bce
class CalibrationContracts(unittest.TestCase):
    def test_stationary_binary_focal_risk(self):
        p=torch.linspace(.02,.98,49,dtype=torch.float64,requires_grad=True);q=focal_posterior(p).detach()
        risk=-q*(1-p)*torch.log(p)-(1-q)*p*torch.log1p(-p)
        risk.sum().backward();self.assertLess(p.grad.abs().max().item(),1e-12)
    def test_link_monotone_and_team_symmetric(self):
        p=torch.linspace(.001,.999,999,dtype=torch.float64);q=focal_posterior(p)
        self.assertTrue(torch.all(q[1:]>q[:-1]));self.assertTrue(torch.allclose(q,1-focal_posterior(1-p)))
        self.assertEqual(focal_posterior(torch.tensor(.5)).item(),.5)
    def test_only_ownership_head_changes(self):
        class Identity(torch.nn.Module):
            context=256
            def forward(self,x):return x
        model=FocalOwnershipLink(Identity());x=torch.tensor([[[.1,.2,.3],[-1.,0.,1.]]]);y=model(x)
        self.assertTrue(torch.equal(x[:,0],y[:,0]));self.assertTrue(torch.allclose(torch.sigmoid(y[:,1]),focal_posterior(torch.sigmoid(x[:,1]))));self.assertEqual(model.context,256)
    def test_bce_masks_dead_and_unknown_and_preserves_swap(self):
        x=torch.randn(1,2,4,requires_grad=True);gt=torch.tensor([[0,1,2,3]]);w=torch.ones(1,4);m=torch.tensor([[True,False,True,True]])
        l=weighted_ownership_bce(x,gt,w,m);sw=x.detach().clone();sw[:,1]*=-1
        self.assertAlmostEqual(l.item(),weighted_ownership_bce(sw,torch.where(gt<2,1-gt,gt),w,m).item(),places=6)
        l.backward();self.assertTrue(torch.all(x.grad[:,0]==0));self.assertTrue(torch.all(x.grad[:,:,2:]==0))
