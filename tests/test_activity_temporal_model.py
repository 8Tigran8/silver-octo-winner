import unittest
import numpy as np
import torch
from tools.activity_temporal_model import ActivityModel,loss,probabilities
class ActivityTemporalContracts(unittest.TestCase):
    def test_loss_uses_only_missing_valid_targets(self):
        x=torch.zeros(1,4,requires_grad=True);y=torch.tensor([[0,2,3,2]]);missing=torch.tensor([[True,True,True,False]])
        loss(x,y,missing).backward();self.assertLess(x.grad[0,1].item(),0);self.assertGreater(x.grad[0,0].item(),0);self.assertTrue(torch.all(x.grad[0,2:]==0))
    def test_temporal_inference_resets_at_periods_and_handles_nan(self):
        torch.set_num_threads(1);torch.manual_seed(1);model=ActivityModel('temporal');rng=np.random.default_rng(5);x=rng.normal(size=(200,64)).astype(np.float32);x[5:8]=np.nan;mean=np.zeros(64,np.float32);scale=np.ones(64,np.float32);parts=[(0,100),(100,200)]
        p=probabilities(model,x,parts,mean,scale);changed=x.copy();changed[100:]*=5;q=probabilities(model,changed,parts,mean,scale)
        self.assertTrue(np.isfinite(p).all());self.assertTrue(np.array_equal(p[:100],q[:100]));self.assertEqual(model.context,128)
    def test_pointwise_network_has_no_extra_temporal_context(self):
        torch.manual_seed(1);model=ActivityModel('pointwise').eval();x=torch.randn(1,128,20);y=x.clone();y[:,:,10:]+=3
        self.assertTrue(torch.equal(model(x)[:,:10],model(y)[:,:10]));self.assertEqual(model.context,0)
