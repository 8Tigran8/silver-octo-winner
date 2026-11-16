import unittest
import numpy as np
import torch
from mtah.mixed_temporal import MixedTemporal,probabilities
from tools.team_gap_compact import SPEC,from_parent
class CompactContracts(unittest.TestCase):
    def test_parent_blocks_and_head_copied_exactly(self):
        parent=MixedTemporal(dict(dilations=[1,2,4,8,16,32,64],context_steps=256));model=from_parent(parent)
        for k,v in model.state_dict().items():
            old='net.9.'+k[len('net.7.'):] if k.startswith('net.7.') else k
            self.assertTrue(torch.equal(v,parent.state_dict()[old]))
        self.assertEqual(1+4*sum(SPEC['dilations']),125);self.assertGreaterEqual(model.context,62)
    def test_inference_is_period_local_and_team_symmetric(self):
        torch.set_num_threads(1);torch.manual_seed(8);model=MixedTemporal(SPEC)
        rng=np.random.default_rng(2);x=rng.normal(size=(180,108)).astype('float32');xs=rng.normal(size=x.shape).astype('float32');mean=np.zeros(108,np.float32);scale=np.ones(108,np.float32);parts=[(0,90),(90,180)]
        p=probabilities(model,x,xs,parts,mean,scale);sw=probabilities(model,xs,x,parts,mean,scale)
        self.assertTrue(np.allclose(p[0],sw[0]));self.assertTrue(np.allclose(p[1],1-sw[1]))
        changed=x.copy();changed[90:]*=10;q=probabilities(model,changed,xs,parts,mean,scale)
        for a,b in zip(p,q):self.assertTrue(np.array_equal(a[:90],b[:90]))
