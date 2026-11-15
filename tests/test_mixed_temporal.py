import sys
from pathlib import Path
import unittest
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from mtah import temporal,mixed_temporal


class MixedTemporalTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(2);torch.set_num_threads(1)

    def test_short_control_exactly_matches_original_inference(self):
        old=temporal.Temporal();new=mixed_temporal.MixedTemporal({'context_steps':64,'dilations':[1,2,4,8,16]})
        new.load_state_dict(old.state_dict());rng=np.random.default_rng(4);x=rng.normal(size=(800,108)).astype(np.float32);xs=rng.normal(size=x.shape).astype(np.float32);mean=np.zeros(108);scale=np.ones(108)
        a=temporal.probabilities(old,x,xs,[(10,400),(450,790)],mean,scale)
        b=mixed_temporal.probabilities(new,x,xs,[(10,400),(450,790)],mean,scale)
        for actual,expected in zip(a,b):np.testing.assert_array_equal(actual,expected)

    def test_long_context_chunk_boundaries_do_not_change_prediction(self):
        model=mixed_temporal.MixedTemporal({'context_steps':256,'dilations':[1,2,4,8,16,32,64]})
        rng=np.random.default_rng(4);x=rng.normal(size=(700,108)).astype(np.float32);xs=rng.normal(size=x.shape).astype(np.float32);mean=np.zeros(108);scale=np.ones(108)
        a=mixed_temporal.probabilities(model,x,xs,[(0,700)],mean,scale,core=64)
        b=mixed_temporal.probabilities(model,x,xs,[(0,700)],mean,scale,core=700)
        for actual,expected in zip(a,b):np.testing.assert_allclose(actual,expected,rtol=1e-5,atol=1e-6)

    def test_unknown_targets_create_no_boundary_loss_or_gradient(self):
        logits=torch.randn(2,2,6,requires_grad=True);unknown=torch.full((2,6),3);weights=torch.ones(2,6)
        loss=mixed_temporal.boundary_loss(logits,unknown,weights);loss.backward()
        self.assertEqual(float(loss.detach()),0.);self.assertEqual(float(logits.grad.abs().sum()),0.)
        logits.grad=None;truth=torch.tensor([[3,0,0,1,1,3],[3,2,2,0,0,3]])
        loss=mixed_temporal.boundary_loss(logits,truth,weights);loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all());self.assertEqual(float(logits.grad[:,:,[0,5]].abs().sum()),0.)


if __name__=='__main__':unittest.main()
