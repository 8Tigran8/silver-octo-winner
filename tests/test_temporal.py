import sys
from pathlib import Path
import unittest
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from mtah import temporal
from temporal_trial import game_split,plan,normalization


class TemporalTests(unittest.TestCase):
    def test_entire_matches_stay_out_of_training(self):
        p=plan();seen=[]
        for k in range(5):
            train,test=game_split('fold'+str(k));self.assertFalse(set(train)&set(test));seen+=test
        self.assertEqual(sorted(seen),sorted(p['training_games']))
        for key,prefix in [('provider_dfl','original/idsse_'),('provider_skill','skillcorner/')]:
            train,test=game_split(key);self.assertTrue(all(g.startswith(prefix) for g in train));self.assertTrue(all(not g.startswith(prefix) for g in test))

    def test_normalization_does_not_read_reference_values(self):
        x=np.arange(216,dtype=np.float32).reshape(2,108);x[0,0]=np.nan
        row={'x':x,'xs':x[::-1],'gt':np.array([0,1])};mean,scale=normalization({'train':[row]});row['gt'][:]=3
        a,b=normalization({'train':[row]});np.testing.assert_array_equal(a,mean);np.testing.assert_array_equal(b,scale)
        self.assertTrue(np.isfinite(temporal.normalize(x,mean,scale)).all())

    def test_predictions_reset_at_periods_and_are_team_symmetric(self):
        torch.set_num_threads(1);torch.manual_seed(4);model=temporal.Temporal(channels=4,dropout=0)
        rng=np.random.default_rng(4);x=rng.normal(size=(220,108)).astype(np.float32);xs=rng.normal(size=x.shape).astype(np.float32);mean=np.zeros(108,np.float32);scale=np.ones(108,np.float32);spans=[(0,100),(120,220)]
        pd,ph=temporal.probabilities(model,x,xs,spans,mean,scale,core=50)
        sd,sh=temporal.probabilities(model,xs,x,spans,mean,scale,core=50)
        np.testing.assert_allclose(pd,sd,atol=1e-7);np.testing.assert_allclose(ph,1-sh,atol=1e-7)
        x[120:]+=100;xs[120:]-=100;newd,newh=temporal.probabilities(model,x,xs,spans,mean,scale,core=50)
        np.testing.assert_array_equal(pd[:100],newd[:100]);np.testing.assert_array_equal(ph[:100],newh[:100])

    def test_unknown_targets_provide_no_gradient(self):
        logits=torch.zeros((1,2,10),requires_grad=True);gt=torch.full((1,10),3);w=torch.ones((1,10));value=temporal.loss(logits,gt,w);value.backward();self.assertEqual(float(value.detach()),0);self.assertEqual(float(logits.grad.abs().sum()),0)


if __name__=='__main__':unittest.main()
