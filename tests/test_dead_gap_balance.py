import unittest
import numpy as np
from tools.dead_gap_balance import balanced_weights,routed_probabilities
class DeadBalanceContracts(unittest.TestCase):
    def test_equal_train_class_mass_and_no_missing_class(self):
        y=np.array([0,1,1,1]);w=balanced_weights(y,np.array([2.,1.,1.,1.]));self.assertAlmostEqual(w[y==0].sum(),w[y==1].sum());self.assertAlmostEqual(w.sum(),4)
        with self.assertRaises(ValueError):balanced_weights(np.ones(3,dtype=int),np.ones(3))
    def test_routing_reconstructs_nn_and_preserves_observed(self):
        old=np.array([.4,.6,.8]);new=np.array([.2,.3,.4]);nd=np.array([.1,.5,.9]);c=dict(pd=.75*nd+.25*old,ph=np.array([.2,.4,.7]),seen=np.array([True,False,False]));focus=np.array([.1,.8,.2])
        pd,ph=routed_probabilities('balanced_50_focus',c,old,new,focus)
        self.assertEqual(pd[0],c['pd'][0]);self.assertTrue(np.allclose(pd[1:],(.5*nd+.5*new)[1:]));self.assertEqual(ph[0],c['ph'][0]);self.assertTrue(np.array_equal(ph[1:],focus[1:]))
    def test_team_swap_and_input_not_mutated(self):
        old=np.array([.4,.6]);new=np.array([.2,.3]);c=dict(pd=np.array([.2,.5]),ph=np.array([.3,.7]),seen=np.array([1,0],bool));focus=np.array([.4,.8]);before={k:v.copy() for k,v in c.items()}
        a=routed_probabilities('activity_75_focus',c,old,new,focus);b=routed_probabilities('activity_75_focus',dict(c,ph=1-c['ph']),old,new,1-focus)
        self.assertTrue(np.array_equal(a[0],b[0]));self.assertTrue(np.allclose(a[1],1-b[1]));self.assertTrue(all(np.array_equal(c[k],v) for k,v in before.items()))
