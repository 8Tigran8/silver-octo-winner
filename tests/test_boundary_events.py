import itertools
import unittest
import numpy as np
from mtah import boundary_events as event
from mtah.methods import boost


class BoundaryTests(unittest.TestCase):
    def test_variable_cost_path_matches_exhaustive_optimum(self):
        rng=np.random.default_rng(71);emissions=rng.normal(size=(5,4));costs=rng.uniform(.1,3,size=(5,4,4))
        for k in range(4):costs[:,k,k]=0
        path=event.duration_path(emissions,costs,[1]*4)
        score=lambda p:sum(emissions[t,k] for t,k in enumerate(p))-sum(costs[t,p[t-1],p[t]] for t in range(1,5))
        self.assertAlmostEqual(score(path),max(score(p) for p in itertools.product(range(4),repeat=5)))

    def test_constant_cost_duration_chain_exactly_matches_original(self):
        rng=np.random.default_rng(72)
        for n in [1,2,9,37]:
            for lengths in [[1,1,1,1],[2,3,2,3],[5,5,5,5]]:
                emissions=rng.normal(size=(n,4));cost=np.array([[0,9,9,np.inf],[9,0,np.inf,9],[9,9,0,np.inf],[9,9,np.inf,0]],float)
                expected=boost._viterbi_chain(emissions,cost,lengths)
                actual=event.duration_path(emissions,np.broadcast_to(cost,(n,4,4)),lengths)
                np.testing.assert_array_equal(actual,expected)

    def test_temporal_differences_do_not_cross_period_boundaries(self):
        x=np.zeros((14,2),np.float32);x[7:]=1000
        out=event.temporal_features(x,['a','b'],[(0,7),(7,14)],['a'])
        np.testing.assert_array_equal(out[:,2:],0)

    def test_unknown_reference_never_becomes_a_transition_target(self):
        gt=np.array([0,0,1,2,3,0,1,2],np.int8)
        np.testing.assert_array_equal(event.targets(gt,[(0,6),(6,8)]),[-1,0,1,2,-1,-1,-1,2])

    def test_transition_evidence_is_type_specific_and_never_rewards_change(self):
        params=dict(boost.DECODER,cost_ha=9,cost_in=9,cost_same=9,cost_other=9)
        q=np.array([[.2,.79,.01],[.2,.01,.79]])
        c=event.transition_costs(q,np.array([.98,.01,.01]),params,2)
        self.assertLess(c[0,0,1],c[0,0,2]);self.assertLess(c[1,0,2],c[1,0,1])
        self.assertTrue(np.all(c[np.isfinite(c)]>=0));self.assertTrue(np.isinf(c[:,0,3]).all())


if __name__=='__main__':unittest.main()
