import unittest
import numpy as np
from mtah.robust_temporal import ood_alpha,combine


class RobustTemporalTests(unittest.TestCase):
    def test_outlier_shrinkage_and_period_reset(self):
        x=np.r_[np.zeros((10,12)),np.full((10,12),10.)];spec={'low':2,'high':4,'top_features':8,'smooth_steps':25}
        a=ood_alpha(x,np.zeros(12),np.ones(12),[(0,10),(10,20)],spec)
        np.testing.assert_array_equal(a[:10],1);np.testing.assert_array_equal(a[10:],0)

    def test_no_observations_means_no_long_model_weight(self):
        a=ood_alpha(np.full((10,12),np.nan),np.zeros(12),np.ones(12),[(0,10)],{'low':2,'high':4,'top_features':8,'smooth_steps':25})
        np.testing.assert_array_equal(a,0)

    def test_channels_are_team_symmetric(self):
        rng=np.random.default_rng(3);od,oh,nd,nh=rng.random((4,20));x=rng.normal(size=(4,12));meta={'full_length':20,'full_segments':[(0,20)],'segments':[(0,4)]}
        for spec in [{'kind':'alpha','dead':.75,'home':.5},{'kind':'temperature','temperature':1.5},{'kind':'ood','low':2,'high':4,'top_features':8,'smooth_steps':25}]:
            d,h=combine(spec,od,oh,nd,nh,x,np.zeros(12),np.ones(12),meta);ds,hs=combine(spec,od,1-oh,nd,1-nh,x,np.zeros(12),np.ones(12),meta)
            np.testing.assert_allclose(d,ds);np.testing.assert_allclose(h,1-hs)


if __name__=='__main__':unittest.main()
