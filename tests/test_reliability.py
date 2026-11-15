import unittest
import numpy as np
from mtah import reliability


class ReliabilityTests(unittest.TestCase):
    def test_team_swap_preserves_dead_and_complements_home(self):
        rng=np.random.default_rng(3);bd,bh,nd,nh=rng.random((4,100));gap=rng.random(100)*30;seen=rng.random(100)
        specs=[{'kind':'gap','scale_seconds':5,'low':.35,'high':.8},{'kind':'visibility','low':.35,'high':.8},{'kind':'confidence','low':.25,'high':.8}]
        for spec in specs:
            d,h=reliability.mix(spec,bd,bh,nd,nh,gap,seen);ds,hs=reliability.mix(spec,bd,1-bh,nd,1-nh,gap,seen)
            np.testing.assert_allclose(d,ds);np.testing.assert_allclose(h,1-hs)

    def test_missing_geometry_is_bounded_and_finite(self):
        p=np.array([.1,.4,.9]);gap=np.array([np.nan,np.inf,-np.inf]);seen=gap.copy()
        for spec in [{'kind':'gap','scale_seconds':5,'low':.35,'high':.8},{'kind':'visibility','low':.35,'high':.8}]:
            wd,wh=reliability.weights(spec,p,p,1-p,1-p,gap,seen)
            self.assertTrue(np.isfinite(wd).all());self.assertTrue(((wd>=.35)&(wd<=.8)).all())

    def test_constant_mixture_is_exact_v4_probability_control(self):
        a=np.linspace(0,1,100);b=1-a
        d,h=reliability.mix({'kind':'fixed','dead':.5,'home':.5},a,a,b,b,a,a)
        np.testing.assert_array_equal(d,.5*a+.5*b);np.testing.assert_array_equal(h,d)


if __name__=='__main__':unittest.main()
