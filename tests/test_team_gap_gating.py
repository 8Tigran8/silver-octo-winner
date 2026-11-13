"""The alternate head may influence only missing-ball frames inside periods."""
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
from team_gap_gating import gap_weight


class GapTests(unittest.TestCase):
    def test_visible_frames_and_outside_period_are_unchanged(self):
        seen=np.ones(100,bool);seen[20:60]=False;seen[80:]=False
        for taper in [False,True]:
            weights=gap_weight(seen,[(10,70)],.5,taper)
            self.assertTrue((weights[seen]==0).all())
            self.assertTrue((weights[70:]==0).all())
            self.assertTrue(((weights>=0)&(weights<=.5)).all())

    def test_taper_is_time_symmetric_and_reduces_boundary_influence(self):
        seen=np.ones(100,bool);seen[20:80]=False
        weights=gap_weight(seen,[(0,100)],1.,True)
        np.testing.assert_allclose(weights,weights[::-1])
        self.assertGreater(weights[49],weights[20])


if __name__=="__main__":unittest.main()
