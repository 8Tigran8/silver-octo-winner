"""Physical eligibility checks for the isolated, unpromoted gap experiment."""
import unittest
import numpy as np
from tools.controlled_gap_study import fill_controlled_gaps
from tools.audit_synthetic_geometry import controlled_tracking

class ControlledGapTests(unittest.TestCase):
    def test_stationary_offset_recovered_without_mutating_input(self):
        d=controlled_tracking(0);original=d['ball'].copy();d['ball'][100:120,:2]=np.nan
        repaired,receipt=fill_controlled_gaps(d,[(0,1500)],2)
        np.testing.assert_allclose(repaired['ball'],original,atol=1e-14)
        self.assertTrue(np.isnan(d['ball'][100:120,:2]).all());self.assertEqual(receipt['filled_frames'],20)

    def test_ambiguous_or_unobserved_player_and_period_edges_not_filled(self):
        for kind in ['other_owner','missing_player','speed_jump','period_boundary','too_long']:
            with self.subTest(kind=kind):
                d=controlled_tracking(0);d['ball'][100:160,:2]=np.nan;periods=[(0,1500)];seconds=10
                if kind=='other_owner':d['ball'][160,:2]=d['players'][160,17,:2]
                if kind=='missing_player':d['players'][120,6,:2]=np.nan
                if kind=='speed_jump':d['players'][120,6,0]+=.2
                if kind=='period_boundary':periods=[(0,130),(130,1500)]
                if kind=='too_long':seconds=2
                repaired,r=fill_controlled_gaps(d,periods,seconds)
                self.assertEqual(r['filled_frames'],0);np.testing.assert_array_equal(repaired['ball'],d['ball'])

if __name__=='__main__':unittest.main()
