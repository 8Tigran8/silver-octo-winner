import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import numpy as np
from possession_emission_study import weights,swapped,blend
class EmissionTests(unittest.TestCase):
 def test_duration_weights_respect_unknown_and_periods(self):
  gt=np.r_[np.zeros(100),[3],np.ones(5)].astype(int)
  w=weights(gt,[(0,100),(101,106)]);self.assertEqual(w[100],0);self.assertTrue(np.all(w[:100]==1));self.assertTrue(np.all(w[101:]==10))
  self.assertTrue(np.all(weights(np.zeros(10),[(0,5),(5,10)])==10))
 def test_swap_involution_and_dead_preserved(self):
  x=np.array([0,1,2,3]);np.testing.assert_array_equal(swapped(swapped(x)),x);np.testing.assert_array_equal(swapped(x),[1,0,2,3])
 def test_joint_fusion_and_team_symmetry(self):
  pd=np.array([.2,.7]);ph=np.array([.4,.8]);q=np.array([[.3,.5,.2],[.1,.1,.8]])
  d,h=blend(pd,ph,q,.5);ds,hs=blend(pd,1-ph,q[:,[1,0,2]],.5);np.testing.assert_allclose(d,ds);np.testing.assert_allclose(h,1-hs)
  np.testing.assert_allclose(d,[.2,.75]);np.testing.assert_allclose((1-d)*h,[.31,.17])
if __name__=='__main__':unittest.main()
