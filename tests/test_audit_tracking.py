import unittest,tempfile
from pathlib import Path
import numpy as np
from mtah import features2,atd,v4
from tests.test_audit_alignment import fixture
class TrackingAudit(unittest.TestCase):
 def test_nearest_tie_is_slot_order_invariant(self):
  d=fixture();d['players'][:,1,:2]=d['players'][:,0,:2];d['players'][:,0,2]=1;d['players'][:,1,2]=7
  a,_=features2.build(d,[(0,600)]);perm=[1,0,2,3];b,_=features2.build(dict(d,players=d['players'][:,perm],slot_team=d['slot_team'][perm]),[(0,600)]);np.testing.assert_allclose(a,b,equal_nan=True)
 def test_distances_in_metres(self):
  d=fixture();d['ball'][:,:2]=[.25,.3];x,n=features2.build(d,[(0,600)]);self.assertEqual(x[0,n.index('h_d1')],0);self.assertAlmostEqual(x[0,n.index('a_d1')],52.5)
 def test_bad_team_rejected(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'m.xml';p.write_text('<R><Team id="h"/><Team id="a"/><Player teamId="unknown"/></R>')
   with self.assertRaises(ValueError):atd.read_meta(p)
 def test_frame_id_endpoint_mapping(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'a.txt';p.write_text('1:0.1,0.2,0;0.3,0.4,0:0.11,0.22,0\n2:0.1,0.2,0;0.3,0.4,0:0.77,0.88,0\n');x=atd.parse_raw(p,2);np.testing.assert_array_equal(x['frame_ids']-1,[0,1]);np.testing.assert_allclose(x['ball'][:,0],[.11,.77])
 def test_resampling_global_phase_and_periods(self):
  pd=np.arange(20)/20;ph=1-pd;d,h=v4.neural_upsample(pd,ph,[(3,43),(52,99)],100)
  self.assertEqual(d[5],pd[1]);self.assertEqual(d[40],pd[8]);self.assertEqual(d[52],pd[11]);self.assertEqual(d[49],.5)
if __name__=='__main__':unittest.main()
