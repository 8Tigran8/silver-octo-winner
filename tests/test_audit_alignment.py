import unittest,tempfile,os
from pathlib import Path
from unittest.mock import patch
import numpy as np
from mtah import features2,atd

def fixture(n=600):
 p=np.zeros((n,4,3));p[:,:2,0]=.25;p[:,2:,0]=.75;p[:,:,1]=[.3,.7,.3,.7];p[:,:,2]=2
 return dict(players=p,ball=np.tile([.3,.4,0.],(n,1)),slot_team=np.array([0,0,1,1]),pitch=(105.,68.))
class AuditAlignment(unittest.TestCase):
 def test_period_isolation(self):
  d=fixture();x,_=features2.build(d,[(0,300),(300,600)]);d['ball'][300:,:2]=.9;y,_=features2.build(d,[(0,300),(300,600)]);np.testing.assert_allclose(x[:300],y[:300],equal_nan=True)
 def test_tiny_period_shape(self):
  for n in [1,2,12,50]:
   x,_=features2.build(fixture(n),[(0,n)]);self.assertEqual(len(x),n)
 def test_content_change_invalidates_cache(self):
  with tempfile.TemporaryDirectory() as t:
   r=Path(t);raw=r/'raw.txt';meta=r/'meta.xml';meta.write_text('<R><Team id="h"><Name>H</Name></Team><Team id="a"><Name>A</Name></Team><Player teamId="h"/><Player teamId="a"/></R>');raw.write_text('1:0.1,0.2,0;0.7,0.8,0:0.3,0.4,0\n');stamp=raw.stat().st_mtime
   with patch.object(atd,'asset',side_effect=lambda name,source: str(meta if name=='atd_meta' else raw)),patch.object(atd,'CACHE_DIR',str(r/'cache')):
    a=atd.load();raw.write_text('1:0.1,0.2,0;0.7,0.8,0:0.9,0.4,0\n');os.utime(raw,(stamp,stamp));b=atd.load();self.assertNotEqual(a['ball'][0,0],b['ball'][0,0])
if __name__=='__main__':unittest.main()
