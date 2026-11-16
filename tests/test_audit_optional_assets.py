import unittest,tempfile
from pathlib import Path
from unittest.mock import patch
from mtah import tagging,features2
from tests.test_audit_alignment import fixture
import numpy as np
class OptionalAssetsAudit(unittest.TestCase):
 def test_tagging_intervals_and_unknown_team(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'tag.xml';p.write_text('<R><instance><code>build up</code><start>1.1</start><end>2</end><label><group>Team</group><text>N/A</text></label></instance><instance><code>SET PIECES</code><start>1.5</start><end>3</end><label><group>Team</group><text>Barcelona</text></label></instance></R>');v=tagging.load(p);self.assertEqual(v[0]['start'],round(1.1*25));self.assertIsNone(v[0]['team']);self.assertEqual(v[1]['team'],'H');self.assertEqual(len(v),2)
 def test_conflicting_unused_tags_cannot_change_features(self):
  d=fixture();x,n=features2.build(d,[(0,600)]);d['smart_tagging']=[{'team':'A','start':0,'end':600},{'team':'H','start':0,'end':600}];y,m=features2.build(d,[(0,600)]);np.testing.assert_array_equal(x,y)
if __name__=='__main__':unittest.main()
