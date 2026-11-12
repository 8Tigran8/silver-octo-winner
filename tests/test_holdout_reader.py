"""Verify that changing lineup register order preserves player identity and frame alignment."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np

TOOLS = Path(__file__).resolve().parents[1] / "tools"
import sys
sys.path.insert(0, str(TOOLS))
SPEC = importlib.util.spec_from_file_location("fresh_holdout", TOOLS / "fresh_metrica_holdout.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class LineupTests(unittest.TestCase):
    def fixture(self, root, second_id=2):
        meta = Path(root) / "meta.xml"; raw = Path(root) / "raw.txt"
        mapping1 = '<SplitRegister><PlayerChannelRef playerChannelId="p1x"/><PlayerChannelRef playerChannelId="p1y"/></SplitRegister><SplitRegister><PlayerChannelRef playerChannelId="p2x"/><PlayerChannelRef playerChannelId="p2y"/></SplitRegister>'
        mapping2 = '<SplitRegister><PlayerChannelRef playerChannelId="p2x"/><PlayerChannelRef playerChannelId="p2y"/></SplitRegister><SplitRegister><PlayerChannelRef playerChannelId="p1x"/><PlayerChannelRef playerChannelId="p1y"/></SplitRegister>'
        meta.write_text('<main><FrameRate>25</FrameRate><FieldSize><Width>105</Width><Height>68</Height></FieldSize><Team id="a"/><Team id="b"/><Player id="p1" teamId="a"/><Player id="p2" teamId="b"/>'+''.join(f'<PlayerChannel id="{p}{c}" playerId="{p}" channelId="{c}"/>' for p in ['p1','p2'] for c in ['x','y'])+'<DataFormatSpecifications>'+''.join(f'<DataFormatSpecification startFrame="{i}" endFrame="{i}"><StringRegister/><SplitRegister>{m}</SplitRegister><SplitRegister/></DataFormatSpecification>' for i,m in [(1,mapping1),(2,mapping2)])+'</DataFormatSpecifications></main>')
        raw.write_text('1:0.1,0.2;0.8,0.9:0.5,0.5\n'+str(second_id)+':0.7,0.8;0.2,0.3:0.5,0.5\n')
        return meta, raw

    def test_lineup_order_changes_are_mapped_by_player_channel(self):
        with tempfile.TemporaryDirectory() as root:
            d, _, count = MODULE.parse_tracking(*self.fixture(root))
            self.assertEqual(count, 2)
            np.testing.assert_allclose(d['players'][:,0,:2],[[.1,.2],[.2,.3]])
            np.testing.assert_allclose(d['players'][:,1,:2],[[.8,.9],[.7,.8]])

    def test_wrong_frame_offset_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError, 'frame IDs'):
                MODULE.parse_tracking(*self.fixture(root, second_id=3))


if __name__ == '__main__':
    unittest.main()
