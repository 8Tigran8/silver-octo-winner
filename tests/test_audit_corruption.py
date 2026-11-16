"""Independent missing-observation and disposable-cache contracts; no match GT."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from mtah import atd, features2
from tests.test_audit_alignment import fixture


class ObservationContracts(unittest.TestCase):
    def test_single_observation_is_retained(self):
        x = np.array([np.nan, 17., np.nan])
        np.testing.assert_array_equal(features2._interp(x, np.isfinite(x)), [17., 17., 17.])

    def test_partial_xy_equals_missing_observation(self):
        for kind in ('ball', 'player'):
            for coordinate in (0, 1):
                with self.subTest(kind=kind, coordinate=coordinate):
                    partial, missing = fixture(), fixture()
                    if kind == 'ball':
                        partial['ball'][100:300, coordinate] = np.nan
                        missing['ball'][100:300, :2] = np.nan
                    else:
                        partial['players'][100:300, 0, coordinate] = np.nan
                        missing['players'][100:300, 0, :2] = np.nan
                    a, _ = features2.build(partial, [(0, 600)])
                    b, _ = features2.build(missing, [(0, 600)])
                    np.testing.assert_allclose(a, b, equal_nan=True)

    def test_infinity_cannot_be_an_observed_coordinate_or_speed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'raw.txt'
            path.write_text('1:inf,0.2,inf;0.7,-inf,2:0.3,inf,0\n')
            data = atd.parse_raw(path, 2)
            self.assertFalse(np.isinf(data['players']).any())
            self.assertFalse(np.isinf(data['ball']).any())

    def test_malformed_player_group_has_actionable_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'raw.txt'
            path.write_text('1:0.1;0.7,0.8,0:0.3,0.4,0\n')
            with self.assertRaisesRegex(ValueError, 'player.*line 1'):
                atd.parse_raw(path, 2)


class CacheContracts(unittest.TestCase):
    def test_damaged_cache_is_rebuilt_from_source(self):
        for damage in ('truncated', 'shape', 'nonconsecutive_ids'):
            with self.subTest(damage=damage), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                meta, raw = root / 'meta.xml', root / 'raw.txt'
                meta.write_text('<R><Team id="h"/><Team id="a"/><Player teamId="h"/><Player teamId="a"/></R>')
                raw.write_text('1:0.1,0.2,0;0.7,0.8,0:0.3,0.4,0\n')
                with patch.object(atd, 'asset', side_effect=lambda name, source: str(meta if name == 'atd_meta' else raw)), patch.object(atd, 'CACHE_DIR', str(root / 'cache')):
                    expected = atd.load()
                    cache, = (root / 'cache').glob('*.npz')
                    if damage == 'truncated':
                        cache.write_bytes(cache.read_bytes()[:50])
                    else:
                        with np.load(cache) as archive:
                            content = dict(archive)
                        if damage == 'shape':
                            content['players'] = content['players'][:, :1]
                        else:
                            content['frame_ids'][0] = 99
                        np.savez(cache, **content)
                    actual = atd.load()
                    for key in ('ball', 'players', 'frame_ids', 'ball_nfields'):
                        np.testing.assert_array_equal(actual[key], expected[key])


if __name__ == '__main__':
    unittest.main()
