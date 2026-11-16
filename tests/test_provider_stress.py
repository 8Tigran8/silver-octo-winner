"""Synthetic checks of reference independence, clocks and observation gaps."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


spec = importlib.util.spec_from_file_location(
    "provider_stress", Path(__file__).resolve().parents[1] / "tools/provider_stress.py")
stress = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stress)


def fixture(hz=25):
    n = 40 * hz
    times = np.arange(n) / hz
    period = np.where(times < 20, 1, 2).astype(np.int8)
    players = np.zeros((n, 2, 3), np.float32)
    players[:, :, 0] = (times % 20)[:, None] * 2 / 100
    players[:, :, 1] = .5
    players[:, :, 2] = np.nan
    ball = np.full((n, 3), .5, np.float32)
    native = dict(players=players, ball=ball, period=period, timestamps=times,
                  source_frame=np.arange(n, dtype=np.int64))
    rows = np.floor(np.arange(1000) * hz / 25).astype(int)
    original = dict(players=players[rows].copy(), ball=ball[rows].copy(),
                    period=period[rows], gt=np.zeros(1000, np.int8),
                    pitch=np.array([100., 60.]), slot_team=np.array([0, 1], np.int8))
    return original, native


def preset(hz=10, noise=.5, rate=12):
    return dict(observation_hz=hz, noise_std_m=noise,
                ball_bursts_per_minute=rate, burst_min_s=2., burst_max_s=6.)


class StressTests(unittest.TestCase):
    def test_labels_cannot_change_geometry_clocks_or_masks(self):
        original, native = fixture()
        before = {k: v.copy() for k, v in original.items()}
        a, ma = stress.make_copy(original, native, 25, preset(), 20261113)
        changed = dict(original, gt=np.arange(1000, dtype=np.int8) % 4)
        b, mb = stress.make_copy(changed, native, 25, preset(), 20261113)
        for key in a:
            if key != "gt":
                np.testing.assert_array_equal(a[key], b[key], err_msg=key)
        self.assertEqual(ma, mb)
        np.testing.assert_array_equal(a["gt"], original["gt"])
        np.testing.assert_array_equal(b["gt"], changed["gt"])
        for key in original:
            np.testing.assert_array_equal(original[key], before[key])

    def test_periods_reset_speed_and_random_streams(self):
        original, native = fixture()
        a, ma = stress.make_copy(original, native, 25, preset(), 14)
        changed = {k: v.copy() for k, v in native.items()}
        changed["players"][:500] = 999
        changed["ball"][:500] = np.nan
        b, _ = stress.make_copy(original, changed, 25, preset(), 14)
        np.testing.assert_array_equal(a["players"][500:], b["players"][500:])
        np.testing.assert_array_equal(a["ball"][500:], b["ball"][500:])
        self.assertTrue(np.isnan(a["players"][[0, 500], :, 2]).all())
        for burst in ma["bursts"]:
            boundary = 20 if burst["period"] == 1 else 40
            self.assertLessEqual(burst["end_s"], boundary)
            self.assertGreaterEqual(burst["start_s"], boundary - 20)
        grid = a["grid_index"]
        np.testing.assert_array_equal(a["period"][grid], native["period"][a["source_index"]])

    def test_dropout_remains_nan_and_no_unobserved_rows_are_filled(self):
        original, native = fixture()
        native["ball"][100:200] = np.nan
        original["ball"][100:200] = np.nan
        out, report = stress.make_copy(original, native, 25, preset(), 15)
        self.assertGreater(report["dropped_ball_observations"], 0)
        self.assertTrue(np.isnan(out["ball"][out["ball_dropout_mask"]]).all())
        self.assertTrue(np.isnan(out["ball"][100:200]).all())
        self.assertTrue(np.isnan(out["ball"][~out["observation_mask"]]).all())
        self.assertTrue(np.isnan(out["players"][~out["observation_mask"]]).all())

    def test_speed_uses_elapsed_time_and_does_not_bridge_hidden_gap(self):
        original, native = fixture()
        # Row 1 is not sampled at 10 Hz, but still breaks this track.
        native["players"][1, 0] = np.nan
        out, _ = stress.make_copy(original, native, 25, preset(noise=0, rate=0), 3)
        self.assertTrue(np.isnan(out["players"][3, 0, 2]))
        self.assertAlmostEqual(float(out["players"][3, 1, 2]), 2., places=5)
        self.assertAlmostEqual(float(out["players"][5, 0, 2]), 2., places=5)
        np.testing.assert_array_equal(out["grid_index"][:4], [0, 3, 5, 8])
        np.testing.assert_allclose(out["native_timestamps_s"][:4], [0., .12, .2, .32])

    def test_ten_hz_source_never_becomes_fifteen_hz_measurements(self):
        original, native = fixture(10)
        a, _ = stress.make_copy(original, native, 10, preset(hz=10), 2)
        b, report = stress.make_copy(original, native, 10, preset(hz=15), 2)
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
        self.assertEqual(report["effective_observation_hz"], 10)
        self.assertEqual(len(np.unique(b["source_index"])), len(b["source_index"]))
        np.testing.assert_allclose(b["native_timestamps_s"][:4], [0., .1, .2, .3])

    def test_half_grid_ties_always_map_forward(self):
        original, native = fixture(10)
        out, _ = stress.make_copy(original, native, 10, preset(noise=0, rate=0), 2)
        # 2.3 s is halfway between rows 57 and 58, despite binary roundoff.
        self.assertEqual(int(out["grid_index"][23]), 58)
        np.testing.assert_array_equal(out["grid_index"], (np.arange(400) * 5 + 1) // 2)

    def test_verification_rejects_incorrect_dropout_counts(self):
        original, native = fixture()
        out, report = stress.make_copy(original, native, 25, preset(), 15)
        stress.verify(original, out, report, native)
        report["dropped_ball_observations"] += 1
        with self.assertRaises(AssertionError):
            stress.verify(original, out, report, native)

    def test_verification_rejects_incorrect_native_timestamps(self):
        original, native = fixture()
        out, report = stress.make_copy(original, native, 25, preset(), 15)
        out["native_timestamps_s"][1] += .04
        with self.assertRaises(AssertionError):
            stress.verify(original, out, report, native)

    def test_fifteen_hz_speed_uses_nonuniform_source_intervals(self):
        original, native = fixture()
        out, _ = stress.make_copy(original, native, 25, preset(hz=15, noise=0, rate=0), 1)
        np.testing.assert_array_equal(out["source_index"][:4], [0, 2, 3, 5])
        np.testing.assert_allclose(out["native_timestamps_s"][:4], [0, .08, .12, .2])
        np.testing.assert_allclose(out["players"][[2, 3, 5], :, 2], 2, atol=1e-5)

    def test_source_half_boundary_between_grid_rows(self):
        original, native = fixture(10)
        native["period"][:201] = 1
        native["period"][201:] = 2
        original["period"][:503] = 1
        original["period"][503:] = 2
        out, report = stress.make_copy(original, native, 10, preset(), 15)
        self.assertTrue(np.isnan(out["players"][503, :, 2]).all())
        stress.verify(original, out, report, native)
        for burst in report["bursts"]:
            if burst["period"] == 1:
                self.assertLessEqual(burst["end_s"], 20.1)
            else:
                self.assertGreaterEqual(burst["start_s"], 20.1)

    def test_outside_periods_and_existing_grid_gaps_stay_empty(self):
        original, native = fixture()
        original["period"][490:510] = 0
        native["period"][490:510] = 0
        original["ball"][100:200] = np.nan
        out, _ = stress.make_copy(original, native, 25, preset(), 4)
        np.testing.assert_array_equal(out["period"], original["period"])
        self.assertTrue(np.isnan(out["ball"][490:510]).all())
        self.assertTrue(np.isnan(out["players"][490:510]).all())
        self.assertTrue(np.isnan(out["ball"][100:200]).all())
        self.assertFalse(out["ball_dropout_mask"][490:510].any())

    def test_skillcorner_adapter_rejects_finite_extrapolated_positions(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            meta = dict(pitch_length=100, pitch_width=60,
                        home_team={"id": 1}, away_team={"id": 2},
                        players=[{"id": 7, "team_id": 1}],
                        match_periods=[{"start_frame": 0, "end_frame": 6, "period": 1},
                                       {"start_frame": 6, "end_frame": 12, "period": 2}])
            (folder / "tiny_match.json").write_text(json.dumps(meta))
            (folder / "tiny_phases_of_play.csv").write_text(
                "frame_start,frame_end,team_in_possession_id,period\n0,6,1,1\n6,12,2,2\n")
            rows = [dict(frame=i, ball_data=dict(x=i, y=0, z=None, is_detected=i != 2),
                         player_data=[dict(player_id=7, x=i, y=0, is_detected=i != 2)])
                    for i in range(13)]
            (folder / "tiny_tracking_extrapolated.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows))
            source = dict(game_id="tiny", raw_files={
                "metadata": {"path": "tiny_match.json"},
                "tracking": {"path": "tiny_tracking_extrapolated.jsonl"}})
            original, native = stress.load_skillcorner(source, folder, folder)
            self.assertTrue(np.isnan(native["players"][2]).all())
            self.assertTrue(np.isnan(native["ball"][2]).all())
            self.assertAlmostEqual(float(native["ball"][3, 0]), .53, places=6)
            self.assertTrue(np.isnan(native["ball"][:, 2]).all())
            out, _ = stress.make_copy(original, native, 10, preset(hz=15, noise=0, rate=0), 1)
            self.assertTrue(np.isnan(out["ball"][5]).all())
            self.assertTrue(np.isnan(out["players"][[0, 15], :, 2]).all())
            np.testing.assert_array_equal(out["gt"], original["gt"])


if __name__ == "__main__":
    unittest.main()
