"""Tests for alignment, missing detections, and phase boundaries in public data."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec = importlib.util.spec_from_file_location("skillcorner_reader", Path(__file__).resolve().parents[1] / "tools/skillcorner_reader.py")
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)


class ReaderTests(unittest.TestCase):
    def test_interpolation_does_not_bridge_missing_source_frames(self):
        source = np.array([[0.], [10.], [np.nan], [30.]])
        actual = reader.upsample_observations(source, np.array([0., .4, 1., 1.4, 2.8, 3.]))[:, 0]
        np.testing.assert_allclose(actual[[0, 1, 2, 5]], [0., 4., 10., 30.])
        self.assertTrue(np.isnan(actual[3:5]).all())

    def test_half_open_phase_boundaries_and_dead_gaps(self):
        meta = {"home_team": {"id": 1}, "away_team": {"id": 2},
                "match_periods": [{"start_frame": 1, "end_frame": 9, "period": 1}]}
        phases = [{"frame_start": 1, "frame_end": 3, "team_in_possession_id": 1, "period": 1},
                  {"frame_start": 3, "frame_end": 5, "team_in_possession_id": 2, "period": 1}]
        gt, _ = reader.phase_labels(meta, phases, 10)
        np.testing.assert_array_equal(gt, [3, 0, 0, 1, 1, 2, 2, 2, 2, 3])
        phases[1]["frame_start"] = 2
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            reader.phase_labels(meta, phases, 10)

    def test_provider_phase_cannot_extend_scored_half(self):
        meta = {"home_team": {"id": 1}, "away_team": {"id": 2},
                "match_periods": [{"start_frame": 2, "end_frame": 6, "period": 1},
                                  {"start_frame": 7, "end_frame": 10, "period": 2}]}
        phases = [{"frame_start": 1, "frame_end": 9, "team_in_possession_id": 1, "period": 1}]
        gt, _ = reader.phase_labels(meta, phases, 11)
        np.testing.assert_array_equal(gt, [3, 3, 0, 0, 0, 0, 3, 2, 2, 2, 3])


if __name__ == "__main__":
    unittest.main()
