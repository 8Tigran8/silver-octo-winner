"""Reference availability and track changes must not fabricate input evidence."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
from legacy_skillcorner_reader import read_match


class LegacyTests(unittest.TestCase):
    def fixture(self,folder):
        meta={"home_team":{"id":1},"away_team":{"id":2},"pitch_length":100,"pitch_width":60,
              "players":[{"team_id":1,"trackable_object":10},{"team_id":1,"trackable_object":11}],
              "ball":{"trackable_object":55}}
        frames=[{"frame":i,"period":1,"possession":{"group":"home team" if i else None},
                 "data":[{"trackable_object":10 if i==0 else 11,"track_id":i if i==0 else 1,"x":i*20,"y":0},
                         {"trackable_object":55,"track_id":-1,"x":0,"y":0,"z":0}]} for i in range(3)]
        (folder/"match_data.json").write_text(json.dumps(meta))
        (folder/"structured_data.json").write_text(json.dumps(frames))
        return frames

    def test_missing_possession_is_unknown_not_dead(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);self.fixture(folder);data=read_match(folder)
            self.assertFalse((data["gt"]==2).any())
            self.assertEqual(data["gt"][0],3)

    def test_reference_annotation_cannot_change_input_positions(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);frames=self.fixture(folder);a=read_match(folder)
            for row in frames:row["possession"]["group"]="away team"
            (folder/"structured_data.json").write_text(json.dumps(frames));b=read_match(folder)
            for key in ("ball","players","slot_team","period"):
                np.testing.assert_allclose(a[key],b[key],equal_nan=True)
            self.assertFalse(np.array_equal(a["gt"],b["gt"]))

    def test_interpolation_does_not_bridge_reassigned_slot(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);self.fixture(folder);data=read_match(folder)
            self.assertTrue(np.isfinite(data["players"][0,0,0]))
            self.assertTrue(np.isnan(data["players"][1:3,0,:]).all())


if __name__=="__main__":unittest.main()
