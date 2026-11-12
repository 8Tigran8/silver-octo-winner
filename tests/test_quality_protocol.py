"""Check the isolation rules that prevent target-match leakage in nested stacking."""
import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location("quality_audit", Path(__file__).resolve().parents[1] / "tools/quality_audit.py")
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class IsolationTests(unittest.TestCase):
    def test_all_nested_fits_are_isolated_and_pairs_reused(self):
        names = [f"external_{i}" for i in range(9)]
        specs = AUDIT.split_specs(names)
        self.assertEqual(len(specs), 45)
        self.assertEqual(len(set(specs)), 45)
        for outer in names:
            AUDIT.assert_exclusion(names, [n for n in names if n != outer], outer)
            for inner in names:
                if inner != outer:
                    omitted = tuple(sorted((outer, inner)))
                    self.assertIn(omitted, specs)
                    AUDIT.assert_exclusion(names, [n for n in names if n not in omitted], outer, inner)

    def test_historical_stack_leakage_is_rejected(self):
        names = ["outer_test", "inner_test", "other_1", "other_2"]
        with self.assertRaisesRegex(ValueError, "leaked"):
            AUDIT.assert_exclusion(names, ["outer_test", "other_1", "other_2"], "outer_test", "inner_test")

    def test_competition_inputs_gt_and_full_data_checkpoint_are_blocked(self):
        for path in [AUDIT.ROOT.parent / "data/tactical/BAR-ATH_tactical_atd.txt",
                     AUDIT.ROOT / "ref/BAR-ATH_possession_gt_public.csv",
                     AUDIT.ROOT / "out/cache/atd_tactical_any.npz",
                     AUDIT.ROOT / "out/training_preflight.pkl",
                     AUDIT.ROOT / "out/submission/Ruses Cules.csv"]:
            self.assertTrue(AUDIT.forbidden(path))
        self.assertFalse(AUDIT.forbidden(AUDIT.ROOT / "external/conv/idsse_J03WMX.npz"))


if __name__ == "__main__":
    unittest.main()
