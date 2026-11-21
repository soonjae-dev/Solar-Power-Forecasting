"""GEFS 교체/추가 후보가 사전 정의한 열을 정확히 쓰는지 합성 검사."""
import unittest
from unittest.mock import patch

import pandas as pd

from src.config import TZ
from src.experiments.gefs_mean_model import main


class MeanModelTest(unittest.TestCase):
    def test_replace_and_add_keep_the_same_control(self):
        stamps = pd.date_range("2024-06-01 12:00", periods=2, freq="h", tz=TZ)
        gfs = pd.DataFrame({f"gfs_base_{i}": [1., 2.] for i in range(20)}, index=stamps)
        more = pd.DataFrame({f"gfs_more_{i}": [3., 4.] for i in range(40)}, index=stamps)
        mean = pd.DataFrame({f"gefs_mean_{i}": [5., 6.] for i in range(20)}, index=stamps)
        for mode, width in (("replace", 60), ("add", 80)):
            with self.subTest(mode=mode), \
                 patch("src.experiments.gefs_mean_model.load_mean", return_value=(mean, 0)), \
                 patch("src.experiments.gefs_mean_model.checked_gfs", return_value=(gfs, [])), \
                 patch("src.experiments.gefs_mean_model.load_more", return_value=(more, [])), \
                 patch("src.experiments.gefs_mean_model.run_comparison") as run:
                main(mode)
                control, candidate = run.call_args.args[:2]
                pd.testing.assert_frame_equal(control, pd.concat([gfs, more], axis=1))
                self.assertEqual(candidate.shape[1], width)
                self.assertTrue(set(mean) <= set(candidate))
                self.assertTrue(set(more) <= set(candidate))
                self.assertEqual(set(gfs) <= set(candidate), mode == "add")

    def test_unknown_mode_is_rejected_before_data_access(self):
        with patch("src.experiments.gefs_mean_model.load_mean") as load:
            with self.assertRaises(ValueError):
                main("unknown")
            load.assert_not_called()

    def test_pipeline_default_runs_two_prespecified_modes_sequentially(self):
        calls = []
        from src.experiments import gefs_mean_model as module
        original = module.main
        with patch("src.experiments.gefs_mean_model.main", side_effect=lambda mode: calls.append(mode)):
            original()
        self.assertEqual(calls, ["replace", "add"])


if __name__ == "__main__":
    unittest.main()
