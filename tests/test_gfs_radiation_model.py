"""새 복사 특징이 기존GFS60열을 보존하는지 합성 검사."""
import unittest
from unittest.mock import patch

import pandas as pd

from src.config import TZ
from src.experiments.gfs_radiation import COLUMNS
from src.experiments.gfs_radiation_model import main


class RadiationModelTest(unittest.TestCase):
    def test_twenty_new_features_preserve_baseline_features(self):
        stamps = pd.date_range("2024-06-01 12:00", periods=2, freq="h", tz=TZ)
        gfs = pd.DataFrame({f"gfs_{i}": [1., 2.] for i in range(20)}, index=stamps)
        more = pd.DataFrame({f"more_{i}": [3., 4.] for i in range(40)}, index=stamps)
        optics = pd.DataFrame({name: [.2, .3] for name in COLUMNS}, index=stamps)
        with patch("src.experiments.gfs_radiation_model.checked_gfs", return_value=(gfs, [])), \
             patch("src.experiments.gfs_radiation_model.load_more", return_value=(more, [])), \
             patch("src.experiments.gfs_radiation_model.load_optics", return_value=(optics, 0)), \
             patch("src.experiments.gfs_radiation_model.run_comparison") as run:
            main()
        base, candidate = run.call_args.args[:2]
        self.assertEqual(base.shape[1], 60)
        self.assertEqual(candidate.shape[1], 80)
        pd.testing.assert_frame_equal(base, candidate[list(base)])
        self.assertEqual(run.call_args.args[2:5], ("sflux_optics20", "p48_gfs_radiation_model", "14.35"))


if __name__ == "__main__":
    unittest.main()
