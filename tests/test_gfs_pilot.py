"""파일럿 영역·구간·판정 검사. 합성 입력만 사용한다."""
import unittest

import numpy as np
import pandas as pd

from src.config import TZ
from src.gfs_pilot import END, blocks, region_masks, shuffled_within_month, summarize, verdict


class GfsPilotTest(unittest.TestCase):
    def test_nine_regions_partition_the_box(self):
        masks = region_masks()
        cells = [m for name, m in masks.items() if name != "all"]
        self.assertEqual(len(cells), 9)
        self.assertTrue((sum(c.astype(int) for c in cells) == masks["all"].astype(int)).all())

    def test_summary_same_value_for_constant_field(self):
        field = np.full((2, *region_masks()["all"].shape), 7.0)
        out = summarize(field, region_masks())
        self.assertEqual(len(out), 10)
        for values in out.values():
            self.assertTrue((values == 7.0).all())

    def test_blocks_train_before_previous_day_17h_and_exclude_holdout(self):
        for b in blocks():
            self.assertEqual(b["cutoff"].hour, 17)
            self.assertLess(b["cutoff"], b["start"])
            self.assertLessEqual(b["end"], END)
        self.assertEqual(blocks()[0]["cutoff"], pd.Timestamp("2024-09-30 17:00", tz=TZ))

    def test_verdict_needs_all_conditions(self):
        good = {k: dict(vs_control=dict(change_pct=v)) for k, v in zip("abc", (-3, -2, 0.5))}
        self.assertTrue(verdict(good, dict(change_pct=-1.5, change_ci90_pct=[-3, -0.1]))["passed"])
        self.assertFalse(verdict(good, dict(change_pct=-0.9, change_ci90_pct=[-3, -0.1]))["passed"])
        self.assertFalse(verdict(good, dict(change_pct=-1.5, change_ci90_pct=[-3, 0.1]))["passed"])
        bad = {k: dict(vs_control=dict(change_pct=v)) for k, v in zip("abc", (-3, 1.5, -1))}
        self.assertFalse(verdict(bad, dict(change_pct=-1.5, change_ci90_pct=[-3, -0.1]))["passed"])

    def test_placebo_shuffles_days_only_within_month(self):
        index = pd.DatetimeIndex([pd.Timestamp(d, tz=TZ) + pd.Timedelta(hours=h)
                                  for d in pd.date_range("2024-06-29", "2024-07-02") for h in (4, 5)])
        days = np.repeat(index.day[::2].to_numpy(), 2).astype(float)
        gfs = pd.DataFrame(dict(x=days + np.tile([0.0, 0.5], 4)), index=index)
        out = shuffled_within_month(gfs, 1)
        self.assertEqual(set(out["x"].iloc[:4]), {29.0, 29.5, 30.0, 30.5})
        self.assertEqual(set(out["x"].iloc[4:]), {1.0, 1.5, 2.0, 2.5})
        self.assertTrue((out["x"].iloc[1::2].to_numpy() % 1 == 0.5).all())


if __name__ == "__main__":
    unittest.main()
