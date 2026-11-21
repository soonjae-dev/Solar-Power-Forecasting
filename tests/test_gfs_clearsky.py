"""청천 지수의 단위, 저일사 처리, 미교체 열 보존 검사. 합성 입력만 사용."""
import unittest

import numpy as np
import pandas as pd

from src.config import TZ
from src.experiments.gfs_clearsky import region_clear_sky, replace_radiation
from src.gfs_pilot import region_masks


class ClearSkyTest(unittest.TestCase):
    def test_units_and_unchanged_other_columns(self):
        ts = pd.DatetimeIndex(["2024-06-01 12:00", "2024-12-01 12:00"], tz=TZ)
        base = pd.DataFrame(index=ts)
        for name, mask in region_masks().items():
            denominator = region_clear_sky(ts, mask)
            self.assertTrue(np.all((denominator > 100) & (denominator < 1200)))
            base[f"gfs_dswrf_{name}"] = .6 * denominator
            base[f"gfs_tcdc_{name}"] = 40.
        out = replace_radiation(base)
        self.assertEqual(list(out.filter(like="gfs_dswrf_")), [])
        np.testing.assert_allclose(out.filter(like="gfs_kt_"), .6, atol=1e-6)
        pd.testing.assert_frame_equal(out.filter(like="tcdc"), base.filter(like="tcdc"))

    def test_dark_hours_cannot_create_large_ratios(self):
        ts = pd.DatetimeIndex(["2024-12-01 04:00"], tz=TZ)
        base = pd.DataFrame({f"gfs_dswrf_{n}": [1.] for n in region_masks()}, index=ts)
        self.assertTrue(replace_radiation(base).isna().all().all())


if __name__ == "__main__":
    unittest.main()
