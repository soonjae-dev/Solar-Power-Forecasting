"""ECMWF 일사 차분·자르기·레코드 찾기 검사. 합성 입력만 사용한다."""
import json
import unittest
from datetime import date

import numpy as np

from src.experiments.ecmwf_fetch import STEPS, crop, deadline, find_ssrd, hourly
from src.gfs_fetch import LATS, LONS


class EcmwfFetchTest(unittest.TestCase):
    def test_hourly_assigns_three_hour_means(self):
        acc = {s: np.full((1, 1), 3600.0 * s * 100) for s in STEPS}       # 일정한 100 W/m² (누적 J/m²)
        acc[33] = acc[30] + 10800 * 400                                      # (30,33] 구간만 400 W/m²
        acc[36] = acc[33] + 10800 * 100
        acc[39], acc[42] = acc[36] + 10800 * 100, acc[36] + 10800 * 200
        h = hourly(acc)[:, 0, 0]
        self.assertEqual(len(h), 17)
        self.assertAlmostEqual(float(h[0]), 100)                            # step 25 → (24,27]
        self.assertAlmostEqual(float(h[31 - 25]), 400)                      # step 31 → (30,33]
        self.assertAlmostEqual(float(h[33 - 25]), 400)                      # step 33 → (30,33]
        self.assertAlmostEqual(float(h[34 - 25]), 100)                      # step 34 → (33,36]

    def test_crop_handles_minus_180_start(self):
        ni, nj = 1440, 721
        lat = 90 - np.arange(nj) * .25
        lon = (180 + np.arange(ni) * .25) % 360                             # 첫 격자 180E(=180W)
        field = (lat[:, None] * 1000 + lon[None, :]).ravel()
        box = crop(field, ni, nj, 90.0, 180.0, .25)
        self.assertEqual(box.shape, (len(LATS), len(LONS)))
        self.assertAlmostEqual(float(box[0, 0]), 40 * 1000 + 123)
        self.assertAlmostEqual(float(box[-1, -1]), 32 * 1000 + 132)

    def test_find_ssrd_and_deadline(self):
        text = "\n".join(json.dumps(x) for x in (
            {"param": "tp", "step": "24", "levtype": "sfc", "_offset": 0, "_length": 10},
            {"param": "ssrd", "step": "24", "levtype": "sfc", "_offset": 10, "_length": 5}))
        self.assertEqual(find_ssrd(text, 24), (10, 14))
        with self.assertRaises(LookupError):
            find_ssrd(text, 27)
        self.assertEqual(deadline(date(2025, 3, 2)).isoformat(), "2025-03-01T08:00:00+00:00")


if __name__ == "__main__":
    unittest.main()
