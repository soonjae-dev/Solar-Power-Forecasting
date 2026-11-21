"""GFS 레코드 찾기·차분·자르기 검사. 합성 입력만 사용한다."""
import unittest

import numpy as np

from src.gfs_fetch import FIRST_STEP, LATS, LONS, PWAT, crop, hourly, parse_aerosol, parse_idx, parse_instant


class GfsFetchTest(unittest.TestCase):
    def test_parse_idx_picks_average_and_byte_range(self):
        text = ("1:0:d=2025053100:TCDC:entire atmosphere:16 hour fcst:\n"
                "2:100:d=2025053100:TCDC:entire atmosphere:12-16 hour ave fcst:\n"
                "3:250:d=2025053100:DSWRF:surface:12-16 hour ave fcst:\n"
                "4:400:d=2025053100:DLWRF:surface:12-16 hour ave fcst:\n")
        self.assertEqual(parse_idx(text, "TCDC", "entire atmosphere"), (100, 249, 12, 16))
        self.assertEqual(parse_idx(text, "DSWRF", "surface"), (250, 399, 12, 16))
        with self.assertRaises(ValueError):
            parse_idx(text, "USWRF", "surface")

    def test_parse_idx_reads_gefs_spread_records(self):
        text = ("25:8003083:d=2021010100:TCDC:entire atmosphere:18-21 hour ave fcst:ens std dev\n"
                "26:8512285:d=2021010100:DSWRF:surface:18-21 hour ave fcst:ens std dev\n"
                "27:9000000:d=2021010100:DLWRF:surface:18-21 hour ave fcst:ens std dev\n")
        self.assertEqual(parse_idx(text, "TCDC", "entire atmosphere"), (8003083, 8512284, 18, 21))
        self.assertEqual(parse_idx(text, "DSWRF", "surface"), (8512285, 8999999, 18, 21))

    def test_parse_instant_matches_step(self):
        text = ("1:0:d=2025053100:PWAT:entire atmosphere (considered as a single layer):22 hour fcst:\n"
                "2:90:d=2025053100:CWAT:entire atmosphere (considered as a single layer):22 hour fcst:\n")
        self.assertEqual(parse_instant(text, *PWAT, 22), (0, 89))
        with self.assertRaises(ValueError):
            parse_instant(text, *PWAT, 21)

    def test_parse_aerosol_picks_550nm_total_and_dust(self):
        text = ("1:0:d=2023060100:AOTK:entire atmosphere:24 hour fcst:aerosol=Total Aerosol:aerosol_size <2e-05:aerosol_wavelength >=3.38e-07,<=3.42e-07\n"
                "5:100:d=2023060100:AOTK:entire atmosphere:24 hour fcst:aerosol=Total Aerosol:aerosol_size <2e-05:aerosol_wavelength >=5.45e-07,<=5.65e-07\n"
                "6:200:d=2023060100:SCTAOTK:entire atmosphere:24 hour fcst:aerosol=Total Aerosol:aerosol_size <2e-05:aerosol_wavelength >=5.45e-07,<=5.65e-07\n"
                "7:300:d=2023060100:AOTK:entire atmosphere:24 hour fcst:aerosol=Dust Dry:aerosol_size <2e-05:aerosol_wavelength >=5.45e-07,<=5.65e-07\n"
                "8:400:d=2023060100:SCTAOTK:entire atmosphere:24 hour fcst:aerosol=Dust Dry:aerosol_size <2e-05:aerosol_wavelength >=5.45e-07,<=5.65e-07\n")
        self.assertEqual(parse_aerosol(text, "Total Aerosol", 24), (100, 199))
        self.assertEqual(parse_aerosol(text, "Dust Dry", 24), (300, 399))
        lower = text.replace("Total Aerosol", "Total aerosol").replace("Dust Dry", "Dust dry")   # 2024-04 이후 표기
        self.assertEqual(parse_aerosol(lower, "Total Aerosol", 24), (100, 199))
        self.assertEqual(parse_aerosol(lower, "Dust Dry", 24), (300, 399))

    def test_hourly_recovers_values_from_reset_averages(self):
        true = {19: 10.0, 20: 20.0, 21: 40.0, 22: 80.0, 23: 5.0, 24: 1.0, 25: 7.0, 26: 9.0}
        averages = {}
        for step in true:
            start = (step - 1) // 6 * 6
            values = [true[s] for s in range(start + 1, step + 1)]
            averages[step] = (start, np.array(sum(values) / len(values)))
        result = hourly(averages)
        for step, value in true.items():
            self.assertAlmostEqual(float(result[step]), value)

    def test_hourly_rejects_mismatched_previous_window(self):
        with self.assertRaises(ValueError):
            hourly({20: (18, np.array(1.0)), 19: (12, np.array(1.0))})

    def test_crop_picks_fixed_box_in_lat_lon_order(self):
        ni, nj = 1440, 721
        lat = 90 - np.arange(nj) * 0.25
        lon = np.arange(ni) * 0.25
        field = (lat[:, None] * 1000 + lon[None, :]).ravel()
        box = crop(field, ni, nj)
        self.assertEqual(box.shape, (len(LATS), len(LONS)))
        self.assertAlmostEqual(float(box[0, 0]), 40 * 1000 + 123, places=1)
        self.assertAlmostEqual(float(box[-1, -1]), 32 * 1000 + 132, places=1)

    def test_first_step_matches_early_morning_ts(self):
        self.assertEqual(FIRST_STEP - 15, 4)


if __name__ == "__main__":
    unittest.main()
