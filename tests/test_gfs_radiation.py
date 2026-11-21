"""native 가우시안 해독·복사 성분 비율·공개 시각·재개 검사. 합성 자료만 사용."""
from datetime import date, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from src.experiments.external_weather_audit import prediction_cutoff
from src.experiments.gfs_radiation import (COLUMNS, FIELDS, REGIONS, SCHEMA, STEPS, cache_valid, classify_day,
                           fetch, fetch_day, gaussian_regions, make_features, preflight, regional_means,
                           required_keys, url_for)


def synthetic_means():
    # CSDSF1000, visible beam200/diffuse100, nearIR beam400/diffuse100.
    means = np.stack([np.full((6, 10), value, dtype="float64") for value in (1000, 200, 100, 400, 100)])
    starts = np.tile([18, 18, 24, 24, 30, 30], (5, 1))
    return means, starts


class RadiationTest(unittest.TestCase):
    def test_component_ratios_and_units(self):
        means, starts = synthetic_means()
        result = make_features(means, starts)
        self.assertEqual(result.shape, (17, 20))
        np.testing.assert_allclose(result[:, :10], .25)
        np.testing.assert_allclose(result[:, 10:], .8)
        self.assertEqual(len(COLUMNS), 20)

    def test_low_sun_is_missing_and_transmittance_is_bounded(self):
        means, starts = synthetic_means()
        self.assertTrue(np.isnan(make_features(means * .001, starts)).all())
        means[0] = 100
        np.testing.assert_array_equal(make_features(means, starts)[:, 10:], 2.)

    def test_three_hour_blocks_recover_cumulative_means(self):
        means, starts = synthetic_means()
        means[1:, 1] *= 1.5  # 두 번째3시간 구간의 플럭스는 원래 값의2배다.
        result = make_features(means, starts)
        np.testing.assert_allclose(result[:3, 10:], .8)
        np.testing.assert_allclose(result[3:6, 10:], 1.6)

    def test_unknown_average_window_is_rejected(self):
        means, starts = synthetic_means()
        starts[0, 0] = 0
        with self.assertRaises(ValueError):
            make_features(means, starts)

    def test_all_files_and_deadline_are_required(self):
        target = date(2024, 8, 16)
        cutoff = prediction_cutoff(target.isoformat())
        listed = {k: cutoff for k in required_keys()}
        self.assertEqual(len(listed), 12)
        self.assertEqual(classify_day(target, listed)["status"], "eligible")
        listed[required_keys()[0]] += timedelta(seconds=1)
        self.assertEqual(classify_day(target, listed)["status"], "late")
        listed.pop(required_keys()[-1])
        self.assertEqual(classify_day(target, listed)["status"], "missing")

    def test_paths_preserve_the_archive_layout(self):
        self.assertNotIn("/atmos/", url_for(date(2021, 3, 1), 27))
        self.assertIn("gfs.20240815/00/atmos/", url_for(date(2024, 8, 16), 30))
        self.assertTrue(url_for(date(2024, 8, 16), 30).endswith("sfluxgrbf030.grib2"))

    def test_real_eccodes_gaussian_decode_matches_coordinate_masking(self):
        import eccodes
        gid = eccodes.codes_grib_new_from_samples("regular_gg_sfc_grib2")
        try:
            order, ni, nj = 64, 256, 128
            lat = np.array(list(eccodes.codes_get_gaussian_latitudes(order)), dtype="float64")
            for key, value in (("N", order), ("Ni", ni), ("Nj", nj),
                               ("latitudeOfFirstGridPointInDegrees", lat[0]),
                               ("latitudeOfLastGridPointInDegrees", lat[-1]),
                               ("longitudeOfFirstGridPointInDegrees", 0.),
                               ("longitudeOfLastGridPointInDegrees", 360. - 360./ni),
                               ("iDirectionIncrementInDegrees", 360./ni),
                               ("stepType", "avg"), ("startStep", 24), ("endStep", 27)):
                eccodes.codes_set(gid, key, value)
            eccodes.codes_set_values(gid, np.zeros(ni * nj))
            latitudes = eccodes.codes_get_array(gid, "latitudes")
            longitudes = eccodes.codes_get_array(gid, "longitudes")
            values = 100 + .2 * latitudes + .03 * longitudes
            eccodes.codes_set_values(gid, values)
            packed_values = eccodes.codes_get_values(gid)
            box = (latitudes >= 33) & (latitudes < 39) & (longitudes >= 124) & (longitudes < 130)
            expected = packed_values[box].mean()
            actual = regional_means(eccodes.codes_get_message(gid), 24, 27)
            self.assertEqual(len(actual), len(REGIONS))
            self.assertAlmostEqual(actual[0], expected, places=7)
            _, indices = gaussian_regions(ni, nj, order)
            np.testing.assert_allclose(actual, [packed_values[index].mean() for index in indices])
        finally:
            eccodes.codes_release(gid)

    def test_resume_validates_stored_features_and_never_overwrites_invalid_cache(self):
        target = date(2024, 8, 16)
        means, starts = synthetic_means()
        def records(day, step):
            j = STEPS.index(step)
            return means[:, j], starts[:, j], ("2024-08-15T03:00:00+00:00",) * 2
        with TemporaryDirectory() as folder:
            directory = Path(folder)
            with patch("src.experiments.gfs_radiation.DIRECTORY", directory), patch("src.experiments.gfs_radiation.read_records", side_effect=records) as get:
                self.assertEqual(fetch_day(target), "downloaded")
                self.assertEqual(get.call_count, 6)
                self.assertEqual(fetch_day(target), "reused")
                self.assertEqual(get.call_count, 6)
                path = directory / "20240816.npz"
                self.assertTrue(cache_valid(path, target))
                with np.load(path) as z:
                    corrupted = {k: z[k] for k in z.files}
                corrupted["features"] = np.zeros((17, 20))
                np.savez(path, **corrupted)
                with self.assertRaises(ValueError):
                    fetch_day(target)
                self.assertEqual(get.call_count, 6)

    def test_failed_audit_cannot_create_a_data_directory(self):
        with self.assertRaises(ValueError):
            fetch({"passed_internal_coverage_gate": False}, workers=1)

    def test_failed_energy_closure_stops_before_bulk_audit_or_collection(self):
        values = np.full((6, 10), 100.)  # 네 성분의 합400이 총일사100과 불일치.
        with TemporaryDirectory() as folder:
            with patch("src.experiments.gfs_radiation.ENV_PATH", Path(folder)/"metrics.json"), \
                 patch("src.experiments.gfs_radiation.read_records", return_value=(values, np.full(6, 24), ())), \
                 patch("src.experiments.gfs_radiation.request") as get:
                result = preflight()
            self.assertFalse(result["ready"])
            self.assertEqual(result["failure_code"], "beam_diffuse_energy_closure_failed")
            get.assert_not_called()

    def test_preflight_success_checks_all_windows_and_repeat_retrieval(self):
        means, _ = synthetic_means()
        def records(target, step, fields=FIELDS):
            values = means[:, 2]
            if len(fields) == 6:
                values = np.concatenate([values, np.full((1, 10), 800.)], axis=0)
            return values, np.full(len(fields), 24), ()
        def get(url):
            step = int(url.split("sfluxgrbf")[1][:3])
            start = ((step - 1) // 6) * 6
            text = "\n".join(f"{i+1}:{100*i}:d=2024081500:{name}:surface:{start}-{step} hour ave fcst:"
                             for i, name in enumerate((*FIELDS, "DSWRF")))
            return text.encode(), {"Last-Modified": "Thu, 15 Aug 2024 03:00:00 GMT"}
        with TemporaryDirectory() as folder:
            with patch("src.experiments.gfs_radiation.ENV_PATH", Path(folder)/"metrics.json"), \
                 patch("src.experiments.gfs_radiation.read_records", side_effect=records) as reads, \
                 patch("src.experiments.gfs_radiation.request", side_effect=get) as gets:
                result = preflight()
        self.assertTrue(result["ready"])
        self.assertTrue(result["repeat_retrieval_regional_means_identical"])
        self.assertEqual(reads.call_count, 4)
        self.assertEqual(gets.call_count, 6)


if __name__ == "__main__":
    unittest.main()
