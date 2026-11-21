"""GEFS 평균 수집의 시간·공개 시각·익명 범위 요청·재시작 검사. 합성 자료만 사용."""
from datetime import date, timedelta
from pathlib import Path
import socket
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from src.experiments.external_weather_audit import prediction_cutoff
from src.experiments.gefs_mean import (SCHEMA, STEPS, block_means, cache_valid, classify_day, expand_blocks,
                       fetch, fetch_day, mean_record, preflight, published_before_cutoff, request,
                       required_keys, run_url, validate_directory)
from src.gfs_fetch import LATS, LONS


class MeanTest(unittest.TestCase):
    def test_averages_and_hour_ends(self):
        truth = {s: np.full((2, 3), float(s)) for s in STEPS}
        averages = {}
        for i, step in enumerate(STEPS):
            if i % 2 == 0:
                averages[step] = (step - 3, truth[step])
            else:
                averages[step] = (step - 6, (truth[step - 3] + truth[step]) / 2)
        blocks = block_means(averages)
        for step in STEPS:
            np.testing.assert_array_equal(blocks[step], truth[step])
        expanded = expand_blocks(blocks)
        self.assertEqual(expanded.shape, (17, 2, 3))
        np.testing.assert_array_equal(expanded[:3], np.full((3, 2, 3), 21.))
        np.testing.assert_array_equal(expanded[-2:], np.full((2, 2, 3), 36.))
        # 6시간 평균은 인접3시간의 길이 가중 평균으로 보존된다.
        np.testing.assert_array_equal(expanded[:6].mean(axis=0), averages[24][1])

    def test_bad_reset_is_rejected(self):
        averages = {s: (s - 3, np.array(1.)) for s in STEPS}
        averages[24] = (12, np.array(1.))
        with self.assertRaises(ValueError):
            block_means(averages)
        averages[24] = (18, np.array(1.))
        averages[21] = (15, np.array(1.))
        with self.assertRaises(ValueError):
            block_means(averages)

    def test_mean_record_rejects_spread_and_wrong_end(self):
        text = ("1:0:d=2024081500:TCDC:entire atmosphere:24-27 hour ave fcst:ens mean\n"
                "2:100:d=2024081500:DSWRF:surface:24-27 hour ave fcst:ens mean\n"
                "3:200:d=2024081500:DLWRF:surface:24-27 hour ave fcst:ens mean\n")
        self.assertEqual(mean_record(text, "DSWRF", "surface", 27), (100, 199, 24, 27))
        with self.assertRaises(ValueError):
            mean_record(text.replace("ens mean", "ens std dev"), "DSWRF", "surface", 27)
        with self.assertRaises(ValueError):
            mean_record(text, "DSWRF", "surface", 30)

    def test_all_bodies_and_indexes_are_required(self):
        target = date(2024, 8, 16)
        cutoff = prediction_cutoff(target.isoformat())
        listed = {k: cutoff for k in required_keys()}
        self.assertEqual(len(listed), 12)
        self.assertEqual(classify_day(target, listed)["status"], "eligible")
        listed[required_keys()[-1]] += timedelta(seconds=1)
        self.assertEqual(classify_day(target, listed)["status"], "late")
        del listed[required_keys()[0]]
        self.assertEqual(classify_day(target, listed)["status"], "missing")

    def test_get_headers_recheck_deadline(self):
        target = date(2024, 8, 16)
        self.assertIsInstance(published_before_cutoff(target, {"Last-Modified": "Thu, 15 Aug 2024 08:00:00 GMT"}), str)
        for headers in ({}, {"last-modified": "Thu, 15 Aug 2024 08:00:01 GMT"}):
            with self.assertRaises(ValueError):
                published_before_cutoff(target, headers)

    def test_range_request_needs_206_and_exact_length(self):
        class Response:
            status = 206
            headers = {"Content-Range": "bytes 10-12/100"}
            def read(self, _):
                return b"abc"
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
        with patch("src.experiments.gefs_mean.urlopen", return_value=Response()) as opened:
            self.assertEqual(request("https://example.invalid/public", (10, 12))[0], b"abc")
            self.assertEqual(opened.call_args.args[0].get_header("Range"), "bytes=10-12")
            Response.status = 200
            with self.assertRaises(ValueError):
                request("https://example.invalid/public", (10, 12))

    def test_dns_failure_is_not_missing_file_or_ready(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "metrics.json"
            with patch("src.experiments.gefs_mean.ENV_PATH", path), \
                 patch("src.experiments.gefs_mean.socket.getaddrinfo", side_effect=socket.gaierror(8, "synthetic")), \
                 patch("src.experiments.gefs_mean.request") as get, \
                 patch("src.experiments.gefs_mean.importlib.metadata.version", return_value="test"):
                result = preflight()
            self.assertFalse(result["ready"])
            self.assertEqual(result["blocked_at"], "dns")
            get.assert_not_called()
            self.assertTrue(path.exists())

    def test_source_day_is_previous_day(self):
        url = run_url(date(2025, 1, 1), 36)
        self.assertIn("gefs.20241231/00/", url)
        self.assertTrue(url.endswith("geavg.t00z.pgrb2s.0p25.f036"))

    def test_cache_validates_time_shape_and_schema(self):
        target = date(2024, 8, 16)
        fields = {k: np.zeros((17, len(LATS), len(LONS)), dtype="float32") for k in ("dswrf", "tcdc")}
        with TemporaryDirectory() as folder:
            path = Path(folder) / "sample.npz"
            stamps = np.array(["2024-08-15T03:00:00+00:00"] * 12)
            np.savez(path, **fields, schema=SCHEMA, target_day=target.isoformat(), last_modified=stamps, source_steps=STEPS)
            self.assertTrue(cache_valid(path, target))
            np.savez(path, **fields, schema="old", target_day=target.isoformat(), last_modified=stamps, source_steps=STEPS)
            self.assertFalse(cache_valid(path, target))
            stamps[-1] = "2024-08-15T08:00:01+00:00"
            np.savez(path, **fields, schema=SCHEMA, target_day=target.isoformat(), last_modified=stamps, source_steps=STEPS)
            self.assertFalse(cache_valid(path, target))

    def test_workspace_cannot_be_data_destination(self):
        from src.config import ROOT
        with self.assertRaises(ValueError):
            validate_directory(ROOT / "external" / "gefs_mean_00utc")

    def test_existing_invalid_cache_is_not_overwritten(self):
        target = date(2024, 8, 16)
        with TemporaryDirectory() as folder:
            directory = Path(folder)
            path = directory / "20240816.npz"
            path.write_bytes(b"synthetic-invalid-cache")
            with patch("src.experiments.gefs_mean.DIRECTORY", directory), patch("src.experiments.gefs_mean.request") as get:
                with self.assertRaises(ValueError):
                    fetch_day(target)
            get.assert_not_called()
            self.assertEqual(path.read_bytes(), b"synthetic-invalid-cache")

    def test_failed_audit_cannot_start_downloads(self):
        with patch("src.experiments.gefs_mean.validate_directory") as directory:
            with self.assertRaises(ValueError):
                fetch({"passed_internal_coverage_gate": False})
            directory.assert_not_called()

    def test_download_and_restart_use_same_verified_cache(self):
        target = date(2024, 8, 16)
        def get(url, byte_range=None):
            step = int(url.removesuffix(".idx").rsplit("f", 1)[-1])
            start = step - 3 if STEPS.index(step) % 2 == 0 else step - 6
            idx = (f"1:0:d=2024081500:TCDC:entire atmosphere:{start}-{step} hour ave fcst:ens mean\n"
                   f"2:100:d=2024081500:DSWRF:surface:{start}-{step} hour ave fcst:ens mean\n"
                   f"3:200:d=2024081500:DLWRF:surface:{start}-{step} hour ave fcst:ens mean\n")
            return (idx.encode("ascii") if byte_range is None else b"synthetic-grib"), {"Last-Modified": "Thu, 15 Aug 2024 03:00:00 GMT"}
        with TemporaryDirectory() as folder:
            directory = Path(folder)
            field = np.full((len(LATS), len(LONS)), 10., dtype="float32")
            with patch("src.experiments.gefs_mean.DIRECTORY", directory), \
                 patch("src.experiments.gefs_mean.request", side_effect=get) as opened, \
                 patch("src.experiments.gefs_mean.decode", return_value=field):
                self.assertEqual(fetch_day(target), "downloaded")
                count = opened.call_count
                self.assertEqual(fetch_day(target), "reused")
                self.assertEqual(opened.call_count, count)
            self.assertTrue(cache_valid(directory / "20240816.npz", target))


if __name__ == "__main__":
    unittest.main()
