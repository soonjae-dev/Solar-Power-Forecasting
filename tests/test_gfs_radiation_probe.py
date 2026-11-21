"""공개 표본 경로·종류·시간 검증. 합성 입력만 사용."""
from datetime import date
import unittest
from unittest.mock import patch

from src.experiments.gfs_radiation_probe import FIELDS, sample, url_for


class RadiationProbeTest(unittest.TestCase):
    def test_archive_paths_before_and_after_atmos_change(self):
        self.assertNotIn("/atmos/", url_for(date(2021, 3, 1)))
        url = url_for(date(2024, 8, 16))
        self.assertIn("gfs.20240815/00/atmos/", url)
        self.assertIn("gfs.t00z.sfluxgrbf027.grib2", url)
        self.assertNotIn("gdas", url)

    def test_network_failure_does_not_read_bodies(self):
        with patch("src.experiments.gfs_radiation_probe.probe", return_value=dict(status="network_unverified")), \
             patch("src.experiments.gfs_radiation_probe.request") as get:
            self.assertEqual(sample(date(2024, 8, 16))["status"], "network_unverified")
            get.assert_not_called()

    def test_success_only_exposes_header_metadata(self):
        idx = "\n".join(f"{i+1}:{i*100}:d=2024081500:{name}:surface:24-27 hour ave fcst:"
                         for i, name in enumerate((*FIELDS, "DLWRF")))
        def get(url, byte_range=None):
            return (idx.encode() if byte_range is None else b"synthetic-grib"), {"Last-Modified": "Thu, 15 Aug 2024 03:00:00 GMT"}
        meta = dict(gridType="regular_gg", Ni=8, Nj=4, numberOfPoints=32,
                    stepType="avg", startStep=24, endStep=27)
        with patch("src.experiments.gfs_radiation_probe.probe", return_value=dict(status="reachable", last_modified="Thu, 15 Aug 2024 03:00:00 GMT")), \
             patch("src.experiments.gfs_radiation_probe.request", side_effect=get), \
             patch("src.experiments.gfs_radiation_probe.metadata", return_value=meta):
            result = sample(date(2024, 8, 16))
        self.assertEqual(result["status"], "eligible_sample")
        self.assertEqual(set(result["fields"]), set(FIELDS))
        self.assertTrue(all(set(v) == set(meta) for v in result["fields"].values()))

    def test_late_publication_prevents_body_download(self):
        with patch("src.experiments.gfs_radiation_probe.probe", return_value=dict(status="reachable", last_modified="Thu, 15 Aug 2024 08:00:01 GMT")), \
             patch("src.experiments.gfs_radiation_probe.request") as get:
            self.assertEqual(sample(date(2024, 8, 16))["status"], "unverified")
            get.assert_not_called()


if __name__ == "__main__":
    unittest.main()
