"""외부 예보 공개 시각 경계와 HTTP 실패 구분. 합성 입력만 사용한다."""
import unittest
from datetime import datetime, timezone
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from src.experiments.external_weather_audit import classify_day, eligible_publication, parse_listing, prediction_cutoff, probe, required_keys


class ExternalWeatherAuditTest(unittest.TestCase):
    def test_exact_cutoff_inclusive_and_one_second_later_rejected(self):
        self.assertTrue(eligible_publication("2025-06-01", datetime.fromisoformat("2025-05-31T08:00:00+00:00")))
        self.assertFalse(eligible_publication("2025-06-01", datetime.fromisoformat("2025-05-31T08:00:01+00:00")))
        self.assertEqual(prediction_cutoff("2025-06-01").astimezone(timezone.utc).hour, 8)

    def test_unknown_publication_is_not_inferred_from_cycle(self):
        self.assertFalse(eligible_publication("2025-06-01", None))
        self.assertFalse(eligible_publication("2025-06-01", datetime(2025, 5, 31)))

    def test_network_failure_does_not_claim_missing_file(self):
        with patch("src.experiments.external_weather_audit.urlopen", side_effect=URLError("synthetic")):
            self.assertEqual(probe("https://example.com")['status'], "network_unverified")
        with patch("src.experiments.external_weather_audit.urlopen", side_effect=HTTPError("https://example.com", 404, "", {}, None)):
            result = probe("https://example.com")
            self.assertEqual(result['status'], "http_error")
            self.assertEqual(result['http_status'], 404)

    def test_classify_day_requires_all_files_before_cutoff(self):
        ok = datetime.fromisoformat("2025-05-31T08:00:00+00:00")
        listed = {k: ok for k in required_keys()}
        self.assertEqual(classify_day("2025-06-01", listed)["status"], "eligible")
        late = {**listed, required_keys()[-1]: datetime.fromisoformat("2025-05-31T08:00:01+00:00")}
        self.assertEqual(classify_day("2025-06-01", late)["status"], "late")
        partial = dict(listed)
        del partial[required_keys()[0]]
        self.assertEqual(classify_day("2025-06-01", partial)["status"], "missing")

    def test_cycle18_keys_shift_six_hours(self):
        keys = required_keys(18)
        self.assertEqual(keys[0], "gfs.t18z.pgrb2.0p25.f021")
        self.assertEqual(keys[24], "gfs.t18z.pgrb2.0p25.f045")

    def test_parse_listing_keeps_filename_and_utc_time(self):
        xml = (b'<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/"><Contents>'
               b'<Key>gfs.20250531/00/atmos/a.f016</Key><LastModified>2025-05-31T03:42:24.000Z</LastModified>'
               b'</Contents></ListBucketResult>')
        self.assertEqual(parse_listing(xml)["a.f016"], datetime.fromisoformat("2025-05-31T03:42:24+00:00"))


if __name__ == "__main__":
    unittest.main()
