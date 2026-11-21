"""기여도 상쇄·작은 코호트 억제·미래 이력 차단. 합성 입력만 사용."""
import unittest
import numpy as np
import pandas as pd
from src.experiments.cb_diagnostic import attribute, recent_state, summary


class DiagnosticTest(unittest.TestCase):
    def test_contributions_sum_to_group_loss_delta_with_cancellation(self):
        ts = pd.Timestamp("2024-01-01 12:00", tz="Asia/Seoul")
        p = pd.DataFrame(dict(site_id=["synthetic1", "synthetic2"], ts=[ts]*2,
                              group=["A"]*2, cap=[100., 100.], energy=[10., 10.],
                              forecast_kwh=[20., 5.], cb=[5., 20.]))
        smp = pd.DataFrame(dict(w=[100.]), index=[ts])
        attributed, delta = attribute(p, smp)
        self.assertAlmostEqual(delta, 0.)
        self.assertAlmostEqual(attributed.contribution.sum(), 0.)
        self.assertTrue((attributed.contribution < 0).any())
        p["cb"] = [10., 10.]
        _, delta = attribute(p, smp)
        self.assertAlmostEqual(delta, -50.)

    def test_small_cohort_only_exposes_count(self):
        p = pd.DataFrame(dict(site_id=["synthetic1"]*4))
        self.assertEqual(summary(p, None), dict(site_count=1, suppressed=True))

    def test_recent_state_cannot_use_future_labels(self):
        cutoff = pd.Timestamp("2024-05-01 17:00", tz="Asia/Seoul")
        ts = pd.date_range("2024-01-01 12:00", "2024-06-01 12:00", freq="D", tz="Asia/Seoul")
        p = pd.DataFrame(dict(site_id="synthetic", ts=ts, cap=100., cosz=.8, energy=60.))
        p.loc[(p.ts > cutoff-pd.Timedelta(days=30)) & (p.ts <= cutoff), "energy"] = 6.
        a = recent_state(p, cutoff)
        p.loc[p.ts > cutoff, "energy"] = 1e6
        b = recent_state(p, cutoff)
        pd.testing.assert_series_equal(a, b)
        self.assertEqual(a.iloc[0], "recent_low")
        self.assertEqual(recent_state(p.iloc[-10:], cutoff).size, 0)


if __name__ == "__main__":
    unittest.main()
