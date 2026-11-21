"""프로필의 시점 분리, 미관측 처리, 타깃 복원 및 손실 동등성. 합성 입력만 사용."""
import unittest

import numpy as np
import pandas as pd

from src.config import TZ
from src.profile_target import (apply_profile, base_target, fit_profile, geometry_bin,
                             normalized_target, quarterly_scales)
from src.train import clear_sky
from src.v6_error_budget import decompose, history_cohort


def synthetic():
    stamps = pd.date_range("2023-01-01 12:00", periods=45, freq="D", tz=TZ)
    parts = []
    for site, kt in (("synthetic_low", .8), ("synthetic_high", 1.2)):
        parts.append(pd.DataFrame(dict(site_id=site, ts=stamps, cap=100., cosz=.8,
                                       azim=180., wa_cloud=0., wb_cloud=0.,
                                       energy=100*clear_sky(np.full(len(stamps), .8))*kt)))
    return pd.concat(parts, ignore_index=True)


class ProfileTargetTest(unittest.TestCase):
    def test_future_labels_cannot_change_fitted_profile(self):
        frame = synthetic()
        future = frame.iloc[:4].copy()
        future["ts"] += pd.Timedelta(days=180)
        a = pd.concat([frame, future], ignore_index=True)
        b = a.copy()
        b.loc[b.ts > pd.Timestamp("2023-03-31", tz=TZ), "energy"] = 1e9
        cutoff = pd.Timestamp("2023-03-31 17:00", tz=TZ)
        pa, pb = fit_profile(a, cutoff), fit_profile(b, cutoff)
        pd.testing.assert_series_equal(pa.site_factor, pb.site_factor)
        pd.testing.assert_series_equal(pa.cell_factor, pb.cell_factor)

    def test_training_profiles_never_use_the_same_quarter_labels(self):
        past = synthetic()
        later = past.copy()
        later["ts"] += pd.Timedelta(days=100)
        frame = pd.concat([past, later], ignore_index=True)
        changed = frame.copy()
        changed.loc[changed.ts >= pd.Timestamp("2023-04-01", tz=TZ), "energy"] = 0
        a, diagnostics = quarterly_scales(frame)
        b, _ = quarterly_scales(changed)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(a[:len(past)], 1.)
        self.assertEqual(diagnostics["2023-04-01"]["profile_cutoff"], "2023-03-31T17:00:00+09:00")

    def test_unknown_and_short_history_sites_use_unit_factor(self):
        frame = synthetic()
        cutoff = pd.Timestamp("2023-02-01", tz=TZ)
        short = frame.iloc[:10].copy()
        profile = fit_profile(short, cutoff)
        np.testing.assert_array_equal(apply_profile(profile, frame), 1.)
        profile = fit_profile(frame, pd.Timestamp("2023-03-01", tz=TZ))
        unknown = frame.iloc[:3].copy()
        unknown["site_id"] = "synthetic_unseen"
        np.testing.assert_array_equal(apply_profile(profile, unknown), 1.)

    def test_factors_are_shrunk_and_geometry_fallback_is_site_level(self):
        frame = synthetic()
        profile = fit_profile(frame, pd.Timestamp("2023-03-01", tz=TZ))
        factor = apply_profile(profile, frame)
        self.assertTrue(np.all((factor >= .5) & (factor <= 1.5)))
        self.assertLess(factor[0], 1.)
        self.assertGreater(factor[0], .8/1.2)
        low_sun = frame.iloc[:3].copy()
        low_sun["cosz"] = .2
        np.testing.assert_allclose(apply_profile(profile, low_sun), profile.site_factor.reindex(low_sun.site_id).to_numpy())

    def test_identical_response_sites_are_identity(self):
        frame = synthetic()
        frame["energy"] = 100*clear_sky(frame.cosz)
        profile = fit_profile(frame, pd.Timestamp("2023-03-01", tz=TZ))
        np.testing.assert_array_equal(apply_profile(profile, frame), 1.)

    def test_target_is_not_clipped_again_and_weight_preserves_l2(self):
        y = np.array([1.9, .8, 1.2])
        factor = np.array([.5, 1.5, .8])
        weight = np.array([1., 2., 3.])
        z, wz = normalized_target(y, factor, weight)
        self.assertGreater(z[0], 2.)
        np.testing.assert_allclose(z*factor, y)
        forecast = np.array([2., .5, 1.])
        self.assertAlmostEqual(float(np.sum(weight*(forecast*factor-y)**2)), float(np.sum(wz*(forecast-z)**2)))

    def test_original_target_clipping_order_matches_v6(self):
        frame = synthetic().iloc[:3].copy()
        frame["energy"] = [-1., 10., 1e6]
        expected = ((frame.energy/frame.cap).clip(0,1.2)/clear_sky(frame.cosz)).clip(0,2)
        np.testing.assert_array_equal(base_target(frame), expected)

    def test_geometry_cells_and_low_sun(self):
        frame = synthetic().iloc[:4].copy()
        frame["cosz"] = [.2, .5, .8, .8]
        frame["azim"] = [180., 100., 180., 250.]
        np.testing.assert_array_equal(geometry_bin(frame), [-1, 0, 4, 5])

    def test_decomposition_is_a_fixed_prediction_bound(self):
        residual = pd.DataFrame([[3., -2.], [4., 2.]])
        penalty, floor = decompose(residual, pd.Series([10., 20.]))
        np.testing.assert_allclose(penalty, [5., 12.])
        np.testing.assert_allclose(floor, [1., 12.])

    def test_history_uses_only_prior_observations(self):
        cutoff = pd.Timestamp("2024-06-01", tz=TZ)
        first = pd.Series([pd.NaT, cutoff-pd.Timedelta(days=100), cutoff-pd.Timedelta(days=500), cutoff-pd.Timedelta(days=800)])
        self.assertEqual(history_cohort(first, cutoff).tolist(), ["unseen", "under_180_days", "180_to_729_days", "730_days_or_more"])


if __name__ == "__main__":
    unittest.main()
