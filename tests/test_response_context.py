"""Synthetic only: causal context, shrinkage, annual freezing and missing input."""
import unittest
import numpy as np
import pandas as pd
from src.experiments.response_context import (COLS, basis, solve_ridge, fit_context, apply_context,
                               annual_training_context, ContextError, metrics)


def fixture():
    ts = pd.date_range("2021-01-01 12:00", "2024-08-01 12:00", freq="7D", tz="Asia/Seoul")
    n = len(ts)
    a = np.linspace(0, 100, n)
    return pd.DataFrame(dict(site_id=["synthetic"]*n, ts=ts, energy=10+(100-a)*.5,
                            cap=100., cosz=.7, azim=180., wa_cloud=a, wb_cloud=a*.8,
                            wa_temp=20., wb_temp=21.))


class ResponseContextTests(unittest.TestCase):
    def test_aggregate_metric_entrypoint(self):
        times = pd.date_range("2024-06-01 01:00", periods=48, freq="h", tz="Asia/Seoul")
        index = pd.MultiIndex.from_product([times, ["A", "B"]], names=["ts", "group"])
        actual = pd.Series(10., index=index)
        smp = pd.DataFrame({"w": np.ones(len(times))}, index=times)
        result = metrics(actual, actual+2, actual+1, smp)
        self.assertEqual(result["change_pct"], -50.)

    def test_ridge_matches_augmented_least_squares(self):
        rng = np.random.default_rng(1)
        x, y, w = rng.normal(size=(20, 8)), rng.normal(size=20), rng.uniform(.1, 1, 20)
        actual = solve_ridge(x, y, w, 3.)
        design = np.vstack([x*np.sqrt(w[:, None]), np.sqrt(3)*np.eye(8)])
        target = np.r_[y*np.sqrt(w), np.zeros(8)]
        np.testing.assert_allclose(actual, np.linalg.lstsq(design, target, rcond=None)[0], atol=1e-12)

    def test_cutoff_excludes_future_targets(self):
        f = fixture()
        origin = pd.Timestamp("2023-01-01", tz="Asia/Seoul")
        cutoff = origin-pd.Timedelta(hours=7)
        a = fit_context(f, cutoff)
        changed = f.copy()
        changed.loc[changed.ts > cutoff, "energy"] = 1e20
        b = fit_context(changed, cutoff)
        pd.testing.assert_frame_equal(a.coefficients, b.coefficients)
        pd.testing.assert_series_equal(a.days, b.days)

    def test_query_targets_and_missing_labels_do_not_affect_input(self):
        f = fixture()
        origin = pd.Timestamp("2023-01-01", tz="Asia/Seoul")
        c = fit_context(f, origin-pd.Timedelta(hours=7))
        query = f.loc[f.ts.between(origin, origin+pd.Timedelta(days=300))].copy()
        a = apply_context(c, query, origin)
        query["energy"] = np.nan
        b = apply_context(c, query, origin)
        pd.testing.assert_frame_equal(a, b)

    def test_unknown_site_and_empty_history(self):
        f = fixture()
        origin = pd.Timestamp("2020-12-31", tz="Asia/Seoul")
        c = fit_context(f, origin-pd.Timedelta(hours=7))
        values = apply_context(c, f.iloc[:1], origin)
        np.testing.assert_array_equal(values[COLS[:10]].to_numpy(), np.zeros((1, 10)))

    def test_missing_query_keeps_coefficients(self):
        f = fixture()
        origin = pd.Timestamp("2023-01-01", tz="Asia/Seoul")
        c = fit_context(f, origin-pd.Timedelta(hours=7))
        q = f.loc[f.ts > origin].iloc[:1].copy()
        q["wa_cloud"] = np.nan
        a = apply_context(c, q, origin)
        self.assertEqual(a.response_query.iloc[0], 0)
        self.assertEqual(a.response_query_valid.iloc[0], 0)
        self.assertGreater(a.response_reliability.iloc[0], 0)

    def test_annual_context_frozen_inside_block_and_no_current_labels(self):
        f = fixture()
        origin = pd.Timestamp("2024-06-01", tz="Asia/Seoul")
        a, diag = annual_training_context(f, origin)
        changed = f.copy()
        changed.loc[changed.ts >= origin, "energy"] = 1e9
        b, _ = annual_training_context(changed, origin)
        pd.testing.assert_frame_equal(a, b)
        mask = (f.ts >= origin)
        self.assertTrue(np.all(a.loc[mask, COLS[:8]].to_numpy() == a.loc[mask, COLS[:8]].iloc[0].to_numpy()))
        self.assertGreater(len(diag), 2)

    def test_right_closed_midnight_belongs_to_previous_day(self):
        f = fixture().iloc[:2].copy()
        f["ts"] = pd.DatetimeIndex(["2023-06-01 00:00", "2023-06-01 01:00"], tz="Asia/Seoul")
        a, diag = annual_training_context(f, pd.Timestamp("2024-06-01", tz="Asia/Seoul"))
        self.assertEqual(len(diag), 2)
        self.assertGreater(a.response_age.iloc[0], .99)
        self.assertEqual(a.response_age.iloc[1], 0)

    def test_zero_generation_retained_and_ridge_shrinks(self):
        f = fixture()
        origin = pd.Timestamp("2024-01-01", tz="Asia/Seoul")
        f.energy = 0
        c = fit_context(f, origin-pd.Timedelta(hours=7))
        self.assertGreater(c.rows, 0)
        np.testing.assert_array_equal(c.coefficients.to_numpy(), np.zeros((1, 8)))


if __name__ == "__main__":
    unittest.main()
