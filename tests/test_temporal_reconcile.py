"""합성 자료만 사용: 시간 계층/정답지원/조정 수학 검사."""
import unittest
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from src.experiments.temporal_reconcile import S, reconcile, labels, common_features, aggregate_features, daily_panel, coarse_predict
from tests.test_sequence_residual import synthetic

threadpool_limits(limits=1)


class TemporalTest(unittest.TestCase):
    def test_boundary_aggregation_and_zero_and_cutoff(self):
        ts = pd.date_range("2023-06-30 01:00", periods=24, freq="h", tz="Asia/Seoul")
        g = pd.DataFrame(dict(site_id="synthetic", ts=ts, energy=0.))
        end = ts[-1]
        day = labels(g, 24, end)
        self.assertEqual(len(day), 1)
        self.assertEqual(day.target.iloc[0], 0)
        self.assertEqual(day.index.get_level_values("day")[0].day, 30)
        cut = pd.Timestamp("2023-06-30 17:00", tz="Asia/Seoul")
        self.assertEqual(len(labels(g, 3, cut)), 5)
        self.assertEqual(len(labels(g, 24, cut)), 0)
        self.assertEqual(len(labels(g.drop(index=10), 3, end)), 7)
        self.assertEqual(len(labels(g.drop(index=10), 24, end)), 0)

    def test_coherent_forecast_identity_night_and_capacity_scaling(self):
        rng = np.random.default_rng(8)
        bottom = rng.uniform(0, 1, (3, 24))
        mask = np.zeros_like(bottom, bool)
        mask[:, 5:19] = True
        bottom[~mask] = 0
        nodes = np.einsum("ij,nj->ni", S, bottom, optimize=False)
        result, stats = reconcile(nodes, mask, np.ones(33))
        np.testing.assert_allclose(result, bottom, atol=1e-12)
        self.assertEqual(stats["constrained_site_days"], 0)
        np.testing.assert_array_equal(result[~mask], 0)
        scaled, _ = reconcile(nodes*10, mask, np.ones(33))
        np.testing.assert_allclose(scaled, result*10, atol=1e-12)

    def test_nonnegative_solution_improves_weighted_objective(self):
        mask = np.ones((1, 24), bool)
        nodes = np.zeros((1, 33))
        nodes[0, 0] = 100
        variance = np.ones(33)
        bottom, stats = reconcile(nodes, mask, variance)
        self.assertGreater(stats["constrained_site_days"], 0)
        self.assertTrue((bottom >= 0).all())
        predicted = np.einsum("ij,nj->ni", S, bottom, optimize=False)
        base = np.einsum("ij,nj->ni", S, nodes[:, :24], optimize=False)
        self.assertLess(((predicted-nodes)**2).sum(), ((base-nodes)**2).sum())

    def test_common_gfs_independent_of_site_and_labels(self):
        p, w, cols, g, sites = synthetic()
        x = aggregate_features(w, cols, g, 3)
        w2 = w.assign(site_id="second")
        w2["cosz"] = w2.cosz*.8
        both = aggregate_features(pd.concat([w, w2]), cols, g, 3)
        a = both.xs("synthetic").sort_index()
        b = both.xs("second").sort_index()
        np.testing.assert_array_equal(a[list(g)].to_numpy(), b[list(g)].to_numpy())
        self.assertEqual(len(x.columns), 146)
        # 정답행을삭제하고수치를바꿔도예보전용집계함수입력에는영향없다.
        p = p.drop(index=10).assign(energy=999.)
        pd.testing.assert_frame_equal(x, aggregate_features(w, cols, g, 3))
        self.assertEqual(len(common_features(g, 3)), 8)

    def test_partial_day_not_filled_as_observed_and_zero_day_retained(self):
        p, w, cols, g, sites = synthetic()
        full = daily_panel(p, sites)
        self.assertEqual(len(full["keys"]), 1)
        self.assertEqual(full["omitted_site_days"], 0)
        with self.assertRaises(ValueError):
            daily_panel(p.drop(index=10), sites)

    def test_coarse_prediction_maps_all_bins_and_night(self):
        class Constant:
            def predict(self, x):
                return np.ones(len(x))
        p, w, cols, g, sites = synthetic()
        days = daily_panel(p, sites)
        for resolution in (3, 24):
            table = aggregate_features(w, cols, g, resolution)
            pred = coarse_predict([Constant()], table, days, resolution)
            self.assertEqual(pred.shape, (1, 24//resolution))
            self.assertAlmostEqual(pred.sum(), float((table.cap*table.cs_sum).sum()))
            dark = ~days["daylight"].reshape(1, 24//resolution, resolution).any(axis=2)
            np.testing.assert_array_equal(pred[dark], 0)


if __name__ == "__main__":
    unittest.main()
