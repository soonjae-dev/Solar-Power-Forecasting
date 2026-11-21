"""합성 자료로 누출 차단·스케일·이웃 수·비음수를 검증한다."""
import unittest

import numpy as np
import pandas as pd

from src.analog_residual import FEATURES, correct, weighted_median


def example(n=30):
    ts = pd.date_range("2023-01-01 12:00", periods=n, freq="D", tz="Asia/Seoul")
    idx = pd.MultiIndex.from_product([ts, ["A", "B"]], names=["ts", "group"])
    h = pd.DataFrame(1., index=idx, columns=FEATURES)
    h["scale"], h["base"], h["weight"], h["residual"] = 100., 50., 100., .1
    q = h.iloc[-2:].copy()
    q.index = pd.MultiIndex.from_product([[ts[-1]+pd.Timedelta(days=2)], ["A", "B"]], names=idx.names)
    return h, q, ts[-1]+pd.Timedelta(hours=5)


class AnalogTest(unittest.TestCase):
    def test_weighted_median(self):
        self.assertEqual(weighted_median([1, 20, 3], [1, 1, 10]), 3)

    def test_future_history_and_query_labels_cannot_change_predictions(self):
        h, q, cutoff = example()
        a, _ = correct(h, q, cutoff)
        future = h.iloc[:2].copy()
        future.index = q.index
        future["residual"] = 1e6
        q["actual"], q["residual"] = -1e8, 1e9
        b, _ = correct(pd.concat([h, future]), q, cutoff)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_allclose(a, 55.)

    def test_missing_and_short_history_are_identity(self):
        h, q, cutoff = example(19)
        a, _ = correct(h, q, cutoff)
        np.testing.assert_array_equal(a, q.base)
        h, q, cutoff = example()
        q[FEATURES[0]] = np.nan
        a, _ = correct(h, q, cutoff)
        np.testing.assert_array_equal(a, q.base)

    def test_group_hour_separation_and_nonnegative_limit(self):
        h, q, cutoff = example()
        h.loc[(slice(None), "A"), "residual"] = -100.
        q["base"] = 2.
        a, _ = correct(h, q, cutoff)
        np.testing.assert_allclose(a, [0., 7.])
        q.index = q.index.set_levels(q.index.levels[0]+pd.Timedelta(hours=1), level="ts")
        b, _ = correct(h, q, cutoff)
        np.testing.assert_array_equal(b, q.base)

    def test_capacity_scaling_is_equivariant(self):
        h, q, cutoff = example()
        a, _ = correct(h, q, cutoff)
        h[["scale", "base", "weight"]] *= 3
        q[["scale", "base", "weight"]] *= 3
        b, _ = correct(h, q, cutoff)
        np.testing.assert_allclose(b, a*3)

    def test_duplicate_and_overlap_are_rejected(self):
        h, q, cutoff = example()
        with self.assertRaises(ValueError):
            correct(pd.concat([h, h.iloc[:1]]), q, cutoff)
        with self.assertRaises(ValueError):
            correct(h, h.iloc[-2:], cutoff)


if __name__ == "__main__":
    unittest.main()
