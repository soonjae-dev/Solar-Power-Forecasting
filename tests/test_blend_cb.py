"""v6·CatBoost 결합 집계 검사. 합성 입력만 사용한다."""
import unittest

import numpy as np
import pandas as pd

from src.experiments.blend_cb import blend_metrics
from src.direct_groups import metrics


def synthetic(seed=0, hours=24*21):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-06-01 01:00", periods=hours, freq="h")
    idx = pd.MultiIndex.from_product([ts, ["A", "B"]], names=["ts", "group"])
    a = pd.Series(rng.uniform(0, 100, len(idx)), idx)
    base = (a + rng.normal(0, 10, len(idx))).clip(lower=0)
    cb = (a + rng.normal(0, 10, len(idx))).clip(lower=0)
    smp = pd.DataFrame({"w": rng.uniform(50, 150, len(ts))}, index=ts)
    return a, base, cb, smp


class BlendTest(unittest.TestCase):
    def test_blend_is_half_half_of_group_sums(self):
        a, base, cb, smp = synthetic()
        expected = metrics(a, base, cb, smp)["direct_penalty"]
        r = blend_metrics(a, base, cb, smp, expected)
        self.assertEqual(r["direct_penalty"], metrics(a, base, .5*base + .5*cb, smp)["direct_penalty"])
        self.assertEqual(r["base_penalty"], metrics(a, base, base, smp)["base_penalty"])

    def test_blend_penalty_not_above_mean_of_two(self):
        a, base, cb, smp = synthetic(1)
        alone = metrics(a, base, cb, smp)
        r = blend_metrics(a, base, cb, smp, alone["direct_penalty"])
        self.assertLessEqual(r["direct_penalty"], (alone["base_penalty"] + alone["direct_penalty"])/2 + 1)

    def test_reproduction_mismatch_stops(self):
        a, base, cb, smp = synthetic(2)
        wrong = metrics(a, base, cb, smp)["direct_penalty"] + 1
        with self.assertRaises(ValueError):
            blend_metrics(a, base, cb, smp, wrong)


if __name__ == "__main__":
    unittest.main()
