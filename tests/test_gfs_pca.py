"""PCA 시점 분리와 판정 검사. 합성 입력만 사용한다."""
import unittest

import numpy as np
import pandas as pd

from src.config import TZ
from src.gfs_pca import scores, verdict


def records(changes, static, base=100.0):
    names = ("summer_support", "autumn_support", "development", "selection")
    out = {n: dict(vs_matched_v3=dict(change_pct=c, base_penalty=base, direct_penalty=base * (1 + c / 100))) for n, c in zip(names, changes)}
    out["fold1_static"] = dict(vs_matched_v3=dict(change_pct=static))
    return out


class GfsPcaTest(unittest.TestCase):
    def test_future_values_do_not_change_fit(self):
        stamps = pd.date_range("2024-01-01 07:00", periods=2000, freq="h", tz=TZ)
        rng = np.random.default_rng(1)
        field = rng.normal(size=(2000, 20)).astype("float32")
        cutoff = stamps[1500]
        changed = field.copy()
        changed[1501:] += 100
        a, ra = scores(field, stamps, cutoff, components=3)
        b, rb = scores(changed, stamps, cutoff, components=3)
        self.assertEqual(ra, rb)
        np.testing.assert_allclose(a[:1501], b[:1501])

    def test_too_little_history_is_rejected(self):
        stamps = pd.date_range("2024-01-01 07:00", periods=100, freq="h", tz=TZ)
        with self.assertRaises(ValueError):
            scores(np.zeros((100, 5), dtype="float32"), stamps, stamps[-1], components=2)

    def test_verdict_needs_all_conditions(self):
        self.assertTrue(verdict(records((-3, -2, -1, 1), -0.5))["passed"])
        self.assertFalse(verdict(records((-1, -1, -1, 2), -0.5))["passed"])
        self.assertFalse(verdict(records((-3, -2, -1, 1), 0.5))["passed"])
        self.assertFalse(verdict(records((-5, -5, 4, -1), -0.5))["passed"])
        self.assertFalse(verdict(records((-5, -5, 1, 1), -0.5))["passed"])


if __name__ == "__main__":
    unittest.main()
