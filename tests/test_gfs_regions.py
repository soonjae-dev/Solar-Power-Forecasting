"""고정 영역이 원래 상자를 정확히 분할하는지 합성 검사한다."""
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import TZ
from src.gfs_pilot import region_masks, summarize
from src.gfs_regions import early_failure, fine_masks
from src.tune_v3 import run_stage


class FineRegionsTest(unittest.TestCase):
    def test_early_failure_requires_an_irreversible_constraint_failure(self):
        def row(change):
            return dict(vs_matched_v3=dict(change_pct=change))
        self.assertIsNone(early_failure({"summer_support": row(.5), "fold1_static": row(-.1)}))
        self.assertEqual(early_failure({"fold1_static": row(.1)}), "fixed_11_months_worse")
        self.assertEqual(early_failure({"summer_support": row(3.1)}), "season_above_3pct")
        self.assertEqual(early_failure({"summer_support": row(0), "autumn_support": row(.1)}), "two_nonimproving_seasons")
    def test_partition_matches_original_box(self):
        masks = fine_masks()
        self.assertEqual(len(masks), 37)
        coverage = np.stack([m for n, m in masks.items() if n != "all"]).sum(axis=0)
        np.testing.assert_array_equal(coverage, region_masks()["all"].astype(int))
        self.assertTrue(all(m.sum() == 16 for n, m in masks.items() if n != "all"))

    def test_coarse_means_recovered_from_fine(self):
        rng = np.random.default_rng(2026)
        masks = fine_masks()
        field = rng.normal(size=(3, *masks["all"].shape))
        fine = summarize(field, masks)
        coarse = summarize(field, region_masks())
        for i in range(3):
            for j in range(3):
                recovered = np.mean([fine[f"r{2*i+a}{2*j+b}"] for a in range(2) for b in range(2)], axis=0)
                np.testing.assert_allclose(recovered, coarse[f"r{i}{j}"], atol=1e-12)

    def test_replacement_is_not_an_addition_to_model_inputs(self):
        cutoff = pd.Timestamp("2024-05-31 17:00", tz=TZ)
        stamps = pd.DatetimeIndex([cutoff, cutoff + pd.Timedelta(hours=8)])
        d = pd.DataFrame(dict(site_id=["synthetic", "synthetic"], ts=stamps, cosz=[.8, .8], cap=[100., 100.], energy=[50., 50.]))
        common = pd.DataFrame(dict(old=[1., 1.], keep=[2., 2.], new=[3., 3.]), index=stamps)
        a = pd.Series([50.])
        cfg_fold = dict(val_start=stamps[-1], val_end=stamps[-1])
        class Model:
            def predict(self, frame):
                return np.zeros(len(frame))
        calls = []
        def fit(cfg, frame, y, weight):
            calls.append(list(frame))
            return Model()
        with patch("src.tune_v3.evaluation_groups", return_value={cutoff: ["summer_support"]}), \
             patch("src.tune_v3.fold", return_value=cfg_fold), \
             patch("src.tune_v3.T.fold_features", return_value=d), \
             patch("src.tune_v3.T.select_features", return_value=["cosz", "cap"]), \
             patch("src.tune_v3.T.fit_model", side_effect=fit), \
             patch("src.tune_v3.evaluation_context", return_value=dict(a=a, b3=a, actual=None)), \
             patch("src.tune_v3.grouped_prediction", return_value=a), \
             patch("src.tune_v3.metrics", return_value=dict(change_pct=0.)), \
             patch("src.tune_v3.screen", return_value=dict(screen_pass=False)), \
             patch("src.tune_v3.save"):
            run_stage(d, None, None, pd.DataFrame(dict(w=[])), {}, Path("unused"), 1,
                      ["replacement"], variants={"replacement": {}}, added=common,
                      control_columns=["old", "keep"], candidate_columns=["new", "keep"])
        self.assertEqual(calls, [["cosz", "cap", "old", "keep"], ["cosz", "cap", "new", "keep"]])

    def test_early_rejection_skips_later_training_and_preserves_partial_records(self):
        cutoff = pd.Timestamp("2024-05-31 17:00", tz=TZ)
        stamps = pd.DatetimeIndex([cutoff, cutoff + pd.Timedelta(hours=8)])
        d = pd.DataFrame(dict(site_id=["synthetic"] * 2, ts=stamps, cosz=[.8] * 2,
                              cap=[100.] * 2, energy=[50.] * 2))
        a = pd.Series([50.])
        class Model:
            def predict(self, frame):
                return np.zeros(len(frame))
        result = {}
        groups = {cutoff: ["summer_support"], cutoff + pd.Timedelta(days=90): ["autumn_support"]}
        with patch("src.tune_v3.evaluation_groups", return_value=groups), \
             patch("src.tune_v3.fold", return_value=dict(val_start=stamps[-1], val_end=stamps[-1])), \
             patch("src.tune_v3.T.fold_features", return_value=d), \
             patch("src.tune_v3.T.select_features", return_value=["cosz", "cap"]), \
             patch("src.tune_v3.T.fit_model", return_value=Model()) as fit, \
             patch("src.tune_v3.evaluation_context", return_value=dict(a=a, b3=a, actual=None)), \
             patch("src.tune_v3.grouped_prediction", return_value=a), \
             patch("src.tune_v3.metrics", return_value=dict(change_pct=4.)), \
             patch("src.tune_v3.screen") as screen, patch("src.tune_v3.save"):
            passed = run_stage(d, None, None, pd.DataFrame(dict(w=[])), result, Path("unused"),
                               1, ["candidate"], variants={"candidate": {}}, early_reject=early_failure)
            self.assertEqual(fit.call_count, 2)
            # 결과 기록 직후 중단돼 기각 플래그가 없더라도 재개 시 다시 판단한다.
            del result["seeds1_early_rejections"]
            fit.reset_mock()
            run_stage(d, None, None, pd.DataFrame(dict(w=[])), result, Path("unused"),
                      1, ["candidate"], variants={"candidate": {}}, early_reject=early_failure)
        self.assertEqual(fit.call_count, 0)
        self.assertEqual(passed, [])
        self.assertEqual(list(result["seeds1"]["candidate"]), ["summer_support"])
        self.assertTrue(result["seeds1_verdicts"]["candidate"]["early_rejected"])
        screen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
