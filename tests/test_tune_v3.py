"""학습 재사용 범위·holdout 차단·후보 승격 규칙 검사."""
import copy
import unittest

from src.diagnose_season import FOLDS
from src.tune_v3 import evaluation_groups,screen


class TuneV3Test(unittest.TestCase):
    def records(self):
        return {n:{c:dict(base_penalty=100,direct_penalty=96,change_pct=-4.)
                   for c in ("vs_v3_3seeds","vs_matched_v3")}
                for n in FOLDS+("fold1_static",)}

    def test_shared_training_cutoff_and_holdout_rejection(self):
        groups=evaluation_groups()
        self.assertEqual(len(groups),4)
        self.assertTrue(any(set(names)=={"summer_support","fold1_static"} for names in groups.values()))
        with self.assertRaises(ValueError):
            evaluation_groups(("final_holdout",))

    def test_long_horizon_worsening_blocks_promotion(self):
        rows=self.records()
        self.assertTrue(screen(rows,True)["screen_pass"])
        rows["fold1_static"]["vs_v3_3seeds"]["change_pct"]=.01
        self.assertFalse(screen(rows,True)["screen_pass"])

    def test_bad_season_is_not_hidden_by_aggregate_gain(self):
        rows=self.records()
        for n in FOLDS:
            for c in rows[n]:
                rows[n][c].update(direct_penalty=92,change_pct=-8.)
        bad=copy.deepcopy(rows)
        bad[FOLDS[0]]["vs_v3_3seeds"].update(direct_penalty=104,change_pct=4.)
        self.assertLess(screen(bad,True)["vs_v3_3seeds_seasonal_change_pct"],-3)
        self.assertFalse(screen(bad,True)["screen_pass"])


if __name__=="__main__":
    unittest.main()
