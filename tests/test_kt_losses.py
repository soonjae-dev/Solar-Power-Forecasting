"""공통 실행기의 대조군 보존과 실제 손실 선택을 합성 입력으로 검사."""
import copy
import unittest

import numpy as np
import pandas as pd

from src import train as T
from src.experiments.kt_losses import LOSSES
from src.tune_v3 import candidate_config,VARIANTS
from src.weather_teacher import CFG


class KTLossTest(unittest.TestCase):
    def test_baseline_and_existing_variants_unchanged(self):
        before=copy.deepcopy(CFG)
        old=copy.deepcopy(VARIANTS)
        self.assertEqual(candidate_config("baseline",LOSSES),CFG)
        self.assertEqual(candidate_config("kt_l1",LOSSES)["objective"],"l1")
        self.assertEqual(candidate_config("min_child500",VARIANTS)["min_child_samples"],500)
        self.assertEqual(CFG,before)
        self.assertEqual(VARIANTS,old)
        with self.assertRaises(ValueError):
            candidate_config("unknown",LOSSES)

    def test_native_l1_uses_median_in_constant_prediction_problem(self):
        X=pd.DataFrame({"cosz":np.full(10,.5)})
        y=np.r_[np.zeros(9),1.]
        override=dict(n_estimators=1,num_leaves=2,min_child_samples=1)
        l2=T.fit_model({**candidate_config("baseline",LOSSES),**override},X,y,np.ones(10))
        l1=T.fit_model({**candidate_config("kt_l1",LOSSES),**override},X,y,np.ones(10))
        np.testing.assert_allclose(l2.predict(X),.1)
        np.testing.assert_allclose(l1.predict(X),0.)
        huber=T.make_model(candidate_config("kt_huber",LOSSES))
        self.assertEqual(huber.get_params()["objective"],"huber")
        self.assertEqual(huber.get_params()["alpha"],.1)


if __name__=="__main__":
    unittest.main()
