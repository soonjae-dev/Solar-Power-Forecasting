"""식별열/정답 차단과 기존모델 설정 보존. 합성 입력만 사용."""
import unittest
import numpy as np
import pandas as pd
from src.experiments.cb_nosite import SPEC, numeric_input
from src.cb_onehot import PARAMS, SPEC as ORIGINAL, make_model


class SharedTest(unittest.TestCase):
    def test_existing_spec_preserved(self):
        self.assertEqual(ORIGINAL["features"], 144)
        self.assertEqual(SPEC["features"], 143)
        self.assertEqual(SPEC["params"], PARAMS)
        self.assertEqual(ORIGINAL["seeds"], SPEC["seeds"])

    def test_identifier_and_label_rejected(self):
        for column in ("site", "site_id", "energy"):
            with self.assertRaises(ValueError):
                numeric_input(pd.DataFrame({column: [1, 2], "cosz": [.5, .8]}))

    def test_categorical_rejected_numeric_missing_preserved(self):
        with self.assertRaises(ValueError):
            numeric_input(pd.DataFrame(dict(a=pd.Categorical([1, 2]))))
        frame = pd.DataFrame(dict(a=[1., np.nan], b=[1, 2]))
        pd.testing.assert_frame_equal(numeric_input(frame), frame)

    def test_actual_fit_without_site(self):
        frame = numeric_input(pd.DataFrame(dict(a=[1., np.nan]*20, b=[1, 2]*20)))
        model = make_model(2026)
        model.set_params(iterations=3)
        model.fit(frame, [.3, .7]*20, cat_features=[])
        self.assertEqual(model.get_cat_feature_indices(), [])
        self.assertTrue(np.isfinite(model.predict(frame)).all())


if __name__ == "__main__":
    unittest.main()
