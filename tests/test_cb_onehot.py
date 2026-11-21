"""외부 데이터 없는 범주 처리/설정 검사."""
import unittest
import numpy as np
import pandas as pd
from src.cb_onehot import PARAMS, checked_input, make_model


class OneHotTest(unittest.TestCase):
    def test_native_onehot_and_no_file_writing(self):
        p = make_model(2026).get_params()
        self.assertEqual(p["one_hot_max_size"], 255)
        self.assertFalse(p["allow_writing_files"])
        self.assertEqual(p["iterations"], 600)

    def test_category_is_integral_and_extra_categories_rejected(self):
        x = pd.DataFrame(dict(site=pd.Categorical([1, 2, 1]), a=[1., np.nan, 2.]))
        y = checked_input(x)
        self.assertEqual(str(y.site.dtype), "int64")
        self.assertTrue(np.isnan(y.a.iloc[1]))
        x["extra"] = pd.Categorical([1, 2, 1])
        with self.assertRaises(ValueError):
            checked_input(x)

    def test_cardinality_guard(self):
        with self.assertRaises(ValueError):
            checked_input(pd.DataFrame(dict(site=np.arange(256))))

    def test_actual_native_fit_with_missing_numeric(self):
        x = checked_input(pd.DataFrame(dict(site=[1, 2]*20, a=[1., np.nan]*20)))
        m = make_model(2026)
        m.set_params(iterations=3)
        m.fit(x, np.array([.3, .7]*20), cat_features=["site"])
        self.assertTrue(np.isfinite(m.predict(x)).all())


if __name__ == "__main__":
    unittest.main()
