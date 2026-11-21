"""발전소 고유 편향 분리와 과거 배율 적용의 합성 검사."""
import unittest

import numpy as np
import pandas as pd

from src.experiments.site_profile import isolate, apply_past
from src.train import clear_sky


class SiteProfileTest(unittest.TestCase):
    def panel(self):
        p=pd.DataFrame(dict(site_id=["a","b","c"],station=["s","s","s"],
             ts=[pd.Timestamp("2024-01-01 12:00",tz="Asia/Seoul")]*3,
             cap=[10.,20.,30.],cosz=[.6,.6,.6],forecast_kwh=[3.,6.,9.]))
        return p

    def test_common_error_removed_and_leave_one_out(self):
        p=self.panel()
        scale=p.cap*clear_sky(p.cosz)
        p["energy"]=p.forecast_kwh+.2*scale
        self.assertTrue(np.allclose(isolate(p).unique_error,0))
        p.loc[0,"energy"]+=1
        r=isolate(p)
        self.assertAlmostEqual(r.unique_error.iloc[0],1.)
        self.assertLess(r.unique_error.iloc[1],0)

    def test_unknown_site_unchanged_and_no_actual_dependency(self):
        p=self.panel()
        beta=pd.Series({"a":.1,"b":-.1})
        c=apply_past(p,beta)
        np.testing.assert_allclose(c.forecast_kwh,[3.3,5.4,9.])
        self.assertTrue((c.forecast_kwh>=0).all())


if __name__=="__main__":
    unittest.main()
