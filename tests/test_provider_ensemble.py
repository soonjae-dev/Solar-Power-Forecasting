"""제공사 정보 분리·고정 혼합·평가키 연결의 합성 검사."""
import unittest

import numpy as np
import pandas as pd

from src.provider_ensemble import provider_features,blend,grouped_prediction


class ProviderEnsembleTest(unittest.TestCase):
    def test_features_do_not_leak_other_provider(self):
        names=["site","cap","cosz","wa_cloud","wa_cloud_nbrc","wa_dist",
               "wb_cloud","wb_uv_idx","wb_dist","uv_per_cosz",
               "cloud_mean","cloud_gap","day_cloud_gap","rain_gap"]
        a=provider_features(names,"wa")
        b=provider_features(names,"wb")
        self.assertEqual(a,["site","cap","cosz","wa_cloud","wa_cloud_nbrc","wa_dist"])
        self.assertEqual(b,["site","cap","cosz","wb_cloud","wb_uv_idx","wb_dist","uv_per_cosz"])

    def test_fixed_convex_blend_and_input_validation(self):
        p=dict(full=np.array([0.,10.]),wa=np.array([0.,20.]),wb=np.array([0.,5.]))
        np.testing.assert_allclose(blend(p,dict(full=.8,wa=.1,wb=.1)),[0.,10.5])
        np.testing.assert_allclose(blend(p,dict(wa=.5,wb=.5)),[0.,12.5])
        with self.assertRaises(ValueError):
            blend(p,dict(full=.9,wa=.2))
        with self.assertRaises(ValueError):
            blend(dict(full=np.array([-1.])),dict(full=1.))

    def test_forecasts_join_by_site_and_time(self):
        noon=pd.Timestamp("2024-01-01 12:00",tz="Asia/Seoul")
        night=noon+pd.Timedelta(hours=12)
        actual=pd.DataFrame(dict(site_id=["a","b","a","b"],ts=[noon,noon,night,night],group=["A","B","A","B"]))
        val=pd.DataFrame(dict(site_id=["b","a"],ts=[noon,noon]))
        p=grouped_prediction(actual,val,np.array([7.,3.]))
        self.assertEqual(p.loc[(noon,"A")],3.)
        self.assertEqual(p.loc[(noon,"B")],7.)
        self.assertEqual(p.loc[(night,"A")],0.)
        self.assertEqual(p.loc[(night,"B")],0.)


if __name__=="__main__":
    unittest.main()
