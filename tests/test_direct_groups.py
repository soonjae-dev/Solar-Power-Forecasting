"""집합 특징·관측 누락·시간 분리의 합성 입력 검사."""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.direct_groups import WEATHER, aggregate, fit_predict


class DirectGroupsTest(unittest.TestCase):
    def test_capacity_weighting_and_incomplete_group(self):
        ts=pd.Timestamp("2024-01-01 12:00",tz="Asia/Seoul")
        later=ts+pd.Timedelta(hours=1)
        sites=pd.DataFrame(dict(current_group=["A","A","B"],capacity_kw=[10.,30.,20.]),index=["a","b","c"])
        gen=pd.DataFrame(dict(site_id=["a","b","c","b","c"],ts=[ts,ts,ts,later,later],energy=[1.,3.,2.,3.,2.]))
        weather=gen[["site_id","ts"]].copy()
        for c in WEATHER:
            weather[c]=.5 if c=="cosz" else weather.site_id.map(dict(a=20.,b=80.,c=50.))
        g,X=aggregate(gen,weather,sites,sites.index,ts)
        self.assertAlmostEqual(X.loc[(ts,"A"),"wa_cloud"],65.)
        self.assertAlmostEqual(X.loc[(ts,"A"),"wa_cloud_std"],np.sqrt(675.))
        self.assertEqual(g.loc[(ts,"A"),"actual"],4.)
        self.assertNotIn((later,"A"),g.index)
        self.assertIn((later,"B"),g.index)

    def test_future_actual_is_not_used_for_training(self):
        dates=pd.date_range("2024-01-01 12:00",periods=62,freq="D",tz="Asia/Seoul")
        index=pd.MultiIndex.from_product([dates,["A","B"]],names=["ts","group"])
        g=pd.DataFrame(dict(actual=1.,scale=1.),index=index)
        g.loc[(dates[-1],slice(None)),"actual"]=999.
        X=pd.DataFrame(dict(group_id=np.tile([0,1],62),weather=1.),index=index)
        smp=pd.DataFrame(dict(w=100.),index=dates)
        class Spy:
            def __init__(self,**kwargs): pass
            def fit(self,x,y,**kwargs):
                assert len(x)==122 and (y==1).all()
                return self
            def predict(self,x): return np.ones(len(x))
        with patch("src.direct_groups.lgb.LGBMRegressor",Spy):
            pred,meta=fit_predict(g,X,dates[-2],dates[-1],dates[-1],smp)
        self.assertEqual(meta["train_days"],61)
        self.assertTrue((pred==1).all())


if __name__=="__main__":
    unittest.main()
