"""기상 교사의 시간 분리와 발전소 재결합을 합성 입력으로 검사."""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.weather_teacher import attach, corrected, teacher_cutoff


class WeatherTeacherTest(unittest.TestCase):
    def test_future_observation_not_used(self):
        cutoff=pd.Timestamp("2024-05-31 17:00",tz="Asia/Seoul")
        ts=pd.DatetimeIndex([cutoff-pd.Timedelta(hours=5),cutoff+pd.Timedelta(hours=19)])
        idx=pd.MultiIndex.from_arrays([["x","x"],ts],names=["wa_station","ts"])
        X=pd.DataFrame(dict(wa_cloud=[10.,10.],station=pd.Categorical(["x","x"])),index=idx)
        obs=pd.DataFrame(dict(wa_cloud=[15.,99.]),index=idx)
        class Spy:
            def __init__(self,**kwargs): pass
            def fit(self,x,y,**kwargs):
                assert len(y)==1 and float(y.iloc[0])==5
                return self
            def predict(self,x): return np.full(len(x),5.)
        with patch("src.weather_teacher.lgb.LGBMRegressor",Spy):
            result=corrected(X,obs,"wa_cloud",cutoff,np.array([False,True]))
        self.assertEqual(result.tolist(),[15.])
        self.assertEqual(teacher_cutoff(pd.Timestamp("2024-06-01",tz="Asia/Seoul")),cutoff)

    def test_station_mapping_and_cold_start_missing(self):
        ts=pd.Timestamp("2024-01-01 12:00",tz="Asia/Seoul")
        sites=pd.DataFrame(dict(wa_station=["s1","s2"]),index=["a","b"])
        d=pd.DataFrame(dict(site_id=["b","a"],ts=[ts,ts],wa_cloud=[20.,30.]))
        values=pd.DataFrame(dict(teacher_wa_cloud=[40.,np.nan]),index=pd.MultiIndex.from_arrays([["s1","s2"],[ts,ts]],names=["wa_station","ts"]))
        attach(d,sites,"wa",values)
        self.assertTrue(np.isnan(d.teacher_wa_cloud.iloc[0]))
        self.assertEqual(d.teacher_wa_cloud.iloc[1],40.)
        self.assertEqual(d.teacher_wa_cloud_delta.iloc[1],10.)


if __name__=="__main__":
    unittest.main()
