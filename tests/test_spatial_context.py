"""예보 시간 경계와 PCA 학습 시간 분리의 합성 검사."""
import unittest

import numpy as np
import pandas as pd

from src.spatial_context import encode_field, within_day_change, attach


class SpatialContextTest(unittest.TestCase):
    def test_temporal_change_does_not_cross_target_day(self):
        index=pd.date_range("2024-01-01 01:00",periods=48,freq="h",tz="Asia/Seoul")
        s=pd.Series(np.r_[np.ones(24),np.full(24,100.)],index=index)
        delta=within_day_change(s)
        self.assertTrue((delta.dropna()==0).all())
        self.assertTrue(np.isnan(delta.loc["2024-01-02 00:00"]))
        self.assertTrue(np.isnan(delta.loc["2024-01-02 01:00"]))

    def test_future_values_do_not_change_pca_fit(self):
        index=pd.date_range("2023-01-01 01:00",periods=90*24,freq="h",tz="Asia/Seoul")
        wide=pd.DataFrame(np.random.default_rng(42).normal(size=(len(index),5)),index=index)
        cutoff=index[60*24]
        changed=wide.copy()
        changed.loc[changed.index>cutoff,0]+=10000
        a,va=encode_field(wide,cutoff,3)
        b,vb=encode_field(changed,cutoff,3)
        np.testing.assert_allclose(a.loc[a.index<=cutoff],b.loc[b.index<=cutoff])
        np.testing.assert_allclose(va,vb)

    def test_context_is_joined_by_time(self):
        ts=pd.Timestamp("2024-01-01 12:00",tz="Asia/Seoul")
        q=pd.DataFrame(dict(ts=[ts,ts+pd.Timedelta(hours=1),ts]))
        extra=pd.DataFrame(dict(context_x=[3.,8.]),index=[ts,ts+pd.Timedelta(hours=1)])
        attach(q,extra)
        np.testing.assert_array_equal(q.context_x,[3.,8.,3.])


if __name__=="__main__":
    unittest.main()
