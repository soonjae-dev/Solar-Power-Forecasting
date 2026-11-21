"""집합 손실의 수치 미분·곡률 상계·상쇄·추가학습 검사."""
import unittest

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.experiments.group_objective import GroupObjective


class GroupObjectiveTest(unittest.TestCase):
    def setup_objective(self):
        scale=np.array([1.,2.,3.,1.])
        actual=np.array([.4,.6,2.,.3])
        obj=GroupObjective([0,0,1,1],scale,actual,[1.,1.,3.,3.])
        return obj,np.array([.5,.4,.7,.6]),actual/scale

    def test_gradient_matches_finite_difference(self):
        obj,p,y=self.setup_objective()
        grad,_=obj.derivatives(p,y)
        expected=[]
        for i in range(len(p)):
            step=np.zeros_like(p)
            step[i]=1e-6
            expected.append((obj.loss(p+step,y)-obj.loss(p-step,y))/2e-6)
        np.testing.assert_allclose(grad,expected,rtol=1e-6,atol=1e-8)

    def test_hessian_majorizer_is_upper_bound(self):
        obj,p,y=self.setup_objective()
        _,major=obj.derivatives(p,y)
        u=obj.residual(p)
        curve=(1+(u/obj.delta)**2)**-1.5
        exact=np.diag(obj.anchor*obj.s/obj.norm)
        for k in range(len(obj.S)):
            s=np.where(obj.groups==k,obj.s,0)
            exact+=obj.q[k]*curve[k]/obj.S[k]/obj.norm*np.outer(s,s)
        self.assertGreaterEqual(np.linalg.eigvalsh(np.diag(major)-exact).min(),-1e-10)

    def test_opposite_errors_cancel_only_within_group(self):
        p=np.array([2.,0.])
        y=np.ones(2)
        same=GroupObjective([0,0],[1.,1.],y,[1.,1.],anchor=0)
        different=GroupObjective([0,1],[1.,1.],y,[1.,1.],anchor=0)
        np.testing.assert_allclose(same.derivatives(p,y)[0],0)
        grad,_=different.derivatives(p,y)
        self.assertGreater(grad[0],0)
        self.assertLess(grad[1],0)

    def test_continuation_starts_from_original_predictions(self):
        rng=np.random.default_rng(7)
        X=pd.DataFrame(rng.normal(size=(100,2)),columns=["x","z"])
        y=np.asarray(.7+.1*X.x,dtype="float32")
        base=lgb.LGBMRegressor(n_estimators=4,num_leaves=4,min_child_samples=5,
                              n_jobs=1,verbosity=-1).fit(X,y)
        obj=GroupObjective(np.arange(100)//10,np.ones(100),y,np.ones(100))
        first=[]
        def spy(pred,data):
            if not first:
                first.append(pred.copy())
            return obj(pred,data)
        m=lgb.train(dict(objective=spy,num_leaves=4,min_data_in_leaf=5,
                         num_threads=1,verbosity=-1),lgb.Dataset(X,label=y),
                    num_boost_round=2,init_model=base.booster_)
        np.testing.assert_allclose(first[0],base.predict(X))
        self.assertEqual(m.num_trees(),6)
        self.assertEqual(base.booster_.num_trees(),4)
        self.assertLess(obj.loss(m.predict(X),y),obj.loss(base.predict(X),y))


if __name__=="__main__":
    unittest.main()
