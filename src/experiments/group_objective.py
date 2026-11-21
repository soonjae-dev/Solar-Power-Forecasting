"""v3 발전소별 구조를 보존한 집합 손실 추가학습. 집계만 저장.

    .venv/bin/python -m src.experiments.group_objective
"""
import gc
import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from .. import train as T
from ..config import EXP_DIR, PRED_DIR, TZ
from ..data import load_generation, load_sites, load_smp
from ..diagnose_season import FOLDS, run_id
from ..direct_groups import metrics
from ..splits import fold
from ..weather_teacher import CFG

END=pd.Timestamp("2025-05-01",tz=TZ)
ROUNDS=150
DELTA=.05
ANCHOR=.1
CONTINUE=dict(learning_rate=.02,num_leaves=127,min_data_in_leaf=200,
              feature_fraction=.8,bagging_fraction=1.,bagging_freq=0,
              num_threads=4,verbosity=-1,seed=2026,deterministic=True,
              force_col_wise=True,metric="None")


class GroupObjective:
    """L = sum q_g*S_g*rho(E_g/S_g) + alpha/2*sum s_i*(z_i-y_i)^2.

    E_g=sum(s_i*z_i-actual_i), S_g=sum(s_i).
    true Hessian block = q_g*rho''/S_g * s*s.T.
    Cauchy-Schwarz gives this block <= diag(q_g*rho''*s).
    """
    def __init__(self,groups,scale,actual,weights,delta=DELTA,anchor=ANCHOR):
        self.groups=np.asarray(groups,dtype="int32")
        self.s=np.asarray(scale,dtype="float64")
        self.actual=np.asarray(actual,dtype="float64")
        self.delta,self.anchor=delta,anchor
        if not (len(self.groups)==len(self.s)==len(self.actual)==len(weights)):
            raise ValueError("집합 목적함수 행 수 불일치")
        if (self.s<=0).any() or not np.isfinite(self.s).all():
            raise ValueError("집합 목적함수 scale 이상")
        if not np.isfinite(self.actual).all() or (self.groups<0).any():
            raise ValueError("집합 목적함수 실적·그룹 코드 이상")
        self.S=np.bincount(self.groups,weights=self.s)
        if (self.S<=0).any():
            raise ValueError("빈 집합 코드")
        q=np.bincount(self.groups,weights=self.s*np.asarray(weights))/self.S
        if not np.isfinite(q).all() or (q<0).any() or not (q>0).any():
            raise ValueError("집합 가중치 이상")
        self.q=q/np.average(q,weights=self.S)
        self.norm=self.s.mean()

    def residual(self,pred):
        return np.bincount(self.groups,weights=self.s*pred-self.actual)/self.S

    def loss(self,pred,label):
        u=self.residual(pred)
        rho=self.delta**2*(np.sqrt(1+(u/self.delta)**2)-1)
        point=.5*self.anchor*np.sum(self.s*(pred-label)**2)
        return (np.sum(self.q*self.S*rho)+point)/self.norm

    def derivatives(self,pred,label):
        u=self.residual(pred)
        den=np.sqrt(1+(u/self.delta)**2)
        psi=u/den
        curvature=den**-3
        g=self.s*(self.q[self.groups]*psi[self.groups]+self.anchor*(pred-label))/self.norm
        h=self.s*(self.q[self.groups]*curvature[self.groups]+self.anchor)/self.norm
        if not np.isfinite(g).all() or not np.isfinite(h).all() or (h<=0).any():
            raise ValueError("집합 목적함수 미분값 이상")
        return g,h

    def __call__(self,pred,dataset):
        return self.derivatives(pred,dataset.get_label())


def to_groups(actual,val,pred,sites):
    p=val[["site_id","ts"]].copy()
    p["forecast_kwh"]=np.maximum(pred,0)*T.clear_sky(val.cosz)*val.cap.to_numpy()
    g=actual[["site_id","ts","group"]].merge(p,on=["site_id","ts"],how="left",validate="one_to_one")
    g.forecast_kwh=g.forecast_kwh.fillna(0)
    return g.groupby(["ts","group"]).forecast_kwh.sum()


def run(seeds=1):
    print(f"집합 손실 추가학습: {seeds}시드, 기존 v3·L2 추가학습과 비교",flush=True)
    sites=load_sites().set_index("site_id")
    smp=load_smp("train").set_index("ts")
    gen=load_generation()
    gen=gen[gen.ts<=END]
    d=T.load_train_features()
    cols=["site_id","ts","energy",*T.select_features(d,CFG["feats"])]
    d=d.loc[d.ts<=END,cols].copy()
    T._cache.clear()
    out=EXP_DIR/"p17_group_objective"
    out.mkdir(parents=True,exist_ok=True)
    path=out/"metrics.json"
    result=json.loads(path.read_text()) if seeds==3 else dict(generation_config=CFG,delta=DELTA,anchor=ANCHOR,continue_params=CONTINUE,rounds=ROUNDS,results={})
    stage=f"seeds{seeds}"
    result["results"][stage]={}
    for name in FOLDS+("fold1_static",):
        f=fold(name)
        q=T.fold_features(d,CFG,f["cutoff"])
        feats=T.select_features(q,CFG["feats"])
        tr=q[q.ts<=f["cutoff"]].copy()
        val=q[q.ts.between(f["val_start"],f["val_end"])].copy()
        del q
        cs=T.clear_sky(tr.cosz)
        scale=tr.cap.to_numpy()*cs
        label=((tr.energy/tr.cap).clip(0,1.2)/cs).clip(0,2).to_numpy()
        weight=T.kt_weight(cs,T.sample_weight(tr,"cap",smp.w),CFG)
        codes,_=pd.factorize(pd.MultiIndex.from_arrays([tr.ts,tr.site_id.map(sites.current_group)]),sort=False)
        objective=GroupObjective(codes,scale,tr.energy.to_numpy(),tr.ts.map(smp.w).to_numpy())
        key=pd.read_parquet(PRED_DIR/run_id("nbrc_asof",name)/"predictions.parquet",columns=["site_id","ts","forecast_kwh"])
        actual=key[["site_id","ts"]].merge(gen,on=["site_id","ts"],validate="one_to_one")
        if len(actual)!=len(key):
            raise ValueError("평가 실적 키 누락")
        actual["group"]=actual.site_id.map(sites.current_group)
        key["group"]=key.site_id.map(sites.current_group)
        a=actual.groupby(["ts","group"]).energy.sum()
        b3=key.groupby(["ts","group"]).forecast_kwh.sum().reindex(a.index)
        predictions={c:np.zeros(len(val)) for c in ("base","l2_continue","group_continue")}
        for k in range(seeds):
            cfg={**CFG,"seed":2026+k}
            base=T.fit_model(cfg,tr[feats],label,weight)
            predictions["base"]+=base.predict(val[feats])/seeds
            print(f"{name}: v3 seed={2026+k} 학습 완료",flush=True)
            for mode,obj in (("l2_continue","regression"),("group_continue",objective)):
                # 새 dataset에 objective가 필요한 가중치를 한 번만 반영한다.
                ds=lgb.Dataset(tr[feats],label=label,weight=weight if mode=="l2_continue" else None,
                               categorical_feature=["site"],free_raw_data=True)
                m=lgb.train({**CONTINUE,"objective":obj,"seed":2026+k},ds,num_boost_round=ROUNDS,
                            init_model=base.booster_,keep_training_booster=False)
                if m.num_trees()!=base.booster_.num_trees()+ROUNDS:
                    raise ValueError("추가학습 트리 수 불일치")
                predictions[mode]+=m.predict(val[feats])/seeds
                print(f"{name}: {mode} seed={2026+k} 완료",flush=True)
                del m,ds
                gc.collect()
            del base
        grouped={c:to_groups(actual,val,p,sites).reindex(a.index) for c,p in predictions.items()}
        r=dict(vs_matched_v3=metrics(a,grouped["base"],grouped["group_continue"],smp),
               vs_l2_continue=metrics(a,grouped["l2_continue"],grouped["group_continue"],smp),
               vs_v3_3seeds=metrics(a,b3,grouped["group_continue"],smp),
               l2_continue_vs_matched=metrics(a,grouped["base"],grouped["l2_continue"],smp),
               train_rows=len(tr),group_hours=len(objective.S))
        result["results"][stage][name]=r
        path.write_text(json.dumps(result,ensure_ascii=False,indent=2))
        print(f"{name}: 같은 트리 수 L2 대비 {r['vs_l2_continue']['change_pct']:+.3f}%, v3 3시드 대비 {r['vs_v3_3seeds']['change_pct']:+.3f}%",flush=True)
        del tr,val,objective,grouped,predictions
        gc.collect()
    records=result["results"][stage]
    total={}
    for c in ("vs_matched_v3","vs_l2_continue","vs_v3_3seeds"):
        rows=[records[n][c] for n in FOLDS]
        total[c]=round(100*(sum(r["direct_penalty"] for r in rows)/sum(r["base_penalty"] for r in rows)-1),3)
    rows=[records[n]["vs_v3_3seeds"] for n in FOLDS]
    passed=(total["vs_v3_3seeds"]<=-3 and sum(r["change_pct"]<0 for r in rows)>=3
            and max(r["change_pct"] for r in rows)<=3 and records["fold1_static"]["vs_v3_3seeds"]["change_pct"]<=0
            and total["vs_l2_continue"]<0 and records["fold1_static"]["vs_l2_continue"]["change_pct"]<0)
    result[stage+"_verdict"]=dict(seasonal_change_pct=total,screen_pass=bool(passed))
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print("판정",result[stage+"_verdict"],flush=True)
    if seeds==1 and passed:
        del d,gen
        gc.collect()
        run(seeds=3)


if __name__=="__main__":
    run()
