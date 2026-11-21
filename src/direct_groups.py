"""참조 G1/G2 직접 예측. 고정 발전소 집합, 시간순 학습, 집계만 저장.

    MPLCONFIGDIR=/private/tmp/oibc-mpl .venv/bin/python -m src.direct_groups

안정 집합: 2023-06부터 해당 학습 마감까지 실적이 완전한 발전소 고정.
전체 집합: 253개 모두 관측된 시점 이후만 학습. 60일 미만이면 평가 생략.
2025-05 holdout, 제출 파일, 기존 v3는 변경하지 않는다.
"""
import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import CACHE_DIR, EXP_DIR, PRED_DIR, TZ
from .data import load_generation, load_sites, load_smp
from .diagnose_season import FOLDS, run_id
from .splits import fold
from .train import clear_sky

START = pd.Timestamp("2023-06-01 01:00", tz=TZ)
END = pd.Timestamp("2025-05-01 00:00", tz=TZ)
WEATHER = ["wa_cloud", "wb_cloud", "wb_uv_idx", "wa_humidity", "wb_rel_hum",
           "wa_temp", "wb_temp", "wa_rain", "wb_precip_prob", "cosz"]
PARAMS = dict(objective="regression_l1", n_estimators=400, learning_rate=.03,
              num_leaves=15, max_depth=4, min_child_samples=50, reg_lambda=10,
              n_jobs=4, random_state=2026, deterministic=True, force_col_wise=True,
              verbosity=-1)


def fixed_sites(gen, sites, cutoff, scope):
    """학습 기간의 관측 여부만으로 집합을 고른다. 평가 실적으로 고르지 않는다."""
    if scope == "all253":
        ids=sites.index
        first=gen.groupby("site_id")["ts"].min().reindex(ids)
        if first.isna().any() or first.max()>cutoff:
            return None, None
        start=first.max()
    else:
        start=START
        p=gen[gen.ts.between(start,cutoff)]
        expected=len(pd.date_range(start,cutoff,freq="h"))
        counts=p.groupby("site_id").size()
        ids=counts.index[counts==expected]
    if len(ids)==0:
        raise ValueError("고정 발전소 집합 없음")
    return ids,start


def aggregate(gen, weather, sites, ids, start):
    """집합별 완전 관측 시각만 사용. 평가 누락은 호출부에서 거부한다."""
    p=gen[gen.site_id.isin(ids)&gen.ts.between(start,END)].copy()
    if p.duplicated(["site_id","ts"]).any():
        raise ValueError("실적 중복 키")
    p["group"]=p.site_id.map(sites.current_group)
    p["cap"]=p.site_id.map(sites.capacity_kw)
    keys=["ts","group"]
    expected=sites.loc[ids].groupby("current_group").size()
    g=p.groupby(keys).agg(actual=("energy","sum"),count=("energy","size"),cap=("cap","sum"))
    complete=g["count"].to_numpy()==g.index.get_level_values("group").map(expected).to_numpy()
    g=g.loc[complete].copy()
    x=p[["site_id","ts","group","cap"]].merge(weather,on=["site_id","ts"],how="left",validate="one_to_one")
    X=pd.DataFrame(index=g.index)
    for c in WEATHER:
        denom=x["cap"].where(x[c].notna()).groupby([x.ts,x.group]).sum().replace(0,np.nan)
        mean=(x[c]*x.cap).groupby([x.ts,x.group]).sum()/denom
        X[c]=mean
        if c in ("wa_cloud","wb_cloud","wb_uv_idx"):
            second=(x[c]**2*x.cap).groupby([x.ts,x.group]).sum()/denom
            X[c+"_std"]=np.sqrt((second-mean**2).clip(lower=0))
    # 특징 캐시는 주간 행만 있다. 야간의 청천 합은 0으로 둔다.
    x["cs"]=np.where(x.cosz.notna(),clear_sky(x.cosz.fillna(0))*x.cap,0)
    g["scale"]=x.groupby(keys)["cs"].sum().reindex(g.index).fillna(0)
    X["cloud_disagreement"]=X.wa_cloud-X.wb_cloud
    ts=g.index.get_level_values("ts")
    X["hour"]=ts.hour
    X["doy_sin"]=np.sin(2*np.pi*ts.dayofyear/365.25)
    X["doy_cos"]=np.cos(2*np.pi*ts.dayofyear/365.25)
    X["group_id"]=(g.index.get_level_values("group")=="B").astype(int)
    return g,X


def fit_predict(g,X,cutoff,start,end,smp,seeds=1):
    ts=g.index.get_level_values("ts")
    tr=(ts<=cutoff)&(g.scale.to_numpy()>0)
    va=(ts>=start)&(ts<=end)
    day=va&(g.scale.to_numpy()>0)
    train_days=int((ts[tr]-pd.Timedelta(hours=1)).normalize().nunique())
    if train_days<60:
        return None,dict(train_days=train_days,reason="학습 60일 미만")
    y=g.actual.loc[tr].clip(lower=0)/g.scale.loc[tr]
    w=smp.reindex(ts[tr])["w"].to_numpy()*g.scale.loc[tr].to_numpy()
    if not np.isfinite(w).all():
        raise ValueError("학습 SMP 누락")
    pred=pd.Series(0.,index=g.index[va])
    if day.any():
        forecasts=[]
        for k in range(seeds):
            model=lgb.LGBMRegressor(**{**PARAMS,"random_state":2026+k})
            model.fit(X.loc[tr],y,sample_weight=w/w.mean(),categorical_feature=["group_id"])
            forecasts.append(model.predict(X.loc[day]))
        pred.loc[g.index[day]]=np.maximum(np.mean(forecasts,axis=0),0)*g.scale.loc[day]
    return pred,dict(train_days=train_days,train_group_hours=int(tr.sum()),seeds=seeds)


def baseline(name, ids, sites, gen):
    p=pd.read_parquet(PRED_DIR/run_id("nbrc_asof",name)/"predictions.parquet",columns=["site_id","ts","forecast_kwh"])
    p=p[p.site_id.isin(ids)].copy()
    f=fold(name)
    truth=gen[gen.site_id.isin(ids)&gen.ts.between(f["val_start"],f["val_end"])]
    if len(p)!=len(truth) or p.duplicated(["site_id","ts"]).any():
        raise ValueError("v3 비교 키 수 불일치")
    merged=p.merge(truth[["site_id","ts"]],on=["site_id","ts"],validate="one_to_one")
    if len(merged)!=len(p):
        raise ValueError("v3 비교 키 불일치")
    p["group"]=p.site_id.map(sites.current_group)
    return p.groupby(["ts","group"]).forecast_kwh.sum()


def metrics(actual,base,pred,smp):
    if not actual.index.equals(base.index) or not actual.index.equals(pred.index):
        raise ValueError("집합별 평가 키 불일치")
    a,b,c=[v.unstack("group").reindex(columns=["A","B"]) for v in (actual,base,pred)]
    if a.isna().any().any() or b.isna().any().any() or c.isna().any().any():
        raise ValueError("집합별 평가 누락")
    assert np.isfinite(c.to_numpy()).all() and (c.to_numpy()>=0).all()
    w=smp.reindex(a.index)["w"]
    errors=[(v-a).abs().mul(.1*w,axis=0) for v in (b,c)]
    p0,p1=[e.sum(axis=1) for e in errors]
    day=(a.index-pd.Timedelta(hours=1)).normalize()
    daily=pd.DataFrame(dict(base=p0,direct=p1)).groupby(day).sum()
    rng=np.random.default_rng(2026)
    changes=[]
    for _ in range(1000):
        starts=rng.integers(0,max(1,len(daily)-6),size=int(np.ceil(len(daily)/7)))
        idx=np.concatenate([np.arange(s,min(s+7,len(daily))) for s in starts])[:len(daily)]
        s=daily.iloc[idx].sum()
        changes.append(100*(s.direct/s.base-1))
    peak=a.index.hour.isin(range(11,16))
    return dict(base_penalty=round(float(p0.sum())),direct_penalty=round(float(p1.sum())),
                change_pct=round(float(100*(p1.sum()/p0.sum()-1)),3),
                change_ci90_pct=np.quantile(changes,[.05,.95]).round(3).tolist(),
                peak_change_pct=round(float(100*(p1[peak].sum()/p0[peak].sum()-1)),3),
                base_forecast_actual_ratio=round(float(b.to_numpy().sum()/a.to_numpy().sum()),4),
                direct_forecast_actual_ratio=round(float(c.to_numpy().sum()/a.to_numpy().sum()),4),
                group_change_pct={str(k):round(float(100*(errors[1][k].sum()/errors[0][k].sum()-1)),3) for k in a.columns})


def main():
    print("G1/G2 직접 예측: 고정 집합·1시드·holdout 제외",flush=True)
    gen=load_generation()
    gen=gen[gen.ts<=END]  # 5월 holdout은 집계·학습·평가에서 제외
    sites=load_sites().set_index("site_id")
    smp=load_smp("train").set_index("ts")
    weather=pd.read_parquet(CACHE_DIR/"train_features.parquet",columns=["site_id","ts",*WEATHER])
    weather=weather[weather.ts<=END]
    result=dict(params=PARAMS,results={})
    datasets={}
    for scope in ("stable","all253"):
        result["results"][scope]={}
        for name in FOLDS+("fold1_static",):
            f=fold(name)
            ids,start=fixed_sites(gen,sites,f["cutoff"],scope)
            if ids is None:
                result["results"][scope][name]=dict(skipped=True,reason="학습 시점에 253개소 관측 불가")
                print(f"{scope} {name}: 전체 관측 이전, 생략",flush=True)
                continue
            key=(scope,tuple(sorted(ids)),start)
            if key not in datasets:
                datasets[key]=aggregate(gen,weather,sites,ids,start)
            g,X=datasets[key]
            pred,meta=fit_predict(g,X,f["cutoff"],f["val_start"],f["val_end"],smp)
            if pred is None:
                result["results"][scope][name]=dict(skipped=True,**meta)
                print(f"{scope} {name}: 학습 {meta['train_days']}일, 생략",flush=True)
                continue
            expected=pd.MultiIndex.from_product([pd.date_range(f["val_start"],f["val_end"],freq="h"),["A","B"]],names=["ts","group"])
            if not pred.index.equals(expected):
                raise ValueError("고정 집합의 평가 실적 누락")
            b=baseline(name,ids,sites,gen).reindex(pred.index)
            r=metrics(g.actual.reindex(pred.index),b,pred,smp)
            r.update(meta)
            r.update(sites=len(ids),capacity_kw=round(float(sites.loc[ids].capacity_kw.sum()),1),hours=len(pred)//2)
            result["results"][scope][name]=r
            print(f"{scope} {name}: {len(ids)}개소, 페널티 변화 {r['change_pct']:+.3f}%",flush=True)
    stable=result["results"]["stable"]
    change=100*(sum(stable[n]["direct_penalty"] for n in FOLDS)/sum(stable[n]["base_penalty"] for n in FOLDS)-1)
    result["stable_seasonal_change_pct"]=round(change,3)
    result["stable_screen_pass"]=(change<=-3 and sum(stable[n]["change_pct"]<0 for n in FOLDS)>=3 and max(stable[n]["change_pct"] for n in FOLDS)<=3)
    result["note"]="안정 집합 진단은 전체253개 공식점수가 아님. 전체253은 학습 이력 제한으로 별도 비교. 미통과 시 시드 확대·제출 안 함."
    out=EXP_DIR/"p13_direct_groups"
    out.mkdir(parents=True,exist_ok=True)
    (out/"metrics.json").write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(f"안정 집합 계절 합계 {change:+.3f}%, 선택 기준 통과={result['stable_screen_pass']}",flush=True)


if __name__=="__main__":
    main()
