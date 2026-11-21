"""예보→실측 기상 교사 학습. 시간순 OOF 특징을 발전량 모델에 추가.

    .venv/bin/python -m src.weather_teacher [fold ...]

기본: 네 계절 및 고정 모델, 1시드. 데이터 행·모델·예측 파일 저장 없음.
"""
import gc
import json
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import train as T
from .config import EXP_DIR, PRED_DIR, TRAIN_DIR, TZ
from .data import load_forecast, load_generation, load_sites, load_smp
from .diagnose_season import FOLDS, run_id
from .direct_groups import metrics
from .splits import fold

END=pd.Timestamp("2025-05-01",tz=TZ)
TARGETS={"wa":["wa_cloud"],"wb":["wb_cloud","wb_uv_idx"]}
CFG={**T.DEFAULTS,"feats":"fullc","weight":"cap","target":"kt",
     "train_start":"2021-01-01","nbrc_asof":"fold"}
TEACHER=dict(n_estimators=200,learning_rate=.05,num_leaves=31,min_child_samples=100,
             reg_lambda=10,n_jobs=4,verbosity=-1,random_state=2026,
             deterministic=True,force_col_wise=True)


def teacher_cutoff(start):
    return start-pd.Timedelta(hours=7)


def station_data(provider,sites):
    st=f"{provider}_station"
    f=load_forecast("train",provider)
    f=f[(f.ts<=END)&(f.ts==f.ts.dt.floor("h"))]
    cols=[c for c in f if c not in (st,"ts","round")]
    r17=f[f["round"]==17].set_index([st,"ts"])[cols]
    r10=f[f["round"]==10].set_index([st,"ts"])[cols]
    if not r17.index.is_unique or not r10.index.is_unique:
        raise ValueError("예보 관측소 키 중복")
    X=r17.combine_first(r10).sort_index()
    for c in TARGETS[provider]:
        X[c+"_r10"]=r10[c].reindex(X.index).fillna(X[c])
        X[c+"_rdiff"]=X[c]-X[c+"_r10"]
    ts=X.index.get_level_values("ts")
    X["hour"]=ts.hour
    X["doy_sin"]=np.sin(2*np.pi*ts.dayofyear/365.25)
    X["doy_cos"]=np.cos(2*np.pi*ts.dayofyear/365.25)
    X["station"]=pd.Categorical(X.index.get_level_values(0))
    obs=pd.read_parquet(TRAIN_DIR/f"observed_{provider}.parquet",columns=[st,"ts",*TARGETS[provider]])
    obs["ts"]=pd.to_datetime(obs.ts).dt.tz_convert(TZ)
    obs=obs[(obs.ts<=END)&(obs.ts==obs.ts.dt.floor("h"))]
    obs=obs.drop_duplicates([st,"ts"]).set_index([st,"ts"])
    for c in TARGETS[provider]:
        hi=100 if c.endswith("cloud") else 30
        obs[c]=obs[c].where(obs[c].between(0,hi))
    return X,obs.reindex(X.index)


def corrected(X,obs,target,cutoff,predict_mask):
    """전일 17시까지의 관측만 학습. 대상일 이후 실측은 참조하지 않는다."""
    ts=X.index.get_level_values("ts")
    tr=(ts<=cutoff)&ts.hour.isin(range(6,20))&obs[target].notna()&X[target].notna()
    if (ts[tr]>cutoff).any() or (ts[predict_mask]<=cutoff).any():
        raise ValueError("기상 교사 시간 분리 위반")
    m=lgb.LGBMRegressor(**TEACHER)
    m.fit(X.loc[tr],obs.loc[tr,target]-X.loc[tr,target],categorical_feature=["station"])
    raw=X.loc[predict_mask,target].to_numpy()
    hi=100 if target.endswith("cloud") else 30
    return np.clip(raw+m.predict(X.loc[predict_mask]),0,hi).astype("float32")


def make_oof(provider,X,obs):
    ts=X.index.get_level_values("ts")
    out=pd.DataFrame(index=X.index)
    # 첫 반년은 교사 이력이 없어 결측으로 발전량 모델에 전달한다.
    boundaries=pd.date_range("2021-07-01","2025-07-01",freq="6MS",tz=TZ)
    for c in TARGETS[provider]:
        out["teacher_"+c]=np.nan
    for start,stop in zip(boundaries[:-1],boundaries[1:]):
        if start>=END:
            break
        mask=(ts>=start+pd.Timedelta(hours=1))&(ts<=min(stop,END))
        if not mask.any():
            continue
        for c in TARGETS[provider]:
            out.loc[mask,"teacher_"+c]=corrected(X,obs,c,teacher_cutoff(start),mask)
        print(f"교사 OOF {provider}: {start.year}-{start.month:02d}, {int(mask.sum()):,} 관측소·시각",flush=True)
    return out.astype("float32")


def attach(d,sites,provider,values):
    st=d.site_id.map(sites[f"{provider}_station"])
    keys=pd.MultiIndex.from_arrays([st,d.ts],names=[f"{provider}_station","ts"])
    for c in values:
        d[c]=values[c].reindex(keys).to_numpy(dtype="float32")
        original=c.removeprefix("teacher_")
        d[c+"_delta"]=d[c]-d[original]


def weather_scores(X,obs,values,start,end):
    ts=X.index.get_level_values("ts")
    mask=(ts>=start)&(ts<=end)&ts.hour.isin(range(6,20))
    out={}
    for name in values:
        c=name.removeprefix("teacher_")
        raw=X[c].reindex(values.index)
        actual=obs[c].reindex(values.index)
        valid=mask&raw.notna()&actual.notna()&values[name].notna()
        a=(raw[valid]-actual[valid]).to_numpy()
        b=(values.loc[valid,name]-actual[valid]).to_numpy()
        out[c]=dict(count=int(valid.sum()),raw_rmse=round(float(np.sqrt(np.mean(a*a))),4),
                    corrected_rmse=round(float(np.sqrt(np.mean(b*b))),4))
    return out


def evaluate_fold(name,d,sites,station,oof,gen,smp):
    f=fold(name)
    if f["val_end"]>END:
        raise ValueError("5월 holdout 금지")
    # 평가 시점 이후 실측은 교사 입력에 결합하지 않는다.
    q=T.fold_features(d,CFG,f["cutoff"])
    feats=T.select_features(q,CFG["feats"])
    train=q[q.ts<=f["cutoff"]].copy()
    val=q[q.ts.between(f["val_start"],f["val_end"])].copy()
    del q
    if any(np.isinf(part[c].to_numpy()).any() for part in (train,val) for c in feats):
        raise ValueError("발전량 입력에 무한대 포함")
    weather={}
    for p,(X,obs) in station.items():
        attach(train,sites,p,oof[p])
        ts=X.index.get_level_values("ts")
        mask=(ts>=f["val_start"])&(ts<=f["val_end"])
        values=pd.DataFrame(index=X.index)
        for c in TARGETS[p]:
            values["teacher_"+c]=np.nan
            values.loc[mask,"teacher_"+c]=corrected(X,obs,c,f["cutoff"],mask)
        attach(val,sites,p,values)
        weather.update(weather_scores(X,obs,values,f["val_start"],f["val_end"]))
    extra=[c for c in train if c.startswith("teacher_")]
    cs=T.clear_sky(train.cosz)
    y=((train.energy/train.cap).clip(0,1.2)/cs).clip(0,2.)
    w=T.kt_weight(cs,T.sample_weight(train,"cap",smp["w"]),CFG)
    scale=T.clear_sky(val.cosz)*val.cap.to_numpy()
    key=pd.read_parquet(PRED_DIR/run_id("nbrc_asof",name)/"predictions.parquet",columns=["site_id","ts","forecast_kwh"])
    actual=key[["site_id","ts"]].merge(gen,on=["site_id","ts"],validate="one_to_one")
    if len(actual)!=len(key):
        raise ValueError("평가 실적 키 누락")
    actual["group"]=actual.site_id.map(sites.current_group)
    a=actual.groupby(["ts","group"]).energy.sum()
    key["group"]=key.site_id.map(sites.current_group)
    b3=key.groupby(["ts","group"]).forecast_kwh.sum().reindex(a.index)
    forecasts={}
    for label,columns in (("matched_v3_seed1",feats),("teacher_seed1",feats+extra)):
        model=T.fit_model(CFG,train[columns],y,w)
        pred=val[["site_id","ts"]].copy()
        pred["forecast_kwh"]=np.maximum(model.predict(val[columns]),0)*scale
        merged=actual[["site_id","ts","group"]].merge(pred,on=["site_id","ts"],how="left",validate="one_to_one")
        merged.forecast_kwh=merged.forecast_kwh.fillna(0)
        forecasts[label]=merged.groupby(["ts","group"]).forecast_kwh.sum().reindex(a.index)
        print(f"{name}: {label} 학습·예측 완료",flush=True)
        del model
    return dict(weather=weather,train_rows=len(train),feature_count=len(feats),teacher_features=len(extra),
                teacher_initial_missing_pct=round(float(100*train[extra[0]].isna().mean()),3),
                matched_comparison=metrics(a,forecasts["matched_v3_seed1"],forecasts["teacher_seed1"],smp),
                vs_v3_3seeds=metrics(a,b3,forecasts["teacher_seed1"],smp))


def audit_neighbors():
    """기존 이웃 평균 계산 경고가 실제 비정상 값인지 집계로 확인한다."""
    from .features import nbrc_table
    counts=dict(values=0,infinite=0,missing=0)
    cutoffs=sorted({fold(n)["cutoff"] for n in FOLDS+("fold1_static",)})
    for cutoff in cutoffs:
        for p in TARGETS:
            values=nbrc_table(p,cutoff+pd.Timedelta(hours=1))
            a=values.loc[values.index.get_level_values("ts")<=END].to_numpy()
            counts["values"]+=int(a.size)
            counts["infinite"]+=int(np.isinf(a).sum())
            counts["missing"]+=int(np.isnan(a).sum())
    path=EXP_DIR/"p14_weather_teacher"/"metrics.json"
    result=json.loads(path.read_text())
    result["neighbor_numeric_audit"]=counts
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print("이웃 평균 유한성 검사",counts,flush=True)
    if counts["infinite"]:
        raise ValueError("이웃 평균 비정상 값 확인 필요")


def main(names):
    if any(n not in FOLDS+("fold1_static",) for n in names):
        raise ValueError("허용되지 않은 평가 구간")
    print("예보→실측 기상 교사 실험, 반년 단위 시간순 OOF, 1시드",flush=True)
    sites=load_sites().set_index("site_id")
    station={p:station_data(p,sites) for p in TARGETS}
    oof={p:make_oof(p,*station[p]) for p in TARGETS}
    d=T.load_train_features()
    # 실험에 쓰지 않는 캐시 파생 열을 복제하지 않아 메모리를 줄인다.
    needed=["site_id","ts","energy",*T.select_features(d,CFG["feats"])]
    d=d.loc[d.ts<=END,needed].copy()
    T._cache.clear()
    gen=load_generation()
    gen=gen[gen.ts<=END]
    smp=load_smp("train").set_index("ts")
    out=EXP_DIR/"p14_weather_teacher"
    out.mkdir(parents=True,exist_ok=True)
    result=dict(teacher_params=TEACHER,generation_config=CFG,results={})
    for name in names:
        r=evaluate_fold(name,d,sites,station,oof,gen,smp)
        result["results"][name]=r
        (out/"metrics.json").write_text(json.dumps(result,ensure_ascii=False,indent=2))
        print(f"{name}: 추가 특징 효과 {r['matched_comparison']['change_pct']:+.3f}%, v3 3시드 대비 {r['vs_v3_3seeds']['change_pct']:+.3f}%",flush=True)
        gc.collect()
    if all(n in result["results"] for n in FOLDS):
        summary={}
        for comparison in ("matched_comparison","vs_v3_3seeds"):
            rows=[result["results"][n][comparison] for n in FOLDS]
            change=100*(sum(r["direct_penalty"] for r in rows)/sum(r["base_penalty"] for r in rows)-1)
            summary[comparison]=dict(seasonal_change_pct=round(change,3),
                screen_pass=change<=-3 and sum(r["change_pct"]<0 for r in rows)>=3 and max(r["change_pct"] for r in rows)<=3)
        result["summary"]=summary
        (out/"metrics.json").write_text(json.dumps(result,ensure_ascii=False,indent=2))
        print("계절 합계",json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=="__main__":
    if sys.argv[1:]==["audit-neighbors"]:
        audit_neighbors()
    else:
        main(tuple(sys.argv[1:]) or FOLDS+("fold1_static",))
