"""공간·시간 예보 패턴의 추가 정보 진단. 데이터 행·PCA 계수 저장 없음.

    .venv/bin/python -m src.spatial_context
"""
import gc
import json

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from . import train as T
from .config import CACHE_DIR, EXP_DIR, PRED_DIR, TZ
from .correct_total import HISTORY, WEATHER, PARAMS, allocate, table, train, evaluate
from .data import load_forecast, load_generation, load_sites, load_smp
from .diagnose_season import FOLDS, run_id
from .direct_groups import metrics
from .splits import fold
from .weather_teacher import CFG

END=pd.Timestamp("2025-05-01",tz=TZ)
SIGNALS=(("wa","wa_cloud",3),("wb","wb_cloud",3),("wb","wb_uv_idx",2))


def within_day_change(x):
    """시간축이 1시간 간격임을 보장한 뒤 같은 대상일 안에서만 변화 계산."""
    if len(x.index)>1 and not ((x.index[1:]-x.index[:-1])==pd.Timedelta(hours=1)).all():
        raise ValueError("공간 특징의 시간축 누락")
    day=(x.index-pd.Timedelta(hours=1)).normalize()
    return x.groupby(day).shift(-2)-x.groupby(day).shift(2)


def load_fields():
    fields={}
    for p in ("wa","wb"):
        columns=[c for provider,c,_ in SIGNALS if provider==p]
        st=p+"_station"
        f=load_forecast("train",p,columns,end=END)
        f=f[(f.ts==f.ts.dt.floor("h"))&(f.ts<=END)]
        for c in columns:
            frames={r:f[f["round"]==r].pivot(index="ts",columns=st,values=c) for r in (10,17)}
            wide=frames[17].combine_first(frames[10]).sort_index()
            grid=pd.date_range(wide.index.min(),END,freq="h")
            fields[c]=wide.reindex(grid).astype("float32")
    return fields


def encode_field(wide,cutoff,components):
    """미래 예보로 평균·분산·PCA를 적합하지 않는다."""
    mean=wide.mean(axis=1)
    centered=wide.sub(mean,axis=0).fillna(0)
    fit=(wide.index<=cutoff)&wide.index.hour.isin(range(6,20))&(wide.notna().mean(axis=1)>=.8)
    past=centered.loc[fit]
    if len(past)<365:
        raise ValueError("PCA 학습 예보 이력 부족")
    mu=past.mean(axis=0)
    scale=past.std(axis=0).replace(0,1).fillna(1)
    z=(centered-mu)/scale
    model=PCA(n_components=components,svd_solver="full")
    model.fit(z.loc[fit])
    score=pd.DataFrame(model.transform(z),index=wide.index,columns=range(components)).astype("float32")
    assert np.isfinite(score.to_numpy()).all()
    return score,model.explained_variance_ratio_.round(5).tolist()


def context(fields,cutoff):
    controls,extra=pd.DataFrame(),pd.DataFrame()
    variance={}
    for _,c,k in SIGNALS:
        wide=fields[c]
        mean=wide.mean(axis=1)
        controls[c+"_field_mean"]=mean
        controls[c+"_field_change"]=within_day_change(mean)
        pc,variance[c]=encode_field(wide,cutoff,k)
        for i in range(k):
            extra[f"context_{c}_pc{i+1}"]=pc[i]
        for i in range(2):
            extra[f"context_{c}_pc{i+1}_change"]=within_day_change(pc[i])
    return controls.astype("float32"),extra.astype("float32"),variance


def extended(t,controls,extra=None):
    frames=[t["X"],controls.reindex(t["index"])]
    if extra is not None:
        frames.append(extra.reindex(t["index"]))
    return {**t,"X":pd.concat(frames,axis=1)}


def residual_mae(t,model):
    day=t["base"].sum(axis=1)>0
    loss=np.abs(t["y"][day]-model.predict(t["X"].loc[day]))
    return float(np.average(loss,weights=t["w"][day]*t["cap"][day]))


def paired_comparison(t,a,b,smp):
    actual=pd.DataFrame(t["act"],index=t["index"],columns=["A","B"]).rename_axis("group",axis=1).stack()
    forecasts=[]
    for model,val in (a,b):
        corrected=allocate(t["base"],model.predict(val["X"])*t["cap"])
        forecasts.append(pd.DataFrame(corrected,index=t["index"],columns=["A","B"]).rename_axis("group",axis=1).stack())
    return metrics(actual,forecasts[0],forecasts[1],smp)


def main():
    print("공간14개 특징의 추가 설명력 진단, 시간순·holdout 제외",flush=True)
    fields=load_fields()
    weather=pd.read_parquet(CACHE_DIR/"train_features.parquet",columns=["site_id","ts",*WEATHER])
    weather=weather[weather.ts<=END]
    gen=load_generation()
    gen=gen[gen.ts<=END]
    sites=load_sites().set_index("site_id")
    smp=load_smp("train").set_index("ts")
    tables={n:table(n,weather,gen,sites,smp) for n in HISTORY+FOLDS+("fold1_static",)}
    result=dict(residual_params=PARAMS,results={})
    out=EXP_DIR/"p16_spatial_context"
    out.mkdir(parents=True,exist_ok=True)
    for name in FOLDS+("fold1_static",):
        t=tables[name]
        eligible=[n for n in HISTORY+FOLDS if tables[n]["end"]<=t["cutoff"]]
        controls,extra,variance=context(fields,t["cutoff"])
        row=dict(pca_variance=variance,calibration_folds=eligible)
        models={}
        for label,added in (("controls",None),("spatial",extra)):
            past=[extended(tables[n],controls,added) for n in eligible]
            val=extended(t,controls,added)
            model=train(past)
            row[label]=evaluate(val,model)
            row[label]["weighted_residual_mae"]=round(residual_mae(val,model),7)
            models[label]=(model,val)
        row["incremental_comparison"]=paired_comparison(t,models["controls"],models["spatial"],smp)
        row["incremental_penalty_change_pct"]=row["incremental_comparison"]["change_pct"]
        row["incremental_residual_mae_change_pct"]=round(100*(row["spatial"]["weighted_residual_mae"]/row["controls"]["weighted_residual_mae"]-1),3)
        result["results"][name]=row
        (out/"metrics.json").write_text(json.dumps(result,ensure_ascii=False,indent=2))
        print(f"{name}: 대조군 대비 {row['incremental_penalty_change_pct']:+.3f}%, v3 대비 {row['spatial']['change_pct']:+.3f}%",flush=True)
    rows=[result["results"][n] for n in FOLDS]
    change=100*(sum(r["spatial"]["corrected_penalty"] for r in rows)/sum(r["controls"]["corrected_penalty"] for r in rows)-1)
    static=result["results"]["fold1_static"]["incremental_penalty_change_pct"]
    passed=(change<=-1 and sum(r["incremental_penalty_change_pct"]<0 for r in rows)>=3 and static<=-.5 and max(r["spatial"]["change_pct"] for r in rows)<=3)
    result["verdict"]=dict(incremental_seasonal_change_pct=round(change,3),proceed_generation_trial=bool(passed))
    (out/"metrics.json").write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print("판정",json.dumps(result["verdict"],ensure_ascii=False),flush=True)
    if passed:
        del tables,weather,gen,past,val,models
        gc.collect()
        generation_trial(fields)


def attach(q,extra):
    for c in extra:
        q[c]=q.ts.map(extra[c]).astype("float32")


def generation_trial(fields,seeds=1):
    """진단 통과 시 같은 v3 구조에 공간 특징만 추가한다."""
    sites=load_sites().set_index("site_id")
    smp=load_smp("train").set_index("ts")
    gen=load_generation()
    gen=gen[gen.ts<=END]
    d=T.load_train_features()
    needed=["site_id","ts","energy",*T.select_features(d,CFG["feats"])]
    d=d.loc[d.ts<=END,needed].copy()
    T._cache.clear()
    path=EXP_DIR/"p16_spatial_context"/"metrics.json"
    result=json.loads(path.read_text())
    stage=f"generation_seeds{seeds}"
    result[stage]={}
    for name in FOLDS+("fold1_static",):
        f=fold(name)
        _,extra,variance=context(fields,f["cutoff"])
        q=T.fold_features(d,CFG,f["cutoff"])
        feats=T.select_features(q,CFG["feats"])
        tr=q[q.ts<=f["cutoff"]].copy()
        val=q[q.ts.between(f["val_start"],f["val_end"])].copy()
        del q
        attach(tr,extra)
        attach(val,extra)
        cs=T.clear_sky(tr.cosz)
        y=((tr.energy/tr.cap).clip(0,1.2)/cs).clip(0,2)
        sw=T.kt_weight(cs,T.sample_weight(tr,"cap",smp.w),CFG)
        scale=T.clear_sky(val.cosz)*val.cap.to_numpy()
        key=pd.read_parquet(PRED_DIR/run_id("nbrc_asof",name)/"predictions.parquet",columns=["site_id","ts","forecast_kwh"])
        actual=key[["site_id","ts"]].merge(gen,on=["site_id","ts"],validate="one_to_one")
        if len(actual)!=len(key):
            raise ValueError("평가 실적 키 누락")
        actual["group"]=actual.site_id.map(sites.current_group)
        key["group"]=key.site_id.map(sites.current_group)
        a=actual.groupby(["ts","group"]).energy.sum()
        b3=key.groupby(["ts","group"]).forecast_kwh.sum().reindex(a.index)
        forecasts={}
        labels=(("baseline",feats),("context",feats+list(extra))) if seeds==1 else (("context",feats+list(extra)),)
        for label,columns in labels:
            ratio=np.zeros(len(val))
            for k in range(seeds):
                m=T.fit_model({**CFG,"seed":2026+k},tr[columns],y,sw)
                ratio+=m.predict(val[columns])/seeds
            p=val[["site_id","ts"]].copy()
            p["forecast_kwh"]=np.maximum(ratio,0)*scale
            merged=actual[["site_id","ts","group"]].merge(p,on=["site_id","ts"],how="left",validate="one_to_one")
            merged.forecast_kwh=merged.forecast_kwh.fillna(0)
            forecasts[label]=merged.groupby(["ts","group"]).forecast_kwh.sum().reindex(a.index)
            print(f"발전량 {name} {label} {seeds}시드 완료",flush=True)
        ref=forecasts["baseline"] if seeds==1 else b3
        r=dict(matched_comparison=metrics(a,ref,forecasts["context"],smp),
               vs_v3_3seeds=metrics(a,b3,forecasts["context"],smp),pca_variance=variance,
               train_rows=len(tr),base_features=len(feats),context_features=len(extra))
        result[stage][name]=r
        path.write_text(json.dumps(result,ensure_ascii=False,indent=2))
        del tr,val,m,actual,key
        gc.collect()
    rows=[result[stage][n]["vs_v3_3seeds"] for n in FOLDS]
    change=100*(sum(r["direct_penalty"] for r in rows)/sum(r["base_penalty"] for r in rows)-1)
    passed=(change<=-3 and sum(r["change_pct"]<0 for r in rows)>=3 and max(r["change_pct"] for r in rows)<=3 and result[stage]["fold1_static"]["vs_v3_3seeds"]["change_pct"]<=0)
    result[stage+"_verdict"]=dict(seasonal_change_pct=round(change,3),screen_pass=bool(passed))
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    if seeds==1 and passed:
        del d,gen
        gc.collect()
        generation_trial(fields,seeds=3)


if __name__=="__main__":
    main()
