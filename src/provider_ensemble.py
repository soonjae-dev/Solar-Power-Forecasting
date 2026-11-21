"""제공사별 독립 모델의 고정 앙상블. 집계만 출력·저장.

    .venv/bin/python -m src.provider_ensemble
"""
import gc
import json

import numpy as np
import pandas as pd

from . import train as T
from .config import EXP_DIR, PRED_DIR, TZ
from .data import load_generation, load_sites, load_smp
from .diagnose_season import FOLDS, run_id
from .direct_groups import metrics
from .splits import fold
from .weather_teacher import CFG

END=pd.Timestamp("2025-05-01",tz=TZ)
STATIC={"elev","azim","cosz","hour","doy_sin","doy_cos","cap","site","lat","lon"}
WEIGHTS={"full":.8,"wa":.1,"wb":.1}


def provider_features(names,provider):
    if provider not in ("wa","wb"):
        raise ValueError("알 수 없는 제공사")
    return [c for c in names if c in STATIC or c.startswith(provider+"_")
            or (provider=="wb" and c=="uv_per_cosz")]


def blend(predictions,weights):
    if any(w<0 for w in weights.values()) or not np.isclose(sum(weights.values()),1):
        raise ValueError("혼합 가중치 이상")
    arrays=[np.asarray(predictions[c],dtype="float64") for c in weights]
    if any(a.shape!=arrays[0].shape for a in arrays):
        raise ValueError("앙상블 예측 행 수 불일치")
    if any(not np.isfinite(a).all() or (a<0).any() for a in arrays):
        raise ValueError("앙상블 예측값 이상")
    return sum(w*a for w,a in zip(weights.values(),arrays))


def grouped_prediction(actual,val,values):
    p=val[["site_id","ts"]].copy()
    p["forecast_kwh"]=values
    g=actual[["site_id","ts","group"]].merge(p,on=["site_id","ts"],how="left",validate="one_to_one")
    g.forecast_kwh=g.forecast_kwh.fillna(0)
    return g.groupby(["ts","group"]).forecast_kwh.sum()


def diversity(actual,forecasts,smp):
    a=actual.unstack("group")
    e={k:(v.unstack("group")-a).to_numpy() for k,v in forecasts.items()}
    day=np.logical_or(a.sum(axis=1).to_numpy()>0,forecasts["full"].unstack("group").sum(axis=1).to_numpy()>0)
    weights=np.repeat(smp.reindex(a.index).w.to_numpy()[day],a.shape[1])
    out={}
    for left,right in (("full","wa"),("full","wb"),("wa","wb")):
        x,y=e[left][day].ravel(),e[right][day].ravel()
        mx,my=np.average(x,weights=weights),np.average(y,weights=weights)
        denom=np.sqrt(np.average((x-mx)**2,weights=weights)*np.average((y-my)**2,weights=weights))
        corr=np.average((x-mx)*(y-my),weights=weights)/denom if denom else None
        out[left+"_"+right]=dict(weighted_error_correlation=round(float(corr),5) if corr is not None else None,
                                opposite_sign_weight_pct=round(float(100*np.average(x*y<0,weights=weights)),3))
    return out


def run(seeds=1):
    print(f"제공사별 독립 모델 앙상블 {seeds}시드, 혼합 비율 고정",flush=True)
    sites=load_sites().set_index("site_id")
    smp=load_smp("train").set_index("ts")
    gen=load_generation()
    gen=gen[gen.ts<=END]
    d=T.load_train_features()
    names=T.select_features(d,CFG["feats"])
    d=d.loc[d.ts<=END,["site_id","ts","energy",*names]].copy()
    T._cache.clear()
    out=EXP_DIR/"p18_provider_ensemble"
    out.mkdir(parents=True,exist_ok=True)
    path=out/"metrics.json"
    result=json.loads(path.read_text()) if seeds==3 else dict(generation_config=CFG,weights=WEIGHTS,results={})
    stage=f"seeds{seeds}"
    result["results"][stage]={}
    for name in FOLDS+("fold1_static",):
        f=fold(name)
        q=T.fold_features(d,CFG,f["cutoff"])
        names=T.select_features(q,CFG["feats"])
        features={"full":names,**{p:provider_features(names,p) for p in ("wa","wb")}}
        tr=q[q.ts<=f["cutoff"]].copy()
        val=q[q.ts.between(f["val_start"],f["val_end"])].copy()
        del q
        cs=T.clear_sky(tr.cosz)
        y=((tr.energy/tr.cap).clip(0,1.2)/cs).clip(0,2).to_numpy()
        weight=T.kt_weight(cs,T.sample_weight(tr,"cap",smp.w),CFG)
        scale=T.clear_sky(val.cosz)*val.cap.to_numpy()
        forecasts={c:np.zeros(len(val)) for c in features}
        for label,cols in features.items():
            for k in range(seeds):
                model=T.fit_model({**CFG,"seed":2026+k},tr[cols],y,weight)
                forecasts[label]+=np.maximum(model.predict(val[cols]),0)*scale/seeds
                print(f"{name}: {label} seed={2026+k} 완료",flush=True)
                del model
                gc.collect()
        forecasts["mixture"]=blend(forecasts,WEIGHTS)
        forecasts["experts_mean"]=blend(forecasts,{"wa":.5,"wb":.5})
        key=pd.read_parquet(PRED_DIR/run_id("nbrc_asof",name)/"predictions.parquet",columns=["site_id","ts","forecast_kwh"])
        actual=key[["site_id","ts"]].merge(gen,on=["site_id","ts"],validate="one_to_one")
        if len(actual)!=len(key):
            raise ValueError("평가 실적 키 누락")
        actual["group"]=actual.site_id.map(sites.current_group)
        key["group"]=key.site_id.map(sites.current_group)
        a=actual.groupby(["ts","group"]).energy.sum()
        b3=key.groupby(["ts","group"]).forecast_kwh.sum().reindex(a.index)
        grouped={c:grouped_prediction(actual,val,p).reindex(a.index) for c,p in forecasts.items()}
        r=dict(vs_matched_full=metrics(a,grouped["full"],grouped["mixture"],smp),
               vs_v3_3seeds=metrics(a,b3,grouped["mixture"],smp),
               diagnostics={c:metrics(a,b3,grouped[c],smp) for c in ("wa","wb","experts_mean")},
               diversity=diversity(a,grouped,smp),feature_counts={c:len(v) for c,v in features.items()},
               train_rows=len(tr),seeds=seeds)
        result["results"][stage][name]=r
        path.write_text(json.dumps(result,ensure_ascii=False,indent=2))
        print(f"{name}: 같은 시드 full 대비 {r['vs_matched_full']['change_pct']:+.3f}%, v3 3시드 대비 {r['vs_v3_3seeds']['change_pct']:+.3f}%",flush=True)
        del tr,val,forecasts,grouped,actual,key
        gc.collect()
    records=result["results"][stage]
    total={}
    for c in ("vs_matched_full","vs_v3_3seeds"):
        rows=[records[n][c] for n in FOLDS]
        total[c]=round(100*(sum(r["direct_penalty"] for r in rows)/sum(r["base_penalty"] for r in rows)-1),3)
    rows=[records[n]["vs_v3_3seeds"] for n in FOLDS]
    passed=(total["vs_v3_3seeds"]<=-3 and sum(r["change_pct"]<0 for r in rows)>=3 and max(r["change_pct"] for r in rows)<=3
            and records["fold1_static"]["vs_v3_3seeds"]["change_pct"]<=0 and total["vs_matched_full"]<0
            and records["fold1_static"]["vs_matched_full"]["change_pct"]<0)
    result[stage+"_verdict"]=dict(seasonal_change_pct=total,screen_pass=bool(passed))
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print("판정",result[stage+"_verdict"],flush=True)
    if seeds==1 and passed:
        del d,gen
        gc.collect()
        run(seeds=3)


if __name__=="__main__":
    run()
