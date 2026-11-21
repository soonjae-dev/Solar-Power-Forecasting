"""현재 v3의 사전 고정 학습 설정 3개 비교. 데이터 행 저장·출력 없음.

    .venv/bin/python -m src.tune_v3

동일 마감의 여름/고정 검증은 한 번 학습해 평가한다. 집계 체크포인트 재개 가능.
"""
import gc
import json
import time

import numpy as np
import pandas as pd

from . import train as T
from .config import CACHE_DIR, EXP_DIR, PRED_DIR, TZ
from .data import load_generation, load_sites, load_smp
from .diagnose_season import FOLDS, run_id
from .direct_groups import metrics
from .provider_ensemble import grouped_prediction
from .splits import fold
from .weather_teacher import CFG

END=pd.Timestamp("2025-05-01",tz=TZ)
WINDOWS=FOLDS+("fold1_static",)
VARIANTS={"min_child500":{"min_child_samples":500},
          "leaves255":{"num_leaves":255},
          "slow1200":{"learning_rate":.03,"n_estimators":1200}}


def evaluation_groups(names=WINDOWS):
    groups={}
    for name in names:
        if name not in WINDOWS:
            raise ValueError("허용되지 않은 검증 구간")
        f=fold(name)
        if f["val_end"]>END:
            raise ValueError("5월 holdout 사용 금지")
        groups.setdefault(f["cutoff"],[]).append(name)
    return groups


def seasonal_change(records,comparison):
    rows=[records[n][comparison] for n in FOLDS]
    return float(100*(sum(r["direct_penalty"] for r in rows)/sum(r["base_penalty"] for r in rows)-1))


def screen(records,matched_required):
    change=seasonal_change(records,"vs_v3_3seeds")
    rows=[records[n]["vs_v3_3seeds"] for n in FOLDS]
    passed=(change<=-3 and sum(r["change_pct"]<0 for r in rows)>=3
            and max(r["change_pct"] for r in rows)<=3
            and records["fold1_static"]["vs_v3_3seeds"]["change_pct"]<=0)
    summary=dict(vs_v3_3seeds_seasonal_change_pct=round(change,3))
    if matched_required:
        matched=seasonal_change(records,"vs_matched_v3")
        passed=passed and matched<0 and records["fold1_static"]["vs_matched_v3"]["change_pct"]<0
        summary["vs_matched_v3_seasonal_change_pct"]=round(matched,3)
    summary["screen_pass"]=bool(passed)
    return summary


def save(path,result):
    temp=path.with_suffix(".tmp")
    temp.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    temp.replace(path)


def evaluation_context(name,gen,sites):
    key=pd.read_parquet(PRED_DIR/run_id("nbrc_asof",name)/"predictions.parquet",columns=["site_id","ts","forecast_kwh"])
    actual=key[["site_id","ts"]].merge(gen,on=["site_id","ts"],validate="one_to_one")
    if len(actual)!=len(key):
        raise ValueError("평가 실적 키 누락")
    actual["group"]=actual.site_id.map(sites.current_group)
    key["group"]=key.site_id.map(sites.current_group)
    a=actual.groupby(["ts","group"]).energy.sum()
    b3=key.groupby(["ts","group"]).forecast_kwh.sum().reindex(a.index)
    return dict(actual=actual,a=a,b3=b3)


def candidate_config(name,variants):
    if name=="baseline":
        return dict(CFG)
    if name not in variants:
        raise ValueError("등록되지 않은 후보 설정")
    return {**CFG,**variants[name]}


def run_stage(d,gen,sites,smp,result,path,seeds,candidates,variants=None,added=None,control_columns=(),candidate_columns=None,early_reject=None):
    """added는 시각별 공통 열(또는 학습 마감별 생성 함수)이다.

    대조군에는 control_columns만 넣는다. 후보에는 기본적으로 added 전체를
    넣고, candidate_columns가 있으면 그 열만 선택해 기존 열 교체를 지원한다.
    """
    variants=VARIANTS if variants is None else variants
    stage=f"seeds{seeds}"
    result.setdefault(stage,{})
    rejected=result.setdefault(stage+"_early_rejections",{}) if early_reject is not None else {}
    for c in candidates:
        result[stage].setdefault(c,{})
        if early_reject is not None and c not in rejected:
            reason=early_reject(result[stage][c])
            if reason:
                rejected[c]=reason
                save(path,result)
    for cutoff,names in evaluation_groups().items():
        remaining=[c for c in candidates if c not in rejected and not all(n in result[stage][c] for n in names)]
        if not remaining:
            print(f"학습 마감 {cutoff:%Y-%m-%d}: 완료/기각 체크포인트 사용",flush=True)
            continue
        q=T.fold_features(d,CFG,cutoff)
        feats=T.select_features(q,CFG["feats"])
        tr=q[q.ts<=cutoff].copy()
        first=min(fold(n)["val_start"] for n in names)
        last=max(fold(n)["val_end"] for n in names)
        val=q[q.ts.between(first,last)].copy()
        del q
        if tr.ts.max()>cutoff or val.ts.max()>END:
            raise ValueError("학습·평가 시점 분리 위반")
        common=added(cutoff) if callable(added) else added
        if common is not None:
            tr[list(common)]=common.reindex(tr.ts).to_numpy()
            val[list(common)]=common.reindex(val.ts).to_numpy()
        cs=T.clear_sky(tr.cosz)
        y=((tr.energy/tr.cap).clip(0,1.2)/cs).clip(0,2).to_numpy()
        weight=T.kt_weight(cs,T.sample_weight(tr,"cap",smp.w),CFG)
        scale=T.clear_sky(val.cosz)*val.cap.to_numpy()
        contexts={n:evaluation_context(n,gen,sites) for n in names}
        masks={n:val.ts.between(fold(n)["val_start"],fold(n)["val_end"]).to_numpy() for n in names}
        baseline={}
        baseline_rmse=None
        baseline_mae=None
        models=["baseline",*remaining] if seeds==1 else remaining
        for c in models:
            cfg=candidate_config(c,variants)
            columns=feats+(list(control_columns) if c=="baseline" else
                           list(candidate_columns) if candidate_columns is not None else
                           list(common) if common is not None else [])
            pred=np.zeros(len(val))
            train_pred=np.zeros(len(tr))
            started=time.perf_counter()
            for k in range(seeds):
                m=T.fit_model({**cfg,"seed":2026+k},tr[columns],y,weight)
                pred+=m.predict(val[columns])/seeds
                train_pred+=m.predict(tr[columns])/seeds
                print(f"마감 {cutoff:%Y-%m-%d}: {c} seed={2026+k} 학습·예측 완료",flush=True)
                del m
                gc.collect()
            seconds=time.perf_counter()-started
            pred=np.maximum(pred,0)*scale
            train_rmse=float(np.sqrt(np.average((train_pred-y)**2,weights=weight)))
            train_mae=float(np.average(np.abs(train_pred-y),weights=weight))
            if c=="baseline":
                baseline_rmse=train_rmse
                baseline_mae=train_mae
            for n,ctx in contexts.items():
                mask=masks[n]
                g=grouped_prediction(ctx["actual"],val.loc[mask,["site_id","ts"]],pred[mask]).reindex(ctx["a"].index)
                if c=="baseline":
                    baseline[n]=g
                    continue
                r=dict(config=cfg,seeds=seeds,train_cutoff=cutoff.isoformat(),
                       validation_start=fold(n)["val_start"].isoformat(),validation_end=fold(n)["val_end"].isoformat(),
                       train_rows=len(tr),train_features=len(columns),
                       weighted_train_rmse_kt=round(train_rmse,6),training_predict_seconds=round(seconds,2),
                       weighted_train_mae_kt=round(train_mae,6),
                       vs_v3_3seeds=metrics(ctx["a"],ctx["b3"],g,smp))
                if seeds==1:
                    r["baseline_train_rmse_kt"]=round(baseline_rmse,6)
                    r["baseline_train_mae_kt"]=round(baseline_mae,6)
                    r["vs_matched_v3"]=metrics(ctx["a"],baseline[n],g,smp)
                result[stage][c][n]=r
                save(path,result)
                print(f"{c} {n}: v3 3시드 대비 {r['vs_v3_3seeds']['change_pct']:+.3f}%",flush=True)
            if c != "baseline" and early_reject is not None:
                reason=early_reject(result[stage][c])
                if reason:
                    rejected[c]=reason
                    save(path,result)
                    print(f"{c}: 사전 필수조건 위반으로 조기 기각 ({reason})",flush=True)
            del pred,train_pred
            gc.collect()
        del tr,val,contexts,baseline
        gc.collect()
    verdicts={c:dict(screen_pass=False,early_rejected=True,reason=rejected[c]) if c in rejected
              else screen(result[stage][c],matched_required=seeds==1) for c in candidates}
    result[stage+"_verdicts"]=verdicts
    save(path,result)
    print(stage,"판정",json.dumps(verdicts,ensure_ascii=False),flush=True)
    return [c for c,v in verdicts.items() if v["screen_pass"]]


def main():
    print("현재v3의 사전 고정 설정3개 비교, 동일 마감 모델 재사용, holdout 제외",flush=True)
    sites=load_sites().set_index("site_id")
    smp=load_smp("train").set_index("ts")
    gen=load_generation()
    gen=gen[gen.ts<=END]
    d=T.load_train_features()
    cols=["site_id","ts","energy",*T.select_features(d,CFG["feats"])]
    d=d.loc[d.ts<=END,cols].copy()
    T._cache.clear()
    out=EXP_DIR/"p19_tune_v3"
    out.mkdir(parents=True,exist_ok=True)
    path=out/"metrics.json"
    if path.exists():
        result=json.loads(path.read_text())
        if result["baseline_config"]!=CFG or result["variants"]!=VARIANTS or result.get("cache_name")!=CACHE_DIR.name:
            raise ValueError("체크포인트의 설정이 현재 계획과 다름")
    else:
        result=dict(baseline_config=CFG,variants=VARIANTS,cache_name=CACHE_DIR.name)
    eligible=run_stage(d,gen,sites,smp,result,path,1,list(VARIANTS))
    if eligible:
        finalists=run_stage(d,gen,sites,smp,result,path,3,eligible)
        result["selected_candidate"]=min(finalists,key=lambda c:result["seeds3"][c]["fold1_static"]["vs_v3_3seeds"]["direct_penalty"]) if finalists else None
    else:
        result["selected_candidate"]=None
    save(path,result)
    print("최종 후보",result["selected_candidate"],flush=True)


if __name__=="__main__":
    main()
