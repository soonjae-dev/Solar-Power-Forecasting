"""전체 보정 봄 악화 진단. 데이터 행 없이 집계만 저장·출력한다."""
import json

import numpy as np
import pandas as pd

from ..config import CACHE_DIR, EXP_DIR
from ..correct_total import HISTORY, WEATHER, allocate, table, train, evaluate
from ..data import load_generation, load_sites, load_smp
from ..diagnose_season import FOLDS


def diagnostics(t, model):
    b, a = t["base"], t["act"]
    c = allocate(b, model.predict(t["X"]) * t["cap"])
    actual = a.sum(axis=1)
    before, after = b.sum(axis=1) - actual, c.sum(axis=1) - actual
    change = c.sum(axis=1) - b.sum(axis=1)
    p0 = .1 * t["w"] * np.abs(b - a).sum(axis=1)
    p1 = .1 * t["w"] * np.abs(c - a).sum(axis=1)
    lb0, lb1 = .1 * t["w"] * np.abs(before), .1 * t["w"] * np.abs(after)
    day = b.sum(axis=1) > 0
    frame = pd.DataFrame(dict(actual=actual, before=before, after=after, change=change,
                              p0=p0, p1=p1), index=t["index"])
    daily = frame.groupby((frame.index - pd.Timedelta(hours=1)).normalize()).sum()
    assert np.isclose((p1-p0).sum(), (lb1-lb0).sum()+((p1-lb1)-(p0-lb0)).sum())
    assert np.isfinite(c).all() and (c>=0).all()

    def bucket(d):
        return dict(days=len(d), penalty_change=round(float((d.p1-d.p0).sum())),
                    correction_kwh=round(float(d.change.sum())),
                    original_signed_error_kwh=round(float(d.before.sum())),
                    corrected_signed_error_kwh=round(float(d.after.sum())))

    return dict(
        forecast_actual_ratio_before=round(float(b.sum()/a.sum()), 5),
        forecast_actual_ratio_after=round(float(c.sum()/a.sum()), 5),
        correction_energy_pct=round(float(100*change.sum()/b.sum()), 4),
        daylight_up_pct=round(float(100*np.mean(change[day] > 0)), 2),
        daylight_down_pct=round(float(100*np.mean(change[day] < 0)), 2),
        daylight_needed_up_pct=round(float(100*np.mean(before[day] < 0)), 2),
        wrong_direction_pct=round(float(100*np.mean((before[day]*change[day]) > 0)), 2),
        penalty_change=round(float((p1-p0).sum())),
        total_error_lower_bound_change=round(float((lb1-lb0).sum())),
        group_split_excess_change=round(float(((p1-lb1)-(p0-lb0)).sum())),
        by_baseline_daily_bias={"over":bucket(daily[daily.before>0]), "under":bucket(daily[daily.before<0])},
        days_improved=int((daily.p1<daily.p0).sum()), days_worsened=int((daily.p1>daily.p0).sum()),
        observed_capacity_kw=dict(min=round(float(t["cap"].min()),1),max=round(float(t["cap"].max()),1)))


def shift(past, target):
    X = pd.concat([t["X"].loc[t["base"].sum(axis=1)>0] for t in past])
    Y = target["X"].loc[target["base"].sum(axis=1)>0]
    out = {}
    for c in X:
        x,y=X[c].dropna(),Y[c].dropna()
        scale=x.std()
        lo,hi=x.quantile([.01,.99])
        out[c]=dict(train_mean=round(float(x.mean()),4),evaluation_mean=round(float(y.mean()),4),
                    mean_shift_train_sd=round(float((y.mean()-x.mean())/scale),3) if scale else None,
                    outside_train_1_99_pct=round(float(100*((y<lo)|(y>hi)).mean()),2))
    return out


def main():
    print("봄 악화 진단: 기존 보정 설정 고정, holdout 제외",flush=True)
    weather=pd.read_parquet(CACHE_DIR/"train_features.parquet",columns=["site_id","ts",*WEATHER])
    truth=load_generation()[["site_id","ts","energy"]]
    sites=load_sites().set_index("site_id")
    smp=load_smp("train").set_index("ts")
    tables={}
    for name in HISTORY+FOLDS+("fold1_static",):
        tables[name]=table(name,weather,truth,sites,smp)
        print(f"진단 집계 준비: {name}",flush=True)
    result={}
    for name in ("selection","fold1_static"):
        t=tables[name]
        eligible=[n for n in HISTORY+FOLDS if tables[n]["end"]<=t["cutoff"]]
        past=[tables[n] for n in eligible]
        model=train(past)
        r=dict(evaluation=diagnostics(t,model),feature_shift=shift(past,t),training_folds={})
        # 학습 구간 재적용은 방향 진단 전용. 성능 검증으로 사용하지 않는다.
        for n in eligible:
            r["training_folds"][n]=diagnostics(tables[n],model)
        # 모델 노후화와 달력 기간을 분리해 동일한 봄의 고정 모델도 비교한다.
        if name=="fold1_static":
            m=t["index"].isin(tables["selection"]["index"])
            spring={k:(v.loc[m] if isinstance(v,pd.DataFrame) else v[m] if isinstance(v,(np.ndarray,pd.DatetimeIndex)) else v) for k,v in t.items()}
            r["spring_subset"]=diagnostics(spring,model)
        # TreeSHAP의 평균 기여도(관측 용량 대비 보정량 %p), 인과 효과 아님.
        day=t["base"].sum(axis=1)>0
        contrib=model.predict(t["X"].loc[day],pred_contrib=True)
        r["correction_contributions_pp"]={str(c):dict(mean=round(float(contrib[:,i].mean()*100),4),
                                                      mean_abs=round(float(np.abs(contrib[:,i]).mean()*100),4))
                                               for i,c in enumerate(list(t["X"].columns)+["intercept"])}
        result[name]=r
        print(name,json.dumps(r["evaluation"],ensure_ascii=False),flush=True)
    # 사후 원인 진단용 제거 실험. 이 결과로 후보를 선정하거나 서버 제출하지 않는다.
    no_calendar={n:{**t,"X":t["X"].drop(columns=["doy_sin","doy_cos"])} for n,t in tables.items()}
    result["posthoc_no_calendar"]={}
    for name in FOLDS+("fold1_static",):
        t=no_calendar[name]
        eligible=[n for n in HISTORY+FOLDS if tables[n]["end"]<=t["cutoff"]]
        r=evaluate(t,train([no_calendar[n] for n in eligible]))
        result["posthoc_no_calendar"][name]=r
        print(f"사후 계절특징 제거 {name}: {r['change_pct']:+.3f}%",flush=True)
    out=EXP_DIR/"p12_correction_diagnosis"
    out.mkdir(parents=True,exist_ok=True)
    (out/"metrics.json").write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print("집계 진단 저장 완료",flush=True)


if __name__=="__main__":
    main()
