"""p53 봄을 동일 재현하고 코호트 기여만 집계. 행/모델 저장 없음."""
import gc
import json
import traceback

import numpy as np
import pandas as pd

from .. import train as T
from ..cb_onehot import SPEC, checked_input, make_model
from ..config import EXP_DIR
from ..data import load_generation, load_sites, load_smp
from ..direct_groups import metrics
from ..gfs_more import load_more
from ..gfs_v3 import checked_gfs
from ..profile_target import base_target
from ..splits import fold
from ..tune_v3 import END, evaluation_context, save
from ..v6_error_budget import history_cohort, load_reference
from ..weather_teacher import CFG

PATH = EXP_DIR/"p54_cb_diagnostic"/"metrics.json"


def recent_state(d, cutoff):
    p = d.loc[d.ts.between(cutoff-pd.Timedelta(days=120), cutoff),
              ["site_id", "ts", "energy", "cap", "cosz"]].copy()
    p["kt"] = base_target(p)
    p["day"] = (p.ts-pd.Timedelta(hours=1)).dt.normalize()
    daily = p.groupby(["site_id", "day"]).kt.median().reset_index()
    split = (cutoff-pd.Timedelta(days=30)).normalize()
    a = daily.loc[daily.day > split].groupby("site_id").kt.agg(["median", "size"])
    b = daily.loc[daily.day <= split].groupby("site_id").kt.agg(["median", "size"])
    eligible = (a["size"] >= 20) & (b["size"].reindex(a.index) >= 20)
    ratio = a["median"]/b["median"].reindex(a.index).replace(0, np.nan)
    return pd.Series(np.where(ratio < .5, "recent_low", "not_recent_low"), index=a.index).where(eligible & ratio.notna(), "unknown")


def cloud_bin(values):
    return np.select([values.isna(), values <= 30, values <= 70],
                     ["unknown", "clear", "mixed"], default="cloudy")


def attribute(p, smp):
    p = p.copy()
    p["e_base"], p["e_cb"] = p.forecast_kwh-p.energy, p.cb-p.energy
    grouped = p.groupby(["ts", "group"])[["e_base", "e_cb"]].sum()
    index = pd.MultiIndex.from_frame(p[["ts", "group"]])
    w = p.ts.map(smp.w).to_numpy()
    p["contribution"] = .1*w*(p.e_cb.to_numpy()*np.sign(grouped.e_cb.reindex(index).to_numpy())
                                  -p.e_base.to_numpy()*np.sign(grouped.e_base.reindex(index).to_numpy()))
    true_delta = float((grouped.e_cb.abs()-grouped.e_base.abs()).mul(
        .1*smp.w.reindex(grouped.index.get_level_values("ts")).to_numpy()).sum())
    if not np.isclose(p.contribution.sum(), true_delta, atol=1e-5, rtol=1e-10):
        raise ValueError("기여도 합계 불일치")
    return p, true_delta


def summary(p, smp):
    n = int(p.site_id.nunique())
    if n < 5:
        return dict(site_count=n, suppressed=True)
    weight = p.ts.map(smp.w).to_numpy()
    denom = float(np.sum(p.cap*weight))
    actual = float(p.energy.sum())
    return dict(site_count=n, suppressed=False, rows=len(p),
                contribution_won=round(float(p.contribution.sum())),
                base_prediction_actual_ratio=round(float(p.forecast_kwh.sum()/actual), 4) if actual else None,
                cb_prediction_actual_ratio=round(float(p.cb.sum()/actual), 4) if actual else None,
                base_weighted_bias_per_capacity=round(float(np.sum(p.e_base*weight)/denom), 6),
                cb_weighted_bias_per_capacity=round(float(np.sum(p.e_cb*weight)/denom), 6))


def diagnose(p, smp):
    p, delta = attribute(p, smp)
    result = dict(penalty_change_won=round(delta), dimensions={})
    for axis in ("group", "age", "recent", "cloud"):
        result["dimensions"][axis] = {str(k): summary(v, smp) for k, v in p.groupby(axis)}
        if axis != "group":
            result["dimensions"][axis+"_by_group"] = {
                str(g)+":"+str(k): summary(v, smp) for (g, k), v in p.groupby(["group", axis])}
    # 집합 시각별 운량 구간은 실제 집합 절대오차로 집계한다.
    keys = ["ts", "group"]
    cloud_sum = (p.cloud_value*p.cap).groupby([p.ts, p.group]).sum()
    valid_cap = p.cap.where(p.cloud_value.notna()).groupby([p.ts, p.group]).sum()
    c = cloud_sum/valid_cap.replace(0, np.nan)
    group = p.groupby(keys)[["e_base", "e_cb"]].sum()
    group["regime"] = cloud_bin(c.reindex(group.index))
    w = .1*smp.w.reindex(group.index.get_level_values("ts")).to_numpy()
    group["base_loss"], group["cb_loss"] = w*group.e_base.abs(), w*group.e_cb.abs()
    result["group_weather"] = {}
    for k, v in group.groupby("regime"):
        a, b = float(v.base_loss.sum()), float(v.cb_loss.sum())
        result["group_weather"][str(k)] = dict(group_hours=len(v), base_penalty=round(a), cb_penalty=round(b),
                                              change_pct=round(100*(b/a-1), 3) if a else None)
    return result


def main():
    PATH.parent.mkdir(parents=True, exist_ok=True)
    if PATH.exists() and json.loads(PATH.read_text()).get("complete"):
        print("완료 진단: 재학습하지 않음", flush=True)
        return
    f = fold("selection")
    if f["val_end"] > END:
        raise ValueError("holdout 차단")
    old = json.loads((EXP_DIR/"p53_cb_onehot"/"metrics.json").read_text())
    if old["spec"] != SPEC:
        raise ValueError("p53 설정 불일치")
    gen = load_generation()
    gen = gen.loc[gen.ts <= END]
    sites = load_sites().set_index("site_id")
    smp = load_smp("train").set_index("ts")
    d = T.load_train_features()
    features = T.select_features(d, CFG["feats"])
    d = d.loc[d.ts <= END, ["site_id", "ts", "energy", *features]].copy()
    T._cache.clear()
    recent = recent_state(d, f["cutoff"])
    first = gen.loc[gen.ts <= f["cutoff"]].groupby("site_id").ts.min().reindex(sites.index)
    age = history_cohort(first, f["cutoff"])
    q = T.fold_features(d, CFG, f["cutoff"])
    tr = q.loc[q.ts <= f["cutoff"]].copy()
    val = q.loc[q.ts.between(f["val_start"], f["val_end"])].copy()
    del q, d
    common = pd.concat([checked_gfs()[0], load_more()[0]], axis=1)
    tr[list(common)] = common.reindex(tr.ts).to_numpy()
    val[list(common)] = common.reindex(val.ts).to_numpy()
    features += list(common)
    if len(features) != 144 or len(tr) != old["results"]["selection"]["train_rows"]:
        raise ValueError("재현 학습행/입력 불일치")
    X, V = checked_input(tr[features]), checked_input(val[features])
    y = base_target(tr)
    weight = T.kt_weight(T.clear_sky(tr.cosz), T.sample_weight(tr, "cap", smp.w), CFG)
    prediction = np.zeros(len(val))
    print("봄 진단 재현 시작: 동일3시드, 후보변경 없음", flush=True)
    for seed in SPEC["seeds"]:
        model = make_model(seed)
        model.fit(X, y, sample_weight=weight, cat_features=["site"])
        prediction += model.predict(V)/3
        del model
        gc.collect()
        print(f"재현 seed={seed} 완료", flush=True)
    val["cb"] = np.maximum(prediction, 0)*T.clear_sky(val.cosz)*val.cap.to_numpy()
    ctx = evaluation_context("selection", gen, sites)
    p = load_reference("selection", ctx["actual"], sites)
    p = p.merge(val[["site_id", "ts", "cb", "wa_cloud", "wb_cloud"]],
                on=["site_id", "ts"], how="left", validate="one_to_one")
    p["cb"] = p.cb.fillna(0.)
    group = p.groupby(["ts", "group"])[["energy", "forecast_kwh", "cb"]].sum()
    comparison = metrics(group.energy, group.forecast_kwh, group.cb, smp)
    for key in ("base_penalty", "direct_penalty"):
        if comparison[key] != old["results"]["selection"][key]:
            raise ValueError("p53 점수 재현 불일치")
    p["age"] = p.site_id.map(age).fillna("unseen")
    p["recent"] = p.site_id.map(recent).fillna("unknown")
    p["cloud_value"] = p[["wa_cloud", "wb_cloud"]].mean(axis=1)
    p["cloud"] = cloud_bin(p.cloud_value)
    result = dict(rule="14.41", spec=SPEC, complete=True, reproduces_p53=True,
                  comparison=comparison, diagnostics=diagnose(p, smp))
    save(PATH, result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        frames = [dict(function=f.name, line=f.lineno) for f in traceback.extract_tb(exc.__traceback__)
                  if "/src/" in f.filename]
        print(json.dumps(dict(status="failed", error_type=type(exc).__name__, code_frames=frames)), flush=True)
        raise SystemExit(1) from None
