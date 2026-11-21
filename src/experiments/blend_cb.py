"""v6·CatBoost(p53 설정) 발전소별 고정 0.5:0.5 결합. 3시드·집계만, holdout 제외.

    MPLCONFIGDIR=/tmp/oibc-mpl PYTHONWARNINGS=ignore caffeinate -i -s .venv/bin/python -m src.experiments.blend_cb

CatBoost 단독 페널티가 p53과 원 단위로 같아야 진행한다. 결합은 선형이라 집합합도 0.5·v6 + 0.5·cb다.
"""
import gc
import json
import time
import traceback

import numpy as np
import pandas as pd

from .. import train as T
from ..analog_residual import rejection
from ..cb_onehot import PARAMS, checked_input, make_model, seasonal_summary
from ..config import EXP_DIR
from ..data import load_generation, load_sites, load_smp
from ..direct_groups import metrics
from ..gfs_more import load_more
from ..gfs_v3 import checked_gfs
from ..profile_target import base_target
from ..provider_ensemble import grouped_prediction
from ..splits import fold
from ..tune_v3 import END, WINDOWS, evaluation_context, evaluation_groups, save
from ..v6_error_budget import load_reference
from ..weather_teacher import CFG

WEIGHT = .5
SPEC = dict(rule="14.57", params=PARAMS, seeds=[2026, 2027, 2028], features=144, cb_weight=WEIGHT)
PATH = EXP_DIR/"p61_blend_cb"/"metrics.json"
P53 = EXP_DIR/"p53_cb_onehot"/"metrics.json"
DROP = ("doy_sin", "doy_cos")
SPEC_NODOY = dict(rule="14.60", params=PARAMS, seeds=[2026, 2027, 2028], features=142, cb_weight=WEIGHT, dropped=list(DROP))
PATH_NODOY = EXP_DIR/"p64_blend_cb_nodoy"/"metrics.json"


def blend_metrics(a, base, cb, smp, expected_cb):
    """CatBoost 단독 재현을 확인하고 결합 집계를 낸다. 세 입력은 (ts, group) 집합합."""
    alone = metrics(a, base, cb, smp)
    if expected_cb is not None and alone["direct_penalty"] != expected_cb:
        raise ValueError("p53 CatBoost 재현 불일치")
    r = metrics(a, base, (1-WEIGHT)*base + WEIGHT*cb, smp)
    r["cb_alone_change_pct"] = alone["change_pct"]
    return r


def main(SPEC=SPEC, PATH=PATH, drop=()):
    PATH.parent.mkdir(parents=True, exist_ok=True)
    result = json.loads(PATH.read_text()) if PATH.exists() else dict(spec=SPEC, results={})
    if result["spec"] != SPEC:
        raise ValueError("계획 불일치")
    if result.get("complete"):
        print("완료 후보: 재학습하지 않음", json.dumps({k: result.get(k) for k in ("passed", "seasonal_change_pct", "early_rejection")}, ensure_ascii=False), flush=True)
        return
    p53 = json.loads(P53.read_text())["results"]
    baseline = json.loads((EXP_DIR/"p26_gfs_more"/"metrics.json").read_text())["seeds3"]["gfs60"]
    gen = load_generation()
    gen = gen.loc[gen.ts <= END]
    sites = load_sites().set_index("site_id")
    smp = load_smp("train").set_index("ts")
    d = T.load_train_features()
    features = T.select_features(d, CFG["feats"])
    d = d.loc[d.ts <= END, ["site_id", "ts", "energy", *features]].copy()
    T._cache.clear()
    common = pd.concat([checked_gfs()[0], load_more()[0]], axis=1)
    features = [c for c in features + list(common) if c not in drop]
    if len(features) != SPEC["features"]:
        raise ValueError("입력 수 불일치")
    for cutoff, names in evaluation_groups().items():
        if all(n in result["results"] for n in names):
            continue
        q = T.fold_features(d, CFG, cutoff)
        tr = q.loc[q.ts <= cutoff].copy()
        val = q.loc[q.ts.between(min(fold(n)["val_start"] for n in names),
                                max(fold(n)["val_end"] for n in names))].copy()
        del q
        if any(len(tr) != baseline[n]["train_rows"] for n in names) or val.ts.max() > END:
            raise ValueError("학습행/holdout 조건 위반")
        tr[list(common)] = common.reindex(tr.ts).to_numpy()
        val[list(common)] = common.reindex(val.ts).to_numpy()
        X, V = checked_input(tr[features]), checked_input(val[features])
        y = base_target(tr)
        weight = T.kt_weight(T.clear_sky(tr.cosz), T.sample_weight(tr, "cap", smp.w), CFG)
        pred = np.zeros(len(val))
        started = time.monotonic()
        print(f"마감{cutoff:%Y-%m-%d}: CatBoost 3시드 시작", flush=True)
        for seed in SPEC["seeds"]:
            model = make_model(seed)
            model.fit(X, y, sample_weight=weight, cat_features=["site"])
            pred += model.predict(V)/3
            del model
            gc.collect()
            print(f"seed={seed} 완료, 누적{time.monotonic()-started:.0f}초", flush=True)
        pred = np.maximum(pred, 0)*T.clear_sky(val.cosz)*val.cap.to_numpy()
        for name in names:
            ctx = evaluation_context(name, gen, sites)
            base = load_reference(name, ctx["actual"], sites).groupby(["ts", "group"]).forecast_kwh.sum().reindex(ctx["a"].index)
            f = fold(name)
            mask = val.ts.between(f["val_start"], f["val_end"]).to_numpy()
            cb = grouped_prediction(ctx["actual"], val.loc[mask, ["site_id", "ts"]], pred[mask]).reindex(ctx["a"].index)
            r = blend_metrics(ctx["a"], base, cb, smp, None if drop else p53[name]["direct_penalty"])
            if r["base_penalty"] != baseline[name]["vs_v3_3seeds"]["direct_penalty"]:
                raise ValueError("v6 대조 불일치")
            r.update(train_rows=len(tr), train_cutoff=cutoff.isoformat(), seconds=round(time.monotonic()-started, 1))
            result["results"][name] = r
            save(PATH, result)
            print(name, json.dumps(r, ensure_ascii=False), flush=True)
        reason = rejection(result["results"])
        if reason:
            result.update(complete=True, passed=False, early_rejection=reason,
                          uncompleted_windows=[n for n in WINDOWS if n not in result["results"]])
            if not result["uncompleted_windows"]:
                seasonal_summary(result)
            save(PATH, result)
            print("조기 기각:", reason, flush=True)
            return
        del X, V, tr, val, pred
        gc.collect()
    seasonal_summary(result)
    seasons = [r for n, r in result["results"].items() if n != "fold1_static"]
    change = result["seasonal_change_pct"]
    result.update(complete=True, seasonal_change_pct=round(change, 3),
                  passed=change <= -1 and sum(r["change_pct"] < 0 for r in seasons) >= 3
                  and max(r["change_pct"] for r in seasons) <= 3
                  and result["results"]["fold1_static"]["change_pct"] <= 0
                  and (not drop or result["results"]["selection"]["change_pct"] <= 0))
    save(PATH, result)
    print("최종", json.dumps({k: result[k] for k in ("passed", "seasonal_change_pct", "improved_seasons")}, ensure_ascii=False), flush=True)


def holdout():
    """개발 기준 통과 후 2025-05 1회 확인(14.57). v6는 p27_gfs_holdout_gfs60과 원 단위 재현을 요구한다."""
    from .gfs_holdout import NAME
    from ..gfs_pilot import load_gfs
    dev = json.loads(PATH.read_text())
    if not dev.get("passed"):
        raise ValueError("개발 기준 미통과: 2025-05 확인 안 함")
    if "final_holdout" in dev:
        print("이미 확인함", json.dumps(dev["final_holdout"], ensure_ascii=False), flush=True)
        return
    ref = json.loads((EXP_DIR/"p27_gfs_holdout_gfs60"/"metrics.json").read_text())["vs_v3_3seeds"]
    f = fold(NAME)
    gfs = pd.concat([load_gfs("2021-01-02", "2025-05-31")[0], load_more("2021-01-02", "2025-05-31")[0]], axis=1)
    sites = load_sites().set_index("site_id")
    smp = load_smp("train").set_index("ts")
    gen = load_generation()
    gen = gen[gen.ts <= f["val_end"]]
    d = T.load_train_features()
    d = d.loc[d.ts <= f["val_end"], ["site_id", "ts", "energy", *T.select_features(d, CFG["feats"])]].copy()
    T._cache.clear()
    q = T.fold_features(d, CFG, f["cutoff"])
    columns = T.select_features(q, CFG["feats"]) + list(gfs)
    tr = q[q.ts <= f["cutoff"]].copy()
    val = q[q.ts.between(f["val_start"], f["val_end"])].copy()
    del q, d
    if tr.ts.max() > f["cutoff"] or len(columns) != SPEC["features"]:
        raise ValueError("시점 분리/입력 수 위반")
    for frame in (tr, val):
        frame[list(gfs)] = gfs.reindex(frame.ts).to_numpy()
    y = base_target(tr)
    weight = T.kt_weight(T.clear_sky(tr.cosz), T.sample_weight(tr, "cap", smp.w), CFG)
    v6, cb = np.zeros(len(val)), np.zeros(len(val))
    X, V = checked_input(tr[columns]), checked_input(val[columns])
    for seed in SPEC["seeds"]:
        v6 += T.fit_model({**CFG, "seed": seed}, tr[columns], y, weight).predict(val[columns])/3
        model = make_model(seed)
        model.fit(X, y, sample_weight=weight, cat_features=["site"])
        cb += model.predict(V)/3
        del model
        gc.collect()
        print(f"seed={seed} 완료(v6·CatBoost)", flush=True)
    scale = T.clear_sky(val.cosz)*val.cap.to_numpy()
    ctx = evaluation_context(NAME, gen, sites)
    group = lambda p: grouped_prediction(ctx["actual"], val[["site_id", "ts"]], np.maximum(p, 0)*scale).reindex(ctx["a"].index)
    gv6, gcb = group(v6), group(cb)
    if metrics(ctx["a"], ctx["b3"], gv6, smp)["direct_penalty"] != ref["direct_penalty"]:
        raise ValueError("2025-05 v6 재현 불일치")
    r = metrics(ctx["a"], gv6, (1-WEIGHT)*gv6 + WEIGHT*gcb, smp)
    r.update(cb_alone_change_pct=metrics(ctx["a"], gv6, gcb, smp)["change_pct"], train_cutoff=f["cutoff"].isoformat(),
             train_rows=len(tr), submission_candidate=r["direct_penalty"] <= r["base_penalty"])
    dev["final_holdout"] = r
    save(PATH, dev)
    print("final_holdout", json.dumps(r, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    import sys
    try:
        if sys.argv[1:] == ["holdout"]:
            holdout()
        elif sys.argv[1:] == ["nodoy"]:
            main(SPEC_NODOY, PATH_NODOY, DROP)
        else:
            main()
    except Exception as exc:
        frames = [dict(function=f.name, line=f.lineno) for f in traceback.extract_tb(exc.__traceback__)
                  if "/src/" in f.filename]
        print(json.dumps(dict(status="failed", error_type=type(exc).__name__, code_frames=frames)), flush=True)
        raise SystemExit(1) from None
