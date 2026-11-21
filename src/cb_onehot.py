"""CTR 없는 site 범주 CatBoost 대안. 3시드·집계만, holdout 제외."""
import gc
import json
import time
import traceback

from catboost import CatBoostRegressor
import numpy as np
import pandas as pd

from . import train as T
from .analog_residual import rejection
from .config import EXP_DIR
from .data import load_generation, load_sites, load_smp
from .direct_groups import metrics
from .gfs_more import load_more
from .gfs_v3 import checked_gfs
from .profile_target import base_target
from .provider_ensemble import grouped_prediction
from .splits import fold
from .tune_v3 import END, WINDOWS, evaluation_context, evaluation_groups, save
from .v6_error_budget import load_reference
from .weather_teacher import CFG

PARAMS = dict(iterations=600, learning_rate=.06, depth=8, loss_function="RMSE",
              thread_count=4, verbose=False, allow_writing_files=False,
              one_hot_max_size=255)
SPEC = dict(rule="14.40", params=PARAMS, seeds=[2026, 2027, 2028], features=144)
PATH = EXP_DIR/"p53_cb_onehot"/"metrics.json"


def checked_input(frame):
    if "site" not in frame or frame.site.isna().any() or frame.site.nunique() > 255:
        raise ValueError("범주 조건 위반")
    if any(isinstance(frame[c].dtype, pd.CategoricalDtype) for c in frame if c != "site"):
        raise ValueError("추가 범주 입력 거부")
    x = frame.copy()
    x["site"] = x.site.astype("int64")
    return x


def make_model(seed):
    return CatBoostRegressor(**PARAMS, random_seed=seed)


def numeric_input(frame):
    if any(c in frame for c in ("site", "site_id", "energy")):
        raise ValueError("식별/정답 입력 거부")
    if any(not pd.api.types.is_numeric_dtype(frame[c].dtype) for c in frame):
        raise ValueError("비수치 입력 거부")
    return frame.copy()


def run_candidate(spec=SPEC, path=PATH, drop_site=False):
    """기본 p53을 보존하고 사전 고정된 식별열 제거 후보를 같은 절차로 검증한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    result = json.loads(path.read_text()) if path.exists() else dict(spec=spec, results={})
    if result["spec"] != spec or drop_site != (spec["features"] == 143):
        raise ValueError("계획 불일치")
    if result.get("complete"):
        print("완료 후보: 재학습하지 않음", flush=True)
        return
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
    if drop_site:
        features = [c for c in features if c != "site"]
    features = features + list(common)
    if len(features) != spec["features"]:
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
        prepare = numeric_input if drop_site else checked_input
        X, V = prepare(tr[features]), prepare(val[features])
        y = base_target(tr)
        cs = T.clear_sky(tr.cosz)
        weight = T.kt_weight(cs, T.sample_weight(tr, "cap", smp.w), CFG)
        pred = np.zeros(len(val))
        started = time.monotonic()
        result["running"] = dict(cutoff=cutoff.isoformat(), completed_seeds=[])
        save(path, result)
        print(f"마감{cutoff:%Y-%m-%d}: CatBoost {len(features)}입력·3시드 시작", flush=True)
        for seed in spec["seeds"]:
            model = make_model(seed)
            model.fit(X, y, sample_weight=weight, cat_features=[] if drop_site else ["site"])
            pred += model.predict(V)/3
            del model
            gc.collect()
            result["running"]["completed_seeds"].append(seed)
            save(path, result)
            print(f"CatBoost seed={seed} 완료, 누적{time.monotonic()-started:.0f}초", flush=True)
        pred = np.maximum(pred, 0)*T.clear_sky(val.cosz)*val.cap.to_numpy()
        for name in names:
            ctx = evaluation_context(name, gen, sites)
            panel = load_reference(name, ctx["actual"], sites)
            base = panel.groupby(["ts", "group"]).forecast_kwh.sum().reindex(ctx["a"].index)
            f = fold(name)
            mask = val.ts.between(f["val_start"], f["val_end"]).to_numpy()
            forecast = grouped_prediction(ctx["actual"], val.loc[mask, ["site_id", "ts"]], pred[mask]).reindex(ctx["a"].index)
            r = metrics(ctx["a"], base, forecast, smp)
            if r["base_penalty"] != baseline[name]["vs_v3_3seeds"]["direct_penalty"]:
                raise ValueError("v6 대조 불일치")
            r.update(train_rows=len(tr), features=len(features), seconds=round(time.monotonic()-started, 1),
                     train_cutoff=cutoff.isoformat(), seeds=spec["seeds"])
            result["results"][name] = r
            save(path, result)
            print(name, json.dumps(r, ensure_ascii=False), flush=True)
        reason = rejection(result["results"])
        result.pop("running", None)
        if reason:
            result.update(complete=True, passed=False, early_rejection=reason,
                          uncompleted_windows=[n for n in WINDOWS if n not in result["results"]])
            if not result["uncompleted_windows"]:
                seasonal_summary(result)
            save(path, result)
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
                  and result["results"]["fold1_static"]["change_pct"] <= 0)
    save(path, result)


def seasonal_summary(result):
    seasons = [r for n, r in result["results"].items() if n != "fold1_static"]
    base = sum(r["base_penalty"] for r in seasons)
    candidate = sum(r["direct_penalty"] for r in seasons)
    result.update(seasonal_base_penalty=base, seasonal_candidate_penalty=candidate,
                  seasonal_change_pct=100*(candidate/base-1), evaluation_complete=True,
                  improved_seasons=sum(r["change_pct"] < 0 for r in seasons))


def main():
    run_candidate()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        frames = [dict(function=f.name, line=f.lineno) for f in traceback.extract_tb(exc.__traceback__)
                  if "/src/" in f.filename]
        print(json.dumps(dict(status="failed", error_type=type(exc).__name__, code_frames=frames)), flush=True)
        raise SystemExit(1) from None
