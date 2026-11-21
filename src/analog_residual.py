"""시간순 OOF 유사 예보의 집합 잔차 보정. 집계만 출력/저장, 2025-05 제외."""
import json

import numpy as np
import pandas as pd

from .config import CACHE_DIR, EXP_DIR
from .data import load_generation, load_sites, load_smp
from .direct_groups import metrics
from .gfs_more import load_more
from .gfs_v3 import checked_gfs
from .splits import fold
from .train import clear_sky
from .tune_v3 import END, evaluation_context, save
from .v6_error_budget import load_reference

HISTORY = tuple(f"oof6m_{i}" for i in range(1, 6))
ORDER = ("summer_support", "fold1_static", "autumn_support", "development", "selection")
WEATHER = ["wa_cloud", "wb_cloud", "wb_uv_idx", "cosz"]
GFS = [f"gfs_{c}_all" for c in ("dswrf", "tcdc", "lcdc", "mcdc", "hcdc", "pwat")]
FEATURES = WEATHER + ["base_kt"] + GFS
SPEC = dict(neighbors=20, shrink_days=20, max_correction=.10, rule="14.39",
            history=list(HISTORY), features=FEATURES, baseline="gfs60_3seeds")


def weighted_median(values, weights):
    values, weights = np.asarray(values), np.asarray(weights)
    if not np.isfinite(values).all() or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("중앙값 입력 조건 위반")
    order = np.argsort(values, kind="stable")
    return float(values[order][np.searchsorted(np.cumsum(weights[order]), weights.sum()/2)])


def aggregate(p, weather, common, smp):
    keys = ["ts", "group"]
    x = p.merge(weather, on=["site_id", "ts"], how="left", validate="one_to_one")
    g = x.groupby(keys).agg(base=("forecast_kwh", "sum"), actual=("energy", "sum"))
    # 특징 캐시가 없는 야간 행은 scale=0, 자동으로 보정 제외.
    x["scale"] = np.where(x.cosz.notna() & (x.cosz > 0),
                           clear_sky(x.cosz.fillna(0))*x.cap, 0)
    g["scale"] = x.groupby(keys).scale.sum()
    for c in WEATHER:
        denom = x.cap.where(x[c].notna()).groupby([x.ts, x.group]).sum()
        g[c] = (x[c]*x.cap).groupby([x.ts, x.group]).sum()/denom.replace(0, np.nan)
    g["base_kt"] = g.base/g.scale.replace(0, np.nan)
    ts = g.index.get_level_values("ts")
    for c in GFS:
        g[c] = common[c].reindex(ts).to_numpy()
    g["weight"] = smp.w.reindex(ts).to_numpy()*g.scale
    g["residual"] = (g.actual-g.base)/g.scale.replace(0, np.nan)
    return g


def correct(history, query, cutoff):
    """fit은 cutoff 이전만. query의 actual/residual은 절대로 참조하지 않는다."""
    if not history.index.is_unique or not query.index.is_unique:
        raise ValueError("집합 시각 중복")
    ht = history.index.get_level_values("ts")
    qt = query.index.get_level_values("ts")
    if (qt <= cutoff).any():
        raise ValueError("학습/평가 시각 겹침")
    valid = ((ht <= cutoff) & (history.scale > 0) & (history.weight > 0)
             & np.isfinite(history[FEATURES]).all(axis=1)
             & np.isfinite(history.residual) & np.isfinite(history.weight))
    h = history.loc[valid]
    prediction = query.base.copy()
    changed, insufficient, missing = 0, 0, 0
    for group in ("A", "B"):
        for hour in range(24):
            train = h.loc[(h.index.get_level_values("group") == group)
                          & (h.index.get_level_values("ts").hour == hour)]
            mask = ((query.index.get_level_values("group") == group) & (qt.hour == hour)
                    & (query.scale > 0))
            q = query.loc[mask]
            finite = np.isfinite(q[FEATURES]).all(axis=1)
            missing += int((~finite).sum())
            q = q.loc[finite]
            if len(train) < SPEC["neighbors"]:
                insufficient += len(q)
                continue
            X = train[FEATURES].to_numpy(dtype=float)
            mean, std = X.mean(axis=0), X.std(axis=0)
            std[std < 1e-8] = 1
            X = (X-mean)/std
            y, w = train.residual.to_numpy(), train.weight.to_numpy()
            for key, row in q.iterrows():
                d = np.sum((X-(row[FEATURES].to_numpy(dtype=float)-mean)/std)**2, axis=1)
                nearest = np.argsort(d, kind="stable")[:SPEC["neighbors"]]
                shrink = len(nearest)/(len(nearest)+SPEC["shrink_days"])
                delta = np.clip(shrink*weighted_median(y[nearest], w[nearest]),
                                -SPEC["max_correction"], SPEC["max_correction"])
                prediction.loc[key] = max(0., row.base + delta*row.scale)
                changed += int(delta != 0)
    return prediction, dict(training_hours=len(h), training_days=int(h.index.get_level_values("ts").normalize().nunique()),
                            corrected_group_hours=changed, missing_feature_hours=missing,
                            insufficient_history_hours=insufficient)


def rejection(results):
    if results.get("fold1_static", {}).get("change_pct", 0) > 0:
        return "고정11개월 악화"
    seasons = [r["change_pct"] for n, r in results.items() if n != "fold1_static"]
    if any(v > 3 for v in seasons):
        return "계절+3% 초과"
    if sum(v >= 0 for v in seasons) >= 2:
        return "2개 계절 비개선"
    return None


def main():
    path = EXP_DIR/"p52_analog_residual"/"metrics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        prior = json.loads(path.read_text())
        if prior["spec"] != SPEC:
            raise ValueError("체크포인트 계획 불일치")
        if prior.get("complete"):
            print("완료된 후보: 재평가하지 않음", flush=True)
            return
    result = dict(spec=SPEC, results={}, complete=False)
    print("유사 예보 집합잔차 보정: v6 3시드 OOF, 2025-05 제외", flush=True)
    gen = load_generation()
    gen = gen.loc[gen.ts <= END]
    sites = load_sites().set_index("site_id")
    smp = load_smp("train").set_index("ts")
    weather = pd.read_parquet(CACHE_DIR/"train_features.parquet", columns=["site_id", "ts", *WEATHER])
    weather = weather.loc[weather.ts <= END]
    gfs, _ = checked_gfs()
    more, _ = load_more()
    common = pd.concat([gfs, more], axis=1)[GFS]
    history = []
    for name in HISTORY:
        f = fold(name)
        actual = gen.loc[gen.ts.between(f["val_start"], f["val_end"])].copy()
        actual["group"] = actual.site_id.map(sites.current_group)
        p = load_reference(name, actual, sites)
        history.append(aggregate(p, weather, common, smp))
        print(f"과거 OOF 집계 준비: {name}", flush=True)
    history = pd.concat(history).sort_index()
    if not history.index.is_unique:
        raise ValueError("OOF 중복")
    reference = json.loads((EXP_DIR/"p26_gfs_more"/"metrics.json").read_text())["seeds3"]["gfs60"]
    for name in ORDER:
        f = fold(name)
        if f["val_end"] > END:
            raise ValueError("holdout 차단")
        ctx = evaluation_context(name, gen, sites)
        p = load_reference(name, ctx["actual"], sites)
        query = aggregate(p, weather, common, smp)
        pred, meta = correct(history, query, f["cutoff"])
        r = metrics(query.actual, query.base, pred, smp)
        if r["base_penalty"] != reference[name]["vs_v3_3seeds"]["direct_penalty"]:
            raise ValueError("p26 대조 페널티 불일치")
        r.update(meta)
        result["results"][name] = r
        reason = rejection(result["results"])
        result.update(early_rejection=reason, complete=bool(reason))
        if reason:
            result.update(passed=False, uncompleted_windows=[n for n in ORDER if n not in result["results"]])
        save(path, result)
        print(f"{name}: {r['base_penalty']}→{r['direct_penalty']}, {r['change_pct']:+.3f}%, CI90={r['change_ci90_pct']}", flush=True)
        if reason:
            print(f"조기 기각: {reason}", flush=True)
            return
    seasons = [r for n, r in result["results"].items() if n != "fold1_static"]
    change = 100*(sum(r["direct_penalty"] for r in seasons)/sum(r["base_penalty"] for r in seasons)-1)
    result.update(complete=True, seasonal_change_pct=round(change, 3),
                  passed=change <= -1 and sum(r["change_pct"] < 0 for r in seasons) >= 3
                  and max(r["change_pct"] for r in seasons) <= 3
                  and result["results"]["fold1_static"]["change_pct"] <= 0)
    save(path, result)
    print(json.dumps({k: result[k] for k in ("complete", "seasonal_change_pct", "passed")}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # 데이터가 포함될 수 있는 exception message/traceback은 출력하지 않는다.
        import traceback
        frames = [dict(function=f.name, line=f.lineno) for f in traceback.extract_tb(exc.__traceback__)
                  if "/src/" in f.filename]
        print(json.dumps(dict(status="failed", error_type=type(exc).__name__, code_frames=frames)), flush=True)
        raise SystemExit(1) from None
