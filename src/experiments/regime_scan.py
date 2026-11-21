"""기상 조건별 v6 집합 편향 탐색. OOF로 배율을 정하고 fold1_static에서 확인한다. 집계만 저장.

    MPLCONFIGDIR=/tmp/oibc-mpl .venv/bin/python -m src.experiments.regime_scan
"""
import json
import traceback

import numpy as np
import pandas as pd

from .. import train as T
from ..config import EXP_DIR, PRED_DIR
from ..data import load_sites, load_smp
from ..tune_v3 import save

PATH = EXP_DIR/"p65_regime_scan"/"metrics.json"
OOF = [f"p1_gfs60seed_oof6m_{k}" for k in range(1, 6)]
STATIC = "p1_gfs60seed_fold1_static"
VARS = ["wa_cloud", "wa_temp", "wa_humidity", "wa_ground_press", "wa_wind_speed", "wa_rain", "wa_rain_daysum", "wa_cloud_rdiff",
        "wb_temp", "wb_rel_hum", "wb_vis", "wb_ceiling", "wb_uv_idx", "wb_precip_prob", "wb_total_liq", "wb_ice_prob",
        "wb_cloud", "wb_cloud_rdiff", "dew_spread", "cloud_absgap", "rain_gap", "day_cloud_gap"]
SHRINK = 500
SEASON = {6: "summer", 7: "summer", 8: "summer", 9: "autumn", 10: "autumn", 11: "autumn",
          12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring"}


def group_hours(names, feats, sites):
    """집합·시각별 실적·예측 합과 변수의 용량가중 평균."""
    p = pd.concat([pd.read_parquet(PRED_DIR/n/"predictions.parquet", columns=["site_id", "ts", "energy", "forecast_kwh"]) for n in names])
    if p.duplicated(["site_id", "ts"]).any():
        raise ValueError("예측 구간 중복")
    p = p.dropna(subset=["energy"]).merge(feats, on=["site_id", "ts"], how="left", validate="one_to_one")
    p["group"] = p.site_id.map(sites.current_group)
    p["cap"] = p.site_id.map(sites.capacity_kw)
    keys = [p.ts, p.group]
    out = p.groupby(keys)[["energy", "forecast_kwh"]].sum()
    for v in VARS:
        valid = p[v].notna()
        out[v] = (p[v]*p.cap).where(valid).groupby(keys).sum(min_count=1)/p.cap.where(valid).groupby(keys).sum().replace(0, np.nan)
    out = out[(out.energy > 0) | (out.forecast_kwh > 0)]
    return out.reset_index()


def penalty(g, forecast, w):
    return .1*w*(forecast-g.energy).abs()


def factors(g, v):
    edges = np.unique(np.nanquantile(g[v], np.linspace(0, 1, 11)))
    if len(edges) < 3:
        return None
    b = np.clip(np.searchsorted(edges, g[v], side="right")-1, 0, len(edges)-2)
    b = np.where(g[v].isna(), -1, b)
    c = {}
    for k in range(len(edges)-1):
        m = b == k
        raw = g.energy[m].sum()/g.forecast_kwh[m].sum() if g.forecast_kwh[m].sum() > 0 else 1.
        c[k] = 1+(raw-1)*m.sum()/(m.sum()+SHRINK)
    return edges, c


def apply(g, v, fit):
    edges, c = fit
    b = np.clip(np.searchsorted(edges, g[v], side="right")-1, 0, len(edges)-2)
    scale = np.array([c[k] for k in b])
    return g.forecast_kwh*np.where(g[v].isna(), 1., scale)


def block_ci(g, base, cand, rng, n=2000):
    day = (g.ts-pd.Timedelta(hours=1)).dt.normalize()
    blk = ((day-day.min()).dt.days//7).to_numpy()
    s, b = np.bincount(blk, weights=cand-base), np.bincount(blk, weights=base)
    idx = rng.integers(0, len(s), (n, len(s)))
    return [round(float(x), 3) for x in np.percentile(100*s[idx].sum(1)/b[idx].sum(1), [5, 95])]


def main():
    PATH.parent.mkdir(parents=True, exist_ok=True)
    sites = load_sites().set_index("site_id")
    w = load_smp("train").set_index("ts").w
    d = T.load_train_features()
    d["dew_spread"] = d.wb_temp-d.wb_dew_point
    feats = d[["site_id", "ts", *VARS]].copy()
    del d
    T._cache.clear()
    oof, static = group_hours(OOF, feats, sites), group_hours([STATIC], feats, sites)
    wo, ws = oof.ts.map(w).to_numpy(), static.ts.map(w).to_numpy()
    if np.isnan(wo).any() or np.isnan(ws).any():
        raise ValueError("가중치 누락")
    base_o, base_s = penalty(oof, oof.forecast_kwh, wo), penalty(static, static.forecast_kwh, ws)
    result = dict(rule="14.61", oof_group_hours=len(oof), static_group_hours=len(static),
                  oof_base=round(float(base_o.sum())), static_base=round(float(base_s.sum())), variables={})
    rng = np.random.default_rng(2026)
    season = static.ts.map(lambda t: SEASON.get((t-pd.Timedelta(hours=1)).month, "may")).to_numpy()
    for v in VARS:
        fit = factors(oof, v)
        if fit is None:
            result["variables"][v] = dict(skipped="분위 부족")
            continue
        po = penalty(oof, apply(oof, v, fit), wo)
        ps = penalty(static, apply(static, v, fit), ws)
        seasons = {s: round(float(100*(ps[season == s].sum()/base_s[season == s].sum()-1)), 3) for s in ("summer", "autumn", "winter", "spring")}
        result["variables"][v] = dict(
            missing_share=round(float(oof[v].isna().mean()), 4),
            factors=[round(float(x), 4) for x in fit[1].values()],
            oof_change_pct=round(float(100*(po.sum()/base_o.sum()-1)), 3),
            static_change_pct=round(float(100*(ps.sum()/base_s.sum()-1)), 3),
            static_ci90_pct=block_ci(static, base_s.to_numpy(), ps.to_numpy(), rng),
            static_seasons=seasons)
        print(v, json.dumps({k: result["variables"][v][k] for k in ("oof_change_pct", "static_change_pct")}), flush=True)
    ok = {k: r for k, r in result["variables"].items() if "skipped" not in r}
    pick = min(ok, key=lambda k: ok[k]["oof_change_pct"])
    r = ok[pick]
    result["selected"] = pick
    result["passed"] = bool(r["static_change_pct"] <= -1 and r["static_ci90_pct"][1] < 0
                            and sum(x < 0 for x in r["static_seasons"].values()) >= 3)
    result["complete"] = True
    save(PATH, result)
    print("선택", pick, json.dumps(r, ensure_ascii=False), "통과", result["passed"], flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        frames = [dict(function=f.name, line=f.lineno) for f in traceback.extract_tb(exc.__traceback__)
                  if "/src/" in f.filename]
        print(json.dumps(dict(status="failed", error_type=type(exc).__name__, code_frames=frames)), flush=True)
        raise SystemExit(1) from None
