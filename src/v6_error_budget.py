"""v6의 오차 예산과 관측 이력별 기여를 집계한다. 원본 행/예측은 출력·저장하지 않는다."""
import json

import numpy as np
import pandas as pd

from .config import CACHE_DIR, EXP_DIR, PRED_DIR
from .data import load_generation, load_sites, load_smp
from .diagnose_season import run_id
from .splits import fold
from .tune_v3 import END, WINDOWS, evaluation_context, save


def history_cohort(first_seen, cutoff):
    age = (cutoff - first_seen).dt.total_seconds() / 86400
    return pd.Series(np.select([age.isna(), age < 180, age < 730],
                               ["unseen", "under_180_days", "180_to_729_days"], default="730_days_or_more"),
                     index=first_seen.index)


def decompose(residuals, weights):
    penalty = .1 * residuals.abs().sum(axis=1) * weights
    floor = .1 * residuals.sum(axis=1).abs() * weights
    return penalty, floor


def load_reference(name, actual, sites):
    paths = [PRED_DIR/run_id(c, name)/"predictions.parquet" for c in ("gfs60seed", "gfs60")]
    path = next((p for p in paths if p.exists()), None)
    if path is None:
        raise ValueError("현재v6 비교 예측 없음")
    predicted = pd.read_parquet(path, columns=["site_id", "ts", "forecast_kwh"])
    keys = actual[["site_id", "ts", "energy", "group"]]
    p = keys.merge(predicted, on=["site_id", "ts"], how="left", validate="one_to_one")
    if p.forecast_kwh.isna().any():
        raise ValueError("현재v6 예측과 평가키 불일치")
    p["cap"] = p.site_id.map(sites.capacity_kw)
    return p


def main():
    print("v6 잔여오차 집계, 2025-05 제외", flush=True)
    gen = load_generation()
    gen = gen[gen.ts <= END]
    sites = load_sites().set_index("site_id")
    smp = load_smp("train").set_index("ts")
    weather = pd.read_parquet(CACHE_DIR/"train_features.parquet", columns=["site_id", "ts", "wa_cloud", "wb_cloud"])
    weather = weather[weather.ts <= END]
    prior = json.loads((EXP_DIR/"p26_gfs_more"/"metrics.json").read_text())["seeds3"]["gfs60"]
    result = dict(rule="14.36", baseline="gfs60_3seeds", windows={})
    out = EXP_DIR/"p50_v6_error_budget"
    out.mkdir(parents=True, exist_ok=True)
    for name in WINDOWS:
        f = fold(name)
        if f["val_end"] > END:
            raise ValueError("holdout 차단")
        ctx = evaluation_context(name, gen, sites)
        p = load_reference(name, ctx["actual"], sites)
        p["e"] = p.forecast_kwh - p.energy
        residual = p.groupby(["ts", "group"]).e.sum().unstack("group")
        w = smp.w.reindex(residual.index)
        penalty, floor = decompose(residual, w)
        reference = prior[name]["vs_v3_3seeds"]["direct_penalty"]
        if round(float(penalty.sum())) != reference:
            raise ValueError("현재v6 페널티가p26과 불일치")
        group_index = pd.MultiIndex.from_frame(p[["ts", "group"]])
        group_error = residual.stack().reindex(group_index).to_numpy()
        p["aligned_contribution"] = .1 * p.ts.map(smp.w).to_numpy() * p.e.to_numpy() * np.sign(group_error)
        first = gen[gen.ts <= f["cutoff"]].groupby("site_id").ts.min().reindex(sites.index)
        cohort = history_cohort(first, f["cutoff"])
        p["cohort"] = p.site_id.map(cohort)
        cohorts = {}
        for c, rows in p.groupby("cohort"):
            count = rows.site_id.nunique()
            cohorts[c] = dict(site_count=int(count), suppressed=count < 5)
            if count >= 5:
                cohorts[c].update(aligned_penalty_contribution=round(float(rows.aligned_contribution.sum())),
                                  prediction_actual_ratio=round(float(rows.forecast_kwh.sum()/rows.energy.sum()), 4))
        joined = p[["site_id", "ts", "cap"]].merge(weather, on=["site_id", "ts"], how="left", validate="one_to_one")
        cloud = joined[["wa_cloud", "wb_cloud"]].mean(axis=1)
        weighted = cloud.mul(joined.cap).groupby(joined.ts).sum()
        denom = joined.cap.where(cloud.notna()).groupby(joined.ts).sum().replace(0, np.nan)
        cloud = (weighted/denom).reindex(penalty.index)
        regimes = np.select([cloud.isna(), cloud <= 30, cloud <= 70], ["unknown", "clear", "mixed"], default="cloudy")
        regime_penalty = penalty.groupby(regimes).sum()
        daily = penalty.groupby((penalty.index - pd.Timedelta(hours=1)).normalize()).sum()
        peak = penalty.index.hour.isin(range(11, 16))
        r = dict(penalty=reference, reproduces_p26=True, total_residual_floor=round(float(floor.sum())),
                 allocation_excess=round(float((penalty-floor).sum())),
                 allocation_excess_pct=round(float(100*(penalty-floor).sum()/penalty.sum()), 3),
                 peak_penalty_share_pct=round(float(100*penalty[peak].sum()/penalty.sum()), 3),
                 worst_10pct_days_share_pct=round(float(100*daily.nlargest(int(np.ceil(.1*len(daily)))).sum()/daily.sum()), 3),
                 cloud_regime_penalty_share_pct={k:round(float(100*v/penalty.sum()), 3) for k,v in regime_penalty.items()},
                 observation_history_cohorts=cohorts)
        if not np.isclose(p.aligned_contribution.sum(), penalty.sum()):
            raise ValueError("코호트 기여의 합과 페널티 불일치")
        result["windows"][name] = r
        save(out/"metrics.json", result)
        print(name, json.dumps(r, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # 예외 본문에 원본 행이 포함될 수 있어 유형만 출력한다.
        print(json.dumps(dict(status="stopped", error_type=type(exc).__name__)), flush=True)
        raise SystemExit(1)
