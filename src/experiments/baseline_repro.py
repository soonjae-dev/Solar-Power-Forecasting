"""노트북 17-A 기준선(17시 예보, L2, 참조 배정)을 로컬에서 재현한다.

    python -m src.experiments.baseline_repro [fold]        # 기본 development
"""
import json
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from ..config import EXP_DIR, PRED_DIR
from ..data import load_forecast, load_generation, load_sites, load_smp
from ..scoring import check_capacity, score_groups, to_groups
from ..solar import solar
from ..splits import fold

TRAIN_START = pd.Timestamp("2023-06-01 01:00", tz="Asia/Seoul")
WA_COLS = ["wa_cloud", "wa_temp", "wa_humidity", "wa_rain", "wa_wind_speed", "wa_ground_press"]
WB_COLS = ["wb_cloud", "wb_temp", "wb_rel_hum", "wb_uv_idx", "wb_precip_prob",
           "wb_total_liq", "wb_vis", "wb_wind_speed", "wb_dew_point"]
FEATS = WA_COLS + WB_COLS + ["elev", "azim", "cosz", "doy_sin", "doy_cos", "hour", "cap", "lat", "lon"]
PARAMS = dict(objective="regression", n_estimators=600, learning_rate=0.06, num_leaves=127,
              min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
              verbosity=-1, n_jobs=4, random_state=2026, deterministic=True, force_col_wise=True)


def features(base, sites, fa, fb):
    d = base.merge(sites, on="site_id", how="left", validate="many_to_one")
    for weather, station in [(fa, "wa_station"), (fb, "wb_station")]:
        d = d.merge(weather.drop(columns="round"), on=[station, "ts"], how="left", validate="many_to_one")
    assert len(d) == len(base)
    ts = pd.DatetimeIndex(d["ts"])
    d["elev"], d["azim"], d["cosz"] = solar(ts, d["lat"].to_numpy(), d["lon"].to_numpy())
    d["doy_sin"] = np.sin(2*np.pi*ts.dayofyear.to_numpy()/365.25).astype("float32")
    d["doy_cos"] = np.cos(2*np.pi*ts.dayofyear.to_numpy()/365.25).astype("float32")
    d["hour"] = ts.hour.to_numpy().astype("int8")
    d["cap"] = d["capacity_kw"].astype("float32")
    return d


def main(fold_name: str = "development") -> None:
    t0 = time.perf_counter()
    f = fold(fold_name)
    run_id = f"baseline_r17_{fold_name}"
    sites = load_sites()
    gen = load_generation()
    train = gen[gen["ts"].between(TRAIN_START, f["cutoff"])]
    val = gen[gen["ts"].between(f["val_start"], f["val_end"])]
    assert val.groupby("ts")["site_id"].nunique().eq(len(sites)).all(), "검증 구간 정답이 완전하지 않다"
    weather = []
    for provider, cols in [("wa", WA_COLS), ("wb", WB_COLS)]:
        w = load_forecast("train", provider, cols, TRAIN_START, f["val_end"])
        weather.append(w[w["round"] == 17])

    d = features(train, sites, *weather)
    d = d[d["elev"] > 0]
    y = (d["energy"] / d["cap"]).clip(0, 1.2).astype("float32")
    print(f"주간 학습 행 {len(d):,}  특징 {len(FEATS)}", flush=True)
    model = lgb.LGBMRegressor(**PARAMS).fit(d[FEATS].astype("float32"), y)
    del d, y

    d = features(val, sites, *weather)
    ratio = np.where(d["elev"].to_numpy() > 0, model.predict(d[FEATS].astype("float32")), 0.0)
    pred = d[["site_id", "ts", "energy"]].copy()
    pred["forecast_kwh"] = np.clip(ratio, 0, None) * d["cap"].to_numpy()

    assign = sites.set_index("site_id")["current_group"].map({"A": "G1", "B": "G2"})
    check_capacity(assign, sites)
    metrics = score_groups(to_groups(pred, assign), load_smp("train"))
    metrics.update(run_id=run_id, fold=fold_name, val_rows=len(pred),
                   elapsed_seconds=round(time.perf_counter() - t0, 1))
    out = EXP_DIR / run_id
    out.mkdir(parents=True, exist_ok=True)
    (PRED_DIR / run_id).mkdir(parents=True, exist_ok=True)
    pred.to_parquet(PRED_DIR / run_id / "predictions.parquet", index=False)
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:2])
