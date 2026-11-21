"""gfs60에 ECMWF ssrd 10열을 더해 재학습 검증한다. 집계만 저장한다.

    .venv/bin/python -m src.experiments.ecmwf_model

ssrd가 없는 날(2024-03 이전, 늦게 공개된 날)은 결측이다. 2025-05 holdout은 쓰지 않는다.
"""
import json

import numpy as np
import pandas as pd

from .. import train as T
from ..config import CACHE_DIR, EXP_DIR, TZ
from ..data import load_generation, load_sites, load_smp
from .ecmwf_fetch import FIRST_TS_HOUR, HOURS, OUT_DIR
from ..gfs_more import load_more
from ..gfs_pca import three_seed_verdict, verdict
from ..gfs_pilot import region_masks
from ..gfs_v3 import checked_gfs
from ..tune_v3 import END, run_stage, save
from ..weather_teacher import CFG

CANDIDATES = {"ec10": {}}


def load_ecmwf(first="2024-03-03", last="2026-05-31"):
    masks = region_masks()
    rows, missing = [], []
    for day in pd.date_range(first, last):
        path = OUT_DIR / f"{day:%Y%m%d}.npz"
        if not path.exists():
            missing.append(day.date().isoformat())
            continue
        z = np.load(path)["ssrd"]
        if z.shape[0] != HOURS:
            raise ValueError("시간 수 불일치")
        index = pd.DatetimeIndex([day + pd.Timedelta(hours=FIRST_TS_HOUR + i) for i in range(HOURS)]).tz_localize(TZ)
        rows.append(pd.DataFrame({f"ec_ssrd_{name}": z[:, mask].mean(axis=1) for name, mask in masks.items()}, index=index))
    return pd.concat(rows).astype("float32").sort_index(), missing


def alignment(ec, gfs):
    """학습 기간에서 ECMWF 전체 영역 일사와 GFS 일사의 상관이 시차 0에서 가장 커야 한다."""
    a = ec["ec_ssrd_all"]
    b = gfs["gfs_dswrf_all"]
    return {lag: round(float(a.corr(b.shift(lag).reindex(a.index))), 4) for lag in (-1, 0, 1)}


def main():
    print("gfs60+ECMWF ssrd 재학습 검증, holdout 제외", flush=True)
    gfs, _ = checked_gfs()
    base = pd.concat([gfs, load_more()[0]], axis=1)
    ec, missing = load_ecmwf(last=f"{END - pd.Timedelta(hours=1):%Y-%m-%d}")
    ec = ec.loc[ec.index <= END]
    lags = alignment(ec, gfs)
    print(f"ECMWF {len(ec)}시각, 빠진 날 {len(missing)}, 시차 상관 {lags}", flush=True)
    if ec.shape[1] != 10 or not ec.index.is_unique or max(lags, key=lags.get) != 0:
        raise ValueError("ECMWF 열·시각·정렬 조건 위반")
    common = pd.concat([base, ec.reindex(base.index)], axis=1)
    sites = load_sites().set_index("site_id")
    smp = load_smp("train").set_index("ts")
    gen = load_generation()
    gen = gen[gen.ts <= END]
    d = T.load_train_features()
    cols = ["site_id", "ts", "energy", *T.select_features(d, CFG["feats"])]
    d = d.loc[d.ts <= END, cols].copy()
    T._cache.clear()
    out = EXP_DIR / "p68_ecmwf_model"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "metrics.json"
    result = json.loads(path.read_text()) if path.exists() else dict(
        rule="14.64", baseline_config=CFG, candidates=CANDIDATES, cache_name=CACHE_DIR.name, control="gfs60",
        ecmwf_hours=len(ec), ecmwf_missing_days=len(missing), alignment=lags)
    run_stage(d, gen, sites, smp, result, path, 1, list(CANDIDATES), variants=CANDIDATES, added=common, control_columns=list(base))
    result["verdict_vs_control"] = verdict(result["seeds1"]["ec10"])
    save(path, result)
    print("판정", json.dumps(result["verdict_vs_control"], ensure_ascii=False), flush=True)
    if result["verdict_vs_control"]["passed"]:
        run_stage(d, gen, sites, smp, result, path, 3, list(CANDIDATES), variants=CANDIDATES, added=common, control_columns=list(base))
        ctrl = json.loads((EXP_DIR / "p26_gfs_more" / "metrics.json").read_text())["seeds3"]["gfs60"]
        result["verdict3_vs_control"] = three_seed_verdict(result["seeds3"]["ec10"], ctrl)
        save(path, result)
        print("3시드 판정", json.dumps(result["verdict3_vs_control"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
