"""gfs60에 기상청 LDAPS 하향 단파 10열을 더해 재학습 검증한다. 집계만 저장한다.

    .venv/bin/python -m src.ldaps_model
"""
import json

import numpy as np
import pandas as pd

from . import train as T
from .config import CACHE_DIR, EXP_DIR, TZ
from .data import load_generation, load_sites, load_smp
from .gfs_more import load_more
from .gfs_pca import three_seed_verdict, verdict
from .gfs_v3 import checked_gfs
from .ldaps_fetch import OUT_DIR
from .tune_v3 import END, run_stage, save
from .weather_teacher import CFG

CANDIDATES = {"ldaps10": {}}
USE_STEPS = (23, 25, 27, 29, 31)


def load_ldaps(first="2021-01-02", last="2026-03-31"):
    """ts 08~16시 2시간 간격 5시각을 선형 보간해 ts 04~20시 17시각으로 편다."""
    rows, missing = [], []
    for day in pd.date_range(first, last):
        path = OUT_DIR / f"{day:%Y%m%d}.npz"
        if not path.exists():
            missing.append(day.date().isoformat())
            continue
        z = np.load(path)
        keep = np.isin(z["steps"], USE_STEPS)                 # 전 기간 같은 정의: ts 08·10·12·14·16시만 쓴다(14.72 개정)
        if keep.sum() != len(USE_STEPS):
            missing.append(day.date().isoformat())
            continue
        hours = np.r_[4, z["steps"][keep] - 15, 20]           # ts 04·20시는 0으로 두고 사이를 선형 보간
        full = np.arange(4, 21)
        data = z["dswrf"][keep]
        vals = np.stack([np.interp(full, hours, np.r_[0, data[:, k], 0]) for k in range(data.shape[1])], axis=1)
        index = pd.DatetimeIndex([day + pd.Timedelta(hours=int(h)) for h in full]).tz_localize(TZ)
        rows.append(pd.DataFrame(np.clip(vals, 0, None), index=index, columns=[f"ldaps_dswrf_{r}" for r in z["regions"]]))
    return pd.concat(rows).astype("float32").sort_index(), missing


def main():
    print("gfs60+LDAPS 재학습 검증, holdout 제외", flush=True)
    gfs, _ = checked_gfs()
    base = pd.concat([gfs, load_more()[0]], axis=1)
    ld, missing = load_ldaps(last=f"{END - pd.Timedelta(hours=1):%Y-%m-%d}")
    ld = ld.loc[ld.index <= END]
    lags = {lag: round(float(ld.ldaps_dswrf_all.corr(gfs.gfs_dswrf_all.shift(lag).reindex(ld.index))), 4) for lag in (-1, 0, 1)}
    print(f"LDAPS {len(ld)}시각, 빠진 날 {len(missing)}, 시차 상관 {lags}", flush=True)
    if ld.shape[1] != 10 or not ld.index.is_unique or max(lags, key=lags.get) != 0:
        raise ValueError("LDAPS 열·시각·정렬 조건 위반")
    common = pd.concat([base, ld.reindex(base.index)], axis=1)
    sites = load_sites().set_index("site_id")
    smp = load_smp("train").set_index("ts")
    gen = load_generation()
    gen = gen[gen.ts <= END]
    d = T.load_train_features()
    d = d.loc[d.ts <= END, ["site_id", "ts", "energy", *T.select_features(d, CFG["feats"])]].copy()
    T._cache.clear()
    out = EXP_DIR / "p72_ldaps_model"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "metrics.json"
    result = json.loads(path.read_text()) if path.exists() else dict(rule="14.72", baseline_config=CFG, candidates=CANDIDATES,
                                                                    cache_name=CACHE_DIR.name, control="gfs60", missing_days=len(missing), alignment=lags)
    run_stage(d, gen, sites, smp, result, path, 1, list(CANDIDATES), variants=CANDIDATES, added=common, control_columns=list(base))
    result["verdict_vs_control"] = verdict(result["seeds1"]["ldaps10"])
    save(path, result)
    print("판정", json.dumps(result["verdict_vs_control"], ensure_ascii=False), flush=True)
    if result["verdict_vs_control"]["passed"]:
        run_stage(d, gen, sites, smp, result, path, 3, list(CANDIDATES), variants=CANDIDATES, added=common, control_columns=list(base))
        ctrl = json.loads((EXP_DIR / "p26_gfs_more" / "metrics.json").read_text())["seeds3"]["gfs60"]
        result["verdict3_vs_control"] = three_seed_verdict(result["seeds3"]["ldaps10"], ctrl)
        save(path, result)
        print("3시드 판정", json.dumps(result["verdict3_vs_control"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
