"""gfs60에 GEFS 에어로졸 광학 두께 20열을 더해 재학습 검증한다. 집계만 저장한다.

    .venv/bin/python -m src.gfs_aod

판정 기준은 실행 전에 고정했다(docs/validation.md). 2025-05 holdout은 쓰지 않는다.
"""
import json

import numpy as np
import pandas as pd

from . import train as T
from .config import CACHE_DIR, EXP_DIR, TZ
from .data import load_generation, load_sites, load_smp
from .gfs_fetch import CHEM_DIR, FIRST_STEP, LAST_STEP
from .gfs_more import load_more
from .gfs_pca import three_seed_verdict, verdict
from .gfs_pilot import FIRST_TS_HOUR, region_masks
from .gfs_v3 import FIRST_DAY, LAST_DAY, checked_gfs
from .tune_v3 import END, run_stage, save
from .weather_teacher import CFG

CANDIDATES = {"aod20": {}}
HOURS = LAST_STEP - FIRST_STEP + 1


def load_aod(first=FIRST_DAY, last=LAST_DAY):
    """낮 5개 시각(ts 06~18시)의 영역 평균을 다시 평균해 그날 ts 04~20시에 같은 값으로 편다."""
    masks = region_masks()
    rows, missing = [], []
    for day in pd.date_range(first, last):
        path = CHEM_DIR / f"{day:%Y%m%d}.npz"
        if not path.exists():
            missing.append(day.date().isoformat())
            continue
        z = np.load(path)
        values = {f"gefs_{key}_{name}": float(z[key][:, mask].mean()) for key in ("aod", "dust") for name, mask in masks.items()}
        index = pd.DatetimeIndex([day + pd.Timedelta(hours=FIRST_TS_HOUR + i) for i in range(HOURS)]).tz_localize(TZ)
        rows.append(pd.DataFrame({k: np.full(HOURS, v) for k, v in values.items()}, index=index))
    return pd.concat(rows).astype("float32").sort_index(), missing


def main():
    print("gfs60+GEFS 에어로졸 재학습 검증, holdout 제외", flush=True)
    gfs, _ = checked_gfs()
    base = pd.concat([gfs, load_more()[0]], axis=1)
    aod, missing = load_aod()
    if aod.shape[1] != 20 or not aod.index.is_unique or aod.index.max() > END:
        raise ValueError("에어로졸 열·시각 조건 위반")
    print(f"에어로졸 {len(aod)}시각, 빠진 날 {len(missing)}", flush=True)
    common = pd.concat([base, aod], axis=1)
    sites = load_sites().set_index("site_id")
    smp = load_smp("train").set_index("ts")
    gen = load_generation()
    gen = gen[gen.ts <= END]
    d = T.load_train_features()
    cols = ["site_id", "ts", "energy", *T.select_features(d, CFG["feats"])]
    d = d.loc[d.ts <= END, cols].copy()
    T._cache.clear()
    out = EXP_DIR / "p33_gefs_aod"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "metrics.json"
    result = json.loads(path.read_text()) if path.exists() else dict(baseline_config=CFG, candidates=CANDIDATES, cache_name=CACHE_DIR.name, control="gfs60")
    run_stage(d, gen, sites, smp, result, path, 1, list(CANDIDATES), variants=CANDIDATES, added=common, control_columns=list(base))
    result["verdict_vs_control"] = verdict(result["seeds1"]["aod20"])
    save(path, result)
    print("판정", json.dumps(result["verdict_vs_control"], ensure_ascii=False), flush=True)
    if result["verdict_vs_control"]["passed"]:
        run_stage(d, gen, sites, smp, result, path, 3, list(CANDIDATES), variants=CANDIDATES, added=common, control_columns=list(base))
        ctrl = json.loads((EXP_DIR / "p26_gfs_more" / "metrics.json").read_text())["seeds3"]["gfs60"]
        result["verdict3_vs_control"] = three_seed_verdict(result["seeds3"]["aod20"], ctrl)
        save(path, result)
        print("3시드 판정", json.dumps(result["verdict3_vs_control"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
