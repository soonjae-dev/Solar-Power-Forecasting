"""GFS 20열에 구름 층·가강수량 40열을 더해 재학습 검증한다. 집계만 저장한다.

    .venv/bin/python -m src.gfs_more

판정 기준은 실행 전에 고정했다(docs/validation.md). 2025-05 holdout은 쓰지 않는다.
"""
import json

import numpy as np
import pandas as pd

from . import train as T
from .config import CACHE_DIR, DATA_ROOT, EXP_DIR, TZ
from .data import load_generation, load_sites, load_smp
from .gfs_pca import three_seed_verdict, verdict
from .gfs_pilot import FIRST_TS_HOUR, region_masks, summarize
from .gfs_v3 import FIRST_DAY, LAST_DAY, checked_gfs
from .tune_v3 import END, run_stage, save
from .weather_teacher import CFG

MORE_DIR = DATA_ROOT / "external" / "gfs_00utc_more"
CANDIDATES = {"gfs60": {}}
LIMITS = (("lcdc", 0, 100), ("mcdc", 0, 100), ("hcdc", 0, 100), ("pwat", 0, None))


def load_more(first=FIRST_DAY, last=LAST_DAY):
    masks = region_masks()
    rows, missing = [], []
    for day in pd.date_range(first, last):
        path = MORE_DIR / f"{day:%Y%m%d}.npz"
        if not path.exists():
            missing.append(day.date().isoformat())
            continue
        z = np.load(path)
        index = pd.DatetimeIndex([day + pd.Timedelta(hours=FIRST_TS_HOUR + i) for i in range(z["pwat"].shape[0])]).tz_localize(TZ)
        frame = {}
        for key, low, high in LIMITS:
            for name, values in summarize(np.clip(z[key], low, high), masks).items():
                frame[f"gfs_{key}_{name}"] = values
        rows.append(pd.DataFrame(frame, index=index))
    return pd.concat(rows).astype("float32").sort_index(), missing


def main():
    print("v3+GFS 60열 재학습 검증, holdout 제외", flush=True)
    gfs, _ = checked_gfs()
    more, missing = load_more()
    if more.shape[1] != 40 or not more.index.is_unique or more.index.max() > END:
        raise ValueError("추가 GFS 열·시각 조건 위반")
    print(f"추가 GFS {len(more)}시각, 빠진 날 {len(missing)}", flush=True)
    common = pd.concat([gfs, more], axis=1)
    sites = load_sites().set_index("site_id")
    smp = load_smp("train").set_index("ts")
    gen = load_generation()
    gen = gen[gen.ts <= END]
    d = T.load_train_features()
    cols = ["site_id", "ts", "energy", *T.select_features(d, CFG["feats"])]
    d = d.loc[d.ts <= END, cols].copy()
    T._cache.clear()
    out = EXP_DIR / "p26_gfs_more"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "metrics.json"
    result = json.loads(path.read_text()) if path.exists() else dict(baseline_config=CFG, candidates=CANDIDATES, cache_name=CACHE_DIR.name, control="gfs20")
    run_stage(d, gen, sites, smp, result, path, 1, list(CANDIDATES), variants=CANDIDATES, added=common, control_columns=list(gfs))
    result["verdict_vs_gfs20"] = verdict(result["seeds1"]["gfs60"])
    save(path, result)
    print("판정", json.dumps(result["verdict_vs_gfs20"], ensure_ascii=False), flush=True)
    if result["verdict_vs_gfs20"]["passed"]:
        run_stage(d, gen, sites, smp, result, path, 3, list(CANDIDATES), variants=CANDIDATES, added=common, control_columns=list(gfs))
        ctrl = json.loads((EXP_DIR / "p24_gfs_v3" / "metrics.json").read_text())["seeds3"]["gfs20"]
        result["verdict3_vs_gfs20"] = three_seed_verdict(result["seeds3"]["gfs60"], ctrl)
        save(path, result)
        print("3시드 판정", json.dumps(result["verdict3_vs_gfs20"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
