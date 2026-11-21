"""표준 대비 미시험 GFS 구조 특징 검증. A: 앞뒤 시각, B: 칸 안 표준편차. 집계만 저장한다.

    .venv/bin/python -m src.experiments.gfs_struct_model lag|disp
"""
import json
import sys

import numpy as np
import pandas as pd

from .. import train as T
from ..config import CACHE_DIR, EXP_DIR, TZ
from ..data import load_generation, load_sites, load_smp
from ..gfs_more import load_more
from ..gfs_pca import verdict
from ..gfs_pilot import FIRST_TS_HOUR, GFS_DIR, region_masks
from ..gfs_v3 import FIRST_DAY, LAST_DAY, checked_gfs
from ..ldaps_model import load_ldaps
from ..tune_v3 import END, run_stage, save
from ..weather_teacher import CFG


def lag_features(gfs):
    day = (gfs.index - pd.Timedelta(hours=1)).normalize()
    cols = [c for c in gfs if c.startswith(("gfs_dswrf_", "gfs_tcdc_"))]
    g = gfs[cols].groupby(day)
    prev = g.shift(1).add_suffix("_prev")
    nxt = g.shift(-1).add_suffix("_next")
    return pd.concat([prev, nxt], axis=1).astype("float32")


def disp_features(first=FIRST_DAY, last=LAST_DAY):
    masks = region_masks()
    rows = []
    for day in pd.date_range(first, last):
        p = GFS_DIR / f"{day:%Y%m%d}.npz"
        if not p.exists():
            continue
        z = np.load(p)
        idx = pd.DatetimeIndex([day + pd.Timedelta(hours=FIRST_TS_HOUR + i) for i in range(z["dswrf"].shape[0])]).tz_localize(TZ)
        cols = {}
        for key, low, high in (("dswrf", 0, None), ("tcdc", 0, 100)):
            f = np.clip(z[key], low, high)
            for name, m in masks.items():
                cols[f"gfs_{key}_std_{name}"] = f[:, m].std(axis=1)
        rows.append(pd.DataFrame(cols, index=idx))
    return pd.concat(rows).astype("float32").sort_index()


def main(kind):
    gfs, _ = checked_gfs()
    base = pd.concat([gfs, load_more()[0]], axis=1)
    ld = load_ldaps(last=f"{END - pd.Timedelta(hours=1):%Y-%m-%d}")[0].reindex(base.index)
    control = pd.concat([base, ld], axis=1)
    extra = lag_features(gfs) if kind == "lag" else disp_features().reindex(base.index)
    if extra.shape[1] != (40 if kind == "lag" else 20):
        raise ValueError("열 수 불일치")
    common = pd.concat([control, extra], axis=1)
    name = f"gfs{kind}"
    cand = {name: {}}
    sites = load_sites().set_index("site_id"); smp = load_smp("train").set_index("ts")
    gen = load_generation(); gen = gen[gen.ts <= END]
    d = T.load_train_features()
    d = d.loc[d.ts <= END, ["site_id", "ts", "energy", *T.select_features(d, CFG["feats"])]].copy()
    T._cache.clear()
    out = EXP_DIR / ("p81a_gfs_lag" if kind == "lag" else "p81b_gfs_disp"); out.mkdir(parents=True, exist_ok=True)
    path = out / "metrics.json"
    result = json.loads(path.read_text()) if path.exists() else dict(rule="14.81", control="gfs60+ldaps", cache_name=CACHE_DIR.name)
    run_stage(d, gen, sites, smp, result, path, 1, list(cand), variants=cand, added=common, control_columns=list(control))
    result["verdict_vs_control"] = verdict(result["seeds1"][name])
    save(path, result)
    print("판정", json.dumps(result["verdict_vs_control"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
