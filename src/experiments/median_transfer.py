"""후보 13 배율의 모델 간 이전성 확인. 재학습 없음, 집계만 저장.

    MPLCONFIGDIR=/tmp/oibc-mpl .venv/bin/python -m src.experiments.median_transfer
"""
import json
import traceback

import numpy as np
import pandas as pd

from .. import train as T
from ..config import EXP_DIR
from ..data import load_sites, load_smp
from .median_scale import apply, fit, group_hours
from .regime_scan import OOF, block_ci, penalty
from ..tune_v3 import save

PATH = EXP_DIR/"p91_median_transfer"/"metrics.json"
TARGETS = {"v6": ["p1_gfs60seed_fold1_static"], "v3_asof": ["p1_nbrc_asof_fold1_static"], "gfs20": ["p1_gfs20_fold1_static"],
           "official_baseline": ["p4_official_baseline_fold1_static"], "cb_blend": ["p1_blend_cb_development", "p1_blend_cb_selection"],
           "cb": ["p1_cb_development", "p1_cb_selection"]}


def main():
    PATH.parent.mkdir(parents=True, exist_ok=True)
    sites = load_sites().set_index("site_id")
    w = load_smp("train").set_index("ts").w
    d = T.load_train_features()
    feats = d[["site_id", "ts", "cosz"]].copy()
    del d
    T._cache.clear()
    tr = group_hours(OOF, feats, sites)
    tr = tr[tr.ts < pd.Timestamp("2024-06-01 01:00", tz=tr.ts.dt.tz)].reset_index(drop=True)
    f = fit(tr, tr.ts.map(w).to_numpy(), "median")
    level = apply(tr, f).sum()/tr.forecast_kwh.sum()
    f = (f[0], {k: v/level for k, v in f[1].items()})
    result = dict(rule="14.91", level_divisor=round(float(level), 4), factors=[round(float(x), 4) for x in f[1].values()], models={})
    rng = np.random.default_rng(2026)
    for name, runs in TARGETS.items():
        g = group_hours(runs, feats, sites)
        wg = g.ts.map(w).to_numpy()
        base = penalty(g, g.forecast_kwh, wg).to_numpy()
        cand = penalty(g, apply(g, f), wg).to_numpy()
        result["models"][name] = dict(period=f"{g.ts.min():%Y-%m-%d}~{g.ts.max():%Y-%m-%d}", group_hours=len(g),
                                      change_pct=round(100*(cand.sum()/base.sum()-1), 3), ci90=block_ci(g, base, cand, rng),
                                      total_ratio=round(float(apply(g, f).sum()/g.forecast_kwh.sum()), 4))
    result["complete"] = True
    save(PATH, result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        frames = [dict(function=f.name, line=f.lineno) for f in traceback.extract_tb(exc.__traceback__)
                  if "/src/" in f.filename]
        print(json.dumps(dict(status="failed", error_type=type(exc).__name__, code_frames=frames)), flush=True)
        raise SystemExit(1) from None
