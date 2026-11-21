"""후보 13 손해 원인 진단: 기준 수준에 따른 중앙값 배율 효과. 집계만 저장.

    MPLCONFIGDIR=/tmp/oibc-mpl .venv/bin/python -m src.experiments.median_level
"""
import json
import traceback

import numpy as np
import pandas as pd

from .. import train as T
from .cold_cv import BLOCKS
from ..config import EXP_DIR
from ..data import load_sites, load_smp
from .median_scale import apply, fit, group_hours
from .regime_scan import OOF, STATIC, penalty
from ..tune_v3 import save

PATH = EXP_DIR/"p92_median_level"/"metrics.json"
LEVELS = np.round(np.arange(.96, 1.101, .01), 2)
SERVER_DOWN = 1.82        # 8번 ×0.984의 서버 페널티 변화(%), −251만 원 / 약 1.378억 원


def main():
    PATH.parent.mkdir(parents=True, exist_ok=True)
    sites = load_sites().set_index("site_id")
    w = load_smp("train").set_index("ts").w
    d = T.load_train_features()
    feats = d[["site_id", "ts", "cosz"]].copy()
    del d
    T._cache.clear()
    g = pd.concat([group_hours(OOF, feats, sites), group_hours([STATIC], feats, sites)], ignore_index=True)
    day = g.ts-pd.Timedelta(hours=1)
    block = np.full(len(g), -1)
    for k, (a, b) in enumerate(BLOCKS):
        block[((day >= pd.Timestamp(a, tz=day.dt.tz)) & (day < pd.Timestamp(b, tz=day.dt.tz))).to_numpy()] = k
    wg = g.ts.map(w).to_numpy()
    result = dict(rule="14.92", server_down_pct=SERVER_DOWN, blocks={})
    for k in range(len(BLOCKS)):
        m = block == k
        tr = g[~m].reset_index(drop=True)
        f = fit(tr, wg[~m], "median")
        s = apply(tr, f).sum()/tr.forecast_kwh.sum()
        f = (f[0], {j: v/s for j, v in f[1].items()})
        te0, wt = g[m].reset_index(drop=True), wg[m]
        rows = []
        for c in LEVELS:
            te = te0.assign(forecast_kwh=te0.forecast_kwh*c, q=te0.q*c)
            b = penalty(te, te.forecast_kwh, wt).sum()
            rows.append(dict(c=float(c), down_pct=round(100*(penalty(te, te.forecast_kwh*.984, wt).sum()/b-1), 3),
                             rule_pct=round(100*(penalty(te, apply(te, f), wt).sum()/b-1), 3)))
        down = np.array([r["down_pct"] for r in rows])
        rule = np.array([r["rule_pct"] for r in rows])
        o = np.argsort(down)
        at = float(np.interp(SERVER_DOWN, down[o], rule[o])) if down.min() <= SERVER_DOWN <= down.max() else None
        c_at = float(np.interp(SERVER_DOWN, down[o], LEVELS[o])) if at is not None else None
        result["blocks"][f"{BLOCKS[k][0]}~{BLOCKS[k][1]}"] = dict(
            level_ratio=round(float(te0.energy.sum()/te0.forecast_kwh.sum()), 4), curve=rows,
            server_like_c=None if c_at is None else round(c_at, 3), rule_at_server_like_pct=None if at is None else round(at, 3))
    at = [v["rule_at_server_like_pct"] for v in result["blocks"].values()]
    result["worse_blocks_at_server_like"] = int(sum(x is not None and x > 0 for x in at))
    result["supported"] = result["worse_blocks_at_server_like"] >= 3
    result["complete"] = True
    save(PATH, result)
    print(json.dumps({k: v for k, v in result.items() if k != "blocks"}, ensure_ascii=False))
    for name, v in result["blocks"].items():
        print(name, v["level_ratio"], v["server_like_c"], v["rule_at_server_like_pct"],
              [(r["c"], r["down_pct"], r["rule_pct"]) for r in v["curve"]], flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        frames = [dict(function=f.name, line=f.lineno) for f in traceback.extract_tb(exc.__traceback__)
                  if "/src/" in f.filename]
        print(json.dumps(dict(status="failed", error_type=type(exc).__name__, code_frames=frames)), flush=True)
        raise SystemExit(1) from None
