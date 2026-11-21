"""추운 조건 과대예측 보정의 연도 블록 교차검증. 재학습 없음, 집계만 저장.

    MPLCONFIGDIR=/tmp/oibc-mpl .venv/bin/python -m src.experiments.cold_cv
"""
import json
import traceback

import numpy as np
import pandas as pd

from .. import train as T
from ..config import EXP_DIR
from ..data import load_sites, load_smp
from .regime_scan import OOF, SHRINK, STATIC, apply, factors, group_hours, penalty
from ..tune_v3 import save

PATH = EXP_DIR/"p66_cold_cv"/"metrics.json"
VAR = "wa_temp"
BLOCKS = [("2022-01-01", "2022-06-01"), ("2022-06-01", "2023-06-01"), ("2023-06-01", "2024-06-01"), ("2024-06-01", "2025-05-01")]


def main():
    PATH.parent.mkdir(parents=True, exist_ok=True)
    sites = load_sites().set_index("site_id")
    w = load_smp("train").set_index("ts").w
    d = T.load_train_features()
    d["dew_spread"] = d.wb_temp-d.wb_dew_point
    from .regime_scan import VARS
    feats = d[["site_id", "ts", *VARS]].copy()
    del d
    T._cache.clear()
    g = pd.concat([group_hours(OOF, feats, sites), group_hours([STATIC], feats, sites)], ignore_index=True)
    day = g.ts-pd.Timedelta(hours=1)
    block = np.full(len(g), -1)
    for k, (a, b) in enumerate(BLOCKS):
        block[((day >= pd.Timestamp(a, tz=day.dt.tz)) & (day < pd.Timestamp(b, tz=day.dt.tz))).to_numpy()] = k
    if (block < 0).any():
        raise ValueError("블록 밖 시각")
    wg = g.ts.map(w).to_numpy()
    base = penalty(g, g.forecast_kwh, wg).to_numpy()
    rule, const = np.zeros(len(g)), np.zeros(len(g))
    result = dict(rule="14.62", variable=VAR, shrink=SHRINK, blocks={})
    for k in range(len(BLOCKS)):
        tr, te = g[block != k].reset_index(drop=True), g[block == k].reset_index(drop=True)
        fit = factors(tr, VAR)
        c = tr.energy.sum()/tr.forecast_kwh.sum()
        m = block == k
        rule[m] = penalty(te, apply(te, VAR, fit), wg[m]).to_numpy()
        const[m] = penalty(te, te.forecast_kwh*c, wg[m]).to_numpy()
        b, r, q = base[m].sum(), rule[m].sum(), const[m].sum()
        result["blocks"][f"{BLOCKS[k][0]}~{BLOCKS[k][1]}"] = dict(
            group_hours=int(m.sum()), base=int(round(float(b))), rule_change_pct=round(100*(r/b-1), 3),
            const_change_pct=round(100*(q/b-1), 3), const_factor=round(float(c), 4),
            factors=[round(float(x), 4) for x in fit[1].values()])
    seen = np.arange(len(BLOCKS)) < 3
    b, r, q = base.sum(), rule.sum(), const.sum()
    b3, r3 = base[np.isin(block, np.where(seen)[0])].sum(), rule[np.isin(block, np.where(seen)[0])].sum()
    result.update(total_vs_v6_pct=round(100*(r/b-1), 3), total_vs_const_pct=round(100*(r/q-1), 3),
                  const_vs_v6_pct=round(100*(q/b-1), 3), blocks1_3_vs_v6_pct=round(100*(r3/b3-1), 3),
                  improved_blocks=int(sum(v["rule_change_pct"] < 0 for v in result["blocks"].values())))
    result["passed"] = bool(result["total_vs_v6_pct"] <= -1 and result["total_vs_const_pct"] <= -.5
                            and result["improved_blocks"] >= 3 and result["blocks1_3_vs_v6_pct"] <= -1)
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
