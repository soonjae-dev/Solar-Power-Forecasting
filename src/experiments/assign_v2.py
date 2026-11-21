"""배정 재탐색. 신규 개소 포함 전체 자유, 탐색·검증 구간 분리. 집계만 저장한다."""
import json
import sys

import numpy as np
import pandas as pd

from .assign_gfs60 import block_ci, hourly
from ..compose import local_search
from ..config import EXP_DIR, MIN_GROUP_KW, PRED_DIR, SUB_DIR
from ..data import load_sites, load_smp
from ..headroom import random_feasible
from ..scoring import composition_penalty

PATH = EXP_DIR / "p88_assign_v2" / "metrics.json"
NAMES = [f"p1_gfs60seed_oof6m_{k}" for k in range(1, 6)] + ["p1_gfs60seed_fold1_static"]


def residual_frame(sites):
    p = pd.concat([pd.read_parquet(PRED_DIR / n / "predictions.parquet", columns=["site_id", "ts", "energy", "forecast_kwh"]) for n in NAMES])
    if p.duplicated(["site_id", "ts"]).any():
        raise ValueError("구간 중복")
    E = p.assign(e=p.forecast_kwh - p.energy).pivot(index="ts", columns="site_id", values="e").reindex(columns=sites.index)
    return E.fillna(0.0)


def search(E, w, cap, ref, nolimit, rng, restarts=8):
    starts = [ref, nolimit]
    while len(starts) < restarts + 2:
        g = random_feasible(cap, rng)
        if MIN_GROUP_KW <= cap[g].sum() <= cap.sum() - MIN_GROUP_KW:
            starts.append(g)
    found = [local_search(E, w, cap, g, rng) for g in starts]
    return min(found, key=lambda r: r[1])[0]


def main(final=False):
    sites = load_sites().set_index("site_id")
    cap = sites.capacity_kw.to_numpy()
    ref = (sites.current_group == "A").to_numpy()
    nolimit = pd.read_csv(SUB_DIR / "composition_oof_gfs60_nolimit.csv").set_index("site_id").group.reindex(sites.index).eq("G1").to_numpy()
    wser = load_smp("train").set_index("ts")["w"]
    Edf = residual_frame(sites)
    ts = Edf.index
    w = wser.reindex(ts).to_numpy()
    E = Edf.to_numpy()
    rng = np.random.default_rng(2026)
    result = json.loads(PATH.read_text()) if PATH.exists() else dict(rule="14.88")
    if not final:
        day = (ts - pd.Timedelta(hours=1))
        val = np.asarray(day >= pd.Timestamp("2024-12-01", tz=ts.tz))
        g = search(E[~val], w[~val], cap, ref, nolimit, rng)
        bv = composition_penalty(E[val], w[val], nolimit); cv = composition_penalty(E[val], w[val], g)
        base = hourly(E[val], w[val], nolimit)
        ci = block_ci(hourly(E[val], w[val], g) - base, ts[val], base, rng)
        result["validation"] = dict(moved_vs_nolimit=int((g != nolimit).sum()), g1_kw=round(float(cap[g].sum()), 1),
                                    nolimit=round(bv), candidate=round(cv), change_pct=round(100 * (cv / bv - 1), 3), block95_pct=ci,
                                    search_change_vs_nolimit_pct=round(100 * (composition_penalty(E[~val], w[~val], g) / composition_penalty(E[~val], w[~val], nolimit) - 1), 3))
        r = result["validation"]
        result["passed"] = bool(r["change_pct"] <= -0.5 and ci[1] < 0)
    else:
        g = search(E, w, cap, ref, nolimit, rng)
        c1 = float(cap[g].sum())
        if not MIN_GROUP_KW <= c1 <= cap.sum() - MIN_GROUP_KW:
            raise ValueError("용량 제약 위반")
        pd.DataFrame({"site_id": sites.index, "group": np.where(g, "G1", "G2")}).to_csv(SUB_DIR / "composition_assign_v2.csv", index=False)
        result["final"] = dict(moved_vs_nolimit=int((g != nolimit).sum()), moved_vs_ref=int((g != ref).sum()), g1_kw=round(c1, 1), g2_kw=round(float(cap.sum() - c1), 1),
                               all_change_vs_nolimit_pct=round(100 * (composition_penalty(E, w, g) / composition_penalty(E, w, nolimit) - 1), 3))
    PATH.parent.mkdir(parents=True, exist_ok=True)
    PATH.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main(final=sys.argv[1:] == ["final"])
