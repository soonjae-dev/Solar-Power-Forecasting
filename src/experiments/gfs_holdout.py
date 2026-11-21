"""2025-05 확인 채점: gfs20을 v3와 같은 행에서 한 번 비교한다. 집계만 저장한다.

    .venv/bin/python -m src.experiments.gfs_holdout

판정 기준은 실행 전에 고정했다(docs/validation.md). 결과로 후보를 고르거나 설정을 바꾸지 않는다.
"""
import json
import sys

import numpy as np
import pandas as pd

from .. import train as T
from ..config import EXP_DIR
from ..data import load_generation, load_sites, load_smp
from ..direct_groups import metrics
from ..gfs_pilot import load_gfs
from ..provider_ensemble import grouped_prediction
from ..splits import fold
from ..tune_v3 import evaluation_context
from ..weather_teacher import CFG

NAME = "final_holdout"


def main(common="gfs20", seeds=3):
    f = fold(NAME)
    gfs, missing = load_gfs("2021-01-02", "2025-05-31")
    if common in ("gfs60", "gfs60aod", "gfs60spr"):
        from ..gfs_more import load_more
        gfs = pd.concat([gfs, load_more("2021-01-02", "2025-05-31")[0]], axis=1)
    if common == "gfs60aod":
        from ..gfs_aod import load_aod
        gfs = pd.concat([gfs, load_aod("2021-01-02", "2025-05-31")[0]], axis=1)
    if common == "gfs60spr":
        from ..gfs_spread import load_spread
        gfs = pd.concat([gfs, load_spread("2021-01-02", "2025-05-31")[0]], axis=1)
    sites = load_sites().set_index("site_id")
    smp = load_smp("train").set_index("ts")
    gen = load_generation()
    gen = gen[gen.ts <= f["val_end"]]
    d = T.load_train_features()
    cols = ["site_id", "ts", "energy", *T.select_features(d, CFG["feats"])]
    d = d.loc[d.ts <= f["val_end"], cols].copy()
    T._cache.clear()
    q = T.fold_features(d, CFG, f["cutoff"])
    feats = T.select_features(q, CFG["feats"])
    tr = q[q.ts <= f["cutoff"]].copy()
    val = q[q.ts.between(f["val_start"], f["val_end"])].copy()
    del q, d
    if tr.ts.max() > f["cutoff"]:
        raise ValueError("학습·평가 시점 분리 위반")
    for frame in (tr, val):
        frame[list(gfs)] = gfs.reindex(frame.ts).to_numpy()
    columns = feats + list(gfs)
    cs = T.clear_sky(tr.cosz)
    y = ((tr.energy / tr.cap).clip(0, 1.2) / cs).clip(0, 2).to_numpy()
    weight = T.kt_weight(cs, T.sample_weight(tr, "cap", smp.w), CFG)
    pred = np.zeros(len(val))
    for k in range(seeds):
        pred += T.fit_model({**CFG, "seed": 2026 + k}, tr[columns], y, weight).predict(val[columns]) / seeds
        print(f"seed {2026 + k} 완료", flush=True)
    pred = np.maximum(pred, 0) * T.clear_sky(val.cosz) * val.cap.to_numpy()
    ctx = evaluation_context(NAME, gen, sites)
    g = grouped_prediction(ctx["actual"], val[["site_id", "ts"]], pred).reindex(ctx["a"].index)
    r = dict(rule="14.19", common=common, seeds=seeds, train_cutoff=f["cutoff"].isoformat(), missing_gfs_days=len(missing),
             vs_v3_3seeds=metrics(ctx["a"], ctx["b3"], g, smp))
    r["investigate_before_submission"] = r["vs_v3_3seeds"]["change_pct"] > 10
    out = EXP_DIR / ("p27_gfs_holdout" if common == "gfs20" else f"p27_gfs_holdout_{common}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(r, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:2])
