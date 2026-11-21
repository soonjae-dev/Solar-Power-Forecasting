"""채택된 GFS 열에 전전날 18UTC 실행 20열과 두 실행의 차이 20열을 더해 재학습 검증한다. 집계만 저장한다.

    .venv/bin/python -m src.experiments.gfs_lag

판정 기준은 실행 전에 고정했다(docs/validation.md). 2025-05 holdout은 쓰지 않는다.
"""
import json

import pandas as pd

from .. import train as T
from ..config import CACHE_DIR, EXP_DIR
from ..data import load_generation, load_sites, load_smp
from ..gfs_fetch import LAG_DIR
from ..gfs_more import load_more
from ..gfs_pca import three_seed_verdict, verdict
from ..gfs_pilot import load_gfs
from ..gfs_v3 import FIRST_DAY, LAST_DAY, checked_gfs
from ..tune_v3 import END, run_stage, save
from ..weather_teacher import CFG

CANDIDATES = {"lag18": {}}


def adopted_control():
    """gfs60이 1시드와 3시드 판정을 모두 통과했으면 gfs60, 아니면 gfs20."""
    path = EXP_DIR / "p26_gfs_more" / "metrics.json"
    if path.exists():
        m = json.loads(path.read_text())
        if m.get("verdict_vs_gfs20", {}).get("passed") and m.get("verdict3_vs_gfs20", {}).get("passed"):
            return "gfs60"
    return "gfs20"


def lag_columns(gfs):
    lag, missing = load_gfs(FIRST_DAY, LAST_DAY, directory=LAG_DIR, prefix="lag18")
    lag = lag.reindex(gfs.index.union(lag.index))
    diff = gfs.reindex(lag.index).to_numpy() - lag.to_numpy()
    diff = pd.DataFrame(diff, index=lag.index, columns=[c.replace("lag18_", "runchange_") for c in lag.columns])
    if lag.index.max() > END:
        raise ValueError("시각 조건 위반")
    return pd.concat([lag, diff], axis=1).astype("float32"), missing


def main():
    print("채택 GFS 열+18UTC 이전 실행 재학습 검증, holdout 제외", flush=True)
    gfs, _ = checked_gfs()
    control = adopted_control()
    base = pd.concat([gfs, load_more()[0]], axis=1) if control == "gfs60" else gfs
    extra, missing = lag_columns(gfs)
    if extra.shape[1] != 40:
        raise ValueError("이전 실행 열 수 오류")
    print(f"대조군 {control}, 이전 실행 빠진 날 {len(missing)}", flush=True)
    common = pd.concat([base, extra], axis=1)
    sites = load_sites().set_index("site_id")
    smp = load_smp("train").set_index("ts")
    gen = load_generation()
    gen = gen[gen.ts <= END]
    d = T.load_train_features()
    cols = ["site_id", "ts", "energy", *T.select_features(d, CFG["feats"])]
    d = d.loc[d.ts <= END, cols].copy()
    T._cache.clear()
    out = EXP_DIR / "p29_gfs_lag"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "metrics.json"
    result = json.loads(path.read_text()) if path.exists() else dict(baseline_config=CFG, candidates=CANDIDATES, cache_name=CACHE_DIR.name, control=control)
    if result["control"] != control:
        raise ValueError("체크포인트의 대조군이 다르다")
    run_stage(d, gen, sites, smp, result, path, 1, list(CANDIDATES), variants=CANDIDATES, added=common, control_columns=list(base))
    result["verdict_vs_control"] = verdict(result["seeds1"]["lag18"])
    save(path, result)
    print("판정", json.dumps(result["verdict_vs_control"], ensure_ascii=False), flush=True)
    if result["verdict_vs_control"]["passed"]:
        run_stage(d, gen, sites, smp, result, path, 3, list(CANDIDATES), variants=CANDIDATES, added=common, control_columns=list(base))
        ctrl = json.loads((EXP_DIR / ("p26_gfs_more" if control == "gfs60" else "p24_gfs_v3") / "metrics.json").read_text())["seeds3"][control]
        result["verdict3_vs_control"] = three_seed_verdict(result["seeds3"]["lag18"], ctrl)
        save(path, result)
        print("3시드 판정", json.dumps(result["verdict3_vs_control"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
