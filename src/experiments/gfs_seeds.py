"""gfs60을 10시드로 평균해 3시드와 비교한다. 집계만 저장한다.

    .venv/bin/python -m src.experiments.gfs_seeds

판정 기준은 실행 전에 고정했다(docs/validation.md). 2025-05 holdout은 쓰지 않는다.
"""
import json

import pandas as pd

from .. import train as T
from ..config import CACHE_DIR, EXP_DIR
from ..data import load_generation, load_sites, load_smp
from ..gfs_more import load_more
from ..gfs_pca import three_seed_verdict
from ..gfs_v3 import checked_gfs
from ..tune_v3 import END, run_stage, save
from ..weather_teacher import CFG

CANDIDATES = {"gfs60_s10": {}}
SEEDS = 10


def main():
    print("gfs60 10시드 평균, holdout 제외", flush=True)
    gfs, _ = checked_gfs()
    common = pd.concat([gfs, load_more()[0]], axis=1)
    sites = load_sites().set_index("site_id")
    smp = load_smp("train").set_index("ts")
    gen = load_generation()
    gen = gen[gen.ts <= END]
    d = T.load_train_features()
    cols = ["site_id", "ts", "energy", *T.select_features(d, CFG["feats"])]
    d = d.loc[d.ts <= END, cols].copy()
    T._cache.clear()
    out = EXP_DIR / "p31_gfs_seeds"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "metrics.json"
    result = json.loads(path.read_text()) if path.exists() else dict(baseline_config=CFG, candidates=CANDIDATES, cache_name=CACHE_DIR.name, control="gfs60 3시드")
    run_stage(d, gen, sites, smp, result, path, SEEDS, list(CANDIDATES), variants=CANDIDATES, added=common)
    ctrl = json.loads((EXP_DIR / "p26_gfs_more" / "metrics.json").read_text())["seeds3"]["gfs60"]
    v = three_seed_verdict(result[f"seeds{SEEDS}"]["gfs60_s10"], ctrl)
    v["passed"] = bool(v["seasonal_change_pct"] <= 0 and v["changes_pct"]["fold1_static"] <= 0)   # 14.22 기준
    result["verdict_vs_gfs60_3seeds"] = v
    save(path, result)
    print("판정", json.dumps(v, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
