"""v3 입력에 공개 GFS 20열을 더해 재학습 검증한다. 집계만 저장한다.

    .venv/bin/python -m src.gfs_v3

판정 기준은 실행 전에 고정했다(docs/validation.md). 2025-05 holdout은 쓰지 않는다.
"""
import json

from . import train as T
from .config import CACHE_DIR, EXP_DIR
from .data import load_generation, load_sites, load_smp
from .gfs_pilot import load_gfs
from .tune_v3 import END, run_stage, save
from .weather_teacher import CFG

CANDIDATES = {"gfs20": {}}
FIRST_DAY, LAST_DAY = "2021-01-02", "2025-04-30"


def checked_gfs():
    gfs, missing = load_gfs(FIRST_DAY, LAST_DAY)
    if gfs.shape[1] != 20 or not gfs.index.is_unique or gfs.index.max() > END:
        raise ValueError("GFS 열·시각 조건 위반")
    return gfs, missing


def main():
    print("v3+GFS 20열 재학습 검증, holdout 제외", flush=True)
    gfs, missing = checked_gfs()
    print(f"GFS {len(gfs)}시각, 빠진 날 {len(missing)}", flush=True)
    sites = load_sites().set_index("site_id")
    smp = load_smp("train").set_index("ts")
    gen = load_generation()
    gen = gen[gen.ts <= END]
    d = T.load_train_features()
    cols = ["site_id", "ts", "energy", *T.select_features(d, CFG["feats"])]
    d = d.loc[d.ts <= END, cols].copy()
    T._cache.clear()
    out = EXP_DIR / "p24_gfs_v3"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "metrics.json"
    if path.exists():
        result = json.loads(path.read_text())
        if result["baseline_config"] != CFG or result["candidates"] != CANDIDATES or result["cache_name"] != CACHE_DIR.name:
            raise ValueError("체크포인트 설정이 현재 계획과 다름")
    else:
        result = dict(baseline_config=CFG, candidates=CANDIDATES, cache_name=CACHE_DIR.name, gfs_hours=len(gfs), missing_days=len(missing))
    eligible = run_stage(d, gen, sites, smp, result, path, 1, list(CANDIDATES), variants=CANDIDATES, added=gfs)
    if eligible:
        finalists = run_stage(d, gen, sites, smp, result, path, 3, eligible, variants=CANDIDATES, added=gfs)
        result["selected_candidate"] = finalists[0] if finalists else None
    else:
        result["selected_candidate"] = None
    save(path, result)
    print("최종 후보", result["selected_candidate"], flush=True)


if __name__ == "__main__":
    main()
