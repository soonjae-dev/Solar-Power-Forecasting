"""gfs60에 휴일(주말·공휴일) 1열을 더해 재학습 검증한다. 집계만 저장한다.

    .venv/bin/python -m src.experiments.offday_model
"""
import json
from datetime import date

import holidays
import pandas as pd

from .. import train as T
from ..config import CACHE_DIR, EXP_DIR
from ..data import load_generation, load_sites, load_smp
from ..gfs_more import load_more
from ..gfs_pca import three_seed_verdict, verdict
from ..gfs_v3 import checked_gfs
from ..tune_v3 import END, run_stage, save
from ..weather_teacher import CFG

CANDIDATES = {"offday": {}}
EXTRA = {date(2025, 6, 3)}                       # 제21대 대통령선거일. holidays 0.58에 없다


def offday(index):
    kr = set(holidays.KR(years=range(2021, 2027))) | EXTRA
    day = (index - pd.Timedelta(hours=1))
    flag = (day.dayofweek >= 5) | pd.Index(day.date).isin(kr)
    return pd.DataFrame({"offday": flag.astype("float32")}, index=index)


def main():
    print("gfs60+휴일 1열 재학습 검증, holdout 제외", flush=True)
    gfs, _ = checked_gfs()
    base = pd.concat([gfs, load_more()[0]], axis=1)
    common = pd.concat([base, offday(base.index)], axis=1)
    print("휴일 비율", round(float(common.offday.mean()), 4), flush=True)
    sites = load_sites().set_index("site_id")
    smp = load_smp("train").set_index("ts")
    gen = load_generation()
    gen = gen[gen.ts <= END]
    d = T.load_train_features()
    d = d.loc[d.ts <= END, ["site_id", "ts", "energy", *T.select_features(d, CFG["feats"])]].copy()
    T._cache.clear()
    out = EXP_DIR / "p69_offday"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "metrics.json"
    result = json.loads(path.read_text()) if path.exists() else dict(rule="14.66", baseline_config=CFG, candidates=CANDIDATES,
                                                                    cache_name=CACHE_DIR.name, control="gfs60")
    run_stage(d, gen, sites, smp, result, path, 1, list(CANDIDATES), variants=CANDIDATES, added=common, control_columns=list(base))
    result["verdict_vs_control"] = verdict(result["seeds1"]["offday"])
    save(path, result)
    print("판정", json.dumps(result["verdict_vs_control"], ensure_ascii=False), flush=True)
    if result["verdict_vs_control"]["passed"]:
        run_stage(d, gen, sites, smp, result, path, 3, list(CANDIDATES), variants=CANDIDATES, added=common, control_columns=list(base))
        ctrl = json.loads((EXP_DIR / "p26_gfs_more" / "metrics.json").read_text())["seeds3"]["gfs60"]
        result["verdict3_vs_control"] = three_seed_verdict(result["seeds3"]["offday"], ctrl)
        save(path, result)
        print("3시드 판정", json.dumps(result["verdict3_vs_control"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
