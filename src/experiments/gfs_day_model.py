"""v6+LDAPS에 GFS 하루 맥락 20열(영역별 일사 합·운량 평균)을 더해 검증한다. 집계만 저장한다."""
import json

import pandas as pd

from .. import train as T
from ..config import CACHE_DIR, EXP_DIR
from ..data import load_generation, load_sites, load_smp
from ..gfs_more import load_more
from ..gfs_pca import verdict
from ..gfs_v3 import checked_gfs
from ..ldaps_model import load_ldaps
from ..tune_v3 import END, run_stage, save
from ..weather_teacher import CFG

CANDIDATES = {"gfsday": {}}


def day_context(gfs):
    """같은 대상일(ts−1시간의 날짜) 안의 영역별 GFS 일사 합과 운량 평균을 각 시각에 편다."""
    day = (gfs.index - pd.Timedelta(hours=1)).normalize()
    out = {}
    for c in gfs.columns:
        if c.startswith("gfs_dswrf_"):
            out[c.replace("gfs_dswrf_", "gfs_dswrf_day_")] = gfs[c].clip(lower=0).groupby(day).transform("sum")
        elif c.startswith("gfs_tcdc_"):
            out[c.replace("gfs_tcdc_", "gfs_tcdc_day_")] = gfs[c].groupby(day).transform("mean")
    return pd.DataFrame(out, index=gfs.index).astype("float32")


def main():
    gfs, _ = checked_gfs()
    base = pd.concat([gfs, load_more()[0]], axis=1)
    ld = load_ldaps(last=f"{END - pd.Timedelta(hours=1):%Y-%m-%d}")[0].reindex(base.index)
    dc = day_context(gfs)
    if dc.shape[1] != 20:
        raise ValueError("일 맥락 열 수 불일치")
    control = pd.concat([base, ld], axis=1)
    common = pd.concat([control, dc], axis=1)
    sites = load_sites().set_index("site_id"); smp = load_smp("train").set_index("ts")
    gen = load_generation(); gen = gen[gen.ts <= END]
    d = T.load_train_features()
    d = d.loc[d.ts <= END, ["site_id", "ts", "energy", *T.select_features(d, CFG["feats"])]].copy()
    T._cache.clear()
    out = EXP_DIR / "p79_gfs_day"; out.mkdir(parents=True, exist_ok=True)
    path = out / "metrics.json"
    result = json.loads(path.read_text()) if path.exists() else dict(rule="14.79", control="gfs60+ldaps", cache_name=CACHE_DIR.name)
    run_stage(d, gen, sites, smp, result, path, 1, list(CANDIDATES), variants=CANDIDATES, added=common, control_columns=list(control))
    result["verdict_vs_control"] = verdict(result["seeds1"]["gfsday"])
    save(path, result)
    print("판정", json.dumps(result["verdict_vs_control"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
