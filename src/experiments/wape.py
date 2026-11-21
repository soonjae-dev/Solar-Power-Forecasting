"""집합별 시간당 가중 절대 오차율(Σ|예측−실적| ÷ Σ실적, 집합 실적이 있는 시각)을 계절별로 낸다. 집계만 출력한다.

    .venv/bin/python -m src.experiments.wape p4_official_baseline p1_nbrc_drop p1_gfs60

발표 장표 8의 "익숙한 척도"다. 배정은 참조(A→G1, B→G2).
"""
import sys

import pandas as pd

from ..config import PRED_DIR
from ..data import load_sites

SEASONS = ("summer_support", "autumn_support", "development", "selection")


def wape(prefix, season, group):
    p = pd.read_parquet(PRED_DIR / f"{prefix}_{season}" / "predictions.parquet", columns=["site_id", "ts", "energy", "forecast_kwh"])
    p = p.dropna(subset=["energy"]).assign(g=lambda x: x.site_id.map(group))
    agg = p.groupby(["g", "ts"])[["energy", "forecast_kwh"]].sum()
    agg = agg[agg.energy > 0]
    return 100 * (agg.forecast_kwh - agg.energy).abs().sum() / agg.energy.sum()


def main(*prefixes):
    group = load_sites().set_index("site_id")["current_group"]
    for prefix in prefixes:
        print(prefix, {s: round(wape(prefix, s, group), 1) for s in SEASONS}, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
