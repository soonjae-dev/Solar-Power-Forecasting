"""gfs60_snowgate 제출 파일: 발전소-시각마다 예보상 지난 72시간 강설이 있으면 눈 모델 예측, 없으면 기본(v6) 예측.

    python -m src.snowgate_submission NAME BASE SNOW COMPOSITION_CSV
    예: python -m src.snowgate_submission v6_gfs60_snowgate_nolimit v6_gfs60_ref_submit v6_gfs60_snowacc_ref \
            $OIBC_DATA_ROOT/submissions/composition_oof_gfs60_nolimit.csv

BASE·SNOW는 make_submission이 만든 SUB_DIR/<이름>/site_forecast_daylight.parquet. 게이트 열은 CACHE_DIR의 test_features.parquet
(features.py의 snow 묶음). 규칙은 src/snow_gate.py와 같고 결과를 보고 바꾸지 않는다. 합산·저장은 make_submission과 같다.
"""
import sys

import pandas as pd

from .config import CACHE_DIR, SUB_DIR
from .data import load_smp
from .validate_submission import validate

GATE = ["wa_snow_sum72", "wb_snow_sum72"]
KEY = ["site_id", "ts"]


def main(name, base, snow, composition):
    b = pd.read_parquet(SUB_DIR / base / "site_forecast_daylight.parquet")
    s = pd.read_parquet(SUB_DIR / snow / "site_forecast_daylight.parquet").set_index(KEY)["kwh"]
    gate = pd.read_parquet(CACHE_DIR / "test_features.parquet", columns=KEY + GATE).set_index(KEY)
    idx = pd.MultiIndex.from_frame(b[KEY])
    assert len(s) == len(b) and s.index.isin(idx).all() and gate.index.isin(idx).all() and len(gate) == len(b), "두 예측·게이트의 행이 다르다"
    on = (gate.reindex(idx)[GATE].fillna(0) > 0).any(axis=1).to_numpy()
    test = b.copy()
    test.loc[on, "kwh"] = s.reindex(idx).to_numpy()[on]
    print(f"눈 모델 적용 행 {on.mean():.1%} ({on.sum():,} / {len(on):,}), 연간 합 변화 {(test['kwh'].sum() / b['kwh'].sum() - 1) * 100:+.2f}%")

    comp = pd.read_csv(composition)
    out = SUB_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    test.to_parquet(out / "site_forecast_daylight.parquet", index=False)
    day = (test.assign(group=test["site_id"].map(comp.set_index("site_id")["group"]))
               .groupby(["group", "ts"])["kwh"].sum())
    grid = pd.MultiIndex.from_product([["G1", "G2"], load_smp("test")["ts"]], names=["group", "ts"])
    fc = day.reindex(grid, fill_value=0.0).rename("forecast_kwh").reset_index()   # 야간은 0
    fc["ts"] = fc["ts"].map(lambda t: t.isoformat(sep=" "))
    fc.to_csv(out / "forecast.csv", index=False)
    comp.sort_values("site_id").to_csv(out / "composition.csv", index=False)
    validate(out / "composition.csv", out / "forecast.csv")
    print(f"저장: {out}")


if __name__ == "__main__":
    main(*sys.argv[1:5])
