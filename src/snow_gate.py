"""후보 gfs60_snowgate: 예보상 최근 72시간 강설이 있는 발전소-시각만 눈 모델 예측, 나머지는 기준 예측.

    python -m src.snow_gate fold1_static
결과는 PRED_DIR/p1_gfs60_snowgate_asof_<fold>/predictions.parquet. 재학습 없음.
"""
import sys

import pandas as pd

from .config import DATA_ROOT, PRED_DIR

GATE = ["wa_snow_sum72", "wb_snow_sum72"]


def main(fold, cand="snowacc", out_name="snowgate"):
    base = pd.read_parquet(PRED_DIR / f"p1_gfs60_asof_{fold}" / "predictions.parquet")
    snow = pd.read_parquet(PRED_DIR / f"p1_gfs60_{cand}_asof_{fold}" / "predictions.parquet")
    feat = pd.read_parquet(DATA_ROOT / f"cache_asof_{fold}" / "train_features.parquet", columns=["site_id", "ts"] + GATE)
    key = ["site_id", "ts"]
    snow = snow.set_index(key).loc[pd.MultiIndex.from_frame(base[key])].reset_index()   # 같은 행 순서
    gate = base[key].merge(feat, on=key, how="left", validate="one_to_one")
    on = ((gate[GATE].fillna(0) > 0).any(axis=1)).to_numpy()
    out = base.copy()
    for c in [c for c in base.columns if c.startswith("forecast_kwh")]:
        out.loc[on, c] = snow.loc[on, c].to_numpy()
    assert (out["energy"].to_numpy() == snow["energy"].to_numpy()).all()
    d = PRED_DIR / f"p1_gfs60_{out_name}_asof_{fold}"
    d.mkdir(exist_ok=True)
    out.to_parquet(d / "predictions.parquet", index=False)
    print(f"{fold}: 눈 모델 적용 행 {on.mean():.1%} ({on.sum():,} / {len(on):,})")


if __name__ == "__main__":
    main(*sys.argv[1:4])
