"""후보 D: C에 snowgate를 얹는다. D = C + 0.5·(snowgate − v6), 발전소·시각 kWh. 집계만 출력.

    .venv/bin/python -m src.make_blend_snow

snowgate 발전소별 예측은 다시 만든 결과를 먼저 쓰고, 없으면 팀원 묶음을 풀지 않고 메모리로 읽는다. 게이트 밖 행이 v6와 같아야 진행한다.
"""
import io
import json
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_ROOT, SUB_DIR
from .data import load_smp
from .validate_submission import validate

# 팀원의 적설 전용 모델(snowgate) 발전소별 예측. scripts/snowgate_run.sh로 다시 만든 결과(SNOW_DIR)를 먼저 쓰고,
# 없으면 당시 받은 묶음(OIBC_SNOWGATE_ZIP, 기본 $OIBC_DATA_ROOT/external/)에서 읽는다(README의 "재현 범위" 참고).
SNOW_DIR = SUB_DIR / "v6_gfs60_snowgate_nolimit"
ZIP = Path(os.environ.get("OIBC_SNOWGATE_ZIP", DATA_ROOT / "external" / "oibc_snowgate_20261108.zip"))
SNOW = "oibc_snowgate_20261108/submission/v6_gfs60_snowgate_nolimit/site_forecast_daylight.parquet"
C, V6, NAME = "v8_gfs60_cbblend_nolimit", "repro_v6_gfs60", "v9_gfs60_cbblend_snowgate_nolimit"


def load_snowgate() -> pd.DataFrame:
    if (SNOW_DIR / "site_forecast_daylight.parquet").exists():
        return pd.read_parquet(SNOW_DIR / "site_forecast_daylight.parquet")
    if ZIP.exists():
        return pd.read_parquet(io.BytesIO(zipfile.ZipFile(ZIP).read(SNOW)))
    raise SystemExit(f"snowgate 예측이 없다. scripts/snowgate_run.sh로 {SNOW_DIR}를 만들거나 OIBC_SNOWGATE_ZIP으로 묶음을 지정한다.")


def main():
    keys = ["site_id", "ts"]
    c = pd.read_parquet(SUB_DIR/C/"site_forecast_daylight.parquet")
    v6 = pd.read_parquet(SUB_DIR/V6/"site_forecast_daylight.parquet")
    snow = load_snowgate()
    if list(snow.columns) != list(v6.columns):
        raise ValueError(f"열 불일치: {list(snow.columns)}")
    for f in (v6, snow):
        f["ts"] = pd.to_datetime(f["ts"])
    p = c.merge(v6, on=keys, suffixes=("", "_v6"), validate="one_to_one").merge(
        snow.rename(columns={"kwh": "kwh_snow"}), on=keys, validate="one_to_one")
    if len(p) != len(c) or len(p) != len(v6) or len(p) != len(snow):
        raise ValueError("행 불일치")
    gated = ~np.isclose(p.kwh_snow, p.kwh_v6, rtol=0, atol=1e-9)
    print(json.dumps(dict(rows=len(p), gated_share=round(float(gated.mean()), 4),
                          snow_minus_v6_gwh=round(float((p.kwh_snow-p.kwh_v6).sum()/1e6), 4)), ensure_ascii=False), flush=True)
    if not .02 < gated.mean() < .1:
        raise ValueError("게이트 비율이 전달 기록(약 5.6%)과 다르다")
    p["kwh"] = np.clip(p.kwh + .5*(p.kwh_snow-p.kwh_v6), 0, None)
    out = SUB_DIR/NAME
    out.mkdir(parents=True, exist_ok=True)
    p[keys+["kwh"]].to_parquet(out/"site_forecast_daylight.parquet", index=False)
    comp = pd.read_csv(SUB_DIR/C/"composition.csv")
    grid = pd.MultiIndex.from_product([["G1", "G2"], load_smp("test")["ts"]], names=["group", "ts"])
    fc = (p.assign(group=p.site_id.map(comp.set_index("site_id").group)).groupby(["group", "ts"]).kwh.sum()
          .reindex(grid, fill_value=0.0).rename("forecast_kwh").reset_index())
    fc["ts"] = fc["ts"].map(lambda t: t.isoformat(sep=" "))
    fc.to_csv(out/"forecast.csv", index=False)
    comp.to_csv(out/"composition.csv", index=False)
    validate(out/"composition.csv", out/"forecast.csv")
    (out/"manifest.json").write_text(json.dumps(dict(rule="14.65 후보 D", formula="C + 0.5*(snowgate - v6)", c=C, v6=V6,
                                                     snowgate=SNOW, gated_share=round(float(gated.mean()), 4)), ensure_ascii=False, indent=1))
    print("저장", out.name, flush=True)


if __name__ == "__main__":
    main()
