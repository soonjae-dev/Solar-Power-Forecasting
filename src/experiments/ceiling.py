"""날씨를 미리 안다면 점수가 어디까지 오르는지 잰다(제출용 아님, 상한 진단용).

같은 설정(v4: fullc, 청천 지수 타깃, 정지일 제외, 시드 1개)으로 두 모델을 학습한다.
  ref    : 제출과 같은 입력(전일 예보 + 발전소 메타)
  oracle : 위 입력 + 대상 시각의 실측 기상(관측소 값). 평가 구간에는 없는 정보라 실제로는 쓸 수 없다.
oracle의 페널티는 '하루전 예보가 관측만큼 정확해진다면'의 근사 하한이다. 관측소 날씨도 발전소 상공의
구름을 다 담지 못하므로 진짜 하한은 이보다 낮을 수 있지만, 예보로는 관측보다 정확할 수 없다.

    python -m src.experiments.ceiling [fold ...]
"""
import json
import sys

import numpy as np
import pandas as pd

from .. import train as T
from ..config import EXP_DIR, TRAIN_DIR, TZ
from ..data import load_sites

OBS = {"wa": ["wa_cloud", "wa_temp", "wa_humidity", "wa_rain", "wa_snow"],
       "wb": ["wb_cloud", "wb_uv_idx", "wb_vis", "wb_ceiling", "wb_rel_hum", "wb_precip_1h", "wb_temp"]}
OFFICIAL = json.loads((EXP_DIR / "p4_official_baseline" / "metrics.json").read_text())
CFG = dict(feats="fullc", target="kt", outage="drop", weight="cap", train_start="2021-01-01")   # v4와 같은 설정(시드 1개)


def observed(provider: str) -> pd.DataFrame:
    st = f"{provider}_station"
    o = pd.read_parquet(TRAIN_DIR / f"observed_{provider}.parquet", columns=[st, "ts", *OBS[provider]])
    o["ts"] = pd.to_datetime(o["ts"]).dt.tz_convert(TZ)
    o = o[o["ts"] == o["ts"].dt.floor("h")].drop_duplicates([st, "ts"])
    for c in OBS[provider]:                                    # 이상값(운량 > 100, 9999 등)은 결측으로
        x = o[c].astype("float32")
        hi = 100 if c.endswith(("cloud", "hum", "humidity")) else 5000
        o[c] = x.where(x.between(-50, hi))
    return o.rename(columns={c: f"obs_{c}" for c in OBS[provider]})


def with_observed(d: pd.DataFrame) -> pd.DataFrame:
    sites = load_sites().set_index("site_id")
    out = d
    for p in ("wa", "wb"):
        st = f"{p}_station"
        out = out.assign(**{st: out["site_id"].map(sites[st])}).merge(observed(p), on=[st, "ts"], how="left").drop(columns=st)
    return out


def main(folds) -> None:
    base = T.load_train_features()
    result = {}
    for fold in folds:
        off = OFFICIAL[fold]["scores"]["official"]["penalty"]
        res = {}
        for name, d in (("ref", base), ("oracle", None)):
            if name == "oracle":
                d = with_observed(base)
                print(f"  실측 결합: 결측률 " + ", ".join(f"{c} {d[c].isna().mean():.1%}" for c in d.columns if c.startswith("obs_")), flush=True)
            T._cache["train"] = d
            m = T.run(f"ceiling_{name}", fold=fold, **CFG)
            res[name] = dict(penalty=m["penalty_ref"], lower_bound=m["penalty_lb"],
                             normalized=round(100 * (1 - m["penalty_ref"] / off), 2))
            del d
        T._cache["train"] = base
        res["official_penalty"] = off
        result[fold] = res
        print(f"{fold}: 정규화 점수 ref {res['ref']['normalized']:+.1f}  oracle {res['oracle']['normalized']:+.1f}", flush=True)
    out = EXP_DIR / "explain"
    out.mkdir(parents=True, exist_ok=True)
    (out / "ceiling.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"저장: {out / 'ceiling.json'}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["development"])
