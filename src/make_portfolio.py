"""제출 포트폴리오 4~7번을 만든다. 집계만 출력한다.

    .venv/bin/python -m src.make_portfolio train cbns|v6_10|cb_10   # 구성 요소 최종 학습(발전소 kWh 저장)
    .venv/bin/python -m src.make_portfolio build 4|5|6|7            # 후보 조립·검증·등록

학습 설정은 v6 제출과 같다(gfs60, fullc, cap, kt, 2021-01-01~2025-05-31 17시, nbrc_asof=no).
"""
import io
import json
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CACHE_DIR, DATA_ROOT, FIRST_TEST_CUTOFF, SUB_DIR, TZ
from .data import load_smp
from .train import DEFAULTS, check_options, clear_sky, drop_outage, fit_model, kt_weight, load_train_features, sample_weight, select_features
from .validate_submission import validate

COMP = SUB_DIR / "components"
COMPOSITION = SUB_DIR / "v8_gfs60_cbblend_nolimit" / "composition.csv"
# 팀원의 적설 전용 모델(snowgate) 발전소별 예측. scripts/snowgate_run.sh로 다시 만든 결과(SNOW_DIR)를 먼저 쓰고,
# 없으면 당시 받은 묶음(OIBC_SNOWGATE_ZIP, 기본 $OIBC_DATA_ROOT/external/)에서 읽는다(README의 "재현 범위" 참고).
SNOW_DIR = SUB_DIR / "v6_gfs60_snowgate_nolimit"
ZIP = Path(os.environ.get("OIBC_SNOWGATE_ZIP", DATA_ROOT / "external" / "oibc_snowgate_20261108.zip"))
SNOW = "oibc_snowgate_20261108/submission/v6_gfs60_snowgate_nolimit/site_forecast_daylight.parquet"
CFG = {**DEFAULTS, "feats": "fullc", "weight": "cap", "target": "kt", "train_start": "2021-01-01"}
NAMES = {"4": "p4_cb_snowgate_nolimit", "5": "p5_cbns_snowgate_nolimit", "6": "p6_v6cbcbns_snowgate_nolimit", "7": "p7_d_seed10_nolimit"}
KEYS = ["site_id", "ts"]


def load_snowgate() -> pd.DataFrame:
    if (SNOW_DIR / "site_forecast_daylight.parquet").exists():
        return pd.read_parquet(SNOW_DIR / "site_forecast_daylight.parquet")
    if ZIP.exists():
        return pd.read_parquet(io.BytesIO(zipfile.ZipFile(ZIP).read(SNOW)))
    raise SystemExit(f"snowgate 예측이 없다. scripts/snowgate_run.sh로 {SNOW_DIR}를 만들거나 OIBC_SNOWGATE_ZIP으로 묶음을 지정한다.")


def prepare(ldaps=False):
    check_options(CFG)
    train = load_train_features()
    train = train[(train["ts"] >= pd.Timestamp(CFG["train_start"] + " 01:00", tz=TZ)) & (train["ts"] <= pd.Timestamp(FIRST_TEST_CUTOFF, tz=TZ))]
    train = drop_outage(train, CFG)
    test = pd.read_parquet(CACHE_DIR / "test_features.parquet")
    feats = select_features(train, CFG["feats"])
    from .gfs_more import load_more
    from .gfs_pilot import load_gfs
    gfs = pd.concat([load_gfs("2021-01-02", "2026-05-31")[0], load_more("2021-01-02", "2026-05-31")[0]], axis=1)
    train = train.copy()
    train[list(gfs)] = gfs.reindex(train["ts"]).to_numpy()
    test[list(gfs)] = gfs.reindex(test["ts"]).to_numpy()
    if ldaps:                                                   # 14.72: LDAPS 10열. 2026-04 이후는 결측
        from .ldaps_model import load_ldaps
        gfs = pd.concat([gfs, load_ldaps("2021-01-02", "2026-03-31")[0].reindex(gfs.index)], axis=1)
        train[[c for c in gfs if c.startswith("ldaps_")]] = gfs.filter(like="ldaps_").reindex(train["ts"]).to_numpy()
        test[[c for c in gfs if c.startswith("ldaps_")]] = gfs.filter(like="ldaps_").reindex(test["ts"]).to_numpy()
    feats = feats + list(gfs)
    if len(feats) != (154 if ldaps else 144):
        raise ValueError("입력 수 불일치")
    y = (train["energy"] / train["cap"]).clip(0, 1.2)
    sw = sample_weight(train, CFG["weight"], load_smp("train").set_index("ts")["w"])
    cs = clear_sky(train["cosz"])
    return train, test, feats, (y / cs).clip(0, 2.0), kt_weight(cs, sw, CFG), clear_sky(test["cosz"])


def train_component(kind):
    from .cb_onehot import checked_input, make_model, numeric_input
    train, test, feats, y, sw, scale = prepare(ldaps=kind in ("v6l", "v6l_10", "cbl", "cbnsl"))
    seeds = 10 if kind.endswith("_10") else 3
    pred = np.zeros(len(test))
    for k in range(seeds):
        seed = CFG["seed"] + k
        if kind in ("v6_10", "v6l", "v6l_10"):
            pred += fit_model({**CFG, "seed": seed}, train[feats], y, sw).predict(test[feats]) / seeds
        elif kind in ("cb_10", "cbl"):
            m = make_model(seed)
            m.fit(checked_input(train[feats]), y, sample_weight=sw, cat_features=["site"])
            pred += m.predict(checked_input(test[feats])) / seeds
        elif kind in ("cbns", "cbnsl"):
            cols = [c for c in feats if c != "site"]
            m = make_model(seed)
            m.fit(numeric_input(train[cols]), y, sample_weight=sw, cat_features=[])
            pred += m.predict(numeric_input(test[cols])) / seeds
        else:
            raise ValueError(kind)
        print(f"{kind} seed {seed} 완료", flush=True)
    out = test[KEYS].copy()
    out["kwh"] = np.clip(pred * scale, 0, None) * test["cap"].to_numpy()
    COMP.mkdir(parents=True, exist_ok=True)
    out.to_parquet(COMP / f"{kind}.parquet", index=False)
    print(f"저장 {kind}: 연간 {out.kwh.sum() / 1e6:.3f} GWh", flush=True)


def components(*need):
    v6 = pd.read_parquet(SUB_DIR / "repro_v6_gfs60" / "site_forecast_daylight.parquet")
    c = pd.read_parquet(SUB_DIR / "v8_gfs60_cbblend_nolimit" / "site_forecast_daylight.parquet")
    sg = load_snowgate()
    sg["ts"] = pd.to_datetime(sg["ts"])
    p = v6.rename(columns={"kwh": "v6"}).merge(c.rename(columns={"kwh": "c"}), on=KEYS, validate="one_to_one") \
          .merge(sg.rename(columns={"kwh": "sg"}), on=KEYS, validate="one_to_one")
    p["cb"] = (2 * p.c - p.v6).clip(lower=0)
    for kind in need:
        x = pd.read_parquet(COMP / f"{kind}.parquet").rename(columns={"kwh": kind})
        p = p.merge(x, on=KEYS, validate="one_to_one")
    if len(p) != len(v6):
        raise ValueError("행 불일치")
    p["g"] = ~np.isclose(p.sg, p.v6, rtol=0, atol=1e-9)
    return p


def build(number):
    need = {"4": (), "5": ("cbns",), "6": ("cbns",), "7": ("v6_10", "cb_10")}[number]
    p = components(*need)
    if number == "4":
        kwh = np.where(p.g, p.sg, p.cb)
    elif number == "5":
        kwh = np.where(p.g, p.sg, p.cbns)
    elif number == "6":
        kwh = (p.sg + p.cb + p.cbns) / 3
    else:
        kwh = 0.5 * p.v6_10 + 0.5 * p.cb_10 + 0.5 * (p.sg - p.v6)
    p["kwh"] = np.clip(kwh, 0, None)
    name = NAMES[number]
    out = SUB_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    p[KEYS + ["kwh"]].to_parquet(out / "site_forecast_daylight.parquet", index=False)
    comp = pd.read_csv(COMPOSITION)
    grid = pd.MultiIndex.from_product([["G1", "G2"], load_smp("test")["ts"]], names=["group", "ts"])
    fc = (p.assign(group=p.site_id.map(comp.set_index("site_id").group)).groupby(["group", "ts"]).kwh.sum()
          .reindex(grid, fill_value=0.0).rename("forecast_kwh").reset_index())
    fc["ts"] = fc["ts"].map(lambda t: t.isoformat(sep=" "))
    fc.to_csv(out / "forecast.csv", index=False)
    comp.to_csv(out / "composition.csv", index=False)
    validate(out / "composition.csv", out / "forecast.csv")
    (out / "manifest.json").write_text(json.dumps(dict(rule="14.70", candidate=number, components=list(need),
                                                     gated_share=round(float(p.g.mean()), 4),
                                                     total_vs_v6=round(float(p.kwh.sum() / p.v6.sum()), 4)), ensure_ascii=False, indent=1))
    from .submission_log import register
    register(name, f"python -m src.make_portfolio build {number}", f"14.70 후보 {number}")
    print(json.dumps(dict(name=name, gated_share=round(float(p.g.mean()), 4), total_vs_v6=round(float(p.kwh.sum() / p.v6.sum()), 4)), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    {"train": train_component, "build": build}[sys.argv[1]](sys.argv[2])
