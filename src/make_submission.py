"""학습 구간 전체로 다시 학습해 평가 구간을 예측하고 제출 파일을 만든다.

    python -m src.make_submission NAME COMPOSITION_CSV [seeds=3] [common=gfs20|gfs60|gfs60aod|gfs60spr] [blend_cb=1] [key=value ...]

COMPOSITION_CSV가 'reference'이면 sites.csv의 참조 배정(A→G1, B→G2)을 쓴다.
blend_cb=1이면 같은 학습행으로 CatBoost(p53 설정) 같은 수의 시드를 학습해 발전소·시각 kWh를 0.5:0.5로 섞는다(후보 C).
이때 LightGBM만의 예측은 참조 배정 forecast_lgbm_reference.csv로도 남겨 v6 재현을 확인한다.
"""
import json
import sys

import numpy as np
import pandas as pd

from .config import CACHE_DIR, FIRST_TEST_CUTOFF, SUB_DIR, TZ
from .data import load_sites, load_smp
from .train import DEFAULTS, _parse, check_options, clear_sky, drop_outage, fit_model, kt_weight, load_train_features, sample_weight, select_features
from .validate_submission import validate


def main(name: str, composition: str, seeds: int = 3, common: str = "no", blend_cb: int = 0, **overrides) -> None:
    cfg = {**DEFAULTS, **overrides}
    check_options(cfg)
    if cfg["nbrc_asof"] != "no":
        raise ValueError("제출 모델의 이웃은 이미 평가 시작 전 예보로만 고른다. nbrc_asof는 검증 구간 전용이다")
    sites = load_sites()
    if composition == "reference":
        comp = pd.DataFrame({"site_id": sites["site_id"],
                             "group": sites["current_group"].map({"A": "G1", "B": "G2"})})
    else:
        comp = pd.read_csv(composition)

    train = load_train_features()
    train = train[(train["ts"] >= pd.Timestamp(cfg["train_start"] + " 01:00", tz=TZ))
                  & (train["ts"] <= pd.Timestamp(FIRST_TEST_CUTOFF, tz=TZ))]   # 예측시점 뒤의 실적은 쓰지 않는다
    train = drop_outage(train, cfg)
    test = pd.read_parquet(CACHE_DIR / "test_features.parquet")
    feats = select_features(train, cfg["feats"])
    if common != "no":                                         # 모든 발전소에 같은 외부 열. 전일 17시 전 공개분만 받아 둔다
        if common not in ("gfs20", "gfs60", "gfs60aod", "gfs60spr"):
            raise ValueError(f"알 수 없는 common: {common!r}")
        from .gfs_pilot import load_gfs
        gfs, _ = load_gfs("2021-01-02", "2026-05-31")
        if common in ("gfs60", "gfs60aod", "gfs60spr"):
            from .gfs_more import load_more
            gfs = pd.concat([gfs, load_more("2021-01-02", "2026-05-31")[0]], axis=1)
        if common == "gfs60aod":
            from .gfs_aod import load_aod
            gfs = pd.concat([gfs, load_aod("2021-01-02", "2026-05-31")[0]], axis=1)
        if common == "gfs60spr":
            from .gfs_spread import load_spread
            gfs = pd.concat([gfs, load_spread("2021-01-02", "2026-05-31")[0]], axis=1)
        train = train.copy()
        train[list(gfs)] = gfs.reindex(train["ts"]).to_numpy()
        test[list(gfs)] = gfs.reindex(test["ts"]).to_numpy()
        feats = feats + list(gfs)
        day = test["ts"].dt.hour.between(4, 20)
        core = [c for c in gfs if not c.startswith("gefs_")]          # 에어로졸은 늦게 공개된 4일이 결측이라 검사에서 뺀다
        if test.loc[day, core].isna().any().any():
            raise ValueError("평가 구간 낮 시각에 GFS가 비어 있다")
    y = (train["energy"] / train["cap"]).clip(0, 1.2)
    sw = sample_weight(train, cfg["weight"], load_smp("train").set_index("ts")["w"])
    scale = 1.0
    if cfg["target"] == "kt":
        cs = clear_sky(train["cosz"])
        y, sw, scale = (y / cs).clip(0, 2.0), kt_weight(cs, sw, cfg), clear_sky(test["cosz"])
    ratio = np.zeros(len(test))
    for k in range(seeds):
        model = fit_model({**cfg, "seed": cfg["seed"] + k}, train[feats], y, sw)
        ratio += model.predict(test[feats]) / seeds
        print(f"seed {cfg['seed'] + k} 완료", flush=True)
    test["kwh"] = np.clip(ratio * scale, 0, None) * test["cap"].to_numpy()

    out = SUB_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    grid = pd.MultiIndex.from_product([["G1", "G2"], load_smp("test")["ts"]], names=["group", "ts"])
    if blend_cb:
        from .cb_onehot import checked_input, make_model
        ref = test["site_id"].map(sites.set_index("site_id")["current_group"].map({"A": "G1", "B": "G2"}))
        lg = test.assign(group=ref).groupby(["group", "ts"])["kwh"].sum().reindex(grid, fill_value=0.0).rename("forecast_kwh").reset_index()
        lg["ts"] = lg["ts"].map(lambda t: t.isoformat(sep=" "))
        lg.to_csv(out / "forecast_lgbm_reference.csv", index=False)
        X, V = checked_input(train[feats]), checked_input(test[feats])
        cb = np.zeros(len(test))
        for k in range(seeds):
            m = make_model(cfg["seed"] + k)
            m.fit(X, y, sample_weight=sw, cat_features=["site"])
            cb += m.predict(V) / seeds
            print(f"CatBoost seed {cfg['seed'] + k} 완료", flush=True)
        test["kwh"] = 0.5 * test["kwh"] + 0.5 * np.clip(cb * scale, 0, None) * test["cap"].to_numpy()
    test[["site_id", "ts", "kwh"]].to_parquet(out / "site_forecast_daylight.parquet", index=False)
    day = (test.assign(group=test["site_id"].map(comp.set_index("site_id")["group"]))
               .groupby(["group", "ts"])["kwh"].sum())
    fc = day.reindex(grid, fill_value=0.0).rename("forecast_kwh").reset_index()   # 야간은 0
    fc["ts"] = fc["ts"].map(lambda t: t.isoformat(sep=" "))
    fc.to_csv(out / "forecast.csv", index=False)
    comp.sort_values("site_id").to_csv(out / "composition.csv", index=False)
    validate(out / "composition.csv", out / "forecast.csv")
    (out / "manifest.json").write_text(json.dumps(
        dict(config=cfg, common=common, blend_cb=blend_cb, cache=CACHE_DIR.name, seeds=seeds, composition=composition, train_cutoff=FIRST_TEST_CUTOFF,
             train_rows=len(train), features=len(feats)), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"저장: {out}")


if __name__ == "__main__":
    kw = {k: _parse(v) for k, v in (a.split("=", 1) for a in sys.argv[3:])}
    main(sys.argv[1], sys.argv[2], **kw)
