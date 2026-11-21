"""배포 baseline.py를 검증 구간에서 그대로 재현해, 서버 정규화 점수의 기대 범위를 잡는다.

    python -m src.official_baseline fit       # 네 구간에서 baseline.py 방식으로 학습·예측해 PRED_DIR에 저장
    python -m src.official_baseline report    # 채택 설정과 W 비율·기대 정규화 점수를 출력하고 experiments/p4_official_baseline/metrics.json에 저장

서버 0점은 배포 baseline.py다(설명회 슬라이드 20). 정규화 점수 = 100 × (1 − W팀 / W기준선).
노트북 17-A 재현(src/experiments/baseline_repro.py, 17시 단일 모델)과 달리 라운드 10·17 모델의 평균이다.
특징 함수·특징 목록·태양 계산은 baseline.py 파일에서 그대로 가져오고, 모델 설정은 baseline.py main()과 같다.
baseline.py가 평가 구간에서 쓰는 학습 범위(대상 첫날 2년 전 01시 ~ 첫날 00시)를 각 구간에 같은 간격으로 옮긴다.
행은 출력하지 않는다.
"""
import importlib.util
import json
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import EXP_DIR, PRED_DIR, ROOT
from .data import load_forecast, load_generation, load_sites, load_smp
from .diagnose_season import FOLDS, run_id, score
from .splits import fold as get_fold

_spec = importlib.util.spec_from_file_location("official_baseline_py", ROOT / "handover 2" / "01_participant_dataset" / "baseline.py")
B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B)

RUN = "p4_official_baseline"


def model() -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(n_estimators=600, learning_rate=0.06, num_leaves=127,
                             min_child_samples=200, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.8, verbose=-1, n_jobs=-1)


def fit_predict(fold_name: str, complete_only: bool = True) -> None:
    f = get_fold(fold_name)
    first_day = f["val_start"] - pd.Timedelta(hours=1)
    train_from = first_day - pd.DateOffset(years=2) + pd.Timedelta(hours=1)
    sites = load_sites()[["site_id", "current_group", "capacity_kw", "lat", "lon", "wa_station", "wb_station"]]
    gen = load_generation()
    train_gen = gen[(gen["ts"] >= train_from) & (gen["ts"] <= first_day)]
    fa = load_forecast("train", "wa", B.WA_COLS, start=train_from, end=f["val_end"])
    fb = load_forecast("train", "wb", B.WB_COLS, start=train_from, end=f["val_end"])
    fa_tr, fb_tr = fa[fa["ts"] <= first_day], fb[fb["ts"] <= first_day]

    truth = gen[gen["ts"].between(f["val_start"], f["val_end"])]
    hours = truth["ts"].nunique()
    n = truth.groupby("site_id")["ts"].size()
    complete = n.index[n == hours] if complete_only else n.index       # diagnose_season과 같은 발전소 집합
    index = pd.DatetimeIndex(sorted(truth["ts"].unique()))
    scope = sites[sites["site_id"].isin(complete)]

    preds = []
    for r in B.ROUNDS:
        d = B.features(train_gen, sites, fa_tr, fb_tr, None, r)
        d = d[d.elev > 0]
        y = (d.energy / d.cap).clip(0, 1.2).astype("float32")
        m = model().fit(d[B.FEATS].astype("float32"), y)
        n_train = len(d)
        del d, y
        v = B.features(None, scope, fa, fb, index, r)
        p = np.where(v.elev.to_numpy() > 0, m.predict(v[B.FEATS].astype("float32")), 0.0)
        v["kwh"] = np.clip(p, 0, None) * v.cap.to_numpy()
        preds.append(v[["site_id", "ts", "kwh"]])
        print(f"  {fold_name:<15} round={r} 학습 {n_train:>9,}행", flush=True)
    pred = (pd.concat(preds).groupby(["site_id", "ts"], as_index=False)["kwh"].mean()
              .rename(columns={"kwh": "forecast_kwh"}))
    out = truth[truth["site_id"].isin(complete)].merge(pred, on=["site_id", "ts"], how="left")
    if out["forecast_kwh"].isna().any():
        raise ValueError("예측이 빠진 (발전소, 시각)이 있다")
    out["energy"] = out["energy"].clip(lower=0)
    (PRED_DIR / f"{RUN}_{fold_name}").mkdir(parents=True, exist_ok=True)
    out.to_parquet(PRED_DIR / f"{RUN}_{fold_name}" / "predictions.parquet", index=False)
    print(f"{fold_name:<15} 학습 {train_from:%Y-%m-%d %H시} ~ {first_day:%Y-%m-%d %H시}  발전소 {len(complete)}  시각 {hours}", flush=True)


def panel(path, sites: pd.Series, smp: pd.DataFrame) -> pd.DataFrame:
    p = pd.read_parquet(path, columns=["site_id", "ts", "energy", "forecast_kwh"])
    p["energy"] = p["energy"].clip(lower=0)
    p["err"] = p["forecast_kwh"] - p["energy"]
    p["group"] = p["site_id"].map(sites)
    return p.merge(smp[["ts", "smp", "k", "w"]], on="ts", how="left")


def report() -> None:
    sites = load_sites().set_index("site_id")["current_group"]
    smp = load_smp("train")
    result = {}
    print("구간별 W(참조 배정 페널티) 비교. 정규화 점수 = 100 × (1 − W채택 / W공식기준선)")
    for fold_name in FOLDS:
        panels = {
            "official": panel(PRED_DIR / f"{RUN}_{fold_name}" / "predictions.parquet", sites, smp),
            "notebook_17A": panel(PRED_DIR / run_id("baseline", fold_name) / "predictions.parquet", sites, smp),
            "adopted": panel(PRED_DIR / run_id("adopted", fold_name) / "predictions.parquet", sites, smp),
        }
        keys = None
        for p in panels.values():                                      # 세 예측이 같은 (발전소, 시각)을 채점하게 한다
            k = pd.MultiIndex.from_frame(p[["site_id", "ts"]])
            keys = k if keys is None else keys.intersection(k)
        sizes = {c: len(p) for c, p in panels.items()}
        panels = {c: p.set_index(["site_id", "ts"]).loc[keys].reset_index() for c, p in panels.items()}
        s = {c: score(p) for c, p in panels.items()}
        w_off, w_nb, w_ad = (s[c]["penalty"] for c in ("official", "notebook_17A", "adopted"))
        res = dict(sites=int(panels["adopted"]["site_id"].nunique()), rows=len(keys), rows_before=sizes,
                   scores=s, normalized_adopted=round(100 * (1 - w_ad / w_off), 2),
                   normalized_notebook_17A=round(100 * (1 - w_nb / w_off), 2))
        result[fold_name] = res
        print(f"  {fold_name:<15} 발전소 {res['sites']}  공식 기준선 {s['official']['pen_pct']:.3f}%  노트북 17-A {s['notebook_17A']['pen_pct']:.3f}%  "
              f"채택 {s['adopted']['pen_pct']:.3f}%  → 채택 정규화 {res['normalized_adopted']:+.1f}점 (17-A {res['normalized_notebook_17A']:+.1f}점)  "
              f"예측/실적 공식 {s['official']['bias']:.3f}", flush=True)
    vals = [r["normalized_adopted"] for r in result.values()]
    result["summary"] = dict(normalized_min=min(vals), normalized_max=max(vals), normalized_mean=round(float(np.mean(vals)), 2))
    print(f"채택 설정의 기대 정규화 점수: 구간별 {min(vals):+.1f} ~ {max(vals):+.1f}점, 평균 {np.mean(vals):+.1f}점")
    out = EXP_DIR / RUN
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    print(f"저장: {out / 'metrics.json'}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "fit":
        for name in (sys.argv[2:] or FOLDS):
            fit_predict(name)
    elif cmd == "report":
        report()
    else:
        raise SystemExit(__doc__)
