"""시간순 OOF v3 잔차로 전체 발전량을 보정한다. 행 출력·외부 저장 없음.

    .venv/bin/python -m src.correct_total

기존 nbrc_asof 3시드 예측만 읽고 2025-05는 사용하지 않는다.
"""
import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import CACHE_DIR, EXP_DIR, PRED_DIR
from .data import load_generation, load_sites, load_smp
from .diagnose_season import run_id, FOLDS
from .splits import fold

HISTORY = ("autumn_2022", "janfeb_2023", "spring_2023", "summer_2023",
           "autumn_2023", "janfeb_2024", "spring_2024")
WEATHER = ["wa_cloud", "wb_cloud", "wb_uv_idx", "wa_humidity", "wb_rel_hum", "cosz"]
PARAMS = dict(objective="quantile", alpha=0.5, n_estimators=150,
              learning_rate=0.03, num_leaves=7, max_depth=3,
              min_child_samples=150, reg_lambda=10, verbosity=-1,
              n_jobs=4, random_state=2026, deterministic=True, force_col_wise=True)


def allocate(base, delta):
    """공통 배율로 분배: 비음수, 집합 간 비율 유지, 야간 0 유지."""
    total = base.sum(axis=1)
    target = np.maximum(total + delta, 0)
    factor = np.divide(target, total, out=np.ones_like(total), where=total > 0)
    return base * factor[:, None]


def table(name, weather, truth, sites, smp):
    f = fold(name)
    if f["val_end"] > pd.Timestamp("2025-05-01", tz="Asia/Seoul"):
        raise ValueError("5월 holdout 사용 금지")
    path = PRED_DIR / run_id("nbrc_asof", name) / "predictions.parquet"
    if not path.exists():
        raise ValueError(f"필요한 OOF 예측 없음: {name}")
    # 저장된 energy는 음수 절단되었을 수 있어 채점 실적은 원본을 재결합한다.
    p = pd.read_parquet(path, columns=["site_id", "ts", "forecast_kwh"])
    if p.duplicated(["site_id", "ts"]).any():
        raise ValueError("중복 예측 키")
    rows = len(p)
    p = p.merge(truth, on=["site_id", "ts"], validate="one_to_one")
    if len(p) != rows:
        raise ValueError("예측에 대응하는 실적 누락")
    p["group"] = p["site_id"].map(sites["current_group"])
    p["cap"] = p["site_id"].map(sites["capacity_kw"])
    g = p.groupby(["ts", "group"])[["forecast_kwh", "energy"]].sum().unstack("group")
    base = g["forecast_kwh"].reindex(columns=["A", "B"])
    act = g["energy"].reindex(columns=["A", "B"])
    if base.isna().any().any() or act.isna().any().any():
        raise ValueError("집합 누락")
    x = p[["site_id", "ts", "cap"]].merge(weather, on=["site_id", "ts"], how="left", validate="one_to_one")
    cap = p.groupby("ts")["cap"].sum().reindex(base.index)
    features = pd.DataFrame(index=base.index)
    for c in WEATHER:
        valid_cap = x["cap"].where(x[c].notna()).groupby(x["ts"]).sum()
        features[c] = (x[c] * x["cap"]).groupby(x["ts"]).sum() / valid_cap.replace(0, np.nan)
        if c in ("wa_cloud", "wb_cloud", "wb_uv_idx"):
            features[c + "_std"] = x.groupby("ts")[c].std()
    features["cloud_disagreement"] = features["wa_cloud"] - features["wb_cloud"]
    features["base_cf"] = base.sum(axis=1) / cap
    features["hour"] = base.index.hour
    features["doy_sin"] = np.sin(2 * np.pi * base.index.dayofyear / 365.25)
    features["doy_cos"] = np.cos(2 * np.pi * base.index.dayofyear / 365.25)
    w = smp.reindex(base.index)["w"].to_numpy()
    if not np.isfinite(w).all():
        raise ValueError("SMP 누락")
    return dict(X=features, base=base.to_numpy(), act=act.to_numpy(), cap=cap.to_numpy(),
                y=((act.sum(axis=1) - base.sum(axis=1)) / cap).to_numpy(), w=w,
                index=base.index, cutoff=f["cutoff"], end=f["val_end"])


def train(tables):
    X = pd.concat([t["X"] for t in tables])
    y = np.concatenate([t["y"] for t in tables])
    w = np.concatenate([t["w"] * t["cap"] for t in tables])
    day = np.concatenate([t["base"].sum(axis=1) > 0 for t in tables])
    return lgb.LGBMRegressor(**PARAMS).fit(X.loc[day], y[day], sample_weight=w[day] / w[day].mean())


def evaluate(t, model):
    delta = model.predict(t["X"]) * t["cap"]
    corrected = allocate(t["base"], delta)
    assert np.isfinite(corrected).all() and (corrected >= 0).all()
    losses = [0.1 * t["w"] * np.abs(a - t["act"]).sum(axis=1) for a in (t["base"], corrected)]
    daily = pd.DataFrame(dict(base=losses[0], corrected=losses[1]), index=t["index"])
    daily = daily.groupby((daily.index - pd.Timedelta(hours=1)).normalize()).sum()
    rng = np.random.default_rng(2026)
    samples = []
    # 이동 블록 부트스트랩(7일), 같은 날의 양 모델을 함께 재표집한다.
    for _ in range(1000):
        starts = rng.integers(0, max(1, len(daily) - 6), size=int(np.ceil(len(daily) / 7)))
        idx = np.concatenate([np.arange(s, min(s + 7, len(daily))) for s in starts])[:len(daily)]
        sums = daily.iloc[idx].sum()
        samples.append(100 * (sums["corrected"] / sums["base"] - 1))
    peak = t["index"].hour.isin(range(11, 16))
    result = dict(hours=len(t["index"]), base_penalty=round(float(losses[0].sum())),
                  corrected_penalty=round(float(losses[1].sum())),
                  change_pct=round(float(100 * (losses[1].sum() / losses[0].sum() - 1)), 3),
                  change_ci90_pct=np.quantile(samples, [.05, .95]).round(3).tolist(),
                  peak_change_pct=round(float(100 * (losses[1][peak].sum() / losses[0][peak].sum() - 1)), 3))
    return result


def main():
    print("전체 오차 보정: 시간순 OOF, 고정 설정, holdout 제외", flush=True)
    weather = pd.read_parquet(CACHE_DIR / "train_features.parquet", columns=["site_id", "ts", *WEATHER])
    truth = load_generation()[["site_id", "ts", "energy"]]
    sites = load_sites().set_index("site_id")
    smp = load_smp("train").set_index("ts")
    names = HISTORY + FOLDS + ("fold1_static",)
    tables = {}
    for name in names:
        tables[name] = table(name, weather, truth, sites, smp)
        print(f"집계 준비 {name}: {len(tables[name]['X'])}시각", flush=True)
    results = {}
    for name in FOLDS + ("fold1_static",):
        t = tables[name]
        eligible = [n for n in HISTORY + FOLDS if tables[n]["end"] <= t["cutoff"]]
        assert name not in eligible
        model = train([tables[n] for n in eligible])
        r = evaluate(t, model)
        r["calibration_folds"] = eligible
        results[name] = r
        print(f"{name}: 페널티 변화 {r['change_pct']:+.3f}% (90% 구간 {r['change_ci90_pct']})", flush=True)
    seasons = [results[n] for n in FOLDS]
    change = 100 * (sum(r["corrected_penalty"] for r in seasons) / sum(r["base_penalty"] for r in seasons) - 1)
    passed = change <= -3 and sum(r["change_pct"] < 0 for r in seasons) >= 3 and max(r["change_pct"] for r in seasons) <= 3
    out = EXP_DIR / "p11_total_correction"
    out.mkdir(parents=True, exist_ok=True)
    summary = dict(params=PARAMS, results=results, seasonal_change_pct=round(change, 3),
                   seasonal_screen_pass=passed, static_improves=results["fold1_static"]["change_pct"] < 0,
                   note="과거 부분 관측 집합 포함. 고정 모델 검증은 보정기도 2024-06 이전에 동결. 독립 holdout 아님")
    (out / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"계절 합계 {change:+.3f}%, 사전 선택 기준 통과={passed}", flush=True)


if __name__ == "__main__":
    main()
