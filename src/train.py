"""캐시된 특징으로 한 fold를 학습·채점한다. 발전소별 예측을 저장한다.

    python -m src.train NAME [key=value ...]
    예) python -m src.train l1_site fold=development objective=l1 weight=cap

채점은 참조 배정의 페널티(ref)와 배정과 무관한 하한(lb)을 함께 낸다.
"""
import json
import sys
import time

import lightgbm as lgb
from catboost import CatBoostRegressor
import numpy as np
import pandas as pd

from .config import CACHE_DIR, EXP_DIR, PRED_DIR
from .data import load_generation, load_sites, load_smp
from .features import EXTRA, EXTRA_GROUPS, feature_names, with_nbrc_asof
from .scoring import composition_penalty, lower_bound, residual_matrix
from .splits import fold as get_fold

BASELINE_FEATS = ["wa_cloud", "wa_temp", "wa_humidity", "wa_rain", "wa_wind_speed", "wa_ground_press",
                  "wb_cloud", "wb_temp", "wb_rel_hum", "wb_uv_idx", "wb_precip_prob", "wb_total_liq",
                  "wb_vis", "wb_wind_speed", "wb_dew_point",
                  "elev", "azim", "cosz", "doy_sin", "doy_cos", "hour", "cap", "lat", "lon"]
DEFAULTS = dict(fold="development", feats="full", objective="l2", alpha=0.5, weight="none", target="cf",
                train_start="2023-06-01", n_estimators=600, learning_rate=0.06, num_leaves=127,
                min_child_samples=200, seed=2026, model="lgbm", depth=8,   # model=cb: CatBoost(n_estimators, learning_rate, depth)
                outage="keep",                                             # outage=drop: 정지 의심 발전소-일을 학습에서 뺀다
                nbrc_asof="no",                                            # nbrc_asof=fold: 이웃을 구간 예측시점까지의 예보로 고른다
                kt_weight="cs")                                            # kt_weight=cs2: 청천 지수 타깃의 가중치에 청천 일사 제곱을 쓴다

_cache = {}


def check_options(cfg: dict) -> None:
    if cfg["target"] not in ("cf", "kt"):
        raise ValueError(f"알 수 없는 target: {cfg['target']!r}")
    if cfg["nbrc_asof"] not in ("no", "fold"):
        raise ValueError(f"알 수 없는 nbrc_asof: {cfg['nbrc_asof']!r}")
    if cfg["kt_weight"] not in ("cs", "cs2"):
        raise ValueError(f"알 수 없는 kt_weight: {cfg['kt_weight']!r}")


def fold_features(d: pd.DataFrame, cfg: dict, cutoff: pd.Timestamp) -> pd.DataFrame:
    if cfg["nbrc_asof"] == "no":
        return d
    return with_nbrc_asof(d, cutoff + pd.Timedelta(hours=1))   # 예측시점 시각의 예보까지 포함한다


def load_train_features() -> pd.DataFrame:
    if "train" not in _cache:
        d = pd.read_parquet(CACHE_DIR / "train_features.parquet")
        d["energy"] = d["energy"].clip(lower=0)                # 음수 40건(-1, -0.1, -0.0025)은 0으로 본다
        _cache["train"] = d
    return _cache["train"]


def select_features(d: pd.DataFrame, feats: str) -> list:
    if feats == "baseline":
        return BASELINE_FEATS
    if feats == "baseline_site":
        return BASELINE_FEATS + ["site"]
    base, *groups = feats.split("+")                           # 예: "full+var+spa"
    if base not in ("full", "full_nosite", "fullc") or any(g not in EXTRA_GROUPS for g in groups):
        raise ValueError(f"알 수 없는 feats: {feats!r}")
    extra = {c for g in groups + (["nbrc"] if base == "fullc" else []) for c in EXTRA_GROUPS[g]}
    names = [c for c in feature_names(d) if c not in EXTRA or c in extra]
    if base == "fullc":                                        # 인접 관측소 평균을 좌표 기반(_nbr)에서 예보 상관 기반(_nbrc)으로
        names = [c for c in names if not c.endswith("_nbr")]
    return [c for c in names if c != "site"] if base == "full_nosite" else names


def sample_weight(d: pd.DataFrame, kind: str, smp: pd.Series):
    """타깃이 이용률이므로 kWh 오차 단위로 맞추려면 용량을 곱해야 한다."""
    if kind not in ("none", "cap", "smp", "capsmp", "capk", "capsmpk"):
        raise ValueError(f"알 수 없는 weight: {kind!r}")
    if kind == "none":
        return None
    w = np.ones(len(d))
    if "cap" in kind:
        w = w * d["cap"].to_numpy()
    if "smp" in kind:
        w = w * smp.reindex(d["ts"]).to_numpy()
    if kind in ("capk", "capsmpk"):                            # 정산의 시간대 가중치 k. 시각만으로 정해져 평가 구간에서도 같다
        w = w * np.where(d["ts"].dt.hour.between(11, 15), 3.0, 1.0)
    return w / w.mean()


OUTAGE_REF_MIN = 0.3      # 기준(같은 A 관측소 발전소들의 중앙값) 일 청천 지수가 이 값 이상인 날만 판정한다
OUTAGE_RATIO = 0.1        # 그날 발전소 청천 지수가 기준의 10% 미만이면 정지 의심


def outage_mask(d: pd.DataFrame) -> np.ndarray:
    """학습 행 중 '정지 의심 발전소-일'에 속하는 행. 그날 같은 A 관측소를 쓰는 발전소들은 정상인데
    이 발전소만 거의 발전하지 않은 날이다(그 관측소 발전소가 3곳 미만이면 전체 발전소를 기준으로 쓴다).
    같은 날의 학습 실적만 쓰므로 학습 구간 밖 정보는 들어가지 않는다. 검증·평가의 정답은 건드리지 않는다."""
    sites = load_sites().set_index("site_id")["wa_station"]
    day = (d["ts"] - pd.Timedelta(hours=1)).dt.normalize()
    cs = d["cap"].to_numpy() * clear_sky(d["cosz"])
    sd = (pd.DataFrame({"site_id": d["site_id"].to_numpy(), "day": day.to_numpy(), "e": d["energy"].to_numpy(), "cs": cs})
            .groupby(["site_id", "day"], sort=False)[["e", "cs"]].sum())
    sd["kt"] = sd["e"] / sd["cs"]
    sd["st"] = sd.index.get_level_values("site_id").map(sites)
    g = sd.groupby(["st", sd.index.get_level_values("day")])["kt"]
    n = g.transform("size")
    local = g.transform("median")                              # 발전소가 3곳 이상이면 자기 자신이 섞여도 중앙값은 정상 쪽에 머문다
    by_day = sd.groupby(level="day")
    fleet = by_day["e"].transform("sum") / by_day["cs"].transform("sum")
    ref = local.where(n >= 3, fleet)
    flag = (ref >= OUTAGE_REF_MIN) & (sd["kt"] < OUTAGE_RATIO * ref)
    bad = pd.MultiIndex.from_tuples(flag[flag].index, names=["site_id", "day"]) if flag.any() else None
    if bad is None:
        return np.zeros(len(d), dtype=bool)
    return pd.MultiIndex.from_arrays([d["site_id"].to_numpy(), day.to_numpy()]).isin(bad)


def drop_outage(train: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    if cfg["outage"] == "keep":
        return train
    if cfg["outage"] != "drop":
        raise ValueError(f"알 수 없는 outage: {cfg['outage']!r}")
    m = outage_mask(train)
    print(f"  정지 의심 발전소-일 제외: 학습 {len(train):,}행 중 {int(m.sum()):,}행", flush=True)
    return train[~m]


def clear_sky(cosz) -> np.ndarray:
    """Haurwitz 청천 수평면 일사(kW/m2). 타깃 정규화용이라 절대값의 정확도는 중요하지 않다."""
    c = np.clip(np.asarray(cosz, dtype="float64"), 0.02, None)
    return 1.098 * c * np.exp(-0.057 / c)


def kt_weight(cs, sw, cfg: dict):
    """청천 지수 타깃의 표본 가중치. cs2면 L2에서 kWh 제곱오차에 비례한다(kWh 오차 = cap × cs × 청천 지수 오차)."""
    w = cs if cfg["kt_weight"] == "cs" else cs ** 2
    return w / w.mean() if sw is None else sw * w / w.mean()


def make_model(cfg: dict):
    if cfg["model"] == "cb":
        return CatBoostRegressor(iterations=cfg["n_estimators"], learning_rate=cfg["learning_rate"], depth=cfg["depth"],
                                 loss_function="RMSE", random_seed=cfg["seed"], thread_count=4, verbose=False,
                                 allow_writing_files=False)
    if cfg["model"] != "lgbm":
        raise ValueError(f"알 수 없는 model: {cfg['model']!r}")
    objective = {"l2": "regression", "l1": "regression_l1", "huber": "huber",
                 "quantile": "quantile"}[cfg["objective"]]
    extra = {"alpha": cfg["alpha"]} if cfg["objective"] in ("quantile", "huber") else {}
    return lgb.LGBMRegressor(
        objective=objective, n_estimators=cfg["n_estimators"], learning_rate=cfg["learning_rate"],
        num_leaves=cfg["num_leaves"], min_child_samples=cfg["min_child_samples"],
        subsample=0.8, subsample_freq=1, colsample_bytree=cfg.get("colsample_bytree", 0.8), verbosity=-1, n_jobs=4,
        random_state=cfg["seed"], deterministic=True, force_col_wise=True, **extra)


def fit_model(cfg: dict, X: pd.DataFrame, y, sw):
    cat = ["site"] if "site" in X.columns else None
    if cfg["model"] == "cb":
        return make_model(cfg).fit(X, y, sample_weight=sw, cat_features=cat)
    return make_model(cfg).fit(X, y, sample_weight=sw, categorical_feature=cat or "auto")


def night_filled(pred_day: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """주간 예측을 (발전소, 시각) 전체 격자로 넓힌다. 특징이 없는 야간은 0이다."""
    out = truth.merge(pred_day[["site_id", "ts", "forecast_kwh"]], on=["site_id", "ts"], how="left")
    out["forecast_kwh"] = out["forecast_kwh"].fillna(0.0)
    return out


def run(name: str, **overrides) -> dict:
    unknown = set(overrides) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"알 수 없는 설정: {unknown}")
    cfg = {**DEFAULTS, **overrides}
    check_options(cfg)
    t0 = time.perf_counter()
    f = get_fold(cfg["fold"])
    d = fold_features(load_train_features(), cfg, f["cutoff"])
    feats = select_features(d, cfg["feats"])
    smp = load_smp("train").set_index("ts")["w"]
    train = d[(d["ts"] >= pd.Timestamp(cfg["train_start"] + " 01:00", tz="Asia/Seoul")) & (d["ts"] <= f["cutoff"])]
    val = d[d["ts"].between(f["val_start"], f["val_end"])]
    train = drop_outage(train, cfg)
    y = (train["energy"] / train["cap"]).clip(0, 1.2)
    sw = sample_weight(train, cfg["weight"], smp)
    scale = 1.0
    if cfg["target"] == "kt":                                  # 청천 지수. 가중치에 청천 일사를 곱해 kWh 오차 단위를 유지한다
        cs = clear_sky(train["cosz"])
        y = (y / cs).clip(0, 2.0)
        sw = kt_weight(cs, sw, cfg)
        scale = clear_sky(val["cosz"])
    model = fit_model(cfg, train[feats], y, sw)
    pred = val[["site_id", "ts"]].copy()
    pred["forecast_kwh"] = np.clip(model.predict(val[feats]) * scale, 0, None) * val["cap"].to_numpy()

    if "gen" not in _cache:
        _cache["gen"] = load_generation()                      # 채점용 실적은 원본 그대로
    gen = _cache["gen"]
    pred = night_filled(pred, gen[gen["ts"].between(f["val_start"], f["val_end"])])
    E = residual_matrix(pred)
    w = smp.reindex(E.index).to_numpy()
    sites = load_sites().set_index("site_id").loc[E.columns]
    ref = composition_penalty(E.to_numpy(), w, (sites["current_group"] == "A").to_numpy())
    lb = lower_bound(E.to_numpy(), w)
    bias = float(w @ E.sum(axis=1).to_numpy()) / float(w.sum())
    metrics = dict(name=name, **cfg, cache=CACHE_DIR.name, n_feats=len(feats), train_rows=len(train),
                   penalty_ref=round(ref), penalty_lb=round(lb), weighted_bias_kwh=round(bias, 1),
                   seconds=round(time.perf_counter() - t0, 1))
    run_id = f"{name}_{cfg['fold']}"
    out = EXP_DIR / run_id
    out.mkdir(parents=True, exist_ok=True)
    (PRED_DIR / run_id).mkdir(parents=True, exist_ok=True)
    pred.to_parquet(PRED_DIR / run_id / "predictions.parquet", index=False)
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"{name:<28}{cfg['fold']:<13} ref {ref:>12,.0f}  lb {lb:>12,.0f}  bias {bias:>8.1f}  "
          f"{metrics['seconds']:>5.0f}s", flush=True)
    return metrics


def _parse(v: str):
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


if __name__ == "__main__":
    run(sys.argv[1], **{k: _parse(v) for k, v in (a.split("=", 1) for a in sys.argv[2:])})
