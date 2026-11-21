"""특징 행렬 생성과 캐시. 주간(구간 중앙 태양고도 > 0) 행만 만든다. 야간 예측은 0이다.

    python -m src.features            # CACHE_DIR의 train_features.parquet, test_features.parquet

쓰는 정보는 대상일 전일 10시·17시 발행 예보와 발전소 메타뿐이다.
실측 기상·발전 실적 lag는 평가 구간에 없으므로 쓰지 않는다.
"""
import os

import numpy as np
import pandas as pd

from .config import CACHE_DIR, TRAIN_DIR, TZ
from .data import forecast_columns, load_forecast, load_generation, load_sites, load_smp
from .solar import solar

# 하루전 예보에서 wa_temp_max·wa_temp_min은 wa_temp와 같은 값이라 뺀다 (README).
DROP = {"wa_temp_max", "wa_temp_min"}
R10_COLS = {"wa": ["wa_cloud", "wa_rain", "wa_humidity"],
            "wb": ["wb_cloud", "wb_uv_idx", "wb_precip_prob", "wb_total_liq", "wb_vis", "wb_ceiling"]}
NEIGHBOR_COLS = {"wa": ["wa_cloud"], "wb": ["wb_cloud", "wb_uv_idx"]}
DAILY = {"wa": {"wa_cloud": "mean", "wa_rain": "sum"},
         "wb": {"wb_cloud": "mean", "wb_uv_idx": "max", "wb_total_liq": "sum", "wb_precip_prob": "max"}}

# 구름 양상 특징(2026-10-13 추가). 기본 특징 집합("full")에는 넣지 않고 feats="full+var+..."로 골라 쓴다.
# 모두 같은 대상일(8~18시) 또는 같은 시각의 예보에서 만든다.
_REGIME = ("wa_cloud", "wb_cloud", "wb_uv_idx")
EXTRA_GROUPS = {
    "var": [f"{c}_daystd" for c in _REGIME] + [f"{c}_trend" for c in _REGIME]          # 하루 안의 변동
           + ["wa_cloud_daymin", "wa_cloud_daymax", "wb_cloud_daymin", "wb_cloud_daymax", "wb_uv_idx_daymean"],
    "ampm": [f"{c}_{h}" for c in _REGIME for h in ("am", "pm")],                       # 오전·오후
    "dis": [f"{c}_rdiff_dayabs" for c in _REGIME] + ["day_cloud_gap", "rain_gap", "cloud_absgap"],   # 발행 간·제공사 간 불일치
    "spa": [f"{c}_regstd" for c in _REGIME] + [f"{c}_regdev" for c in _REGIME],        # 공간 분산
    # 인접 관측소를 좌표 대신 예보 상관으로 고른 평균(2026-10-15 추가). feats="fullc"가 _nbr 대신 쓴다.
    "nbrc": [f"{c}_nbrc" for cols in NEIGHBOR_COLS.values() for c in cols],
}
EXTRA = {c for cols in EXTRA_GROUPS.values() for c in cols}


def neighbor_weights(sites: pd.DataFrame, st: str, stations: list, k: int = 4) -> np.ndarray:
    """관측소 좌표가 없어 연결된 발전소들의 평균 좌표로 대신하고, 가까운 k곳의 평균을 내는 행렬을 만든다."""
    xy = sites.groupby(st)[["lat", "lon"]].mean().loc[stations].to_numpy()
    xy = xy * np.array([111.0, 111.0 * np.cos(np.radians(36.0))])          # km
    dist = np.linalg.norm(xy[:, None] - xy[None], axis=2)
    np.fill_diagonal(dist, np.inf)
    W = np.zeros_like(dist)
    np.put_along_axis(W, np.argsort(dist, axis=1)[:, :k], 1.0 / k, axis=1)
    return W


CORR_END = pd.Timestamp("2025-06-01 01:00", tz=TZ)          # 평가 첫 시각. 제출용 이웃의 상관은 그 전 예보로만 잰다


def correlation_weights(wide: pd.DataFrame, k: int = 4, end: pd.Timestamp = CORR_END) -> np.ndarray:
    """좌표를 쓰지 않고, 학습 구간 예보 운량 시계열의 상관이 가장 높은 k곳의 평균을 내는 행렬을 만든다.
    설명회 슬라이드 17: 좌표는 노이즈가 들어간 값이며 거리·근접 구조 추정용이 아니다."""
    x = wide[wide.index < end]
    corr = x.corr(min_periods=24 * 365).to_numpy()
    corr = np.where(np.isfinite(corr), corr, -np.inf)
    np.fill_diagonal(corr, -np.inf)
    W = np.zeros_like(corr)
    np.put_along_axis(W, np.argsort(-corr, axis=1)[:, :k], 1.0 / k, axis=1)
    return W


ROUND_BLEND = os.environ.get("OIBC_ROUND_BLEND") == "1"   # 1이면 기본 예보를 17시 단독 대신 10시·17시 가중 결합으로 만든다
BLEND_FIT_END = pd.Timestamp("2024-01-01 01:00", tz=TZ)    # 결합 가중치는 모든 검증 구간보다 앞선 2021~2023년 실측으로만 정한다


def blend_weight(provider: str) -> float:
    """Bates–Granger 최적 결합 가중치(17시 쪽). 낮 8~18시 운량 예보 오차의 분산·공분산으로 구한다(src.experiments.explain combine과 같은 식)."""
    st, c = f"{provider}_station", f"{provider}_cloud"
    f = load_forecast("train", provider, [c], end=BLEND_FIT_END)
    f = f[(f["ts"] == f["ts"].dt.floor("h")) & f["ts"].dt.hour.between(8, 18)]
    w = f.pivot_table(index=[st, "ts"], columns="round", values=c, aggfunc="first")
    o = pd.read_parquet(TRAIN_DIR / f"observed_{provider}.parquet", columns=[st, "ts", c])
    o["ts"] = pd.to_datetime(o["ts"]).dt.tz_convert(TZ)
    o = o[o[c].between(0, 100)].set_index([st, "ts"])[c]
    j = w.join(o.rename("obs"), how="inner").dropna()
    e10, e17 = j[10] - j["obs"], j[17] - j["obs"]
    v10, v17, cov = e10.var(), e17.var(), float(np.cov(e10, e17)[0, 1])
    return float((v10 - cov) / (v10 + v17 - 2 * cov))


def station_table(provider: str, sites: pd.DataFrame, weighted_average=None) -> pd.DataFrame:
    """관측소×시각 격자의 예보 특징. 17시 예보가 기본이고 빈 곳은 10시 예보로 채운다."""
    st = f"{provider}_station"
    cols = [c for c in forecast_columns(provider) if c not in DROP]
    f = pd.concat([load_forecast("train", provider, cols), load_forecast("test", provider, cols)])
    f = f[f["ts"] == f["ts"].dt.floor("h")]                    # 학습 B의 정각 아닌 92행은 연결하지 않는다
    grid = pd.MultiIndex.from_product(
        [sorted(f[st].unique()), pd.date_range(f["ts"].min(), f["ts"].max(), freq="h")], names=[st, "ts"])
    r17 = f[f["round"] == 17].drop(columns="round").set_index([st, "ts"]).reindex(grid)
    r10 = f[f["round"] == 10].drop(columns="round").set_index([st, "ts"]).reindex(grid)
    del f
    key = cols[-1]                                             # wa_snow·wb_cloud: 행이 있으면 비어 있지 않다
    out = r17.fillna(r10)
    if ROUND_BLEND:                                            # 두 라운드가 다 있으면 가중 결합, 하나만 있으면 그 값
        wt = blend_weight(provider)
        print(f"  {provider} 결합 가중치(17시) {wt:.3f}", flush=True)
        out = out.where(r17.isna() | r10.isna(), wt * r17 + (1 - wt) * r10)
    out[f"{provider}_r17_missing"] = r17[key].isna().astype("float32")
    for c in R10_COLS[provider]:
        out[f"{c}_r10"] = r10[c].fillna(r17[c])
        out[f"{c}_rdiff"] = out[c] - out[f"{c}_r10"]
    del r17, r10

    g = out.groupby(level=0, sort=False)
    for c in NEIGHBOR_COLS[provider]:
        out[f"{c}_prev"] = g[c].shift(1)
        out[f"{c}_next"] = g[c].shift(-1)
        out[f"{c}_roll3"] = (out[c] + out[f"{c}_prev"] + out[f"{c}_next"]) / 3
    ts = out.index.get_level_values("ts")
    day = (ts - pd.Timedelta(hours=1)).normalize()             # 우측닫힘: 00시는 전날에 속한다
    daytime = (ts.hour >= 8) & (ts.hour <= 18)
    for c, how in DAILY[provider].items():
        out[f"{c}_day{how}"] = (out[c].where(daytime)
                                .groupby([out.index.get_level_values(0), day]).transform(how))
    keys = [out.index.get_level_values(0), day]
    for c in NEIGHBOR_COLS[provider]:                          # 구름 양상 특징 (EXTRA_GROUPS)
        x = out[c].where(daytime)
        out[f"{c}_daystd"] = x.groupby(keys).transform("std")
        if c.endswith("cloud"):
            out[f"{c}_daymin"] = x.groupby(keys).transform("min")
            out[f"{c}_daymax"] = x.groupby(keys).transform("max")
        else:
            out[f"{c}_daymean"] = x.groupby(keys).transform("mean")
        out[f"{c}_am"] = out[c].where((ts.hour >= 8) & (ts.hour <= 12)).groupby(keys).transform("mean")
        out[f"{c}_pm"] = out[c].where((ts.hour >= 13) & (ts.hour <= 18)).groupby(keys).transform("mean")
        out[f"{c}_trend"] = out[f"{c}_next"] - out[f"{c}_prev"]
        out[f"{c}_rdiff_dayabs"] = out[f"{c}_rdiff"].abs().where(daytime).groupby(keys).transform("mean")
        out[f"{c}_regstd"] = out[c].groupby(level="ts").transform("std")
    stations = list(out.index.levels[0])
    W = neighbor_weights(sites, st, stations)
    Wc = correlation_weights(out[NEIGHBOR_COLS[provider][0]].unstack(level=0)[stations])
    for c in NEIGHBOR_COLS[provider]:
        out[f"{c}_region"] = out[c].groupby(level="ts").transform("mean")    # 그 시각의 광역 날씨
        out[f"{c}_regdev"] = out[c] - out[f"{c}_region"]
        wide = out[c].unstack(level=0)[stations]
        filled = wide.T.fillna(wide.mean(axis=1)).T.to_numpy()
        nearby = filled @ W.T if weighted_average is None else weighted_average(filled, W)
        correlated = filled @ Wc.T if weighted_average is None else weighted_average(filled, Wc)
        out[f"{c}_nbr"] = pd.DataFrame(nearby, index=wide.index, columns=stations).unstack().reindex(out.index)
        out[f"{c}_nbrc"] = pd.DataFrame(correlated, index=wide.index, columns=stations).unstack().reindex(out.index)
    return out.astype("float32").reset_index()


def nbrc_table(provider: str, end: pd.Timestamp, weighted_average=None) -> pd.DataFrame:
    """(관측소, 시각)별 _nbrc 값. 상관을 end 전 예보로만 재서 이웃을 고른다. station_table의 계산과 같다."""
    if ROUND_BLEND:
        raise ValueError("nbrc_table은 OIBC_ROUND_BLEND와 함께 쓰지 않는다")
    st, cols = f"{provider}_station", NEIGHBOR_COLS[provider]
    f = pd.concat([load_forecast("train", provider, cols), load_forecast("test", provider, cols)])
    f = f[f["ts"] == f["ts"].dt.floor("h")]
    grid = pd.MultiIndex.from_product(
        [sorted(f[st].unique()), pd.date_range(f["ts"].min(), f["ts"].max(), freq="h")], names=[st, "ts"])
    r17 = f[f["round"] == 17].drop(columns="round").set_index([st, "ts"]).reindex(grid)
    r10 = f[f["round"] == 10].drop(columns="round").set_index([st, "ts"]).reindex(grid)
    out = r17.fillna(r10)
    stations = list(out.index.levels[0])
    Wc = correlation_weights(out[cols[0]].unstack(level=0)[stations], end=end)
    res = {}
    for c in cols:
        wide = out[c].unstack(level=0)[stations]
        filled = wide.T.fillna(wide.mean(axis=1)).T.to_numpy()
        values = filled @ Wc.T if weighted_average is None else weighted_average(filled, Wc)
        res[f"{c}_nbrc"] = pd.DataFrame(values, index=wide.index, columns=stations).unstack().reindex(out.index)
    return pd.DataFrame(res).astype("float32")


def with_nbrc_asof(d: pd.DataFrame, end: pd.Timestamp) -> pd.DataFrame:
    """_nbrc 열을 end 전 예보로 고른 이웃의 값으로 바꾼 사본. 검증 구간의 예측시점 뒤 예보가 이웃 선택에 섞이지 않게 한다."""
    sites = load_sites().set_index("site_id")
    out = d.copy(deep=False)
    for provider in ("wa", "wb"):
        table = nbrc_table(provider, end)
        keys = pd.MultiIndex.from_arrays([d["site_id"].map(sites[f"{provider}_station"]), d["ts"]])
        for c in table.columns:
            out[c] = table[c].reindex(keys).to_numpy()
    return out


def build(base: pd.DataFrame, sites: pd.DataFrame, tables: dict) -> pd.DataFrame:
    d = base.merge(sites.drop(columns="current_group"), on="site_id", how="left", validate="many_to_one")
    ts = pd.DatetimeIndex(d["ts"])
    d["elev"], d["azim"], d["cosz"] = solar(ts, d["lat"].to_numpy(), d["lon"].to_numpy())
    d = d[d["elev"] > 0].reset_index(drop=True)
    ts = pd.DatetimeIndex(d["ts"])
    for provider in ("wa", "wb"):
        d = d.merge(tables[provider], on=[f"{provider}_station", "ts"], how="left")
    doy = ts.dayofyear.to_numpy()
    d["doy_sin"] = np.sin(2*np.pi*doy/365.25).astype("float32")
    d["doy_cos"] = np.cos(2*np.pi*doy/365.25).astype("float32")
    d["hour"] = ts.hour.to_numpy().astype("int8")
    d["cap"] = d["capacity_kw"].astype("float32")
    d["site"] = d["site_id"].str[2:].astype("int16")
    d["cloud_mean"] = (d["wa_cloud"] + d["wb_cloud"]) / 2
    d["cloud_gap"] = d["wa_cloud"] - d["wb_cloud"]
    d["uv_per_cosz"] = d["wb_uv_idx"] / d["cosz"].clip(lower=0.05)
    d["day_cloud_gap"] = (d["wa_cloud_daymean"] - d["wb_cloud_daymean"]).abs()
    d["rain_gap"] = d["wa_rain_daysum"] - d["wb_total_liq_daysum"]
    d["cloud_absgap"] = d["cloud_gap"].abs()
    for c in ("lat", "lon", "wa_dist", "wb_dist"):
        d[c] = d[c].astype("float32")
    return d.drop(columns=["capacity_kw", "wa_station", "wb_station"])


def feature_names(d: pd.DataFrame) -> list:
    return [c for c in d.columns if c not in ("site_id", "ts", "energy")]


def main() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    sites = load_sites()
    tables = {p: station_table(p, sites) for p in ("wa", "wb")}
    gen = load_generation()
    train = build(gen, sites, tables)
    train.to_parquet(CACHE_DIR / "train_features.parquet", index=False)
    print("train", train.shape, f"{train.memory_usage().sum()/1e6:.0f} MB", flush=True)
    del train, gen
    grid = pd.MultiIndex.from_product([sites["site_id"], load_smp("test")["ts"]],
                                      names=["site_id", "ts"]).to_frame(index=False)
    test = build(grid, sites, tables)
    test.to_parquet(CACHE_DIR / "test_features.parquet", index=False)
    print("test", test.shape, "특징", len(feature_names(test)))
    print("test 특징별 결측 행(0 초과만):")
    na = test[feature_names(test)].isna().sum()
    print(na[na > 0].to_string() if (na > 0).any() else "  없음")


if __name__ == "__main__":
    main()
