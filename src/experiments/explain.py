"""결과의 원인 진단. 성능 비교가 아니라 "왜 그런 결과가 나왔는가"를 집계 지표로 확인한다.

    python -m src.experiments.explain neighbors   # 인접 관측소: 좌표 기반과 예보 상관 기반이 왜 같은 성능을 내는가, 인접 평균은 왜 도움이 되는가

결과는 experiments/explain/<주제>.json에 저장한다. 행은 출력하지 않는다.
"""
import json
import sys

import numpy as np
import pandas as pd

from ..config import CACHE_DIR, EXP_DIR, TRAIN_DIR, TZ
from ..data import load_forecast, load_sites
from ..features import CORR_END, correlation_weights, neighbor_weights
from ..train import clear_sky, load_train_features

OUT = EXP_DIR / "explain"


def _station_cloud(provider: str) -> pd.DataFrame:
    """관측소×시각 운량(17시 예보, 빈 곳은 10시). features.station_table과 같은 규칙, 학습 구간만."""
    st, c = f"{provider}_station", f"{provider}_cloud"
    f = load_forecast("train", provider, [c])
    f = f[(f["ts"] == f["ts"].dt.floor("h")) & (f["ts"] < CORR_END)]
    w = {r: f[f["round"] == r].pivot(index="ts", columns=st, values=c) for r in (10, 17)}
    return w[17].combine_first(w[10])


def _r2(y: np.ndarray, *xs: np.ndarray) -> float:
    X = np.column_stack([np.ones_like(y)] + list(xs))
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(1 - np.var(y - X @ beta) / np.var(y))


def neighbors() -> dict:
    sites = load_sites()
    res = {}
    print("가설 1: 예보 상관은 거리에 따라 줄어들므로, 두 방식은 대부분 같은 이웃을 고른다")
    for provider in ("wa", "wb"):
        st = f"{provider}_station"
        wide = _station_cloud(provider)
        stations = sorted(wide.columns)
        wide = wide[stations]
        Wd = neighbor_weights(sites, st, stations) > 0
        Wc = correlation_weights(wide) > 0
        overlap = (Wd & Wc).sum(axis=1)                                # 관측소별 공통 이웃 수(4곳 중)
        xy = sites.groupby(st)[["lat", "lon"]].mean().loc[stations].to_numpy() * np.array([111.0, 111.0 * np.cos(np.radians(36.0))])
        dist = np.linalg.norm(xy[:, None] - xy[None], axis=2)          # 진단용 근사 거리(km). 모델 입력이 아님
        corr = wide.corr(min_periods=24 * 365).to_numpy()
        iu = np.triu_indices(len(stations), 1)
        ok = np.isfinite(corr[iu])
        rank_corr = float(pd.Series(dist[iu][ok]).corr(pd.Series(corr[iu][ok]), method="spearman"))
        near = dist[iu][ok] < np.quantile(dist[iu][ok], 0.1)
        r = dict(stations=len(stations),
                 overlap_mean=round(float(overlap.mean()), 2),
                 overlap_dist={int(k): int(v) for k, v in pd.Series(overlap).value_counts().sort_index().items()},
                 spearman_distance_vs_corr=round(rank_corr, 3),
                 corr_nearest10pct_pairs=round(float(np.mean(corr[iu][ok][near])), 3),
                 corr_other_pairs=round(float(np.mean(corr[iu][ok][~near])), 3),
                 neighbor_dist_km_coord=round(float(dist[Wd].mean()), 1),
                 neighbor_dist_km_corr=round(float(dist[Wc].mean()), 1),
                 neighbor_corr_coord=round(float(np.nanmean(np.where(Wd, corr, np.nan))), 3),
                 neighbor_corr_corr=round(float(np.nanmean(np.where(Wc, corr, np.nan))), 3))
        res[provider] = r
        print(f"  {provider}: 관측소 {r['stations']}곳, 공통 이웃 평균 {r['overlap_mean']}/4 (분포 {r['overlap_dist']}), "
              f"거리-상관 순위상관 {r['spearman_distance_vs_corr']}, 가장 가까운 10% 쌍의 상관 {r['corr_nearest10pct_pairs']} vs 나머지 {r['corr_other_pairs']}")
        print(f"      이웃까지 평균 거리: 좌표 기반 {r['neighbor_dist_km_coord']} km, 상관 기반 {r['neighbor_dist_km_corr']} km   "
              f"이웃과의 평균 상관: 좌표 기반 {r['neighbor_corr_coord']}, 상관 기반 {r['neighbor_corr_corr']}")

    print("가설 2: 한 관측소의 예보는 발전소 위 하늘을 잡음 섞인 점 하나로 본다. 이웃 평균이 그 잡음을 줄인다")
    cols = ["energy", "cap", "cosz", "hour"] + [f"{p}_cloud{s}" for p in ("wa", "wb") for s in ("", "_nbr", "_nbrc", "_region")]
    d = load_train_features()[cols]
    d = d[(d["cosz"] > 0.3) & d[cols].notna().all(axis=1)]             # 해가 충분히 높은 낮 시간
    kt = (d["energy"] / (d["cap"] * clear_sky(d["cosz"]))).clip(0, 2).to_numpy()
    res["signal"] = {"rows": int(len(d))}
    for p in ("wa", "wb"):
        own, nbr, nbrc, reg = (d[f"{p}_cloud{s}"].to_numpy(dtype="float64") for s in ("", "_nbr", "_nbrc", "_region"))
        r = dict(corr_own=round(float(np.corrcoef(kt, own)[0, 1]), 3),
                 corr_nbr=round(float(np.corrcoef(kt, nbr)[0, 1]), 3),
                 corr_nbrc=round(float(np.corrcoef(kt, nbrc)[0, 1]), 3),
                 corr_region=round(float(np.corrcoef(kt, reg)[0, 1]), 3),
                 r2_own=round(_r2(kt, own), 4), r2_own_nbr=round(_r2(kt, own, nbr), 4),
                 r2_own_nbrc=round(_r2(kt, own, nbrc), 4), r2_own_region=round(_r2(kt, own, reg), 4))
        res["signal"][p] = r
        print(f"  {p}: 청천 지수와의 상관  자기 관측소 {r['corr_own']}  좌표 이웃 평균 {r['corr_nbr']}  상관 이웃 평균 {r['corr_nbrc']}  광역 평균 {r['corr_region']}")
        print(f"      선형 R²  자기 {r['r2_own']}  +좌표 이웃 {r['r2_own_nbr']}  +상관 이웃 {r['r2_own_nbrc']}  +광역 {r['r2_own_region']}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "neighbors.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"저장: {OUT / 'neighbors.json'}")
    return res


def field() -> dict:
    """설명회 슬라이드 25의 질문을 채택 모델(v2)의 검증 구간 예측으로 확인한다.
    같은 기상 지점 발전소는 함께 틀리나 / 큰 발전소의 오차 비중 / 시간대·계절별 상쇄 / 크게 틀리는 날 / A·B 예보 불일치와 오차."""
    from ..config import PRED_DIR
    from ..data import load_smp
    from ..diagnose_season import FOLDS, run_id
    sites = load_sites().set_index("site_id")
    smp = load_smp("train").set_index("ts")
    cloud = load_train_features()[["site_id", "ts", "wa_cloud", "wb_cloud"]]
    res = {}
    for fold in FOLDS:
        p = pd.read_parquet(PRED_DIR / run_id("adopted", fold) / "predictions.parquet", columns=["site_id", "ts", "energy", "forecast_kwh"])
        p["energy"] = p["energy"].clip(lower=0)
        p["e"] = p["forecast_kwh"] - p["energy"]
        E = p.pivot(index="ts", columns="site_id", values="e")
        E = E[(E != 0).any(axis=1)]                                    # 야간(전 발전소 0) 제외
        w = smp["w"].reindex(E.index).to_numpy()
        k3 = (smp["k"].reindex(E.index) == 3).to_numpy()
        ids = E.columns
        grp = (sites.loc[ids, "current_group"] == "A").to_numpy()
        r = {"sites": len(ids)}

        # 1. 같은 A 관측소를 쓰는 발전소끼리 오차가 더 비슷한가
        C = E.corr().to_numpy()
        st = sites.loc[ids, "wa_station"].to_numpy()
        same = st[:, None] == st[None, :]
        iu = np.triu_indices(len(ids), 1)
        r["corr_same_station"] = round(float(np.nanmean(C[iu][same[iu]])), 3)
        r["corr_diff_station"] = round(float(np.nanmean(C[iu][~same[iu]])), 3)
        r["same_station_pairs"] = int(same[iu].sum())

        # 2. 큰 발전소(900 kW 이상)의 비중
        cap = sites.loc[ids, "capacity_kw"].to_numpy()
        big = cap >= 900
        site_err = (np.abs(E.to_numpy()) * w[:, None]).sum(axis=0)
        r["big_sites"] = int(big.sum())
        r["big_cap_share"] = round(float(cap[big].sum() / cap.sum()), 3)
        r["big_err_share"] = round(float(site_err[big].sum() / site_err.sum()), 3)
        r["big_in_G1"] = int((big & grp).sum())

        # 3. 집합으로 묶었을 때의 상쇄율 = 1 − Σw(|E1|+|E2|) / Σw Σ|e_i|, 시간대별
        A = E.to_numpy()
        e1, e2 = A[:, grp].sum(axis=1), A[:, ~grp].sum(axis=1)
        pen = w * (np.abs(e1) + np.abs(e2))
        indiv = w * np.abs(A).sum(axis=1)
        lb = w * np.abs(e1 + e2)
        for name, m in (("all", np.ones_like(k3)), ("k3", k3), ("k1", ~k3)):
            r[f"cancel_{name}"] = round(float(1 - pen[m].sum() / indiv[m].sum()), 3)
            r[f"common_share_{name}"] = round(float(lb[m].sum() / pen[m].sum()), 3)   # 페널티 중 배정으로 못 없애는 몫

        # 4. 크게 틀리는 날: 페널티 상위 10% 날의 비중과, 그날 공통 오차의 몫
        day = (E.index - pd.Timedelta(hours=1)).normalize()
        dp = pd.Series(pen, index=day).groupby(level=0).sum()
        dl = pd.Series(lb, index=day).groupby(level=0).sum()
        top = dp >= dp.quantile(0.9)
        r["days"] = int(len(dp))
        r["top10_days_penalty_share"] = round(float(dp[top].sum() / dp.sum()), 3)
        r["top10_days_common_share"] = round(float(dl[top].sum() / dp[top].sum()), 3)
        r["other_days_common_share"] = round(float(dl[~top].sum() / dp[~top].sum()), 3)

        # 5. A·B 예보가 서로 다를수록 크게 틀리나 (낮 8~18시 운량 차의 일평균)
        c = cloud[cloud["ts"].isin(E.index) & cloud["ts"].dt.hour.between(8, 18)]
        gap = (c["wa_cloud"] - c["wb_cloud"]).abs().groupby((c["ts"] - pd.Timedelta(hours=1)).dt.normalize()).mean()
        mcl = ((c["wa_cloud"] + c["wb_cloud"]) / 2).groupby((c["ts"] - pd.Timedelta(hours=1)).dt.normalize()).mean()
        j = pd.DataFrame({"pen": dp, "gap": gap, "cloud": mcl}).dropna()
        r["spearman_gap_vs_daypen"] = round(float(j["pen"].corr(j["gap"], method="spearman")), 3)
        r["spearman_cloud_vs_daypen"] = round(float(j["pen"].corr(j["cloud"], method="spearman")), 3)
        hi = j["gap"] >= j["gap"].quantile(0.75)
        r["daypen_ratio_gap_top25_vs_rest"] = round(float(j.loc[hi, "pen"].mean() / j.loc[~hi, "pen"].mean()), 2)
        res[fold] = r
        print(f"{fold:<15} 발전소 {r['sites']}")
        print(f"  같은 A 관측소 발전소끼리 오차 상관 {r['corr_same_station']} (쌍 {r['same_station_pairs']}) vs 다른 관측소 {r['corr_diff_station']}")
        print(f"  900 kW 이상 {r['big_sites']}개소(G1에 {r['big_in_G1']}): 용량 비중 {r['big_cap_share']:.1%}, 발전소 단위 오차 비중 {r['big_err_share']:.1%}")
        print(f"  상쇄율 전체 {r['cancel_all']:.1%} (11~15시 {r['cancel_k3']:.1%}, 그 외 {r['cancel_k1']:.1%})  "
              f"페널티 중 공통 오차 몫 {r['common_share_all']:.1%} (11~15시 {r['common_share_k3']:.1%})")
        print(f"  페널티 상위 10% 날({int(top.sum())}일)의 비중 {r['top10_days_penalty_share']:.1%}, 그날 공통 오차 몫 {r['top10_days_common_share']:.1%} (나머지 날 {r['other_days_common_share']:.1%})")
        print(f"  A·B 운량 차와 일 페널티 순위상관 {r['spearman_gap_vs_daypen']}, 운량 자체와 {r['spearman_cloud_vs_daypen']}, "
              f"운량 차 상위 25% 날의 일 페널티는 나머지의 {r['daypen_ratio_gap_top25_vs_rest']}배")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "field.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"저장: {OUT / 'field.json'}")
    return res


def rounds() -> dict:
    """17시 예보가 10시 예보보다 정확한가, 10시 예보가 17시에 없는 정보를 더하나. 학습 구간만 쓴다.
    (1) 같은 관측소·시각의 실측 운량과 비교한 예보 오차 (2) 발전소 청천 지수와의 관계."""
    res = {}
    print("(1) 실측 운량 대비 예보 오차, 낮 8~18시, 학습 구간")
    for provider in ("wa", "wb"):
        st, c = f"{provider}_station", f"{provider}_cloud"
        f = load_forecast("train", provider, [c])
        f = f[(f["ts"] == f["ts"].dt.floor("h")) & f["ts"].dt.hour.between(8, 18)]
        w = f.pivot_table(index=[st, "ts"], columns="round", values=c, aggfunc="first")
        o = pd.read_parquet(TRAIN_DIR / f"observed_{provider}.parquet", columns=[st, "ts", c])
        o["ts"] = pd.to_datetime(o["ts"]).dt.tz_convert(TZ)
        o = o[o[c].between(0, 100)].set_index([st, "ts"])[c]           # 센티널(9999 등) 제외
        j = w.join(o.rename("obs"), how="inner").dropna()
        e10, e17 = (j[10] - j["obs"]).abs(), (j[17] - j["obs"]).abs()
        lead = j.index.get_level_values("ts").hour
        r = dict(rows=int(len(j)), mae_r10=round(float(e10.mean()), 2), mae_r17=round(float(e17.mean()), 2),
                 r17_closer_share=round(float((e17 < e10).mean()), 3), tie_share=round(float((e17 == e10).mean()), 3),
                 identical_share=round(float((j[10] == j[17]).mean()), 3),
                 corr_r10_r17=round(float(j[10].corr(j[17])), 3))
        res[provider] = r
        print(f"  {provider}: 비교 {r['rows']:,}시각  MAE 10시 {r['mae_r10']}  17시 {r['mae_r17']}  "
              f"17시가 더 가까운 비율 {r['r17_closer_share']:.1%} (같음 {r['tie_share']:.1%})  두 라운드 값이 똑같은 비율 {r['identical_share']:.1%}  두 라운드 상관 {r['corr_r10_r17']}")

    print("(2) 발전소 청천 지수와의 관계(cos 천정각 > 0.3, 17시 예보가 있는 행)")
    cols = ["energy", "cap", "cosz", "wa_r17_missing", "wb_r17_missing",
            "wa_cloud", "wa_cloud_r10", "wb_cloud", "wb_cloud_r10", "wb_uv_idx", "wb_uv_idx_r10"]
    d = load_train_features()[cols]
    d = d[(d["cosz"] > 0.3) & (d["wa_r17_missing"] == 0) & (d["wb_r17_missing"] == 0)].dropna()
    kt = (d["energy"] / (d["cap"] * clear_sky(d["cosz"]))).clip(0, 2).to_numpy()
    res["signal"] = {"rows": int(len(d))}
    for v in ("wa_cloud", "wb_cloud", "wb_uv_idx"):
        x17, x10 = d[v].to_numpy(dtype="float64"), d[f"{v}_r10"].to_numpy(dtype="float64")
        r = dict(r2_r17=round(_r2(kt, x17), 4), r2_r10=round(_r2(kt, x10), 4), r2_both=round(_r2(kt, x17, x10), 4),
                 r2_avg=round(_r2(kt, (x17 + x10) / 2), 4))
        res["signal"][v] = r
        print(f"  {v}: 선형 R²  17시 {r['r2_r17']}  10시 {r['r2_r10']}  둘 다 {r['r2_both']}  평균 {r['r2_avg']}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "rounds.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"저장: {OUT / 'rounds.json'}")
    return res


def combine() -> dict:
    """10시·17시 예보의 최적 결합 가중치(Bates–Granger). 학습 구간 실측 운량 대비 오차의 분산·공분산으로 구한다.
    결합 예보 = w·17시 + (1−w)·10시, 오차분산을 최소화하는 w = (σ10² − c) / (σ10² + σ17² − 2c), c = 두 오차의 공분산.
    가중치가 대상 시각·계절·예보 수정 폭에 따라 달라지면 '조건부 결합'의 근거가 된다."""
    res = {}
    for provider in ("wa", "wb"):
        st, c = f"{provider}_station", f"{provider}_cloud"
        f = load_forecast("train", provider, [c])
        f = f[(f["ts"] == f["ts"].dt.floor("h")) & f["ts"].dt.hour.between(8, 18)]
        w = f.pivot_table(index=[st, "ts"], columns="round", values=c, aggfunc="first")
        o = pd.read_parquet(TRAIN_DIR / f"observed_{provider}.parquet", columns=[st, "ts", c])
        o["ts"] = pd.to_datetime(o["ts"]).dt.tz_convert(TZ)
        o = o[o[c].between(0, 100)].set_index([st, "ts"])[c]
        j = w.join(o.rename("obs"), how="inner").dropna().reset_index()
        j["e10"], j["e17"] = j[10] - j["obs"], j[17] - j["obs"]
        j["rev"] = (j[17] - j[10]).abs()
        ts = j["ts"]
        j["hour_band"] = pd.cut(ts.dt.hour, [7, 10, 13, 15, 18], labels=["08-10시", "11-13시", "14-15시", "16-18시"])
        j["season"] = ts.dt.month.map({12: "겨울", 1: "겨울", 2: "겨울", 3: "봄", 4: "봄", 5: "봄", 6: "여름", 7: "여름", 8: "여름", 9: "가을", 10: "가을", 11: "가을"})
        j["revision"] = pd.cut(j["rev"], [-0.1, 0, 10, 30, 101], labels=["수정 없음", "1~10", "11~30", "30 초과"])

        def stats(t):
            v10, v17, cov = t["e10"].var(), t["e17"].var(), t[["e10", "e17"]].cov().iloc[0, 1]
            wt = (v10 - cov) / (v10 + v17 - 2 * cov) if (v10 + v17 - 2 * cov) > 0 else 1.0
            blend = wt * t["e17"] + (1 - wt) * t["e10"]
            return dict(n=int(len(t)), rmse10=round(float(np.sqrt((t["e10"] ** 2).mean())), 2),
                        rmse17=round(float(np.sqrt((t["e17"] ** 2).mean())), 2),
                        err_corr=round(float(t["e10"].corr(t["e17"])), 3), w17=round(float(wt), 3),
                        rmse_blend=round(float(np.sqrt((blend ** 2).mean())), 2),
                        rmse_avg=round(float(np.sqrt((((t["e10"] + t["e17"]) / 2) ** 2).mean())), 2))
        r = {"all": stats(j)}
        a = r["all"]
        print(f"{provider}: 전체 {a['n']:,}시각  RMSE 10시 {a['rmse10']}  17시 {a['rmse17']}  오차 상관 {a['err_corr']}  "
              f"최적 17시 가중 {a['w17']} → RMSE {a['rmse_blend']} (단순 평균 {a['rmse_avg']})")
        for key in ("hour_band", "season", "revision"):
            r[key] = {}
            for v, t in j.groupby(key, observed=True):
                s = stats(t)
                r[key][str(v)] = s
                print(f"    {key:<9} {str(v):<8} n={s['n']:>9,}  RMSE 10시 {s['rmse10']:>6}  17시 {s['rmse17']:>6}  오차 상관 {s['err_corr']:>6}  "
                      f"최적 17시 가중 {s['w17']:>6} → {s['rmse_blend']}")
        res[provider] = r
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "combine.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
    return res


def imputation_ceiling() -> dict:
    """예보 결손을 어떤 방법(GAIN 포함)으로 채워도 얻을 수 있는 이득의 상한.
    평가 구간에서 대체값이 들어간 발전소-시각이 정산 가중치(SMP·k·용량·청천 일사)에서 차지하는 몫을 잰다.
    완벽한 대체라도 그 몫 이상의 페널티는 줄일 수 없다."""
    from ..data import load_smp
    t = pd.read_parquet(CACHE_DIR / "test_features.parquet",
                        columns=["site_id", "ts", "cap", "cosz", "wa_r17_missing", "wb_r17_missing",
                                 "wa_cloud", "wa_cloud_r10", "wb_cloud", "wb_cloud_r10"])
    smp = load_smp("test").set_index("ts")["w"]
    wt = t["cap"].to_numpy() * clear_sky(t["cosz"]) * smp.reindex(t["ts"]).to_numpy()
    res = {"daytime_rows": int(len(t))}
    for p in ("wa", "wb"):
        r17_missing = t[f"{p}_r17_missing"] == 1
        r10_missing = (t[f"{p}_cloud_r10"] == t[f"{p}_cloud"]) & ~r17_missing      # 10시가 비어 17시로 채운 행(값이 우연히 같은 행 포함, 상한 쪽으로 셈)
        res[p] = dict(r17_missing_rows=int(r17_missing.sum()),
                      r17_missing_weight_share=float(wt[r17_missing.to_numpy()].sum() / wt.sum()),
                      r10_filled_or_equal_rows=int(r10_missing.sum()))
        print(f"{p}: 17시 결손을 10시로 채운 주간 발전소-시각 {res[p]['r17_missing_rows']:,} / {len(t):,}  "
              f"정산 가중 몫 {res[p]['r17_missing_weight_share']:.4%}")
    (OUT / "imputation_ceiling.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
    return res


def _pair(a: str, b: str, fold: str) -> pd.DataFrame:
    """두 실험의 발전소별 예측을 같은 (발전소, 시각)으로 맞춘다. 열: site_id, ts, energy, fa, fb, w, group."""
    from ..config import PRED_DIR
    from ..data import load_smp
    from ..diagnose_season import run_id
    def load(r):
        name = next(n for n in (r, f"{r}_{fold}", run_id(r, fold)) if (PRED_DIR / n).exists())
        return pd.read_parquet(PRED_DIR / name / "predictions.parquet",
                               columns=["site_id", "ts", "energy", "forecast_kwh"]).set_index(["site_id", "ts"])
    pa, pb = load(a), load(b)
    d = pa.rename(columns={"forecast_kwh": "fa"}).join(pb["forecast_kwh"].rename("fb"), how="inner").reset_index()
    d["energy"] = d["energy"].clip(lower=0)
    smp = load_smp("train").set_index("ts")
    d["w"] = d["ts"].map(smp["w"])
    d["k"] = d["ts"].map(smp["k"])
    d["group"] = d["site_id"].map(load_sites().set_index("site_id")["current_group"])
    return d


def _hour_pen(d: pd.DataFrame, col: str) -> pd.Series:
    """시각별 참조 배정 페널티(원)."""
    e = (d[col] - d["energy"]).groupby([d["ts"], d["group"]]).sum().abs().groupby(level=0).sum()
    return 0.1 * e * d.groupby("ts")["w"].first()


def _split(d: pd.DataFrame, key: pd.Series, label: str) -> dict:
    """시각 단위 범주별 페널티 변화(b/a−1)와 전체 변화분 중 몫."""
    pa, pb = _hour_pen(d, "fa"), _hour_pen(d, "fb")
    k = key.reindex(pa.index)
    t = pd.DataFrame({"a": pa, "b": pb, "k": k}).dropna().groupby("k", observed=True)[["a", "b"]].sum()
    delta = t["b"].sum() - t["a"].sum()
    out = {}
    for v, r in t.iterrows():
        out[str(v)] = dict(change=round(float(r.b / r.a - 1), 4), share_of_delta=round(float((r.b - r.a) / delta), 3) if delta else None)
    print(f"    {label}: " + "  ".join(f"{v} {o['change']:+.1%}(몫 {o['share_of_delta']:+.0%})" for v, o in out.items()))
    return out


def _site_split(d: pd.DataFrame, key: pd.Series, label: str) -> dict:
    """발전소 단위 가중 절대오차 변화와 예측/실적, 발전소 범주별."""
    g = d.assign(c=d["site_id"].map(key), ea=d["w"] * (d["fa"] - d["energy"]).abs(), eb=d["w"] * (d["fb"] - d["energy"]).abs())
    t = g.groupby("c", observed=True).agg(n=("site_id", "nunique"), ea=("ea", "sum"), eb=("eb", "sum"),
                                         fa=("fa", "sum"), fb=("fb", "sum"), act=("energy", "sum"))
    out = {str(v): dict(sites=int(r.n), err_change=round(float(r.eb / r.ea - 1), 4),
                        bias_a=round(float(r.fa / r.act), 3), bias_b=round(float(r.fb / r.act), 3)) for v, r in t.iterrows()}
    print(f"    {label}: " + "  ".join(f"{v}({o['sites']}곳) 오차 {o['err_change']:+.1%} 예측/실적 {o['bias_a']}→{o['bias_b']}" for v, o in out.items()))
    return out


def _day_weather() -> pd.DataFrame:
    """대상일별 광역 날씨(낮 시간 예보 일집계의 발전소 평균). 날씨 구분과 '드문 날' 판정에만 쓴다."""
    cols = ["ts", "wa_cloud_daymean", "wb_cloud_daymean", "wb_uv_idx_daymax", "wa_rain_daysum"]
    c = load_train_features()[cols]
    return c.groupby((c["ts"] - pd.Timedelta(hours=1)).dt.normalize())[cols[1:]].mean()


def unexplained() -> dict:
    """설명 현황 표의 미설명 항목. 각 항목은 가설이 맞다면 나타나야 할 현상을 저장된 예측으로 확인한다."""
    from ..diagnose_season import site_cohort
    from ..splits import fold as get_fold
    res = {}
    dw = _day_weather()
    sky = pd.qcut(dw[["wa_cloud_daymean", "wb_cloud_daymean"]].mean(axis=1), 3, labels=["맑음", "중간", "흐림"])
    cohort = site_cohort()

    # 1. 청천 지수 타깃 (d0_nbr → d1_kt)
    print("[1] 청천 지수 타깃. 가설: 태양 기하로 정해지는 몫을 타깃에서 빼면, 모델은 구름 감쇠만 배우면 된다.\n"
          "    → 이력에 그 계절이 없는 발전소(외삽)와 태양 고도가 낮은 시간에서 개선이 커야 한다.")
    feats = load_train_features()[["site_id", "ts", "cosz"]]
    res["target"] = {}
    for fold in ("development", "selection"):
        d = _pair("d0_nbr", "d1_kt", fold)
        f = get_fold(fold)
        # 검증 달과 같은 달을 학습 이력에서 본 적이 있는가
        months = set(pd.date_range(f["val_start"], f["val_end"], freq="D").month)
        hist = load_train_features()[["site_id", "ts"]]
        hist = hist[hist["ts"] <= f["cutoff"]]
        seen = hist.groupby("site_id")["ts"].apply(lambda s: months.issubset(set(s.dt.month)))
        seen = seen.map({True: "그 계절 이력 있음", False: "그 계절 이력 없음"})
        cz = d[["ts"]].merge(feats[feats["ts"].between(f["val_start"], f["val_end"])].groupby("ts")["cosz"].mean().rename("cz"), on="ts", how="left")
        elev = pd.Series(pd.cut(cz["cz"], [-1, 0.3, 0.6, 1.1], labels=["고도 낮음", "중간", "높음"]).to_numpy(), index=d["ts"]).groupby(level=0).first()
        print(f"  {fold} 전체 {_hour_pen(d, 'fb').sum() / _hour_pen(d, 'fa').sum() - 1:+.1%}")
        res["target"][fold] = dict(seen=_site_split(d, seen, "발전소 이력"), elev=_split(d, elev, "태양 고도"),
                                   sky=_split(d, pd.Series(sky.reindex((pd.Series(d['ts'].unique()) - pd.Timedelta(hours=1)).dt.normalize()).to_numpy(), index=d['ts'].unique()), "날씨"))

    # 2. 2021년부터 학습 (c1 → c3)
    print("[2] 2021년부터 학습. 가설: 짧은 기간(2023-06~)에는 드문 날씨의 표본이 적다. 학습 기간이 길면 그런 날의 예측이 나아진다.\n"
          "    → 개선은 짧은 학습 기간에 비슷한 날이 적었던 '드문 날'에 몰려야 한다.")
    res["history"] = {}
    z = (dw - dw.mean()) / dw.std()
    for fold in ("development", "selection"):
        d = _pair("c1_full_cap", "c3_from2021", fold)
        f = get_fold(fold)
        vdays = sorted(((pd.Series(d["ts"].unique()) - pd.Timedelta(hours=1)).dt.normalize()).unique())
        short = z[(z.index >= pd.Timestamp("2023-06-01", tz=TZ)) & (z.index < f["cutoff"].normalize())]
        longw = z[(z.index >= pd.Timestamp("2021-01-01", tz=TZ)) & (z.index < f["cutoff"].normalize())]
        def n_similar(win, day):
            dist = np.sqrt(((win - z.loc[day]) ** 2).sum(axis=1))
            return int((dist < 0.5).sum())
        cnt = pd.Series({dd: n_similar(short, dd) for dd in vdays if dd in z.index})
        cnt_long = pd.Series({dd: n_similar(longw, dd) for dd in vdays if dd in z.index})
        rare = pd.qcut(cnt.rank(method="first"), 3, labels=["드문 날", "중간", "흔한 날"])
        hk = pd.Series(((pd.Series(d["ts"].unique()) - pd.Timedelta(hours=1)).dt.normalize()).map(rare).to_numpy(), index=d["ts"].unique())
        print(f"  {fold} 전체 {_hour_pen(d, 'fb').sum() / _hour_pen(d, 'fa').sum() - 1:+.1%}  "
              f"비슷한 날 수(중앙값) 2023-06~ {int(cnt.median())}일 vs 2021~ {int(cnt_long.median())}일")
        res["history"][fold] = dict(rarity=_split(d, hk, "날의 드묾"), cohort=_site_split(d, cohort, "편입 시기"),
                                    similar_median_short=int(cnt.median()), similar_median_long=int(cnt_long.median()))

    # 3. L1 vs L2 (b1 → b3, 개발)
    print("[3] L1 손실. 가설: L1은 발전소별 중앙값을 맞춘다. 발전소 오차 분포가 비대칭이면 중앙값의 합은 평균의 합(집합의 기댓값)과 어긋난다.\n"
          "    → L1은 발전소 단위 절대오차는 같거나 좋고, 집합 단위에서만 나빠야 한다. 집합 합계는 한쪽으로 치우쳐야 한다.")
    d = _pair("b1_full_cap_development", "b3_full_l1_cap_development", "development")
    site_a = float((d["w"] * (d["fa"] - d["energy"]).abs()).sum())
    site_b = float((d["w"] * (d["fb"] - d["energy"]).abs()).sum())
    grp_a, grp_b = float(_hour_pen(d, "fa").sum()), float(_hour_pen(d, "fb").sum())
    day = (d["ts"] - pd.Timedelta(hours=1)).dt.normalize()
    dd = d.assign(sky=day.map(sky))
    bias = dd.groupby("sky", observed=True)[["fa", "fb", "energy"]].sum()
    e = (d["fa"] - d["energy"])[d["energy"] > 0]
    skew = float(((e - e.mean()) ** 3).mean() / e.std() ** 3)
    res["loss"] = dict(site_err_change=round(site_b / site_a - 1, 4), group_pen_change=round(grp_b / grp_a - 1, 4),
                       bias_l2={str(k): round(float(r.fa / r.energy), 3) for k, r in bias.iterrows()},
                       bias_l1={str(k): round(float(r.fb / r.energy), 3) for k, r in bias.iterrows()},
                       site_error_skew_l2=round(skew, 2))
    print(f"  발전소 단위 가중 절대오차 L2→L1 {res['loss']['site_err_change']:+.1%}, 집합 페널티 {res['loss']['group_pen_change']:+.1%}")
    print(f"  예측/실적 L2 {res['loss']['bias_l2']}  L1 {res['loss']['bias_l1']}  L2 발전소 오차의 비대칭도 {res['loss']['site_error_skew_l2']}")

    # 4. k 가중 (adopted → k3)
    print("[4] 11~15시 가중. 가설: 모델에 시각 특징이 있어 11~15시와 그 밖을 따로 맞출 수 있으므로, 가중치를 바꿔도 맞춘 값이 거의 변하지 않는다.\n"
          "    → 두 모델의 예측 차이가 오차보다 훨씬 작아야 한다. 또 청천 일사 가중이 이미 11~15시에 무게를 싣는다.")
    tr = load_train_features()[["ts", "cap", "cosz"]]
    k3 = tr["ts"].dt.hour.between(11, 15).to_numpy()
    cs = clear_sky(tr["cosz"])
    shares = {name: round(float(w[k3].sum() / w.sum()), 3) for name, w in
              (("행 수", np.ones(len(tr))), ("용량", tr["cap"].to_numpy()), ("용량×청천일사(채택)", tr["cap"].to_numpy() * cs),
               ("용량×청천일사×k", tr["cap"].to_numpy() * cs * np.where(k3, 3, 1)))}
    res["kweight"] = {"k3_weight_share": shares}
    print(f"  학습 가중치 중 11~15시 몫: " + "  ".join(f"{k} {v:.1%}" for k, v in shares.items()))
    for fold in ("development", "selection"):
        d = _pair("adopted", "k3", fold)
        m = d["k"] == 3
        diff = float((d.loc[m, "fb"] - d.loc[m, "fa"]).abs().sum() / (d.loc[m, "fa"] - d.loc[m, "energy"]).abs().sum())
        res["kweight"][fold] = round(diff, 3)
        print(f"  {fold}: 11~15시 두 모델 예측 차이의 합 / 채택 모델 절대오차의 합 = {diff:.1%}")

    # 5. 특징 확장 (a0 → a1 → a2/a3, 개발)
    print("[5] 특징 확장(24→81개). 분해: 발전소 ID 추가(a0→a1)와 예보 파생 특징 추가(a0→a2)를 따로 본다.\n"
          "    가설: 발전소 ID는 발전소마다 고정된 효율·음영 차이를, 파생 특징은 날씨 판단을 개선한다.\n"
          "    → ID는 발전소별 예측/실적의 흩어짐을 줄이고, 파생 특징은 흐린 날 오차를 더 줄여야 한다.")
    res["features"] = {}
    for a, b, label in (("a0_baseline_development", "a1_site_development", "ID 추가"),
                        ("a0_baseline_development", "a2_full_nosite_development", "파생 특징 추가")):
        d = _pair(a, b, "development")
        per = d.groupby("site_id")[["fa", "fb", "energy"]].sum()
        disp_a, disp_b = float((per["fa"] / per["energy"]).std()), float((per["fb"] / per["energy"]).std())
        hk = pd.Series(((pd.Series(d["ts"].unique()) - pd.Timedelta(hours=1)).dt.normalize()).map(sky).to_numpy(), index=d["ts"].unique())
        print(f"  {label}: 집합 페널티 {_hour_pen(d, 'fb').sum() / _hour_pen(d, 'fa').sum() - 1:+.1%}  발전소별 예측/실적의 표준편차 {disp_a:.3f}→{disp_b:.3f}")
        res["features"][label] = dict(dispersion=(round(disp_a, 3), round(disp_b, 3)), sky=_split(d, hk, "날씨"))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "unexplained.json").write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str))
    print(f"저장: {OUT / 'unexplained.json'}")
    return res


def calibration() -> dict:
    """정규화 점수 상승이 물리적 오차 감소에서 왔는지, 페널티에 맞춘 수준 조정(캘리브레이션)에서 왔는지 가른다.
    공식 기준선(p4)과 v4 설정(nbrc_drop)을 참조 배정의 집합×시각 단위로 비교한다."""
    from ..config import PRED_DIR
    from ..data import load_smp
    from ..diagnose_season import FOLDS, run_id
    grp = load_sites().set_index("site_id")["current_group"]
    smp = load_smp("train").set_index("ts")
    scales = np.round(np.arange(0.80, 1.2001, 0.005), 3)
    res = {}

    def group_hours(path):
        p = pd.read_parquet(path, columns=["site_id", "ts", "energy", "forecast_kwh"])
        p["energy"] = p["energy"].clip(lower=0)
        h = p.groupby(["ts", p["site_id"].map(grp).rename("group")])[["energy", "forecast_kwh"]].sum().reset_index()
        h["smp"], h["k"] = h["ts"].map(smp["smp"]), h["ts"].map(smp["k"])
        return h

    def metrics(h, f):
        e = (f - h["energy"]).abs()
        day = h["energy"] > 0
        W = lambda c: float((h["smp"] * h["k"] * (c * f - h["energy"]).abs()).sum())
        best = min(scales, key=W)
        return dict(W=float((h["smp"] * h["k"] * e).sum()), kwh=float(e.sum()), k_only=float((h["k"] * e).sum()),
                    smp_only=float((h["smp"] * e).sum()), bias=float(f.sum() / h["energy"].sum()),
                    corr=float(np.corrcoef(f[day], h["energy"][day])[0, 1]), best_scale=float(best), W_best=W(best),
                    W_k3=float((h["smp"] * h["k"] * e)[h["k"] == 3].sum()), W_k1=float((h["smp"] * h["k"] * e)[h["k"] == 1].sum()))
    print("정규화 이득(%) = 100 × (1 − v4 / 기준선). 같은 비율이면 가중치와 무관하게 오차 자체가 줄었다는 뜻")
    for fold in FOLDS:
        b = group_hours(PRED_DIR / f"p4_official_baseline_{fold}" / "predictions.parquet")
        v = group_hours(PRED_DIR / run_id("nbrc_drop", fold) / "predictions.parquet")
        m = b.merge(v[["ts", "group", "forecast_kwh"]].rename(columns={"forecast_kwh": "fv"}), on=["ts", "group"])
        mb, mv = metrics(m, m["forecast_kwh"]), metrics(m, m["fv"])
        gain = lambda k: round(100 * (1 - mv[k] / mb[k]), 1)
        r = dict(baseline=mb, v4=mv, gain={k: gain(k) for k in ("W", "kwh", "k_only", "smp_only", "W_best", "W_k3", "W_k1")})
        res[fold] = r
        g = r["gain"]
        print(f"{fold:<15} 정규화 점수(W) {g['W']:+.1f}  |  가중치 없는 kWh 오차 {g['kwh']:+.1f}  k만 {g['k_only']:+.1f}  SMP만 {g['smp_only']:+.1f}  |  "
              f"11~15시 {g['W_k3']:+.1f}  그 외 {g['W_k1']:+.1f}")
        print(f"{'':<15} 예측/실적 기준선 {mb['bias']:.3f} → v4 {mv['bias']:.3f}  |  최적 사후 배율 기준선 {mb['best_scale']:.3f}, v4 {mv['best_scale']:.3f}  |  "
              f"둘 다 최적 배율로 보정한 뒤의 이득 {g['W_best']:+.1f}  |  시간별 상관 {mb['corr']:.3f} → {mv['corr']:.3f}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "calibration.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"저장: {OUT / 'calibration.json'}")
    return res


def ktcf() -> dict:
    """청천 지수(kt) 모델과 이용률(cf) 모델의 혼합. 가설: kt는 분모(청천 일사)가 작은 저고도 시간에 불안정하고,
    cf를 섞으면 그 시간 오차가 준다. 흐린 날도 따로 본다. 혼합 이득의 전제인 두 모델의 오차 상관도 잰다."""
    from ..splits import fold as get_fold
    from ..diagnose_season import FOLDS
    dw = _day_weather()
    sky = pd.qcut(dw[["wa_cloud_daymean", "wb_cloud_daymean"]].mean(axis=1), 3, labels=["맑음", "중간", "흐림"])
    feats = load_train_features()[["ts", "cosz"]]
    res = {}
    for fold in FOLDS:
        f = get_fold(fold)
        cz = feats[feats["ts"].between(f["val_start"], f["val_end"])].groupby("ts")["cosz"].mean()
        elev = pd.cut(cz, [-1, 0.3, 0.6, 1.1], labels=["고도 낮음", "중간", "높음"])
        d = _pair("nbrc_drop", "nbrc_drop_cf", fold)
        ea, eb = d["fa"] - d["energy"], d["fb"] - d["energy"]
        day = d["energy"] > 0
        r = {"err_corr": round(float(np.corrcoef(ea[day], eb[day])[0, 1]), 3)}
        site = d.assign(c=d["ts"].map(elev), ea=d["w"] * ea.abs(), eb=d["w"] * eb.abs()).groupby("c", observed=True)[["ea", "eb"]].sum()
        r["site_err_cf_vs_kt_by_elev"] = {str(k): round(float(v.eb / v.ea - 1), 4) for k, v in site.iterrows()}
        print(f"== {fold}  두 모델 발전소 오차 상관 {r['err_corr']}")
        print("    발전소 단위 오차, cf 모델 / kt 모델 − 1: " + "  ".join(f"{k} {v:+.1%}" for k, v in r["site_err_cf_vs_kt_by_elev"].items()))
        b = _pair("nbrc_drop", "blend_kt_cf_73", fold)
        hk = pd.Series(((pd.Series(b["ts"].unique()) - pd.Timedelta(hours=1)).dt.normalize()).map(sky).to_numpy(), index=b["ts"].unique())
        r["blend73_by_elev"] = _split(b, elev, "7:3 혼합의 집합 페널티 변화(태양 고도)")
        r["blend73_by_sky"] = _split(b, hk, "7:3 혼합의 집합 페널티 변화(날씨)")
        res[fold] = r
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ktcf.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"저장: {OUT / 'ktcf.json'}")
    return res


def autumn() -> dict:
    """2024년 가을(9~11월) 과대예측의 원인 가르기. 연도별 9~11월을 비교한다.
    (a) 예보가 실측보다 맑게 나왔나: 관측소 운량 예보(17시) − 실측 운량, 낮 8~18시
    (b) 같은 예보에서 발전이 적었나: 2021-09 이전부터 있던 발전소의 청천 지수를 예보 운량 구간별로 비교"""
    res = {"forecast_bias": {}, "kt_given_cloud": {}}
    print("(a) 9~11월 낮 시간 운량: 예보(17시) − 실측의 평균(양수 = 예보가 더 흐림)")
    for provider in ("wa", "wb"):
        st, c = f"{provider}_station", f"{provider}_cloud"
        f = load_forecast("train", provider, [c])
        f = f[(f["round"] == 17) & f["ts"].dt.month.isin([9, 10, 11]) & f["ts"].dt.hour.between(8, 18)].set_index([st, "ts"])[c]
        o = pd.read_parquet(TRAIN_DIR / f"observed_{provider}.parquet", columns=[st, "ts", c])
        o["ts"] = pd.to_datetime(o["ts"]).dt.tz_convert(TZ)
        o = o[o[c].between(0, 100)].set_index([st, "ts"])[c]
        f = f.groupby(level=[0, 1]).mean()                         # 정각 아닌 행·중복 키는 평균으로 묶는다
        o = o.groupby(level=[0, 1]).mean()
        j = pd.concat([f.rename("fc"), o.rename("ob")], axis=1, join="inner").dropna()
        y = j.index.get_level_values("ts").year
        t = j.groupby(y).agg(fc=("fc", "mean"), ob=("ob", "mean"))
        t["bias"] = t["fc"] - t["ob"]
        res["forecast_bias"][provider] = t.round(2).to_dict(orient="index")
        print(f"  {provider}: " + "  ".join(f"{yr}: 예보 {r.fc:.1f} 실측 {r.ob:.1f} 차 {r.bias:+.1f}" for yr, r in t.iterrows()))

    print("(b) 9~11월, 2021-09 이전부터 있던 발전소의 청천 지수(발전 ÷ 용량×청천 일사), 예보 운량(A·B 평균) 구간별")
    d = load_train_features()[["site_id", "ts", "energy", "cap", "cosz", "wa_cloud", "wb_cloud"]]
    first = d.groupby("site_id")["ts"].min()
    old = first.index[first < pd.Timestamp("2021-09-01", tz=TZ)]
    d = d[d["site_id"].isin(old) & d["ts"].dt.month.isin([9, 10, 11]) & (d["cosz"] > 0.3)]
    d = d.assign(cs=d["cap"] * clear_sky(d["cosz"]), cloud=(d["wa_cloud"] + d["wb_cloud"]) / 2, year=d["ts"].dt.year)
    d["bin"] = pd.cut(d["cloud"], [-1, 20, 50, 80, 101], labels=["맑음(0~20)", "약간(20~50)", "많음(50~80)", "흐림(80~100)"])
    t = d.groupby(["bin", "year"], observed=True).apply(lambda x: x["energy"].sum() / x["cs"].sum()).unstack("year").round(3)
    share = d.groupby(["year", "bin"], observed=True)["cs"].sum().unstack("bin")
    share = share.div(share.sum(axis=1), axis=0).round(3)
    res["kt_given_cloud"] = {str(k): v for k, v in t.to_dict(orient="index").items()}
    res["cloud_bin_share"] = {int(k): v for k, v in share.to_dict(orient="index").items()}
    res["old_sites"] = int(len(old))
    print(f"  발전소 {len(old)}곳")
    print("  " + t.to_string().replace("\n", "\n  "))
    print("  예보 운량 구간의 비중(청천 일사 가중):")
    print("  " + share.to_string().replace("\n", "\n  "))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "autumn.json").write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str))
    return res


if __name__ == "__main__":
    {"neighbors": neighbors, "field": field, "rounds": rounds, "combine": combine,
     "imputation": imputation_ceiling, "unexplained": unexplained, "calibration": calibration, "ktcf": ktcf, "autumn": autumn}[sys.argv[1]]()
