"""계절 일반화 진단. 여름·가을 보조 구간에서 기준선과 채택 설정을 같은 발전소 집합으로 비교한다.

    python -m src.diagnose_season fit        # 보조 구간 2개를 학습·예측해 PRED_DIR에 저장
    python -m src.diagnose_season report     # 네 구간의 집계 지표를 출력하고 experiments/p1_season/metrics.json에 저장
    python -m src.diagnose_season ablate     # 가을 구간에서 설정을 하나씩 바꿔 과대예측의 원인을 가른다
    python -m src.diagnose_season compare k3 # 후보 설정을 네 구간에서 채택 설정과 비교
    python -m src.diagnose_season drift      # 같은 발전소·같은 달의 연도별 청천 지수 (학습 기간 안의 추세)

보조 구간은 정답이 완전한 발전소만 쓴다(여름 205개소, 가을 252개소). 253개소 전체의 공식 집합 점수와
같지 않으므로 원화 총액을 개발·선택 구간과 직접 비교하지 않고, 페널티/매출 비율과 기준선 대비 변화로 본다.
행은 출력하지 않는다.
"""
import json
import sys

import numpy as np
import pandas as pd

from .config import CACHE_DIR, EXP_DIR, PRED_DIR, TZ
from .data import load_generation, load_sites, load_smp
from .splits import fold as get_fold
from .train import DEFAULTS, check_options, clear_sky, drop_outage, fit_model, fold_features, kt_weight, load_train_features, sample_weight, select_features

CONFIGS = {
    "baseline": dict(feats="baseline", seeds=1),
    "adopted": dict(weight="cap", train_start="2021-01-01", target="kt", seeds=3),
}
# 가을 과대예측(기준선 1.000 → 채택 1.032)의 원인 가르기. 모두 1시드.
ABLATION = {
    "b0 기준선(24특징·2023~·cf·무가중)": dict(feats="baseline"),
    "b1 기준선 + 2021~": dict(feats="baseline", train_start="2021-01-01"),
    "s1 전체특징·용량가중·2023~·cf": dict(weight="cap"),
    "s2 s1 + 2021~": dict(weight="cap", train_start="2021-01-01"),
    "s3 s2 + 청천타깃 (채택 1시드)": dict(weight="cap", train_start="2021-01-01", target="kt"),
    "r1 채택에서 2023~로": dict(weight="cap", target="kt"),
    "r2 채택에서 2022-06~로": dict(weight="cap", train_start="2022-06-01", target="kt"),
    "r3 채택에서 용량가중 제거": dict(train_start="2021-01-01", target="kt"),
    "r4 채택에서 24특징으로": dict(feats="baseline", weight="cap", train_start="2021-01-01", target="kt"),
}
CONFIGS.update({k: {**v, "seeds": 1} for k, v in ABLATION.items()})
CANDIDATES = {  # 채택 설정에서 한 가지만 바꾼 후보. 채택과 같은 3시드
    "k3": dict(weight="capk", train_start="2021-01-01", target="kt", seeds=3),
    "f_all": dict(feats="full+var+ampm+dis+spa", weight="cap", train_start="2021-01-01", target="kt", seeds=3),
    "f_var": dict(feats="full+var", weight="cap", train_start="2021-01-01", target="kt", seeds=3),
    "f_ampm": dict(feats="full+ampm", weight="cap", train_start="2021-01-01", target="kt", seeds=3),
    "f_dis": dict(feats="full+dis", weight="cap", train_start="2021-01-01", target="kt", seeds=3),
    "f_spa": dict(feats="full+spa", weight="cap", train_start="2021-01-01", target="kt", seeds=3),
    # 2026-10-15 제한적 시험. nbrc: 인접 관측소를 예보 상관으로 고름(좌표 미사용). cb: CatBoost 1시드. blend_cb: 채택과 cb의 평균
    "nbrc": dict(feats="fullc", weight="cap", train_start="2021-01-01", target="kt", seeds=3),
    "nbrc_asof": dict(feats="fullc", weight="cap", train_start="2021-01-01", target="kt", nbrc_asof="fold", seeds=3),
    "gfs20": dict(feats="fullc", weight="cap", train_start="2021-01-01", target="kt", nbrc_asof="fold", common="gfs20", seeds=3),   # v3+GFS 20열
    "gfs60": dict(feats="fullc", weight="cap", train_start="2021-01-01", target="kt", nbrc_asof="fold", common="gfs60", seeds=3),   # gfs20+구름 층·가강수량(14.18)
    "gfs60seed": dict(feats="fullc", weight="cap", train_start="2021-01-01", target="kt", nbrc_asof="fold", common="gfs60", seeds=3),   # gfs60과 같은 설정, 팀원 전달용 시드별 열 출력
    "nbrc_asof_2023": dict(feats="fullc", weight="cap", train_start="2023-01-01", target="kt", nbrc_asof="fold", seeds=3),   # 팀원 fold2와 같은 학습 시작
    "nbrc_cs2": dict(feats="fullc", weight="cap", train_start="2021-01-01", target="kt", nbrc_asof="fold", kt_weight="cs2", seeds=3),
    "cb": dict(model="cb", n_estimators=1500, learning_rate=0.1, depth=8, weight="cap", train_start="2021-01-01", target="kt", seeds=1),
    "blend_cb": dict(blend=("adopted", "cb")),
    # 정지 의심 발전소-일을 학습에서 뺀 v3 설정. 비교 기준은 nbrc(v3와 같은 설정)
    # 기본 예보를 10시·17시 가중 결합으로 만든 캐시(OIBC_CACHE_NAME=cache_blend)에서 v3와 같은 설정. 비교 기준은 nbrc
    "nbrc_blend": dict(feats="fullc", weight="cap", train_start="2021-01-01", target="kt", seeds=3),
    "nbrc_drop": dict(feats="fullc", weight="cap", train_start="2021-01-01", target="kt", outage="drop", seeds=3),
    # 2026-10-15 사용자 제안: v4 설정에서 타깃만 이용률(cf)로 바꾼 모델과 v4의 가중 평균. 비교 기준은 nbrc_drop(v4)
    "nbrc_drop_cf": dict(feats="fullc", weight="cap", train_start="2021-01-01", target="cf", outage="drop", seeds=3),
    "blend_kt_cf_82": dict(blend=("nbrc_drop", "nbrc_drop_cf"), weights=(0.8, 0.2)),
    "blend_kt_cf_73": dict(blend=("nbrc_drop", "nbrc_drop_cf"), weights=(0.7, 0.3)),
}
CONFIGS.update(CANDIDATES)
SUPPORT = ("summer_support", "autumn_support")
STORED = {  # 개발·선택 구간은 이미 만든 예측을 다시 쓴다
    ("baseline", "development"): "c0_baseline_development", ("baseline", "selection"): "c0_baseline_selection",
    ("adopted", "development"): "e1_kt_ens3_development", ("adopted", "selection"): "e1_kt_ens3_selection",
}
FOLDS = SUPPORT + ("development", "selection")


def run_id(config: str, fold: str) -> str:
    return STORED.get((config, fold), f"p1_{config.split()[0]}_{fold}")


def blend(config: str, fold_name: str) -> None:
    """구성원 예측의 단순 평균. 구성원이 없으면 먼저 만든다."""
    members = CONFIGS[config]["blend"]
    parts = []
    for c in members:
        if not (PRED_DIR / run_id(c, fold_name) / "predictions.parquet").exists():
            fit_predict(c, fold_name)
        parts.append(pd.read_parquet(PRED_DIR / run_id(c, fold_name) / "predictions.parquet").set_index(["site_id", "ts"]))
    out = parts[0].copy()
    fc = [p["forecast_kwh"].reindex(out.index) for p in parts]
    if any(len(p) != len(out) or f.isna().any() for p, f in zip(parts, fc)):
        raise ValueError("구성원 예측의 (발전소, 시각)이 다르다")
    wts = CONFIGS[config].get("weights", [1 / len(fc)] * len(fc))
    out["forecast_kwh"] = sum(w * f for w, f in zip(wts, fc))
    (PRED_DIR / run_id(config, fold_name)).mkdir(parents=True, exist_ok=True)
    out.reset_index().to_parquet(PRED_DIR / run_id(config, fold_name) / "predictions.parquet", index=False)
    print(f"{config:<9}{fold_name:<15} 평균: {' + '.join(members)} 가중치 {CONFIGS[config].get('weights', '균등')}", flush=True)


def fit_predict(config: str, fold_name: str, complete_only: bool = True, per_seed: bool = False) -> None:
    if "blend" in CONFIGS[config]:
        return blend(config, fold_name)
    cfg = {**DEFAULTS, **{k: v for k, v in CONFIGS[config].items() if k not in ("seeds", "common")}}
    check_options(cfg)
    f = get_fold(fold_name)
    d = fold_features(load_train_features(), cfg, f["cutoff"])
    feats = select_features(d, cfg["feats"])
    smp = load_smp("train").set_index("ts")["w"]
    train = d[(d["ts"] >= pd.Timestamp(cfg["train_start"] + " 01:00", tz=TZ)) & (d["ts"] <= f["cutoff"])]
    val = d[d["ts"].between(f["val_start"], f["val_end"])]
    train = drop_outage(train, cfg)
    if CONFIGS[config].get("common") in ("gfs20", "gfs60"):     # 모든 발전소에 같은 GFS 열(전일 17시 전 공개분)
        from .gfs_pilot import load_gfs
        gfs, _ = load_gfs("2021-01-02", f["val_end"].strftime("%Y-%m-%d"))
        if CONFIGS[config]["common"] == "gfs60":
            from .gfs_more import load_more
            gfs = pd.concat([gfs, load_more("2021-01-02", f["val_end"].strftime("%Y-%m-%d"))[0]], axis=1)
        train, val = train.copy(), val.copy()
        train[list(gfs)] = gfs.reindex(train["ts"]).to_numpy()
        val[list(gfs)] = gfs.reindex(val["ts"]).to_numpy()
        feats = feats + list(gfs)
    y = (train["energy"] / train["cap"]).clip(0, 1.2)
    sw = sample_weight(train, cfg["weight"], smp)
    scale = 1.0
    if cfg["target"] == "kt":
        cs = clear_sky(train["cosz"])
        y = (y / cs).clip(0, 2.0)
        sw = kt_weight(cs, sw, cfg)
        scale = clear_sky(val["cosz"])
    seeds = CONFIGS[config]["seeds"]
    ratio = np.zeros(len(val))
    pred = val[["site_id", "ts"]].copy()
    for k in range(seeds):
        model = fit_model({**cfg, "seed": cfg["seed"] + k}, train[feats], y, sw)
        one = model.predict(val[feats])
        ratio += one / seeds
        if per_seed:                                                    # 팀원 요청: 시드별 예측 열(forecast_kwh_s0 등)
            pred[f"forecast_kwh_s{k}"] = np.clip(one * scale, 0, None) * val["cap"].to_numpy()
    pred["forecast_kwh"] = np.clip(ratio * scale, 0, None) * val["cap"].to_numpy()

    gen = load_generation()
    truth = gen[gen["ts"].between(f["val_start"], f["val_end"])]
    hours = truth["ts"].nunique()
    n = truth.groupby("site_id")["ts"].size()
    complete = n.index[n == hours] if complete_only else n.index       # 기본은 정답이 완전한 발전소만. 아니면 실적이 있는 행을 모두 채점한다
    out = truth[truth["site_id"].isin(complete)].merge(pred, on=["site_id", "ts"], how="left")
    for c in [c for c in out.columns if c.startswith("forecast_kwh")]:
        out[c] = out[c].fillna(0.0)                                     # 야간은 0
    out["energy"] = out["energy"].clip(lower=0)
    (PRED_DIR / run_id(config, fold_name)).mkdir(parents=True, exist_ok=True)
    out.to_parquet(PRED_DIR / run_id(config, fold_name) / "predictions.parquet", index=False)
    print(f"{config:<9}{fold_name:<15} 학습 {len(train):>9,}행  발전소 {len(complete)}  시각 {hours}", flush=True)


def site_cohort() -> pd.Series:
    """편입 시기: 첫 실적 시각 기준."""
    first = load_generation().groupby("site_id")["ts"].min()
    return pd.cut(first, [pd.Timestamp("2000-01-01", tz=TZ), pd.Timestamp("2023-06-01", tz=TZ),
                          pd.Timestamp("2024-06-01", tz=TZ), pd.Timestamp("2100-01-01", tz=TZ)],
                  labels=["~2023-05", "2023-06~2024-05", "2024-06~"], right=False).astype(str)


def day_cloud() -> pd.Series:
    """대상일별 광역 평균 운량(B 예보, 8~18시 평균). 날씨 구분에만 쓴다."""
    c = pd.read_parquet(CACHE_DIR / "train_features.parquet", columns=["ts", "wb_cloud_daymean"])
    return c.groupby(c["ts"].dt.normalize())["wb_cloud_daymean"].mean()


def score(p: pd.DataFrame) -> dict:
    """p: ts, group, energy, forecast_kwh, w, smp. 참조 배정 페널티와 하한, 매출 대비 비율."""
    g = p.groupby(["ts", "group"]).agg(e=("err", "sum")).unstack("group")["e"]
    t = p.groupby("ts").agg(w=("w", "first"), smp=("smp", "first"), act=("energy", "sum"), fc=("forecast_kwh", "sum"))
    pen = 0.1 * float((t["w"] * g.abs().sum(axis=1)).sum())
    lb = 0.1 * float((t["w"] * g.sum(axis=1).abs()).sum())
    sales = float((t["smp"] * t["act"]).sum())
    return dict(penalty=round(pen), lower_bound=round(lb), pen_pct=round(100 * pen / sales, 3),
                bias=round(float(t["fc"].sum() / t["act"].sum()), 4))


def load_panel(config: str, fold_name: str, sites: pd.DataFrame, smp: pd.DataFrame, cohort: pd.Series, cloud_cut) -> pd.DataFrame:
    p = pd.read_parquet(PRED_DIR / run_id(config, fold_name) / "predictions.parquet")
    p["energy"] = p["energy"].clip(lower=0)
    p["err"] = p["forecast_kwh"] - p["energy"]
    p["group"] = p["site_id"].map(sites["current_group"])
    p["cohort"] = p["site_id"].map(cohort)
    p = p.merge(smp[["ts", "smp", "k", "w"]], on="ts", how="left")
    day = (p["ts"] - pd.Timedelta(hours=1)).dt.normalize()
    p["month"] = day.dt.strftime("%Y-%m")
    p["sky"] = cloud_cut(day)
    return p


def report() -> None:
    sites = load_sites().set_index("site_id")
    smp = load_smp("train")
    cohort = site_cohort()
    cloud = day_cloud()
    days = pd.concat([pd.Series(pd.date_range(get_fold(f)["val_start"].normalize(), get_fold(f)["val_end"] - pd.Timedelta(days=1), freq="D")) for f in FOLDS])
    lo, hi = cloud.reindex(days).quantile([1/3, 2/3])
    cloud_cut = lambda day: pd.cut(day.map(cloud), [-1, lo, hi, 101], labels=["맑음", "중간", "흐림"]).astype(str)
    print(f"날씨 구분: 대상일 광역 평균 운량 3분위 경계 {lo:.1f}% / {hi:.1f}% (네 구간의 날 전체 기준)")
    print(f"편입 시기별 발전소 수: {cohort.value_counts().sort_index().to_dict()}\n")

    result = {}
    for fold_name in FOLDS:
        panels = {c: load_panel(c, fold_name, sites, smp, cohort, cloud_cut) for c in ("baseline", "adopted")}
        a = panels["adopted"]
        n_sites, cap = a["site_id"].nunique(), float(sites.loc[a["site_id"].unique(), "capacity_kw"].sum())
        res = {"sites": n_sites, "capacity_kw": round(cap, 1), "total": {c: score(p) for c, p in panels.items()}}
        b, ad = res["total"]["baseline"], res["total"]["adopted"]
        print(f"=== {fold_name}  발전소 {n_sites} ({cap:,.0f} kW)")
        print(f"  전체        기준선 {b['pen_pct']:.3f}%  채택 {ad['pen_pct']:.3f}%  변화 {ad['penalty']/b['penalty']-1:+.1%}   "
              f"채택의 하한 대비 {ad['penalty']/ad['lower_bound']-1:+.1%}  예측/실적 {ad['bias']:.3f} (기준선 {b['bias']:.3f})")
        for key, label in (("month", "월"), ("k", "가중치 k"), ("sky", "날씨")):
            res[key] = {}
            for v in sorted(a[key].astype(str).unique()):
                sb = score(panels["baseline"][panels["baseline"][key].astype(str) == v])
                sa = score(a[a[key].astype(str) == v])
                res[key][v] = {"baseline": sb, "adopted": sa}
                share = sa["penalty"] / ad["penalty"]
                print(f"  {label} {v:<9} 기준선 {sb['pen_pct']:.3f}%  채택 {sa['pen_pct']:.3f}%  변화 {sa['penalty']/sb['penalty']-1:+.1%}   "
                      f"페널티 비중 {share:.0%}  예측/실적 {sa['bias']:.3f}")
        res["cohort_bias"] = {}
        for v in sorted(a["cohort"].unique()):
            m = a[a["cohort"] == v]
            by_month = (m.groupby("month")["forecast_kwh"].sum() / m.groupby("month")["energy"].sum()).round(3)
            res["cohort_bias"][v] = {"sites": int(m["site_id"].nunique()), "by_month": by_month.to_dict()}
            print(f"  편입 {v:<16} {m['site_id'].nunique():>3}개소  월별 예측/실적 " + "  ".join(f"{k[5:]}월 {x:.3f}" for k, x in by_month.items()))
        result[fold_name] = res
        print()
    out = EXP_DIR / "p1_season"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"저장: {out / 'metrics.json'}")


def ablate(fold_name: str = "autumn_support") -> None:
    sites = load_sites().set_index("site_id")
    smp = load_smp("train")
    result = {}
    print(f"{fold_name}: 설정별 페널티/매출, 예측/실적(전체 · 11~15시 · 월별)")
    for config in ABLATION:
        path = PRED_DIR / run_id(config, fold_name) / "predictions.parquet"
        if not path.exists():
            fit_predict(config, fold_name)
        p = pd.read_parquet(path)
        p["err"] = p["forecast_kwh"] - p["energy"]
        p["group"] = p["site_id"].map(sites["current_group"])
        p = p.merge(smp[["ts", "smp", "k", "w"]], on="ts", how="left")
        month = (p["ts"] - pd.Timedelta(hours=1)).dt.strftime("%m")
        s = score(p)
        peak = score(p[p["k"] == 3])["bias"]
        by_month = (p.groupby(month)["forecast_kwh"].sum() / p.groupby(month)["energy"].sum()).round(3).to_dict()
        result[config] = {**s, "bias_k3": peak, "bias_by_month": by_month}
        print(f"  {config:<34} {s['pen_pct']:.3f}%  예측/실적 {s['bias']:.3f}  11~15시 {peak:.3f}  "
              + "  ".join(f"{m}월 {v:.3f}" for m, v in by_month.items()), flush=True)
    out = EXP_DIR / "p2_bias_ablation"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{fold_name}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))


def compare(config: str, base: str = "adopted") -> None:
    sites = load_sites().set_index("site_id")
    smp = load_smp("train")
    result = {}
    for fold_name in FOLDS:
        if not (PRED_DIR / run_id(config, fold_name) / "predictions.parquet").exists():
            fit_predict(config, fold_name)
        row = {}
        for c in (base, config):
            p = pd.read_parquet(PRED_DIR / run_id(c, fold_name) / "predictions.parquet")
            p["energy"] = p["energy"].clip(lower=0)
            p["err"] = p["forecast_kwh"] - p["energy"]
            p["group"] = p["site_id"].map(sites["current_group"])
            p = p.merge(smp[["ts", "smp", "k", "w"]], on="ts", how="left")
            month = (p["ts"] - pd.Timedelta(hours=1)).dt.strftime("%Y-%m")
            row[c] = {"total": score(p), "k3": score(p[p["k"] == 3]), "k1": score(p[p["k"] == 1]),
                      "month": {m: score(p[month == m]) for m in sorted(month.unique())}}
        a, b = row[base], row[config]
        chg = lambda x, y: f"{y['penalty'] / x['penalty'] - 1:+.1%}"
        print(f"{fold_name:<15} {base} {a['total']['pen_pct']:.3f}% → {config} {b['total']['pen_pct']:.3f}%  변화 {chg(a['total'], b['total'])}  "
              f"(11~15시 {chg(a['k3'], b['k3'])}, 그 외 {chg(a['k1'], b['k1'])})  예측/실적 {a['total']['bias']:.3f} → {b['total']['bias']:.3f}  "
              f"하한 변화 {b['total']['lower_bound'] / a['total']['lower_bound'] - 1:+.1%}", flush=True)
        print("    월별 변화 " + "  ".join(f"{m[2:]} {chg(a['month'][m], b['month'][m])}" for m in a["month"]), flush=True)
        result[fold_name] = row
    out = EXP_DIR / f"p2_candidate_{config}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))


def drift() -> None:
    """2021년 가을부터 실적이 있는 발전소만 골라, 같은 달(9~11월·1~2월)의 연도별 청천 지수와 예보 운량을 본다."""
    d = load_train_features()[["site_id", "ts", "energy", "cap", "cosz", "wb_cloud", "wa_cloud", "wb_uv_idx"]]
    first = d.groupby("site_id")["ts"].min()
    old = first.index[first < pd.Timestamp("2021-09-01", tz=TZ)]
    d = d[d["site_id"].isin(old)]
    d = d.assign(cs=clear_sky(d["cosz"]) * d["cap"], year=d["ts"].dt.year, month=d["ts"].dt.month)
    print(f"2021-09 이전부터 실적이 있는 발전소 {len(old)}개소 (용량 {d.groupby('site_id')['cap'].first().sum():,.0f} kW)")
    for label, months in (("9~11월", (9, 10, 11)), ("1~2월", (1, 2)), ("6~8월", (6, 7, 8)), ("3~4월", (3, 4))):
        m = d[d["month"].isin(months)]
        g = m.groupby("year")
        tab = pd.DataFrame({"발전소": g["site_id"].nunique(), "청천지수": (g["energy"].sum() / g["cs"].sum()).round(4),
                            "B운량": g["wb_cloud"].mean().round(1), "A운량": g["wa_cloud"].mean().round(1),
                            "B자외선": g["wb_uv_idx"].mean().round(2)})
        print(f"  {label}\n" + "\n".join("    " + line for line in tab.to_string().splitlines()))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "fit":
        for fold_name in SUPPORT:
            for config in ("baseline", "adopted"):
                fit_predict(config, fold_name)
    elif cmd == "ablate":
        ablate(*sys.argv[2:3])
    elif cmd == "compare":
        compare(*sys.argv[2:4])
    elif cmd == "drift":
        drift()
    else:
        report()
