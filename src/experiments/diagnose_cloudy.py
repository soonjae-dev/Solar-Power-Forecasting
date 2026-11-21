"""흐린 날 편향이 다른 해에도 나타나는지 확인한다. 집계만 출력하고 experiments/p8_cloudy/metrics.json에 저장한다.

    python -m src.experiments.diagnose_cloudy fit      # 과거 7개 구간을 v3 설정(nbrc_asof)으로 학습·예측한다
    python -m src.experiments.diagnose_cloudy report   # 기존 4개 구간과 합쳐 판정한다

실측 흐림은 결과로 조건을 건 구분이라 평균으로의 회귀가 섞이고 예측시점에는 알 수 없다. 후보 착수는 예보 흐림으로 판정한다.
가설과 판정 기준은 실행 전에 고정했다(docs/validation.md). 5월 holdout은 쓰지 않는다.
"""
import json
import sys

import numpy as np
import pandas as pd

from ..config import EXP_DIR, PRED_DIR
from ..data import load_sites, load_smp
from .diagnose_days import forecast_cloud, observed_cloud, target_day
from ..diagnose_season import fit_predict, run_id
from ..official_baseline import panel

CONFIG = "nbrc_asof"
NEW_OVER = ("autumn_2022", "janfeb_2023", "autumn_2023", "janfeb_2024")      # 과대 계절의 새 구간
OLD_OVER = ("autumn_support", "development")
CONTROL = ("summer_2023", "spring_2023", "spring_2024", "summer_support", "selection")
FOLDS = NEW_OVER + OLD_OVER + CONTROL
NEW = NEW_OVER + ("summer_2023", "spring_2023", "spring_2024")
COHORT_FOLD = "autumn_2022"                                                  # 가장 이른 구간에 정답이 완전한 발전소
CUTS, LABELS = [-1, 30, 70, 101], ["맑음", "중간", "흐림"]
MIN_RATIO, MIN_DAYS = 1.05, 5
N_BOOT = 2000


def fold_panel(name: str, sites: pd.Series, smp: pd.DataFrame) -> pd.DataFrame:
    return panel(PRED_DIR / run_id(CONFIG, name) / "predictions.parquet", sites, smp)


def day_table(p: pd.DataFrame, obs: pd.Series, fcst: pd.Series) -> pd.DataFrame:
    g = p.groupby(["ts", "group"])["err"].sum().unstack("group")
    t = p.groupby("ts").agg(w=("w", "first"), act=("energy", "sum"), fc=("forecast_kwh", "sum"))
    t["pen"] = 0.1 * t["w"] * g.abs().sum(axis=1)
    d = t.groupby(target_day(t.index.to_series())).agg(pen=("pen", "sum"), act=("act", "sum"), fc=("fc", "sum"))
    d["obs"], d["fcst"] = obs.reindex(d.index), fcst.reindex(d.index)
    return d


def ratio_ci(x: pd.DataFrame, seed: int = 2026) -> list:
    """일 단위 부트스트랩 90% 구간."""
    fc, act = x["fc"].to_numpy(), x["act"].to_numpy()
    i = np.random.default_rng(seed).integers(0, len(x), (N_BOOT, len(x)))
    q = np.quantile(fc[i].sum(axis=1) / act[i].sum(axis=1), [0.05, 0.95])
    return [round(float(q[0]), 3), round(float(q[1]), 3)]


def buckets(d: pd.DataFrame, col: str) -> dict:
    cut = pd.cut(d[col], CUTS, labels=LABELS)
    out = {}
    for k in LABELS:
        x = d[cut == k]
        if len(x) == 0:
            out[k] = dict(days=0)
            continue
        out[k] = dict(days=len(x), ratio=round(float(x["fc"].sum() / x["act"].sum()), 3), ci90=ratio_ci(x),
                      pen_share=round(float(x["pen"].sum() / d["pen"].sum()), 3),
                      err_pct_of_fold=round(float(100 * (x["fc"] - x["act"]).sum() / d["act"].sum()), 2))
    return out


def fold_report(p: pd.DataFrame, obs: pd.Series, fcst: pd.Series) -> dict:
    d = day_table(p, obs, fcst)
    zero = p[(p["forecast_kwh"] == 0) & (p["energy"] > 0)]["energy"].sum() / p["energy"].sum()   # 예측이 없어 0으로 채워진 실적의 상한
    return dict(sites=int(p["site_id"].nunique()), days=len(d), ratio=round(float(d["fc"].sum() / d["act"].sum()), 3),
                pen_pct_of_revenue=round(float(100 * d["pen"].sum() / (p["smp"] * p["energy"]).sum()), 3),
                energy_share_at_zero_forecast=round(float(zero), 4), days_no_fcst=int(d["fcst"].isna().sum()),
                observed=buckets(d, "obs"), forecast=buckets(d, "fcst"))


def passes(b: dict) -> bool:
    return b.get("days", 0) >= MIN_DAYS and b["ratio"] >= MIN_RATIO and b["ci90"][0] > 1


def fit() -> None:
    for name in NEW:
        if (PRED_DIR / run_id(CONFIG, name) / "predictions.parquet").exists():
            print(f"{name}: 이미 있음", flush=True)
            continue
        fit_predict(CONFIG, name)


def show(name: str, r: dict) -> None:
    def cell(b):
        return f"{b['days']}일 {b['ratio']:.3f} [{b['ci90'][0]:.3f},{b['ci90'][1]:.3f}] 몫{b['pen_share']:.2f} 오차{b['err_pct_of_fold']:+.1f}%" if b["days"] else "0일"
    print(f"\n{name}: {r['sites']}개소 {r['days']}일 전체 {r['ratio']:.3f} 페널티/매출 {r['pen_pct_of_revenue']}% 0예측 실적 몫 {r['energy_share_at_zero_forecast']} 예보 없는 날 {r['days_no_fcst']}")
    print("  실측 흐림 " + cell(r["observed"]["흐림"]))
    print("  예보 흐림 " + cell(r["forecast"]["흐림"]))
    print("  예보 중간 " + cell(r["forecast"]["중간"]))
    print("  예보 맑음 " + cell(r["forecast"]["맑음"]))


def report() -> None:
    sites, smp = load_sites().set_index("site_id")["current_group"], load_smp("train")
    obs, fcst = observed_cloud(), forecast_cloud()
    cohort = fold_panel(COHORT_FOLD, sites, smp)["site_id"].unique()
    res = dict(all={}, cohort={})
    for name in FOLDS:
        p = fold_panel(name, sites, smp)
        res["all"][name] = fold_report(p, obs, fcst)
        res["cohort"][name] = fold_report(p[p["site_id"].isin(cohort)], obs, fcst)
    a = [n for n in NEW_OVER if passes(res["all"][n]["observed"]["흐림"])]
    b = [n for n in NEW_OVER + OLD_OVER if passes(res["all"][n]["forecast"]["흐림"])]
    b_cohort = [n for n in NEW_OVER + OLD_OVER if passes(res["cohort"][n]["forecast"]["흐림"])]
    res["verdict"] = dict(A_observed_pass=a, A=len(a) == len(NEW_OVER), B_forecast_pass=b, B=len(b) >= 5, B_cohort_pass=b_cohort)
    out = EXP_DIR / "p8_cloudy"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"공통 발전소 {len(cohort)}개소")
    for scope in ("all", "cohort"):
        print(f"\n===== {scope} =====")
        for name in FOLDS:
            show(name, res[scope][name])
    print("\n판정:", json.dumps(res["verdict"], ensure_ascii=False))


if __name__ == "__main__":
    {"fit": fit, "report": report}[sys.argv[1]]()
