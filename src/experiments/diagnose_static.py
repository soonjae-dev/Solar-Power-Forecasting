"""한 번 학습한 v3 모델로 11개월을 예측해 재학습 없이 성능이 유지되는지 본다. 집계만 출력하고 experiments/p9_static/metrics.json에 저장한다.

    python -m src.experiments.diagnose_static fit
    python -m src.experiments.diagnose_static report

실적이 있는 발전소-시각만 채점한다(편입 전 발전소는 그 시각의 집합 합계에서 빠진다). 가설과 판정 기준은 실행 전에 고정했다(docs/validation.md).
5월 holdout은 쓰지 않는다.
"""
import json
import sys

import pandas as pd

from ..config import EXP_DIR, PRED_DIR
from ..data import load_sites, load_smp
from .diagnose_days import target_day
from ..diagnose_season import FOLDS as RETRAIN_FOLDS, fit_predict, run_id, site_cohort
from ..official_baseline import RUN as BASE_RUN, fit_predict as base_fit_predict, panel

CONFIG, FOLD = "nbrc_asof", "fold1_static"
LATE = "2024-06~"                                # 시작 시점에 학습 이력이 없는 편입 묶음


def v3_path(fold_name: str):
    return PRED_DIR / run_id(CONFIG, fold_name) / "predictions.parquet"


def base_path(fold_name: str):
    return PRED_DIR / f"{BASE_RUN}_{fold_name}" / "predictions.parquet"


def fit() -> None:
    if not v3_path(FOLD).exists():
        fit_predict(CONFIG, FOLD, complete_only=False)
    if not base_path(FOLD).exists():
        base_fit_predict(FOLD, complete_only=False)


def penalty(p: pd.DataFrame) -> float:
    g = p.groupby(["ts", "group"])["err"].sum().unstack("group")
    return 0.1 * float((p.groupby("ts")["w"].first() * g.abs().sum(axis=1)).sum())


def pair(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """같은 행의 v3(a)와 기준선(b)."""
    wa, wb = penalty(a), penalty(b)
    return dict(sites=int(a["site_id"].nunique()), hours=int(a["ts"].nunique()), penalty=round(wa), base_penalty=round(wb),
                score=round(100 * (1 - wa / wb), 2), ratio=round(float(a["forecast_kwh"].sum() / a["energy"].sum()), 3),
                pen_pct_of_revenue=round(100 * wa / float((a["smp"] * a["energy"]).sum()), 3))


def on_keys(p: pd.DataFrame, keys: pd.DataFrame) -> pd.DataFrame:
    return p.merge(keys, on=["site_id", "ts"])


def retrain_compare(a: pd.DataFrame, b: pd.DataFrame, sites: pd.Series, smp: pd.DataFrame, cohort: pd.Series) -> dict:
    """재학습 구간과 같은 발전소-시각에서 고정 모델과 재학습 모델을 비교한다."""
    out = {}
    for name in RETRAIN_FOLDS:
        ra, rb = panel(v3_path(name), sites, smp), panel(base_path(name), sites, smp)
        keys = ra[["site_id", "ts"]]
        sa, sb = on_keys(a, keys), on_keys(b, keys)
        if len(sa) != len(ra) or len(sb) != len(ra):
            raise ValueError(f"{name}: 고정 모델 예측의 행이 재학습 구간과 다르다")
        row = dict(retrain=pair(ra, rb), static=pair(sa, sb))
        row["static_vs_retrain_pct"] = round(100 * (penalty(sa) / penalty(ra) - 1), 2)
        late = cohort.index[cohort == LATE]
        if ra["site_id"].isin(late).any():
            r_late, s_late = ra[ra["site_id"].isin(late)], sa[sa["site_id"].isin(late)]
            row["late_static_vs_retrain_pct"] = round(100 * (penalty(s_late) / penalty(r_late) - 1), 2)
            row["late_sites"] = int(r_late["site_id"].nunique())
            r_old, s_old = ra[~ra["site_id"].isin(late)], sa[~sa["site_id"].isin(late)]
            row["old_static_vs_retrain_pct"] = round(100 * (penalty(s_old) / penalty(r_old) - 1), 2)
            row["old_ratio"] = dict(retrain=pair(r_old, r_old)["ratio"], static=pair(s_old, s_old)["ratio"])
            row["late_ratio"] = dict(retrain=pair(r_late, r_late)["ratio"], static=pair(s_late, s_late)["ratio"])
        out[name] = row
    return out


def report() -> None:
    sites, smp = load_sites().set_index("site_id")["current_group"], load_smp("train")
    a, b = panel(v3_path(FOLD), sites, smp), panel(base_path(FOLD), sites, smp)
    if len(a) != len(b) or len(a) != len(a.merge(b[["site_id", "ts"]], on=["site_id", "ts"])):
        raise ValueError("v3와 기준선의 행이 다르다")
    cohort = site_cohort()
    res = dict(total=pair(a, b), zero_forecast_energy_share=round(float(a[(a["forecast_kwh"] == 0) & (a["energy"] > 0)]["energy"].sum() / a["energy"].sum()), 4))
    month_a, month_b = target_day(a["ts"]).dt.strftime("%Y-%m"), target_day(b["ts"]).dt.strftime("%Y-%m")
    res["by_month"] = {m: pair(a[month_a == m], b[month_b == m]) for m in sorted(month_a.unique())}
    split = "2025-01-01"
    res["by_period"] = {"2024-06~12": pair(a[month_a < split[:7]], b[month_b < split[:7]]),
                        "2025-01~04": pair(a[month_a >= split[:7]], b[month_b >= split[:7]])}
    res["by_cohort"] = {}
    for c in sorted(cohort.unique()):
        ids = cohort.index[cohort == c]
        res["by_cohort"][c] = pair(a[a["site_id"].isin(ids)], b[b["site_id"].isin(ids)])
    res["retrain_compare"] = retrain_compare(a, b, sites, smp, cohort)
    out = EXP_DIR / "p9_static"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

    def line(k, r):
        print(f"{k:<12} {r['sites']:>3}개소 점수 {r['score']:+6.2f}  페널티 {r['penalty']:>11,} 기준선 {r['base_penalty']:>11,}  예/실 {r['ratio']:.3f}  페널티/매출 {r['pen_pct_of_revenue']:.3f}%")
    line("전체", res["total"])
    print("0예측 실적 몫", res["zero_forecast_energy_share"])
    for name in ("by_period", "by_cohort", "by_month"):
        print(f"\n{name}")
        for k, r in res[name].items():
            line(k, r)
    print("\n재학습 대비 (같은 발전소-시각)")
    for k, r in res["retrain_compare"].items():
        late = (f"  신규 {r['late_sites']}개소 {r['late_static_vs_retrain_pct']:+.2f}% 예/실 {r['late_ratio']['retrain']}→{r['late_ratio']['static']}"
                f"  기존 {r['old_static_vs_retrain_pct']:+.2f}% 예/실 {r['old_ratio']['retrain']}→{r['old_ratio']['static']}") if "late_sites" in r else ""
        print(f"{k:<15} 점수 재학습 {r['retrain']['score']:+6.2f} → 고정 {r['static']['score']:+6.2f}  페널티 변화 {r['static_vs_retrain_pct']:+.2f}%{late}")


if __name__ == "__main__":
    {"fit": fit, "report": report}[sys.argv[1]]()
