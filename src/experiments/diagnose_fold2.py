"""Min fold2 조건으로 v3 기준점을 만든다. 집계만 출력하고 experiments/p10_min_fold2/metrics.json에 저장한다.

    python -m src.experiments.diagnose_fold2 fit
    python -m src.experiments.diagnose_fold2 report

학습 시작 2023-01(팀원 조건)과 2021-01(v3 원래 설정)을 같은 학습 마감으로 비교한다. 5월 holdout은 쓰지 않는다.
"""
import json
import sys

from ..config import EXP_DIR, PRED_DIR
from ..data import load_sites, load_smp
from .diagnose_days import target_day
from ..diagnose_season import fit_predict, run_id, site_cohort
from .diagnose_static import base_path, pair, penalty
from ..official_baseline import fit_predict as base_fit_predict, panel

FOLD = "min_fold2"
CONFIGS = {"2023-01 시작": "nbrc_asof_2023", "2021-01 시작": "nbrc_asof"}
PERIODS = {"2025-01~02": ("2025-01", "2025-02"), "2025-03~04": ("2025-03", "2025-04")}


def path(config: str):
    return PRED_DIR / run_id(config, FOLD) / "predictions.parquet"


def fit() -> None:
    for config in CONFIGS.values():
        if not path(config).exists():
            fit_predict(config, FOLD)
    if not base_path(FOLD).exists():
        base_fit_predict(FOLD)


def report() -> None:
    sites, smp = load_sites().set_index("site_id")["current_group"], load_smp("train")
    b = panel(base_path(FOLD), sites, smp)
    cohort = site_cohort()
    res = {}
    for label, config in CONFIGS.items():
        a = panel(path(config), sites, smp)
        if len(a) != len(b) or len(a) != len(a.merge(b[["site_id", "ts"]], on=["site_id", "ts"])):
            raise ValueError("v3와 기준선의 행이 다르다")
        ma, mb = target_day(a["ts"]).dt.strftime("%Y-%m"), target_day(b["ts"]).dt.strftime("%Y-%m")
        r = dict(total=pair(a, b), by_month={m: pair(a[ma == m], b[mb == m]) for m in sorted(ma.unique())})
        r["by_period"] = {k: pair(a[ma.isin(v)], b[mb.isin(v)]) for k, v in PERIODS.items()}
        r["by_cohort"] = {c: pair(a[a["site_id"].isin(cohort.index[cohort == c])], b[b["site_id"].isin(cohort.index[cohort == c])]) for c in sorted(cohort.unique())}
        res[label] = r
    x, y = res["2023-01 시작"]["total"]["penalty"], res["2021-01 시작"]["total"]["penalty"]
    res["start_2023_vs_2021_pct"] = round(100 * (x / y - 1), 2)
    out = EXP_DIR / "p10_min_fold2"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

    def line(k, r):
        print(f"  {k:<16}{r['sites']:>4}개소 점수 {r['score']:+6.2f}  페널티 {r['penalty']:>11,} 기준선 {r['base_penalty']:>11,}  예/실 {r['ratio']:.3f}  페널티/매출 {r['pen_pct_of_revenue']:.3f}%")
    for label in CONFIGS:
        print(label)
        line("전체", res[label]["total"])
        for name in ("by_period", "by_cohort", "by_month"):
            for k, r in res[label][name].items():
                line(k, r)
    print("2023 시작 / 2021 시작 페널티 변화", res["start_2023_vs_2021_pct"], "%")


if __name__ == "__main__":
    {"fit": fit, "report": report}[sys.argv[1]]()
