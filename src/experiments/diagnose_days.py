"""여름·가을 페널티가 어떤 날에 몰리는지 일 단위로 진단한다. 집계만 출력하고 experiments/p7_days/metrics.json에 저장한다.

    python -m src.experiments.diagnose_days

v3 설정(구간별 이웃, 예측 p1_nbrc_asof_<구간>)의 참조 배정 페널티를 쓴다. 날씨 구분에 쓰는 실측 운량은 진단에만 쓴다.
질문: (1) 페널티가 소수의 날에 몰리나 (2) 예보가 틀린 날인가 (3) 주말·공휴일에 과대예측이 몰리나(출력제어 가설).
요일·공휴일은 특징에 없고, 이전에 진단한 적이 없다.
"""
import json

import numpy as np
import pandas as pd

from ..config import CACHE_DIR, EXP_DIR, PRED_DIR, TRAIN_DIR, TZ
from ..data import load_sites, load_smp
from ..diagnose_season import FOLDS, run_id
from ..official_baseline import panel
from ..train import clear_sky

# 공개된 2021~2024 관공서 공휴일(대체·임시 포함). 2024는 임시공휴일 10-01을 포함한다.
HOLIDAYS = pd.to_datetime([
    "2021-03-01", "2021-05-05", "2021-05-19", "2021-06-06", "2021-08-16", "2021-09-20", "2021-09-21", "2021-09-22",
    "2021-10-04", "2021-10-11",
    "2022-03-01", "2022-03-09", "2022-05-05", "2022-05-08", "2022-06-01", "2022-06-06", "2022-08-15", "2022-09-09",
    "2022-09-10", "2022-09-12", "2022-10-03", "2022-10-10",
    "2023-03-01", "2023-05-05", "2023-05-27", "2023-05-29", "2023-06-06", "2023-08-15", "2023-09-28", "2023-09-29",
    "2023-09-30", "2023-10-02", "2023-10-03", "2023-10-09",
    "2024-03-01", "2024-04-10", "2024-05-05", "2024-05-06", "2024-05-15", "2024-06-06", "2024-08-15", "2024-09-16",
    "2024-09-17", "2024-09-18", "2024-10-01", "2024-10-03", "2024-10-09",
    "2025-03-01", "2025-03-03", "2025-05-05", "2025-05-06",
])
OFF_DAY_NAMES = {5, 6}


def target_day(ts: pd.Series) -> pd.Series:
    """우측닫힘: 00시는 전날에 속한다."""
    return (ts - pd.Timedelta(hours=1)).dt.tz_localize(None).dt.normalize()


def day_kind(days: pd.Series) -> pd.Series:
    off = days.dt.dayofweek.isin(OFF_DAY_NAMES) | days.isin(HOLIDAYS)
    return off.map({True: "휴일", False: "평일"})


def observed_cloud() -> pd.Series:
    """일별 광역 실측 운량(8~18시, A·B 관측소 평균). 진단 전용이다."""
    parts = []
    for provider in ("wa", "wb"):
        c = f"{provider}_cloud"
        o = pd.read_parquet(TRAIN_DIR / f"observed_{provider}.parquet", columns=["ts", c])
        o["ts"] = pd.to_datetime(o["ts"]).dt.tz_convert(TZ)
        o = o[o[c].between(0, 100) & o["ts"].dt.hour.between(8, 18)]
        parts.append(o.groupby(target_day(o["ts"]))[c].mean())
    return pd.concat(parts, axis=1).mean(axis=1)


def forecast_cloud() -> pd.Series:
    """일별 광역 예보 운량(A·B의 8~18시 일평균 특징의 평균)."""
    c = pd.read_parquet(CACHE_DIR / "train_features.parquet", columns=["ts", "wa_cloud_daymean", "wb_cloud_daymean"])
    v = c[["wa_cloud_daymean", "wb_cloud_daymean"]].mean(axis=1)
    return v.groupby(target_day(c["ts"])).mean()


def day_table(fold_name: str, sites: pd.Series, smp: pd.DataFrame) -> pd.DataFrame:
    p = panel(PRED_DIR / run_id("nbrc_asof", fold_name) / "predictions.parquet", sites, smp)
    g = p.groupby(["ts", "group"])["err"].sum().unstack("group")
    t = p.groupby("ts").agg(w=("w", "first"), act=("energy", "sum"), fc=("forecast_kwh", "sum"))
    t["pen"] = 0.1 * t["w"] * g.abs().sum(axis=1)
    t["day"] = target_day(t.index.to_series())
    d = t.groupby("day").agg(pen=("pen", "sum"), act=("act", "sum"), fc=("fc", "sum"))
    d["err"] = d["fc"] - d["act"]
    d["kind"] = day_kind(d.index.to_series())
    d["obs"] = observed_cloud().reindex(d.index)
    d["fcst"] = forecast_cloud().reindex(d.index)
    return d


def group_stats(d: pd.DataFrame, by) -> dict:
    total = d["pen"].sum()
    out = {}
    for k, x in d.groupby(by, observed=True):
        out[str(k)] = dict(days=len(x), pen_share=round(float(x["pen"].sum() / total), 3),
                           pen_per_day=round(float(x["pen"].mean())), bias=round(float(x["fc"].sum() / x["act"].sum()), 3))
    return out


def fold_report(d: pd.DataFrame) -> dict:
    s = d["pen"].sort_values(ascending=False)
    top = {f"top{int(q * 100)}pct": round(float(s.iloc[:max(1, int(len(s) * q))].sum() / s.sum()), 3) for q in (0.1, 0.2)}
    d = d.assign(cloud=pd.cut(d["obs"], [-1, 30, 70, 101], labels=["실측 맑음", "실측 중간", "실측 흐림"]),
                 miss=pd.cut(d["fcst"] - d["obs"], [-200, -20, 20, 200], labels=["예보가 더 맑음(-20 미만)", "비슷", "예보가 더 흐림(+20 초과)"]))
    worst = d[d["pen"] >= s.iloc[max(1, int(len(s) * 0.1)) - 1]]
    return dict(days=len(d), concentration=top, by_cloud=group_stats(d, "cloud"), by_miss=group_stats(d, "miss"),
                by_kind=group_stats(d, "kind"),
                by_kind_clear=group_stats(d[d["cloud"] == "실측 맑음"], "kind"),
                worst_days=dict(over=int((worst["err"] > 0).sum()), under=int((worst["err"] < 0).sum()),
                                mean_abs_miss=round(float((worst["fcst"] - worst["obs"]).abs().mean()), 1),
                                other_abs_miss=round(float((d.drop(worst.index)["fcst"] - d.drop(worst.index)["obs"]).abs().mean()), 1)))


def off_day_history() -> dict:
    """모델과 무관한 확인: 실측이 맑은 날의 발전소 평균 청천 지수를 연·월별로 평일과 휴일로 나눠 본다.
    휴일에 체계적으로 낮으면 수요가 적은 날의 출력제어 신호다. 같은 발전소 구성의 영향을 줄이려 발전소-일 단위로 평균한다."""
    c = pd.read_parquet(CACHE_DIR / "train_features.parquet", columns=["site_id", "ts", "energy", "cap", "cosz"])
    c = c[c["ts"] < pd.Timestamp("2025-05-01 01:00", tz=TZ)]                  # 5월 holdout은 후보 탐색 진단에 넣지 않는다
    c["cs"] = c["cap"] * clear_sky(c["cosz"])
    c["day"] = target_day(c["ts"])
    g = c.groupby(["site_id", "day"], sort=False)[["energy", "cs"]].sum()
    g["kt"] = g["energy"] / g["cs"]
    fleet = g.groupby(level="day")["kt"].mean().to_frame("kt")
    fleet["obs"] = observed_cloud().reindex(fleet.index)
    fleet["kind"] = day_kind(fleet.index.to_series())
    clear = fleet[fleet["obs"] <= 30].copy()
    clear["year"] = clear.index.year
    clear["season"] = clear.index.month.map({12: "겨울", 1: "겨울", 2: "겨울", 3: "봄", 4: "봄", 5: "봄", 6: "여름", 7: "여름", 8: "여름",
                                             9: "가을", 10: "가을", 11: "가을"})
    out = {}
    for (y, s), x in clear.groupby(["year", "season"]):
        a, b = x[x["kind"] == "평일"]["kt"], x[x["kind"] == "휴일"]["kt"]
        if len(b) >= 3 and len(a) >= 3:
            out[f"{y} {s}"] = dict(weekday_days=len(a), off_days=len(b), weekday_kt=round(float(a.mean()), 3), off_kt=round(float(b.mean()), 3),
                                   off_vs_weekday_pct=round(float(100 * (b.mean() / a.mean() - 1)), 1))
    return out


def main() -> None:
    sites = load_sites().set_index("site_id")["current_group"]
    smp = load_smp("train")
    res = {f: fold_report(day_table(f, sites, smp)) for f in FOLDS}
    res["off_day_history_clear"] = off_day_history()
    out = EXP_DIR / "p7_days"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    for f in FOLDS:
        r = res[f]
        print(f"\n== {f} ({r['days']}일)  상위 10%일 페널티 몫 {r['concentration']['top10pct']}  상위 20% {r['concentration']['top20pct']}")
        print(f"  최악 10%일: 과대 {r['worst_days']['over']} 과소 {r['worst_days']['under']}  |예보−실측 운량| {r['worst_days']['mean_abs_miss']} (나머지 {r['worst_days']['other_abs_miss']})")
        for name in ("by_cloud", "by_miss", "by_kind", "by_kind_clear"):
            print(f"  {name}: " + "  ".join(f"{k}[{v['days']}일 몫{v['pen_share']} 예/실{v['bias']} 일{v['pen_per_day']:,}]" for k, v in r[name].items()))
    print("\n맑은 날 청천 지수(휴일 vs 평일):")
    for k, v in res["off_day_history_clear"].items():
        print(f"  {k}: 평일 {v['weekday_kt']}({v['weekday_days']}일) 휴일 {v['off_kt']}({v['off_days']}일) {v['off_vs_weekday_pct']:+.1f}%")


if __name__ == "__main__":
    main()
