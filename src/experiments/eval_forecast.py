"""예측 후보 판정: 배정을 고정하고 fold1_static·fold2에서 기준 예측과 후보 예측의 페널티를 비교한다.

    python -m src.experiments.eval_forecast CAND [BASE] [COMPOSITION]
    기본값: BASE=gfs60_asof, COMPOSITION=gfs60_nolimit (SUB_DIR/composition_oof_{COMPOSITION}.csv)

판정: fold1 −1% 초과 개선, 시드 3개 모두 개선, fold2 악화 없음. 참고: 7일 블록 95% 구간, 참조 배정 기준, 월별 변화.
"""
import sys

import numpy as np
import pandas as pd

from ..config import SUB_DIR
from ..data import load_sites, load_smp
from .eval_comp import block_ci, hourly, load_fold
from ..scoring import composition_penalty


def main(cand, base="gfs60_asof", comp="gfs60_nolimit"):
    sites = load_sites().set_index("site_id")
    w_all = load_smp("train").set_index("ts")["w"]
    groups = {comp: (pd.read_csv(SUB_DIR / f"composition_oof_{comp}.csv").set_index("site_id")
                     .loc[sites.index, "group"] == "G1").to_numpy(),
              "참조": (sites["current_group"] == "A").to_numpy()}
    rng = np.random.default_rng(2026)
    print(f"후보 {cand}  기준 {base}  배정 {comp}")
    res = {}
    for f in ("fold1_static", "fold2"):
        Eb, w, ts = load_fold(f"p1_{base}_{f}", sites, w_all)
        Ec, wc, tc = load_fold(f"p1_{cand}_{f}", sites, w_all)
        assert ts.equals(tc), "두 예측의 시각이 다르다"
        for gname, g in groups.items():
            for c in Eb:
                b, x = composition_penalty(Eb[c], w, g), composition_penalty(Ec[c], w, g)
                res[f, gname, c] = 100 * (x / b - 1)
                lab = "평균" if c == "forecast_kwh" else c[-2:]
                print(f"  {f:<13}{gname:<14}{lab:<4} 기준 {b:>13,.0f}  후보 {x:>13,.0f}  {res[f, gname, c]:+.2f}%")
        g = groups[comp]
        hb, hc = hourly(Eb["forecast_kwh"], w, g), hourly(Ec["forecast_kwh"], w, g)
        lo, hi = block_ci(hc - hb, ts, hb, rng)
        print(f"  {f:<13}7일 블록 95% 구간 [{lo:+.2f}%, {hi:+.2f}%]")
        month = (ts.tz_localize(None) - pd.Timedelta(hours=1)).to_period("M")
        m = pd.DataFrame({"b": hb, "c": hc}, index=month).groupby(level=0).sum()
        print("  월별 변화(%): " + "  ".join(f"{p.strftime('%y-%m')} {100 * (r.c / r.b - 1):+.1f}" for p, r in m.iterrows()))
    seeds = [c for c in Eb if c != "forecast_kwh"]
    checks = {"fold1 −1% 초과": res["fold1_static", comp, "forecast_kwh"] < -1,
              f"시드 {len(seeds)}개 모두 개선": all(res["fold1_static", comp, c] < 0 for c in seeds),
              "fold2 악화 없음": res["fold2", comp, "forecast_kwh"] <= 0}
    for k, v in checks.items():
        print(f"  {'통과' if v else '실패'}  {k}")
    print(f"  판정: {'채택 후보' if all(checks.values()) else '미채택'}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
