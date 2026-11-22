"""4단계: 예측을 고정하고 fold1_static·fold2에서 {TAG}_ref 배정과 후보 배정을 비교한다.

    python -m src.experiments.eval_comp [OOF_NAME] [FOLD1_DIR] [FOLD2_DIR] [TAG]
    기본값: oof_nbrc_asof p1_nbrc_asof_fold1_static p1_nbrc_asof_fold2 v3
    후보는 search_oof가 만든 {TAG}_nolimit, {TAG}_ref_mv20. 비교 기준 {TAG}_ref는 sites.csv 참조 배정.

판정: fold1 −1% 초과 개선, 시드 3개 같은 방향, fold2 악화 없음, fold1 무작위 유효 배정 200개 중 낮은 쪽 5% 안.
진단(판정 제외): OOF 이력 없는 고정 발전소(48개소) 잔차를 0으로 둔 페널티 차.
시드별 열(forecast_kwh_s*)이 없으면 시드 기준은 확인 불가로 두고 판정은 보류한다.
7일 블록 부트스트랩 95% 구간은 참고. 결측 잔차는 0(mask_site와 동치).
"""
import sys

import numpy as np
import pandas as pd

from ..config import MIN_GROUP_KW, PRED_DIR, SUB_DIR
from ..data import load_sites, load_smp
from ..headroom import random_feasible
from ..scoring import PENALTY_RATE, composition_penalty

def load_fold(pred_dir, sites, w_all):
    p = pd.read_parquet(PRED_DIR / pred_dir / "predictions.parquet")
    out = {}
    for c in ["forecast_kwh"] + sorted(c for c in p.columns if c.startswith("forecast_kwh_s")):
        E = (p.assign(e=p[c] - p["energy"]).pivot(index="ts", columns="site_id", values="e")
              .reindex(columns=sites.index).fillna(0.0))
        out[c] = E.to_numpy()
    w = w_all.reindex(E.index).to_numpy()
    assert np.isfinite(w).all()
    return out, w, E.index


def hourly(E, w, g):
    return PENALTY_RATE * w * (np.abs(E[:, g].sum(1)) + np.abs(E[:, ~g].sum(1)))


def block_ci(d, ts, base, rng, n=2000):
    """시각별 페널티 차 d를 7일 블록으로 묶어 재표집한 상대 차(%)의 95% 구간."""
    blk = ((ts - ts[0]).days // 7).to_numpy()
    s = np.bincount(blk, weights=d)
    b = np.bincount(blk, weights=base)
    idx = rng.integers(0, len(s), (n, len(s)))
    r = 100 * s[idx].sum(1) / b[idx].sum(1)
    return np.percentile(r, [2.5, 97.5])


def main(oof_name="oof_nbrc_asof", fold1_dir="p1_nbrc_asof_fold1_static",
         fold2_dir="p1_nbrc_asof_fold2", tag="v3"):
    sites = load_sites().set_index("site_id")
    cap = sites["capacity_kw"].to_numpy()
    ref = (sites["current_group"] == "A").to_numpy()
    w_all = load_smp("train").set_index("ts")["w"]
    cands = {n: (pd.read_csv(SUB_DIR / f"composition_oof_{n}.csv").set_index("site_id")
                 .loc[sites.index, "group"] == "G1").to_numpy() for n in (f"{tag}_nolimit", f"{tag}_ref_mv20")}
    rng = np.random.default_rng(2026)
    folds = {"fold1_static": load_fold(fold1_dir, sites, w_all), "fold2": load_fold(fold2_dir, sites, w_all)}
    print(f"OOF {oof_name}  fold1 {fold1_dir}  fold2 {fold2_dir}  기준 {tag}_ref")

    # 무작위 유효 배정: 후보 탐색과 같게 OOF 이력 없는 발전소는 {tag}_ref 그룹에 고정
    E1, w1, _ = folds["fold1_static"]
    free = sites.index.isin(pd.read_parquet(PRED_DIR / oof_name / "predictions.parquet",
                                            columns=["site_id"])["site_id"].unique())
    rand = []
    while len(rand) < 200:
        g = np.where(free, random_feasible(cap, rng), ref)
        if MIN_GROUP_KW <= cap[g].sum() <= cap.sum() - MIN_GROUP_KW:
            rand.append(composition_penalty(E1["forecast_kwh"], w1, g))
    rand = np.array(rand)
    print(f"fold1 무작위 200개 5/50/95%: " + " / ".join(f"{x:,.0f}" for x in np.percentile(rand, [5, 50, 95])))

    for n, g in cands.items():
        print(f"\n== 후보 {n} (이동 {(g != ref).sum()}개소, G1 {cap[g].sum():,.1f} kW)")
        res = {}
        for f, (Es, w, ts) in folds.items():
            for c in Es:
                r, x = composition_penalty(Es[c], w, ref), composition_penalty(Es[c], w, g)
                res[f, c] = 100 * (x / r - 1)
                lab = "평균" if c == "forecast_kwh" else c[-2:]
                print(f"  {f:<13}{lab:<4} {tag}_ref {r:>13,.0f}  후보 {x:>13,.0f}  {res[f, c]:+.2f}%")
            base = hourly(Es["forecast_kwh"], w, ref)
            lo, hi = block_ci(hourly(Es["forecast_kwh"], w, g) - base, ts, base, rng)
            print(f"  {f:<13}7일 블록 95% 구간 [{lo:+.2f}%, {hi:+.2f}%]")
            Ef = Es["forecast_kwh"] * free                             # 진단(판정 제외): OOF 이력 없는 고정 발전소 잔차를 0으로
            r, x = composition_penalty(Ef, w, ref), composition_penalty(Ef, w, g)
            print(f"  {f:<13}고정 {(~free).sum()}개소 제외 진단 {tag}_ref {r:>13,.0f}  후보 {x:>13,.0f}  {100 * (x / r - 1):+.2f}%")
        x1 = composition_penalty(E1["forecast_kwh"], w1, g)
        pct = 100 * (rand < x1).mean()
        seeds = [c for c in E1 if c != "forecast_kwh"]
        checks = {"fold1 -1% 초과": res["fold1_static", "forecast_kwh"] < -1,
                  f"시드 {len(seeds)}개 같은 방향": all(res["fold1_static", c] < 0 for c in seeds) if seeds else None,
                  "fold2 악화 없음": res["fold2", "forecast_kwh"] <= 0,
                  f"무작위 200개 하위 5% (위치 {pct:.1f}%)": pct < 5}
        for k, v in checks.items():
            print(f"  {'확인 불가' if v is None else '통과' if v else '실패'}  {k}")
        verdict = ("미채택" if False in checks.values() else
                   "보류(시드 열 없음)" if None in checks.values() else "채택 후보")
        print(f"  판정: {verdict}")


if __name__ == "__main__":
    main(*sys.argv[1:])
