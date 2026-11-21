"""발전소별 예측을 고정했을 때 배정이 바꿀 수 있는 페널티의 범위를 잰다.

    python -m src.headroom ../OIBC_2026_data/predictions/baseline_r17_development/predictions.parquet
"""
import sys

import numpy as np
import pandas as pd

from .config import MIN_GROUP_KW
from .data import load_sites, load_smp
from .scoring import composition_penalty, lower_bound, residual_matrix


def load_problem(pred_path):
    """잔차 행렬 E(시각×발전소), 가중치 w, 발전소 메타(E의 열 순서). 'a.parquet+b.parquet'은 이어 붙인다."""
    E = residual_matrix(pd.concat([pd.read_parquet(p) for p in str(pred_path).split("+")]))
    smp = load_smp("train").set_index("ts")["w"]
    w = smp.reindex(E.index).to_numpy()
    assert np.isfinite(w).all()
    sites = load_sites().set_index("site_id").loc[E.columns]
    return E.to_numpy(), w, sites


def random_feasible(cap: np.ndarray, rng, strata=None) -> np.ndarray:
    """용량 제약을 만족하는 무작위 배정. strata가 있으면 층마다 절반씩 나눈다."""
    while True:
        if strata is None:
            g1 = rng.random(len(cap)) < 0.5
        else:
            g1 = np.zeros(len(cap), dtype=bool)
            for s in np.unique(strata):
                idx = rng.permutation(np.flatnonzero(strata == s))
                half = len(idx) // 2 + (rng.random() < 0.5) * (len(idx) % 2)
                g1[idx[:half]] = True
        c1 = cap[g1].sum()
        if c1 >= MIN_GROUP_KW and cap.sum() - c1 >= MIN_GROUP_KW:
            return g1


def main(pred_path: str, n: int = 300) -> None:
    E, w, sites = load_problem(pred_path)
    cap = sites["capacity_kw"].to_numpy()
    rng = np.random.default_rng(2026)
    lb = lower_bound(E, w)
    ref = composition_penalty(E, w, (sites["current_group"] == "A").to_numpy())
    indep = 0.1 * float(w @ np.abs(E).sum(axis=1))        # 상쇄가 전혀 없을 때
    lat_split = composition_penalty(E, w, sites["lat"].to_numpy() > np.median(sites["lat"]))

    def dist(strata):
        return np.array([composition_penalty(E, w, random_feasible(cap, rng, strata)) for _ in range(n)])

    plain = dist(None)
    by_station = dist(sites["wa_station"].to_numpy())
    print(f"상쇄 없음(발전소별 절대오차 합)  {indep:>14,.0f}")
    print(f"남북 분할(위도 중앙값)           {lat_split:>14,.0f}   하한 대비 +{100*(lat_split/lb-1):.1f}%")
    print(f"참조 배정                        {ref:>14,.0f}   하한 대비 +{100*(ref/lb-1):.1f}%")
    for name, x in [("무작위 배정", plain), ("관측소 층화 배정", by_station)]:
        print(f"{name:<14} 중앙값 {np.median(x):>14,.0f}  최소 {x.min():>14,.0f}  최대 {x.max():>14,.0f}"
              f"   하한 대비 +{100*(np.median(x)/lb-1):.1f}%")
    print(f"하한(전체를 한 집합으로 본 오차)  {lb:>14,.0f}")


if __name__ == "__main__":
    main(sys.argv[1])
