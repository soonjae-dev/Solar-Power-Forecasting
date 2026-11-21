"""집합 구성 탐색. 발전소별 잔차를 고정하고 J = Σ w(|E1|+|E2|)를 줄인다.

    python -m src.compose FIT_PRED [EVAL_PRED]     # 예측 parquet. EVAL은 탐색에 쓰지 않은 구간

|E1|+|E2| >= |E1+E2| 이므로 두 집합이 서로의 축소 복제본일수록 좋다.
탐색 결과는 FIT 구간에 과적합될 수 있으니 EVAL 구간에서 층화 배정과 비교해 채택한다.
"""
import sys

import numpy as np
import pandas as pd

from .config import MIN_GROUP_KW, SUB_DIR
from .headroom import load_problem, random_feasible
from .scoring import composition_penalty, lower_bound


def local_search(E, w, cap, g1, rng, swap_tries=20_000, free=None, g0=None, max_moves=None):
    """단일 이동은 전수, 쌍 교환은 무작위로 시도한다. 용량 제약을 유지한다.
    free: 움직일 수 있는 발전소(bool). g0·max_moves: g0와 다른 발전소 수를 max_moves 이하로 유지."""
    total = E.sum(axis=1)
    lo, hi = MIN_GROUP_KW, cap.sum() - MIN_GROUP_KW
    free = np.ones(len(cap), dtype=bool) if free is None else free
    g0 = g1.copy() if g0 is None else g0
    max_moves = len(cap) if max_moves is None else max_moves
    g1 = g1.copy()
    e1 = E[:, g1].sum(axis=1)
    cost = lambda x: float(w @ (np.abs(x) + np.abs(total - x)))
    best = cost(e1)
    improved = True
    while improved:
        improved = False
        sign = np.where(g1, -1.0, 1.0)                         # G1에서 빼거나 G1에 넣거나
        cand = e1[:, None] + E * sign
        costs = w @ (np.abs(cand) + np.abs(total[:, None] - cand))
        c1 = cap[g1].sum() + sign * cap
        costs[(c1 < lo) | (c1 > hi)] = np.inf
        moved = g1 != g0
        dist = moved.sum()
        costs[~free | (~moved & (dist + 1 > max_moves))] = np.inf
        i = int(np.argmin(costs))
        if costs[i] < best - 1e-9:
            e1, best, g1[i], improved = cand[:, i], float(costs[i]), not g1[i], True
            continue
        ins, outs = np.flatnonzero(g1 & free), np.flatnonzero(~g1 & free)
        c1 = cap[g1].sum()
        for _ in range(swap_tries):
            a, b = rng.choice(ins), rng.choice(outs)
            if not lo <= c1 - cap[a] + cap[b] <= hi:
                continue
            if dist + (1 - 2 * moved[a]) + (1 - 2 * moved[b]) > max_moves:   # 교환 후 g0와 다른 수
                continue
            x = e1 - E[:, a] + E[:, b]
            c = cost(x)
            if c < best - 1e-9:
                e1, best, improved = x, c, True
                g1[a], g1[b] = False, True
                break
    return g1, 0.1 * best


def main(fit_path: str, eval_path: str = None, restarts: int = 8) -> None:
    E, w, sites = load_problem(fit_path)
    cap = sites["capacity_kw"].to_numpy()
    strata = sites["wa_station"].to_numpy()
    rng = np.random.default_rng(2026)
    ref = (sites["current_group"] == "A").to_numpy()
    held = load_problem(eval_path) if eval_path else None
    if held:
        assert list(held[2].index) == list(sites.index)

    def report(name, g1):
        line = f"{name:<12} fit {composition_penalty(E, w, g1):>12,.0f}"
        if held:
            line += f"   eval {composition_penalty(held[0], held[1], g1):>12,.0f}"
        print(line + f"   G1 {cap[g1].sum():,.0f} kW / {g1.sum()}개소", flush=True)

    line = f"{'하한':<12} fit {lower_bound(E, w):>12,.0f}"
    print(line + (f"   eval {lower_bound(held[0], held[1]):>12,.0f}" if held else ""))
    report("참조 배정", ref)
    starts = [random_feasible(cap, rng, strata) for _ in range(restarts)]
    strat_fit = [composition_penalty(E, w, g) for g in starts]
    report("층화(중앙값)", starts[int(np.argsort(strat_fit)[len(starts) // 2])])
    found = [local_search(E, w, cap, g, rng) for g in starts]
    best_g1, _ = min(found, key=lambda r: r[1])
    for k, (g, _) in enumerate(found):
        report(f"탐색 #{k}", g)

    SUB_DIR.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({"site_id": sites.index, "group": np.where(best_g1, "G1", "G2")}).sort_values("site_id")
    out.to_csv(SUB_DIR / "composition_search.csv", index=False)
    print(f"저장: {SUB_DIR / 'composition_search.csv'} (FIT 구간 최저 탐색 결과)")


if __name__ == "__main__":
    main(*sys.argv[1:3])
