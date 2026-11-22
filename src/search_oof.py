"""3단계: OOF 잔차로 G1/G2 배정 후보 2개를 만든다. 결측 잔차는 0(mask_site와 동치).

    python -m src.search_oof [OOF_NAME] [TAG] [restarts]
    기본값: oof_nbrc_asof v3 8   (gfs60 예: python -m src.search_oof oof_gfs60 gfs60)

후보 {TAG}_nolimit: 제한 없는 탐색(무작위 시작 + {TAG}_ref 시작). 후보 {TAG}_ref_mv20: {TAG}_ref에서 최대 20개소 이동.
{TAG}_ref는 sites.csv 참조 배정(A→G1, B→G2). OOF 이력이 없는 발전소는 {TAG}_ref 그룹에 고정한다.
결과는 SUB_DIR/composition_oof_{TAG}_{nolimit,ref_mv20}.csv, composition_oof_{TAG}_random200.npy.
"""
import sys
import time

import numpy as np
import pandas as pd

from .compose import local_search
from .config import MIN_GROUP_KW, PRED_DIR, SUB_DIR
from .data import load_sites, load_smp
from .headroom import random_feasible
from .scoring import composition_penalty, lower_bound


def load_oof(oof_name="oof_nbrc_asof", col="forecast_kwh"):
    p = pd.read_parquet(PRED_DIR / oof_name / "predictions.parquet")
    sites = load_sites().set_index("site_id")
    E = (p.assign(e=p[col] - p["energy"]).pivot(index="ts", columns="site_id", values="e")
          .reindex(columns=sites.index))
    has = E.notna().any().to_numpy()
    w = load_smp("train").set_index("ts")["w"].reindex(E.index).to_numpy()
    assert np.isfinite(w).all()
    return E.fillna(0.0).to_numpy(), w, sites, has


def main(oof_name: str = "oof_nbrc_asof", tag: str = "v3", restarts: int = 8) -> None:
    E, w, sites, free = load_oof(oof_name)
    cap = sites["capacity_kw"].to_numpy()
    ref = (sites["current_group"] == "A").to_numpy()
    rng = np.random.default_rng(2026)
    pen = lambda g: composition_penalty(E, w, g)
    print(f"하한 {lower_bound(E, w):,.0f}  {tag}_ref {pen(ref):,.0f}  고정 {(~free).sum()}개소", flush=True)

    def start():                                    # 고정 발전소는 {tag}_ref 그룹, 나머지는 무작위
        while True:
            g = np.where(free, random_feasible(cap, rng), ref)
            if MIN_GROUP_KW <= cap[g].sum() <= cap.sum() - MIN_GROUP_KW:
                return g

    found = [local_search(E, w, cap, ref, rng, free=free)]
    for k in range(restarts):
        t = time.time()
        found.append(local_search(E, w, cap, start(), rng, free=free))
        print(f"  탐색 #{k} {found[-1][1]:,.0f}  {time.time() - t:.0f}s", flush=True)
    cands = {f"{tag}_nolimit": min(found, key=lambda r: r[1])[0],
             f"{tag}_ref_mv20": local_search(E, w, cap, ref, rng, free=free, g0=ref, max_moves=20)[0]}

    rand = np.array([pen(start()) for _ in range(200)])
    q = np.percentile(rand, [5, 50, 95])
    print(f"무작위 유효 배정 200개 OOF 페널티 5/50/95%: {q[0]:,.0f} / {q[1]:,.0f} / {q[2]:,.0f}")
    SUB_DIR.mkdir(parents=True, exist_ok=True)
    for name, g in cands.items():
        c1 = cap[g].sum()
        assert (g[~free] == ref[~free]).all(), "고정 발전소가 움직였다"
        assert MIN_GROUP_KW <= c1 <= cap.sum() - MIN_GROUP_KW, "용량 제약 위반"
        assert pen(g) <= pen(ref) + 1e-6, f"OOF 페널티가 {tag}_ref보다 높다"
        print(f"{name:<15} OOF {pen(g):,.0f}  {tag}_ref 대비 {100 * (pen(g) / pen(ref) - 1):+.2f}%  "
              f"이동 {(g != ref).sum()}개소  G1 {c1:,.1f} kW / G2 {cap.sum() - c1:,.1f} kW")
        pd.DataFrame({"site_id": sites.index, "group": np.where(g, "G1", "G2")}).to_csv(
            SUB_DIR / f"composition_oof_{name}.csv", index=False)
    np.save(SUB_DIR / f"composition_oof_{tag}_random200.npy", rand)


if __name__ == "__main__":
    a = sys.argv[1:4]
    main(*a[:2], *map(int, a[2:3]))
