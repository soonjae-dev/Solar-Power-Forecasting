"""gfs60 예측으로 G1/G2 배정을 다시 찾고 판정한다(팀원의 v3 절차와 같음). 집계만 저장한다.

    .venv/bin/python -m src.experiments.assign_gfs60

OOF 잔차(p1_gfs60seed_oof6m_1~5)로 후보 2개를 만들고, fold1_static·min_fold2에서 참조 배정과 비교한다.
결측 잔차는 0(mask_site와 동치). OOF 이력이 없는 발전소는 참조 그룹에 고정한다.
"""
import json

import numpy as np
import pandas as pd

from ..compose import local_search
from ..config import EXP_DIR, MIN_GROUP_KW, PENALTY_RATE, PRED_DIR, SUB_DIR
from ..data import load_sites, load_smp
from ..headroom import random_feasible
from ..scoring import composition_penalty, lower_bound

COLS = ["forecast_kwh", "forecast_kwh_s0", "forecast_kwh_s1", "forecast_kwh_s2"]
OOF = [f"oof6m_{k}" for k in range(1, 6)]
FOLDS = {"fold1": "fold1_static", "fold2": "min_fold2"}


def residuals(names, sites, w_all):
    p = pd.concat([pd.read_parquet(PRED_DIR / f"p1_gfs60seed_{n}" / "predictions.parquet") for n in names])
    if p.duplicated(["site_id", "ts"]).any():
        raise ValueError("예측 구간이 겹친다")
    out = {}
    for c in COLS:
        E = p.assign(e=p[c] - p["energy"]).pivot(index="ts", columns="site_id", values="e").reindex(columns=sites.index)
        out[c] = E.fillna(0.0).to_numpy()
    has = E.notna().any().to_numpy()
    w = w_all.reindex(E.index).to_numpy()
    if not np.isfinite(w).all():
        raise ValueError("가중치가 비어 있다")
    return out, w, E.index, has


def hourly(E, w, g):
    return PENALTY_RATE * w * (np.abs(E[:, g].sum(1)) + np.abs(E[:, ~g].sum(1)))


def block_ci(d, ts, base, rng, n=2000):
    """시각별 페널티 차 d를 7일 블록으로 묶어 재표집한 상대 차(%)의 95% 구간."""
    blk = ((ts - ts[0]).days // 7).to_numpy()
    s, b = np.bincount(blk, weights=d), np.bincount(blk, weights=base)
    idx = rng.integers(0, len(s), (n, len(s)))
    return [round(float(x), 3) for x in np.percentile(100 * s[idx].sum(1) / b[idx].sum(1), [2.5, 97.5])]


def main(restarts=8):
    sites = load_sites().set_index("site_id")
    cap = sites["capacity_kw"].to_numpy()
    ref = (sites["current_group"] == "A").to_numpy()
    w_all = load_smp("train").set_index("ts")["w"]
    rng = np.random.default_rng(2026)
    oof, w, _, free = residuals(OOF, sites, w_all)
    E = oof["forecast_kwh"]
    pen = lambda g: composition_penalty(E, w, g)
    result = dict(rule="14.26", fixed_sites=int((~free).sum()), fixed_kw=round(float(cap[~free].sum()), 1),
                  ref_kw=[round(float(cap[ref].sum()), 1), round(float(cap[~ref].sum()), 1)],
                  oof=dict(lower_bound=round(lower_bound(E, w)), ref=round(pen(ref))))
    print(f"OOF 하한 {result['oof']['lower_bound']:,}  참조 {result['oof']['ref']:,}  고정 {result['fixed_sites']}개소", flush=True)

    def start():                                     # 고정 발전소는 참조 그룹, 나머지는 무작위
        while True:
            g = np.where(free, random_feasible(cap, rng), ref)
            if MIN_GROUP_KW <= cap[g].sum() <= cap.sum() - MIN_GROUP_KW:
                return g

    found = [local_search(E, w, cap, ref, rng, free=free)]
    for k in range(restarts):
        found.append(local_search(E, w, cap, start(), rng, free=free))
        print(f"  탐색 #{k} {found[-1][1]:,.0f}", flush=True)
    cands = {"gfs60_nolimit": min(found, key=lambda r: r[1])[0],
             "gfs60_ref_mv20": local_search(E, w, cap, ref, rng, free=free, g0=ref, max_moves=20)[0]}
    rand_oof = np.array([pen(start()) for _ in range(200)])
    result["oof"]["random200_q5_50_95"] = [round(float(x)) for x in np.percentile(rand_oof, [5, 50, 95])]
    result["oof"]["search_from_ref_is_best"] = bool(int(np.argmin([r[1] for r in found])) == 0)
    SUB_DIR.mkdir(parents=True, exist_ok=True)
    for name, g in cands.items():
        c1 = float(cap[g].sum())
        if not (g[~free] == ref[~free]).all() or not MIN_GROUP_KW <= c1 <= cap.sum() - MIN_GROUP_KW:
            raise ValueError("고정·용량 제약 위반")
        result[name] = dict(moved=int((g != ref).sum()), g1_kw=round(c1, 1), g2_kw=round(float(cap.sum() - c1), 1),
                            oof=round(pen(g)), oof_change_pct=round(100 * (pen(g) / pen(ref) - 1), 3))
        pd.DataFrame({"site_id": sites.index, "group": np.where(g, "G1", "G2")}).to_csv(SUB_DIR / f"composition_oof_{name}.csv", index=False)
        print(name, json.dumps(result[name], ensure_ascii=False), flush=True)

    folds = {k: residuals([v], sites, w_all) for k, v in FOLDS.items()}
    E1, w1 = folds["fold1"][0], folds["fold1"][1]
    rand_g = []
    while len(rand_g) < 200:
        g = np.where(free, random_feasible(cap, rng), ref)
        if MIN_GROUP_KW <= cap[g].sum() <= cap.sum() - MIN_GROUP_KW:
            rand_g.append(g)
    rand1 = np.array([composition_penalty(E1["forecast_kwh"], w1, g) for g in rand_g])
    ref1 = composition_penalty(E1["forecast_kwh"], w1, ref)
    result["fold1_random200"] = dict(q5_50_95_pct=[round(float(100 * (x / ref1 - 1)), 3) for x in np.percentile(rand1, [5, 50, 95])],
                                     ref_position_pct=round(float(100 * (rand1 < ref1).mean()), 1))
    # 진단: OOF 이력이 없는 48개소를 뺀 fold1. 판정에 쓰지 않는다
    sub = lambda Es, g: composition_penalty(Es[:, free], w1, g[free])
    rand_sub = np.array([sub(E1["forecast_kwh"], g) for g in rand_g])
    ref_sub = sub(E1["forecast_kwh"], ref)
    result["diag_fold1_without_new"] = dict(sites=int(free.sum()), ref_position_pct=round(float(100 * (rand_sub < ref_sub).mean()), 1),
                                            random200_q5_50_95_pct=[round(float(100 * (x / ref_sub - 1)), 3) for x in np.percentile(rand_sub, [5, 50, 95])])
    for name, g in cands.items():
        r = result[name]
        for k, (Es, wk, ts, _) in folds.items():
            for c in COLS:
                base, x = composition_penalty(Es[c], wk, ref), composition_penalty(Es[c], wk, g)
                r[f"{k}_{c[-2:] if c != 'forecast_kwh' else 'mean'}_pct"] = round(100 * (x / base - 1), 3)
            r[f"{k}_ref"] = round(composition_penalty(Es["forecast_kwh"], wk, ref))
            base = hourly(Es["forecast_kwh"], wk, ref)
            r[f"{k}_block95_pct"] = block_ci(hourly(Es["forecast_kwh"], wk, g) - base, ts, base, rng)
        r["fold1_random_position_pct"] = round(float(100 * (rand1 < composition_penalty(E1["forecast_kwh"], w1, g)).mean()), 1)
        r["diag_without_new_pct"] = round(100 * (sub(E1["forecast_kwh"], g) / ref_sub - 1), 3)
        r["checks"] = {"fold1_gt_1pct": r["fold1_mean_pct"] < -1, "seeds_same_direction": all(r[f"fold1_{s}_pct"] < 0 for s in ("s0", "s1", "s2")),
                       "fold2_not_worse": r["fold2_mean_pct"] <= 0, "random_bottom5": r["fold1_random_position_pct"] < 5}
        r["passed"] = all(r["checks"].values())
        print(name, json.dumps(r, ensure_ascii=False), flush=True)
    out = EXP_DIR / "p36_assign_gfs60"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print("무작위·진단", json.dumps({k: result[k] for k in ("fold1_random200", "diag_fold1_without_new")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
