"""v6 집합 오차를 마감 전 상태 코호트로 나눠 집계한다. 재학습·행 저장 없음, 2025-05 제외.

    MPLCONFIGDIR=/tmp/oibc-mpl .venv/bin/python -m src.experiments.v6_cohort
"""
import json
import traceback

import numpy as np
import pandas as pd

from .. import train as T
from ..config import EXP_DIR
from ..data import load_generation, load_sites, load_smp
from ..direct_groups import metrics
from ..profile_target import base_target
from ..splits import fold
from ..tune_v3 import END, evaluation_context, save
from ..v6_error_budget import history_cohort, load_reference
from ..weather_teacher import CFG

PATH = EXP_DIR/"p62_v6_cohort"/"metrics.json"
SEASONS = ("summer_support", "autumn_support", "development", "selection")
AXES = ("age", "recent", "zero_days", "capacity")


def recent_state(d, cutoff):
    """마감 전 30일 일별 kt 중앙값 ÷ 그 이전 90일 중앙값으로 3단계 분류."""
    p = d.loc[(d.ts > cutoff-pd.Timedelta(days=120)) & (d.ts <= cutoff), ["site_id", "ts", "energy", "cap", "cosz"]].copy()
    p["kt"] = base_target(p)
    p["day"] = (p.ts-pd.Timedelta(hours=1)).dt.normalize()
    daily = p.groupby(["site_id", "day"]).kt.median().reset_index()
    split = (cutoff-pd.Timedelta(days=30)).normalize()
    a = daily.loc[daily.day > split].groupby("site_id").kt.agg(["median", "size"])
    b = daily.loc[daily.day <= split].groupby("site_id").kt.agg(["median", "size"]).reindex(a.index)
    ratio = a["median"]/b["median"].replace(0, np.nan)
    ok = (a["size"] >= 20) & (b["size"] >= 20) & ratio.notna()
    label = np.select([ratio < .5, ratio < .9], ["recent_low", "recent_dip"], default="normal")
    return pd.Series(label, index=a.index).where(ok, "unknown")


def zero_days(gen, cutoff):
    g = gen.loc[(gen.ts > cutoff-pd.Timedelta(days=60)) & (gen.ts <= cutoff)].dropna(subset=["energy"])
    daily = g.groupby(["site_id", (g.ts-pd.Timedelta(hours=1)).dt.normalize()]).energy.sum()
    n = (daily == 0).groupby(level=0).sum()
    return pd.Series(np.select([n == 0, n < 7], ["none", "some"], default="many"), index=n.index)


def contributions(p, w):
    """발전소 오차의 집합 기여. 합계는 집합 페널티와 같다."""
    e = p.forecast_kwh-p.energy
    E = e.groupby([p.ts, p.group]).transform("sum")
    c = .1*w*e*np.sign(E)
    return e, c


def group_penalty(p, e, w):
    E = e.groupby([p.ts, p.group]).sum()
    return float((.1*w.groupby([p.ts, p.group]).first()*E.abs()).sum())


def cell(p, e, c, w, mask, total_pen, cap_total):
    sites = p.loc[mask, "site_id"].unique()
    if len(sites) < 5:
        return dict(site_count=int(len(sites)), suppressed=True)
    rows = p.loc[mask]
    actual = float(rows.energy.sum())
    oracle = group_penalty(p, e.where(~mask, 0.), w)
    return dict(site_count=int(len(sites)), suppressed=False,
                capacity_share=round(float(p.drop_duplicates("site_id").set_index("site_id").cap.reindex(sites).sum()/cap_total), 4),
                contribution_share=round(float(c[mask].sum()/total_pen), 4),
                contribution_won=round(float(c[mask].sum())),
                ratio=round(float(rows.forecast_kwh.sum()/actual), 4) if actual else None,
                bias_per_cap=round(float((e[mask]*w[mask]).sum()/(rows.cap*w[mask]).sum()), 6),
                oracle_reduction_won=round(total_pen-oracle))


def window(name, gen, d, sites, smp, expected):
    cutoff = fold(name)["cutoff"]
    ctx = evaluation_context(name, gen, sites)
    p = load_reference(name, ctx["actual"], sites).reset_index(drop=True)
    base = p.groupby(["ts", "group"]).forecast_kwh.sum().reindex(ctx["a"].index)
    if metrics(ctx["a"], base, base, smp)["base_penalty"] != expected:
        raise ValueError("v6 대조 불일치")
    first = gen.loc[gen.ts <= cutoff].dropna(subset=["energy"]).groupby("site_id").ts.min().reindex(sites.index)
    labels = dict(age=history_cohort(first, cutoff), recent=recent_state(d, cutoff), zero_days=zero_days(gen, cutoff),
                  capacity=pd.qcut(sites.capacity_kw, 4, labels=["q1", "q2", "q3", "q4"]).astype(str))
    for axis, s in labels.items():
        p[axis] = p.site_id.map(s).fillna("unseen" if axis == "age" else "unknown")
    w = p.ts.map(smp.w)
    e, c = contributions(p, w)
    total = group_penalty(p, e, w)
    if not np.isclose(c.sum(), total, rtol=1e-9) or round(total) != expected:
        raise ValueError("기여도 합계 불일치")
    cap_total = float(p.drop_duplicates("site_id").cap.sum())
    out = dict(penalty=round(total), group_share={g: round(float(c[p.group == g].sum()/total), 4) for g in ("A", "B")},
               group_ratio={g: round(float(p.loc[p.group == g].forecast_kwh.sum()/p.loc[p.group == g].energy.sum()), 4) for g in ("A", "B")},
               cohorts={})
    for axis in AXES:
        for k in sorted(p[axis].unique()):
            m = p[axis] == k
            out["cohorts"][f"{axis}:{k}"] = cell(p, e, c, w, m.to_numpy(), total, cap_total)
            for g in ("A", "B"):
                out["cohorts"][f"{axis}:{k}:{g}"] = cell(p, e, c, w, (m & (p.group == g)).to_numpy(), total, cap_total)
    return out


def promotion(result):
    """14.58 승격 조건을 4계절 합산으로 판정한다."""
    seasonal = sum(result["windows"][n]["penalty"] for n in SEASONS)
    keys = set().union(*(result["windows"][n]["cohorts"] for n in SEASONS))
    out = {}
    for k in sorted(keys):
        cells = [result["windows"][n]["cohorts"].get(k, dict(suppressed=True)) for n in SEASONS]
        if any(x["suppressed"] for x in cells):
            continue
        signs = [np.sign(x["bias_per_cap"]) for x in cells]
        same = max(signs.count(1.), signs.count(-1.))
        oracle = 100*sum(x["oracle_reduction_won"] for x in cells)/seasonal
        contrib = sum(x["contribution_won"] for x in cells)/seasonal
        cap = float(np.mean([x["capacity_share"] for x in cells]))
        out[k] = dict(same_sign_seasons=int(same), oracle_reduction_pct=round(oracle, 3),
                      contribution_share=round(contrib, 4), capacity_share=round(cap, 4),
                      promote=bool(same >= 3 and oracle >= 1 and contrib >= 1.5*cap))
    return out


def main():
    PATH.parent.mkdir(parents=True, exist_ok=True)
    baseline = json.loads((EXP_DIR/"p26_gfs_more"/"metrics.json").read_text())["seeds3"]["gfs60"]
    gen = load_generation()
    gen = gen.loc[gen.ts <= END]
    sites = load_sites().set_index("site_id")
    smp = load_smp("train").set_index("ts")
    d = T.load_train_features()
    d = d.loc[d.ts <= END, ["site_id", "ts", "energy", "cap", "cosz"]].copy()
    T._cache.clear()
    result = dict(rule="14.58", windows={})
    for name in (*SEASONS, "fold1_static"):
        result["windows"][name] = window(name, gen, d, sites, smp, baseline[name]["vs_v3_3seeds"]["direct_penalty"])
        print(name, "완료", flush=True)
    result["promotion"] = promotion(result)
    result["promoted"] = [k for k, v in result["promotion"].items() if v["promote"]]
    result["complete"] = True
    save(PATH, result)
    print(json.dumps(dict(promoted=result["promoted"],
                          groups={n: dict(share=w["group_share"], ratio=w["group_ratio"]) for n, w in result["windows"].items()}),
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        frames = [dict(function=f.name, line=f.lineno) for f in traceback.extract_tb(exc.__traceback__)
                  if "/src/" in f.filename]
        print(json.dumps(dict(status="failed", error_type=type(exc).__name__, code_frames=frames)), flush=True)
        raise SystemExit(1) from None
