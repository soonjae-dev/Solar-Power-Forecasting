"""예측 청천도 구간별 가중 중앙값 배율의 연도 블록 교차검증. 재학습 없음, 집계만 저장.

    MPLCONFIGDIR=/tmp/oibc-mpl .venv/bin/python -m src.experiments.median_scale
"""
import json
import sys
import traceback

import numpy as np
import pandas as pd

from .. import train as T
from .cold_cv import BLOCKS
from ..config import EXP_DIR, PRED_DIR
from ..data import load_sites, load_smp
from .regime_scan import OOF, SHRINK, STATIC, penalty
from ..tune_v3 import save

PATH = EXP_DIR/"p89_median_scale"/"metrics.json"


def group_hours(names, feats, sites):
    """집합·시각별 실적·예측 합과 청천 잠재량 합."""
    p = pd.concat([pd.read_parquet(PRED_DIR/n/"predictions.parquet", columns=["site_id", "ts", "energy", "forecast_kwh"]) for n in names])
    if p.duplicated(["site_id", "ts"]).any():
        raise ValueError("예측 구간 중복")
    p = p.dropna(subset=["energy"]).merge(feats, on=["site_id", "ts"], how="left", validate="one_to_one")
    p["group"] = p.site_id.map(sites.current_group)
    p["cs"] = p.site_id.map(sites.capacity_kw)*T.clear_sky(p.cosz.fillna(0))*(p.cosz > 0)
    out = p.groupby([p.ts, p.group])[["energy", "forecast_kwh", "cs"]].sum()
    out = out[(out.energy > 0) | (out.forecast_kwh > 0)]
    out["q"] = out.forecast_kwh/out.cs.replace(0, np.nan)
    return out.reset_index()


def wmedian(x, wt):
    o = np.argsort(x)
    c = np.cumsum(wt[o])
    return float(x[o][np.searchsorted(c, c[-1]/2)])


def shrink(c, n):
    return 1+(c-1)*n/(n+SHRINK)


def fit(g, w, how):
    m = (g.forecast_kwh > 0).to_numpy() & g.q.notna().to_numpy()
    edges = np.unique(np.nanquantile(g.q[m], np.linspace(0, 1, 11)))
    b = np.clip(np.searchsorted(edges, g.q, side="right")-1, 0, len(edges)-2)
    c = {}
    for k in range(len(edges)-1):
        s = m & (b == k)
        a, p = g.energy.to_numpy()[s], g.forecast_kwh.to_numpy()[s]
        raw = wmedian(a/p, w[s]*p) if how == "median" else a.sum()/p.sum()
        c[k] = shrink(raw, s.sum())
    return edges, c


def apply(g, f):
    edges, c = f
    b = np.clip(np.searchsorted(edges, g.q, side="right")-1, 0, len(edges)-2)
    return g.forecast_kwh*np.where(g.q.isna(), 1., np.array([c[k] for k in b]))


def main():
    PATH.parent.mkdir(parents=True, exist_ok=True)
    sites = load_sites().set_index("site_id")
    w = load_smp("train").set_index("ts").w
    d = T.load_train_features()
    feats = d[["site_id", "ts", "cosz"]].copy()
    del d
    T._cache.clear()
    g = pd.concat([group_hours(OOF, feats, sites), group_hours([STATIC], feats, sites)], ignore_index=True)
    day = g.ts-pd.Timedelta(hours=1)
    block = np.full(len(g), -1)
    for k, (a, b) in enumerate(BLOCKS):
        block[((day >= pd.Timestamp(a, tz=day.dt.tz)) & (day < pd.Timestamp(b, tz=day.dt.tz))).to_numpy()] = k
    if (block < 0).any():
        raise ValueError("블록 밖 시각")
    wg = g.ts.map(w).to_numpy()
    base = penalty(g, g.forecast_kwh, wg).to_numpy()
    rule, const, mean = np.zeros(len(g)), np.zeros(len(g)), np.zeros(len(g))
    result = dict(rule="14.89", shrink=SHRINK, blocks={})
    for k in range(len(BLOCKS)):
        m = block == k
        tr, te = g[~m].reset_index(drop=True), g[m].reset_index(drop=True)
        f, fm = fit(tr, wg[~m], "median"), fit(tr, wg[~m], "mean")
        pos = (tr.forecast_kwh > 0).to_numpy()
        c = shrink(wmedian((tr.energy/tr.forecast_kwh).to_numpy()[pos], (wg[~m]*tr.forecast_kwh.to_numpy())[pos]), pos.sum())
        rule[m] = penalty(te, apply(te, f), wg[m]).to_numpy()
        mean[m] = penalty(te, apply(te, fm), wg[m]).to_numpy()
        const[m] = penalty(te, te.forecast_kwh*c, wg[m]).to_numpy()
        bb = base[m].sum()
        result["blocks"][f"{BLOCKS[k][0]}~{BLOCKS[k][1]}"] = dict(
            group_hours=int(m.sum()), base=int(round(float(bb))), rule_change_pct=round(100*(rule[m].sum()/bb-1), 3),
            const_change_pct=round(100*(const[m].sum()/bb-1), 3), mean_change_pct=round(100*(mean[m].sum()/bb-1), 3),
            const_factor=round(float(c), 4), q_edges=[round(float(x), 3) for x in f[0]],
            factors=[round(float(x), 4) for x in f[1].values()], mean_factors=[round(float(x), 4) for x in fm[1].values()])
    b, r, q, mn = base.sum(), rule.sum(), const.sum(), mean.sum()
    s3 = block < 3
    result.update(total_vs_v6_pct=round(100*(r/b-1), 3), total_vs_const_pct=round(100*(r/q-1), 3),
                  const_vs_v6_pct=round(100*(q/b-1), 3), mean_vs_v6_pct=round(100*(mn/b-1), 3),
                  blocks1_3_vs_v6_pct=round(100*(rule[s3].sum()/base[s3].sum()-1), 3),
                  improved_blocks=int(sum(v["rule_change_pct"] < 0 for v in result["blocks"].values())))
    result["passed"] = bool(result["total_vs_v6_pct"] <= -1 and result["total_vs_const_pct"] <= -.5
                            and result["improved_blocks"] >= 3 and result["blocks1_3_vs_v6_pct"] <= -1)
    result["complete"] = True
    save(PATH, result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


def build():
    """후보 13: 후보 9에 수준 중립 중앙값 배율을 곱한다."""
    from ..config import CACHE_DIR, SUB_DIR
    from ..submission_log import register
    from ..validate_submission import validate
    sites = load_sites().set_index("site_id")
    w = load_smp("train").set_index("ts").w
    d = T.load_train_features()
    feats = d[["site_id", "ts", "cosz"]].copy()
    del d
    T._cache.clear()
    g = pd.concat([group_hours(OOF, feats, sites), group_hours([STATIC], feats, sites)], ignore_index=True)
    f = fit(g, g.ts.map(w).to_numpy(), "median")
    level = apply(g, f).sum()/g.forecast_kwh.sum()
    f = (f[0], {k: v/level for k, v in f[1].items()})
    src = SUB_DIR/"p9_p6_ldaps_nolimit"
    comp = pd.read_csv(src/"composition.csv")
    p = pd.read_parquet(src/"site_forecast_daylight.parquet")
    test = pd.read_parquet(CACHE_DIR/"test_features.parquet", columns=["site_id", "ts", "cosz", "cap"])
    p = p.merge(test, on=["site_id", "ts"], how="left", validate="one_to_one")
    if p.cosz.isna().any():
        raise ValueError("cosz 결측")
    p["group"] = p.site_id.map(comp.set_index("site_id").group)
    p["cs"] = p.cap*T.clear_sky(p.cosz)*(p.cosz > 0)
    gp = p.groupby(["group", "ts"])[["kwh", "cs"]].sum().reset_index().rename(columns={"kwh": "forecast_kwh"})
    gp["q"] = gp.forecast_kwh/gp.cs.replace(0, np.nan)
    gp["new"] = np.clip(apply(gp, f), 0, None)
    old = pd.read_csv(src/"forecast.csv")
    old["key"] = pd.to_datetime(old.ts)
    gp["key"] = gp.ts
    fc = old.merge(gp[["group", "key", "forecast_kwh", "new"]], on=["group", "key"], how="left", validate="one_to_one")
    if not np.allclose(fc.forecast_kwh_y.fillna(0), fc.forecast_kwh_x, atol=1e-6):
        raise ValueError("후보 9 집합 합산 불일치")
    fc["forecast_kwh"] = fc.new.fillna(fc.forecast_kwh_x)
    name = "p13_p9_median_shape_nolimit"
    out = SUB_DIR/name
    out.mkdir(parents=True, exist_ok=True)
    fc[["group", "ts", "forecast_kwh"]].to_csv(out/"forecast.csv", index=False)
    r = (gp.new/gp.forecast_kwh.replace(0, np.nan)).fillna(1.).set_axis(pd.MultiIndex.from_frame(gp[["group", "ts"]]))
    p["kwh"] = p.kwh*r.reindex(pd.MultiIndex.from_frame(p[["group", "ts"]])).to_numpy()
    p[["site_id", "ts", "kwh"]].to_parquet(out/"site_forecast_daylight.parquet", index=False)
    comp.to_csv(out/"composition.csv", index=False)
    validate(out/"composition.csv", out/"forecast.csv")
    info = dict(rule="14.90", base="p9_p6_ldaps_nolimit", level_divisor=round(float(level), 4),
                factors=[round(float(x), 4) for x in f[1].values()], q_edges=[round(float(x), 3) for x in f[0]],
                total_vs_p9=round(float(fc.forecast_kwh.sum()/fc.forecast_kwh_x.sum()), 4))
    (out/"manifest.json").write_text(json.dumps(info, ensure_ascii=False, indent=1))
    register(name, "python -m src.experiments.median_scale build", "14.90 후보 13")
    print(json.dumps(info, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    try:
        build() if sys.argv[1:] == ["build"] else main()
    except Exception as exc:
        frames = [dict(function=f.name, line=f.lineno) for f in traceback.extract_tb(exc.__traceback__)
                  if "/src/" in f.filename]
        print(json.dumps(dict(status="failed", error_type=type(exc).__name__, code_frames=frames)), flush=True)
        raise SystemExit(1) from None
