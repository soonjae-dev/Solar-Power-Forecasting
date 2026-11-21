"""v6+LDAPS LightGBM 하이퍼파라미터 탐색. 2024-05 마감 고정11개월에서 고르고 이후 3개 마감에서 검증한다."""
import gc
import json
import traceback

import numpy as np
import pandas as pd

from .. import train as T
from ..config import EXP_DIR
from ..data import load_generation, load_sites, load_smp
from ..direct_groups import metrics
from ..gfs_more import load_more
from ..gfs_v3 import checked_gfs
from ..ldaps_model import load_ldaps
from ..profile_target import base_target
from ..provider_ensemble import grouped_prediction
from ..splits import fold
from ..tune_v3 import END, evaluation_context, evaluation_groups, save
from ..weather_teacher import CFG

PATH = EXP_DIR / "p84_tune" / "metrics.json"
CONFIGS = json.load(open("/tmp/oibc_kma/tune_configs.json"))
DEFAULT = dict(num_leaves=127, min_child_samples=200, learning_rate=0.06, n_estimators=600, colsample_bytree=0.8)


def setup():
    gen = load_generation(); gen = gen.loc[gen.ts <= END]
    sites = load_sites().set_index("site_id"); smp = load_smp("train").set_index("ts")
    d = T.load_train_features()
    base = T.select_features(d, CFG["feats"])
    d = d.loc[d.ts <= END, ["site_id", "ts", "energy", *base]].copy()
    T._cache.clear()
    gfs = pd.concat([checked_gfs()[0], load_more()[0]], axis=1)
    common = pd.concat([gfs, load_ldaps(last=f"{END - pd.Timedelta(hours=1):%Y-%m-%d}")[0].reindex(gfs.index)], axis=1)
    return gen, sites, smp, d, base + list(common), common


def split(d, common, cutoff, names, smp):
    q = T.fold_features(d, CFG, cutoff)
    tr = q.loc[q.ts <= cutoff].copy()
    val = q.loc[q.ts.between(min(fold(n)["val_start"] for n in names), max(fold(n)["val_end"] for n in names))].copy()
    tr[list(common)] = common.reindex(tr.ts).to_numpy(); val[list(common)] = common.reindex(val.ts).to_numpy()
    y = base_target(tr)
    w = T.kt_weight(T.clear_sky(tr.cosz), T.sample_weight(tr, "cap", smp.w), CFG)
    return tr, val, y, w


def score(cfg, tr, val, y, w, cols, names, gen, sites, smp):
    m = T.fit_model({**CFG, **cfg, "seed": 2026}, tr[cols], y, w)
    p = np.maximum(m.predict(val[cols]), 0) * T.clear_sky(val.cosz) * val.cap.to_numpy()
    del m; gc.collect()
    out = {}
    for name in names:
        ctx = evaluation_context(name, gen, sites)
        f = fold(name); mask = val.ts.between(f["val_start"], f["val_end"]).to_numpy()
        out[name] = grouped_prediction(ctx["actual"], val.loc[mask, ["site_id", "ts"]], p[mask]).reindex(ctx["a"].index), ctx["a"]
    return out


def main():
    PATH.parent.mkdir(parents=True, exist_ok=True)
    result = json.loads(PATH.read_text()) if PATH.exists() else dict(rule="14.84", configs=CONFIGS, default=DEFAULT, search={})
    gen, sites, smp, d, cols, common = setup()
    groups = list(evaluation_groups().items())
    cut0, names0 = groups[0]
    tr, val, y, w = split(d, common, cut0, names0, smp)
    base = score(DEFAULT, tr, val, y, w, cols, names0, gen, sites, smp)["fold1_static"]
    b0 = metrics(base[1], base[0], base[0], smp)["base_penalty"]
    result["search"]["default"] = b0
    for i, c in enumerate(CONFIGS):
        k = f"c{i}"
        if k in result["search"]:
            continue
        g, a = score(c, tr, val, y, w, cols, names0, gen, sites, smp)["fold1_static"]
        result["search"][k] = metrics(a, base[0], g, smp)["direct_penalty"]
        save(PATH, result)
        print(k, c, f"{100 * (result['search'][k] / b0 - 1):+.3f}%", flush=True)
    del tr, val; gc.collect()
    best = min((k for k in result["search"] if k != "default"), key=lambda k: result["search"][k])
    result["best"] = best
    result["best_static_change_pct"] = round(100 * (result["search"][best] / b0 - 1), 3)
    save(PATH, result)
    print("선택", best, result["best_static_change_pct"], flush=True)
    if result["search"][best] >= b0:
        result.update(passed=False, reason="기본값보다 나은 설정 없음"); save(PATH, result); return
    bc = CONFIGS[int(best[1:])]
    val_res = {}
    for cutoff, names in groups[1:]:
        tr, val, y, w = split(d, common, cutoff, names, smp)
        dflt = score(DEFAULT, tr, val, y, w, cols, names, gen, sites, smp)
        tuned = score(bc, tr, val, y, w, cols, names, gen, sites, smp)
        for n in names:
            r = metrics(dflt[n][1], dflt[n][0], tuned[n][0], smp)
            val_res[n] = r
            print(n, r["change_pct"], r["change_ci90_pct"], flush=True)
        del tr, val; gc.collect()
    result["validation"] = val_res
    tot = 100 * (sum(r["direct_penalty"] for r in val_res.values()) / sum(r["base_penalty"] for r in val_res.values()) - 1)
    result.update(validation_change_pct=round(tot, 3),
                  passed=bool(tot <= -1 and all(r["change_pct"] < 0 for r in val_res.values()) and max(r["change_pct"] for r in val_res.values()) <= 3))
    save(PATH, result)
    print("판정", json.dumps({k: result[k] for k in ("best", "best_static_change_pct", "validation_change_pct", "passed")}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps(dict(status="failed", error_type=type(exc).__name__,
                              code_frames=[dict(function=f.name, line=f.lineno) for f in traceback.extract_tb(exc.__traceback__) if "/src/" in f.filename])))
        raise SystemExit(1) from None
