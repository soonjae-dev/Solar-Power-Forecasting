"""수준 중립 재평가. 구간마다 각 모델을 최적 상수 배율로 맞춘 뒤 페널티를 비교한다. 집계만 저장한다."""
import gc
import json
import traceback

import numpy as np
import pandas as pd

from .. import train as T
from ..cb_onehot import checked_input, make_model
from ..config import EXP_DIR
from ..data import load_generation, load_sites, load_smp
from ..direct_groups import metrics
from .gfs_day_model import day_context
from ..gfs_more import load_more
from ..gfs_v3 import checked_gfs
from ..ldaps_model import load_ldaps
from ..profile_target import base_target
from ..provider_ensemble import grouped_prediction
from ..splits import fold
from ..tune_v3 import END, evaluation_context, evaluation_groups, save
from ..weather_teacher import CFG

PATH = EXP_DIR / "p87_weight_neutral" / "metrics.json"
C11 = dict(num_leaves=255, min_child_samples=500, learning_rate=0.03, n_estimators=1200, colsample_bytree=0.5)


def pen(a, f, smp):
    w = smp.reindex(a.index.get_level_values("ts"))["w"].to_numpy()
    return float((0.1 * w * (f - a).abs()).sum())


def main():
    PATH.parent.mkdir(parents=True, exist_ok=True)
    result = json.loads(PATH.read_text()) if PATH.exists() else dict(rule="14.87", seed=2026, results={})
    gen = load_generation(); gen = gen.loc[gen.ts <= END]
    sites = load_sites().set_index("site_id"); smp = load_smp("train").set_index("ts")
    d = T.load_train_features(); base = T.select_features(d, CFG["feats"])
    d = d.loc[d.ts <= END, ["site_id", "ts", "energy", *base]].copy(); T._cache.clear()
    g0 = checked_gfs()[0]
    gfs = pd.concat([g0, load_more()[0]], axis=1)
    ld = load_ldaps(last=f"{END - pd.Timedelta(hours=1):%Y-%m-%d}")[0].reindex(gfs.index)
    dc = day_context(g0)
    common = pd.concat([gfs, ld, dc], axis=1)
    cols = {"ctrl": base + list(gfs) + list(ld), "day": base + list(gfs) + list(ld) + list(dc),
            "gfs": base + list(gfs), "ldaps": base + list(ld)}
    for cutoff, names in evaluation_groups().items():
        if all(n in result["results"] for n in names):
            continue
        q = T.fold_features(d, CFG, cutoff)
        tr = q.loc[q.ts <= cutoff].copy()
        val = q.loc[q.ts.between(min(fold(n)["val_start"] for n in names), max(fold(n)["val_end"] for n in names))].copy(); del q
        tr[list(common)] = common.reindex(tr.ts).to_numpy(); val[list(common)] = common.reindex(val.ts).to_numpy()
        y = base_target(tr)
        sc = T.clear_sky(val.cosz) * val.cap.to_numpy()
        P = {}
        for k, kind in (("ctrl", "cap"), ("capk", "capk"), ("capsmpk", "capsmpk")):
            w = T.kt_weight(T.clear_sky(tr.cosz), T.sample_weight(tr, kind, smp.smp), CFG)
            m = T.fit_model({**CFG, "seed": 2026}, tr[cols["ctrl"]], y, w)
            P[k] = np.maximum(m.predict(val[cols["ctrl"]]), 0) * sc; del m; gc.collect()
        print(f"{cutoff:%Y-%m-%d} 학습 완료", flush=True)
        for name in names:
            ctx = evaluation_context(name, gen, sites); a = ctx["a"]
            f = fold(name); mask = val.ts.between(f["val_start"], f["val_end"]).to_numpy()
            G = {k: grouped_prediction(ctx["actual"], val.loc[mask, ["site_id", "ts"]], v[mask]).reindex(a.index) for k, v in P.items()}
            row = {}
            for k, g in G.items():
                s = float(a.sum() / g.sum())
                row[k] = dict(raw=round(pen(a, g, smp)), scale=round(s, 4), neutral=round(pen(a, g * s, smp)))
            result["results"][name] = row
            save(PATH, result)
            print(name, json.dumps({k: (v["scale"], round(100 * (v["raw"] / row["ctrl"]["raw"] - 1), 2), round(100 * (v["neutral"] / row["ctrl"]["neutral"] - 1), 2)) for k, v in row.items() if True}), flush=True)
        del tr, val; gc.collect()
    seasons = [n for n in result["results"] if n != "fold1_static"]
    summ = {}
    for k in ("capk", "capsmpk"):
        for kind in ("raw", "neutral"):
            ch = [100 * (result["results"][n][k][kind] / result["results"][n]["ctrl"][kind] - 1) for n in seasons]
            tot = 100 * (sum(result["results"][n][k][kind] for n in seasons) / sum(result["results"][n]["ctrl"][kind] for n in seasons) - 1)
            st = 100 * (result["results"]["fold1_static"][k][kind] / result["results"]["fold1_static"]["ctrl"][kind] - 1)
            summ[f"{k}_{kind}"] = dict(seasonal=round(tot, 3), improved=sum(c < 0 for c in ch), worst=round(max(ch), 3), static=round(st, 3),
                                       passed=bool(tot <= -1 and sum(c < 0 for c in ch) >= 3 and max(ch) <= 3 and st <= 0))
    result["summary"] = summ; result["complete"] = True
    save(PATH, result)
    for k, v in summ.items(): print(k, json.dumps(v), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps(dict(status="failed", error_type=type(exc).__name__,
                              code_frames=[dict(function=f.name, line=f.lineno) for f in traceback.extract_tb(exc.__traceback__) if "/src/" in f.filename])))
        raise SystemExit(1) from None
