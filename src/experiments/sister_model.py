"""원천별 자매 모델(GFS 모델·LDAPS 모델) 결합 vs 단일 통합 모델. 1시드, 집계만 저장한다."""
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

PATH = EXP_DIR / "p83_sister" / "metrics.json"


def main():
    PATH.parent.mkdir(parents=True, exist_ok=True)
    result = json.loads(PATH.read_text()) if PATH.exists() else dict(rule="14.83", seed=2026, results={})
    gen = load_generation(); gen = gen.loc[gen.ts <= END]
    sites = load_sites().set_index("site_id"); smp = load_smp("train").set_index("ts")
    d = T.load_train_features()
    base = T.select_features(d, CFG["feats"])
    d = d.loc[d.ts <= END, ["site_id", "ts", "energy", *base]].copy()
    T._cache.clear()
    gfs = pd.concat([checked_gfs()[0], load_more()[0]], axis=1)
    ld = load_ldaps(last=f"{END - pd.Timedelta(hours=1):%Y-%m-%d}")[0].reindex(gfs.index)
    common = pd.concat([gfs, ld], axis=1)
    sets = {"all": base + list(common), "gfs": base + list(gfs), "ldaps": base + list(ld)}
    if [len(v) for v in sets.values()] != [154, 144, 94]:
        raise ValueError("입력 수 불일치")
    for cutoff, names in evaluation_groups().items():
        if all(n in result["results"] for n in names):
            continue
        q = T.fold_features(d, CFG, cutoff)
        tr = q.loc[q.ts <= cutoff].copy()
        val = q.loc[q.ts.between(min(fold(n)["val_start"] for n in names), max(fold(n)["val_end"] for n in names))].copy()
        del q
        tr[list(common)] = common.reindex(tr.ts).to_numpy(); val[list(common)] = common.reindex(val.ts).to_numpy()
        y = base_target(tr)
        w = T.kt_weight(T.clear_sky(tr.cosz), T.sample_weight(tr, "cap", smp.w), CFG)
        scale = T.clear_sky(val.cosz) * val.cap.to_numpy()
        pred = {}
        for k, cols in sets.items():
            m = T.fit_model({**CFG, "seed": 2026}, tr[cols], y, w)
            pred[k] = np.maximum(m.predict(val[cols]), 0) * scale
            del m; gc.collect()
            print(f"{cutoff:%Y-%m-%d} {k} 완료", flush=True)
        pred["sister"] = 0.5 * pred["gfs"] + 0.5 * pred["ldaps"]
        for name in names:
            ctx = evaluation_context(name, gen, sites)
            f = fold(name); mask = val.ts.between(f["val_start"], f["val_end"]).to_numpy()
            g = {k: grouped_prediction(ctx["actual"], val.loc[mask, ["site_id", "ts"]], v[mask]).reindex(ctx["a"].index) for k, v in pred.items()}
            r = metrics(ctx["a"], g["all"], g["sister"], smp)
            r["ldaps_only_vs_all_pct"] = metrics(ctx["a"], g["all"], g["ldaps"], smp)["change_pct"]
            r["gfs_only_vs_all_pct"] = metrics(ctx["a"], g["all"], g["gfs"], smp)["change_pct"]
            result["results"][name] = r
            save(PATH, result)
            print(name, json.dumps({k: r[k] for k in ("change_pct", "change_ci90_pct", "ldaps_only_vs_all_pct", "gfs_only_vs_all_pct")}), flush=True)
        del tr, val; gc.collect()
    seasons = [r for n, r in result["results"].items() if n != "fold1_static"]
    ch = 100 * (sum(r["direct_penalty"] for r in seasons) / sum(r["base_penalty"] for r in seasons) - 1)
    result.update(seasonal_change_pct=round(ch, 3), improved=sum(r["change_pct"] < 0 for r in seasons),
                  passed=bool(ch <= -1 and sum(r["change_pct"] < 0 for r in seasons) >= 3 and max(r["change_pct"] for r in seasons) <= 3
                              and result["results"]["fold1_static"]["change_pct"] <= 0), complete=True)
    save(PATH, result)
    print("판정", json.dumps({k: result[k] for k in ("seasonal_change_pct", "improved", "passed")}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps(dict(status="failed", error_type=type(exc).__name__,
                              code_frames=[dict(function=f.name, line=f.lineno) for f in traceback.extract_tb(exc.__traceback__) if "/src/" in f.filename])))
        raise SystemExit(1) from None
