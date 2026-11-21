"""CatBoost(p53 설정)에 LDAPS 10열을 더한 효과를 1시드로 검증한다. 집계만 저장한다."""
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
from ..gfs_more import load_more
from ..gfs_v3 import checked_gfs
from ..ldaps_model import load_ldaps
from ..profile_target import base_target
from ..provider_ensemble import grouped_prediction
from ..splits import fold
from ..tune_v3 import END, WINDOWS, evaluation_context, evaluation_groups, save
from ..weather_teacher import CFG

PATH = EXP_DIR / "p76_cb_ldaps" / "metrics.json"


def main():
    PATH.parent.mkdir(parents=True, exist_ok=True)
    result = json.loads(PATH.read_text()) if PATH.exists() else dict(rule="14.76", seed=2026, results={})
    gen = load_generation(); gen = gen.loc[gen.ts <= END]
    sites = load_sites().set_index("site_id")
    smp = load_smp("train").set_index("ts")
    d = T.load_train_features()
    base_feats = T.select_features(d, CFG["feats"])
    d = d.loc[d.ts <= END, ["site_id", "ts", "energy", *base_feats]].copy()
    T._cache.clear()
    common = pd.concat([checked_gfs()[0], load_more()[0]], axis=1)
    ld = load_ldaps(last=f"{END - pd.Timedelta(hours=1):%Y-%m-%d}")[0]
    common = pd.concat([common, ld.reindex(common.index)], axis=1)
    f0 = base_feats + [c for c in common if not c.startswith("ldaps_")]
    f1 = base_feats + list(common)
    if (len(f0), len(f1)) != (144, 154):
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
        preds = {}
        for tag, feats in (("cb", f0), ("cb_ldaps", f1)):
            m = make_model(2026)
            m.fit(checked_input(tr[feats]), y, sample_weight=w, cat_features=["site"])
            preds[tag] = np.maximum(m.predict(checked_input(val[feats])), 0) * scale
            del m; gc.collect()
            print(f"{cutoff:%Y-%m-%d} {tag} 완료", flush=True)
        for name in names:
            ctx = evaluation_context(name, gen, sites)
            f = fold(name); mask = val.ts.between(f["val_start"], f["val_end"]).to_numpy()
            g = {k: grouped_prediction(ctx["actual"], val.loc[mask, ["site_id", "ts"]], v[mask]).reindex(ctx["a"].index) for k, v in preds.items()}
            r = metrics(ctx["a"], g["cb"], g["cb_ldaps"], smp)
            result["results"][name] = r
            save(PATH, result)
            print(name, json.dumps({k: r[k] for k in ("base_penalty", "direct_penalty", "change_pct", "change_ci90_pct")}), flush=True)
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
