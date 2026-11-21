"""CatBoost 최근 마감 악화 원인 진단. 여름·봄 마감 비교, 집계와 열 이름만 저장. 2025-05 제외.

    OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/oibc-mpl PYTHONWARNINGS=ignore caffeinate -i -s .venv/bin/python -m src.experiments.cb_recent
"""
import gc
import json
import traceback

import numpy as np
import pandas as pd

from .. import train as T
from ..cb_onehot import SPEC, checked_input, make_model
from ..config import EXP_DIR
from ..data import load_generation, load_sites, load_smp
from ..direct_groups import metrics
from ..gfs_more import load_more
from ..gfs_v3 import checked_gfs
from ..profile_target import base_target
from ..splits import fold
from ..tune_v3 import END, evaluation_context, save
from ..v6_error_budget import load_reference
from ..weather_teacher import CFG

PATH = EXP_DIR/"p63_cb_recent"/"metrics.json"
WINDOWS = ("summer_support", "selection")
SOLAR = ("cosz", "elev", "azim", "clear", "solar", "zen")
CALENDAR = ("doy", "hour", "month", "dow", "week", "day", "year")


def family(name, gfs):
    if name == "site":
        return "site"
    if name in gfs:
        return "gfs"
    if name.endswith("_nbrc"):
        return "nbrc"
    if name.startswith("wa_"):
        return "forecast_A"
    if name.startswith("wb_"):
        return "forecast_B"
    if any(k in name for k in SOLAR):
        return "solar"
    if any(k in name for k in CALENDAR):
        return "calendar"
    return "other"


def ratios(frame, pred_cols, by):
    out = {}
    for k, v in frame.groupby(by):
        actual = float(v.energy.sum())
        out[str(k) if not isinstance(k, tuple) else ":".join(map(str, k))] = {
            c: round(float(v[c].sum()/actual), 4) if actual > 0 else None for c in pred_cols}
    return out


def importance(values, features, gfs):
    s = pd.Series(np.asarray(values, float), index=features)
    s = s/s.sum()
    fam = s.groupby([family(c, gfs) for c in s.index]).sum().round(4).to_dict()
    return dict(families=fam, top15={k: round(float(v), 4) for k, v in s.nlargest(15).items()})


def run_window(name, d, common, gen, sites, smp, p53, features):
    f = fold(name)
    cutoff = f["cutoff"]
    q = T.fold_features(d, CFG, cutoff)
    tr = q.loc[q.ts <= cutoff].copy()
    val = q.loc[q.ts.between(f["val_start"], f["val_end"])].copy()
    del q
    tr[list(common)] = common.reindex(tr.ts).to_numpy()
    val[list(common)] = common.reindex(val.ts).to_numpy()
    X, V = checked_input(tr[features]), checked_input(val[features])
    y = base_target(tr)
    weight = T.kt_weight(T.clear_sky(tr.cosz), T.sample_weight(tr, "cap", smp.w), CFG)
    cb_val = np.zeros(len(val))
    for seed in SPEC["seeds"]:
        model = make_model(seed)
        model.fit(X, y, sample_weight=weight, cat_features=["site"])
        cb_val += model.predict(V)/3
        if seed == SPEC["seeds"][0]:
            cb_imp = model.get_feature_importance()
            cb_tr = model.predict(X)
        del model
        gc.collect()
        print(f"{name} CatBoost seed={seed} 완료", flush=True)
    lgbm = T.fit_model({**CFG, "seed": SPEC["seeds"][0]}, tr[features], y, weight)
    lgb_imp = lgbm.booster_.feature_importance("gain")
    lgb_tr = lgbm.predict(tr[features])
    del lgbm, X, V
    gc.collect()
    print(f"{name} LightGBM 완료", flush=True)

    # 검증: p53 재현 후 v6 패널에 붙인다
    scale_val = T.clear_sky(val.cosz)*val.cap.to_numpy()
    val["cb"] = np.maximum(cb_val, 0)*scale_val
    val["scale"] = scale_val
    ctx = evaluation_context(name, gen, sites)
    p = load_reference(name, ctx["actual"], sites)
    p = p.merge(val[["site_id", "ts", "cb", "scale"]], on=["site_id", "ts"], how="left", validate="one_to_one")
    p["cb"] = p.cb.fillna(0.)
    g = p.groupby(["ts", "group"])[["energy", "forecast_kwh", "cb"]].sum()
    m = metrics(g.energy, g.forecast_kwh, g.cb, smp)
    if m["direct_penalty"] != p53[name]["direct_penalty"] or m["base_penalty"] != p53[name]["base_penalty"]:
        raise ValueError("p53 재현 불일치")
    p["half"] = p.ts.dt.strftime("%Y-%m") + np.where((p.ts-pd.Timedelta(hours=1)).dt.day <= 15, "a", "b")
    kt = (p.forecast_kwh/p.scale.replace(0, np.nan))
    p["v6_kt"] = np.select([kt.isna(), kt < .3, kt < .7], ["night", "low", "mid"], default="high")
    out = dict(train_cutoff=cutoff.isoformat(), train_rows=len(tr), cb_change_pct=m["change_pct"],
               d1_half_month=ratios(p, ["forecast_kwh", "cb"], ["group", "half"]),
               d3_v6_kt=ratios(p.loc[p.v6_kt != "night"], ["forecast_kwh", "cb"], ["group", "v6_kt"]))

    # 학습 적합(표본 내, seed2026)
    scale_tr = T.clear_sky(tr.cosz)*tr.cap.to_numpy()
    t = pd.DataFrame(dict(energy=tr.energy.to_numpy(), cb=np.maximum(cb_tr, 0)*scale_tr, lgbm=np.maximum(lgb_tr, 0)*scale_tr,
                          group=tr.site_id.map(sites.current_group).to_numpy(), ts=tr.ts.to_numpy()))
    t = t.dropna(subset=["energy"])
    t["year"] = (t.ts-pd.Timedelta(hours=1)).dt.year.astype(str)
    out["d2_train_year"] = ratios(t, ["cb", "lgbm"], ["group", "year"])
    out["d2_train_last90"] = ratios(t.loc[t.ts > cutoff-pd.Timedelta(days=90)], ["cb", "lgbm"], "group")
    gfs = set(common)
    out["d4_importance"] = dict(catboost=importance(cb_imp, features, gfs), lightgbm=importance(lgb_imp, features, gfs))
    return out


def main():
    PATH.parent.mkdir(parents=True, exist_ok=True)
    p53 = json.loads((EXP_DIR/"p53_cb_onehot"/"metrics.json").read_text())["results"]
    gen = load_generation()
    gen = gen.loc[gen.ts <= END]
    sites = load_sites().set_index("site_id")
    smp = load_smp("train").set_index("ts")
    d = T.load_train_features()
    features = T.select_features(d, CFG["feats"])
    d = d.loc[d.ts <= END, ["site_id", "ts", "energy", *features]].copy()
    T._cache.clear()
    common = pd.concat([checked_gfs()[0], load_more()[0]], axis=1)
    features = features + list(common)
    if len(features) != 144:
        raise ValueError("입력 수 불일치")
    fam = pd.Series([family(c, set(common)) for c in features]).value_counts().to_dict()
    result = dict(rule="14.59", feature_families=fam,
                  other_features=[c for c in features if family(c, set(common)) == "other"], windows={})
    print("계열 열 수", json.dumps(fam, ensure_ascii=False), flush=True)
    for name in WINDOWS:
        result["windows"][name] = run_window(name, d, common, gen, sites, smp, p53, features)
        save(PATH, result)
    result["complete"] = True
    save(PATH, result)
    print("완료", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        frames = [dict(function=f.name, line=f.lineno) for f in traceback.extract_tb(exc.__traceback__)
                  if "/src/" in f.filename]
        print(json.dumps(dict(status="failed", error_type=type(exc).__name__, code_frames=frames)), flush=True)
        raise SystemExit(1) from None
