"""p60: provided forecast snow accumulation audit and two earlier winters.

Only aggregate metrics are persisted. No forecast rows/models/submissions are written.
"""
import argparse
import gc
import json
from pathlib import Path
import time
import traceback
import numpy as np
import pandas as pd

from .. import train as T
from ..config import EXP_DIR, TZ
from ..data import load_forecast, load_generation, load_sites, load_smp
from ..gfs_more import load_more
from ..gfs_v3 import checked_gfs
from ..profile_target import base_target
from ..provider_ensemble import grouped_prediction
from .sequence_residual import sequence_fold_features
from ..tune_v3 import save
from ..weather_teacher import CFG

COLS = [f"{p}_snow_sum{h}" for p in ("wa", "wb") for h in (24, 72)]
WINTERS = {"winter2022": ["2022-12-01", "2023-04-01"],
           "winter2023": ["2023-12-01", "2024-04-01"]}
SPEC = dict(rule="14.55", seeds=[2026, 2027, 2028], windows=WINTERS,
            gate="wa_snow_sum72>0 or wb_snow_sum72>0", baseline="v6_ref", threads=4)
PATH = EXP_DIR / "p60_snowgate" / "metrics.json"


class SnowError(ValueError):
    pass


def issued(ts, rounds):
    return (ts-pd.Timedelta(hours=1)).dt.normalize()-pd.Timedelta(days=1)+pd.to_timedelta(rounds, unit="h")


def rolling_snow(raw, provider):
    st, field = f"{provider}_station", f"{provider}_snow"
    if not raw["round"].isin([10, 17]).all():
        raise SnowError("unexpected_round")
    exact = raw.ts == raw.ts.dt.floor("h")
    f = raw.loc[exact].copy()
    if f.duplicated([st, "round", "ts"]).any():
        raise SnowError("duplicate_forecast_keys")
    if (f[field].dropna() < 0).any():
        raise SnowError("negative_snow")
    grid = pd.MultiIndex.from_product([sorted(f[st].unique()),
            pd.date_range(f.ts.min(), f.ts.max(), freq="h")], names=[st, "ts"])
    a = f.loc[f["round"] == 17].set_index([st, "ts"])[field].reindex(grid)
    b = f.loc[f["round"] == 10].set_index([st, "ts"])[field].reindex(grid)
    chosen = a.fillna(b)
    rounds = np.where(a.notna(), 17, 10)
    ts = pd.Series(grid.get_level_values("ts"), index=grid)
    # Seconds retain exact hour precision through rolling float conversions.
    issue = pd.Series(issued(ts, rounds).astype("int64").to_numpy()/1e9, index=grid).where(chosen.notna())
    cutoff = issued(ts, np.full(len(ts), 17)).astype("int64").to_numpy()/1e9
    out = pd.DataFrame(index=grid)
    audit = dict(source_rows=len(raw), nonhourly_rows=int((~exact).sum()),
                 fallback10_rows=int((a.isna() & b.notna()).sum()), missing_selected_rows=int(chosen.isna().sum()))
    for h in (24, 72):
        g = chosen.groupby(level=0, sort=False)
        out[f"{provider}_snow_sum{h}"] = g.transform(lambda x: x.rolling(h, min_periods=1).sum())
        latest = issue.groupby(level=0, sort=False).transform(lambda x: x.rolling(h, min_periods=1).max())
        count = g.transform(lambda x: x.rolling(h, min_periods=1).count())
        violations = int((latest.to_numpy() > cutoff).sum())
        audit[str(h)] = dict(issue_violations=violations, empty_windows=int((count == 0).sum()),
                              partial_windows=int(((count > 0) & (count < h)).sum()))
        if violations:
            raise SnowError("forecast_issue_after_target_cutoff")
    return out, audit


def gate(values):
    return ((values.wa_snow_sum72.fillna(0) > 0) | (values.wb_snow_sum72.fillna(0) > 0)).to_numpy()


def choose(base, snow, on):
    out = np.where(on, snow, base)
    if not np.array_equal(out[~on], base[~on]):
        raise SnowError("gate_off_changed")
    return out


def load_snow(sites):
    tables, audits = {}, {}
    for p in ("wa", "wb"):
        st, field = f"{p}_station", f"{p}_snow"
        train, test = [load_forecast(s, p, [field]) for s in ("train", "test")]
        overlap = pd.MultiIndex.from_frame(train[[st, "round", "ts"]]).intersection(
            pd.MultiIndex.from_frame(test[[st, "round", "ts"]]))
        if len(overlap):
            raise SnowError("train_test_duplicate_forecasts")
        tables[p], audits[p] = rolling_snow(pd.concat([train, test], ignore_index=True), p)
        separate, _ = rolling_snow(test, p)
        aligned = tables[p].reindex(separate.index)
        first = test.ts.min()
        mask = separate.index.get_level_values("ts") < first+pd.Timedelta(hours=72)
        audits[p]["boundary_rows_changed_if_reset"] = int((~np.isclose(
            aligned.loc[mask].to_numpy(), separate.loc[mask].to_numpy(), equal_nan=True)).any(axis=1).sum())
        audits[p]["train_test_overlap"] = len(overlap)
        del train, test, separate
        print(json.dumps(dict(stage="snow_audit", provider=p, audit=audits[p])), flush=True)
    return tables, audits


def add_snow(frame, tables, sites):
    out = pd.DataFrame(index=frame.index)
    for p in ("wa", "wb"):
        keys = pd.MultiIndex.from_arrays([frame.site_id.map(sites[f"{p}_station"]), frame.ts])
        for c in tables[p]:
            out[c] = tables[p][c].reindex(keys).to_numpy()
    return out


def compare(actual, baseline, prediction, smp):
    if not actual.index.equals(baseline.index) or not actual.index.equals(prediction.index):
        raise SnowError("group_keys_mismatch")
    if not np.isfinite(np.r_[baseline.to_numpy(), prediction.to_numpy()]).all():
        raise SnowError("nonfinite_group_prediction")
    weight = smp.w.reindex(actual.index.get_level_values("ts")).to_numpy()
    b = .1*weight*np.abs(baseline.to_numpy()-actual.to_numpy())
    p = .1*weight*np.abs(prediction.to_numpy()-actual.to_numpy())
    day = (actual.index.get_level_values("ts")-pd.Timedelta(hours=1)).normalize()
    daily = pd.DataFrame(dict(base=b, candidate=p), index=day).groupby(level=0).sum().to_numpy()
    rng = np.random.default_rng(2026)
    changes = []
    for _ in range(1000):
        starts = rng.integers(0, max(1, len(daily)-6), size=int(np.ceil(len(daily)/7)))
        idx = np.concatenate([np.arange(s, min(s+7, len(daily))) for s in starts])[:len(daily)]
        sums = daily[idx].sum(axis=0)
        changes.append(100*(sums[1]/sums[0]-1))
    return dict(base_penalty=round(float(b.sum())), candidate_penalty=round(float(p.sum())),
                change_pct=round(float(100*(p.sum()/b.sum()-1)), 4),
                change_ci95_pct=np.quantile(changes, [.025, .975]).round(4).tolist())


def decide(records):
    rows = list(records.values())
    change = 100*(sum(r["gate"]["candidate_penalty"] for r in rows)/sum(r["gate"]["base_penalty"] for r in rows)-1)
    all_seed_improve = all(sum(r["paired_seeds"][str(s)]["candidate_penalty"] for r in rows) <
                           sum(r["paired_seeds"][str(s)]["base_penalty"] for r in rows) for s in SPEC["seeds"])
    sufficient = all(r["gate_days"] >= 10 for r in rows)
    passed = len(rows) == 2 and sufficient and change <= -1 and all_seed_improve and all(r["gate"]["change_pct"] <= 0 for r in rows)
    return dict(passed=bool(passed), pooled_change_pct=round(change, 4),
                all_paired_seeds_improve=bool(all_seed_improve), sufficient_gate_days=bool(sufficient))


def main(audit_only=False):
    PATH.parent.mkdir(parents=True, exist_ok=True)
    result = json.loads(PATH.read_text()) if PATH.exists() else dict(spec=SPEC, results={})
    # JSON changes tuples to lists; windows already use tuple in SPEC.
    if result.get("spec") != json.loads(json.dumps(SPEC)):
        raise SnowError("checkpoint_spec_mismatch")
    if result.get("complete"):
        print(json.dumps(result["verdict"]), flush=True)
        return
    sites = load_sites().set_index("site_id")
    tables, result["audit"] = load_snow(sites)
    result["audit_passed"] = True
    save(PATH, result)
    if audit_only:
        return
    last = pd.Timestamp("2024-04-01", tz=TZ)
    original = T.load_train_features()
    columns = T.select_features(original, CFG["feats"])
    d = original.loc[original.ts <= last, ["site_id", "ts", "energy", *columns]].copy()
    del original
    T._cache.clear()
    if d.duplicated(["site_id", "ts"]).any():
        raise SnowError("duplicate_daylight_keys")
    snow = add_snow(d, tables, sites)
    del tables
    d[COLS] = snow
    del snow
    common = pd.concat([checked_gfs()[0], load_more()[0]], axis=1)
    base_features = columns+list(common)
    if len(base_features) != 144:
        raise SnowError("base_feature_count")
    gen = load_generation()
    gen = gen.loc[gen.ts <= last].copy()
    gen["group"] = gen.site_id.map(sites.current_group)
    smp = load_smp("train").set_index("ts")
    for name, (start, stop) in WINTERS.items():
        if name in result["results"]:
            continue
        start, stop = pd.Timestamp(start, tz=TZ), pd.Timestamp(stop, tz=TZ)
        cutoff = start-pd.Timedelta(hours=7)
        q = sequence_fold_features(d, sites, cutoff)
        tr, val = q.loc[q.ts <= cutoff].copy(), q.loc[q.ts.between(start+pd.Timedelta(hours=1), stop)].copy()
        del q
        tr[list(common)] = common.reindex(tr.ts).to_numpy()
        val[list(common)] = common.reindex(val.ts).to_numpy()
        truth = gen.loc[gen.ts.between(start+pd.Timedelta(hours=1), stop)].copy()
        actual = truth.groupby(["ts", "group"]).energy.sum()
        on = gate(val)
        days = (val.ts-pd.Timedelta(hours=1)).dt.normalize()
        aggregate = lambda p: grouped_prediction(truth, val[["site_id", "ts"]], p).reindex(actual.index)
        y = base_target(tr)
        weight = T.kt_weight(T.clear_sky(tr.cosz), T.sample_weight(tr, "cap", smp.w), CFG)
        scale = T.clear_sky(val.cosz)*val.cap.to_numpy()
        total_base, total_snow = np.zeros(len(val)), np.zeros(len(val))
        paired = {}
        started = time.monotonic()
        print(json.dumps(dict(stage="winter_ready", winter=name, train_rows=len(tr), validation_rows=len(val),
                              gate_days=int(days[on].nunique()), gate_share=float(on.mean()))), flush=True)
        for seed in SPEC["seeds"]:
            predictions = []
            for label, feats in (("base", base_features), ("snow", base_features+COLS)):
                model = T.make_model({**CFG, "seed": seed}).set_params(n_jobs=4)
                model.fit(tr[feats], y, sample_weight=weight, categorical_feature=["site"])
                predictions.append(model.predict(val[feats])*scale)
                del model
                gc.collect()
                print(json.dumps(dict(stage="model_complete", winter=name, seed=seed, model=label)), flush=True)
            b, s = predictions
            total_base += b/3
            total_snow += s/3
            paired[str(seed)] = compare(actual, aggregate(np.maximum(b, 0)),
                                       aggregate(choose(np.maximum(b, 0), np.maximum(s, 0), on)), smp)
        total_base, total_snow = np.maximum(total_base, 0), np.maximum(total_snow, 0)
        b, s = aggregate(total_base), aggregate(total_snow)
        g = aggregate(choose(total_base, total_snow, on))
        result["results"][name] = dict(train_cutoff=cutoff.isoformat(), train_rows=len(tr), validation_rows=len(val),
            gate_days=int(days[on].nunique()), gate_share=float(on.mean()),
            missing_both72_rows=int(val[["wa_snow_sum72", "wb_snow_sum72"]].isna().all(axis=1).sum()),
            snow=compare(actual, b, s, smp), gate=compare(actual, b, g, smp), paired_seeds=paired,
            seconds=round(time.monotonic()-started, 2))
        save(PATH, result)
        print(json.dumps(dict(winter=name, result=result["results"][name])), flush=True)
        del tr, val, truth, total_base, total_snow, predictions
        gc.collect()
    result.update(complete=True, verdict=decide(result["results"]))
    result["selected"] = result["verdict"]["passed"]
    save(PATH, result)
    print(json.dumps(result["verdict"]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    try:
        main(args.audit_only)
    except Exception as exc:
        report = dict(status="stopped", error_type=type(exc).__name__)
        if isinstance(exc, SnowError):
            report["failure_code"] = str(exc)
        else:
            report["source_locations"] = [dict(file=Path(f.filename).name, function=f.name, line=f.lineno)
                                          for f in traceback.extract_tb(exc.__traceback__)[-4:]]
        print(json.dumps(report), flush=True)
        raise SystemExit(1)
