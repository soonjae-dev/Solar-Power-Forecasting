"""p59: 과거 예보-발전량 관계의 연간 동결 특징. 집계만 출력/저장."""
from dataclasses import dataclass
import gc
import json
from pathlib import Path
import time
import traceback

import numpy as np
import pandas as pd

from .. import train as T
from ..config import CACHE_DIR, EXP_DIR, TZ
from ..data import load_generation, load_sites, load_smp
from ..direct_groups import metrics
from ..gfs_more import load_more
from ..gfs_pca import verdict
from ..gfs_regions import early_failure
from ..gfs_v3 import checked_gfs
from ..profile_target import base_target, comparison_records
from ..provider_ensemble import grouped_prediction
from .sequence_residual import sequence_fold_features
from ..splits import fold
from ..tune_v3 import END, WINDOWS, evaluation_context, save
from ..v6_error_budget import load_reference
from ..weather_teacher import CFG

SPEC = dict(rule="14.53", basis=8, global_ridge=.001, site_ridge=60.,
            min_cosz=.1, seeds=[2026, 2027, 2028],
            context="evaluation_month_day_annual_blocks_previous_day_17KST")
COLS = [f"response_delta_{i}" for i in range(8)] + [
    "response_query", "response_reliability", "response_age", "response_query_valid"]
PATH = EXP_DIR / "p59_response_context" / "metrics.json"


class ContextError(ValueError):
    pass


def basis(frame):
    a = np.clip(frame.wa_cloud.to_numpy(dtype=float), 0, 100)/100
    b = np.clip(frame.wb_cloud.to_numpy(dtype=float), 0, 100)/100
    temp = np.clip((frame.wa_temp.to_numpy(dtype=float)+frame.wb_temp.to_numpy(dtype=float))/2, -40, 50)/50
    angle = np.deg2rad(frame.azim.to_numpy(dtype=float))
    x = np.column_stack([np.ones(len(frame)), a, b, a*b,
                         frame.cosz.to_numpy(dtype=float), np.sin(angle), np.cos(angle), temp])
    valid = np.isfinite(x).all(axis=1)
    return np.where(np.isfinite(x), x, 0), valid


def solve_ridge(x, y, weight, ridge):
    gram = np.einsum("ni,n,nj->ij", x, weight, x, optimize=False)
    rhs = np.einsum("ni,n,n->i", x, weight, y, optimize=False)
    answer = np.linalg.solve(gram + ridge*np.eye(8), rhs)
    if not np.isfinite(answer).all():
        raise ContextError("nonfinite_ridge")
    return answer


@dataclass
class Context:
    coefficients: pd.DataFrame
    days: pd.Series
    cutoff: pd.Timestamp
    rows: int


def fit_context(frame, cutoff):
    # Only the past is selected before any target conversion or aggregation.
    past = frame.loc[(frame.ts <= cutoff) & (frame.cosz > SPEC["min_cosz"]) & (frame.cap > 0)].copy()
    x, valid = basis(past)
    y = base_target(past)
    valid &= np.isfinite(y)
    past, x, y = past.loc[valid].copy(), x[valid], y[valid]
    if not len(past):
        return Context(pd.DataFrame(columns=COLS[:8], dtype=float), pd.Series(dtype=float), cutoff, 0)
    day = (past.ts-pd.Timedelta(hours=1)).dt.normalize()
    count = past.groupby([past.site_id, day], observed=True).ts.transform("size").to_numpy()
    weight = 1/count
    beta = solve_ridge(x, y, weight/weight.sum(), SPEC["global_ridge"])
    residual = y-np.einsum("ni,i->n", x, beta, optimize=False)
    codes, ids = pd.factorize(past.site_id, sort=False)
    coefficients, days = [], []
    for i in range(len(ids)):
        mask = codes == i
        coefficients.append(solve_ridge(x[mask], residual[mask], weight[mask], SPEC["site_ridge"]))
        days.append(int(day.iloc[np.flatnonzero(mask)].nunique()))
    return Context(pd.DataFrame(coefficients, index=ids, columns=COLS[:8]),
                   pd.Series(days, index=ids, dtype=float), cutoff, len(past))


def apply_context(context, frame, origin):
    if context.cutoff != origin-pd.Timedelta(hours=7):
        raise ContextError("context_origin_mismatch")
    day = (frame.ts-pd.Timedelta(hours=1)).dt.normalize()
    age = (day-origin).dt.total_seconds().to_numpy()/86400/365.25
    if (age < 0).any() or (age > 1.01).any():
        raise ContextError("query_outside_annual_block")
    delta = context.coefficients.reindex(frame.site_id).fillna(0).to_numpy(dtype=float)
    n = context.days.reindex(frame.site_id).fillna(0).to_numpy()
    x, valid = basis(frame)
    query = np.einsum("ni,ni->n", delta, x, optimize=False)
    values = np.column_stack([delta, np.where(valid, query, 0), n/(n+60), age, valid.astype(float)])
    if not np.isfinite(values).all():
        raise ContextError("nonfinite_context_features")
    return pd.DataFrame(values.astype("float32"), index=frame.index, columns=COLS)


def annual_training_context(frame, origin):
    """Each target day belongs to one non-overlapping frozen annual context."""
    day = (frame.ts-pd.Timedelta(hours=1)).dt.normalize()
    values = pd.DataFrame(np.nan, index=frame.index, columns=COLS)
    diagnostics = []
    for year in range(int(day.dt.year.min())-1, int(day.dt.year.max())+1):
        start = pd.Timestamp(year=year, month=origin.month, day=origin.day, tz=TZ)
        stop = start+pd.DateOffset(years=1)
        mask = (day >= start) & (day < stop)
        if not mask.any():
            continue
        fitted = fit_context(frame, start-pd.Timedelta(hours=7))
        values.loc[mask] = apply_context(fitted, frame.loc[mask], start).to_numpy()
        diagnostics.append(dict(origin=start.isoformat(), cutoff=fitted.cutoff.isoformat(),
                                history_rows=fitted.rows, sites_with_history=len(fitted.days), query_rows=int(mask.sum())))
    if not np.isfinite(values.to_numpy()).all():
        raise ContextError("annual_context_coverage")
    return values.astype("float32"), diagnostics


def context_summary(values):
    return dict(rows=len(values), history_share=float((values.response_reliability > 0).mean()),
                valid_query_share=float(values.response_query_valid.mean()),
                query_quantiles={str(q): float(values.response_query.quantile(q)) for q in (.1, .5, .9)},
                max_age_years=float(values.response_age.max()))


def main():
    PATH.parent.mkdir(parents=True, exist_ok=True)
    identity = dict(spec=SPEC, baseline_config=CFG, cache_name=CACHE_DIR.name)
    result = json.loads(PATH.read_text()) if PATH.exists() else {**identity, "results": {}}
    if any(result.get(k) != v for k, v in identity.items()):
        raise ContextError("checkpoint_signature")
    if result.get("complete"):
        print(json.dumps(result["verdict"], ensure_ascii=False), flush=True)
        return
    baseline = json.loads((EXP_DIR/"p26_gfs_more"/"metrics.json").read_text())["seeds3"]["gfs60"]
    sites = load_sites().set_index("site_id")
    gen = load_generation()
    gen = gen.loc[gen.ts <= END]
    smp = load_smp("train").set_index("ts")
    original = T.load_train_features()
    base_cols = T.select_features(original, CFG["feats"])
    d = original.loc[original.ts <= END, ["site_id", "ts", "energy", *base_cols]].copy()
    del original
    T._cache.clear()
    if d.duplicated(["site_id", "ts"]).any():
        raise ContextError("duplicate_input_keys")
    common = pd.concat([checked_gfs()[0], load_more()[0]], axis=1)
    features = base_cols+list(common)+COLS
    if len(base_cols)+len(common.columns) != 144 or len(features) != 156:
        raise ContextError("feature_count")
    for cutoff, names in evaluation_groups_local().items():
        if all(n in result["results"] for n in names):
            continue
        start = min(fold(n)["val_start"] for n in names)
        origin = start.normalize()
        if cutoff != origin-pd.Timedelta(hours=7):
            raise ContextError("fold_origin")
        q = sequence_fold_features(d, sites, cutoff)
        tr = q.loc[q.ts <= cutoff].copy()
        val = q.loc[q.ts.between(start, max(fold(n)["val_end"] for n in names))].copy()
        del q
        if any(len(tr) != baseline[n]["train_rows"] for n in names):
            raise ContextError("baseline_train_rows")
        history, diag = annual_training_context(tr, origin)
        fitted = fit_context(tr, cutoff)
        val_history = apply_context(fitted, val, origin)
        tr[COLS], val[COLS] = history, val_history
        tr[list(common)], val[list(common)] = common.reindex(tr.ts).to_numpy(), common.reindex(val.ts).to_numpy()
        weight = T.kt_weight(T.clear_sky(tr.cosz), T.sample_weight(tr, "cap", smp.w), CFG)
        y = base_target(tr)
        prediction = np.zeros(len(val))
        started = time.monotonic()
        print(json.dumps(dict(stage="context_ready", train_rows=len(tr), features=len(features),
                              annual_blocks=len(diag), validation=context_summary(val_history)), ensure_ascii=False), flush=True)
        for seed in SPEC["seeds"]:
            model = T.make_model({**CFG, "seed": seed}).set_params(n_jobs=1)
            model.fit(tr[features], y, sample_weight=weight, categorical_feature=["site"])
            prediction += model.predict(val[features])/len(SPEC["seeds"])
            del model
            gc.collect()
            print(json.dumps(dict(stage="seed_complete", seed=seed, cutoff=cutoff.isoformat())), flush=True)
        prediction = np.maximum(prediction, 0)*T.clear_sky(val.cosz)*val.cap.to_numpy()
        for name in names:
            ctx = evaluation_context(name, gen, sites)
            base = load_reference(name, ctx["actual"], sites)
            b = base.groupby(["ts", "group"]).forecast_kwh.sum().reindex(ctx["a"].index)
            f = fold(name)
            mask = val.ts.between(f["val_start"], f["val_end"])
            p = grouped_prediction(ctx["actual"], val.loc[mask, ["site_id", "ts"]], prediction[mask]).reindex(ctx["a"].index)
            comparison = metrics(ctx["a"], b, p, smp)
            if comparison["base_penalty"] != baseline[name]["vs_v3_3seeds"]["direct_penalty"]:
                raise ContextError("baseline_penalty")
            result["results"][name] = dict(vs_v6_3seeds=comparison, features=len(features), train_rows=len(tr),
                train_cutoff=cutoff.isoformat(), annual_contexts=diag, training_context=context_summary(history),
                validation_context=context_summary(val_history.loc[mask]), seconds=round(time.monotonic()-started, 2))
            save(PATH, result)
            print(name, json.dumps(comparison, ensure_ascii=False), flush=True)
        reason = early_failure(comparison_records(result["results"]))
        if reason:
            result.update(complete=True, selected=False, verdict=dict(passed=False, early_rejected=True, reason=reason,
                          uncompleted_windows=[n for n in WINDOWS if n not in result["results"]]))
            save(PATH, result)
            return
        del tr, val, prediction, history, val_history
        gc.collect()
    decision = verdict(comparison_records(result["results"]))
    result.update(complete=True, selected=decision["passed"], verdict=decision)
    save(PATH, result)
    print(json.dumps(decision, ensure_ascii=False), flush=True)


def evaluation_groups_local():
    from ..tune_v3 import evaluation_groups
    return evaluation_groups()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        report = dict(status="stopped", error_type=type(exc).__name__)
        if isinstance(exc, ContextError):
            report["failure_code"] = str(exc)
        else:
            report["source_locations"] = [dict(file=Path(f.filename).name, function=f.name, line=f.lineno)
                                          for f in traceback.extract_tb(exc.__traceback__)[-4:]]
        print(json.dumps(report, ensure_ascii=False), flush=True)
        raise SystemExit(1)
