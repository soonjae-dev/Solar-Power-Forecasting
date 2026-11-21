"""p58: 독립3h/24h 예측과 v6의 비음수 WLS 조정. 집계만 저장한다."""
import argparse
import gc
import json
import os
from pathlib import Path
import time
import traceback
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from threadpoolctl import threadpool_limits

from .. import train as T
from ..config import EXP_DIR, PRED_DIR, TZ
from ..data import load_smp
from ..diagnose_season import run_id
from ..direct_groups import metrics
from .sequence_audit import day_slots
from .sequence_residual import prepare, sequence_fold_features
from ..solar import solar
from ..splits import fold
from ..tune_v3 import END, WINDOWS, evaluation_context, save
from ..v6_error_budget import load_reference
from ..weather_teacher import CFG

PATH = EXP_DIR/"p58_temporal_reconcile"/"metrics.json"
SPEC = dict(rule="14.48", resolutions=[3, 24], seeds=[2026, 2027, 2028],
            n_estimators=600, learning_rate=.06, num_leaves=127, min_child_samples=200,
            variance_floor=1e-6, variance_ddof=1, oof5_gate_pct=-1.)


def summing_matrix():
    h = np.zeros((8, 24))
    for i in range(8):
        h[i, 3*i:3*i+3] = 1
    return np.vstack([np.eye(24), h, np.ones((1, 24))])


S = summing_matrix()


def reconcile(nodes, daylight, variance):
    """입력은 용량 정규화. 정답/관측mask를 받지 않는다."""
    nodes, daylight, variance = np.asarray(nodes), np.asarray(daylight, bool), np.asarray(variance)
    if nodes.shape != (len(daylight), 33) or daylight.shape[1] != 24 or variance.shape != (33,):
        raise ValueError("조정 행렬 크기 위반")
    if not np.isfinite(nodes).all() or not np.isfinite(variance).all() or (variance <= 0).any():
        raise ValueError("조정 입력 유한성/분산 위반")
    out = np.zeros((len(nodes), 24))
    patterns, codes = np.unique(daylight, axis=0, return_inverse=True)
    weight = 1/np.sqrt(variance)
    constrained = 0
    for k, mask in enumerate(patterns):
        ids = np.flatnonzero(codes == k)
        if not mask.any():
            continue
        a = S[:, mask]*weight[:, None]
        gram = np.einsum("ji,jk->ik", a, a, optimize=False)
        projection = np.linalg.solve(gram, a.T)
        rhs = nodes[ids]*weight
        values = np.einsum("ij,nj->ni", projection, rhs, optimize=False)
        bad = (values < -1e-10).any(axis=1)
        for j in np.flatnonzero(bad):
            values[j] = nnls(a, rhs[j], maxiter=10*a.shape[1])[0]
        constrained += int(bad.sum())
        out[np.ix_(ids, np.flatnonzero(mask))] = np.maximum(values, 0)
    return out, dict(site_days=len(out), constrained_site_days=constrained)


def labels(gen, resolution, cutoff):
    p = gen.loc[(gen.ts <= cutoff) & (gen.ts >= pd.Timestamp("2021-01-01 01:00", tz=TZ))].copy()
    day, slot = day_slots(p.ts)
    p["day"], p["bin"] = day, slot//resolution
    if p.duplicated(["site_id", "ts"]).any():
        raise ValueError("원본 관측 키 중복")
    p["target"] = p.energy.clip(lower=0)
    grouped = p.groupby(["site_id", "day", "bin"], observed=True)
    y = grouped.agg(target=("target", "sum"), observed=("ts", "size"), last_ts=("ts", "max"))
    return y.loc[(y.observed == resolution) & (y.last_ts <= cutoff), ["target"]]


def common_features(gfs, resolution):
    """공통기상은 발전소/태양위치/관측지원과 독립적으로 먼저 시간집계."""
    g = gfs.copy()
    ts = pd.Series(g.index)
    day, slot = day_slots(ts)
    g["day"], g["bin"] = day.to_numpy(), slot//resolution
    return g.groupby(["day", "bin"], observed=True).mean()


def aggregate_features(weather, columns, gfs, resolution):
    p = weather[["site_id", "ts", *columns]].copy()
    day, slot = day_slots(p.ts)
    p["day"], p["bin"] = day, slot//resolution
    p["cs_sum"] = T.clear_sky(p.cosz)
    grouped = p.groupby(["site_id", "day", "bin"], observed=True)
    x = grouped[columns].mean()
    x["cs_sum"] = grouped.cs_sum.sum()
    x["daylight_hours"] = grouped.size()
    x["bin_number"] = x.index.get_level_values("bin")
    common = common_features(gfs, resolution)
    keys = pd.MultiIndex.from_arrays([x.index.get_level_values("day"), x.index.get_level_values("bin")])
    x[list(gfs)] = common.reindex(keys).to_numpy()
    if len(x.columns) != 146 or np.isinf(x.to_numpy()).any():
        raise ValueError("집계 특징 수/무한값 위반")
    return x


def daily_panel(panel, sites):
    p = panel[["site_id", "ts", "energy", "forecast_kwh"]].copy()
    if p.duplicated(["site_id", "ts"]).any():
        raise ValueError("비교 예측 키 중복")
    day, slot = day_slots(p.ts)
    p["day"], p["slot"] = day, slot
    complete = p.groupby(["site_id", "day"]).ts.transform("size") == 24
    omitted = p.loc[~complete, ["site_id", "day"]].drop_duplicates().shape[0]
    p = p.loc[complete].sort_values(["site_id", "ts"]).reset_index(drop=True)
    if p.empty or not np.array_equal(p.slot.to_numpy(), np.tile(np.arange(24), len(p)//24)):
        raise ValueError("완전24시간 비교지원 불일치")
    keys = p.iloc[::24][["site_id", "day"]].reset_index(drop=True)
    cap = keys.site_id.map(sites.capacity_kw).to_numpy()
    elev, _, _ = solar(pd.DatetimeIndex(p.ts), p.site_id.map(sites.lat).to_numpy(), p.site_id.map(sites.lon).to_numpy())
    base = p.forecast_kwh.to_numpy().reshape(-1, 24)
    actual = p.energy.to_numpy().reshape(-1, 24)
    if not np.isfinite(base).all() or (base < 0).any() or not np.isfinite(actual).all():
        raise ValueError("일비교 값 유한성 위반")
    return dict(panel=p, keys=keys, cap=cap, base=base, actual=actual,
                daylight=(elev > 0).reshape(-1, 24), omitted_site_days=omitted)


def coarse_predict(models, table, days, resolution):
    keys = days["keys"].merge(pd.DataFrame(dict(bin=np.arange(24//resolution))), how="cross")
    index = pd.MultiIndex.from_frame(keys[["site_id", "day", "bin"]])
    x = table.reindex(index)
    scale = (x.cap*x.cs_sum).to_numpy()
    valid = np.isfinite(scale) & (scale > 0)
    pred = np.zeros(len(x))
    if valid.any():
        kt = sum(m.predict(x.loc[valid])/len(models) for m in models)
        pred[valid] = np.maximum(kt, 0)*scale[valid]
    # 누락이 실제야간인지 별도확인: 주간노드가없으면0처리금지.
    expected = days["daylight"].reshape(len(days["cap"]), 24//resolution, resolution).any(axis=2).ravel()
    if not np.array_equal(valid, expected):
        raise ValueError("독립모델 주간 특징 누락")
    return pred.reshape(-1, 24//resolution)


def make_nodes(days, predictions):
    return np.column_stack([days["base"], predictions[3], predictions[24]])/days["cap"][:, None]


def fit_coarse(weather, columns, gfs, gen, cutoff):
    tables, models, report = {}, {}, {}
    for resolution in SPEC["resolutions"]:
        table = aggregate_features(weather, columns, gfs, resolution)
        y = labels(gen, resolution, cutoff)
        tr = table.join(y, how="inner")
        scale = (tr.cap*tr.cs_sum).to_numpy()
        valid = np.isfinite(scale) & (scale > 0)
        tr, scale = tr.loc[valid], scale[valid]
        if len(tr) == 0:
            raise ValueError("집계 학습 표본 없음")
        features = list(table.columns)
        target = (tr.target.to_numpy()/scale).clip(0, 2)
        if tr.index.get_level_values("day").max() > cutoff or not np.isfinite(target).all():
            raise ValueError("집계 학습 마감/정답 위반")
        fitted = []
        started = time.perf_counter()
        for seed in SPEC["seeds"]:
            cfg = {**CFG, **{k:SPEC[k] for k in ("n_estimators", "learning_rate", "num_leaves", "min_child_samples")}, "seed":seed}
            model = T.make_model(cfg).set_params(n_jobs=1)
            model.fit(tr[features], target, sample_weight=scale/scale.mean())
            fitted.append(model)
            print(json.dumps(dict(cutoff=cutoff.isoformat(), resolution=resolution, seed=seed,
                                  train_rows=len(tr), status="fitted")), flush=True)
        models[resolution], tables[resolution] = fitted, table
        report[str(resolution)] = dict(train_rows=len(tr), zero_target_blocks=int((tr.target == 0).sum()),
            features=len(features), seconds=round(time.perf_counter()-started, 2))
        del tr
        gc.collect()
    return tables, models, report


def forecast_coarse(tables, models, days):
    return {r:coarse_predict(models[r], tables[r], days, r) for r in SPEC["resolutions"]}


def diagnostics(days, predicted):
    out = {}
    for r in SPEC["resolutions"]:
        n = len(days["cap"])
        actual = days["actual"].reshape(n, 24//r, r).sum(axis=2)/days["cap"][:, None]
        baseline = days["base"].reshape(n, 24//r, r).sum(axis=2)/days["cap"][:, None]
        candidate = predicted[r]/days["cap"][:, None]
        e, b = (candidate-actual).ravel(), (baseline-actual).ravel()
        denom = np.sqrt(((e-e.mean())**2).mean()*((b-b.mean())**2).mean())
        out[str(r)] = dict(base_rmse_cf=float(np.sqrt((b*b).mean())), coarse_rmse_cf=float(np.sqrt((e*e).mean())),
            base_mae_cf=float(np.abs(b).mean()), coarse_mae_cf=float(np.abs(e).mean()),
            residual_correlation=float(((e-e.mean())*(b-b.mean())).mean()/denom) if denom else None)
    return out


def compare(days, values, sites, smp):
    p = days["panel"].copy()
    p["candidate"] = values.ravel()
    p["group"] = p.site_id.map(sites.current_group)
    grouped = p.groupby(["ts", "group"])[["energy", "forecast_kwh", "candidate"]].sum()
    return metrics(grouped.energy, grouped.forecast_kwh, grouped.candidate, smp)


def reference_days(name, gen, sites):
    path = PRED_DIR/run_id("gfs60seed", name)/"predictions.parquet"
    p = pd.read_parquet(path, columns=["site_id", "ts", "forecast_kwh"])
    f = fold(name)
    p = p.merge(gen, on=["site_id", "ts"], how="left", validate="one_to_one")
    if p.energy.isna().any() or not p.ts.between(f["val_start"], f["val_end"]).all():
        raise ValueError("OOF 비교지원/구간 위반")
    return daily_panel(p, sites)


def main(stage):
    threadpool_limits(limits=1)
    warnings.filterwarnings("error", category=RuntimeWarning)
    Path("/tmp/oibc_temporal_reconcile.pid").write_text(str(os.getpid()))
    PATH.parent.mkdir(parents=True, exist_ok=True)
    result = json.loads(PATH.read_text()) if PATH.exists() else dict(spec=SPEC)
    if result["spec"] != SPEC:
        raise ValueError("설정 체크포인트 불일치")
    if result.get("complete"):
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return
    _, data, columns, gfs, sites, gen, restored = prepare(build_blocks=False)
    calibration = reference_days("oof6m_4", gen, sites)
    selection = reference_days("oof6m_5", gen, sites)
    result["prepare"] = dict(restored_daylight_rows=restored,
        calibration_site_days=len(calibration["cap"]), selection_site_days=len(selection["cap"]),
        calibration_omitted_partial_days=calibration["omitted_site_days"],
        selection_omitted_partial_days=selection["omitted_site_days"],
        calibration_zero_days=int((calibration["actual"] == 0).all(axis=1).sum()),
        selection_zero_days=int((selection["actual"] == 0).all(axis=1).sum()),
        scored=False)
    save(PATH, result)
    print(json.dumps(dict(prepare=result["prepare"])), flush=True)
    if stage == "prepare":
        return
    smp = load_smp("train").set_index("ts")
    if "calibration" not in result:
        cutoff = fold("oof6m_4")["cutoff"]
        weather = sequence_fold_features(data.loc[data.ts <= fold("oof6m_4")["val_end"]], sites, cutoff)
        tables, models, report = fit_coarse(weather, columns, gfs, gen, cutoff)
        forecast = forecast_coarse(tables, models, calibration)
        errors = make_nodes(calibration, forecast)-np.einsum("ij,nj->ni", S,
            calibration["actual"]/calibration["cap"][:, None], optimize=False)
        variance = np.maximum(errors.var(axis=0, ddof=1), SPEC["variance_floor"])
        result["calibration"] = dict(training=report, variance=variance.tolist(),
            diagnostics=diagnostics(calibration, forecast), used_for_parameter_selection=False)
        save(PATH, result)
        del models, tables, weather
        gc.collect()
    variance = np.array(result["calibration"]["variance"])
    cutoff = fold("oof6m_5")["cutoff"]
    weather = sequence_fold_features(data.loc[data.ts <= fold("oof6m_5")["val_end"]], sites, cutoff)
    tables, models, report = fit_coarse(weather, columns, gfs, gen, cutoff)
    forecast = forecast_coarse(tables, models, selection)
    values, solver = reconcile(make_nodes(selection, forecast), selection["daylight"], variance)
    evaluation = compare(selection, values*selection["cap"][:, None], sites, smp)
    result["oof5"] = dict(metrics=evaluation, training=report, solver=solver,
        diagnostics=diagnostics(selection, forecast), passed=evaluation["change_pct"] <= SPEC["oof5_gate_pct"])
    save(PATH, result)
    print(json.dumps(dict(oof5=result["oof5"]), ensure_ascii=False), flush=True)
    del models, tables, weather
    gc.collect()
    if not result["oof5"]["passed"]:
        result.update(complete=True, selected=False, reason="OOF5 시간별 집합 페널티 1% 개선 미달")
        save(PATH, result)
        return
    result.setdefault("external", {})
    grouped = {}
    for name in WINDOWS:
        grouped.setdefault(fold(name)["cutoff"], []).append(name)
    prior = json.loads((EXP_DIR/"p26_gfs_more"/"metrics.json").read_text())["seeds3"]["gfs60"]
    for cutoff, names in grouped.items():
        if all(name in result["external"] for name in names):
            continue
        weather = sequence_fold_features(data, sites, cutoff)
        tables, models, report = fit_coarse(weather, columns, gfs, gen, cutoff)
        for name in names:
            ctx = evaluation_context(name, gen, sites)
            days = daily_panel(load_reference(name, ctx["actual"], sites), sites)
            if days["omitted_site_days"]:
                raise ValueError("외부 평가 완전지원 불일치")
            forecast = forecast_coarse(tables, models, days)
            values, solver = reconcile(make_nodes(days, forecast), days["daylight"], variance)
            r = compare(days, values*days["cap"][:, None], sites, smp)
            if r["base_penalty"] != prior[name]["vs_v3_3seeds"]["direct_penalty"]:
                raise ValueError("v6 기준 페널티 재현 실패")
            result["external"][name] = dict(metrics=r, training=report, solver=solver)
            save(PATH, result)
            print(json.dumps({name:r}), flush=True)
        del models, tables, weather
        gc.collect()
        rows = result["external"]
        if any(v["metrics"]["change_pct"] > (0 if k == "fold1_static" else 3) for k,v in rows.items()):
            result.update(complete=True, selected=False, reason="외부 최악계절/static 비악화 기준 미달")
            save(PATH, result)
            return
    seasons = [result["external"][n]["metrics"] for n in WINDOWS if n != "fold1_static"]
    change = 100*(sum(v["direct_penalty"] for v in seasons)/sum(v["base_penalty"] for v in seasons)-1)
    result.update(complete=True, seasonal_change_pct=change, selected=change <= -1 and
        sum(v["change_pct"] < 0 for v in seasons) >= 3 and max(v["change_pct"] for v in seasons) <= 3 and
        result["external"]["fold1_static"]["metrics"]["change_pct"] <= 0)
    save(PATH, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["prepare", "run"], default="prepare")
    args = parser.parse_args()
    try:
        main(args.stage)
    except Exception as exc:
        frames = [dict(function=f.name, line=f.lineno) for f in traceback.extract_tb(exc.__traceback__)
                  if "/src/" in f.filename]
        print(json.dumps(dict(status="failed", error_type=type(exc).__name__, code_frames=frames)), flush=True)
        raise SystemExit(1) from None
