"""과거 정상출력 프로필로 타깃을 정규화한다. 3시드·집계만 저장, 2025-05 제외.

    MPLCONFIGDIR=/tmp/oibc-mpl .venv/bin/python -m src.profile_target
"""
from dataclasses import dataclass
import gc
import json
from pathlib import Path
import time
import traceback

import numpy as np
import pandas as pd

from . import train as T
from .config import CACHE_DIR, EXP_DIR, TZ
from .data import load_generation, load_sites, load_smp
from .direct_groups import metrics
from .gfs_more import load_more
from .gfs_pca import SEASONS, verdict
from .gfs_regions import early_failure
from .gfs_v3 import checked_gfs
from .provider_ensemble import grouped_prediction
from .splits import fold
from .tune_v3 import END, WINDOWS, evaluation_context, evaluation_groups, save
from .v6_error_budget import load_reference
from .weather_teacher import CFG

SPEC = dict(schema="profile_target_v1", quantile=.9, min_days=20, shrink_days=60,
            factor_bounds=[.5, 1.5], azimuth_edges=[150, 210], cosz_edge=.6,
            clear_cosz_min=.3, clear_cloud_max=30, seeds=[2026, 2027, 2028],
            training_profile_asof="calendar_quarter_previous_day_17KST",
            loss="original_kt_weighted_l2_up_to_global_constant")
PATH = EXP_DIR/"p51_profile_target"/"metrics.json"


class ProfileError(ValueError):
    pass


def base_target(frame):
    """v6와 같은 순서로 절단하며, 정규화 후에는 다시 절단하지 않는다."""
    return np.clip(np.clip(frame.energy.to_numpy()/frame.cap.to_numpy(), 0, 1.2)
                   / T.clear_sky(frame.cosz), 0, 2)


def geometry_bin(frame):
    direction = np.select([frame.azim < 150, frame.azim <= 210], [0, 1], default=2)
    cell = direction + 3*(frame.cosz.to_numpy() > .6)
    return np.where(frame.cosz.to_numpy() > .3, cell, -1)


@dataclass
class Profile:
    site_factor: pd.Series
    cell_factor: pd.Series
    cutoff: pd.Timestamp
    clear_rows: int


def empty_profile(cutoff, clear_rows=0):
    index = pd.MultiIndex.from_arrays([[], []], names=["site_id", "cell"])
    return Profile(pd.Series(dtype="float64"), pd.Series(index=index, dtype="float64"), cutoff, clear_rows)


def fit_profile(frame, cutoff):
    mask = ((frame.ts <= cutoff) & (frame.cosz > .3)
            & (frame.wa_cloud <= 30) & (frame.wb_cloud <= 30))
    past = frame.loc[mask, ["site_id", "ts", "energy", "cap", "cosz", "azim"]].copy()
    if past.empty:
        return empty_profile(cutoff)
    if past.ts.max() > cutoff or not (past.cap > 0).all():
        raise ProfileError("profile_past_or_capacity_invalid")
    past["kt"] = base_target(past)
    past["cell"] = geometry_bin(past)
    past["day"] = (past.ts - pd.Timedelta(hours=1)).dt.normalize()
    global_q = float(past.kt.quantile(.9))
    if not np.isfinite(global_q) or global_q <= 1e-6:
        return empty_profile(cutoff, len(past))
    group = past.groupby("site_id", observed=True)
    q, days = group.kt.quantile(.9), group.day.nunique()
    reliable = days >= 20
    weight = days/(days+60)
    site = (weight*(q/global_q) + 1-weight).where(reliable).dropna().clip(.5, 1.5)
    global_cell = past.groupby("cell").kt.quantile(.9)
    cells = past.groupby(["site_id", "cell"], observed=True)
    cell_q, cell_days = cells.kt.quantile(.9), cells.day.nunique()
    denominator = global_cell.reindex(cell_q.index.get_level_values("cell")).to_numpy()
    prior = site.reindex(cell_q.index.get_level_values("site_id")).fillna(1).to_numpy()
    ratio = cell_q.to_numpy()/np.where(denominator > 1e-6, denominator, np.nan)
    weight = cell_days.to_numpy()/(cell_days.to_numpy()+60)
    factor = weight*ratio + (1-weight)*prior
    good = (cell_days.to_numpy() >= 20) & np.isfinite(factor)
    cell = pd.Series(np.clip(factor[good], .5, 1.5), index=cell_q.index[good])
    return Profile(site, cell, cutoff, len(past))


def apply_profile(profile, frame):
    scale = profile.site_factor.reindex(frame.site_id).fillna(1).to_numpy()
    if len(profile.cell_factor):
        keys = pd.MultiIndex.from_arrays([frame.site_id.to_numpy(), geometry_bin(frame)], names=["site_id", "cell"])
        local = profile.cell_factor.reindex(keys).to_numpy()
        scale = np.where(np.isfinite(local), local, scale)
    if not np.isfinite(scale).all() or (scale < .5).any() or (scale > 1.5).any():
        raise ProfileError("profile_scale_invalid")
    return scale.astype("float32")


def scale_summary(values):
    values = np.asarray(values)
    return dict(rows=int(len(values)), unit_factor_share_pct=round(float(100*np.mean(np.abs(values-1) < 1e-6)), 3),
                factor_quantiles={str(q):round(float(np.quantile(values, q)), 5) for q in (0, .1, .5, .9, 1)})


def quarterly_scales(frame):
    values = np.ones(len(frame), dtype="float32")
    diagnostics = {}
    for year in range(frame.ts.min().year, frame.ts.max().year+1):
        for month in (1, 4, 7, 10):
            start = pd.Timestamp(year=year, month=month, day=1, tz=TZ)
            mask = ((frame.ts >= start) & (frame.ts < start + pd.DateOffset(months=3))).to_numpy()
            if not mask.any():
                continue
            cutoff = start - pd.Timedelta(hours=7)
            profile = fit_profile(frame, cutoff)
            values[mask] = apply_profile(profile, frame.loc[mask])
            diagnostics[start.date().isoformat()] = dict(profile_cutoff=cutoff.isoformat(), clear_past_rows=profile.clear_rows,
                sites_with_profile=len(profile.site_factor), geometry_cells=len(profile.cell_factor), **scale_summary(values[mask]))
    return values, diagnostics


def normalized_target(y, factor, weight):
    factor = np.asarray(factor, dtype="float64")
    if not np.isfinite(factor).all() or (factor <= 0).any():
        raise ProfileError("target_scale_invalid")
    return np.asarray(y)/factor, np.asarray(weight)*factor**2


def comparison_records(rows):
    return {n: dict(vs_matched_v3=r["vs_v6_3seeds"]) for n, r in rows.items()}


def main():
    identity = dict(spec=SPEC, baseline_config=CFG, cache_name=CACHE_DIR.name, rule="14.37")
    PATH.parent.mkdir(parents=True, exist_ok=True)
    result = json.loads(PATH.read_text()) if PATH.exists() else {**identity, "results": {}}
    if any(result.get(k) != v for k, v in identity.items()):
        raise ProfileError("checkpoint_spec_mismatch")
    if "verdict" in result:
        print("완료 체크포인트", json.dumps(result["verdict"], ensure_ascii=False), flush=True)
        return
    # 후보 출력 평가 전에 참조 모델 재현을 확인한다.
    diagnostic = json.loads((EXP_DIR/"p50_v6_error_budget"/"metrics.json").read_text())
    if not all(diagnostic["windows"][n]["reproduces_p26"] for n in WINDOWS):
        raise ProfileError("reference_diagnostic_missing")
    baseline = json.loads((EXP_DIR/"p26_gfs_more"/"metrics.json").read_text())["seeds3"]["gfs60"]
    sites = load_sites().set_index("site_id")
    gen = load_generation()
    gen = gen[gen.ts <= END]
    smp = load_smp("train").set_index("ts")
    d = T.load_train_features()
    base_features = T.select_features(d, CFG["feats"])
    d = d.loc[d.ts <= END, ["site_id", "ts", "energy", *base_features]].copy()
    T._cache.clear()
    if d.duplicated(["site_id", "ts"]).any():
        raise ProfileError("duplicate_training_keys")
    print("프로필의 분기별 과거 척도 계산, 2025-05 제외", flush=True)
    d["_profile_oof"], result["quarterly_profile_diagnostics"] = quarterly_scales(d)
    save(PATH, result)
    common = pd.concat([checked_gfs()[0], load_more()[0]], axis=1)
    features = base_features + list(common)
    if len(features) != 144 or "_profile_oof" in features:
        raise ProfileError("input_features_changed")
    for cutoff, names in evaluation_groups().items():
        if all(name in result["results"] for name in names):
            continue
        q = T.fold_features(d, CFG, cutoff)
        tr = q[q.ts <= cutoff].copy()
        start = min(fold(n)["val_start"] for n in names)
        end = max(fold(n)["val_end"] for n in names)
        val = q[q.ts.between(start, end)].copy()
        del q
        if val.ts.max() > END or tr.ts.max() > cutoff:
            raise ProfileError("training_validation_boundary_invalid")
        if any(len(tr) != baseline[n]["train_rows"] for n in names):
            raise ProfileError("baseline_training_row_count_changed")
        tr[list(common)] = common.reindex(tr.ts).to_numpy()
        val[list(common)] = common.reindex(val.ts).to_numpy()
        profile = fit_profile(d, cutoff)
        factor = tr["_profile_oof"].to_numpy()
        val_factor = apply_profile(profile, val)
        cs = T.clear_sky(tr.cosz)
        y = base_target(tr)
        base_weight = T.kt_weight(cs, T.sample_weight(tr, "cap", smp.w), CFG)
        z, weight = normalized_target(y, factor, base_weight)
        weight = weight/weight.mean()
        prediction = np.zeros(len(val))
        train_prediction = np.zeros(len(tr))
        started = time.monotonic()
        for seed in SPEC["seeds"]:
            model = T.fit_model({**CFG, "seed": seed}, tr[features], z, weight)
            prediction += model.predict(val[features])/3
            train_prediction += model.predict(tr[features])/3
            del model
            gc.collect()
            print(f"마감{cutoff:%Y-%m-%d} profile_target seed={seed} 완료", flush=True)
        prediction = np.maximum(prediction, 0)*val_factor*T.clear_sky(val.cosz)*val.cap.to_numpy()
        original_rmse = float(np.sqrt(np.average((train_prediction*factor-y)**2, weights=base_weight)))
        for name in names:
            ctx = evaluation_context(name, gen, sites)
            base_panel = load_reference(name, ctx["actual"], sites)
            b = base_panel.groupby(["ts", "group"]).forecast_kwh.sum().reindex(ctx["a"].index)
            f = fold(name)
            mask = val.ts.between(f["val_start"], f["val_end"]).to_numpy()
            p = grouped_prediction(ctx["actual"], val.loc[mask, ["site_id", "ts"]], prediction[mask]).reindex(ctx["a"].index)
            comparison = metrics(ctx["a"], b, p, smp)
            if comparison["base_penalty"] != baseline[name]["vs_v3_3seeds"]["direct_penalty"]:
                raise ProfileError("baseline_penalty_not_reproduced")
            result["results"][name] = dict(vs_v6_3seeds=comparison, seeds=SPEC["seeds"], features=len(features),
                train_rows=len(tr), train_cutoff=cutoff.isoformat(), weighted_train_rmse_original_kt=round(original_rmse, 6),
                training_predict_seconds=round(time.monotonic()-started, 2), training_profile=scale_summary(factor),
                validation_profile=scale_summary(val_factor[mask]), fitted_profile_sites=len(profile.site_factor),
                fitted_geometry_cells=len(profile.cell_factor), baseline_reproduces_p26=True)
            save(PATH, result)
            print(name, json.dumps(comparison, ensure_ascii=False), flush=True)
        reason = early_failure(comparison_records(result["results"]))
        if reason:
            result["verdict"] = dict(passed=False, early_rejected=True, reason=reason, seeds=3,
                uncompleted_windows=[n for n in WINDOWS if n not in result["results"]])
            save(PATH, result)
            print("3시드 판정", json.dumps(result["verdict"], ensure_ascii=False), flush=True)
            return
        del tr, val, prediction, train_prediction
        gc.collect()
    result["verdict"] = {**verdict(comparison_records(result["results"])), "seeds": 3}
    save(PATH, result)
    print("3시드 판정", json.dumps(result["verdict"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        report = dict(status="stopped", error_type=type(exc).__name__)
        if isinstance(exc, ProfileError):
            report["failure_code"] = str(exc)
        else:
            report["source_locations"] = [dict(file=Path(f.filename).name, function=f.name, line=f.lineno)
                                          for f in traceback.extract_tb(exc.__traceback__)[-4:]]
        print(json.dumps(report, ensure_ascii=False), flush=True)
        raise SystemExit(1)
