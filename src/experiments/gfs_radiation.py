"""GFS sflux 복사 성분: 공개시각 감사→지역 요약 수집→검증. 집계만 출력.

    .venv/bin/python -m src.experiments.gfs_radiation pipeline
    .venv/bin/python -m src.experiments.gfs_radiation status

데이터는 외부 DATA_ROOT의 전용 디렉터리에만 저장한다. 제출은 하지 않는다.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
import json
import sys
import time

import numpy as np

from ..config import DATA_ROOT, EXP_DIR, ROOT
from .external_weather_audit import parse_listing, prediction_cutoff
from .gefs_mean import (DEVELOPMENT_LAST, EVALUATION_FIRST, FIRST, LAST, STEPS, PipelineError,
                       block_means, error_summary, expand_blocks, published_before_cutoff, request, save)
from ..gfs_fetch import parse_idx
from .gfs_radiation_probe import FIELDS, TARGETS

BUCKET = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
REGIONS = ("all",) + tuple(f"r{i}{j}" for i in range(3) for j in range(3))
COLUMNS = tuple(f"sflux_{key}_{region}" for key in ("diffuse_fraction", "transmittance") for region in REGIONS)
SCHEMA = "gfs_sflux_optics20_3hour_v1"
DIRECTORY = DATA_ROOT / "external" / "gfs_sflux_optics_00utc"
AUDIT_PATH = EXP_DIR / "p46_gfs_radiation_coverage" / "metrics.json"
DOWNLOAD_PATH = EXP_DIR / "p47_gfs_radiation_download" / "metrics.json"
MODEL_PATH = EXP_DIR / "p48_gfs_radiation_model" / "metrics.json"
ENV_PATH = EXP_DIR / "p49_gfs_radiation_environment" / "metrics.json"


def run_prefix(target):
    run = target - timedelta(days=1)
    folder = "atmos/" if run >= date(2021, 3, 23) else ""
    return f"gfs.{run:%Y%m%d}/00/{folder}gfs.t00z.sfluxgrbf"


def url_for(target, step):
    return f"{BUCKET}/{run_prefix(target)}{step:03d}.grib2"


def required_keys():
    bodies = [f"gfs.t00z.sfluxgrbf{s:03d}.grib2" for s in STEPS]
    return bodies + [n + ".idx" for n in bodies]


@lru_cache(maxsize=4)
def gaussian_regions(ni, nj, order):
    """사이트와 무관한 고정 영역의 전지구 평탄 인덱스. 전지구 좌표 배열 생성 불필요."""
    import eccodes
    if nj != 2 * order or ni != 4 * order:
        raise PipelineError("unexpected_gaussian_grid_shape")
    lat = np.array(list(eccodes.codes_get_gaussian_latitudes(order)), dtype="float64")
    lon = np.arange(ni, dtype="float64") * (360. / ni)
    boxes = [(33, 39, 124, 130)] + [(33 + 2*i, 35 + 2*i, 124 + 2*j, 126 + 2*j)
                                           for i in range(3) for j in range(3)]
    indices = []
    for south, north, west, east in boxes:
        rows = np.flatnonzero((lat >= south) & (lat < north))
        cols = np.flatnonzero((lon >= west) & (lon < east))
        if not len(rows) or not len(cols):
            raise PipelineError("empty_fixed_region")
        indices.append((rows[:, None] * ni + cols[None, :]).ravel())
    return lat, indices


def regional_means(message, start, end):
    import eccodes
    gid = eccodes.codes_new_from_message(message)
    try:
        get = lambda key: eccodes.codes_get(gid, key)
        if get("gridType") != "regular_gg" or get("stepType") != "avg":
            raise PipelineError("unexpected_grid_or_step_type")
        if get("startStep") != start or get("endStep") != end:
            raise PipelineError("grib_idx_interval_mismatch")
        for flag in ("iScansNegatively", "jScansPositively", "jPointsAreConsecutive", "alternativeRowScanning"):
            if get(flag) != 0:
                raise PipelineError("unsupported_grid_scanning_order")
        ni, nj, order = int(get("Ni")), int(get("Nj")), int(get("N"))
        lat, indices = gaussian_regions(ni, nj, order)
        expected = (lat[0], lat[-1], 0., 360. - 360. / ni)
        actual = tuple(get(k) for k in ("latitudeOfFirstGridPointInDegrees", "latitudeOfLastGridPointInDegrees",
                                       "longitudeOfFirstGridPointInDegrees", "longitudeOfLastGridPointInDegrees"))
        if not np.allclose(actual, expected, atol=1e-5, rtol=0):
            raise PipelineError("gaussian_coordinates_mismatch")
        values = eccodes.codes_get_values(gid)
        if len(values) != ni * nj:
            raise PipelineError("grib_value_count_mismatch")
        missing = get("missingValue") if get("bitmapPresent") else None
        result = []
        for index in indices:
            chosen = values[index]
            if missing is not None:
                chosen = np.where(chosen == missing, np.nan, chosen)
            if not np.isfinite(chosen).all():
                raise PipelineError("fixed_region_values_nonfinite")
            result.append(float(chosen.mean()))
        return np.array(result)
    finally:
        eccodes.codes_release(gid)


def make_features(means, starts):
    if np.shape(means) != (5, 6, 10) or np.shape(starts) != (5, 6):
        raise PipelineError("regional_mean_shape_mismatch")
    fields = {}
    for i, key in enumerate(FIELDS):
        averages = {step: (int(starts[i, j]), means[i, j]) for j, step in enumerate(STEPS)}
        fields[key] = np.maximum(expand_blocks(block_means(averages)), 0)
    beam = fields["VBDSF"] + fields["NBDSF"]
    diffuse = fields["VDDSF"] + fields["NDDSF"]
    total = beam + diffuse
    fraction = np.full(total.shape, np.nan)
    transmittance = np.full(total.shape, np.nan)
    np.divide(diffuse, total, out=fraction, where=total >= 20)
    np.divide(total, fields["CSDSF"], out=transmittance, where=fields["CSDSF"] >= 20)
    return np.concatenate([np.clip(fraction, 0, 1), np.clip(transmittance, 0, 2)], axis=1).astype("float32")


def read_records(target, step, fields=FIELDS):
    url = url_for(target, step)
    payload, headers = request(url + ".idx")
    idx_stamp = published_before_cutoff(target, headers)
    text = payload.decode("ascii")
    means, starts, body_stamps = [], [], []
    for name in fields:
        try:
            a, b, start, end = parse_idx(text, name, "surface")
        except ValueError:
            raise PipelineError("required_radiation_record_missing") from None
        if end != step:
            raise PipelineError("idx_end_step_mismatch")
        message, headers = request(url, (a, b))
        body_stamps.append(published_before_cutoff(target, headers))
        means.append(regional_means(message, start, end))
        starts.append(start)
    if len(set(body_stamps)) != 1:
        raise PipelineError("source_changed_during_read")
    return np.stack(means), np.array(starts), (idx_stamp, body_stamps[0])


def preflight():
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(), ready=False,
                  pipeline_name="gfs_sflux_optics20", competition_data_accessed=False,
                  full_global_arrays_saved=False, samples=[])
    try:
        for target in TARGETS:
            values, starts, _ = read_records(target, 27, (*FIELDS, "DSWRF"))
            if not np.all(starts == 24):
                raise PipelineError("sample_average_window_mismatch")
            total = values[1] + values[2] + values[3] + values[4]
            error = np.abs(total - values[5])
            closed = bool(np.all(error <= 2 + .01 * np.abs(values[5])))
            result["samples"].append(dict(target_day=target.isoformat(), energy_closure_passed=closed,
                                         max_closure_error_wm2=round(float(error.max()), 4)))
            if not closed:
                raise PipelineError("beam_diffuse_energy_closure_failed")
            if target == TARGETS[1]:
                repeated, repeated_starts, _ = read_records(target, 27)
                identical = np.array_equal(values[:5], repeated) and np.array_equal(starts[:5], repeated_starts)
                result["repeat_retrieval_regional_means_identical"] = bool(identical)
                if not identical:
                    raise PipelineError("repeat_retrieval_mismatch")
        # 다운로드 전 고정 최신 표본에서 모든3/6시간 창이 복원 가능한지 확인한다.
        windows = {name: {} for name in FIELDS}
        for step in STEPS:
            payload, headers = request(url_for(TARGETS[-1], step) + ".idx")
            published_before_cutoff(TARGETS[-1], headers)
            for name in FIELDS:
                _, _, start, end = parse_idx(payload.decode("ascii"), name, "surface")
                if end != step:
                    raise PipelineError("sample_end_step_mismatch")
                windows[name][step] = (start, np.array(0.))
        for averages in windows.values():
            block_means(averages)
        result["ready"] = True
    except Exception as exc:
        result.update(error_summary(exc))
    save(ENV_PATH, result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def classify_day(target, listed):
    missing = [k for k in required_keys() if k not in listed]
    if missing:
        return dict(status="missing", missing=len(missing))
    latest = max(listed[k] for k in required_keys())
    return dict(status="eligible" if latest <= prediction_cutoff(target.isoformat()) else "late",
                last_modified=latest.isoformat())


def check_day(target):
    try:
        prefix = run_prefix(target) + "0"
        payload, _ = request(f"{BUCKET}/?list-type=2&prefix={prefix}")
        return target.isoformat(), classify_day(target, parse_listing(payload))
    except Exception as exc:
        return target.isoformat(), dict(status="network_unverified", **error_summary(exc))


def audit(workers):
    identity = dict(schema=SCHEMA, first=FIRST.isoformat(), last=LAST.isoformat(), required_keys=required_keys())
    result = json.loads(AUDIT_PATH.read_text()) if AUDIT_PATH.exists() else {**identity, "days": {}}
    if any(result.get(k) != v for k, v in identity.items()):
        raise PipelineError("audit_checkpoint_mismatch")
    days = [FIRST + timedelta(days=i) for i in range((LAST - FIRST).days + 1)]
    remaining = [d for d in days if result["days"].get(d.isoformat(), {}).get("status") not in ("eligible", "late", "missing")]
    with ThreadPoolExecutor(workers) as pool:
        futures = [pool.submit(check_day, d) for d in remaining]
        for i, future in enumerate(as_completed(futures), 1):
            day, row = future.result()
            result["days"][day] = row
            if i % 25 == 0:
                save(AUDIT_PATH, result)
                print(json.dumps(dict(audit_completed=len(result["days"]), total_days=len(days))), flush=True)
    for part, start, end in (("train", FIRST, EVALUATION_FIRST - timedelta(days=1)),
                             ("evaluation", EVALUATION_FIRST, LAST)):
        rows = [r for d, r in result["days"].items() if start <= date.fromisoformat(d) <= end]
        counts = Counter(r["status"] for r in rows)
        result[part] = dict(days=len(rows), **dict(counts), eligible_share=counts["eligible"] / len(rows))
    result["passed_internal_coverage_gate"] = all(result[k]["eligible_share"] >= .95 for k in ("train", "evaluation"))
    save(AUDIT_PATH, result)
    print(json.dumps({k: result[k] for k in ("train", "evaluation", "passed_internal_coverage_gate")}), flush=True)
    return result


def cache_valid(path, target):
    try:
        with np.load(path, allow_pickle=False) as z:
            expected = make_features(z["means"], z["starts"])
            return bool(str(z["schema"]) == SCHEMA and str(z["target_day"]) == target.isoformat()
                        and np.array_equal(z["source_steps"], np.array(STEPS))
                        and np.array_equal(z["columns"], np.array(COLUMNS))
                        and np.isfinite(z["means"]).all() and not np.isinf(z["features"]).any()
                        and z["features"].shape == (17, 20)
                        and np.allclose(z["features"], expected, equal_nan=True, rtol=1e-6, atol=1e-6)
                        and z["last_modified"].shape == (12,)
                        and all(datetime.fromisoformat(s) <= prediction_cutoff(target.isoformat())
                                for s in z["last_modified"].tolist()))
    except Exception:
        return False


def fetch_day(target):
    path = DIRECTORY / f"{target:%Y%m%d}.npz"
    if path.exists():
        if not cache_valid(path, target):
            raise PipelineError("existing_cache_invalid")
        return "reused"
    means, starts, stamps = [], [], []
    for step in STEPS:
        values, beginnings, modified = read_records(target, step)
        means.append(values)
        starts.append(beginnings)
        stamps.extend(modified)
    means, starts = np.stack(means, axis=1), np.stack(starts, axis=1)
    features = make_features(means, starts)
    temp = path.with_suffix(".tmp.npz")
    np.savez_compressed(temp, means=means, starts=starts, features=features, schema=SCHEMA,
                        target_day=target.isoformat(), source_steps=STEPS, columns=COLUMNS,
                        last_modified=np.array(stamps))
    temp.replace(path)
    return "downloaded"


def fetch(result, workers):
    if not result.get("passed_internal_coverage_gate"):
        raise PipelineError("coverage_gate_failed")
    directory = DIRECTORY.resolve()
    if ROOT == directory or ROOT in directory.parents or directory != (DATA_ROOT / "external" / "gfs_sflux_optics_00utc").resolve():
        raise PipelineError("unsafe_data_destination")
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    days = sorted(date.fromisoformat(d) for d, row in result["days"].items()
                  if row["status"] == "eligible" and FIRST.isoformat() <= d <= DEVELOPMENT_LAST.isoformat())
    counts, errors = Counter(), Counter()
    started = time.monotonic()
    def checkpoint():
        elapsed = max(1., time.monotonic() - started)
        completed = sum(counts.values())
        summary = dict(schema=SCHEMA, first=FIRST.isoformat(), last=DEVELOPMENT_LAST.isoformat(),
                       eligible_days=len(days), completed=completed, counts=dict(counts), error_codes=dict(errors),
                       elapsed_seconds=round(elapsed, 1),
                       eta_minutes=round((len(days) - completed) * elapsed / completed / 60, 1) if completed else None,
                       complete=completed == len(days) and counts["failed"] == 0)
        save(DOWNLOAD_PATH, summary)
        print(json.dumps(summary), flush=True)
        return summary
    checkpoint()
    with ThreadPoolExecutor(workers) as pool:
        futures = [pool.submit(fetch_day, d) for d in days]
        for i, future in enumerate(as_completed(futures), 1):
            try:
                counts[future.result()] += 1
            except Exception as exc:
                counts["failed"] += 1
                errors[str(exc) if isinstance(exc, PipelineError) else type(exc).__name__] += 1
            if i % 10 == 0:
                checkpoint()
            if counts["failed"] >= 4 and counts["downloaded"] + counts["reused"] == 0:
                for pending in futures:
                    pending.cancel()
                break
    return checkpoint()


def status():
    def read(path):
        return json.loads(path.read_text()) if path.exists() else {}
    environment, coverage, downloaded, model = (read(p) for p in (ENV_PATH, AUDIT_PATH, DOWNLOAD_PATH, MODEL_PATH))
    return dict(environment_ready=environment.get("ready"), audited_days=len(coverage.get("days", {})),
                coverage_passed=coverage.get("passed_internal_coverage_gate"),
                download=downloaded, model_windows={s: {k: len(v) for k, v in model.get(s, {}).items()}
                                                  for s in ("seeds1", "seeds3")},
                verdict1=model.get("verdict_vs_control"), verdict3=model.get("verdict3_vs_control"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "audit", "fetch", "validate", "pipeline", "status"))
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=4)
    args = parser.parse_args()
    try:
        if args.action == "status":
            print(json.dumps(status(), ensure_ascii=False), flush=True)
            return 0
        if args.action == "validate":
            from .gfs_radiation_model import main as validate
            validate()
            return 0
        environment = preflight()
        if not environment["ready"]:
            return 2
        if args.action == "preflight":
            return 0
        result = audit(args.workers) if args.action in ("audit", "pipeline") else json.loads(AUDIT_PATH.read_text())
        if not result.get("passed_internal_coverage_gate"):
            return 3
        if args.action == "audit":
            return 0
        downloaded = fetch(result, args.workers)
        if not downloaded["complete"]:
            return 4
        if args.action == "pipeline":
            from .gfs_radiation_model import main as validate
            validate()
        return 0
    except Exception as exc:
        print(json.dumps(dict(status="stopped", **error_summary(exc))), flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
