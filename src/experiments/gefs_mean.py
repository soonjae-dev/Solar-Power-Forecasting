"""GEFS 평균: 익명 HTTPS 환경 점검, 공개 시각 감사, 재개 가능한 수집.

    .venv/bin/python -m src.experiments.gefs_mean preflight
    .venv/bin/python -m src.experiments.gefs_mean pipeline

pipeline은 감사 통과 후 2025-04-30까지 수집하고 교체·추가 모델을 순차 검증한다.
제공 데이터는 네트워크 요청에 사용하지 않으며 출력은 집계뿐이다.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
import importlib.metadata
import json
from pathlib import Path
import socket
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np

from ..config import DATA_ROOT, EXP_DIR, ROOT
from .external_weather_audit import parse_listing, prediction_cutoff
from ..gfs_fetch import FIELDS, LATS, LONS, decode, parse_idx

BUCKET = "https://noaa-gefs-pds.s3.amazonaws.com"
STEPS = (21, 24, 27, 30, 33, 36)
FIRST, LAST = date(2021, 1, 2), date(2026, 5, 31)
DEVELOPMENT_LAST = date(2025, 4, 30)
EVALUATION_FIRST = date(2025, 6, 1)
DIRECTORY = DATA_ROOT / "external" / "gefs_mean_00utc"
ENV_PATH = EXP_DIR / "p40_gefs_mean_environment" / "metrics.json"
AUDIT_PATH = EXP_DIR / "p41_gefs_mean_coverage" / "metrics.json"
DOWNLOAD_PATH = EXP_DIR / "p42_gefs_mean_download" / "metrics.json"
SCHEMA = "gefs_mean_3hour_blocks_v1"


class PipelineError(ValueError):
    """공개 가능한 고정 오류 코드만 담는다. 응답/데이터 본문은 넣지 않는다."""


def save(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    temp.replace(path)


def error_summary(exc):
    """예외 본문/응답/데이터 행/인증정보를 출력하지 않는다."""
    reason = getattr(exc, "reason", exc)
    summary = dict(error_type=type(exc).__name__, reason_type=type(reason).__name__,
                   errno=getattr(reason, "errno", None), http_status=getattr(exc, "code", None))
    if isinstance(exc, PipelineError):
        summary["failure_code"] = str(exc)
    return summary


def request(url, byte_range=None):
    headers = {"Range": f"bytes={byte_range[0]}-{byte_range[1]}"} if byte_range else {}
    for attempt in range(3):
        try:
            with urlopen(Request(url, headers=headers), timeout=20) as response:
                info = dict(response.headers)
                payload = response.read(32 * 1024 * 1024 + 1)
                if len(payload) > 32 * 1024 * 1024:
                    raise PipelineError("response_size_limit")
                if byte_range:
                    a, b = byte_range
                    content_range = response.headers.get("Content-Range", "")
                    if response.status != 206 or not content_range.startswith(f"bytes {a}-{b}/") or len(payload) != b - a + 1:
                        raise PipelineError("range_response_mismatch")
                return payload, info
        except HTTPError as exc:
            if exc.code < 500 or attempt == 2:
                raise
        except (URLError, OSError, TimeoutError):
            if attempt == 2:
                raise


def run_url(target, step):
    run = target - timedelta(days=1)
    return f"{BUCKET}/gefs.{run:%Y%m%d}/00/atmos/pgrb2sp25/geavg.t00z.pgrb2s.0p25.f{step:03d}"


def required_keys():
    names = [f"geavg.t00z.pgrb2s.0p25.f{s:03d}" for s in STEPS]
    return names + [n + ".idx" for n in names]


def published_before_cutoff(target, headers):
    value = next((v for k, v in headers.items() if k.lower() == "last-modified"), None)
    if value is None:
        raise PipelineError("last_modified_missing")
    stamp = parsedate_to_datetime(value)
    if stamp.tzinfo is None or stamp > prediction_cutoff(target.isoformat()):
        raise PipelineError("last_modified_after_cutoff")
    return stamp.isoformat()


def mean_record(text, name, level, step):
    try:
        a, b, start, end = parse_idx(text, name, level)
    except ValueError:
        raise PipelineError("idx_average_missing_or_ambiguous") from None
    selected = [line.split(":") for line in text.splitlines() if line and int(line.split(":")[1]) == a]
    if (end != step or len(selected) != 1
            or not any(part.strip() == "ens mean" for part in selected[0][6:])):
        raise PipelineError("idx_not_mean_or_wrong_step")
    return a, b, start, end


def block_means(averages):
    """3/6시간 누적 평균에서 겹치지 않는3시간 평균을 복원한다."""
    out = {}
    for step in STEPS:
        start, mean = averages[step]
        if step - start == 3:
            out[step] = mean
        elif step - start == 6:
            if step - 3 not in averages or averages[step - 3][0] != start:
                raise PipelineError("preceding_average_missing_or_misaligned")
            out[step] = 2 * mean - averages[step - 3][1]
        else:
            raise PipelineError("unsupported_average_interval")
    return out


def expand_blocks(blocks):
    """f019~f035의 시간 끝시각을 해당3시간 평균에 연결한다. 선형 보간 아님."""
    return np.stack([blocks[((s - 1) // 3 + 1) * 3] for s in range(19, 36)])


def preflight():
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(), ready=False,
                  public_requests_only=True, competition_data_accessed=False, dependencies={})
    for name in ("eccodes", "numpy", "pandas", "lightgbm"):
        try:
            result["dependencies"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result["dependencies"][name] = None
    if not all(result["dependencies"].values()):
        result["blocked_at"] = "dependencies"
        save(ENV_PATH, result)
        return result
    target = date(2024, 8, 16)
    try:
        socket.getaddrinfo("noaa-gefs-pds.s3.amazonaws.com", 443)
        result["dns"] = "ok"
        intervals = Counter()
        sample_averages = {k: {} for k, _, _ in FIELDS}
        for step in STEPS:
            payload, headers = request(run_url(target, step) + ".idx")
            published_before_cutoff(target, headers)
            text = payload.decode("ascii")
            for key, name, level in FIELDS:
                _, _, start, end = mean_record(text, name, level, step)
                intervals[str(end - start)] += 1
                sample_averages[key][step] = (start, np.array(0.0))
        for values in sample_averages.values():
            block_means(values)
        result["idx_average_width_counts"] = dict(intervals)
        payload, _ = request(run_url(target, 27) + ".idx")
        text = payload.decode("ascii")
        identical = []
        for _, name, level in FIELDS:
            a, b, _, _ = mean_record(text, name, level, 27)
            first, h1 = request(run_url(target, 27), (a, b))
            second, h2 = request(run_url(target, 27), (a, b))
            published_before_cutoff(target, h1)
            published_before_cutoff(target, h2)
            x, y = decode(first), decode(second)
            if x.shape != (len(LATS), len(LONS)) or not np.isfinite(x).all():
                raise PipelineError("decoded_grid_invalid")
            identical.append(bool(np.array_equal(x, y)))
        if not all(identical):
            raise PipelineError("repeat_retrieval_mismatch")
        result.update(ready=True, https="ok", decoded_fields=len(identical),
                      repeat_retrieval_identical=all(identical))
    except Exception as exc:
        result.update(blocked_at="dns" if result.get("dns") != "ok" else "public_sample", **error_summary(exc))
    save(ENV_PATH, result)
    return result


def classify_day(target, listed):
    absent = [k for k in required_keys() if k not in listed]
    if absent:
        return dict(status="missing", missing=len(absent))
    latest = max(listed[k] for k in required_keys())
    return dict(status="eligible" if latest <= prediction_cutoff(target.isoformat()) else "late",
                last_modified=latest.isoformat())


def check_day(target):
    run = target - timedelta(days=1)
    prefix = f"gefs.{run:%Y%m%d}/00/atmos/pgrb2sp25/geavg.t00z.pgrb2s.0p25.f0"
    try:
        payload, _ = request(f"{BUCKET}/?list-type=2&prefix={prefix}")
        return target.isoformat(), classify_day(target, parse_listing(payload))
    except Exception as exc:
        return target.isoformat(), dict(status="network_unverified", **error_summary(exc))


def summarize_audit(rows):
    def part(a, b):
        chosen = [r for d, r in rows.items() if a <= date.fromisoformat(d) <= b]
        counts = Counter(r["status"] for r in chosen)
        return dict(days=len(chosen), **{s: counts[s] for s in ("eligible", "late", "missing", "network_unverified")},
                    eligible_share=counts["eligible"] / max(1, len(chosen)))
    return dict(train=part(FIRST, EVALUATION_FIRST - timedelta(days=1)),
                evaluation=part(EVALUATION_FIRST, LAST))


def audit(workers=8):
    identity = dict(schema=SCHEMA, first=FIRST.isoformat(), last=LAST.isoformat(),
                    required_keys=required_keys(), rule="all bodies and idx LastModified <= previous day 08UTC")
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
    result.update(summarize_audit(result["days"]))
    result["passed_internal_coverage_gate"] = all(result[k]["eligible_share"] >= .95 for k in ("train", "evaluation"))
    save(AUDIT_PATH, result)
    print(json.dumps({k: result[k] for k in ("train", "evaluation", "passed_internal_coverage_gate")}), flush=True)
    return result


def validate_directory(directory):
    directory = directory.resolve()
    if directory == ROOT or ROOT in directory.parents or directory != (DATA_ROOT / "external" / "gefs_mean_00utc").resolve():
        raise PipelineError("unsafe_data_destination")


def cache_valid(path, target):
    try:
        with np.load(path, allow_pickle=False) as z:
            return (str(z["schema"]) == SCHEMA and str(z["target_day"]) == target.isoformat()
                    and all(z[k].shape == (17, len(LATS), len(LONS)) and np.isfinite(z[k]).all() for k, _, _ in FIELDS)
                    and np.array_equal(z["source_steps"], np.array(STEPS))
                    and (z["dswrf"] >= 0).all() and ((z["tcdc"] >= 0) & (z["tcdc"] <= 100)).all()
                    and z["last_modified"].shape == (12,)
                    and all(datetime.fromisoformat(s) <= prediction_cutoff(target.isoformat()) for s in z["last_modified"].tolist()))
    except Exception:
        return False


def fetch_day(target):
    path = DIRECTORY / f"{target:%Y%m%d}.npz"
    if path.exists():
        if not cache_valid(path, target):
            raise PipelineError("existing_cache_invalid")
        return "reused"
    averages = {k: {} for k, _, _ in FIELDS}
    last_modified = []
    for step in STEPS:
        url = run_url(target, step)
        payload, headers = request(url + ".idx")
        last_modified.append(published_before_cutoff(target, headers))
        text = payload.decode("ascii")
        body_stamps = []
        for key, name, level in FIELDS:
            a, b, start, _ = mean_record(text, name, level, step)
            payload, headers = request(url, (a, b))
            body_stamps.append(published_before_cutoff(target, headers))
            averages[key][step] = (start, decode(payload))
        if body_stamps[0] != body_stamps[1]:
            raise PipelineError("source_changed_during_read")
        last_modified.append(body_stamps[0])
    arrays = {k: np.clip(expand_blocks(block_means(v)), 0, 100 if k == "tcdc" else None).astype("float32")
              for k, v in averages.items()}
    if any(not np.isfinite(v).all() for v in arrays.values()):
        raise PipelineError("decoded_values_nonfinite")
    temp = path.with_suffix(".tmp.npz")
    np.savez_compressed(temp, **arrays, schema=SCHEMA, target_day=target.isoformat(),
                        last_modified=np.array(last_modified), source_steps=np.array(STEPS))
    temp.replace(path)
    return "downloaded"


def fetch(result, workers=8):
    if not result.get("passed_internal_coverage_gate"):
        raise PipelineError("coverage_gate_failed")
    validate_directory(DIRECTORY)
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    days = [date.fromisoformat(d) for d, r in result["days"].items()
            if r["status"] == "eligible" and FIRST <= date.fromisoformat(d) <= DEVELOPMENT_LAST]
    counts, errors = Counter(), Counter()
    with ThreadPoolExecutor(workers) as pool:
        futures = [pool.submit(fetch_day, d) for d in sorted(days)]
        for i, future in enumerate(as_completed(futures), 1):
            try:
                counts[future.result()] += 1
            except Exception as exc:
                counts["failed"] += 1
                errors[str(exc) if isinstance(exc, PipelineError) else type(exc).__name__] += 1
            if i % 25 == 0:
                print(json.dumps(dict(download_completed=i, eligible_days=len(days), **counts)), flush=True)
    summary = dict(schema=SCHEMA, first=FIRST.isoformat(), last=DEVELOPMENT_LAST.isoformat(),
                   eligible_days=len(days), counts=dict(counts), error_types=dict(errors),
                   complete=counts["failed"] == 0 and counts["downloaded"] + counts["reused"] == len(days))
    save(DOWNLOAD_PATH, summary)
    print(json.dumps(summary), flush=True)
    return summary


def status():
    """파일 내용 대신 집계 metrics와 일별 파일 개수로 상태를 확인한다."""
    def read(path):
        return json.loads(path.read_text()) if path.exists() else {}
    environment, coverage, downloaded = read(ENV_PATH), read(AUDIT_PATH), read(DOWNLOAD_PATH)
    model = read(EXP_DIR / "p43_gefs_mean_model" / "metrics.json")
    added_model = read(EXP_DIR / "p44_gefs_mean_added" / "metrics.json")
    days = coverage.get("days", {})
    expected = sum(r.get("status") == "eligible" and FIRST.isoformat() <= d <= DEVELOPMENT_LAST.isoformat()
                   for d, r in days.items())
    files = sum(1 for p in DIRECTORY.glob("*.npz") if len(p.stem) == 8 and p.stem.isdecimal()
                and f"{FIRST:%Y%m%d}" <= p.stem <= f"{DEVELOPMENT_LAST:%Y%m%d}") if DIRECTORY.exists() else 0
    verdict1, verdict3 = model.get("verdict_vs_control"), model.get("verdict3_vs_control")
    def terminal(m):
        v3, v1 = m.get("verdict3_vs_control"), m.get("verdict_vs_control")
        if v3 is not None:
            return bool(v3["passed"])
        if v1 is not None and not v1["passed"]:
            return False
        return None
    replaced, added = terminal(model), terminal(added_model)
    if not environment.get("ready"):
        state = "environment_not_ready"
    elif "passed_internal_coverage_gate" not in coverage:
        state = "auditing"
    elif not coverage["passed_internal_coverage_gate"]:
        state = "coverage_gate_failed"
    elif not downloaded.get("complete"):
        state = "collecting"
    elif replaced is None:
        state = "model_validation_replace"
    elif added is None:
        state = "model_validation_add"
    else:
        state = "candidate" if replaced or added else "rejected"
    return dict(state=state, environment_ready=environment.get("ready", False),
                audited_days=len(days), total_audit_days=(LAST - FIRST).days + 1,
                audit_statuses=dict(Counter(r.get("status") for r in days.values())),
                development_files=files, eligible_development_days=expected,
                download_report=downloaded,
                model_completed_windows={stage: {k: len(v) for k, v in model.get(stage, {}).items()}
                                         for stage in ("seeds1", "seeds3")},
                verdict1=verdict1, verdict3=verdict3,
                added_model_completed_windows={stage: {k: len(v) for k, v in added_model.get(stage, {}).items()}
                                               for stage in ("seeds1", "seeds3")},
                added_verdict1=added_model.get("verdict_vs_control"),
                added_verdict3=added_model.get("verdict3_vs_control"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("status", "preflight", "audit", "fetch", "validate", "pipeline"))
    parser.add_argument("--workers", type=int, default=8, choices=range(1, 17))
    args = parser.parse_args()
    try:
        if args.action == "status":
            print(json.dumps(status(), ensure_ascii=False), flush=True)
            return 0
        if args.action == "validate":
            from .gefs_mean_model import main as validate
            validate()
            return 0
        environment = preflight()
        print(json.dumps(environment, ensure_ascii=False), flush=True)
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
            from .gefs_mean_model import main as validate
            validate()
        return 0
    except Exception as exc:
        print(json.dumps(dict(status="stopped", **error_summary(exc))), flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
