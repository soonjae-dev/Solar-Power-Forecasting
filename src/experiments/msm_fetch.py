"""JMA MSM(교토대 RISH 보관본)의 1시간 평균 일사·전운량을 고정 영역 평균으로 저장한다. 대회 데이터는 읽지 않는다.

    .venv/bin/python -m src.experiments.msm_fetch 2024-03-01 2025-05-31 [작업자 수] [간격 일수]

대상일 D마다 D-1 00UTC 실행의 FH16-33·FH34-39 지표 파일을 메모리로 받아 step 19~35(ts 04~20시 KST)만 쓴다.
두 파일의 Last-Modified가 D-1 08:00 UTC 이하인 날만 저장한다. 영역은 GFS와 같은 고정 3×3 2도 칸 + 전체다.
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import numpy as np

from ..config import DATA_ROOT

BASE = "http://database.rish.kyoto-u.ac.jp/arch/jmadata/data/gpv/original"
OUT_DIR = DATA_ROOT / "external" / "msm_00utc"
STEPS = range(19, 36)                               # ts 04~20시 KST
AUDIT = Path(__file__).resolve().parents[2] / "experiments/p70_msm_coverage/metrics.json"


def masks():
    """MSM 격자(47.6N→22.4N 0.05도, 120E→150E 0.0625도)에서 GFS와 같은 고정 영역."""
    lat = 47.6 - np.arange(505) * 0.05
    lon = 120 + np.arange(481) * 0.0625
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    out = {"all": (la >= 33) & (la < 39) & (lo >= 124) & (lo < 130)}
    for i, south in enumerate((33, 35, 37)):
        for j, west in enumerate((124, 126, 128)):
            out[f"r{i}{j}"] = (la >= south) & (la < south + 2) & (lo >= west) & (lo < west + 2)
    return out


MASKS = masks()


def get(url, retries=5):
    for attempt in range(retries):
        try:
            with urlopen(url, timeout=300) as r:
                return r.read(), parsedate_to_datetime(r.headers["Last-Modified"])
        except HTTPError as e:
            if e.code == 404:
                raise FileNotFoundError(url) from None
            if attempt == retries - 1:
                raise
        except (URLError, OSError, TimeoutError):
            if attempt == retries - 1:
                raise


def fields(blob):
    """다중 필드 GRIB2에서 {('dswrf'|'tcc', 끝 step): 영역 평균 배열}을 만든다. eccodes는 실제 파일 핸들이 필요해 임시 파일을 쓴다."""
    import tempfile
    import eccodes
    eccodes.codes_grib_multi_support_on()
    out = {}
    with tempfile.TemporaryFile() as handle:
        handle.write(blob)
        handle.seek(0)
        while True:
            g = eccodes.codes_grib_new_from_file(handle)
            if g is None:
                break
            try:
                key = (eccodes.codes_get(g, "parameterCategory"), eccodes.codes_get(g, "parameterNumber"))
                name = {(4, 7): "dswrf", (6, 1): "tcc"}.get(key)
                if name is None:
                    continue
                if (eccodes.codes_get(g, "Ni"), eccodes.codes_get(g, "Nj")) != (481, 505) or eccodes.codes_get(g, "jScansPositively") != 0:
                    raise ValueError("격자가 예상과 다르다")
                rng = str(eccodes.codes_get(g, "stepRange"))
                end = int(rng.split("-")[-1])
                if name == "dswrf" and rng != f"{end - 1}-{end}":
                    raise ValueError(f"일사 평균 구간이 1시간이 아니다: {rng}")
                v = eccodes.codes_get_values(g).reshape(505, 481)
                out[(name, end)] = np.array([v[m].mean() for m in MASKS.values()], dtype="float32")
            finally:
                eccodes.codes_release(g)
    return out


def deadline(target):
    return datetime(target.year, target.month, target.day, 8, tzinfo=timezone.utc) - timedelta(days=1)


def fetch_day(target, out_dir=OUT_DIR):
    path = out_dir / f"{target:%Y%m%d}.npz"
    if path.exists():
        return "skip"
    for mark in ("late", "missing"):
        if (out_dir / f"{target:%Y%m%d}.{mark}").exists():
            return mark
    run = target - timedelta(days=1)
    stem = f"{BASE}/{run:%Y/%m/%d}/Z__C_RJTD_{run:%Y%m%d}000000_MSM_GPV_Rjp_Lsurf"
    got, latest = {}, None
    try:
        for part in ("FH16-33", "FH34-39"):
            blob, modified = get(f"{stem}_{part}_grib2.bin")
            got.update(fields(blob))
            latest = modified if latest is None else max(latest, modified)
    except FileNotFoundError:
        (out_dir / f"{target:%Y%m%d}.missing").touch()
        return "missing"
    except (URLError, OSError, TimeoutError):
        return "error"
    if latest > deadline(target):
        (out_dir / f"{target:%Y%m%d}.late").touch()
        return "late"
    if any((k, s) not in got for k in ("dswrf", "tcc") for s in STEPS):
        (out_dir / f"{target:%Y%m%d}.missing").touch()
        return "missing"
    arrays = {k: np.stack([got[(k, s)] for s in STEPS]) for k in ("dswrf", "tcc")}
    temp = path.with_suffix(".tmp.npz")
    np.savez_compressed(temp, **arrays, regions=np.array(list(MASKS)), last_modified=np.array(latest.isoformat()))
    temp.replace(path)
    return "new"


def main(first, last, workers=8, every=1):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    start, stop = date.fromisoformat(first), date.fromisoformat(last)
    days = [start + timedelta(days=i) for i in range(0, (stop - start).days + 1, int(every))]
    counts = {}
    with ThreadPoolExecutor(int(workers)) as pool:
        for i, status in enumerate(pool.map(fetch_day, days), 1):
            counts[status] = counts.get(status, 0) + 1
            if i % 20 == 0 or i == len(days):
                print(f"{i}/{len(days)} {counts}", flush=True)
    result = dict(first=first, last=last, every=int(every), target_days=len(days), counts=counts,
                  stored=len(list(OUT_DIR.glob("*.npz"))), late=len(list(OUT_DIR.glob("*.late"))),
                  missing=len(list(OUT_DIR.glob("*.missing"))))
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:5])
