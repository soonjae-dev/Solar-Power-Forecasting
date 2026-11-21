"""공개 ECMWF IFS(open data) 일사 ssrd를 받아 고정 상자 시간값으로 저장한다. 대회 데이터는 읽지 않는다.

    .venv/bin/python -m src.experiments.ecmwf_fetch 2024-03-03 2026-05-31

대상일 D마다 D-2 18z scda 실행의 step 24~42(3시간 간격) 누적 일사를 받는다.
받은 GRIB의 Last-Modified 최댓값이 D-1 08:00 UTC 이하인 날만 저장한다. 다시 실행하면 이어 받는다.
"""
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np

from ..config import DATA_ROOT
from ..gfs_fetch import LATS, LONS

BUCKET = "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com"
OUT_DIR = DATA_ROOT / "external" / "ecmwf_18utc"
STEPS = tuple(range(24, 43, 3))                     # 누적 일사 경계. 구간 (24,27] … (39,42]
FIRST_TS_HOUR, HOURS = 4, 17                        # ts 04~20시 KST = step 25~41
AUDIT = Path(__file__).resolve().parents[2] / "experiments/p67_ecmwf_coverage/metrics.json"


def get(url, byte_range=None, retries=7):
    headers = {"Range": f"bytes={byte_range[0]}-{byte_range[1]}"} if byte_range else {}
    for attempt in range(retries):
        try:
            with urlopen(Request(url, headers=headers), timeout=60) as r:
                return r.read(), parsedate_to_datetime(r.headers["Last-Modified"])
        except HTTPError as e:
            if e.code in (403, 404):
                raise FileNotFoundError(url) from None
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt + random.random())                # 503 요청 과다는 간격을 늘려 다시 시도한다
        except (URLError, OSError, TimeoutError):
            if attempt == retries - 1:
                raise


def run_url(run_day, step, stream="scda"):
    d = f"{run_day:%Y%m%d}"
    return f"{BUCKET}/{d}/18z/ifs/0p25/{stream}/{d}180000-{step}h-{stream}-fc"


def locate(run_day, step):
    """18z 실행은 2025-12~2026-05 사이에 스트림 이름이 scda에서 oper로 바뀌었다. 같은 실행이다."""
    for stream in ("scda", "oper"):
        try:
            return run_url(run_day, step, stream), get(run_url(run_day, step, stream) + ".index")[0]
        except FileNotFoundError:
            continue
    raise FileNotFoundError(f"{run_day} {step}")


def find_ssrd(index_text, step):
    found = [json.loads(line) for line in index_text.splitlines() if line.strip()]
    found = [x for x in found if x.get("param") == "ssrd" and x.get("step") == str(step) and x.get("levtype") == "sfc"]
    if len(found) != 1:
        raise LookupError(f"ssrd 레코드 {len(found)}개")
    return found[0]["_offset"], found[0]["_offset"] + found[0]["_length"] - 1


def crop(values, ni, nj, lat0, lon0, inc):
    """정규 위경도 격자(북→남, 서→동)에서 GFS와 같은 고정 상자를 자른다."""
    if (ni, nj, inc) != (1440, 721, .25) or lat0 != 90:
        raise ValueError("격자가 예상과 다르다")
    field = values.reshape(nj, ni)
    rows = np.round((lat0 - LATS) * 4).astype(int)
    cols = np.round(((LONS - lon0) % 360) * 4).astype(int)
    return field[np.ix_(rows, cols)].astype("float64")


def decode(message, step):
    import eccodes
    gid = eccodes.codes_new_from_message(message)
    try:
        if eccodes.codes_get(gid, "shortName") != "ssrd" or str(eccodes.codes_get(gid, "stepRange")) not in (str(step), f"0-{step}"):
            raise ValueError("레코드 종류/시각 불일치")
        if eccodes.codes_get(gid, "jScansPositively") != 0 or eccodes.codes_get(gid, "iScansNegatively") != 0:
            raise ValueError("격자 순서가 예상과 다르다")
        return crop(eccodes.codes_get_values(gid), eccodes.codes_get(gid, "Ni"), eccodes.codes_get(gid, "Nj"),
                    eccodes.codes_get(gid, "latitudeOfFirstGridPointInDegrees"),
                    eccodes.codes_get(gid, "longitudeOfFirstGridPointInDegrees"),
                    eccodes.codes_get(gid, "iDirectionIncrementInDegrees"))
    finally:
        eccodes.codes_release(gid)


def hourly(acc):
    """{step: 누적 J/m²} → ts 04~20시 각 시각의 W/m²(속한 3시간 구간 평균)."""
    means = {b: (acc[b] - acc[b - 3]) / 10800 for b in STEPS[1:]}
    out = []
    for step in range(25, 42):
        out.append(means[-(-step // 3) * 3])
    return np.clip(np.stack(out), 0, None).astype("float32")


def deadline(target):
    return datetime(target.year, target.month, target.day, 8, tzinfo=timezone.utc) - timedelta(days=1)


def fetch_day(target, out_dir=OUT_DIR):
    path = out_dir / f"{target:%Y%m%d}.npz"
    if path.exists():
        return "skip"
    for mark in ("late", "missing"):
        if (out_dir / f"{target:%Y%m%d}.{mark}").exists():
            return mark
    run_day = target - timedelta(days=2)
    acc, latest = {}, None
    try:
        for step in STEPS:
            url, text = locate(run_day, step)
            message, modified = get(url + ".grib2", find_ssrd(text.decode(), step))
            acc[step] = decode(message, step)
            latest = modified if latest is None else max(latest, modified)
    except (FileNotFoundError, LookupError):
        (out_dir / f"{target:%Y%m%d}.missing").touch()
        return "missing"
    except (URLError, OSError, TimeoutError):                       # 일시 오류는 표시하지 않고 다음 실행에서 다시 받는다
        return "error"
    if latest > deadline(target):
        (out_dir / f"{target:%Y%m%d}.late").touch()
        return "late"
    temp = path.with_suffix(".tmp.npz")
    np.savez_compressed(temp, ssrd=hourly(acc), last_modified=np.array(latest.isoformat()))
    temp.replace(path)
    return "new"


def main(first, last, workers=16):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    days = [date.fromisoformat(first) + timedelta(days=i) for i in range((date.fromisoformat(last) - date.fromisoformat(first)).days + 1)]
    counts = {}
    with ThreadPoolExecutor(workers) as pool:
        for i, (day, status) in enumerate(zip(days, pool.map(fetch_day, days)), 1):
            counts[status] = counts.get(status, 0) + 1
            if i % 50 == 0 or i == len(days):
                print(f"{i}/{len(days)} {counts}", flush=True)
    stored = sorted(p.stem for p in OUT_DIR.glob("*.npz") if not p.stem.endswith(".tmp"))
    result = dict(rule="14.64", first=first, last=last, target_days=len(days),
                  stored=len(stored), first_stored=stored[0] if stored else None,
                  late=sorted(p.stem for p in OUT_DIR.glob("*.late")), missing=sorted(p.stem for p in OUT_DIR.glob("*.missing")))
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: (v if not isinstance(v, list) else len(v)) for k, v in result.items()}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:3], *[int(x) for x in sys.argv[3:4]])
