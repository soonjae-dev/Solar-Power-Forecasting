"""공개 GFS에서 필요한 레코드만 받아 고정 영역 시간값으로 저장한다. 대회 데이터는 읽지 않는다.

    <eccodes가 있는 python> -m src.gfs_fetch 2024-06-01 2025-04-30 [more]

전날 00UTC 실행의 f019~f035(ts 04~20시 KST)에서 DSWRF, TCDC 구간 평균을 받는다.
p22 감사에서 eligible이 아닌 대상일은 받지 않는다. 일별 파일로 저장하므로 다시 실행하면 이어 받는다.
"""
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import numpy as np

from .config import DATA_ROOT

BUCKET = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
OUT_DIR = DATA_ROOT / "external" / "gfs_00utc"
FIRST_STEP, LAST_STEP = 19, 35
FIELDS = (("dswrf", "DSWRF", "surface"), ("tcdc", "TCDC", "entire atmosphere"))
MORE_DIR = DATA_ROOT / "external" / "gfs_00utc_more"
LAG_DIR = DATA_ROOT / "external" / "gfs_18utc_lag"
SNOW_DIR = DATA_ROOT / "external" / "gfs_00utc_snow"
SNOW_STEP = 27                                                     # 대상일 12시 KST. 적설 깊이는 하루 한 번만 쓴다
AUDIT18 = Path(__file__).resolve().parents[1] / "experiments/p28_gfs18_coverage/metrics.json"
MORE_FIELDS = (("lcdc", "LCDC", "low cloud layer"), ("mcdc", "MCDC", "middle cloud layer"), ("hcdc", "HCDC", "high cloud layer"))
PWAT = ("PWAT", "entire atmosphere (considered as a single layer)")   # 순간값이라 차분하지 않는다
# 발전소 위치·용량을 보지 않고 정한 고정 상자. 위도는 북→남, 경도는 서→동 순서로 저장한다.
LATS = np.arange(40, 31.75, -0.25)
LONS = np.arange(123, 132.25, 0.25)
AUDIT = Path(__file__).resolve().parents[1] / "experiments/p22_gfs_coverage/metrics.json"


def get(url, byte_range=None, retries=4):
    headers = {"Range": f"bytes={byte_range[0]}-{byte_range[1]}"} if byte_range else {}
    for attempt in range(retries):
        try:
            with urlopen(Request(url, headers=headers), timeout=60) as response:
                return response.read()
        except (URLError, OSError, TimeoutError):
            if attempt == retries - 1:
                raise


def parse_idx(text, name, level):
    """idx에서 구간 평균 레코드의 바이트 범위와 구간 시작·끝을 찾는다. 범위 끝은 다음 줄 시작 직전이다."""
    lines = [line.split(":") for line in text.strip().splitlines()]
    found = []
    for i, x in enumerate(lines):
        m = re.fullmatch(r"(\d+)-(\d+) hour ave fcst", x[5])
        if x[3] == name and x[4] == level and m:
            end = int(lines[i + 1][1]) - 1 if i + 1 < len(lines) else None
            found.append((int(x[1]), end, int(m.group(1)), int(m.group(2))))
    if len(found) != 1 or found[0][1] is None:
        raise ValueError(f"idx 레코드가 하나가 아니다: {name} {level} {len(found)}")
    return found[0]


def parse_instant(text, name, level, step):
    lines = [line.split(":") for line in text.strip().splitlines()]
    found = [(int(x[1]), int(lines[i + 1][1]) - 1) for i, x in enumerate(lines)
             if x[3] == name and x[4] == level and x[5] == f"{step} hour fcst" and i + 1 < len(lines)]
    if len(found) != 1:
        raise ValueError(f"idx 레코드가 하나가 아니다: {name} {level} {len(found)}")
    return found[0]


def hourly(averages):
    """{끝 step: (구간 시작, 구간 평균)}을 {끝 step: 그 1시간 평균}으로 바꾼다. 구간 시작 직후는 차분 없이 그대로다."""
    out = {}
    for step, (start, mean) in averages.items():
        if step - start == 1:
            out[step] = mean
            continue
        prev_start, prev = averages[step - 1]
        if prev_start != start:
            raise ValueError("차분할 앞 구간의 시작이 다르다")
        out[step] = (step - start) * mean - (step - 1 - start) * prev
    return out


def crop(values, ni, nj):
    """0.25도 전구 격자(90N→90S, 0→359.75E)에서 고정 상자를 자른다."""
    if (ni, nj) != (1440, 721):
        raise ValueError("격자 크기가 예상과 다르다")
    field = values.reshape(nj, ni)
    rows = np.round((90 - LATS) * 4).astype(int)
    cols = np.round(LONS * 4).astype(int)
    return field[np.ix_(rows, cols)].astype("float32")


def run_url(run_day, step, cycle=0):
    d = f"{run_day:%Y%m%d}"
    folder = "atmos/" if run_day >= date(2021, 3, 23) else ""     # 2021-03-22 실행까지는 atmos/ 없는 경로다
    return f"{BUCKET}/gfs.{d}/{cycle:02d}/{folder}gfs.t{cycle:02d}z.pgrb2.0p25.f{step:03d}"


def decode(message):
    import eccodes
    gid = eccodes.codes_new_from_message(message)
    try:
        if eccodes.codes_get(gid, "jScansPositively") != 0 or eccodes.codes_get(gid, "iScansNegatively") != 0:
            raise ValueError("격자 순서가 예상과 다르다")
        values = eccodes.codes_get_values(gid)
        if eccodes.codes_get(gid, "bitmapPresent"):                   # 바다 등 정의되지 않은 격자(적설 깊이 등)는 결측으로 둔다
            values = np.where(values == eccodes.codes_get(gid, "missingValue"), np.nan, values)
        return crop(values, eccodes.codes_get(gid, "Ni"), eccodes.codes_get(gid, "Nj"))
    finally:
        eccodes.codes_release(gid)


def fetch_day(target, fields=FIELDS, out_dir=OUT_DIR, pwat=False, cycle=0):
    """cycle=18이면 전전날 18UTC 실행에서 같은 대상 시각(f025~f041)을 받는다. 저장 순서는 ts 04~20시로 같다."""
    path = out_dir / f"{target:%Y%m%d}.npz"
    if path.exists():
        return "skip"
    run_day = target - timedelta(days=2 if cycle == 18 else 1)
    shift = 6 if cycle == 18 else 0
    averages = {key: {} for key, _, _ in fields}
    instant = []
    for step in range(FIRST_STEP + shift, LAST_STEP + shift + 1):
        url = run_url(run_day, step, cycle)
        text = get(url + ".idx").decode("ascii")
        for key, name, level in fields:
            start, end, a, b = parse_idx(text, name, level)
            if b != step:
                raise ValueError("idx의 구간 끝이 예보 시각과 다르다")
            averages[key][step] = (a, decode(get(url, (start, end))))
        if pwat:
            instant.append(decode(get(url, parse_instant(text, *PWAT, step))))
    arrays = {"pwat": np.stack(instant).astype("float32")} if pwat else {}
    for key, _, _ in fields:
        hours = hourly(averages[key])
        arrays[key] = np.stack([hours[s] for s in range(FIRST_STEP + shift, LAST_STEP + shift + 1)]).astype("float32")
    temp = path.with_suffix(".tmp.npz")
    np.savez_compressed(temp, **arrays)
    temp.replace(path)
    return "new"


def fetch_snow_day(target):
    path = SNOW_DIR / f"{target:%Y%m%d}.npz"
    if path.exists():
        return "skip"
    url = run_url(target - timedelta(days=1), SNOW_STEP)
    text = get(url + ".idx").decode("ascii")
    snod = decode(get(url, parse_instant(text, "SNOD", "surface", SNOW_STEP)))
    temp = path.with_suffix(".tmp.npz")
    np.savez_compressed(temp, snod=snod[None].astype("float32"))
    temp.replace(path)
    return "new"


CHEM_DIR = DATA_ROOT / "external" / "gefs_chem_00utc"
CHEM_AUDIT = Path(__file__).resolve().parents[1] / "experiments/p32_gefs_chem_coverage/metrics.json"
CHEM_STEPS = (21, 24, 27, 30, 33)                                  # 대상일 ts 06, 09, 12, 15, 18시 KST
CHEM_FIELDS = (("aod", "Total Aerosol"), ("dust", "Dust Dry"))   # 550nm 에어로졸 광학 두께


def parse_aerosol(text, aerosol, step):
    """GEFS 화학 idx에서 550nm AOTK 한 레코드를 찾는다."""
    lines = [line.split(":") for line in text.strip().splitlines()]
    found = [(int(x[1]), int(lines[i + 1][1]) - 1) for i, x in enumerate(lines)
             if x[3] == "AOTK" and x[5] == f"{step} hour fcst" and x[6].lower() == f"aerosol={aerosol}".lower()   # 2024-04부터 소문자 표기(Total aerosol)
             and x[8] == "aerosol_wavelength >=5.45e-07,<=5.65e-07" and i + 1 < len(lines)]
    if len(found) != 1:
        raise ValueError(f"idx 레코드가 하나가 아니다: AOTK {aerosol} {len(found)}")
    return found[0]


def fetch_chem_day(target):
    path = CHEM_DIR / f"{target:%Y%m%d}.npz"
    if path.exists():
        return "skip"
    run_day = target - timedelta(days=1)
    arrays = {key: [] for key, _ in CHEM_FIELDS}
    for step in CHEM_STEPS:
        url = f"https://noaa-gefs-pds.s3.amazonaws.com/gefs.{run_day:%Y%m%d}/00/chem/pgrb2ap25/gefs.chem.t00z.a2d_0p25.f{step:03d}.grib2"
        text = get(url + ".idx").decode("ascii")
        for key, aerosol in CHEM_FIELDS:
            arrays[key].append(decode(get(url, parse_aerosol(text, aerosol, step))))
    temp = path.with_suffix(".tmp.npz")
    np.savez_compressed(temp, **{k: np.stack(v).astype("float32") for k, v in arrays.items()})
    temp.replace(path)
    return "new"


SPREAD_DIR = DATA_ROOT / "external" / "gefs_spread_00utc"
SPREAD_AUDIT = Path(__file__).resolve().parents[1] / "experiments/p34_gefs_spread_coverage/metrics.json"


def fetch_spread_day(target):
    """GEFS 앙상블 표준편차(DSWRF, TCDC 구간 평균)를 CHEM_STEPS에서 받는다. 표준편차는 차분하지 않는다."""
    path = SPREAD_DIR / f"{target:%Y%m%d}.npz"
    if path.exists():
        return "skip"
    run_day = target - timedelta(days=1)
    arrays = {key: [] for key, _, _ in FIELDS}
    for step in CHEM_STEPS:
        url = f"https://noaa-gefs-pds.s3.amazonaws.com/gefs.{run_day:%Y%m%d}/00/atmos/pgrb2sp25/gespr.t00z.pgrb2s.0p25.f{step:03d}"
        text = get(url + ".idx").decode("ascii")
        for key, name, level in FIELDS:
            start, end, _, b = parse_idx(text, name, level)
            if b != step:
                raise ValueError("idx의 구간 끝이 예보 시각과 다르다")
            arrays[key].append(decode(get(url, (start, end))))
    temp = path.with_suffix(".tmp.npz")
    np.savez_compressed(temp, **{k: np.stack(v).astype("float32") for k, v in arrays.items()})
    temp.replace(path)
    return "new"


def target_days(first, last, audit=AUDIT):
    bad = {r["day"] for r in json.loads(audit.read_text())["not_eligible"]}
    days = [first + timedelta(days=i) for i in range((last - first).days + 1)]
    return [d for d in days if d.isoformat() not in bad], len(days)


def main(first, last, kind="base", workers=24):
    single = {"snow": (fetch_snow_day, SNOW_DIR, AUDIT), "chem": (fetch_chem_day, CHEM_DIR, CHEM_AUDIT), "spread": (fetch_spread_day, SPREAD_DIR, SPREAD_AUDIT)}
    if kind in single:
        single[kind][1].mkdir(parents=True, exist_ok=True)
    else:
        args = {"base": (FIELDS, OUT_DIR, False, 0), "more": (MORE_FIELDS, MORE_DIR, True, 0), "lag18": (FIELDS, LAG_DIR, False, 18)}[kind]
        args[1].mkdir(parents=True, exist_ok=True)
    workers = 48 if kind == "more" else 24
    days, total = target_days(date.fromisoformat(first), date.fromisoformat(last), AUDIT18 if kind == "lag18" else single[kind][2] if kind in single else AUDIT)
    print(f"대상일 {total}일 중 eligible {len(days)}일", flush=True)
    done = failed = 0
    with ThreadPoolExecutor(workers) as pool:
        for day, future in [(d, pool.submit(single[kind][0], d) if kind in single else pool.submit(fetch_day, d, *args)) for d in days]:
            try:
                future.result()
                done += 1
            except Exception as exc:
                failed += 1
                print(f"실패 {day} {type(exc).__name__}: {exc}", flush=True)
            if (done + failed) % 25 == 0:
                print(f"진행 {done + failed}/{len(days)} 실패 {failed}", flush=True)
    print(f"완료 {done} 실패 {failed}", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:4])
