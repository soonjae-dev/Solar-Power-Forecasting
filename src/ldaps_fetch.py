"""기상청 LDAPS(UM 국지 1.5km) 하향 단파복사(GRIB 코드 4200)를 API허브에서 받아 고정 영역 평균으로 저장한다.

    KMA_AUTH_KEY=... .venv/bin/python -m src.ldaps_fetch 2024-03-08 2025-05-31 [작업자] [간격일] [step 목록]

대상일 D마다 D-1 00UTC 실행의 step(기본 19~35 = ts 04~20시 KST)을 한반도 상자(격자 i 137~505, j 49~489)만 이진으로 받는다.
인증키는 환경 변수로만 받으며 파일에 저장하지 않는다. 대회 데이터는 읽지 않는다.
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path
from urllib.request import urlopen

import numpy as np

from .config import DATA_ROOT

URL = ("https://apihub.kma.go.kr/api/typ06/cgi-bin/url/nph-um_grib_xy_txt1?group=UMKR&nwp=N128&data=U&varn=4200"
       "&map=S&sub=137,49,505,489&tmfc={tmfc}&hf={hf}&disp=B&authKey={key}")
OUT_DIR = DATA_ROOT / "external" / "ldaps_00utc"
GRID = DATA_ROOT / "external" / "ldaps_grid.npz"
AUDIT = Path(__file__).resolve().parents[1] / "experiments/p71_ldaps_coverage/metrics.json"


def masks():
    g = np.load(GRID)
    la, lo = g["lat"][48:489, 136:505], g["lon"][48:489, 136:505]
    out = {"all": (la >= 33) & (la < 39) & (lo >= 124) & (lo < 130)}
    for i, south in enumerate((33, 35, 37)):
        for j, west in enumerate((124, 126, 128)):
            out[f"r{i}{j}"] = (la >= south) & (la < south + 2) & (lo >= west) & (lo < west + 2)
    return out


MASKS = masks()


def field(tmfc, hf, key, retries=5):
    for attempt in range(retries):
        try:
            with urlopen(URL.format(tmfc=tmfc, hf=hf, key=key), timeout=120) as r:
                b = r.read()
            nx, ny = np.frombuffer(b[:8], "<i4")
            if (nx, ny) != (369, 441) or len(b) != 8 + 4 * nx * ny:
                raise ValueError("응답 형식 불일치")
            return np.frombuffer(b[8:], "<f4").reshape(ny, nx)
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def fetch_day(target, steps, key):
    path = OUT_DIR / f"{target:%Y%m%d}.npz"
    if path.exists():
        return "skip"
    tmfc = f"{target - timedelta(days=1):%Y%m%d}00"
    try:
        vals = [field(tmfc, s, key) for s in steps]
    except Exception:
        return "error"
    arr = np.stack([[v[m].mean() for m in MASKS.values()] for v in vals]).astype("float32")
    if not np.isfinite(arr).all() or arr.min() < -1:
        return "invalid"
    np.savez_compressed(path, dswrf=arr, steps=np.array(steps), regions=np.array(list(MASKS)))
    return "new"


def main(first, last, workers=8, every=1, steps="19-35"):
    key = os.environ["KMA_AUTH_KEY"]
    a, b = (int(x) for x in steps.split("-")) if "-" in steps else (None, None)
    step_list = list(range(a, b + 1)) if a is not None else [int(x) for x in steps.split(",")]
    start, stop = date.fromisoformat(first), date.fromisoformat(last)
    days = [start + timedelta(days=i) for i in range(0, (stop - start).days + 1, int(every))]
    counts = {}
    with ThreadPoolExecutor(int(workers)) as pool:
        for i, s in enumerate(pool.map(lambda d: fetch_day(d, step_list, key), days), 1):
            counts[s] = counts.get(s, 0) + 1
            if i % 10 == 0 or i == len(days):
                print(f"{i}/{len(days)} {counts}", flush=True)
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT.write_text(json.dumps(dict(first=first, last=last, every=int(every), steps=step_list, counts=counts), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:6])
