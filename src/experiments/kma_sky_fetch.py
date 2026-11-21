"""기상청 단기예보 격자 하늘상태(SKY)를 API허브에서 받아 고정 영역 평균으로 저장한다.

    KMA_AUTH_KEY=... .venv/bin/python -m src.experiments.kma_sky_fetch 2024-03-08 2025-05-31 [작업자] [간격일]

대상일 D마다 D-1 14시(KST) 발표분의 D 10·12·14시 SKY. 대회 데이터는 읽지 않는다. 인증키는 환경 변수로만 받는다.
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

from ..config import DATA_ROOT

URL = "https://apihub.kma.go.kr/api/typ01/cgi-bin/url/nph-dfs_shrt_grd?tmfc={tmfc}&tmef={tmef}&vars=SKY&authKey={key}"
OUT_DIR = DATA_ROOT / "external" / "kma_sky_14kst"
HOURS = (10, 12, 14)
NX, NY = 149, 253
AUDIT = Path(__file__).resolve().parents[2] / "experiments/p74_kma_sky_coverage/metrics.json"


def grid_latlon():
    """기상청 동네예보 격자(1-기반 x,y)의 위경도. 공식 Lambert 변환식."""
    re, grid, slat1, slat2, olon, olat, xo, yo = 6371.00877, 5.0, 30.0, 60.0, 126.0, 38.0, 43, 136
    d = np.pi / 180
    re /= grid
    s1, s2, olo, ola = slat1 * d, slat2 * d, olon * d, olat * d
    sn = np.log(np.cos(s1) / np.cos(s2)) / np.log(np.tan(np.pi / 4 + s2 / 2) / np.tan(np.pi / 4 + s1 / 2))
    sf = np.tan(np.pi / 4 + s1 / 2) ** sn * np.cos(s1) / sn
    ro = re * sf / np.tan(np.pi / 4 + ola / 2) ** sn
    x, y = np.meshgrid(np.arange(1, NX + 1), np.arange(1, NY + 1))
    xn, yn = x - xo, ro - (y - yo)
    ra = np.sqrt(xn ** 2 + yn ** 2) * np.sign(sn)
    alat = 2 * np.arctan((re * sf / ra) ** (1 / sn)) - np.pi / 2
    theta = np.where(np.abs(xn) <= 0, 0.0, np.arctan2(xn, yn))
    return alat / d, (theta / sn + olo) / d


def masks():
    la, lo = grid_latlon()
    out = {"all": (la >= 33) & (la < 39) & (lo >= 124) & (lo < 130)}
    for i, s in enumerate((33, 35, 37)):
        for j, w in enumerate((124, 126, 128)):
            out[f"r{i}{j}"] = (la >= s) & (la < s + 2) & (lo >= w) & (lo < w + 2)
    return out


MASKS = masks()


def field(tmfc, tmef, key):
    for attempt in range(5):
        try:
            with urlopen(URL.format(tmfc=tmfc, tmef=tmef, key=key), timeout=120) as r:
                text = r.read().decode("ascii", "ignore")
            v = np.array([float(t) for t in text.replace(",", " ").split()], dtype="float32")
            if v.size != NX * NY:
                raise ValueError(f"값 개수 {v.size}")
            v = v.reshape(NY, NX)
            return np.where(v < 0, np.nan, v)
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)


def fetch_day(target, key):
    path = OUT_DIR / f"{target:%Y%m%d}.npz"
    if path.exists():
        return "skip"
    tmfc = f"{target - timedelta(days=1):%Y%m%d}14"
    try:
        vals = [field(tmfc, f"{target:%Y%m%d}{h:02d}", key) for h in HOURS]
    except Exception:
        return "error"
    arr = np.array([[np.nanmean(v[m]) for m in MASKS.values()] for v in vals], dtype="float32")
    np.savez_compressed(path, sky=arr, hours=np.array(HOURS), regions=np.array(list(MASKS)))
    return "new"


def main(first, last, workers=3, every=1):
    key = os.environ["KMA_AUTH_KEY"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    a, b = date.fromisoformat(first), date.fromisoformat(last)
    days = [a + timedelta(days=i) for i in range(0, (b - a).days + 1, int(every))]
    counts = {}
    with ThreadPoolExecutor(int(workers)) as pool:
        for i, s in enumerate(pool.map(lambda d: fetch_day(d, key), days), 1):
            counts[s] = counts.get(s, 0) + 1
            if i % 10 == 0 or i == len(days):
                print(f"{i}/{len(days)} {counts}", flush=True)
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT.write_text(json.dumps(dict(first=first, last=last, every=int(every), counts=counts), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:5])
