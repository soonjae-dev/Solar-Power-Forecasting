"""v6에 공개 sflux 광학 정보20열을 추가한다. 집계만 저장한다."""
import json

import numpy as np
import pandas as pd

from ..config import TZ
from ..gfs_more import load_more
from .gfs_radiation import AUDIT_PATH, COLUMNS, DIRECTORY, FIRST, DEVELOPMENT_LAST, cache_valid
from ..gfs_regions import run_comparison
from ..gfs_v3 import checked_gfs


def load_optics():
    audit = json.loads(AUDIT_PATH.read_text())
    if not audit.get("passed_internal_coverage_gate"):
        raise ValueError("sflux 공개 시각 감사 미통과")
    rows, missing = [], 0
    for stamp in pd.date_range(FIRST, DEVELOPMENT_LAST):
        target = stamp.date()
        if audit["days"].get(target.isoformat(), {}).get("status") != "eligible":
            missing += 1
            continue
        path = DIRECTORY / f"{target:%Y%m%d}.npz"
        if not cache_valid(path, target):
            raise ValueError("eligible sflux 파일 누락/검증 실패")
        with np.load(path, allow_pickle=False) as z:
            stamps = pd.DatetimeIndex([stamp + pd.Timedelta(hours=i) for i in range(4, 21)]).tz_localize(TZ)
            rows.append(pd.DataFrame(z["features"], index=stamps, columns=COLUMNS))
    if not rows:
        raise ValueError("sflux 입력 없음")
    return pd.concat(rows).astype("float32").sort_index(), missing


def main():
    optics, missing = load_optics()
    base = pd.concat([checked_gfs()[0], load_more()[0]], axis=1)
    candidate = pd.concat([base, optics], axis=1)
    print(f"sflux 광학20열 추가 검증, 공통{candidate.shape[1]}열, 빠진 날{missing}, 2025-05 제외", flush=True)
    run_comparison(base, candidate, "sflux_optics20", "p48_gfs_radiation_model", "14.35", missing)


if __name__ == "__main__":
    main()
