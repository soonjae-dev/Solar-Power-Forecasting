"""GEFS 평균20열로 GFS 일사·전운량20열을 교체. 행 출력/저장 없음."""
import json
import sys

import numpy as np
import pandas as pd

from ..config import TZ
from .gefs_mean import AUDIT_PATH, DIRECTORY, FIRST, DEVELOPMENT_LAST, cache_valid
from ..gfs_more import load_more
from ..gfs_pilot import FIRST_TS_HOUR, region_masks, summarize
from ..gfs_regions import run_comparison
from ..gfs_v3 import checked_gfs


def load_mean():
    audit = json.loads(AUDIT_PATH.read_text())
    if not audit.get("passed_internal_coverage_gate"):
        raise ValueError("GEFS 평균 공개 시각 감사 미통과")
    rows, missing = [], 0
    for target in pd.date_range(FIRST, DEVELOPMENT_LAST):
        day = target.date()
        if audit["days"].get(day.isoformat(), {}).get("status") != "eligible":
            missing += 1
            continue
        path = DIRECTORY / f"{target:%Y%m%d}.npz"
        if not cache_valid(path, day):
            raise ValueError("eligible 날짜의 GEFS 파일 누락/검증 실패")
        with np.load(path, allow_pickle=False) as z:
            frame = {f"gefs_mean_{key}_{name}": values
                     for key in ("dswrf", "tcdc") for name, values in summarize(z[key], region_masks()).items()}
            stamps = pd.DatetimeIndex([target + pd.Timedelta(hours=FIRST_TS_HOUR + i) for i in range(17)]).tz_localize(TZ)
            rows.append(pd.DataFrame(frame, index=stamps))
    if not rows:
        raise ValueError("GEFS 평균 입력 없음")
    common = pd.concat(rows).astype("float32").sort_index()
    if not common.index.is_unique or common.shape[1] != 20:
        raise ValueError("GEFS 평균 열/시각 조건 위반")
    return common, missing


def main(mode="both"):
    if mode == "both":
        main("replace")
        main("add")
        return
    if mode not in ("replace", "add"):
        raise ValueError("GEFS 비교 방식은 replace/add만 허용")
    mean, missing = load_mean()
    gfs, _ = checked_gfs()
    more, _ = load_more()
    base = pd.concat([gfs, more], axis=1)
    candidate = pd.concat([mean, more], axis=1) if mode == "replace" else pd.concat([base, mean], axis=1)
    print(f"GEFS 평균 {mode} 검증: 공통{candidate.shape[1]}열, GEFS 빠진 날{missing}, 2025-05 제외", flush=True)
    if mode == "replace":
        run_comparison(base, candidate, "gefs_mean20", "p43_gefs_mean_model", "14.32", missing)
    else:
        run_comparison(base, candidate, "gefs_mean_added20", "p44_gefs_mean_added", "14.33", missing)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "replace")
