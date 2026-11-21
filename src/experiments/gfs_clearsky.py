"""GFS 일사량 10열을 고정 영역 청천 지수로 교체한다."""
import numpy as np
import pandas as pd

from ..gfs_more import load_more
from ..gfs_pilot import LATS, LONS, region_masks
from ..gfs_regions import run_comparison
from ..gfs_v3 import checked_gfs
from ..solar import solar
from ..train import clear_sky


def region_clear_sky(stamps, mask):
    """공개 격자만으로 영역 평균 청천 일사 W/m² 계산. ts는 우측닫힘."""
    lat, lon = np.meshgrid(LATS, LONS, indexing="ij")
    la, lo = lat[mask], lon[mask]
    if not len(la):
        raise ValueError("빈 고정 영역")
    _, _, cosz = solar(stamps.repeat(len(la)), np.tile(la, len(stamps)), np.tile(lo, len(stamps)))
    flux = np.where(cosz > 0, 1000 * clear_sky(cosz), 0)
    return flux.reshape(len(stamps), len(la)).mean(axis=1)


def replace_radiation(base):
    """낮은 분모(<20 W/m²)는 결측. 나머지50열은 그대로 유지."""
    out = base.copy()
    for name, mask in region_masks().items():
        old = f"gfs_dswrf_{name}"
        denominator = region_clear_sky(base.index, mask)
        denominator = np.where(denominator >= 20, denominator, np.nan)
        out[old] = (out[old] / denominator).astype("float32")
    return out.rename(columns={f"gfs_dswrf_{n}": f"gfs_kt_{n}" for n in region_masks()})


def main():
    print("GFS 청천 지수 교체 검증, 2025-05 제외", flush=True)
    gfs, missing = checked_gfs()
    base = pd.concat([gfs, load_more()[0]], axis=1)
    candidate = replace_radiation(base)
    print(f"청천 지수 결측 {int(candidate.filter(like='gfs_kt_').isna().sum().sum())}건", flush=True)
    run_comparison(base, candidate, "clearsky10", "p38_gfs_clearsky", "14.30", len(missing))


if __name__ == "__main__":
    main()
