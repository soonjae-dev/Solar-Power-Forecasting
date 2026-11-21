"""GFS 20열에 고정 상자 공간 패턴의 주성분 16열을 더해 재학습 검증한다. 집계만 저장한다.

    .venv/bin/python -m src.gfs_pca

판정 기준은 실행 전에 고정했다(docs/validation.md). PCA는 학습 마감 이전 값으로만 적합한다. 2025-05 holdout은 쓰지 않는다.
"""
import json

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from . import train as T
from .config import CACHE_DIR, EXP_DIR, TZ
from .data import load_generation, load_sites, load_smp
from .gfs_pilot import FIRST_TS_HOUR, GFS_DIR, region_masks
from .gfs_v3 import FIRST_DAY, LAST_DAY, checked_gfs
from .tune_v3 import END, WINDOWS, run_stage, save
from .weather_teacher import CFG

COMPONENTS = 8
FIT_HOURS = range(7, 19)
CANDIDATES = {"gfs20_pca16": {}}
SEASONS = ("summer_support", "autumn_support", "development", "selection")


def load_fields():
    """변수별 (시각, 상자 칸) 배열. 상자 평균을 뺀 공간 분포로 돌려준다."""
    box = region_masks()["all"]
    index, parts = [], dict(dswrf=[], tcdc=[])
    for day in pd.date_range(FIRST_DAY, LAST_DAY):
        path = GFS_DIR / f"{day:%Y%m%d}.npz"
        if not path.exists():
            continue
        z = np.load(path)
        for key, low, high in (("dswrf", 0, None), ("tcdc", 0, 100)):
            parts[key].append(np.clip(z[key], low, high)[:, box])
        index += [day + pd.Timedelta(hours=FIRST_TS_HOUR + i) for i in range(z["dswrf"].shape[0])]
    stamps = pd.DatetimeIndex(index).tz_localize(TZ)
    fields = {}
    for key, values in parts.items():
        field = np.concatenate(values).astype("float32")
        fields[key] = field - field.mean(axis=1, keepdims=True)
    return stamps, fields


def scores(field, stamps, cutoff, components=COMPONENTS):
    """cutoff 이전 낮 시각으로만 평균·표준편차·PCA를 적합하고 전체 시각을 변환한다."""
    fit = (stamps <= cutoff) & stamps.hour.isin(FIT_HOURS)
    past = field[fit]
    if len(past) < 365:
        raise ValueError("PCA 학습 이력 부족")
    mean, std = past.mean(axis=0), past.std(axis=0)
    std[std == 0] = 1
    model = PCA(n_components=components, svd_solver="full").fit((past - mean) / std)
    return model.transform((field - mean) / std).astype("float32"), model.explained_variance_ratio_.round(5).tolist()


def common_columns(gfs, stamps, fields, variance):
    def build(cutoff):
        frames = [gfs]
        for key, field in fields.items():
            values, ratio = scores(field, stamps, cutoff)
            variance[f"{cutoff:%Y-%m-%d}_{key}"] = ratio
            frames.append(pd.DataFrame(values, index=stamps, columns=[f"gfs_{key}_pc{i + 1}" for i in range(values.shape[1])]))
        return pd.concat(frames, axis=1)
    return build


def verdict(records):
    """gfs20 동일 1시드 대조군과 비교한다."""
    rows = [records[n]["vs_matched_v3"] for n in ("summer_support", "autumn_support", "development", "selection")]
    change = 100 * (sum(r["direct_penalty"] for r in rows) / sum(r["base_penalty"] for r in rows) - 1)
    changes = [r["change_pct"] for r in rows]
    static = records["fold1_static"]["vs_matched_v3"]["change_pct"]
    return dict(seasonal_change_pct=round(change, 3), seasons_improved=sum(c < 0 for c in changes), worst_season_pct=max(changes),
                static_change_pct=static, passed=bool(change <= -1 and sum(c < 0 for c in changes) >= 3 and max(changes) <= 3 and static <= 0))


def three_seed_verdict(cand, ctrl):
    """같은 v3 3시드 기준 행에서 낸 두 후보의 3시드 페널티를 비교한다. 형식은 14.17과 같다."""
    rows = {n: (ctrl[n]["vs_v3_3seeds"]["direct_penalty"], cand[n]["vs_v3_3seeds"]["direct_penalty"]) for n in WINDOWS}
    changes = {n: 100 * (b / a - 1) for n, (a, b) in rows.items()}
    seasonal = 100 * (sum(rows[n][1] for n in SEASONS) / sum(rows[n][0] for n in SEASONS) - 1)
    season_changes = [changes[n] for n in SEASONS]
    return dict(seasonal_change_pct=round(seasonal, 3), changes_pct={n: round(c, 3) for n, c in changes.items()},
                passed=bool(seasonal <= -1 and sum(c < 0 for c in season_changes) >= 3 and max(season_changes) <= 3
                            and changes["fold1_static"] <= 0))


def main():
    print("v3+GFS 20열+PCA 16열 재학습 검증, holdout 제외", flush=True)
    gfs, _ = checked_gfs()
    stamps, fields = load_fields()
    variance = {}
    sites = load_sites().set_index("site_id")
    smp = load_smp("train").set_index("ts")
    gen = load_generation()
    gen = gen[gen.ts <= END]
    d = T.load_train_features()
    cols = ["site_id", "ts", "energy", *T.select_features(d, CFG["feats"])]
    d = d.loc[d.ts <= END, cols].copy()
    T._cache.clear()
    out = EXP_DIR / "p25_gfs_pca"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "metrics.json"
    result = json.loads(path.read_text()) if path.exists() else dict(baseline_config=CFG, candidates=CANDIDATES, cache_name=CACHE_DIR.name, control="gfs20")
    build = common_columns(gfs, stamps, fields, variance)
    run_stage(d, gen, sites, smp, result, path, 1, list(CANDIDATES), variants=CANDIDATES, added=build, control_columns=list(gfs))
    result["pca_variance"] = variance
    result["verdict_vs_gfs20"] = verdict(result["seeds1"]["gfs20_pca16"])
    save(path, result)
    print("판정", json.dumps(result["verdict_vs_gfs20"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
