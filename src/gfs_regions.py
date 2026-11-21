"""고정 1도 영역으로 GFS 일사량·전운량 열을 교체한다. 집계만 저장한다."""
import json

import numpy as np
import pandas as pd

from . import train as T
from .config import CACHE_DIR, EXP_DIR, TZ
from .data import load_generation, load_sites, load_smp
from .gfs_more import load_more
from .gfs_pca import SEASONS, three_seed_verdict, verdict
from .gfs_pilot import FIRST_TS_HOUR, GFS_DIR, LATS, LONS, summarize
from .gfs_v3 import FIRST_DAY, LAST_DAY, checked_gfs
from .tune_v3 import END, run_stage, save
from .weather_teacher import CFG

def fine_masks():
    """기존 상자 전체와 겹치지 않는 36개 1도 칸. 발전소 정보 불사용."""
    lat, lon = np.meshgrid(LATS, LONS, indexing="ij")
    masks = {"all": (lat >= 33) & (lat < 39) & (lon >= 124) & (lon < 130)}
    for i, south in enumerate(range(33, 39)):
        for j, west in enumerate(range(124, 130)):
            masks[f"r{i}{j}"] = ((lat >= south) & (lat < south + 1)
                                     & (lon >= west) & (lon < west + 1))
    return masks


def load_fine(first=FIRST_DAY, last=LAST_DAY):
    masks = fine_masks()
    rows, missing = [], 0
    for day in pd.date_range(first, last):
        path = GFS_DIR / f"{day:%Y%m%d}.npz"
        if not path.exists():
            missing += 1
            continue
        with np.load(path) as z:
            index = pd.DatetimeIndex([day + pd.Timedelta(hours=FIRST_TS_HOUR + i)
                                      for i in range(z["dswrf"].shape[0])]).tz_localize(TZ)
            frame = {}
            for key, high in (("dswrf", None), ("tcdc", 100)):
                for name, values in summarize(np.clip(z[key], 0, high), masks).items():
                    frame[f"fine_{key}_{name}"] = values
            rows.append(pd.DataFrame(frame, index=index))
    if not rows:
        raise ValueError("GFS 1도 영역 입력 없음")
    result = pd.concat(rows).astype("float32").sort_index()
    if result.shape[1] != 74 or not result.index.is_unique:
        raise ValueError("GFS 1도 영역 열·시각 조건 위반")
    return result, missing


def main():
    print("고정 1도 GFS 영역 교체 검증, 2025-05 제외", flush=True)
    gfs, _ = checked_gfs()
    more, _ = load_more()
    base = pd.concat([gfs, more], axis=1)
    fine, missing = load_fine()
    if fine.index.max() > END or not fine.index.equals(gfs.index):
        raise ValueError("대조군과 후보 GFS 시각 불일치")
    candidate = pd.concat([fine, more], axis=1)
    run_comparison(base, candidate, "regions1deg", "p37_gfs_regions", "14.29", missing)


def early_failure(records):
    """남은 계절이 어떻게 나오더라도 기존 통과 조건을 만족할 수 없을 때만 기각."""
    if "fold1_static" in records and records["fold1_static"]["vs_matched_v3"]["change_pct"] > 0:
        return "fixed_11_months_worse"
    changes = [records[n]["vs_matched_v3"]["change_pct"] for n in SEASONS if n in records]
    if any(c > 3 for c in changes):
        return "season_above_3pct"
    if sum(c >= 0 for c in changes) >= 2:
        return "two_nonimproving_seasons"
    return None


def run_comparison(base, candidate, name, experiment, rule, missing=0):
    """교체 후보와 v6 열을 분리하고 같은 사전 기준으로 검증한다."""
    if not candidate.index.is_unique or candidate.index.max() > END:
        raise ValueError("후보 시각 조건 위반")
    common = pd.concat([base, candidate.loc[:, ~candidate.columns.isin(base.columns)]], axis=1)
    overlap = candidate.columns.intersection(base.columns)
    if not candidate[overlap].equals(base.reindex(candidate.index)[overlap]):
        raise ValueError("같은 이름의 공통 열 값 불일치")
    variants = {name: {}}
    signature = dict(baseline_config=CFG, candidates=variants, cache_name=CACHE_DIR.name,
                     control="gfs60", rule=rule, candidate_columns=list(candidate),
                     control_columns=list(base))
    out = EXP_DIR / experiment
    out.mkdir(parents=True, exist_ok=True)
    path = out / "metrics.json"
    result = json.loads(path.read_text()) if path.exists() else dict(signature)
    if any(result.get(k) != v for k, v in signature.items()):
        raise ValueError("체크포인트 설정 불일치")
    if result.get("verdict_vs_control", {}).get("early_rejected"):
        print("조기 기각된 후보, 재학습하지 않음", flush=True)
        return
    result.update(common_hours=len(candidate), missing_days=missing, common_feature_count=len(candidate.columns))
    save(path, result)
    print(f"공통 후보 {len(candidate.columns)}열, {len(candidate)}시각, 빠진 날 {missing}", flush=True)
    sites = load_sites().set_index("site_id")
    smp = load_smp("train").set_index("ts")
    gen = load_generation()
    gen = gen[gen.ts <= END]
    d = T.load_train_features()
    cols = ["site_id", "ts", "energy", *T.select_features(d, CFG["feats"])]
    d = d.loc[d.ts <= END, cols].copy()
    T._cache.clear()
    kwargs = dict(variants=variants, added=common, control_columns=list(base),
                  candidate_columns=list(candidate))
    run_stage(d, gen, sites, smp, result, path, 1, [name], early_reject=early_failure, **kwargs)
    prior = json.loads((EXP_DIR / "p26_gfs_more" / "metrics.json").read_text())
    matches = {n: r["vs_matched_v3"]["base_penalty"] ==
               prior["seeds1"]["gfs60"][n]["vs_v3_3seeds"]["direct_penalty"]
               for n, r in result["seeds1"][name].items()}
    result["matched_control_reproduces_p26"] = matches
    save(path, result)
    if not all(matches.values()):
        raise ValueError("동일1시드 대조군이 p26과 일치하지 않음")
    failure = result.get("seeds1_early_rejections", {}).get(name)
    if failure:
        result["verdict_vs_control"] = dict(passed=False, early_rejected=True, reason=failure,
            uncompleted_windows=[n for n in (*SEASONS, "fold1_static") if n not in result["seeds1"][name]])
        save(path, result)
        print("1시드 판정", json.dumps(result["verdict_vs_control"], ensure_ascii=False), flush=True)
        return
    result["verdict_vs_control"] = verdict(result["seeds1"][name])
    save(path, result)
    print("1시드 판정", json.dumps(result["verdict_vs_control"], ensure_ascii=False), flush=True)
    if result["verdict_vs_control"]["passed"]:
        run_stage(d, gen, sites, smp, result, path, 3, [name], **kwargs)
        ctrl = prior["seeds3"]["gfs60"]
        result["verdict3_vs_control"] = three_seed_verdict(result["seeds3"][name], ctrl)
        save(path, result)
        print("3시드 판정", json.dumps(result["verdict3_vs_control"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
