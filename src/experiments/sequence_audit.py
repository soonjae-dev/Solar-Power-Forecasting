"""하루 OOF 구조/메모리 감사. 예측 점수 계산·행 출력·학습 없음."""
import importlib.util
import json
import platform
import traceback

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .. import train as T
from ..config import CACHE_DIR, EXP_DIR, PRED_DIR, TRAIN_DIR
from ..data import _kst, load_generation, load_sites
from ..diagnose_season import CONFIGS, run_id
from ..gfs_more import load_more
from ..gfs_v3 import checked_gfs
from ..splits import fold
from ..tune_v3 import END, save
from ..weather_teacher import CFG
from ..solar import solar

HISTORY = [f"oof6m_{i}" for i in range(1, 6)]
PATH = EXP_DIR/"p56_sequence_audit"/"metrics.json"


def day_slots(ts):
    return (ts-pd.Timedelta(hours=1)).dt.normalize(), ((ts.dt.hour-1) % 24).to_numpy()


def pack_day(values, slots, targets, observed):
    """합성/실제 공용 배치 구성 원리. 정답mask는입력과독립,NaN정답0변환금지."""
    slots = np.asarray(slots)
    if len(np.unique(slots)) != len(slots) or (slots < 0).any() or (slots > 23).any():
        raise ValueError("시퀀스 슬롯 중복/범위 위반")
    values = np.asarray(values, dtype="float32")
    x = np.zeros((24, values.shape[1]), dtype="float32")
    m = np.zeros_like(x)
    good = np.isfinite(values)
    x[slots] = np.where(good, values, 0)
    m[slots] = good
    y = np.full(24, np.nan, dtype="float32")
    target_mask = np.zeros(24, dtype=bool)
    observed = np.asarray(observed, dtype=bool)
    y[slots[observed]] = np.asarray(targets)[observed]
    target_mask[slots] = observed
    return x, m, y, target_mask


def coverage(panel):
    day, slots = day_slots(panel.ts)
    p = panel.assign(day=day, slot=slots)
    if p.duplicated(["site_id", "day", "slot"]).any():
        raise ValueError("중복 일중 시각")
    days = p.assign(nonzero=p.energy.ne(0)).groupby(["site_id", "day"], observed=True).agg(
        size=("energy", "size"), nonzero=("nonzero", "sum"))
    return dict(rows=len(p), site_count=int(p.site_id.nunique()), days=int(day.nunique()),
                site_days=len(days), complete_site_days=int((days["size"] == 24).sum()),
                partial_site_days=int((days["size"] != 24).sum()),
                complete_zero_site_days=int(((days["size"] == 24) & (days["nonzero"] == 0)).sum()),
                negative_observed_rows=int((p.energy < 0).sum()))


def environment():
    result = dict(machine=platform.machine(), torch_installed=importlib.util.find_spec("torch") is not None,
                  neural_training_benchmark_run=False)
    if result["torch_installed"]:
        import torch
        result.update(torch_version=torch.__version__, mps_available=torch.backends.mps.is_available())
    return result


def partial_context(panel, sites):
    """부분 관측일의 누락 슬롯이 주간 입력인지, 정답값을 쓰지 않고 확인."""
    day, slots = day_slots(panel.ts)
    p = panel.assign(day=day, slot=slots)
    sizes = p.groupby(["site_id", "day"], observed=True).size()
    partial = sizes[sizes != 24].index.to_frame(index=False)
    if partial.empty:
        return dict(missing_slots=0, missing_daylight_slots=0)
    grid = partial.merge(pd.DataFrame({"slot": np.arange(24)}), how="cross")
    grid["ts"] = grid.day+pd.to_timedelta(grid.slot+1, unit="h")
    keys = pd.MultiIndex.from_frame(p[["site_id", "day", "slot"]])
    missing = ~pd.MultiIndex.from_frame(grid[["site_id", "day", "slot"]]).isin(keys)
    grid = grid.loc[missing]
    elev, _, _ = solar(pd.DatetimeIndex(grid.ts), grid.site_id.map(sites.lat).to_numpy(),
                       grid.site_id.map(sites.lon).to_numpy())
    return dict(missing_slots=len(grid), missing_daylight_slots=int((elev > 0).sum()))


def main():
    PATH.parent.mkdir(parents=True, exist_ok=True)
    print("일단위 OOF 적합성 감사: 점수/모델학습 없음, 2025-05 제외", flush=True)
    gen = load_generation()
    sites = load_sites().set_index("site_id")
    gen = gen.loc[(gen.ts <= END) & (gen.ts >= fold(HISTORY[0])["val_start"]), ["site_id", "ts", "energy"]]
    result = dict(rule="14.44", environment=environment(), blocks={}, forecast_keys={}, scored=False,
                  input_policy=dict(target_mask_is_input=False, retain_zero_days=True,
                                    nbrc_recompute_per_original_fold=True, night_padding="zero_with_input_mask"))
    names = pq.read_schema(CACHE_DIR/"train_features.parquet").names
    features = T.select_features(pd.DataFrame(columns=names), CFG["feats"])
    cached = pd.read_parquet(CACHE_DIR/"train_features.parquet", columns=["site_id", "ts", *features])
    cached = cached.loc[cached.ts <= END]
    if len(features) != 84 or "site" not in features:
        raise ValueError("기존 특징 수 불일치")
    border = cached.ts.dt.hour.isin([0, 1, 23])
    next_day = (cached.ts+pd.Timedelta(hours=1)-pd.Timedelta(hours=1)).dt.normalize()
    day = (cached.ts-pd.Timedelta(hours=1)).dt.normalize()
    result["feature_provenance"] = dict(base_features=84, proposed_without_site=83,
        cached_boundary_rows=int(border.sum()), next_feature_cross_day_rows=int((next_day != day).sum()),
        source_round_contract=[10, 17], actual_publication_timestamp_verified=False,
        note="제공round계약/특징코드감사이며실제배포timestamp재인증아님")
    if (next_day != day).any():
        raise ValueError("다음 대상일 예보 경계")
    gfs = pd.concat([checked_gfs()[0], load_more()[0]], axis=1)
    result["gfs"] = dict(columns=len(gfs.columns), unique_hours=gfs.index.is_unique,
                         rows=len(gfs), missing_cells=int(gfs.isna().sum().sum()))
    if len(gfs.columns) != 60:
        raise ValueError("GFS60 입력 불일치")
    seen = set()
    total_site_days, total_rows = 0, 0
    for name in HISTORY:
        f = fold(name)
        path = PRED_DIR/run_id("gfs60seed", name)/"predictions.parquet"
        columns = pq.read_schema(path).names
        seed_columns = [f"forecast_kwh_s{i}" for i in range(3)]
        if not set(seed_columns).issubset(columns):
            raise ValueError("3시드 예측 열 없음")
        pred = pd.read_parquet(path, columns=["site_id", "ts", "forecast_kwh"])
        if pred.duplicated(["site_id", "ts"]).any() or not pred.ts.between(f["val_start"], f["val_end"]).all():
            raise ValueError("OOF 키/구간 위반")
        if not (pred.ts > f["cutoff"]).all() or f["val_end"] > END:
            raise ValueError("OOF 마감/holdout 위반")
        if not np.isfinite(pred.forecast_kwh).all() or (pred.forecast_kwh < 0).any():
            raise ValueError("OOF 예측 값 조건 위반")
        # 집계 점수 없이 키와 관측지원만 검사한다.
        actual = gen.loc[gen.ts.between(f["val_start"], f["val_end"])]
        p = pred.merge(actual, on=["site_id", "ts"], how="left", validate="one_to_one")
        if p.energy.isna().any():
            raise ValueError("원본 대응 실적 누락")
        days, _ = day_slots(p.ts)
        keys = set(zip(p.site_id, days))
        if seen.intersection(keys):
            raise ValueError("OOF 블록간 일 중복")
        seen.update(keys)
        cf = cached.loc[cached.ts.between(f["val_start"], f["val_end"])]
        joined = p[["site_id", "ts", "forecast_kwh"]].merge(cf[["site_id", "ts", "cosz"]],
                            on=["site_id", "ts"], how="left", validate="one_to_one")
        meta = coverage(p)
        meta["partial_day_context"] = partial_context(p, sites)
        meta.update(three_seed_columns_present=True, keys_unique=True, cutoff_before_all_targets=True,
                    missing_prediction_rows=int(len(actual)-len(p)),
                    daylight_feature_rows=int(joined.cosz.notna().sum()),
                    nonzero_prediction_without_features=int(((joined.forecast_kwh != 0) & joined.cosz.isna()).sum()),
                    complete_gfs_daylight_rows=int(gfs.reindex(joined.loc[joined.cosz.notna(), "ts"]).notna().all(axis=1).sum()))
        result["blocks"][name] = meta
        total_site_days += meta["site_days"]
        total_rows += meta["rows"]
        print(f"{name}: {meta['site_days']}발전소일, 완전{meta['complete_site_days']}, 부분{meta['partial_site_days']}", flush=True)
    for provider in ("wa", "wb"):
        st = f"{provider}_station"
        f = pd.read_parquet(TRAIN_DIR/f"forecast_{provider}.parquet", columns=[st, "round", "ts"])
        f["ts"] = _kst(f.ts)
        f = f.loc[f.ts.between(fold(HISTORY[0])["val_start"], fold(HISTORY[-1])["val_end"])]
        off = f.ts != f.ts.dt.floor("h")
        off_count = int(off.sum())
        f = f.loc[~off]
        rounds = sorted(int(v) for v in f["round"].unique())
        day, _ = day_slots(f.ts)
        sizes = f.groupby([st, "round", day], observed=True).size()
        result["forecast_keys"][provider] = dict(rounds=rounds, rows=len(f), off_hour_rows=off_count,
            duplicate_keys=int(f.duplicated([st, "round", "ts"]).sum()), station_round_days=len(sizes),
            complete_24hour_groups=int((sizes == 24).sum()), incomplete_groups=int((sizes != 24).sum()))
        if rounds != [10, 17] or result["forecast_keys"][provider]["duplicate_keys"]:
            raise ValueError("제공예보 round/키 계약 위반")
    channels = 2*(83+60)+2  # 숫자 입력과 결측mask, v6 kt, daylight flag
    result["memory_estimate"] = dict(site_days=total_site_days, observed_rows=total_rows,
        illustrative_channels=channels, full_dense_sequence_input_mib=round(total_site_days*24*channels*4/2**20, 1),
        batch64_sequence_input_mib=round(64*24*channels*4/2**20, 2),
        shared_gfs_float32_mib=round(gfs.shape[0]*gfs.shape[1]*4/2**20, 2),
        note="입력배열산술값,DataFrame/optimizer/activation/백엔드메모리포함안함")
    result["complete"] = True
    result["can_define_training_plan"] = all(v["nonzero_prediction_without_features"] == 0 and
        v["partial_day_context"]["missing_daylight_slots"] == 0 for v in result["blocks"].values())
    result["can_run_neural_training"] = result["can_define_training_plan"] and result["environment"]["torch_installed"]
    result["reference_config"] = CONFIGS["gfs60seed"]
    save(PATH, result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        frames = [dict(function=f.name, line=f.lineno) for f in traceback.extract_tb(exc.__traceback__)
                  if "/src/" in f.filename]
        print(json.dumps(dict(status="failed", error_type=type(exc).__name__, code_frames=frames)), flush=True)
        raise SystemExit(1) from None
