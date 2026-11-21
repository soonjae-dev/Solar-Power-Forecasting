"""입력 시점 감사. 대상일 전일 17시 이후에 발행된 정보가 특징·학습에 섞이지 않는지 집계 건수로 확인한다.

    python -m src.audit_inputs

행은 출력하지 않는다. 건수·시각 범위·열 이름만 낸다.
"""
import pandas as pd
import pyarrow.parquet as pq

from .config import CACHE_DIR, FIRST_TEST_CUTOFF, TEST_DIR, TRAIN_DIR, TZ
from .data import _kst, load_smp
from .features import EXTRA
from .splits import _FOLDS, fold


def target_day(ts: pd.Series) -> pd.Series:
    """우측닫힘: 00시는 전날 대상일에 속한다."""
    return (ts - pd.Timedelta(hours=1)).dt.normalize()


def audit_forecast_keys() -> None:
    print("[1] 예보 키 구조 — (관측소, round, ts)가 유일하고 대상일마다 한 번 발행되는가")
    for provider in ("wa", "wb"):
        st = f"{provider}_station"
        for split, folder in (("train", TRAIN_DIR), ("test", TEST_DIR)):
            f = pd.read_parquet(folder / f"forecast_{provider}.parquet", columns=[st, "round", "ts"])
            f["ts"] = _kst(f["ts"])
            off_hour = int((f["ts"] != f["ts"].dt.floor("h")).sum())
            f = f[f["ts"] == f["ts"].dt.floor("h")]
            dup = int(f.duplicated([st, "round", "ts"]).sum())
            per_day = f.groupby([st, "round", target_day(f["ts"])], observed=True).size()
            full = float((per_day == 24).mean())
            print(f"  {provider} {split:<5} 행 {len(f):>9,}  round {sorted(f['round'].unique().tolist())}  "
                  f"정각 아닌 행 {off_hour}  키 중복 {dup}  (관측소·round·대상일) 묶음 {len(per_day):,}개 중 24시간 완전 {full:.2%}, "
                  f"묶음 크기 {int(per_day.min())}~{int(per_day.max())}")


def audit_feature_rows() -> None:
    print("[2] 특징 행의 시각 — 앞뒤 1시간 특징이 대상일 경계를 넘을 수 있는 행이 있는가")
    print("    _next(ts+1h)는 ts가 00시일 때만 다음 대상일 예보를 본다. _prev(ts-1h)는 01시일 때 전날 예보를 보며 이는 과거 정보다.")
    for name in ("train", "test"):
        ts = pd.read_parquet(CACHE_DIR / f"{name}_features.parquet", columns=["ts"])["ts"]
        hour = ts.dt.hour
        print(f"  {name:<5} 행 {len(ts):>9,}  ts {ts.min()} ~ {ts.max()}  시 범위 {int(hour.min())}~{int(hour.max())}  "
              f"00시 행 {int((hour == 0).sum())}  01시 행 {int((hour == 1).sum())}  23시 행 {int((hour == 23).sum())}")


def audit_feature_names() -> None:
    print("[3] 특징 이름 — 발전 실적·실측 기상에서 온 열이 없는가")
    names = [c for c in pq.read_schema(CACHE_DIR / "test_features.parquet").names if c not in ("site_id", "ts", "energy")]
    print(f"  (기본 특징 {len([c for c in names if c not in EXTRA])}개 + 구름 양상 후보 {len([c for c in names if c in EXTRA])}개)")
    obs_only = {"wb_feel_temp_shade", "wb_pressure", "wb_appr_temp", "wb_wind_chill_temp", "wb_precip_1h"}
    suspicious = [c for c in names if "energy" in c or "obs" in c or "lag" in c or c in obs_only]
    print(f"  특징 {len(names)}개, 의심 이름 {suspicious or '없음'}")
    print("  " + ", ".join(names))
    te = pd.read_parquet(CACHE_DIR / "test_features.parquet", columns=names)
    print(f"  평가 특징 결측 셀 {int(te.isna().sum().sum())}")


def audit_folds() -> None:
    print("[4] 검증 구간 — 학습 마감이 첫 대상일 전일 17시인가")
    ts = pd.read_parquet(CACHE_DIR / "train_features.parquet", columns=["ts"])["ts"]
    for name in _FOLDS:
        f = fold(name)
        train_max = ts[ts <= f["cutoff"]].max()
        gap = int(((ts > f["cutoff"]) & (ts < f["val_start"])).sum())
        print(f"  {name:<15} 마감 {f['cutoff']}  학습 마지막 행 {train_max}  검증 {f['val_start']} ~ {f['val_end']}  "
              f"마감과 검증 사이에 버려지는 행 {gap:,}")


def audit_submission_training() -> None:
    print("[5] 제출용 재학습 — 평가 첫 대상일의 예측시점(2025-05-31 17:00) 뒤의 실적이 학습에 들어가는가")
    ts = pd.read_parquet(CACHE_DIR / "train_features.parquet", columns=["ts"])["ts"]
    late = ts > pd.Timestamp(FIRST_TEST_CUTOFF, tz=TZ)
    print(f"  학습 특징 행 {len(ts):,} 중 예측시점 뒤 {int(late.sum()):,}행 ({late.mean():.4%}), "
          f"해당 시각 {sorted(ts[late].dt.strftime('%m-%d %H시').unique().tolist())}")
    smp = load_smp("test")["ts"]
    print(f"  평가 시각 {len(smp):,}개  {smp.min()} ~ {smp.max()}")


if __name__ == "__main__":
    for step in (audit_forecast_keys, audit_feature_rows, audit_feature_names, audit_folds, audit_submission_training):
        step()
        print()
