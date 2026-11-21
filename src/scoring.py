"""README 산식에 따른 잠정 채점기. 공식 scoring.py를 확보하면 대조한다.

    S = Σ SMP·실적 − 0.10 × Σ SMP·k·|예측 − 실적|      (집합별로 절대값)

    python -m src.scoring --selftest
"""
import argparse

import numpy as np
import pandas as pd

from .config import MIN_GROUP_KW, PENALTY_RATE


def score_groups(group_frame: pd.DataFrame, smp: pd.DataFrame) -> dict:
    """group_frame: group, ts, actual_kwh, forecast_kwh. smp: ts, smp, k."""
    d = group_frame.merge(smp[["ts", "smp", "k"]], on="ts", how="left", validate="many_to_one")
    if d["smp"].isna().any():
        raise ValueError("SMP가 없는 시각이 있다")
    werr = d["smp"] * d["k"] * (d["forecast_kwh"] - d["actual_kwh"]).abs()
    sales = float((d["smp"] * d["actual_kwh"]).sum())
    penalty = PENALTY_RATE * float(werr.sum())
    return dict(sales=sales, penalty=penalty, S=sales - penalty,
                penalty_pct=100 * penalty / sales if sales else float("nan"),
                penalty_by_group=(PENALTY_RATE * werr.groupby(d["group"]).sum()).to_dict())


def to_groups(site_pred: pd.DataFrame, assign: pd.Series) -> pd.DataFrame:
    """site_pred: site_id, ts, energy, forecast_kwh. assign: site_id -> 'G1'/'G2'."""
    d = site_pred.assign(group=site_pred["site_id"].map(assign))
    if d["group"].isna().any():
        raise ValueError("배정되지 않은 발전소가 있다")
    return (d.groupby(["group", "ts"], as_index=False)
             .agg(actual_kwh=("energy", "sum"), forecast_kwh=("forecast_kwh", "sum")))


def check_capacity(assign: pd.Series, sites: pd.DataFrame) -> pd.Series:
    cap = sites.set_index("site_id")["capacity_kw"].groupby(assign).sum()
    if set(cap.index) != {"G1", "G2"} or (cap < MIN_GROUP_KW).any():
        raise ValueError(f"집합 용량 제약 위반: {cap.to_dict()}")
    return cap


def residual_matrix(site_pred: pd.DataFrame) -> pd.DataFrame:
    """행 ts, 열 site_id, 값 예측−실적. 결측이 있으면 거부한다(0으로 채우지 않는다)."""
    e = (site_pred.assign(e=site_pred["forecast_kwh"] - site_pred["energy"])
                  .pivot(index="ts", columns="site_id", values="e"))
    if e.isna().any().any():
        raise ValueError("잔차 행렬에 결측이 있다. 모든 발전소가 관측된 구간만 써야 한다")
    return e


def composition_penalty(E: np.ndarray, w: np.ndarray, in_g1: np.ndarray) -> float:
    """발전소별 예측을 고정했을 때 배정 in_g1(bool, 발전소 수)의 페널티(원)."""
    e1 = E[:, in_g1].sum(axis=1)
    e2 = E[:, ~in_g1].sum(axis=1)
    return PENALTY_RATE * float(w @ (np.abs(e1) + np.abs(e2)))


def lower_bound(E: np.ndarray, w: np.ndarray) -> float:
    """어떤 배정으로도 내려갈 수 없는 페널티: |E1|+|E2| >= |E1+E2|."""
    return PENALTY_RATE * float(w @ np.abs(E.sum(axis=1)))


def _selftest() -> None:
    ts = pd.date_range("2025-01-01 10:00", periods=3, freq="h", tz="Asia/Seoul")   # 10, 11, 12시
    smp = pd.DataFrame({"ts": ts, "smp": [100.0, 100.0, 0.0], "k": [1, 3, 3]})

    def frame(f1, f2, a1=(10, 10, 10), a2=(20, 20, 20)):
        return pd.DataFrame({"group": ["G1"] * 3 + ["G2"] * 3, "ts": list(ts) * 2,
                             "actual_kwh": list(a1) + list(a2), "forecast_kwh": list(f1) + list(f2)})

    sales = 100 * 30 + 100 * 30
    r = score_groups(frame((10, 10, 10), (20, 20, 20)), smp)
    assert r["penalty"] == 0 and r["S"] == sales, "완전 예측"
    r = score_groups(frame((11, 10, 10), (20, 20, 20)), smp)
    assert np.isclose(r["penalty"], 0.1 * 100 * 1 * 1), "k=1"
    r = score_groups(frame((10, 11, 10), (20, 20, 20)), smp)
    assert np.isclose(r["penalty"], 0.1 * 100 * 3 * 1), "k=3"
    r = score_groups(frame((10, 10, 99), (20, 20, 20)), smp)
    assert r["penalty"] == 0, "SMP=0"
    r = score_groups(frame((12, 10, 10), (18, 20, 20)), smp)
    assert np.isclose(r["penalty"], 0.1 * 100 * 4), "집합 간 반대 오차는 상쇄되지 않는다"

    # 집합 내부 상쇄와 하한
    E = np.array([[5.0, -5.0, 2.0, 1.0]])
    w = np.array([10.0])
    assert np.isclose(composition_penalty(E, w, np.array([True, True, False, False])), 0.1 * 10 * 3)
    assert np.isclose(composition_penalty(E, w, np.array([True, False, True, False])), 0.1 * 10 * 11)
    assert np.isclose(lower_bound(E, w), 0.1 * 10 * 3)

    sites = pd.DataFrame({"site_id": ["a", "b"], "capacity_kw": [25_000.0, 19_999.0]})
    try:
        check_capacity(pd.Series({"a": "G1", "b": "G2"}), sites)
        raise AssertionError("용량 미달을 잡지 못했다")
    except ValueError:
        pass
    print("scoring selftest 통과")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest:
        _selftest()
