"""제출 파일 검사 (README 기준. 무효 조건의 정본은 문제정의서).

자체 검사 뒤 주최 측 공식 형식 검증기(작업 폴더 최상위 validate_submission.py)도 돌린다.

    python -m src.validate_submission ../OIBC_2026_data/submissions/NAME/composition.csv ../OIBC_2026_data/submissions/NAME/forecast.csv
"""
import subprocess
import sys

import numpy as np
import pandas as pd

from .config import DATA_DIR, MIN_GROUP_KW, ROOT, N_SITES, N_TEST_HOURS, TZ
from .data import load_sites, load_smp


def validate(composition_path, forecast_path) -> None:
    sites = load_sites()
    comp = pd.read_csv(composition_path)
    assert list(comp.columns) == ["site_id", "group"], f"composition 열: {list(comp.columns)}"
    assert len(comp) == N_SITES and comp["site_id"].is_unique, "composition 행 수·중복"
    assert set(comp["site_id"]) == set(sites["site_id"]), "composition의 발전소 집합"
    assert set(comp["group"]) == {"G1", "G2"}, "group 값"
    cap = comp.merge(sites[["site_id", "capacity_kw"]], on="site_id").groupby("group")["capacity_kw"].sum()
    assert (cap >= MIN_GROUP_KW).all(), f"집합 용량 미달: {cap.to_dict()}"

    raw = pd.read_csv(forecast_path, dtype={"ts": str})
    assert list(raw.columns) == ["group", "ts", "forecast_kwh"], f"forecast 열: {list(raw.columns)}"
    assert len(raw) == 2 * N_TEST_HOURS, f"forecast 행 수 {len(raw)}"
    assert raw["ts"].str.endswith("+09:00").all(), "ts에 타임존(+09:00)이 있어야 한다"
    ts = pd.to_datetime(raw["ts"]).dt.tz_convert(TZ)
    assert not raw.assign(ts=ts).duplicated(["group", "ts"]).any(), "forecast 키 중복"
    expected = set(load_smp("test")["ts"])
    for g in ("G1", "G2"):
        assert set(ts[raw["group"] == g]) == expected, f"{g}의 시각이 smp.csv와 다르다"
    v = raw["forecast_kwh"].to_numpy(dtype=float)
    assert np.isfinite(v).all() and (v >= 0).all(), "forecast_kwh는 유한한 0 이상"
    print(f"제출 검사 통과  용량 {cap.round(1).to_dict()}  "
          f"연간 예측 합 {raw.groupby('group')['forecast_kwh'].sum().div(1e6).round(3).to_dict()} GWh")
    official(composition_path, forecast_path)


def official(composition_path, forecast_path) -> None:
    """주최 측 검증기. 종료 코드 0 = 통과, 1 = 무효, 2 = 읽기 실패."""
    r = subprocess.run([sys.executable, str(ROOT / "validate_submission.py"),
                        "--composition", str(composition_path), "--forecast", str(forecast_path),
                        "--data", str(DATA_DIR)], capture_output=True, text=True)
    print(r.stdout, end="")
    assert r.returncode == 0, f"공식 검증기 실패 (종료 코드 {r.returncode}){r.stderr and chr(10) + r.stderr}"


if __name__ == "__main__":
    validate(sys.argv[1], sys.argv[2])
