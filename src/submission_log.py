"""제출 기록. 제출물의 식별값·설정·기대 점수·서버 점수를 한 줄로 묶는다. 집계값만 쓰고 출력한다.

    python -m src.submission_log register NAME [command="..."] [note="..."]
    python -m src.submission_log result NAME VALUE [kind=normalized|penalty|S|pct|unknown] [date=YYYY-MM-DD] [note="..."]
    python -m src.submission_log show

register는 제출 전에, result는 사이트에 올린 뒤 서버 점수를 받아 적을 때 쓴다(1회 제출 = result 1건).
기대 점수는 검증 구간의 월별 페널티/매출 비율(experiments/p1_season)을 평가 구간의 예측 매출에 곱한 값이다.
서버 산식이 로컬 채점기와 같은지 가늠하는 용도이고 성능 추정치가 아니다.
서버가 정규화 점수(100 × (1 − W팀/W기준선), 설명회 슬라이드 20)를 보여 주면 kind=normalized로 적는다.
그 기대 범위는 배포 baseline.py를 검증 구간에서 재현한 결과(experiments/p4_official_baseline)에서 읽는다.
하루 횟수는 KST 날짜로 센다(슬라이드 19: KST 00:00 초기화).
"""
import hashlib
import json
import sys
import tarfile
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from .config import EXP_DIR, ROOT, SUB_DIR, TZ
from .data import load_sites, load_smp
from .validate_submission import validate

LOG_DIR = EXP_DIR / "submission_log"
LOG = LOG_DIR / "metrics.json"
BUDGET = dict(total=20, per_day=3)          # 2차 설명회: 총 20회, 하루 3회, 최고점 유지
SEASON = EXP_DIR / "p1_season" / "metrics.json"
NORMALIZED = EXP_DIR / "p4_official_baseline" / "metrics.json"
FILL = {12: (11, 1), 5: (4, 6)}             # 검증 구간이 없는 달은 앞뒤 달 평균(5월은 미개봉 holdout)


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code_sha() -> str:
    h = hashlib.sha256()
    for p in sorted((ROOT / "src").glob("*.py")):
        h.update(p.name.encode() + b"\0" + p.read_bytes())
    return h.hexdigest()


def _snapshot(sha: str) -> str:
    """git이 없으므로 제출 시점의 src를 묶어 둔다. 같은 코드면 다시 만들지 않는다."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    out = LOG_DIR / f"src_{sha[:12]}.tar.gz"
    if not out.exists():
        with tarfile.open(out, "w:gz") as tar:
            for p in sorted((ROOT / "src").glob("*.py")):
                tar.add(p, arcname=f"src/{p.name}")
    return out.name


def _load() -> dict:
    return json.loads(LOG.read_text(encoding="utf-8")) if LOG.exists() else dict(budget=BUDGET, entries=[])


def _save(log: dict) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    LOG.write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")


def month_ratio() -> dict:
    """월(1~12) -> 채택 설정의 페널티/매출. 검증 구간이 없는 달은 FILL로 메운다."""
    m = json.loads(SEASON.read_text(encoding="utf-8"))
    r = {int(k[-2:]): v["adopted"]["pen_pct"] / 100 for f in m.values() for k, v in f["month"].items()}
    for mo, (a, b) in FILL.items():
        r[mo] = (r[a] + r[b]) / 2
    return r


def _read_forecast(folder) -> pd.DataFrame:
    fc = pd.read_csv(folder / "forecast.csv", dtype={"ts": str})
    fc["ts"] = pd.to_datetime(fc["ts"]).dt.tz_convert(TZ)
    return fc


def expected(fc: pd.DataFrame) -> dict:
    d = fc.merge(load_smp("test")[["ts", "smp"]], on="ts", validate="many_to_one")
    month = (d["ts"] - pd.Timedelta(hours=1)).dt.month     # 우측닫힘: 00시는 전날에 속한다
    sales = (d["smp"] * d["forecast_kwh"]).groupby(month).sum()
    r = pd.Series(month_ratio())
    seasons = [f["total"]["adopted"]["pen_pct"] / 100 for f in json.loads(SEASON.read_text(encoding="utf-8")).values()]
    total = float(sales.sum())
    pen = float((sales * r).sum())
    return dict(sales=round(total), penalty=round(pen), penalty_low=round(total * min(seasons)),
                penalty_high=round(total * max(seasons)), S=round(total - pen), pct=round(100 * pen / total, 3))


def check_consistency(folder, fc: pd.DataFrame) -> float:
    """forecast.csv가 composition.csv와 발전소별 예측의 합과 같은지. 최대 절대차(kWh)를 돌려준다."""
    comp = pd.read_csv(folder / "composition.csv").set_index("site_id")["group"]
    site = pd.read_parquet(folder / "site_forecast_daylight.parquet")
    agg = site.assign(group=site["site_id"].map(comp)).groupby(["group", "ts"])["kwh"].sum()
    got = fc.set_index(["group", "ts"])["forecast_kwh"]
    diff = float((got - agg.reindex(got.index, fill_value=0.0)).abs().max())
    assert diff < 1e-6, f"forecast.csv가 composition.csv·발전소별 예측과 맞지 않는다 (최대차 {diff})"
    return diff


def register(name: str, command: str = "", note: str = "") -> None:
    folder = SUB_DIR / name
    validate(folder / "composition.csv", folder / "forecast.csv")
    fc = _read_forecast(folder)
    check_consistency(folder, fc)
    comp = pd.read_csv(folder / "composition.csv").merge(load_sites()[["site_id", "capacity_kw"]], on="site_id")
    manifest = folder / "manifest.json"
    sha = code_sha()
    entry = dict(
        name=name, registered_at=datetime.now(ZoneInfo(TZ)).isoformat(timespec="seconds"),
        sha256=dict(composition=_sha(folder / "composition.csv"), forecast=_sha(folder / "forecast.csv")),
        code_sha=sha, code_snapshot=_snapshot(sha),
        config=json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else dict(command=command),
        sites=comp.groupby("group")["site_id"].count().to_dict(),
        capacity_kw=comp.groupby("group")["capacity_kw"].sum().round(1).to_dict(),
        annual_gwh=fc.groupby("group")["forecast_kwh"].sum().div(1e6).round(3).to_dict(),
        expected=expected(fc), note=note, results=[])
    log = _load()
    log["entries"] = [e for e in log["entries"] if e["name"] != name or e["results"]]
    assert all(e["name"] != name for e in log["entries"]), f"{name}은 이미 제출 결과가 있다. 이름을 바꿔 새로 등록한다"
    log["entries"].append(entry)
    _save(log)
    x = entry["expected"]
    print(f"등록: {name}  예측 매출 {x['sales']:,}원  기대 페널티 {x['penalty']:,}원 ({x['pct']}%)  "
          f"범위 {x['penalty_low']:,} ~ {x['penalty_high']:,}원  기대 S {x['S']:,}원")
    print(f"  코드 {sha[:12]}  구성 {entry['sha256']['composition'][:12]}  예측 {entry['sha256']['forecast'][:12]}")


def result(name: str, value: float, kind: str = "unknown", date: str = "", note: str = "") -> None:
    log = _load()
    entry = next(e for e in log["entries"] if e["name"] == name)
    folder = SUB_DIR / name
    for k in ("composition", "forecast"):
        assert _sha(folder / f"{k}.csv") == entry["sha256"][k], f"{k}.csv가 등록 뒤 바뀌었다"
    date = date or datetime.now(ZoneInfo(TZ)).date().isoformat()
    used = [r for e in log["entries"] for r in e["results"]]
    assert len(used) < BUDGET["total"], "총 제출 횟수를 넘었다"
    assert sum(r["date"] == date for r in used) < BUDGET["per_day"], f"{date}의 제출 횟수를 넘었다"
    x = entry["expected"]
    norm = json.loads(NORMALIZED.read_text(encoding="utf-8"))["summary"] if kind == "normalized" else None
    ref = dict(penalty=x["penalty"], S=x["S"], pct=x["pct"]).get(kind) or (norm and norm["normalized_mean"])
    entry["results"].append(dict(date=date, value=value, kind=kind, note=note,
                                 vs_expected=round(value / ref, 4) if ref else None))
    _save(log)
    print(f"기록: {name} {date} {kind} {value:,}  (누적 {len(used) + 1}/{BUDGET['total']}회)")
    if ref:
        print(f"  기대값 {ref:,} 대비 {value / ref:.3f}배")
    if kind == "penalty" and not x["penalty_low"] <= value <= x["penalty_high"]:
        print(f"  기대 범위({x['penalty_low']:,} ~ {x['penalty_high']:,}원) 밖이다. 산식·단위·시각 정렬을 먼저 확인한다")
    if norm and not norm["normalized_min"] <= value <= norm["normalized_max"]:
        print(f"  검증 구간 범위({norm['normalized_min']} ~ {norm['normalized_max']}점) 밖이다. 산식·단위·시각 정렬을 먼저 확인한다")


def show() -> None:
    log = _load()
    used = [r for e in log["entries"] for r in e["results"]]
    print(f"제출 {len(used)}/{BUDGET['total']}회 사용")
    for d, n in sorted(pd.Series([r["date"] for r in used], dtype=str).value_counts().items()):
        print(f"  {d}: {n}/{BUDGET['per_day']}회")
    for e in log["entries"]:
        x = e["expected"]
        print(f"{e['name']}  코드 {e['code_sha'][:12]}  예측 {e['sha256']['forecast'][:12]}  "
              f"연간 {sum(e['annual_gwh'].values()):.2f} GWh  기대 페널티 {x['penalty']:,}원 ({x['pct']}%)")
        for r in e["results"]:
            print(f"    {r['date']}  {r['kind']} {r['value']:,}  기대 대비 {r['vs_expected']}  {r['note']}")


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    pos = [a for a in args if "=" not in a]
    kw = dict(a.split("=", 1) for a in args if "=" in a)
    if cmd == "register":
        register(pos[0], **kw)
    elif cmd == "result":
        result(pos[0], float(pos[1]), **kw)
    else:
        show()
