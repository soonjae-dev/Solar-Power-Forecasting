"""진행 상황 요약을 JSON 한 줄로 출력한다. 진행 패널 모드가 주기적으로 부른다.

    .venv/bin/python -m src.experiments.progress_status

파일 개수·수정 시각과 experiments의 집계 metrics.json만 읽는다. 대회 데이터 내용은 읽지 않는다.
"""
import json
import subprocess
import time
from datetime import date

from ..config import DATA_ROOT, EXP_DIR

DOWNLOADS = (("GFS 00UTC", "gfs_00utc", "p22_gfs_coverage", "2021-01-02", "2026-05-31"),
             ("GFS 추가 변수", "gfs_00utc_more", "p22_gfs_coverage", "2021-01-02", "2025-04-30"),
             ("GFS 추가 변수(평가 기간)", "gfs_00utc_more", "p22_gfs_coverage", "2025-05-01", "2026-05-31"),
             ("GFS 18UTC 이전 실행", "gfs_18utc_lag", "p28_gfs18_coverage", "2021-01-02", "2025-04-30"),
             ("GEFS 에어로졸", "gefs_chem_00utc", "p32_gefs_chem_coverage", "2021-01-02", "2026-05-31"),
             ("GEFS 앙상블 퍼짐", "gefs_spread_00utc", "p34_gefs_spread_coverage", "2021-01-02", "2026-05-31"))
EXPERIMENTS = (("gfs20", "p24_gfs_v3"), ("gfs60", "p26_gfs_more"), ("lag18", "p29_gfs_lag"), ("snow10", "p30_gfs_snow"), ("gfs60 10시드", "p31_gfs_seeds"), ("aod20", "p33_gefs_aod"), ("spr20", "p35_gefs_spread"))
JOBS = ("gfs_fetch", "gfs_aod", "gfs_spread", "gfs_snow", "gfs_seeds", "diagnose_season", "gfs_more", "gfs_lag", "gfs_holdout", "make_submission", "gfs_v3", "gfs_pca")
WINDOWS = 5


def eligible(audit, first, last):
    path = EXP_DIR / audit / "metrics.json"
    if not path.exists():
        return None
    bad = {r["day"] for r in json.loads(path.read_text())["not_eligible"]}
    a, b = date.fromisoformat(first), date.fromisoformat(last)
    return sum(1 for i in range((b - a).days + 1) if date.fromordinal(a.toordinal() + i).isoformat() not in bad)


def download(label, folder, audit, first, last, now):
    d = DATA_ROOT / "external" / folder
    times = [f.stat().st_mtime for f in d.glob("*.npz") if ".tmp" not in f.name and first.replace("-", "") <= f.stem <= last.replace("-", "")] if d.exists() else []
    target = eligible(audit, first, last)
    recent = sum(now - t < 600 for t in times)
    eta = round((target - len(times)) / (recent / 10)) if target and recent else None
    return dict(label=label, done=len(times), target=target, per_min=round(recent / 10, 1), eta_min=eta)


def experiment(label, folder):
    path = EXP_DIR / folder / "metrics.json"
    if not path.exists():
        return dict(label=label, state="대기")
    m = json.loads(path.read_text())
    out = dict(label=label, state="진행")
    for stage in ("seeds1", "seeds3", "seeds10"):
        runs = m.get(stage, {})
        for rows in runs.values():
            out[stage] = f"{len(rows)}/{WINDOWS}"
            if "fold1_static" in rows:
                out[f"{stage}_static"] = rows["fold1_static"]["vs_v3_3seeds"]["change_pct"]
    for key in ("verdict_vs_gfs20", "verdict_vs_control", "seeds1_verdicts"):
        if key in m:
            v = m[key]
            passed = v.get("passed") if "passed" in v else next(iter(v.values())).get("screen_pass")
            out["seeds1_pass"] = passed
            break
    for key in ("verdict3_vs_gfs20", "verdict3_vs_control", "seeds3_verdicts"):
        if key in m:
            v = m[key]
            out["seeds3_pass"] = v.get("passed") if "passed" in v else next(iter(v.values())).get("screen_pass")
            break
    return out


def jobs():
    """실제 python 프로세스만 센다. 대기 중인 셸 명령 문자열에 이름이 들어 있어도 세지 않는다."""
    lines = subprocess.run(["ps", "-axo", "command"], capture_output=True, text=True).stdout.splitlines()
    procs = [x for x in lines if "python" in x.split(" ")[0].lower() or x.split(" ")[0].endswith("/Python")]
    return [name for name in JOBS if any(f"src.{name}" in x for x in procs)]


def main():
    now = time.time()
    print(json.dumps(dict(at=time.strftime("%H:%M:%S"), jobs=jobs(),
                          downloads=[download(*x, now) for x in DOWNLOADS],
                          experiments=[experiment(*x) for x in EXPERIMENTS]), ensure_ascii=False))


if __name__ == "__main__":
    main()
