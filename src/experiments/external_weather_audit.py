"""공개 GFS 파일의 HTTP 메타데이터만 확인한다. 대회 데이터는 읽지 않는다.

    .venv/bin/python -m src.experiments.external_weather_audit
    .venv/bin/python -m src.experiments.external_weather_audit coverage [18]

HEAD 성공은 자격 승인이 아니다. 원본 공개 시각/동일본 증빙은 별도 확인한다.
본문 다운로드, 모델 학습, 특징 캐시/제출 파일 변경은 하지 않는다.
"""
import json
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

KST = timezone(timedelta(hours=9))
REGISTRY = "https://registry.opendata.aws/noaa-gfs-bdp-pds/"
ARCHIVE = "https://www.ncei.noaa.gov/products/weather-climate-models/global-forecast"


def prediction_cutoff(target_day):
    """대상일 전날 17시. 대상일은 YYYY-MM-DD로 명시한다."""
    day = datetime.strptime(target_day, "%Y-%m-%d").replace(tzinfo=KST)
    return day - timedelta(hours=7)


def eligible_publication(target_day, published_at):
    """발행 실행 시각 대신 검증된 일반 공개 시각을 전달해야 한다."""
    if published_at is None or published_at.tzinfo is None:
        return False
    return published_at <= prediction_cutoff(target_day)


def probe(url, timeout=5):
    try:
        with urlopen(Request(url, method="HEAD"), timeout=timeout) as response:
            return dict(status="reachable", http_status=response.status,
                        last_modified=response.headers.get("Last-Modified"),
                        etag=response.headers.get("ETag"),
                        content_length=response.headers.get("Content-Length"))
    except HTTPError as exc:
        return dict(status="http_error", http_status=exc.code)
    except (URLError, OSError, TimeoutError) as exc:
        # 오류/응답 본문은 출력하지 않는다. DNS 실패를 파일 부재로 해석하지 않는다.
        return dict(status="network_unverified", error_type=type(exc).__name__)


def required_keys(cycle=0):
    """대상일 ts 01시~익일 00시는 전날 00UTC 실행의 f016~f039다. 구간 평균 차분에 f015가 더 필요하다.
    전전날 18UTC 실행이면 6시간씩 밀린 f021~f045다."""
    shift = 6 if cycle == 18 else 0
    names = [f"gfs.t{cycle:02d}z.pgrb2.0p25.f{h + shift:03d}" for h in range(15, 40)]
    return names + [f"{n}.idx" for n in names]


def classify_day(target_day, listed, cycle=0):
    """listed: 파일명 -> LastModified(시간대 포함). 전부 있고 가장 늦은 시각이 cutoff 이하일 때만 eligible."""
    keys = required_keys(cycle)
    missing = [k for k in keys if k not in listed]
    if missing:
        return dict(status="missing", missing=len(missing), last=None)
    last = max(listed[k] for k in keys)
    return dict(status="eligible" if eligible_publication(target_day, last) else "late", missing=0, last=last)


def parse_listing(xml_bytes):
    root = ET.fromstring(xml_bytes)
    ns = {"s3": root.tag.split("}")[0].strip("{")}
    return {c.find("s3:Key", ns).text.rsplit("/", 1)[-1]:
            datetime.fromisoformat(c.find("s3:LastModified", ns).text.replace("Z", "+00:00"))
            for c in root.findall("s3:Contents", ns)}


def list_run(run_day, retries=3, cycle=0):
    # 2021-03-22 실행까지는 atmos/ 없는 경로다.
    listed = {}
    for folder in ("atmos/", ""):
        prefix = f"gfs.{run_day:%Y%m%d}/{cycle:02d}/{folder}gfs.t{cycle:02d}z.pgrb2.0p25.f0"
        url = f"https://noaa-gfs-bdp-pds.s3.amazonaws.com/?list-type=2&prefix={prefix}"
        for attempt in range(retries):
            try:
                with urlopen(url, timeout=30) as response:
                    listed = parse_listing(response.read())
                break
            except (URLError, OSError, TimeoutError):
                if attempt == retries - 1:
                    return None
        if listed:
            break
    return listed


def check_day(target, cycle=0):
    listed = list_run(target - timedelta(days=2 if cycle == 18 else 1), cycle=cycle)
    if listed is None:
        return target, dict(status="network_unverified", missing=None, last=None)
    return target, classify_day(target.isoformat(), listed, cycle)


def coverage(first=date(2021, 1, 2), last=date(2026, 5, 31), holdout_start=date(2025, 6, 1), cycle=0):
    days = [first + timedelta(days=i) for i in range((last - first).days + 1)]
    with ThreadPoolExecutor(16) as pool:
        rows = list(pool.map(check_chem_day if cycle == "chem" else check_spread_day if cycle == "spread" else (lambda d: check_day(d, cycle)), days))
    def summary(part):
        margin = [(prediction_cutoff(d.isoformat()) - r["last"]).total_seconds() / 3600 for d, r in part if r["last"]]
        margin.sort()
        counts = {s: sum(r["status"] == s for _, r in part) for s in ("eligible", "late", "missing", "network_unverified")}
        return dict(days=len(part), **counts, eligible_share=round(counts["eligible"] / max(1, len(part)), 4),
                    margin_hours_min=round(margin[0], 2) if margin else None,
                    margin_hours_median=round(margin[len(margin) // 2], 2) if margin else None)
    by_year = {str(y): summary([x for x in rows if x[0].year == y]) for y in sorted({d.year for d in days})}
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(),
                  scope="public S3 listing metadata only; no bodies downloaded; no competition data accessed",
                  rule="all f015-f039 body and idx present and latest LastModified <= previous day 08:00 UTC",
                  train=summary([x for x in rows if x[0] < holdout_start]),
                  evaluation=summary([x for x in rows if x[0] >= holdout_start]), by_year=by_year,
                  not_eligible=[dict(day=d.isoformat(), status=r["status"], missing=r["missing"],
                                     last=r["last"].isoformat() if r["last"] else None)
                                for d, r in rows if r["status"] != "eligible"])
    out = Path(__file__).resolve().parents[2] / {0: "experiments/p22_gfs_coverage", 18: "experiments/p28_gfs18_coverage", "chem": "experiments/p32_gefs_chem_coverage",
                                                  "spread": "experiments/p34_gefs_spread_coverage"}[cycle]
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("train", "evaluation", "by_year")}, ensure_ascii=False, indent=1))
    print("not eligible:", len(result["not_eligible"]))
    for r in result["not_eligible"][:40]:
        print(" ", r)


CHEM_STEPS = (21, 24, 27, 30, 33)


def chem_keys():
    names = [f"gefs.chem.t00z.a2d_0p25.f{h:03d}.grib2" for h in CHEM_STEPS]
    return names + [f"{n}.idx" for n in names]


def spread_keys():
    names = [f"gespr.t00z.pgrb2s.0p25.f{h:03d}" for h in CHEM_STEPS]
    return names + [f"{n}.idx" for n in names]


def check_spread_day(target, retries=3):
    run_day = target - timedelta(days=1)
    url = ("https://noaa-gefs-pds.s3.amazonaws.com/?list-type=2&prefix="
           f"gefs.{run_day:%Y%m%d}/00/atmos/pgrb2sp25/gespr.t00z.pgrb2s.0p25.f0")
    return check_chem_day(target, retries, url, spread_keys())


def check_chem_day(target, retries=3, url=None, keys=None):
    run_day = target - timedelta(days=1)
    url = url or ("https://noaa-gefs-pds.s3.amazonaws.com/?list-type=2&prefix="
                  f"gefs.{run_day:%Y%m%d}/00/chem/pgrb2ap25/gefs.chem.t00z.a2d_0p25.f0")
    for attempt in range(retries):
        try:
            with urlopen(url, timeout=30) as response:
                listed = parse_listing(response.read())
            break
        except (URLError, OSError, TimeoutError):
            if attempt == retries - 1:
                return target, dict(status="network_unverified", missing=None, last=None)
    keys = keys or chem_keys()
    missing = [k for k in keys if k not in listed]
    if missing:
        return target, dict(status="missing", missing=len(missing), last=None)
    last = max(listed[k] for k in keys)
    return target, dict(status="eligible" if eligible_publication(target.isoformat(), last) else "late", missing=0, last=last)


def main():
    rows = []
    for run_day in ("20210531", "20240531", "20250531"):
        url = ("https://noaa-gfs-bdp-pds.s3.amazonaws.com/"
               f"gfs.{run_day}/00/atmos/gfs.t00z.pgrb2.0p25.f016")
        rows.append(dict(url=url, **probe(url)))
    result = dict(
        checked_at=datetime.now(timezone.utc).isoformat(),
        scope="public HTTP headers only; no competition data accessed",
        registry_url=REGISTRY, archive_documentation_url=ARCHIVE,
        distribution_license="NODD public data usable as desired; registry documentation",
        probes=rows,
        qualification="pending; not integrated into v3",
        unresolved=["required historical date coverage and repeatable retrieval",
                    "institutional evidence of original publication before each cutoff",
                    "identity of retrieved files with originally published versions",
                    "GRIB radiation averaging intervals and hourly conversion"],
        current_production_status_page_is_historical_publication_evidence=False,
        head_last_modified_alone_is_publication_evidence=False,
    )
    out = Path(__file__).resolve().parents[2] / "experiments/p21_external_weather_audit"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    counts = {s: sum(r["status"] == s for r in rows)
              for s in ("reachable", "http_error", "network_unverified")}
    print(json.dumps(dict(probes=len(rows), status_counts=counts,
                          qualification=result["qualification"]), ensure_ascii=False))


if __name__ == "__main__":
    if sys.argv[1:2] == ["coverage"]:
        arg = sys.argv[2] if len(sys.argv) > 2 else "0"
        coverage(cycle=arg if arg in ("chem", "spread") else int(arg))
    else:
        main()
