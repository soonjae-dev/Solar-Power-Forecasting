"""GFS sflux 복사 성분의 공개 표본/GRIB 헤더만 조사. 데이터 저장·채점 없음."""
from datetime import date, datetime, timedelta, timezone
import json

from ..config import EXP_DIR
from .external_weather_audit import probe
from .gefs_mean import error_summary, published_before_cutoff, request, save
from ..gfs_fetch import parse_idx

TARGETS = (date(2021, 3, 1), date(2023, 8, 11), date(2024, 8, 16))
FIELDS = ("CSDSF", "VBDSF", "VDDSF", "NBDSF", "NDDSF")
PATH = EXP_DIR / "p45_gfs_radiation_probe" / "metrics.json"


def url_for(target):
    run = target - timedelta(days=1)
    folder = "atmos/" if run >= date(2021, 3, 23) else ""
    return (f"https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{run:%Y%m%d}/00/"
            f"{folder}gfs.t00z.sfluxgrbf027.grib2")


def metadata(message):
    import eccodes
    gid = eccodes.codes_new_from_message(message)
    try:
        # 원소값/좌표 배열은 읽거나 출력하지 않는다.
        keys = ("gridType", "Ni", "Nj", "numberOfPoints", "stepType", "startStep", "endStep")
        return {k: eccodes.codes_get(gid, k) for k in keys}
    finally:
        eccodes.codes_release(gid)


def sample(target):
    url = url_for(target)
    row = dict(target_day=target.isoformat(), fields={})
    try:
        head = probe(url)
        if head["status"] != "reachable":
            return {**row, "status": head["status"], "http_status": head.get("http_status"),
                    "error_type": head.get("error_type")}
        row["body_last_modified"] = published_before_cutoff(target, {"last-modified": head["last_modified"]})
        payload, headers = request(url + ".idx")
        row["idx_last_modified"] = published_before_cutoff(target, headers)
        text = payload.decode("ascii")
        for name in FIELDS:
            a, b, start, end = parse_idx(text, name, "surface")
            if end != 27:
                raise ValueError("복사 성분 예보 시각 불일치")
            message, headers = request(url, (a, b))
            published_before_cutoff(target, headers)
            meta = metadata(message)
            if meta["stepType"] != "avg" or meta["startStep"] != start or meta["endStep"] != end:
                raise ValueError("idx와 GRIB 평균 구간 불일치")
            row["fields"][name] = meta
        row["status"] = "eligible_sample"
    except Exception as exc:
        row.update(status="unverified", **error_summary(exc))
    return row


def main():
    rows = [sample(target) for target in TARGETS]
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(), probe_name="gfs_sflux_radiation",
                  rule="14.34: fixed public samples only; no model scores or competition data",
                  samples=rows, eligible_samples=sum(r["status"] == "eligible_sample" for r in rows),
                  model_trained=False, bodies_saved=False)
    result["can_audit_next"] = result["eligible_samples"] > 0
    save(PATH, result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
