"""공개 GFS 요약이 v3 고정 모델 잔차에 정보를 더하는지 본다. 집계만 출력하고 experiments/p23_gfs_pilot/metrics.json에 저장한다.

    .venv/bin/python -m src.gfs_pilot

판정 기준은 실행 전에 고정했다(docs/validation.md). 2025-05 holdout은 쓰지 않는다.
"""
import json

import numpy as np
import pandas as pd

from .config import CACHE_DIR, DATA_ROOT, EXP_DIR, TZ
from .correct_total import WEATHER, evaluate, table, train
from .data import load_generation, load_sites, load_smp
from .spatial_context import paired_comparison

GFS_DIR = DATA_ROOT / "external" / "gfs_00utc"
FIRST_TS_HOUR = 4
LATS = np.arange(40, 31.75, -0.25)
LONS = np.arange(123, 132.25, 0.25)
BLOCKS = (("2024-10-01", "2024-12-01"), ("2024-12-01", "2025-02-01"), ("2025-02-01", "2025-05-01"))
TRAIN_START = pd.Timestamp("2024-06-01 01:00", tz=TZ)
END = pd.Timestamp("2025-05-01", tz=TZ)


def region_masks():
    """위도 33~39N, 경도 124~130E를 2도씩 3×3으로 나눈 고정 칸과 상자 전체. 칸 경계 격자는 한 칸에만 넣는다."""
    lat, lon = np.meshgrid(LATS, LONS, indexing="ij")
    box = (lat >= 33) & (lat < 39) & (lon >= 124) & (lon < 130)
    masks = {"all": box}
    for i, south in enumerate((33, 35, 37)):
        for j, west in enumerate((124, 126, 128)):
            masks[f"r{i}{j}"] = (lat >= south) & (lat < south + 2) & (lon >= west) & (lon < west + 2)
    return masks


def summarize(field, masks):
    """field: (시각, 위도, 경도). 영역 평균을 영역별 열로 돌려준다."""
    return {name: field[:, mask].mean(axis=1) for name, mask in masks.items()}


def load_gfs(first="2024-06-01", last="2025-04-30", directory=GFS_DIR, prefix="gfs"):
    masks = region_masks()
    rows, missing = [], []
    for day in pd.date_range(first, last):
        path = directory / f"{day:%Y%m%d}.npz"
        if not path.exists():
            missing.append(day.date().isoformat())
            continue
        z = np.load(path)
        index = pd.DatetimeIndex([day + pd.Timedelta(hours=FIRST_TS_HOUR + i) for i in range(z["dswrf"].shape[0])]).tz_localize(TZ)
        frame = {}
        for key, low, high in (("dswrf", 0, None), ("tcdc", 0, 100)):
            for name, values in summarize(np.clip(z[key], low, high), masks).items():
                frame[f"{prefix}_{key}_{name}"] = values
        rows.append(pd.DataFrame(frame, index=index))
    return pd.concat(rows).astype("float32").sort_index(), missing


def take(t, mask):
    return {**t, "X": t["X"].loc[mask], "base": t["base"][mask], "act": t["act"][mask], "cap": t["cap"][mask],
            "y": t["y"][mask], "w": t["w"][mask], "index": t["index"][mask]}


def with_columns(t, extra):
    return {**t, "X": pd.concat([t["X"], extra.reindex(t["index"])], axis=1)}


def blocks():
    """각 시험 구간과 그 구간 시작 전날 17시 학습 마감. 시각은 우측닫힘이다."""
    out = []
    for start, stop in BLOCKS:
        day, end = pd.Timestamp(start, tz=TZ), pd.Timestamp(stop, tz=TZ)
        if end > END:
            raise ValueError("5월 holdout 사용 금지")
        out.append(dict(name=f"{start}~{stop}", cutoff=day - pd.Timedelta(hours=7), start=day + pd.Timedelta(hours=1), end=end))
    return out


def diagnostic_corr(t, extra, day):
    """무학습 보조 진단: 낮 시각의 전체 잔차와 GFS 요약의 상관. 대조로 같은 잔차와 제공 예보 운량의 상관도 본다."""
    resid = pd.Series(t["y"], index=t["index"])[day]
    out = {}
    for name, x in (("gfs_dswrf_all", extra["gfs_dswrf_all"]), ("gfs_tcdc_all", extra["gfs_tcdc_all"]),
                    ("wa_cloud", t["X"]["wa_cloud"]), ("wb_cloud", t["X"]["wb_cloud"])):
        pair = pd.concat([resid, x.reindex(resid.index)], axis=1).dropna()
        out[name] = dict(pearson=round(float(pair.corr().iloc[0, 1]), 4), spearman=round(float(pair.corr(method="spearman").iloc[0, 1]), 4), hours=len(pair))
    return out


def verdict(rows, overall):
    changes = [r["vs_control"]["change_pct"] for r in rows.values()]
    return dict(overall_change_pct=overall["change_pct"], overall_ci90_pct=overall["change_ci90_pct"],
                blocks_improved=int(sum(c < 0 for c in changes)), worst_block_pct=max(changes),
                passed=bool(overall["change_pct"] <= -1 and overall["change_ci90_pct"][1] < 0
                            and sum(c < 0 for c in changes) >= 2 and max(changes) <= 1))


def main():
    gfs, missing = load_gfs()
    if not gfs.index.is_unique:
        raise ValueError("GFS 시각 중복")
    print(f"GFS {len(gfs)}시각, {gfs.shape[1]}열, 빠진 날 {len(missing)}", flush=True)
    weather = pd.read_parquet(CACHE_DIR / "train_features.parquet", columns=["site_id", "ts", *WEATHER])
    weather = weather[weather["ts"] <= END]
    truth = load_generation()[["site_id", "ts", "energy"]]
    truth = truth[truth["ts"] <= END]
    sites = load_sites().set_index("site_id")
    smp = load_smp("train").set_index("ts")
    t = table("fold1_static", weather, truth, sites, smp)
    del weather
    day = t["base"].sum(axis=1) > 0
    # 시간 정렬 검사: GFS 일사량이 v3 낮 예측과 같은 시각에 움직여야 한다. 어긋나면 평가하지 않는다.
    lags = {lag: float(pd.Series(t["base"].sum(axis=1), index=t["index"]).corr(gfs["gfs_dswrf_all"].shift(lag).reindex(t["index"]))) for lag in (-1, 0, 1)}
    print("시간 정렬 상관", {k: round(v, 4) for k, v in lags.items()}, flush=True)
    if max(lags, key=lags.get) != 0:
        raise ValueError("GFS 시각 정렬이 어긋난다")
    extra = gfs.reindex(t["index"])
    result = dict(rule="14.15", gfs_hours=len(gfs), missing_days=missing, time_alignment=lags,
                  diagnostic_correlation=diagnostic_corr(t, gfs, day), blocks={})
    ts = pd.Series(t["index"], index=t["index"])
    tests_control, tests_gfs, test_masks = [], [], []
    for b in blocks():
        fit = ((ts >= TRAIN_START) & (ts <= b["cutoff"])).to_numpy()
        test = ((ts >= b["start"]) & (ts <= b["end"])).to_numpy()
        control = train([take(t, fit)])
        added = train([with_columns(take(t, fit), extra)])
        tt = take(t, test)
        cv, gv = tt, with_columns(tt, extra)
        row = dict(train_hours=int((fit & day).sum()), test_hours=int(test.sum()), control_vs_v3=evaluate(cv, control),
                   gfs_vs_v3=evaluate(gv, added), vs_control=paired_comparison(tt, (control, cv), (added, gv), smp))
        result["blocks"][b["name"]] = row
        tests_control.append(control.predict(cv["X"]) * tt["cap"])
        tests_gfs.append(added.predict(gv["X"]) * tt["cap"])
        test_masks.append(test)
        print(f"{b['name']}: GFS/대조군 {row['vs_control']['change_pct']:+.3f}%  대조군/v3 {row['control_vs_v3']['change_pct']:+.3f}%  GFS/v3 {row['gfs_vs_v3']['change_pct']:+.3f}%", flush=True)
    mask = np.logical_or.reduce(test_masks)
    control_delta = np.concatenate(tests_control)
    overall = pooled(t, control_delta, np.concatenate(tests_gfs), mask, smp)
    result["overall_vs_control"] = overall
    result["verdict"] = verdict(result["blocks"], overall)
    tt = take(t, mask)
    zero = np.zeros(len(tt["index"]))
    result["overall_vs_v3"] = dict(control=pooled(t, zero, control_delta, mask, smp),
                                   gfs=pooled(t, zero, np.concatenate(tests_gfs), mask, smp))
    result["placebo"] = placebo(t, gfs, control_delta, mask, smp, overall["change_pct"])
    print("v3 대비 합계", result["overall_vs_v3"]["control"]["change_pct"], result["overall_vs_v3"]["gfs"]["change_pct"], flush=True)
    print("가짜 GFS", json.dumps({k: v for k, v in result["placebo"].items() if k != "changes"}, ensure_ascii=False), flush=True)
    out = EXP_DIR / "p23_gfs_pilot"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print("판정", json.dumps(result["verdict"], ensure_ascii=False), flush=True)
    print("보조 상관", json.dumps(result["diagnostic_correlation"], ensure_ascii=False), flush=True)


def shuffled_within_month(gfs, seed):
    """같은 달 안에서 날짜만 섞는다. 계절·시간 형태는 남고 날짜별 날씨는 사라진다."""
    rng = np.random.default_rng(seed)
    days = pd.DatetimeIndex(sorted(set(gfs.index.normalize())))
    out = gfs.copy()
    values = gfs.to_numpy()
    n = len(values) // len(days)
    for _, group in pd.Series(np.arange(len(days))).groupby(days.strftime("%Y-%m").to_numpy()):
        source = rng.permutation(group.to_numpy())
        for d, src in zip(group.to_numpy(), source):
            out.iloc[d * n:(d + 1) * n] = values[src * n:(src + 1) * n]
    return out


def placebo(t, gfs, control_delta, mask, smp, real_change, seeds=20):
    ts = pd.Series(t["index"], index=t["index"])
    changes = []
    for seed in range(seeds):
        fake = shuffled_within_month(gfs, seed).reindex(t["index"])
        deltas = []
        for b in blocks():
            fit = ((ts >= TRAIN_START) & (ts <= b["cutoff"])).to_numpy()
            test = ((ts >= b["start"]) & (ts <= b["end"])).to_numpy()
            model = train([with_columns(take(t, fit), fake)])
            view = with_columns(take(t, test), fake)
            deltas.append(model.predict(view["X"]) * view["cap"])
        changes.append(pooled(t, control_delta, np.concatenate(deltas), mask, smp)["change_pct"])
    return dict(seeds=seeds, changes=changes, median=float(np.median(changes)), min=min(changes), max=max(changes),
                real=real_change, share_at_or_below_real=float(np.mean(np.array(changes) <= real_change)))


class Fixed:
    """미리 계산한 예측을 paired_comparison에 넘기기 위한 얇은 래퍼. predict는 잔차율을 돌려준다."""

    def __init__(self, ratio):
        self.ratio = ratio

    def predict(self, _):
        return self.ratio


def pooled(t, delta_control, delta_gfs, mask, smp):
    """세 시험 구간을 합친 페널티 비교. 구간별 모델의 예측을 시간순으로 이어 붙인다."""
    tt = take(t, mask)
    return paired_comparison(tt, (Fixed(delta_control / tt["cap"]), tt), (Fixed(delta_gfs / tt["cap"]), tt), smp)


if __name__ == "__main__":
    main()
