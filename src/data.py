"""원본 데이터 로딩. 원본 파일에는 쓰지 않는다."""
import pandas as pd
import pyarrow.parquet as pq

from .config import DATA_DIR, TRAIN_DIR, TEST_DIR, TZ


def _kst(s: pd.Series) -> pd.Series:
    t = pd.to_datetime(s)
    if t.dt.tz is None:
        raise ValueError("timezone 없는 시각은 임의로 해석하지 않는다")
    return t.dt.tz_convert(TZ).astype(f"datetime64[ns, {TZ}]")


def load_sites() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "sites.csv")


def load_generation() -> pd.DataFrame:
    g = pd.read_parquet(TRAIN_DIR / "generation.parquet")
    g["ts"] = _kst(g["ts"])
    return g


def load_smp(split: str) -> pd.DataFrame:
    """split='train'은 ts·smp, 'test'는 ts·smp·k. k는 항상 ts의 시로 다시 계산해 붙인다."""
    s = pd.read_csv((TRAIN_DIR if split == "train" else DATA_DIR) / "smp.csv")
    s["ts"] = _kst(s["ts"])
    k = s["ts"].dt.hour.between(11, 15).map({True: 3, False: 1})
    if "k" in s and not (s["k"] == k).all():
        raise ValueError("smp.csv의 k가 README 규칙과 다르다")
    s["k"] = k
    s["w"] = s["smp"] * s["k"]
    return s


def forecast_columns(provider: str) -> list:
    names = pq.read_schema(TRAIN_DIR / f"forecast_{provider}.parquet").names
    return [c for c in names if c not in (f"{provider}_station", "round", "ts")]


def load_forecast(split: str, provider: str, cols=None, start=None, end=None) -> pd.DataFrame:
    """긴 형식 예보. 키는 ({provider}_station, round, ts)."""
    folder = TRAIN_DIR if split == "train" else TEST_DIR
    keys = [f"{provider}_station", "round", "ts"]
    cols = forecast_columns(provider) if cols is None else list(cols)
    f = pd.read_parquet(folder / f"forecast_{provider}.parquet", columns=keys + cols)
    f["ts"] = _kst(f["ts"])
    if start is not None:
        f = f[f["ts"] >= start]
    if end is not None:
        f = f[f["ts"] <= end]
    f[cols] = f[cols].astype("float32")
    return f
