"""FRED series via the public CSV endpoint (no API key needed)."""

from pathlib import Path

import pandas as pd

from tpshock import config
from tpshock.data.download import fetch


def download(series: str, dest_dir: Path = config.RAW / "fred") -> Path:
    return fetch(config.FRED_URL.format(series=series), dest_dir / f"{series}.csv")


def load(series: str, dest_dir: Path = config.RAW / "fred") -> pd.Series:
    df = pd.read_csv(dest_dir / f"{series}.csv", na_values=["."])
    date_col = df.columns[0]
    s = df.set_index(pd.to_datetime(df[date_col]))[series].astype(float)
    return s.dropna().rename(series)


def load_all(dest_dir: Path = config.RAW / "fred") -> pd.DataFrame:
    out = {}
    for s in config.FRED_SERIES:
        if (dest_dir / f"{s}.csv").exists():
            out[s] = load(s, dest_dir)
    return pd.DataFrame(out)
