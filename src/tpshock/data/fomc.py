"""High-frequency FOMC surprises from Jarocinski and Karadi (CC BY 4.0).

The raw file holds the change in futures prices in a 30-minute window around
each FOMC announcement: fed funds futures (FF1-FF4, MP1), eurodollar / SOFR
futures (ED1-ED4), Treasury futures and the S&P 500. We build our own shocks
from it (see tpshock.shocks) and check them against the published ones.
"""

from pathlib import Path

import pandas as pd

from tpshock import config
from tpshock.data.download import fetch


def download_surprises(dest: Path = config.RAW / "fomc_surprises_jk.csv") -> Path:
    return fetch(config.JK_SURPRISES_URL, dest)


def download_published(dest: Path = config.RAW / "shocks_fed_jk_t.csv") -> Path:
    return fetch(config.JK_SHOCKS_URL, dest)


def load_surprises(path: Path = config.RAW / "fomc_surprises_jk.csv") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["start"])
    return df.set_index("start").sort_index()


def load_published(path: Path = config.RAW / "shocks_fed_jk_t.csv") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["start"])
    return df.set_index("start").sort_index()
