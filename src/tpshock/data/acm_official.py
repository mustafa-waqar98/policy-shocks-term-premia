"""The NY Fed's published ACM term premium, used only to validate our estimates."""

from pathlib import Path

import pandas as pd

from tpshock import config
from tpshock.data.download import fetch


def download(dest: Path = config.RAW / "ACMTermPremium.xls") -> Path:
    return fetch(config.ACM_OFFICIAL_URL, dest)


def load(path: Path = config.RAW / "ACMTermPremium.xls", sheet: str = "ACM Monthly") -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet)
    df["DATE"] = pd.to_datetime(df["DATE"], format="mixed")
    return df.set_index("DATE").sort_index()
