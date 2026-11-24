"""Gurkaynak, Sack and Wright (2007) Treasury zero curve.

The Fed publishes daily Svensson curve parameters. From them we can compute a
zero-coupon yield at any maturity, which is what the ACM model needs (monthly
maturities from 1 to 120 months).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from tpshock import config
from tpshock.data.download import fetch

PARAMS = ["BETA0", "BETA1", "BETA2", "BETA3", "TAU1", "TAU2"]


def download(dest: Path = config.RAW / "feds200628.csv") -> Path:
    return fetch(config.GSW_URL, dest)


def load(path: Path = config.RAW / "feds200628.csv") -> pd.DataFrame:
    """Return the Svensson parameters indexed by date."""
    # The file starts with a few lines of notes; find the header row.
    with open(path) as fh:
        for i, line in enumerate(fh):
            if line.startswith("Date"):
                skip = i
                break
    df = pd.read_csv(path, skiprows=skip, na_values=["NA", "-999.99"])
    dates = pd.to_datetime(df["Date"], format="mixed")
    # Some vintages use 2-digit years; anything parsed into the future is last century.
    future = dates > pd.Timestamp.today()
    dates[future] = dates[future] - pd.DateOffset(years=100)
    df["Date"] = dates
    df = df.set_index("Date").sort_index()[PARAMS]
    df = df.dropna(subset=["BETA0", "BETA1", "BETA2", "TAU1"])
    # Before 1980 the curve is Nelson-Siegel (no second hump term).
    no_hump = df["TAU2"].isna() | (df["TAU2"] <= 0)
    df.loc[no_hump, "BETA3"] = 0.0
    df.loc[no_hump, "TAU2"] = 1.0  # any positive value works when BETA3 = 0
    return df


def svensson(params: pd.DataFrame, maturities_months) -> pd.DataFrame:
    """Continuously compounded zero yields in percent, columns = maturity in months."""
    n = np.asarray(maturities_months, dtype=float)[None, :] / 12.0
    b0, b1, b2, b3, t1, t2 = (params[c].to_numpy()[:, None] for c in PARAMS)
    x1, x2 = n / t1, n / t2
    f1 = (1 - np.exp(-x1)) / x1
    f2 = f1 - np.exp(-x1)
    f3 = (1 - np.exp(-x2)) / x2 - np.exp(-x2)
    y = b0 + b1 * f1 + b2 * f2 + b3 * f3
    return pd.DataFrame(y, index=params.index, columns=list(maturities_months))


def month_end(df: pd.DataFrame) -> pd.DataFrame:
    """Last available observation in each month, indexed by that actual date."""
    return df.groupby(df.index.to_period("M")).tail(1)
