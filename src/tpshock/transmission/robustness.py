"""Robustness checks: do the headline results survive when one assumption changes?

Each variant is a different set of FOMC meetings. For every variant we rerun the local
projections and record the headline numbers. A result that holds across variants is a
finding; one that only appears in one sample is a property of that sample.
"""

import pandas as pd

from tpshock.transmission.lp import local_projections

# The headline results: (shock, outcome, horizon in trading days, plain-English label)
KEY_RESULTS = [
    ("target", "rn120", 0, "Rate surprise: 10y expected rates, same day"),
    ("target", "tp120", 0, "Rate surprise: 10y term premium, same day"),
    ("path", "tp120", 20, "Guidance surprise: 10y term premium after 20 days"),
    ("MP_median", "DFII10", 20, "Policy shock: 10y real yield after 20 days"),
    ("CBI_median", "y24", 60, "Information shock: 2y yield after 60 days"),
    ("target", "tp120_bc", 0, "Rate surprise: 10y term premium, same day (bias-corrected model)"),
    ("path", "tp120_bc", 20, "Guidance surprise: 10y term premium after 20 days (bias-corrected model)"),
]
SHOCK_SETS = [["MP_median", "CBI_median"], ["target", "path"]]


def make_variants(shocks: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Each entry is one robustness check: a name and the meetings it keeps."""
    variants = {
        "All meetings": shocks,
        "1990-2007": shocks.loc[:"2007"],
        "2008-present": shocks.loc["2008":],
    }
    if "scheduled" in shocks:
        variants["Scheduled meetings only"] = shocks[shocks["scheduled"].astype(bool)]

    # YOUR TURN: add a check that drops every meeting in 2020 (Covid emergency cuts).
    # Hint: shocks.index.year gives the year of each meeting.
    variants["Excluding 2020"] = shocks[shocks.index.year != 2020]

    return variants


def robustness_table(daily: pd.DataFrame, shocks: pd.DataFrame) -> pd.DataFrame:
    outcomes = sorted({o for _, o, _, _ in KEY_RESULTS if o in daily})
    horizons = sorted({h for _, _, h, _ in KEY_RESULTS})
    rows = []
    for name, sample in make_variants(shocks).items():
        irfs = pd.concat([local_projections(daily[outcomes], sample, s, horizons) for s in SHOCK_SETS])
        for shock, outcome, h, label in KEY_RESULTS:
            r = irfs[(irfs.shock == shock) & (irfs.outcome == outcome) & (irfs.h == h)]
            if r.empty:
                continue
            coef, se = float(r.coef.iloc[0]), float(r.se.iloc[0])
            rows.append({"check": name, "result": label, "coef": coef, "se": se,
                         "t": coef / se, "meetings": int(r.n.iloc[0])})
    return pd.DataFrame(rows)
