"""Robustness checks: do the headline results survive when one assumption changes?

Each variant is a different set of FOMC meetings. For every variant we rerun the local
projections and record the headline numbers. A result that holds across variants is a
finding; one that only appears in one sample is a property of that sample.
"""

import pandas as pd

from tpshock import config
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
    ("path", "fwd_tp", 20, "Guidance surprise: 5y5y term premium after 20 days"),
    ("path", "fwd_real", 20, "Guidance surprise: 5y5y real forward after 20 days"),
    ("path", "fwd_breakeven", 20, "Guidance surprise: 5y5y breakeven after 20 days"),
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

    # The same checks inside the post-2008 sample, where the guidance results live
    post = shocks.loc["2008":]
    qe = post.index.normalize().isin(pd.to_datetime(config.BALANCE_SHEET_DATES))
    variants["2008-present, excl. balance sheet announcements"] = post[~qe]
    variants["2008-present, excl. 2020"] = post[post.index.year != 2020]
    if "scheduled" in shocks:
        variants["2008-present, scheduled only"] = post[post["scheduled"].astype(bool)]

    return variants


def policy_like_guidance(daily: pd.DataFrame, shocks: pd.DataFrame, horizon: int = 20) -> pd.DataFrame:
    """Post-2008 guidance effect using only guidance surprises where stocks moved the other way.

    The path shock is split in two: meetings where rates and stocks moved opposite ways (looks like
    policy) and the rest (looks like the Fed revealing news about the economy). Both enter one
    regression with the target shock, so each is estimated holding the others fixed.
    """
    post = shocks.loc["2008":].copy()
    same = (post["path"] * post["SP500"]) >= 0
    post["path_policy"] = post["path"].where(~same, 0.0)
    post["path_info"] = post["path"].where(same, 0.0)
    outcomes = sorted({o for s, o, h, _ in KEY_RESULTS if s == "path" and h == horizon and o in daily})
    irf = local_projections(daily[outcomes], post, ["target", "path_policy", "path_info"], [horizon])
    irf = irf[irf.shock == "path_policy"]
    rows = []
    for shock, outcome, h, label in KEY_RESULTS:
        if shock != "path" or h != horizon:
            continue
        r = irf[irf.outcome == outcome]
        if r.empty:
            continue
        coef, se = float(r.coef.iloc[0]), float(r.se.iloc[0])
        rows.append({"check": "2008-present, guidance where stocks fell", "result": label, "coef": coef,
                     "se": se, "t": coef / se, "meetings": int(r.n.iloc[0])})
    return pd.DataFrame(rows)


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
    return pd.concat([pd.DataFrame(rows), policy_like_guidance(daily, shocks)], ignore_index=True)
