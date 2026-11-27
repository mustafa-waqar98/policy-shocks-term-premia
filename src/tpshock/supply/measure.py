"""How much duration the private market has to hold, month by month.

For each month-end and each marketable security: outstanding (MSPD) minus what the Fed holds
(SOMA), matched by CUSIP. Then three summaries of the private remainder:

  - debt_years_gdp: sum of amount x remaining maturity in years, over nominal GDP. This is the
    maturity-weighted debt-to-GDP of Greenwood and Vayanos (2014), the standard supply measure.
  - tenyr_eq_gdp: the same in 10-year equivalents, weighting each security by the duration of
    a zero-coupon bond of its maturity relative to a 10-year one, at a flat 4% yield. It counts
    a 30-year bond as more than three 10-years, which debt-years understates.
  - wam_years: average remaining maturity of privately held debt, in years.

Bills, notes, bonds, TIPS and FRNs are all included. FRNs reset weekly, so they count as
bills (maturity of a week). TIPS are at par.
"""

import numpy as np
import pandas as pd

FLAT_YIELD = 0.04
FRN_YEARS = 7 / 365


def duration_weight(years, y: float = FLAT_YIELD):
    """Duration of a zero of this maturity relative to a 10-year zero, both at yield y,
    continuously compounded. A zero's duration is its maturity, so this is maturity / 10
    after discounting the price: the dollar duration ratio."""
    years = np.asarray(years, float)
    return years * np.exp(-y * years) / (10 * np.exp(-y * 10))


def private_holdings(mspd: pd.DataFrame, soma: pd.DataFrame) -> pd.DataFrame:
    """MSPD rows with the Fed's holding of each security subtracted (from mid-2003 on)."""
    df = mspd.copy()
    df["month"] = df.date.dt.to_period("M")
    df = df.merge(soma, on=["month", "cusip"], how="left")
    first_soma = soma.month.min() if len(soma) else None
    df["fed"] = df.fed.fillna(0.0)
    if first_soma is not None:
        df.loc[df.month < first_soma, "fed"] = np.nan   # no SOMA data yet: private unknown
    df["private"] = (df.outstanding - df.fed).clip(lower=0)
    years = (df.maturity - df.date).dt.days / 365.25
    df["years"] = np.where(df.type.str.contains("Floating", case=False), FRN_YEARS, years.clip(lower=0))
    return df


def monthly_supply(holdings: pd.DataFrame, gdp: pd.Series) -> pd.DataFrame:
    """One row per month-end. gdp: nominal GDP, $bn at an annual rate (FRED GDP)."""
    h = holdings
    w = duration_weight(h.years)
    g = h.assign(dy_tot=h.outstanding * h.years, dy_priv=h.private * h.years,
                 te_tot=h.outstanding * w, te_priv=h.private * w).groupby("date")
    out = pd.DataFrame({
        "marketable_bn": g.outstanding.sum(),
        "private_bn": g.private.sum(min_count=1),
        "fed_bn": g.fed.sum(min_count=1),
        "debt_years_total": g.dy_tot.sum(),
        "debt_years_private": g.dy_priv.sum(min_count=1),
        "tenyr_eq_total": g.te_tot.sum(),
        "tenyr_eq_private": g.te_priv.sum(min_count=1),
    })
    out["wam_years"] = out.debt_years_private / out.private_bn
    # GDP is quarterly: carry each quarter's value through its months
    gdp_m = gdp.resample("ME").last().ffill().reindex(out.index, method="ffill")
    for c in ("debt_years_total", "debt_years_private", "tenyr_eq_total", "tenyr_eq_private"):
        out[c + "_gdp"] = out[c] / gdp_m
    out["fed_share_of_duration"] = 1 - out.tenyr_eq_private / out.tenyr_eq_total
    return out
