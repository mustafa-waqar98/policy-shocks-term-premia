"""Treasury quarterly refunding announcements as supply news.

Each February, May, August and November the Treasury announces the sizes of the three
"refunding" auctions (3-, 10- and 30-year) in its quarterly refunding statement, released at
8:30am ET. Those sizes are announced the same morning as the statement, so in the Treasury's
auction file (fiscaldata auctions_query) the refunding 3/10/30-year auctions carry the statement
date as their announcement date. That gives the event dates and the announced sizes from one
public file, with no hand-coding.

The news in each statement is measured as the change in announced coupon supply against the
previous refunding, in 10-year equivalents (each auction's size weighted by its maturity / 10),
so a $1bn larger 30-year counts three times a $1bn larger 10-year.
Caveat: Treasury often signals size changes a quarter ahead, so part of each change is expected.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from tpshock import config
from tpshock.data.download import fetch

log = logging.getLogger("tpshock")

AUCTIONS_URL = ("https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/od/"
                "auctions_query?fields=cusip,security_type,security_term,announcemt_date,auction_date,"
                "offering_amt,reopening&filter=auction_date:gte:2007-01-01,security_type:in:(Note,Bond)"
                "&sort=auction_date&page[size]=10000&page[number]={page}")
RAW = config.RAW / "supply" / "auctions"
REFUNDING_TERMS = {"3-Year": 3, "10-Year": 10, "30-Year": 30}
REFUNDING_MONTHS = {2, 5, 8, 11}


def download(dest: Path = RAW) -> None:
    first = fetch(AUCTIONS_URL.format(page=1), dest / "page_001.json", deadline=120)
    pages = json.loads(first.read_text())["meta"]["total-pages"]
    for n in range(2, pages + 1):
        fetch(AUCTIONS_URL.format(page=n), dest / f"page_{n:03d}.json", deadline=120)
    log.info("auctions: %d pages", pages)


def load_auctions(src: Path = RAW) -> pd.DataFrame:
    recs = []
    for f in sorted(src.glob("page_*.json")):
        recs += json.loads(f.read_text())["data"]
    df = pd.DataFrame(recs)
    df["announced"] = pd.to_datetime(df.announcemt_date, errors="coerce")
    df["auction"] = pd.to_datetime(df.auction_date, errors="coerce")
    df["size_bn"] = pd.to_numeric(df.offering_amt, errors="coerce") / 1e9
    return df.dropna(subset=["announced", "auction", "size_bn"])


def refunding_events(auctions: pd.DataFrame) -> pd.DataFrame:
    """One row per refunding statement: date, the three announced sizes, and the change in
    announced coupon supply (10-year equivalents) against the previous refunding."""
    a = auctions[auctions.security_term.isin(REFUNDING_TERMS)
                 & auctions.auction.dt.month.isin(REFUNDING_MONTHS)].copy()
    a["quarter"] = a.auction.dt.to_period("Q")
    rows = []
    for _q, g in a.groupby("quarter"):
        ten = g[(g.security_term == "10-Year") & (g.auction.dt.day <= 20)]
        if ten.empty:
            continue
        date = ten.announced.min()
        # only auctions announced the same morning as the refunding 10-year belong to the refunding;
        # this drops the February 30-year TIPS, which the file also labels "30-Year"
        g = g[g.announced == date]
        sizes = {t: g.loc[g.security_term == t, "size_bn"].sum() for t in REFUNDING_TERMS}
        rows.append({"date": date, **{f"s{REFUNDING_TERMS[t]}": v for t, v in sizes.items()}})
    ev = pd.DataFrame(rows).set_index("date").sort_index()
    ev["tenyr_eq_bn"] = sum(ev[f"s{m}"] * m / 10 for m in REFUNDING_TERMS.values())
    ev["d_tenyr_eq_bn"] = ev.tenyr_eq_bn.diff()
    ev["d_total_bn"] = (ev.s3 + ev.s10 + ev.s30).diff()
    return ev


def overlaps(events: pd.DataFrame, fomc_days: pd.DatetimeIndex, window: int = 1) -> pd.Series:
    """True where an FOMC announcement falls on the refunding day or within `window` business
    days after it, so the event-window move can't be told apart from the Fed's."""
    f = pd.DatetimeIndex(fomc_days).normalize()
    out = []
    for d in events.index:
        days = pd.bdate_range(d, periods=window + 1)
        out.append(bool(f.isin(days).any()))
    return pd.Series(out, index=events.index)


def event_study(daily: pd.DataFrame, events: pd.DataFrame, outcomes, horizons=(0, 1, 5)) -> pd.DataFrame:
    """Regress the move from the close before each statement to h days after on the announced
    change in coupon supply. Coefficients in bp per $10bn of 10-year equivalents."""
    from tpshock.transmission.lp import ols
    idx = daily.index
    rows = []
    for h in horizons:
        pos = idx.searchsorted(events.index.normalize())
        ok = (pos >= 1) & (pos + h < len(idx))
        ev, pos = events[ok], pos[ok]
        for col in outcomes:
            chg = (daily[col].to_numpy()[pos + h] - daily[col].to_numpy()[pos - 1]) * 100
            x = ev.d_tenyr_eq_bn.to_numpy() / 10
            m = np.isfinite(chg) & np.isfinite(x)
            if m.sum() < 15:
                continue
            X = np.column_stack([np.ones(m.sum()), x[m]])
            b, se, r2 = ols(chg[m], X, lags=0)
            rows.append({"outcome": col, "h": h, "coef": b[1], "se": se[1], "t": b[1] / se[1],
                         "mean_move": float(chg[m].mean()), "n": int(m.sum()), "r2": r2})
    return pd.DataFrame(rows)


def window_overlaps(events: pd.DataFrame, fomc_days, before: int, after: int) -> pd.Series:
    f = pd.DatetimeIndex(fomc_days).normalize()
    return pd.Series([bool(f.isin(pd.bdate_range(d - pd.offsets.BDay(before), periods=before + after + 1)).any())
                      for d in events.index], index=events.index)


def analyse(daily: pd.DataFrame, events: pd.DataFrame, fomc_days, start: str = "2009") -> tuple[pd.DataFrame, dict]:
    """Event table and results for the dashboard.

    Statement day: close before the 8:30am statement to its close; FOMC within a day excluded.
    Three-day window: from the close before Monday's borrowing estimate to Wednesday's close,
    which also captures the financing estimate; FOMC anywhere in Monday to Thursday excluded.
    Coefficients in bp per $10bn of announced 10-year-equivalent supply.
    From 2009, after the 3-year's return and the 30-year's move to quarterly issuance in 2008-09.
    """
    from tpshock.transmission.lp import ols
    ev = events.loc[start:].copy()
    ev["fomc_overlap"] = overlaps(ev, fomc_days, window=1)
    ev["fomc_overlap_3d"] = window_overlaps(ev, fomc_days, before=2, after=1)
    pos = daily.index.searchsorted(ev.index)
    for col, name in [("y120", "d_y"), ("rn120", "d_rn"), ("tp120", "d_tp")]:
        v = daily[col].to_numpy()
        ev[name] = (v[pos] - v[pos - 1]) * 100
        ev[name + "_3d"] = (v[pos] - v[pos - 3]) * 100
    out = {"n_events": int(len(ev))}
    for key, mask, suffix in [("day", ~ev.fomc_overlap, ""), ("three_day", ~ev.fomc_overlap_3d, "_3d")]:
        e = ev[mask]
        x = e.d_tenyr_eq_bn.to_numpy() / 10
        res = {"n": int(len(e)), "n_changes": int((e.d_tenyr_eq_bn != 0).sum())}
        for name in ["d_y", "d_rn", "d_tp"]:
            y = e[name + suffix].to_numpy()
            b, se, _ = ols(y, np.column_stack([np.ones(len(y)), x]), lags=0)
            res[name] = {"coef": round(float(b[1]), 2), "se": round(float(se[1]), 2),
                         "t": round(float(b[1] / se[1]), 2)}
        out[key] = res
    # without the single largest size change, so one quarter can't drive the slope
    e = ev[~ev.fomc_overlap]
    e = e.drop(e.d_tenyr_eq_bn.abs().idxmax())
    b, se, _ = ols(e.d_y.to_numpy(), np.column_stack([np.ones(len(e)), e.d_tenyr_eq_bn.to_numpy() / 10]), lags=0)
    out["day_ex_largest"] = {"coef": round(float(b[1]), 2), "t": round(float(b[1] / se[1]), 2),
                             "dropped": ev.loc[~ev.fomc_overlap, "d_tenyr_eq_bn"].abs().idxmax().strftime("%b %Y")}
    nz = ev.loc[~ev.fomc_overlap & (ev.d_tenyr_eq_bn != 0), "d_tenyr_eq_bn"].abs()
    out["median_change_bn"] = round(float(nz.median()), 1) if len(nz) else None
    return ev, out
