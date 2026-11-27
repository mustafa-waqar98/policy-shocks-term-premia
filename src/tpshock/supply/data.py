"""Security-level Treasury debt and Fed holdings, the raw material for the supply measure.

Two public sources, both keyed by CUSIP:
  - Treasury's Monthly Statement of the Public Debt (MSPD), table 3: every marketable bill,
    note, bond, TIPS and FRN outstanding at each month-end, since January 2001
  - NY Fed SOMA holdings: what the Fed owns of each security, weekly since July 2003

Matching the two gives what the private market has to hold, security by security.
"""

import json
import logging
import re
from pathlib import Path

import pandas as pd

from tpshock import config
from tpshock.data.download import fetch

log = logging.getLogger("tpshock")

MSPD_URL = ("https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/debt/mspd/"
            "mspd_table_3_market?fields=record_date,security_type_desc,security_class1_desc,"
            "security_class2_desc,maturity_date,outstanding_amt&sort=record_date"
            "&page[size]=10000&page[number]={page}")
SOMA_SUMMARY_URL = "https://markets.newyorkfed.org/api/soma/summary.json"
SOMA_ASOF_URL = "https://markets.newyorkfed.org/api/soma/tsy/get/asof/{date}.json"

RAW = config.RAW / "supply"
CUSIP = re.compile(r"^[0-9]{3}[0-9A-Z]{5}[0-9]$")


def download_mspd(dest: Path = RAW / "mspd") -> None:
    """All pages of MSPD table 3. Re-downloaded each time (about 16 pages)."""
    first = fetch(MSPD_URL.format(page=1), dest / "page_001.json", deadline=120)
    pages = json.loads(first.read_text())["meta"]["total-pages"]
    for n in range(2, pages + 1):
        fetch(MSPD_URL.format(page=n), dest / f"page_{n:03d}.json", deadline=120)
    log.info("MSPD: %d pages", pages)


def soma_month_ends(summary: Path) -> list[str]:
    """The last weekly SOMA date in each month."""
    body = json.loads(summary.read_text())["soma"]
    recs = body["summary"] if isinstance(body, dict) else body
    dates = pd.to_datetime([r["asOfDate"] for r in recs])
    s = pd.Series(dates, index=dates).sort_index()
    return [d.date().isoformat() for d in s.groupby(s.index.to_period("M")).max()]


def download_soma(dest: Path = RAW / "soma") -> None:
    """Fed holdings by CUSIP at each month's last weekly date. Months already on disk are skipped,
    except the latest, which is refreshed."""
    summary = fetch(SOMA_SUMMARY_URL, dest / "summary.json")
    dates = soma_month_ends(summary)
    for i, d in enumerate(dates):
        f = dest / f"{d}.json"
        if f.exists() and i < len(dates) - 2:
            continue
        fetch(SOMA_ASOF_URL.format(date=d), f)
    log.info("SOMA: %d month-end snapshots", len(dates))


def load_mspd(src: Path = RAW / "mspd") -> pd.DataFrame:
    """One row per security per month-end: date, type, cusip, maturity, outstanding ($bn)."""
    recs = []
    for f in sorted(src.glob("page_*.json")):
        recs += json.loads(f.read_text())["data"]
    df = pd.DataFrame(recs)
    df = df[df.security_type_desc.str.contains("Marketable", na=False)]
    df = df[df.security_class2_desc.fillna("").str.strip().str.match(CUSIP)
            & df.maturity_date.notna() & (df.maturity_date != "null")]
    out = pd.DataFrame({
        "date": pd.to_datetime(df.record_date),
        "type": df.security_class1_desc.str.strip(),
        "cusip": df.security_class2_desc.str.strip(),
        "maturity": pd.to_datetime(df.maturity_date, errors="coerce"),
        "outstanding": pd.to_numeric(df.outstanding_amt, errors="coerce") / 1000,  # $mn -> $bn
    }).dropna()
    # a security can appear on more than one line (e.g. reopenings): add them up
    return out.groupby(["date", "type", "cusip", "maturity"], as_index=False).outstanding.sum()


def load_soma(src: Path = RAW / "soma") -> pd.DataFrame:
    """One row per security per month: month, cusip, Fed par holdings ($bn), excluding TIPS
    inflation compensation (MSPD reports TIPS at par too)."""
    rows = []
    for f in sorted(src.glob("????-??-??.json")):
        for h in json.loads(f.read_text())["soma"]["holdings"]:
            rows.append((f.stem, h["cusip"], float(h["parValue"] or 0) / 1e9))
    df = pd.DataFrame(rows, columns=["asof", "cusip", "fed"])
    df["month"] = pd.to_datetime(df["asof"]).dt.to_period("M")
    return df.groupby(["month", "cusip"], as_index=False).fed.sum()
