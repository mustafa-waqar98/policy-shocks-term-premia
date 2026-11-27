"""Treasury supply step, run on its own:

    PYTHONPATH=src python -m tpshock.supply fetch    # download MSPD, SOMA and GDP
    PYTHONPATH=src python -m tpshock.supply build    # write data/reference/supply_monthly.csv

The data is monthly (refunding announcements quarterly), so this runs by hand, about once a month,
not in the daily refresh. Its outputs are small and live in data/reference/, which is in git, so
the dashboard always has them.
"""

import json
import logging
import sys

import pandas as pd

from tpshock import config
from tpshock.data import fred
from tpshock.supply import data, measure, refunding

log = logging.getLogger("tpshock")
REF = config.ROOT / "data" / "reference"


def fetch():
    for name, fn in [("MSPD", data.download_mspd), ("SOMA", data.download_soma),
                     ("FRED GDP", lambda: fred.download("GDP")), ("auctions", refunding.download)]:
        try:
            fn()
            log.info("fetched %s", name)
        except Exception as exc:
            log.warning("could not fetch %s: %s", name, exc)


def build():
    holdings = measure.private_holdings(data.load_mspd(), data.load_soma())
    out = measure.monthly_supply(holdings, fred.load("GDP"))
    REF.mkdir(parents=True, exist_ok=True)
    out.round(5).to_csv(REF / "supply_monthly.csv")
    acm, shocks = config.PROCESSED / "acm_daily.csv", config.PROCESSED / "shocks.csv"
    if (refunding.RAW / "page_001.json").exists() and acm.exists() and shocks.exists():
        daily = pd.read_csv(acm, index_col=0, parse_dates=True)
        fomc_days = pd.read_csv(shocks, index_col=0, parse_dates=True).index
        ev, res = refunding.analyse(daily, refunding.refunding_events(refunding.load_auctions()), fomc_days)
        ev.round(3).to_csv(REF / "refunding_events.csv")
        (REF / "refunding_results.json").write_text(json.dumps(res, indent=2))
        log.info("refunding: %d statements, 10y %+.1fbp per $10bn (t %.1f) on the day",
                 res["n_events"], res["day"]["d_y"]["coef"], res["day"]["d_y"]["t"])
    last = out.dropna().iloc[-1]
    log.info("supply built through %s: private debt %.0fbn, WAM %.1f years, Fed holds %.0f%% of duration",
             out.index[-1].date(), last.private_bn, last.wam_years, 100 * last.fed_share_of_duration)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    steps = sys.argv[1:] or ["fetch", "build"]
    for s in steps:
        {"fetch": fetch, "build": build}[s]()
