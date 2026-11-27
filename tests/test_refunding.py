import numpy as np
import pandas as pd

from tpshock.supply.refunding import analyse, refunding_events


def _auctions():
    rows = []
    for stmt, s10, s30 in ([("2019-01-30", 27, 19), ("2019-05-01", 27, 19), ("2019-07-31", 30, 22),
                               ("2019-10-30", 30, 22)]):
        d = pd.Timestamp(stmt)
        for term, size, lag in [("3-Year", 38, 6), ("10-Year", s10, 7), ("30-Year", s30, 8)]:
            rows.append({"security_term": term, "announced": d, "auction": d + pd.Timedelta(days=lag), "size_bn": size})
    # a 30-year TIPS in the refunding month, announced on another day: must be ignored
    rows.append({"security_term": "30-Year", "announced": pd.Timestamp("2019-02-14"),
                 "auction": pd.Timestamp("2019-02-21"), "size_bn": 7})
    return pd.DataFrame(rows)


def test_events_use_only_same_morning_auctions():
    ev = refunding_events(_auctions())
    assert list(ev.index.strftime("%Y-%m-%d")) == ["2019-01-30", "2019-05-01", "2019-07-31", "2019-10-30"]
    assert ev.s30.iloc[0] == 19                       # TIPS not added in
    assert np.isclose(ev.d_tenyr_eq_bn.iloc[2], 3 * 10 / 10 + 3 * 30 / 10)   # +3bn 10y and +3bn 30y


def test_analyse_finds_a_planted_supply_effect():
    days = pd.bdate_range("2010-01-01", "2020-12-31")
    rng = np.random.default_rng(0)
    y = pd.Series(4 + np.cumsum(rng.normal(0, 0.01, len(days))), index=days)
    stmts = pd.bdate_range("2010-02-03", "2020-11-04", freq="13W-WED")
    change = rng.choice([-10.0, 0.0, 0.0, 10.0], size=len(stmts))
    for d, c in zip(stmts, change, strict=True):
        y.loc[d:] += 0.02 * c / 10                     # +2bp per $10bn on the day
    daily = pd.DataFrame({"y120": y, "rn120": y * 0, "tp120": y})
    ev = pd.DataFrame({"d_tenyr_eq_bn": change}, index=stmts)
    out, res = analyse(daily, ev, fomc_days=pd.DatetimeIndex([]), start="2010")
    assert abs(res["day"]["d_y"]["coef"] - 2.0) < 0.5
    assert not out.fomc_overlap.any()
