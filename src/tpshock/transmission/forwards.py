"""Far-forward rates: where in the future, and in what form, does a Fed shock land?

A 5y5y forward is the 5-year rate that starts 5 years from today. It strips out the
next five years, when the Fed's own policy path dominates, and keeps the far future,
where standard models say monetary policy should barely matter.

We split the nominal 5y5y forward two ways:
  expected rates + term premium     (from the ACM model)
  real rate + breakeven inflation   (from TIPS, the inflation-protected Treasuries)
"""

import pandas as pd


def forward_5y5y(y5, y10):
    """The 5-year rate starting in 5 years, implied by today's 5y and 10y yields.

    Works on numbers, pandas Series or whole columns at once.
    """
    # YOUR TURN: replace the next line with the formula (one line, starts with `return`).
    return (10 * y10 - 5 * y5) / 5


def forward_panel(daily: pd.DataFrame, fred: pd.DataFrame) -> pd.DataFrame:
    """Daily 5y5y forwards. Model-based pieces from ACM, market-based pieces from FRED."""
    out = pd.DataFrame(index=daily.index)
    out["fwd_nominal"] = forward_5y5y(daily["y60"], daily["y120"])
    out["fwd_expected"] = forward_5y5y(daily["rn60"], daily["rn120"])
    out["fwd_tp"] = forward_5y5y(daily["tp60"], daily["tp120"])
    need = {"DGS5", "DGS10", "DFII5", "DFII10"}
    if need <= set(fred.columns):
        f = fred.reindex(daily.index)
        real = forward_5y5y(f["DFII5"], f["DFII10"])
        nominal_par = forward_5y5y(f["DGS5"], f["DGS10"])
        out["fwd_real"] = real
        out["fwd_breakeven"] = nominal_par - real
    return out
