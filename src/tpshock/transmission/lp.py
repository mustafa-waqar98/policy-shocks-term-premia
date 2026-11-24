"""Linking shocks to the curve.

Event study: on each FOMC day, how much of the move in yields came from the
expected rate path and how much from the term premium?

Local projections (Jorda 2005): for each horizon h, regress the change in an
outcome from the day before the meeting to h trading days after on the
shocks. The slope at each h traces out the impulse response. No VAR needed,
and each horizon is just one OLS regression, so it is easy to inspect.
"""

import numpy as np
import pandas as pd


def newey_west(X: np.ndarray, resid: np.ndarray, lags: int) -> np.ndarray:
    """HAC covariance of OLS coefficients (Bartlett kernel)."""
    n = X.shape[0]
    xu = X * resid[:, None]
    S = xu.T @ xu
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        G = xu[L:].T @ xu[:-L]
        S += w * (G + G.T)
    XtX_inv = np.linalg.inv(X.T @ X)
    return XtX_inv @ S @ XtX_inv * n / (n - X.shape[1])


def ols(y: np.ndarray, X: np.ndarray, lags: int = 0):
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    V = newey_west(X, e, lags)
    r2 = 1 - (e @ e) / ((y - y.mean()) @ (y - y.mean()))
    return b, np.sqrt(np.diag(V)), r2


def align_events(event_times: pd.DatetimeIndex, trading_days: pd.DatetimeIndex) -> np.ndarray:
    """Position of the first trading day on or after each announcement."""
    return trading_days.searchsorted(event_times.normalize())


def event_changes(daily: pd.DataFrame, shocks: pd.DataFrame, h: int = 0) -> pd.DataFrame:
    """Change in each daily series from the close before the event to h days after."""
    pos = align_events(shocks.index, daily.index)
    valid = (pos >= 1) & (pos + h < len(daily))
    pos, sh = pos[valid], shocks[valid]
    before = daily.iloc[pos - 1].to_numpy()
    after = daily.iloc[pos + h].to_numpy()
    out = pd.DataFrame(after - before, index=sh.index, columns=daily.columns)
    return out


def local_projections(daily: pd.DataFrame, shocks: pd.DataFrame, shock_cols, horizons,
                      scale: float = 0.10, units: float = 100.0) -> pd.DataFrame:
    """Impulse responses of every column in `daily` to each shock.

    scale: shock size to report (0.10 = a 10bp surprise, since shocks are in %).
    units: multiply outcome changes (100 turns percent into basis points).
    Returns a long table: outcome, shock, h, coef, se, n.
    """
    rows = []
    for h in horizons:
        chg = event_changes(daily, shocks, h) * units
        sh = shocks.loc[chg.index, shock_cols]
        lags = int(np.ceil((h + 1) / 30))  # meetings ~30 trading days apart, windows overlap
        for col in daily.columns:
            ok = chg[col].notna() & sh.notna().all(axis=1)
            if ok.sum() < 30:
                continue
            X = np.column_stack([np.ones(ok.sum()), sh[ok].to_numpy()])
            b, se, _ = ols(chg.loc[ok, col].to_numpy(), X, lags)
            for i, s in enumerate(shock_cols):
                rows.append({"outcome": col, "shock": s, "h": h,
                             "coef": b[i + 1] * scale, "se": se[i + 1] * scale,
                             "n": int(ok.sum())})
    return pd.DataFrame(rows)


def fomc_day_table(daily_acm: pd.DataFrame, shocks: pd.DataFrame, maturity: int = 120) -> pd.DataFrame:
    """Per-meeting split of the yield move into expectations and term premium (bp)."""
    cols = [f"y{maturity}", f"rn{maturity}", f"tp{maturity}"]
    chg = event_changes(daily_acm[cols], shocks, 0) * 100
    chg.columns = ["d_yield", "d_expectations", "d_term_premium"]
    return chg.join(shocks)
