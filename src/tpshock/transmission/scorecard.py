"""Live scorecard for the single-meeting call, one row per meeting since Warsh took over.

The call, as written in Research Notes #1: over the 20 trading days from the close before the
meeting to the close 20 trading days after it, the real 5y5y forward moves in the same direction
the 2-year moved on the meeting day. Everything comes from FRED (DGS2, DFII5, DFII10).

A meeting is a miss if the real forward moves the other way, if it ends the window unchanged
at two decimals, or if the 2-year doesn't move on the day. Meetings where the 2-year moved
less than 3bp are still graded, and flagged, because the backtest says those are close to a
coin flip.
"""

import numpy as np
import pandas as pd

from tpshock.transmission.forwards import forward_5y5y

HORIZON = 20
CUTOFF_BP = 3


def score(fred: pd.DataFrame, meetings: list[str], calls: dict[str, str] | None = None,
          today: pd.Timestamp | None = None) -> pd.DataFrame:
    """One row per meeting: day-0 2y move, 20-day real 5y5y move, and the result so far."""
    calls = calls or {}
    today = pd.Timestamp(today or pd.Timestamp.today()).normalize()
    d = pd.DataFrame({"y2": fred["DGS2"],
                      "real": forward_5y5y(fred["DFII5"], fred["DFII10"])}).dropna()
    idx = d.index
    rows = []
    for m in meetings:
        day = pd.Timestamp(m)
        row = {"meeting": day.date().isoformat(), "note": calls.get(m, ""),
               "d2y_bp": np.nan, "real_bp": np.nan, "window_end": "", "big_move": None,
               "result": "upcoming"}
        i = idx.searchsorted(day)
        if day > today or i >= len(idx) or idx[i] != day:
            if day <= today:
                row["result"] = "waiting for data"
            rows.append(row)
            continue
        d2y = round((d.y2.iloc[i] - d.y2.iloc[i - 1]) * 100, 2)
        row.update(d2y_bp=d2y, big_move=bool(abs(d2y) >= CUTOFF_BP))
        if i + HORIZON >= len(idx):
            row["window_end"] = ""
            row["result"] = f"open ({len(idx) - 1 - i} of {HORIZON} days)"
            rows.append(row)
            continue
        real = round((d.real.iloc[i + HORIZON] - d.real.iloc[i - 1]) * 100, 2)
        row.update(real_bp=real, window_end=idx[i + HORIZON].date().isoformat())
        if d2y == 0:
            row["result"] = "miss (2y unchanged)"
        elif real == 0:
            row["result"] = "miss (real forward unchanged)"
        else:
            row["result"] = "hit" if np.sign(real) == np.sign(d2y) else "miss"
        rows.append(row)
    return pd.DataFrame(rows)


def tally(card: pd.DataFrame) -> dict:
    """Running totals over the meetings whose window has closed."""
    done = card[card.result.str.startswith(("hit", "miss"))]
    big = done[done.big_move.astype(bool)]
    return {"scored": int(len(done)), "hits": int((done.result == "hit").sum()),
            "scored_big": int(len(big)), "hits_big": int((big.result == "hit").sum())}
