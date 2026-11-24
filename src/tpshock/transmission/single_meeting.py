"""Does the average result show up at a single meeting, and does it beat a no-meeting base rate?

Local projections give average responses across many meetings. A call about one meeting needs
something stronger: a hit rate after FOMC meetings that beats the hit rate over comparable
windows with no meeting in them. Otherwise a "call" can just be scoring how the market always
behaves.

Setup, all from FRED so anyone can reproduce it:
  - day 0 is the meeting day; the 2y yield change on day 0 (DGS2) stands in for the surprise
  - the outcome window runs from the close before day 0 to the close 20 trading days later
  - control windows: every other start day whose window contains no FOMC meeting at all
  - neighbouring control windows share most of their days, so the control rate's uncertainty
    comes from a moving-block bootstrap (blocks of 21 trading days); p-values are two-sided
"""

from math import erf, sqrt

import numpy as np
import pandas as pd

from tpshock.transmission.forwards import forward_5y5y

HORIZON = 20
CUTOFFS_BP = [0, 2, 3, 4, 5]   # minimum absolute 2y move on day 0; 0 means "any nonzero move"


def compare(hit: np.ndarray, valid: np.ndarray, is_fomc: np.ndarray, clean: np.ndarray,
            n_boot: int = 4000, block: int = 21, seed: int = 0) -> dict:
    """Hit rate after meetings vs hit rate in meeting-free windows, with a block-bootstrap z-test."""
    rng = np.random.default_rng(seed)
    f, c = valid & is_fomc, valid & clean
    hf, hc = hit[f], hit[c]
    pf, pc = float(hf.mean()), float(hc.mean())
    cidx = np.flatnonzero(c)
    nb = max(1, len(cidx) // block)
    starts = rng.integers(0, max(1, len(cidx) - block), size=(n_boot, nb))
    picks = (starts[:, :, None] + np.arange(block)).reshape(n_boot, -1).clip(0, len(cidx) - 1)
    ctrl = hit[cidx[picks]].mean(axis=1)
    fomc = rng.binomial(len(hf), pf, size=n_boot) / len(hf)
    se = float((fomc - ctrl).std())
    z = (pf - pc) / se if se > 0 else np.nan
    p = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2)))) if np.isfinite(z) else np.nan
    return {"meetings": int(len(hf)), "hits": int(hf.sum()), "fomc_rate": pf,
            "control_rate": pc, "control_windows": int(len(hc)), "z": z, "p_two_sided": p}


def run(fred: pd.DataFrame, daily: pd.DataFrame, shocks: pd.DataFrame, start: str = "2008") -> pd.DataFrame:
    need = {"DGS2", "DGS5", "DGS10", "DFII5", "DFII10"}
    if not need <= set(fred.columns):
        return pd.DataFrame()
    d = pd.DataFrame({
        "y2": fred["DGS2"],
        "real": forward_5y5y(fred["DFII5"], fred["DFII10"]),
        "nominal": forward_5y5y(fred["DGS5"], fred["DGS10"]),
    }).dropna().loc[start:]
    d["breakeven"] = d["nominal"] - d["real"]
    d = d.join(daily[["tp120"]], how="left")
    idx, H = d.index, HORIZON

    meetings = shocks.loc[start:].index.normalize()
    pos = idx.searchsorted(meetings)
    pos = np.unique(pos[(pos >= 1) & (pos + H < len(idx))])
    is_fomc = np.zeros(len(idx), bool)
    is_fomc[pos] = True
    clean = np.ones(len(idx), bool)            # no FOMC day anywhere in [t-1, t+H]
    for p in pos:
        clean[max(p - H, 0):p + 2] = False

    dy2 = (d["y2"] - d["y2"].shift(1)) * 100
    def change(col):
        return (d[col].shift(-H) - d[col].shift(1)) * 100
    real, be, tp = change("real"), change("breakeven"), change("tp120")
    base_valid = (dy2.notna() & real.notna() & be.notna()).to_numpy()

    def same_sign(x):
        return ((np.sign(x) == np.sign(dy2)) & (x != 0)).to_numpy()

    tests = [("Real 5y5y moves more than breakeven 5y5y", (real.abs() > be.abs()).to_numpy(), None)]
    for cut in CUTOFFS_BP:
        label = "any nonzero move" if cut == 0 else f"2y moved at least {cut}bp"
        tests.append((f"Real 5y5y moves with the day-0 2y ({label})", same_sign(real), cut))
    for cut in (0, 3):
        label = "any nonzero move" if cut == 0 else f"2y moved at least {cut}bp"
        tests.append((f"10y term premium moves with the day-0 2y ({label})", same_sign(tp), cut))

    rows = []
    for name, hit, cut in tests:
        valid = base_valid.copy()
        if cut is not None:
            valid &= (dy2.abs() >= max(cut, 1e-9)).to_numpy()
        if "term premium" in name:
            valid &= tp.notna().to_numpy()
        rows.append({"test": name, **compare(hit, valid, is_fomc, clean)})
    out = pd.DataFrame(rows)
    out.attrs["zero_2y_moves_on_meeting_days"] = int(((dy2 == 0).to_numpy() & is_fomc).sum())
    return out
