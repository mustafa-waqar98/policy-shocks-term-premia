"""Turning raw FOMC-window price changes into economically meaningful shocks.

Three layers, each from a well-known paper:

1. Policy surprise (pc1). The first principal component of surprises in rate
   futures from 1 month to 1 year out. One number per meeting summarising
   "how much more hawkish than expected was this announcement".
   (Jarocinski and Karadi 2020, following Nakamura and Steinsson 2018.)

2. Target vs path (Gurkaynak, Sack and Swanson 2005). Two factors: the
   surprise in the current meeting's rate decision, and a second factor that
   moves expected rates further out but not today's rate. The path factor is
   what forward guidance and the dot plot move.

3. Policy vs information (Jarocinski and Karadi 2020). A hawkish surprise
   should push stocks down. When rates and stocks rise together, the market
   is reading the Fed as saying "the economy is stronger than you thought",
   which is an information shock, not a tightening. Sign restrictions split
   pc1 into a monetary policy shock (MP) and a central bank information
   shock (CBI).
"""

import numpy as np
import pandas as pd

from tpshock import config


def _matlab_pca_sign(coeff: np.ndarray) -> np.ndarray:
    """MATLAB's pca makes the largest-magnitude loading of each component positive."""
    idx = np.abs(coeff).argmax(axis=0)
    signs = np.sign(coeff[idx, np.arange(coeff.shape[1])])
    return coeff * signs


def policy_factor(surprises: pd.DataFrame, rate_vars=None, scale_var: str = "ED4") -> pd.Series:
    """First principal component, scaled to the std of ED4 (so units are % p.a.)."""
    rate_vars = rate_vars or config.JK_RATE_VARS
    X = surprises[rate_vars].to_numpy(dtype=float)
    all_missing = np.isnan(X).all(axis=1)
    X = np.nan_to_num(X, nan=0.0)
    X = X / X.std(axis=0, ddof=1)
    # uncentred PCA via SVD, as in JK's code
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    coeff = _matlab_pca_sign(vt.T)
    score = X @ coeff[:, 0]
    score[all_missing] = np.nan
    pc1 = score / np.nanstd(score, ddof=1) * surprises[scale_var].std(ddof=1)
    return pd.Series(pc1, index=surprises.index, name="pc1")


def target_path(surprises: pd.DataFrame, rate_vars=None) -> pd.DataFrame:
    """Gurkaynak-Sack-Swanson target and path factors.

    target: scaled so a 1 unit move = 1pp surprise in the current meeting's rate (MP1)
    path:   uncorrelated with target, no effect on MP1, scaled to move ED4 one for one
    """
    rate_vars = rate_vars or config.JK_RATE_VARS
    raw = surprises[rate_vars].astype(float).fillna(0.0)
    X = (raw - raw.mean()) / raw.std(ddof=1)
    u, s, _ = np.linalg.svd(X.to_numpy(), full_matrices=False)
    F = u[:, :2] * np.sqrt(len(X) - 1)  # two orthogonal unit-variance factors

    # MP1 loads on F with gamma; target is that direction, path is orthogonal to it
    mp1 = raw["MP1"].to_numpy()
    gamma = np.linalg.lstsq(F, mp1 - mp1.mean(), rcond=None)[0]
    z1 = F @ gamma
    z2 = F @ np.array([-gamma[1], gamma[0]])

    def scale(z, target):
        b = np.polyfit(z, target - target.mean(), 1)[0]
        return z * b

    z1 = scale(z1, mp1)
    z2 = scale(z2, raw["ED4"].to_numpy())
    return pd.DataFrame({"target": z1, "path": z2}, index=surprises.index)


def poor_mans(pc1: pd.Series, stocks: pd.Series) -> pd.DataFrame:
    """Each meeting is all MP (rates and stocks move opposite) or all CBI (same direction)."""
    prod = pc1 * stocks
    # a missing stock move counts as neither (as in JK's code)
    return pd.DataFrame({
        "MP_pm": pc1.where(prod < 0, 0.0),
        "CBI_pm": pc1.where(prod >= 0, 0.0),
    })


def median_rotation(pc1: pd.Series, stocks: pd.Series, w: float = 0.5) -> pd.DataFrame:
    """Both shocks can be present at every meeting.

    Out of all rotations of (pc1, stocks) into two orthogonal shocks that meet
    the sign restrictions, pick the median angle. Port of JK's signrestr_median.m.
    """
    M = np.column_stack([pc1.to_numpy(), stocks.to_numpy()])
    ok = ~np.isnan(M).any(axis=1)
    Q, R = np.linalg.qr(M[ok])
    S = np.diag(np.sign(np.diag(R)))
    Q, R = Q @ S, S @ R
    if R[0, 1] > 0:
        a = (1 - w) * np.arctan(R[0, 1] / R[1, 1]) + w * np.pi / 2
    else:
        a = w * np.arctan(-R[1, 1] / R[0, 1])
    P = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
    D = np.diag([R[0, 0] * np.cos(a), R[0, 0] * np.sin(a)])
    U = np.full(M.shape, np.nan)
    U[ok] = Q @ P @ D
    U = np.nan_to_num(U, nan=0.0)
    return pd.DataFrame(U, index=pc1.index, columns=["MP_median", "CBI_median"])


def build_shocks(surprises: pd.DataFrame, start_year: int = config.JK_START_YEAR) -> pd.DataFrame:
    s = surprises[surprises.index.year >= start_year]
    pc1 = policy_factor(s)
    stocks = s[config.JK_STOCK_VAR]
    out = pd.concat([pc1, stocks.rename("SP500"), poor_mans(pc1, stocks),
                     median_rotation(pc1, stocks), target_path(s)], axis=1)
    out.index.name = "start"
    return out
