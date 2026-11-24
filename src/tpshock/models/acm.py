"""Adrian, Crump and Moench (2013), "Pricing the term structure with linear regressions".

The idea in one paragraph: a 10y yield is the average short rate the market
expects over 10 years plus a term premium, the extra compensation for holding
duration risk. Neither piece is observable. ACM back them out with three OLS
regressions instead of the usual hard maximum-likelihood estimation:

  1. Factors. Take the first K principal components of the yield curve (level,
     slope, curvature, ...) and fit a VAR(1) to them. This tells us how the
     curve is expected to evolve, and gives the factor shocks v_{t+1}.
  2. Returns. Regress one-month excess returns on a range of bonds on
     a constant, the factor shocks v_{t+1} and the lagged factors X_t.
     The loading on v_{t+1} is each bond's exposure to each risk (beta).
     The loading on X_t says how expected excess returns move with the state.
  3. Prices of risk. A cross-sectional regression of those return
     coefficients on the betas gives the market prices of risk
     lambda_0 (constant part) and lambda_1 (time-varying part).

With the VAR and the prices of risk we can price every bond. Pricing them
again with the prices of risk set to zero gives "risk-neutral" yields, i.e.
the pure expected short-rate path. Term premium = fitted yield - risk-neutral
yield.

Units inside the model: decimals per month (log prices, monthly short rate).
Inputs and outputs: percent per year, which is what everyone quotes.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ACMResult:
    mean: np.ndarray        # yield means used for PCA (M,)
    loadings: np.ndarray    # PCA weights (M, K)
    mu: np.ndarray          # VAR intercept (K,)
    phi: np.ndarray         # VAR slope (K, K)
    sigma: np.ndarray       # VAR shock covariance (K, K)
    sigma2_e: float         # return pricing-error variance
    lambda0: np.ndarray     # constant price of risk (K,)
    lambda1: np.ndarray     # time-varying price of risk (K, K)
    delta0: float           # short rate = delta0 + delta1' X
    delta1: np.ndarray
    A: np.ndarray           # log price intercepts, index n = 0..N
    B: np.ndarray           # log price loadings, shape (N+1, K)
    A_rn: np.ndarray        # same under risk neutrality
    B_rn: np.ndarray
    maturities: list        # yield maturities (months) used as input
    beta: np.ndarray        # return exposures (K, N_returns)
    return_maturities: list


def kilian_bias_correct(X, mu, phi, V, n_boot=500, seed=0):
    """Small-sample bias correction for a VAR(1) slope (Kilian 1998).

    OLS understates persistence in short samples. Simulate many fake histories from the
    estimated VAR, re-estimate each, and measure how far OLS falls short on average. Then
    add that shortfall back. If the corrected VAR is explosive, shrink the correction until
    it is not. The intercept is reset so the long-run mean of the factors is unchanged.
    """
    rng = np.random.default_rng(seed)
    T, k = X.shape
    draws = V[rng.integers(0, len(V), size=(n_boot, T - 1))]      # resampled shocks
    sims = np.empty((n_boot, T, k))
    sims[:, 0] = X[0]
    for t in range(T - 1):
        sims[:, t + 1] = mu + sims[:, t] @ phi.T + draws[:, t]
    phis = np.empty((n_boot, k, k))
    for b in range(n_boot):
        Z = np.column_stack([np.ones(T - 1), sims[b, :-1]])
        phis[b] = np.linalg.lstsq(Z, sims[b, 1:], rcond=None)[0][1:].T
    bias = phis.mean(axis=0) - phi
    shrink = 1.0
    phi_c = phi - bias
    while np.abs(np.linalg.eigvals(phi_c)).max() >= 1 and shrink > 0:
        shrink -= 0.01
        phi_c = phi - shrink * bias
    mu_c = (np.eye(k) - phi_c) @ X.mean(axis=0)
    return mu_c, phi_c


class ACM:
    def __init__(self, n_factors: int = 5, return_maturities=None, max_maturity: int = 120,
                 bias_correct: bool = False):
        self.bias_correct = bias_correct
        self.k = n_factors
        self.return_maturities = return_maturities or list(range(6, 121, 6))
        self.max_maturity = max_maturity
        self.result: ACMResult | None = None

    # ---- estimation -------------------------------------------------------

    def fit(self, yields: pd.DataFrame, short_rate: pd.Series | None = None) -> "ACM":
        """yields: month-end zero yields in percent, columns = maturities 1..N months."""
        mats = list(yields.columns)
        assert mats[0] == 1, "need the 1-month yield as the short rate"
        assert all(m in mats for m in self.return_maturities)
        Y = yields.to_numpy() / 100.0  # decimal per year
        T = Y.shape[0]
        k = self.k

        # Log prices: p_t^(n) = -(n/12) * y_t^(n)
        n_years = np.array(mats) / 12.0
        logp = -Y * n_years[None, :]
        if short_rate is None:
            rf = Y[:, 0] / 12.0  # 1-month yield read off the fitted curve
        else:
            rf = short_rate.to_numpy() / 100.0 / 12.0  # actual T-bill rate

        # Step 1: principal components and VAR(1)
        mean = Y.mean(axis=0)
        cov = np.cov(Y - mean, rowvar=False)
        eigval, eigvec = np.linalg.eigh(cov)
        W = eigvec[:, np.argsort(eigval)[::-1][:k]]
        # sign convention: first factor loads positively on the 10y (a "level")
        W = W * np.sign(W[-1, :])
        X = (Y - mean) @ W  # (T, K)

        Z = np.column_stack([np.ones(T - 1), X[:-1]])
        coef = np.linalg.lstsq(Z, X[1:], rcond=None)[0]
        mu, phi = coef[0], coef[1:].T
        V = X[1:] - Z @ coef            # factor shocks v_{t+1}, (T-1, K)
        sigma = V.T @ V / (T - 1)

        # Step 2: excess return regressions
        col = {m: i for i, m in enumerate(mats)}
        rx = np.column_stack([
            logp[1:, col[n - 1]] - logp[:-1, col[n]] - rf[:-1]
            for n in self.return_maturities
        ])  # (T-1, N)
        R = np.column_stack([np.ones(T - 1), V, X[:-1]])
        g = np.linalg.lstsq(R, rx, rcond=None)[0]   # (1 + 2K, N)
        a = g[0]                   # (N,)
        beta = g[1:1 + k]          # (K, N)
        c = g[1 + k:].T            # (N, K)
        E = rx - R @ g
        n_ret = rx.shape[1]
        sigma2_e = float((E ** 2).sum() / (n_ret * (T - 1)))

        # Step 3: prices of risk from the cross-section
        b_star = np.array([np.outer(beta[:, i], beta[:, i]).ravel() for i in range(n_ret)])
        bb_inv_b = np.linalg.solve(beta @ beta.T, beta)
        lambda0 = bb_inv_b @ (a + 0.5 * (b_star @ sigma.ravel() + sigma2_e))
        lambda1 = bb_inv_b @ c

        # Short rate equation r_t = delta0 + delta1' X_t
        d = np.linalg.lstsq(np.column_stack([np.ones(T), X]), rf, rcond=None)[0]
        delta0, delta1 = float(d[0]), d[1:]

        A, B = self._recursion(mu - lambda0, phi - lambda1, sigma, sigma2_e, delta0, delta1)
        if self.bias_correct:
            # Only the real-world (P) dynamics change. Pricing (mu - lambda0, phi - lambda1) is pinned
            # by the cross-section of yields, so fitted yields stay the same and the correction moves
            # the split between expected rates and term premium.
            mu_p, phi_p = kilian_bias_correct(X, mu, phi, V)
            lambda0, lambda1 = lambda0 + (mu_p - mu), lambda1 + (phi_p - phi)
            mu, phi = mu_p, phi_p
        A_rn, B_rn = self._recursion(mu, phi, sigma, sigma2_e, delta0, delta1)

        self.result = ACMResult(mean, W, mu, phi, sigma, sigma2_e, lambda0, lambda1,
                                delta0, delta1, A, B, A_rn, B_rn, mats, beta,
                                self.return_maturities)
        return self

    def _recursion(self, mu_q, phi_q, sigma, s2, d0, d1):
        """Bond price recursion: log P_t^(n) = A_n + B_n' X_t."""
        N, k = self.max_maturity, self.k
        A = np.zeros(N + 1)
        B = np.zeros((N + 1, k))
        for n in range(1, N + 1):
            b = B[n - 1]
            A[n] = A[n - 1] + b @ mu_q + 0.5 * (b @ sigma @ b + s2) - d0
            B[n] = phi_q.T @ b - d1
        return A, B

    # ---- applying the model ----------------------------------------------

    def factors(self, yields: pd.DataFrame) -> pd.DataFrame:
        """Project any yield panel (e.g. daily) on the estimated PCA weights."""
        r = self.result
        Y = yields[r.maturities].to_numpy() / 100.0
        X = (Y - r.mean) @ r.loadings
        return pd.DataFrame(X, index=yields.index, columns=[f"PC{i+1}" for i in range(self.k)])

    def decompose(self, yields: pd.DataFrame, maturities=(24, 60, 120)) -> pd.DataFrame:
        """Fitted yield, risk-neutral (expected short rate) yield and term premium, in percent."""
        r = self.result
        X = self.factors(yields).to_numpy()
        out = {}
        for n in maturities:
            fit = -(r.A[n] + X @ r.B[n]) / n * 1200
            rn = -(r.A_rn[n] + X @ r.B_rn[n]) / n * 1200
            out[f"y{n}"] = fit
            out[f"rn{n}"] = rn
            out[f"tp{n}"] = fit - rn
        return pd.DataFrame(out, index=yields.index)
