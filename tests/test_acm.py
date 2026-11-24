"""Simulate a curve from a known affine model, then check ACM recovers its term premium."""

import numpy as np
import pandas as pd

from tpshock.models.acm import ACM


def simulate(T=900, seed=0):
    rng = np.random.default_rng(seed)
    k, N = 3, 120
    phi = np.diag([0.985, 0.94, 0.85])
    sigma = np.diag([0.00025, 0.0002, 0.00015]) ** 2
    mu = np.zeros(k)
    delta0, delta1 = 0.004, np.array([1.0, 0.6, 0.3])
    lambda0 = np.array([-1e-5, 5e-6, 0.0])
    lambda1 = np.array([[-0.02, 0.0, 0.0], [0.0, -0.05, 0.0], [0.0, 0.0, 0.0]])

    def rec(mq, pq):
        A, B = np.zeros(N + 1), np.zeros((N + 1, k))
        for n in range(1, N + 1):
            b = B[n - 1]
            A[n] = A[n - 1] + b @ mq + 0.5 * b @ sigma @ b - delta0
            B[n] = pq.T @ b - delta1
        return A, B

    A, B = rec(mu - lambda0, phi - lambda1)
    Arn, Brn = rec(mu, phi)
    X = np.zeros((T, k))
    chol = np.linalg.cholesky(sigma)
    for t in range(1, T):
        X[t] = mu + phi @ X[t - 1] + chol @ rng.standard_normal(k)
    n = np.arange(1, N + 1)
    Y = -(A[1:][None, :] + X @ B[1:].T) / n * 1200
    Yrn = -(Arn[1:][None, :] + X @ Brn[1:].T) / n * 1200
    Y = Y + rng.normal(0, 0.0005, Y.shape)  # tiny measurement noise
    idx = pd.period_range("1950-01", periods=T, freq="M").to_timestamp("M")
    return pd.DataFrame(Y, index=idx, columns=list(n)), (Y[:, 119] - Yrn[:, 119])


def test_recovers_term_premium_dynamics():
    """The level of the term premium depends on the estimated persistence of
    rates, which is noisy even in long samples. Its movements are identified
    much more tightly. So we test that the shape and the changes are right."""
    for seed in range(3):
        yields, true_tp10 = simulate(seed=seed)
        model = ACM(n_factors=3).fit(yields)
        est = model.decompose(yields, maturities=[120])["tp120"].to_numpy()
        assert np.corrcoef(est, true_tp10)[0, 1] > 0.95
        assert np.corrcoef(np.diff(est), np.diff(true_tp10))[0, 1] > 0.95


def test_fits_yields_and_identity():
    yields, _ = simulate(seed=1)
    model = ACM(n_factors=3).fit(yields)
    out = model.decompose(yields, maturities=[24, 120])
    assert (out["y120"] - yields[120]).abs().mean() < 0.02  # under 2bp
    assert np.allclose(out["tp120"], out["y120"] - out["rn120"])


def test_daily_projection_uses_monthly_parameters():
    yields, _ = simulate(seed=2)
    model = ACM(n_factors=3).fit(yields)
    sub = yields.iloc[-10:]
    a = model.decompose(sub)
    b = model.decompose(yields).iloc[-10:]
    assert np.allclose(a.to_numpy(), b.to_numpy())


def test_bias_correction_raises_persistence_and_keeps_fit():
    yields, _ = simulate(T=300, seed=3)
    plain = ACM(n_factors=3).fit(yields)
    corrected = ACM(n_factors=3, bias_correct=True).fit(yields)
    eig = lambda m: np.abs(np.linalg.eigvals(m.result.phi)).max()  # noqa: E731
    assert eig(corrected) > eig(plain)          # OLS understates persistence
    assert eig(corrected) < 1                   # but the corrected VAR stays stationary
    a = plain.decompose(yields, [120])
    b = corrected.decompose(yields, [120])
    assert np.allclose(a.y120, b.y120)          # fitted yields unchanged
    assert not np.allclose(a.tp120, b.tp120)    # only the split moves
