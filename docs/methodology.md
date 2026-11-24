# Methodology

## 1. Zero-coupon curve

The Fed's GSW dataset gives daily Svensson parameters. The zero yield at maturity $n$ years is

$$
y(n) = \beta_0 + \beta_1 \frac{1-e^{-n/\tau_1}}{n/\tau_1}
+ \beta_2\left(\frac{1-e^{-n/\tau_1}}{n/\tau_1} - e^{-n/\tau_1}\right)
+ \beta_3\left(\frac{1-e^{-n/\tau_2}}{n/\tau_2} - e^{-n/\tau_2}\right)
$$

We evaluate it at monthly maturities 1 to 120. Before 1980 the curve is Nelson-Siegel ($\beta_3 = 0$).
Code: `src/tpshock/data/gsw.py`.

## 2. ACM term premium

Let $p_t^{(n)} = -\tfrac{n}{12} y_t^{(n)}$ be the log price of an $n$-month zero and $r_t$ the one-month rate.

**Factors.** $X_t$ = first $K=5$ principal components of yields, with dynamics
$X_{t+1} = \mu + \Phi X_t + v_{t+1}$, $v \sim N(0, \Sigma)$, estimated by OLS.

**Excess returns.** For bonds $n \in \{6, 12, \dots, 120\}$,
$rx_{t+1}^{(n-1)} = p_{t+1}^{(n-1)} - p_t^{(n)} - r_t$. Stacking across bonds,

$$
rx = a\iota' + \beta' V + c X_- + E
$$

estimated by OLS. $\beta$ ($K \times N$) are risk exposures, $\sigma^2 = \mathrm{tr}(EE')/NT$.

**Prices of risk.** With $B^*$ the $N \times K^2$ matrix with rows $\mathrm{vec}(\beta_i\beta_i')'$,

$$
\lambda_0 = (\beta\beta')^{-1}\beta\left(a + \tfrac12(B^*\mathrm{vec}(\Sigma) + \sigma^2\iota)\right),
\qquad
\lambda_1 = (\beta\beta')^{-1}\beta c
$$

**Pricing.** With $r_t = \delta_0 + \delta_1' X_t$ (OLS), log prices $p_t^{(n)} = A_n + B_n' X_t$ follow

$$
A_n = A_{n-1} + B_{n-1}'(\mu - \lambda_0) + \tfrac12(B_{n-1}'\Sigma B_{n-1} + \sigma^2) - \delta_0,
\qquad
B_n' = B_{n-1}'(\Phi - \lambda_1) - \delta_1'
$$

Setting $\lambda_0 = \lambda_1 = 0$ gives risk-neutral yields: the average expected short rate.
Term premium = fitted yield minus risk-neutral yield.

**Daily series.** Parameters are estimated on month-end data; daily factors are daily yields projected
on the same principal component weights. This is how the NY Fed produces its daily series.

**Known caveat.** The level of the term premium depends on the estimated persistence $\Phi$, which is
biased downward in finite samples (Bauer, Rudebusch and Wu 2012). Changes in the term premium are
much better identified than its level. `tests/test_acm.py` shows this in simulation. This is one reason
the project focuses on changes around FOMC announcements.

Code: `src/tpshock/models/acm.py`.

## 3. FOMC shocks

**Surprises.** For each announcement, the change in futures-implied rates from 10 minutes before to
20 minutes after (Jarocinski and Karadi's data): current-month and 3-month-ahead fed funds futures
(MP1, FF4) and 2nd to 4th quarterly eurodollar / SOFR futures (ED2 to ED4), plus the S&P 500.

**Policy factor.** First principal component of the standardised rate surprises (uncentred, missing
values set to zero), rescaled to the standard deviation of ED4 so the units are percentage points.

**Target and path.** Take the first two principal components $F$ of the standardised surprises and
rotate them so that the second factor has no effect on MP1. The target factor is scaled so it moves
MP1 one for one, the path factor so it moves ED4 one for one (GSS 2005).

**Policy and information.** Let $m$ = policy factor, $s$ = stock surprise. A monetary policy shock
moves $m$ and $s$ in opposite directions; an information shock moves them in the same direction.
- *Poor man's*: the whole of $m$ is MP if $m s < 0$, otherwise CBI.
- *Median rotation*: among all orthogonal rotations of $(m, s)$ that satisfy the sign restrictions,
  take the median angle; both shocks can be non-zero at every meeting.

Code: `src/tpshock/shocks/identify.py`. Replicates the published JK series exactly.

## 4. Transmission

**Event regressions.** For outcome $x$ (fitted 10y, its expectations part, its term premium),
$\Delta x_d = \alpha + \gamma' \varepsilon_d + u_d$ on announcement days $d$, with $\Delta$ the close to
close change. Run on the full sample and on 1990-2007 and 2008-present.

**Local projections.** For each horizon $h$ in trading days,

$$
x_{d+h} - x_{d-1} = \alpha_h + \gamma_h' \varepsilon_d + u_{d,h}
$$

across meetings. $\gamma_h$ traces the impulse response. Windows overlap for $h$ beyond the gap between
meetings, so standard errors are Newey-West with $\lceil (h+1)/30 \rceil$ lags. Responses are reported per
10bp shock.

Code: `src/tpshock/transmission/lp.py`.
