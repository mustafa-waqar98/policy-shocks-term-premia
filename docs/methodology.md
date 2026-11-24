# Methodology

1. [Zero-coupon curve](#1-zero-coupon-curve)
2. [ACM term premium](#2-acm-term-premium)
3. [FOMC shocks](#3-fomc-shocks)
4. [Transmission](#4-transmission)

---

## 1. Zero-coupon curve

The Fed's GSW dataset gives six Svensson parameters per day. The continuously compounded zero yield at
maturity $`n`$ years is

```math
y(n) = \beta_0
+ \beta_1 \, \frac{1 - e^{-n/\tau_1}}{n/\tau_1}
+ \beta_2 \left( \frac{1 - e^{-n/\tau_1}}{n/\tau_1} - e^{-n/\tau_1} \right)
+ \beta_3 \left( \frac{1 - e^{-n/\tau_2}}{n/\tau_2} - e^{-n/\tau_2} \right)
```

We evaluate it at monthly maturities $`n = 1, 2, \dots, 120`$ months. Before 1980 the curve is
Nelson-Siegel, i.e. $`\beta_3 = 0`$.

Code: [`src/tpshock/data/gsw.py`](../src/tpshock/data/gsw.py)

---

## 2. ACM term premium

Adrian, Crump and Moench (2013). Notation: $`p_t^{(n)}`$ is the log price of an $`n`$-month zero-coupon
bond and $`r_t`$ the one-month rate,

```math
p_t^{(n)} = -\frac{n}{12} \, y_t^{(n)}, \qquad r_t = -p_t^{(1)}
```

### Step 1: factors

$`X_t`$ is the vector of the first $`K = 5`$ principal components of the yield curve. Its dynamics are a
VAR(1) estimated by OLS:

```math
X_{t+1} = \mu + \Phi X_t + v_{t+1}, \qquad v_{t+1} \sim N(0, \Sigma)
```

### Step 2: excess returns

For bonds with maturity $`n \in \{6, 12, \dots, 120\}`$, the one-month log excess return is

```math
rx_{t+1}^{(n-1)} = p_{t+1}^{(n-1)} - p_t^{(n)} - r_t
```

Stacking the $`N`$ bonds and $`T`$ months, estimate by OLS

```math
rx = a \iota^\top + \beta^\top V + c X_{-} + E
```

where $`V`$ holds the VAR shocks, $`X_{-}`$ the lagged factors, $`\beta`$ ($`K \times N`$) the risk exposures, and

```math
\sigma^2 = \frac{\operatorname{tr}(E E^\top)}{N T}
```

### Step 3: prices of risk

Let $`B^{\ast}`$ be the $`N \times K^2`$ matrix whose $`i`$-th row is $`\operatorname{vec}(\beta_i \beta_i^\top)^\top`$. Then

```math
\lambda_0 = (\beta \beta^\top)^{-1} \beta \left( a + \tfrac{1}{2} \left( B^{\ast} \operatorname{vec}(\Sigma) + \sigma^2 \iota \right) \right)
```

```math
\lambda_1 = (\beta \beta^\top)^{-1} \beta \, c
```

### Pricing

With the short rate $`r_t = \delta_0 + \delta_1^\top X_t`$ (OLS), log bond prices are affine,
$`p_t^{(n)} = A_n + B_n^\top X_t`$, with $`A_0 = 0`$, $`B_0 = 0`$ and

```math
A_n = A_{n-1} + B_{n-1}^\top (\mu - \lambda_0) + \tfrac{1}{2} \left( B_{n-1}^\top \Sigma B_{n-1} + \sigma^2 \right) - \delta_0
```

```math
B_n^\top = B_{n-1}^\top (\Phi - \lambda_1) - \delta_1^\top
```

Fitted yields are $`y_t^{(n)} = -\tfrac{12}{n}(A_n + B_n^\top X_t)`$. Repeating the recursion with
$`\lambda_0 = 0`$ and $`\lambda_1 = 0`$ gives the **risk-neutral yield**, the average expected short rate.
The **term premium** is the difference:

```math
TP_t^{(n)} = y_t^{(n)} - y_t^{(n), \text{RN}}
```

### Which short rate

The short rate $`r_t`$ anchors the expected-rate path, so its level passes straight into the split. The GSW
curve is fitted to notes and bonds only, so its 1-month point is an extrapolation, and actual Treasury bills
usually yield less than that because investors pay for their cash-like safety. Using the curve's 1-month
yield left our 10y term premium 18bp below the NY Fed's series on average (correlation 0.9997). Using the
3-month T-bill rate (FRED `DTB3`, converted from discount basis to a continuously compounded yield) closes
the gap to about 1bp on average and cuts the root mean squared difference from 18.6bp to 5.9bp. The setting
is `SHORT_RATE` in `config.py`.

### Daily series

Parameters are estimated on month-end data. Daily factors come from projecting daily yields on the same
principal component weights, which is how the NY Fed builds its daily series.

### Small-sample bias correction

The level of the term premium depends on the estimated persistence $`\Phi`$, which OLS understates in finite
samples (Bauer, Rudebusch and Wu 2012). Understated persistence makes expected rates revert too fast, which
pushes too much of each move into the term premium. As a robustness check the model is also estimated with a
Kilian (1998) bootstrap correction: simulate 500 histories from the fitted VAR, re-estimate $`\Phi`$ on each,
and add back the average shortfall,

```math
\Phi^{\text{bc}} = \hat\Phi - \left( \overline{\Phi^{*}} - \hat\Phi \right)
```

shrinking the correction if the result is explosive. Only the real-world dynamics change: the pricing
parameters $`\mu - \lambda_0`$ and $`\Phi - \lambda_1`$ are pinned by the cross-section of yields, so fitted
yields are identical and only the split moves. The largest eigenvalue rises from 0.991 to 0.999 and the
volatility of the 10y term premium falls by about 15%. The main FOMC results hold under the corrected model
(see the robustness table). [`tests/test_acm.py`](../tests/test_acm.py) checks both properties in simulation.

Code: [`src/tpshock/models/acm.py`](../src/tpshock/models/acm.py)

---

## 3. FOMC shocks

### Surprises

For each announcement, the change in futures-implied rates from 10 minutes before to 20 minutes after the
statement (Jarocinski and Karadi's data): current-month and 3-month-ahead fed funds futures (MP1, FF4),
the 2nd to 4th quarterly eurodollar or SOFR futures (ED2 to ED4), and the S&P 500.

### Policy factor

The first principal component of the standardised rate surprises (uncentred, missing values set to zero),
rescaled to the standard deviation of ED4 so it is in percentage points. Call it $`m_t`$.

### Target and path

Take the first two principal components $`F`$ of the standardised surprises and rotate them so that the
second factor has no effect on MP1 (Gurkaynak, Sack and Swanson 2005):

```math
\text{target}_t = F_t \gamma, \qquad \text{path}_t = F_t \gamma^{\perp}, \qquad \gamma = \arg\min_g \lVert \text{MP1} - F g \rVert
```

The target factor is scaled to move MP1 one for one, the path factor to move ED4 one for one.

### Policy and information

With $`s_t`$ the stock market surprise, a monetary policy shock moves $`m_t`$ and $`s_t`$ in opposite
directions, an information shock in the same direction (Jarocinski and Karadi 2020).

| | effect on rates $`m_t`$ | effect on stocks $`s_t`$ |
|---|---|---|
| Monetary policy shock | $`+`$ | $`-`$ |
| Information shock | $`+`$ | $`+`$ |

- **Poor man's:** all of $`m_t`$ is policy if $`m_t s_t < 0`$, otherwise information.
- **Median rotation:** among all orthogonal rotations of $`(m_t, s_t)`$ that satisfy the sign
  restrictions, take the median angle, so both shocks can be non-zero at every meeting.

Code: [`src/tpshock/shocks/identify.py`](../src/tpshock/shocks/identify.py). Replicates the published
Jarocinski-Karadi series to 8 decimal places.

---

## 4. Transmission

### Event regressions

For an outcome $`x`$ (fitted 10y yield, its expected-rate part, its term premium) and shocks
$`\varepsilon_d`$ on announcement days $`d`$:

```math
\Delta x_d = \alpha + \gamma^\top \varepsilon_d + u_d
```

with $`\Delta`$ the close-to-close change. Estimated on the full sample, 1990-2007 and 2008-present.

### Local projections

For each horizon $`h`$ in trading days (Jorda 2005):

```math
x_{d+h} - x_{d-1} = \alpha_h + \gamma_h^\top \varepsilon_d + u_{d,h}
```

The sequence $`\gamma_0, \gamma_1, \dots, \gamma_{60}`$ is the impulse response. Windows overlap once $`h`$
exceeds the gap between meetings, so standard errors are Newey-West with $`\lceil (h+1)/30 \rceil`$ lags.
Responses are reported per 10bp shock.

Code: [`src/tpshock/transmission/lp.py`](../src/tpshock/transmission/lp.py)

---

## References

- Adrian, T., Crump, R. and Moench, E. (2013). Pricing the term structure with linear regressions. *Journal of Financial Economics* 110(1).
- Kilian, L. (1998). Small-sample confidence intervals for impulse response functions. *Review of Economics and Statistics* 80(2).
- Bauer, M., Rudebusch, G. and Wu, J. (2012). Correcting estimation bias in dynamic term structure models. *Journal of Business and Economic Statistics* 30(3).
- Gurkaynak, R., Sack, B. and Swanson, E. (2005). Do actions speak louder than words? *International Journal of Central Banking* 1(1).
- Gurkaynak, R., Sack, B. and Wright, J. (2007). The U.S. Treasury yield curve: 1961 to the present. *Journal of Monetary Economics* 54(8).
- Jarocinski, M. and Karadi, P. (2020). Deconstructing monetary policy surprises: the role of information shocks. *AEJ: Macroeconomics* 12(2).
- Jorda, O. (2005). Estimation and inference of impulse responses by local projections. *American Economic Review* 95(1).
