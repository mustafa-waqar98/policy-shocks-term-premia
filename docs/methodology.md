# Methodology

1. [Zero-coupon curve](#1-zero-coupon-curve)
2. [ACM term premium](#2-acm-term-premium)
3. [FOMC shocks](#3-fomc-shocks)
4. [Transmission](#4-transmission)
5. [Single-meeting tests](#5-single-meeting-tests)

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

## 5. Single-meeting tests

Local projections estimate average responses across many meetings. To check whether a result shows up at a
single meeting, each FOMC meeting since 2008 is scored on a yes/no test over the window from the close before the
meeting to 20 trading days after. The meeting-day change in the 2y Treasury yield (FRED `DGS2`) stands in for the
surprise, as in Hanson and Stein (2015).

The same test is run on every 20-day window since 2008 that contains no FOMC meeting. Neighbouring control
windows share most of their days, so the uncertainty in the control hit rate comes from a moving-block bootstrap
with 21-day blocks; p-values are two-sided. Tests reported: real 5y5y moves more than breakeven 5y5y; real 5y5y
moves in the direction of the meeting-day 2y move, for minimum 2y moves of 0 (any nonzero move), 2, 3, 4 and 5bp;
and the 10y term premium moves in the direction of the 2y move. The 3bp cutoff was chosen once, near the median
meeting-day move, and not tuned; the other cutoffs are shown so that can be judged.

Code: [`src/tpshock/transmission/single_meeting.py`](../src/tpshock/transmission/single_meeting.py)

### Live scorecard

The rule above is also graded meeting by meeting for every FOMC decision since June 2026, when Warsh became
chair and the statement dropped forward guidance. For each meeting in `config.SCORECARD_MEETINGS`: the day-0 2y
move (DGS2), and the change in the real 5y5y forward, 2 x DFII10 - DFII5, from the close before the meeting to
the close 20 trading days after it. A hit means the two have the same sign. It is a miss if the real forward
moves the other way, ends the window unchanged at two decimals, or if the 2y does not move on the day. Meetings
with a 2y move under 3bp are graded the same way and flagged. Windows still open are shown with the days
elapsed. New meetings are added to the config list as the Fed publishes its calendar.

Code: [`src/tpshock/transmission/scorecard.py`](../src/tpshock/transmission/scorecard.py)

### Balance sheet announcements

The post-2008 robustness checks drop FOMC meetings whose statement included balance sheet news (asset purchases,
reinvestment, maturity extension, tapering, runoff). The list is hand-coded in `config.BALANCE_SHEET_DATES`.

---

## 6. Treasury supply

### Private duration supply

For each month-end since January 2001, Treasury's Monthly Statement of the Public Debt (table 3) lists every
marketable bill, note, bond, TIPS and FRN outstanding. From July 2003 the NY Fed's SOMA file gives the Fed's
holding of each security. Matching the two by CUSIP gives the amount the private market holds of each security,
$a_i$, with remaining maturity $m_i$ in years (FRNs count as one week, since they reset weekly; TIPS at par).
Three summaries, each over nominal GDP $Y$:

$$\text{debt-years} = \frac{\sum_i a_i m_i}{Y}, \qquad
\text{10-year equivalents} = \frac{1}{Y}\sum_i a_i \frac{m_i e^{-y m_i}}{10\, e^{-10 y}}, \quad y = 4\%$$

and the share of total duration the Fed holds. Debt-years is the maturity-weighted debt-to-GDP of Greenwood and
Vayanos (2014); the 10-year-equivalent weight is the dollar duration of a zero-coupon bond of maturity $m_i$
relative to a 10-year one.

### Refunding statements

Each quarter the Treasury announces the sizes of the 3-, 10- and 30-year auctions at 8:30am in its refunding
statement. In the auction file those auctions carry the statement date as their announcement date, which gives
the event dates. Only auctions announced the same morning as the 10-year count, which excludes the February
30-year TIPS. Supply news is the change in announced size against the previous refunding, in 10-year
equivalents ($\Delta S = \Delta s_3 \cdot 0.3 + \Delta s_{10} + \Delta s_{30} \cdot 3$). For each statement
since 2009 with no FOMC announcement on the day or the day after,

$$\Delta y_t = \alpha + \beta\, \frac{\Delta S_t}{10} + \varepsilon_t$$

with $\Delta y_t$ the move in the 10y yield (and its expected-rate and term premium pieces) from the close before
the statement to its close. A second window starts at the close before Monday's borrowing estimate, with FOMC
announcements anywhere from Monday to Thursday excluded. Treasury usually signals size changes a quarter ahead,
so $\Delta S$ overstates the news and $\beta$ is a lower bound on the response to genuine surprises.

Code: [`src/tpshock/supply/`](../src/tpshock/supply/)

---

## References

- Adrian, T., Crump, R. and Moench, E. (2013). Pricing the term structure with linear regressions. *Journal of Financial Economics* 110(1).
- Kilian, L. (1998). Small-sample confidence intervals for impulse response functions. *Review of Economics and Statistics* 80(2).
- Bauer, M., Rudebusch, G. and Wu, J. (2012). Correcting estimation bias in dynamic term structure models. *Journal of Business and Economic Statistics* 30(3).
- Gurkaynak, R., Sack, B. and Swanson, E. (2005). Do actions speak louder than words? *International Journal of Central Banking* 1(1).
- Gurkaynak, R., Sack, B. and Wright, J. (2007). The U.S. Treasury yield curve: 1961 to the present. *Journal of Monetary Economics* 54(8).
- Jarocinski, M. and Karadi, P. (2020). Deconstructing monetary policy surprises: the role of information shocks. *AEJ: Macroeconomics* 12(2).
- Greenwood, R. and Vayanos, D. (2014). Bond supply and excess bond returns. *Review of Financial Studies* 27(3).
- Jorda, O. (2005). Estimation and inference of impulse responses by local projections. *American Economic Review* 95(1).
