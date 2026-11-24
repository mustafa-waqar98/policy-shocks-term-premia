# Session 1: the big picture

## The question

A 10y yield can be split into two parts:

    10y yield  =  average expected overnight rate over 10 years  +  term premium

The first part is what you'd earn rolling overnight money for 10 years if rates follow the expected
path. The term premium is the extra yield investors demand for locking money up in a 10y bond instead
of rolling short. It pays for duration risk: if inflation or rates surprise upward, the long bond loses.

Neither part is observable. You see the yield, not the split. A model has to do the splitting.

**Why a desk cares.** Two 10bp selloffs can be very different trades. If expected rates went up, the
front end should move too, the Fed is the driver, and the move tends to stick with the policy path.
If the term premium went up, the driver is risk appetite, supply, inflation uncertainty or
positioning. Steepeners, swap spread trades and how you hedge all depend on which one it is.

## Part 1: the ACM model (`src/tpshock/models/acm.py`)

**Step 0, the curve.** The Fed publishes six parameters per day that draw a smooth zero-coupon curve
(Svensson curve). From them we read off yields at 1, 2, ..., 120 months. (`data/gsw.py`)

**Step 1, factors.** 120 yields are mostly driven by a few common moves. Principal component analysis
(PCA) finds them: PC1 is the level (everything shifts), PC2 the slope (steepen/flatten), PC3 curvature
(belly vs wings). ACM use 5. Then a VAR(1) says how these factors evolve month to month: if the curve
is steep today, what is it expected to look like next month? The part the VAR didn't predict is the
factor shock.

**Step 2, returns.** For bonds of 6 months to 10 years, compute the one-month excess return (return
over the 1-month bill). Regress it on the factor shocks and on the lagged factors. The coefficient on
the shocks is each bond's risk exposure (a 10y has a big exposure to the level shock, a 6m a tiny one).
The coefficient on the lagged factors says when excess returns are predictably high. Example: when
the curve is steep, long bonds tend to outperform. That predictability is the term premium showing up
in returns.

**Step 3, prices of risk.** If expected excess returns line up with exposures across bonds, the ratio
is the market price of each risk. One cross-sectional regression gives a constant price (lambda 0) and a
part that varies with the state of the curve (lambda 1).

**Pricing.** With the factor dynamics and prices of risk, the model prices every bond (the recursion in
`_recursion`). Price them again with prices of risk set to zero: you get what yields would be if
nobody demanded compensation for risk. That is the expected-rate part. Fitted yield minus that is the
term premium.

**The honest caveat.** The expected-rate part leans heavily on how persistent the VAR thinks rates are.
Persistence is hard to estimate, so the *level* of the term premium carries real uncertainty. Its
*changes* are pinned down much better. Our simulation test shows exactly this. That is one reason this
project looks at changes around FOMC days rather than arguing about whether the term premium is 0.3%
or 0.6%.

## Part 2: FOMC shocks (`src/tpshock/shocks/identify.py`)

**Why 30 minutes.** If you regress daily yield changes on the fed funds rate, you mix up cause and
effect: the Fed reacts to the economy and markets react to both. In a 30-minute window around the
2pm statement, almost nothing else happens. So the change in futures prices in that window is the
surprise, the part the market didn't already price.

**The surprises.** Changes in fed funds futures (this meeting, 3 months out) and SOFR (formerly
eurodollar) futures out to about a year. They mostly move together, so PCA again: the first component
is "how hawkish was this meeting overall". That's `pc1`.

**Target vs path.** Some meetings surprise on today's rate (a surprise cut). Others leave today's rate
as expected but change guidance (dots, "extended period", QT signals). GSS rotate two factors so the
*path* factor has zero effect on the current-meeting contract. Since 2008 most of the action is in
the path factor.

**Policy vs information.** A hawkish surprise should hurt stocks. But sometimes rates and stocks rise
together. JK argue the market then reads the Fed as saying "the economy is stronger than you think".
That's an information shock, and it has different effects: it raises expected growth and rates without
the tightening of financial conditions. The stock move separates the two:
- rates up, stocks down: monetary policy shock
- rates up, stocks up: information shock

We rebuild these from the raw data and match JK's published numbers exactly.

## Part 3: linking them (`src/tpshock/transmission/lp.py`)

**Event regression.** On each FOMC day, take the change in the 10y yield, its expected-rate part and
its term premium. Regress each on the shocks. That gives "per 10bp of shock, how many bp of each".

**Local projections.** Repeat with the change from the day before the meeting to 1, 2, 5, ... 60 trading
days after. Each horizon is one regression. Plot the coefficients against the horizon and you have an
impulse response. Newey-West standard errors because the windows overlap once they're longer than the
gap between meetings.

## What the first run shows (data to April 2023, the full sample comes once it runs on your laptop)

- Target surprises (today's rate) move expectations up and *lower* the 10y term premium.
- Path surprises (guidance) move expected rates at first, then the term premium keeps rising for
  about a month, by roughly 8bp per 10bp of shock. That echoes Hanson and Stein (2015), who argue
  reach-for-yield investors amplify guidance shocks into long-term rates through the term premium.
- Information shocks raise the 2y and 10y steadily over the following three months, consistent with
  the market updating on growth.

Treat these as first-pass results until the robustness checks in `docs/roadmap.md` are done.

## How to talk about it in an interview (30 seconds)

"I built a daily decomposition of the Treasury curve into expected rates and term premium using the
ACM model, and paired it with high-frequency FOMC shocks rebuilt from 30-minute futures moves. That
lets me ask which channel each kind of Fed surprise works through. Guidance shocks build up in the
term premium over a few weeks while rate surprises compress it, which matters for how you position
curve trades around meetings. It's validated against the NY Fed's series and JK's published shocks and
refreshes daily on a public dashboard."
