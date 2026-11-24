# Fed shocks and the term premium

[![tests](https://github.com/mustafa-waqar98/policy-shocks-term-premia/actions/workflows/ci.yml/badge.svg)](https://github.com/mustafa-waqar98/policy-shocks-term-premia/actions/workflows/ci.yml)
[![dashboard](https://github.com/mustafa-waqar98/policy-shocks-term-premia/actions/workflows/refresh.yml/badge.svg)](https://mustafa-waqar98.github.io/policy-shocks-term-premia/)

**Live dashboard: https://mustafa-waqar98.github.io/policy-shocks-term-premia/**

When the Fed surprises the market, the 10-year Treasury yield moves. This project asks *why* it moves:
because investors now expect a different path for the policy rate, or because they want a different
premium for holding duration risk. The answer matters for anyone trading the curve. A move driven by
expectations tends to show up across the front end and persist with the policy path; a move driven by
the term premium is about risk appetite and supply, can reverse, and hedges differently.

The project combines two well-known pieces of empirical macro-finance and links them:

1. **A term structure model** (Adrian, Crump and Moench 2013) that splits every point on the Treasury
   curve, every day since 1961, into an expected short-rate path and a term premium.
2. **High-frequency monetary policy shocks** built from 30-minute futures price changes around every
   FOMC announcement since 1990, decomposed two ways:
   - *target vs path* (Gurkaynak, Sack and Swanson 2005): did the Fed surprise on today's rate or on
     guidance about future rates?
   - *policy vs information* (Jarocinski and Karadi 2020): was the surprise a tightening (stocks fall)
     or a signal about the economy (stocks rise)?
3. **Transmission**: event regressions and local projections (Jorda 2005) tracing how each type of
   shock moves expected rates and the term premium over the following three months, split by
   regime (pre-2008 vs the QE and post-QE era), plus a split of 5y5y forwards into expected rates vs
   term premium and real vs breakeven inflation.

Everything is rebuilt from public data every weekday by GitHub Actions and published to the dashboard.

## Findings

Per 10bp shock, FOMC announcements 1990 to 2026. Full tables, robustness checks and charts are on the dashboard.

1. **Rate surprises and guidance surprises move the term premium in opposite directions.** A surprise in today's
   decision raises 10y expected rates by 3.2bp on the day and lowers the 10y term premium by 2.3bp. A guidance
   (path) surprise leaves the term premium flat on the day, then it builds to +7.4bp after 20 trading days.
2. **The guidance effect is a post-2008 phenomenon.** Zero before 2008 (-0.3bp), +17bp since. The response of real
   yields to a tightening shows the same split: +3bp before, +26bp after.
3. **It lands in the far future, and in real rates.** After a guidance surprise the 5y5y forward rises 10.9bp
   over 20 days. Of that, 9.7bp is term premium and 1.2bp expected rates. The real 5y5y forward (TIPS) rises
   13.6bp while breakevens barely move (+2.0bp, not significant). This is the real term premium channel of
   Hanson and Stein (2015).
4. **Robust to the usual objections.** Inside the post-2008 sample, the 20-day guidance effect on the 10y term
   premium is +17.1bp (t 2.7); excluding the 35 balance sheet announcement meetings +17.0bp (t 2.2); excluding
   2020 +16.9bp; scheduled meetings only +15.7bp; only guidance surprises where stocks fell +16.6bp (t 2.5); with a
   small-sample bias-corrected VAR +13.8bp (t 2.0). Full tables in the dashboard's robustness section.
5. **Weak but real at a single meeting.** Since 2008, the real 5y5y forward moved in the direction of the
   meeting-day 2y move over the next 20 trading days after 66% of the 94 meetings where the 2y moved at least 3bp,
   against 54% in 20-day windows with no meeting (p = 0.02, two-sided, block bootstrap). "Real moves more than
   breakeven" shows no edge at all (63% vs 64%): it is how TIPS behave, not a Fed effect. See
   `src/tpshock/transmission/single_meeting.py`.

Caveats: the TIPS pieces start in 2003 (204 meetings) and use par yields; bands are wide beyond 20 trading days;
model-based term premium levels are less reliable than their changes.

## What it is useful for

- **Daily curve attribution.** Was yesterday's 10bp selloff in the 10y expectations or term premium?
- **FOMC post-mortem.** For each meeting: the size of the surprise, whether the market read it as
  policy or as information, and how the 10y move split between the two components.
- **Pre-meeting scenarios.** The dashboard's scenario tool maps a hypothetical surprise (for example a
  hawkish dot plot, which is a path shock) into the historical average response of the 2y, 10y,
  expected rates, term premium, breakevens, credit spreads and the dollar.

## Validation

Both building blocks are checked against published series, and the checks run on every rebuild:

| Component | Benchmark | Result |
|---|---|---|
| ACM term premium | NY Fed published ACM series | correlation 0.9996 in levels, 0.96 in daily changes, 0.99 on FOMC days; average gap about 1bp after anchoring to the T-bill rate (18bp when using the curve's own 1-month point) |
| FOMC shocks (pc1, MP, CBI, both decompositions) | Jarocinski and Karadi published shocks | identical to 8 decimal places on all meetings |
| ACM estimator | simulated curve with a known term premium | recovers term premium dynamics (correlation > 0.95), see `tests/test_acm.py` |
| Bias correction | simulated curve | raises VAR persistence, leaves fitted yields unchanged, see `tests/test_acm.py` |

## Repository layout

```
src/tpshock/
  data/            downloaders and loaders (Fed GSW curve, JK surprises, FRED, NY Fed ACM)
  models/acm.py    the ACM term structure model
  shocks/          policy factor, target/path, policy/information decompositions
  transmission/    event regressions and local projections with Newey-West errors
  site/            static dashboard builder (Plotly + Jinja)
  pipeline.py      the end-to-end steps
  cli.py           `tpshock <step>` command
tests/             unit tests, simulation tests and replication tests
docs/
  methodology.md   the maths, with references
  learning/        plain-language notes written alongside the code
  roadmap.md       what comes next
.github/workflows/ CI (lint + tests) and the daily data refresh and deploy
```

## Running it

Needs Python 3.10 or newer.

```bash
make setup      # create a virtual environment and install
make all        # download data, estimate, build the site
make serve      # view the dashboard at http://localhost:8000
make test       # lint and run the tests
```

Individual steps: `tpshock fetch | fit | validate | shocks | transmit | site | all`.

## Data

- Federal Reserve Board, [Gurkaynak, Sack and Wright Treasury yield curve](https://www.federalreserve.gov/data/nominal-yield-curve.htm)
- Jarocinski and Karadi, [updated FOMC surprises and shocks](https://github.com/marekjarocinski/jkshocks_update_fed) (CC BY 4.0)
- Federal Reserve Bank of New York, [ACM term premia](https://www.newyorkfed.org/research/data_indicators/term-premia-tabs) (validation only)
- FRED: breakevens, TIPS yields, Baa spread, broad dollar index

## References

- Adrian, T., Crump, R. and Moench, E. (2013). Pricing the term structure with linear regressions. *Journal of Financial Economics* 110(1).
- Gurkaynak, R., Sack, B. and Swanson, E. (2005). Do actions speak louder than words? *International Journal of Central Banking* 1(1).
- Gurkaynak, R., Sack, B. and Wright, J. (2007). The U.S. Treasury yield curve: 1961 to the present. *Journal of Monetary Economics* 54(8).
- Hanson, S. and Stein, J. (2015). Monetary policy and long-term real rates. *Journal of Financial Economics* 115(3).
- Jarocinski, M. and Karadi, P. (2020). Deconstructing monetary policy surprises: the role of information shocks. *AEJ: Macroeconomics* 12(2).
- Jorda, O. (2005). Estimation and inference of impulse responses by local projections. *American Economic Review* 95(1).
- Nakamura, E. and Steinsson, J. (2018). High-frequency identification of monetary non-neutrality. *Quarterly Journal of Economics* 133(3).

## License

Code under MIT. Data belongs to its providers under their terms.
