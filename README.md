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
   regime (pre-2008 vs the QE and post-QE era).

Everything is rebuilt from public data every weekday by GitHub Actions and published to the dashboard.

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
| ACM term premium | NY Fed published ACM series | correlation and RMSE shown on the dashboard |
| FOMC shocks (pc1, MP, CBI, both decompositions) | Jarocinski and Karadi published shocks | identical to 8 decimal places on all meetings |
| ACM estimator | simulated curve with a known term premium | recovers term premium dynamics (correlation > 0.95), see `tests/test_acm.py` |

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
