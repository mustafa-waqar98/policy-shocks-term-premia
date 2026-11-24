# Roadmap

## Done (v1.0)
- GSW curve at monthly maturities, ACM model with daily term premium, anchored to the T-bill rate
  (average gap to the NY Fed series about 1bp)
- FOMC shocks rebuilt from raw surprises: policy factor, target/path, policy/information (both versions),
  matching Jarocinski and Karadi to 8 decimal places
- Event regressions by regime, local projections with Newey-West errors
- 5y5y forwards split into expected rates vs term premium and real vs breakeven
- Robustness: by period, scheduled meetings only, excluding 2020, bias-corrected VAR
- Dashboard with meeting explorer, scenario tool and plain-English summaries; daily refresh via GitHub Actions

## Next projects that build on this one
- **Treasury supply.** Refunding announcement and auction surprises, and their effect on the term premium
  series built here.
- **Event volatility.** Option-implied event variance around FOMC and CPI dates vs the realised moves this
  project measures.
- **Global spillovers** (separate repo). These shocks fed into EM sovereign spreads, FX and capital flows.
