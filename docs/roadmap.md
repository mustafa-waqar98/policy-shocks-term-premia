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

## Done (v1.1)
- Live scorecard for the Research Notes #1 call, graded on every meeting since Warsh became chair
- Scheduled-meeting flag taken from the Fed's calendar (the JK labels missed 11 unscheduled announcements)
- Dashboard rebuilt as an argument: one claim per section, validated colour roles, light and dark mode,
  a table view under every headline chart
- **Treasury supply, first pass.** Private duration supply by security since 2003 (MSPD minus SOMA, 10-year
  equivalents) and an event study of the quarterly refunding statements since 2009

## Next
- **Treasury supply, second pass.** Refunding *surprises* rather than raw size changes (sizes are usually
  signalled a quarter ahead): dealer survey expectations, or the change against the previous statement's
  guidance language. Intraday 10y moves around the 8:30am release, once intraday data is available.
- **Event volatility.** Option-implied event variance around FOMC and CPI dates vs the realised moves this
  project measures.
- **Event volatility.** Option-implied event variance around FOMC and CPI dates vs the realised moves this
  project measures.
- **Global spillovers** (separate repo). These shocks fed into EM sovereign spreads, FX and capital flows.
