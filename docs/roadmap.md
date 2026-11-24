# Roadmap

## Done (v0.1)
- GSW curve at monthly maturities, ACM model with daily term premium
- FOMC shocks rebuilt from raw surprises: policy factor, target/path, policy/information (both versions)
- Event regressions by regime, local projections with Newey-West errors
- Dashboard with scenario tool, daily refresh and deploy via GitHub Actions
- Tests: simulation test for ACM, replication test for shocks, unit tests

## Next
1. **Validation pass.** Compare to the NY Fed series, check sample start and short-rate choices
   (GSW 1-month vs T-bill), document differences.
2. **Robustness.** Bias-corrected VAR (Bauer, Rudebusch and Wu 2012) for the term premium level;
   two-day windows; excluding March 2020; controlling for macro data releases on FOMC days.
3. **Real vs nominal.** Split the term premium using TIPS (breakevens vs real yields), as in
   Hanson and Stein (2015).
4. **Regimes.** Rolling or state-dependent local projections (hiking vs cutting cycles, QE on vs off).

## Later projects that plug into this one
- **Global spillovers (#5).** Feed the shocks into a BVAR on EM sovereign spreads, FX and capital flows,
  and estimate a global risk factor.
- **Event volatility (#6).** Option-implied event variance around FOMC and CPI dates vs the realised
  moves this project already measures.
- **Treasury supply (#9).** Refunding announcement surprises and their effect on the term premium series
  built here.
