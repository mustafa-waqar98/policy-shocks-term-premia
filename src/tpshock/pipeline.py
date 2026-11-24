"""End-to-end steps. Each one reads from data/ and writes to data/processed/."""

import json
import logging

import numpy as np
import pandas as pd

from tpshock import config
from tpshock.data import acm_official, fomc, fred, gsw
from tpshock.models.acm import ACM
from tpshock.shocks.identify import build_shocks
from tpshock.transmission.lp import fomc_day_table, local_projections, ols

log = logging.getLogger("tpshock")
P = config.PROCESSED


def fetch():
    config.RAW.mkdir(parents=True, exist_ok=True)
    jobs = [("GSW yield curve", gsw.download), ("JK surprises", fomc.download_surprises),
            ("JK published shocks", fomc.download_published),
            ("NY Fed ACM (validation)", acm_official.download)]
    jobs += [(f"FRED {s}", lambda s=s: fred.download(s)) for s in config.FRED_SERIES]
    for name, fn in jobs:
        try:
            fn()
            log.info("fetched %s", name)
        except Exception as exc:  # keep going; cached copies may exist
            log.warning("could not fetch %s: %s", name, exc)


def fit_term_premium():
    params = gsw.load()
    mats = list(range(1, config.ACM_MAX_MATURITY + 1))
    monthly = gsw.svensson(gsw.month_end(params).loc[config.ACM_START:], mats)
    model = ACM(config.ACM_N_FACTORS, config.ACM_RETURN_MATURITIES, config.ACM_MAX_MATURITY)
    if config.SHORT_RATE == "tbill":
        # T-bill rate on (or just before) each month-end date of the curve
        bill = fred.tbill_cc().sort_index()
        short = bill.reindex(monthly.index, method="ffill")
        log.info("short rate: 3-month T-bill (mean %.2f%%)", short.mean())
        model.fit(monthly, short_rate=short)
    else:
        log.info("short rate: 1-month yield from the fitted curve")
        model.fit(monthly)

    daily_yields = gsw.svensson(params.loc[config.ACM_START:], mats)
    daily = model.decompose(daily_yields, config.REPORT_MATURITIES)
    monthly_out = model.decompose(monthly, config.REPORT_MATURITIES)

    P.mkdir(parents=True, exist_ok=True)
    daily.round(4).to_csv(P / "acm_daily.csv")
    monthly_out.round(4).to_csv(P / "acm_monthly.csv")
    fit_err = (monthly_out["y120"] - monthly[120]).abs().mean() * 100
    meta = {"sample": [str(monthly.index[0].date()), str(monthly.index[-1].date())],
            "n_months": len(monthly), "mean_abs_fit_error_10y_bp": round(float(fit_err), 2),
            "pricing_error_sd_bp_per_month": round(float(np.sqrt(model.result.sigma2_e) * 1e4), 1)}
    (P / "acm_meta.json").write_text(json.dumps(meta, indent=2))
    log.info("ACM fitted on %s months, last date %s", len(monthly), daily.index[-1].date())
    return model


def validate_term_premium():
    """Compare our 10y term premium to the NY Fed's published ACM series."""
    try:
        off = acm_official.load()
    except Exception as exc:
        log.warning("official ACM file not available: %s", exc)
        return None
    ours = pd.read_csv(P / "acm_monthly.csv", index_col=0, parse_dates=True)
    ours.index = ours.index.to_period("M")
    off.index = off.index.to_period("M")
    j = ours[["tp120", "tp24"]].join(off[["ACMTP10", "ACMTP02"]], how="inner").dropna()
    stats = {"months_compared": len(j),
             "corr_10y": round(float(j.tp120.corr(j.ACMTP10)), 4),
             "rmse_10y_bp": round(float(np.sqrt(((j.tp120 - j.ACMTP10) ** 2).mean()) * 100), 1),
             "corr_2y": round(float(j.tp24.corr(j.ACMTP02)), 4)}
    j.to_csv(P / "acm_vs_official.csv")
    (P / "acm_validation.json").write_text(json.dumps(stats, indent=2))
    log.info("validation vs NY Fed: %s", stats)
    return stats


def identify_shocks():
    s = fomc.load_surprises()
    shocks = build_shocks(s)
    shocks.to_csv(P / "shocks.csv")
    try:
        pub = fomc.load_published()
        a, b = shocks.copy(), pub.copy()
        a.index, b.index = a.index.floor("min"), b.index.floor("min")
        j = a.join(b, rsuffix="_pub", how="inner")
        diff = max(float((j[c] - j[c + "_pub"]).abs().max())
                   for c in ["pc1", "MP_pm", "CBI_pm", "MP_median", "CBI_median"])
        (P / "shocks_validation.json").write_text(json.dumps(
            {"meetings": len(j), "max_abs_diff_vs_published": diff}, indent=2))
        log.info("shocks match JK published to within %.1e", diff)
    except Exception as exc:
        log.warning("could not compare with published shocks: %s", exc)
    return shocks


def transmission():
    daily = pd.read_csv(P / "acm_daily.csv", index_col=0, parse_dates=True)
    shocks = pd.read_csv(P / "shocks.csv", index_col=0, parse_dates=True)

    extra = fred.load_all()
    extra = extra[[c for c in config.FRED_OUTCOMES if c in extra]]
    if not extra.empty:
        if "DTWEXBGS" in extra:
            extra["DTWEXBGS"] = np.log(extra["DTWEXBGS"])  # the x100 in the projections then gives % changes
        daily = daily.join(extra, how="left")

    # FOMC-day split for every meeting
    table = fomc_day_table(daily, shocks)
    table.round(5).to_csv(P / "fomc_days.csv")

    # same-day regressions, full sample and by regime
    regimes = {"1990-2007": ("1990", "2007"), "2008-present": ("2008", "2100"), "all": ("1990", "2100")}
    rows = []
    for name, (a, b) in regimes.items():
        t = table.loc[a:b]
        for shock_set in (["MP_median", "CBI_median"], ["target", "path"]):
            for y in ["d_yield", "d_expectations", "d_term_premium"]:
                ok = t[y].notna()
                X = np.column_stack([np.ones(ok.sum()), t.loc[ok, shock_set].to_numpy()])
                coef, se, r2 = ols(t.loc[ok, y].to_numpy(), X, lags=1)
                for i, s in enumerate(shock_set):
                    rows.append({"regime": name, "outcome": y, "shock": s,
                                 "bp_per_10bp": coef[i + 1] * 0.10, "se": se[i + 1] * 0.10,
                                 "r2": r2, "n": int(ok.sum())})
    pd.DataFrame(rows).round(4).to_csv(P / "event_regressions.csv", index=False)

    outcomes = [c for c in ["y24", "y120", "rn120", "tp120", *config.FRED_OUTCOMES] if c in daily]
    irf = pd.concat([
        local_projections(daily[outcomes], shocks, ["MP_median", "CBI_median"], config.LP_HORIZONS),
        local_projections(daily[outcomes], shocks, ["target", "path"], config.LP_HORIZONS),
    ])
    irf.round(4).to_csv(P / "irf.csv", index=False)
    log.info("transmission done: %s meetings", len(table))


def run_all(skip_fetch: bool = False):
    if not skip_fetch:
        fetch()
    fit_term_premium()
    validate_term_premium()
    identify_shocks()
    transmission()
    from tpshock.site.build import build
    build()
