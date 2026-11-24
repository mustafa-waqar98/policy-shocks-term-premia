"""Paths, data sources and model settings in one place."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
SITE_OUT = ROOT / "site"

# Change mustafa-waqar98 once the GitHub repo exists
REPO_URL = "https://github.com/mustafa-waqar98/policy-shocks-term-premia"

# Data sources
GSW_URL = "https://www.federalreserve.gov/data/yield-curve-tables/feds200628.csv"
ACM_OFFICIAL_URL = (
    "https://www.newyorkfed.org/medialibrary/media/research/"
    "data_indicators/ACMTermPremium.xls"
)
JK_BASE = "https://raw.githubusercontent.com/marekjarocinski/jkshocks_update_fed/main/"
JK_SURPRISES_URL = JK_BASE + "source_data/fomc_surprises_jk.csv"
JK_SHOCKS_URL = JK_BASE + "shocks_fed_jk_t.csv"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"

# FRED series used as extra outcomes in the transmission step
FRED_OUTCOMES = {
    "T10YIE": "10y breakeven inflation",
    "DFII10": "10y TIPS real yield",
    "BAA10Y": "Baa corporate minus 10y Treasury",
    "DTWEXBGS": "Broad trade-weighted dollar",
}
# Par yields used to build 5y5y forwards (real from TIPS, nominal from Treasuries)
FRED_FORWARD_INPUTS = {
    "DGS5": "5y Treasury yield",
    "DGS10": "10y Treasury yield",
    "DFII5": "5y TIPS real yield",
}
# Everything downloaded from FRED
FRED_SERIES = {**FRED_OUTCOMES, **FRED_FORWARD_INPUTS, "DTB3": "3-month Treasury bill, discount basis"}
FORWARD_OUTCOMES = {
    "fwd_nominal": "5y5y nominal forward",
    "fwd_expected": "of which: expected rates",
    "fwd_tp": "of which: term premium",
    "fwd_real": "5y5y real forward (TIPS)",
    "fwd_breakeven": "5y5y breakeven inflation",
}

# ACM settings (Adrian, Crump and Moench 2013)
ACM_N_FACTORS = 5
ACM_MAX_MATURITY = 120  # months
ACM_RETURN_MATURITIES = list(range(6, 121, 6))  # bonds used in the return regressions
ACM_START = "1961-06-30"  # NY Fed's published series starts here
REPORT_MATURITIES = [12, 24, 36, 60, 84, 120]  # 1y to 10y, in months
# Which short rate anchors the expected-rate path:
#   "curve" = 1-month yield read off the fitted GSW curve
#   "tbill" = actual 3-month Treasury bill rate from FRED
SHORT_RATE = "tbill"

# Shock settings (Jarocinski and Karadi 2020)
JK_RATE_VARS = ["MP1", "FF4", "ED2", "ED3", "ED4"]
JK_STOCK_VAR = "SP500"
JK_START_YEAR = 1990

# Local projection settings
LP_HORIZONS = [0, 1, 2, 5, 10, 20, 40, 60]  # business days after the meeting
