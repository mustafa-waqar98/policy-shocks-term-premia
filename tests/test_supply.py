import numpy as np
import pandas as pd

from tpshock.supply.measure import duration_weight, monthly_supply, private_holdings


def test_duration_weight_is_one_at_ten_years():
    assert np.isclose(duration_weight(10), 1.0)
    assert duration_weight(30) > 1.0 and duration_weight(2) < 0.3


def _mspd():
    d = pd.Timestamp("2010-01-31")
    return pd.DataFrame({
        "date": [d, d, d],
        "type": ["Notes", "Bonds", "Floating Rate Notes"],
        "cusip": ["912828AA1", "912810AA1", "912828ZZ1"],
        "maturity": [d + pd.Timedelta(days=round(2 * 365.25)), d + pd.Timedelta(days=round(10 * 365.25)),
                     d + pd.Timedelta(days=730)],
        "outstanding": [100.0, 50.0, 20.0],
    })


def test_fed_holdings_come_out_of_private_supply():
    soma = pd.DataFrame({"month": [pd.Period("2010-01", "M")], "cusip": ["912810AA1"], "fed": [20.0]})
    h = private_holdings(_mspd(), soma)
    assert h.set_index("cusip").private.to_dict() == {"912828AA1": 100.0, "912810AA1": 30.0, "912828ZZ1": 20.0}
    gdp = pd.Series([1000.0], index=[pd.Timestamp("2010-01-01")])
    m = monthly_supply(h, gdp).iloc[0]
    assert np.isclose(m.private_bn, 150.0)
    assert np.isclose(m.debt_years_private, 100 * 2 + 30 * 10 + 20 * 7 / 365, rtol=1e-3)  # FRNs count as a week
    assert np.isclose(m.debt_years_private_gdp, m.debt_years_private / 1000)
    assert 0 < m.fed_share_of_duration < 1


def test_private_unknown_before_soma_starts():
    soma = pd.DataFrame({"month": [pd.Period("2011-01", "M")], "cusip": ["912810AA1"], "fed": [1.0]})
    h = private_holdings(_mspd(), soma)
    assert h.private.isna().all()
