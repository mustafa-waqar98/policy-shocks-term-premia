import numpy as np
import pandas as pd

from tpshock.data.gsw import month_end, svensson


def params(**kw):
    base = dict(BETA0=4.0, BETA1=-2.0, BETA2=1.0, BETA3=0.5, TAU1=1.5, TAU2=8.0)
    base.update(kw)
    return pd.DataFrame([base], index=pd.to_datetime(["2020-01-31"]))


def test_short_end_is_beta0_plus_beta1():
    y = svensson(params(), [0.001])
    assert np.isclose(y.iloc[0, 0], 2.0, atol=1e-3)


def test_long_end_converges_to_beta0():
    y = svensson(params(), [12 * 1000])
    assert np.isclose(y.iloc[0, 0], 4.0, atol=0.01)


def test_month_end_picks_last_day():
    idx = pd.to_datetime(["2020-01-02", "2020-01-31", "2020-02-03", "2020-02-28"])
    df = pd.DataFrame({"x": [1, 2, 3, 4]}, index=idx)
    assert list(month_end(df).x) == [2, 4]
