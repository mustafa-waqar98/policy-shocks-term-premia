import numpy as np
import pandas as pd
import pytest

from tpshock.transmission.forwards import forward_5y5y, forward_panel


def test_flat_curve_forward_equals_spot():
    assert forward_5y5y(4.0, 4.0) == pytest.approx(4.0)


def test_upward_curve_forward_is_above_10y():
    # 5 years at 2%, then 5 years at f, must earn the same as 10 years at 3%:  5*2 + 5*f = 10*3
    assert forward_5y5y(2.0, 3.0) == pytest.approx(4.0)


def test_works_on_whole_columns():
    y5, y10 = pd.Series([1.0, 2.0, 3.0]), pd.Series([1.5, 2.5, 3.0])
    assert np.allclose(forward_5y5y(y5, y10), [2.0, 3.0, 3.0])


def test_pieces_add_up():
    idx = pd.date_range("2020-01-01", periods=3)
    daily = pd.DataFrame({"y60": [3.0, 3.1, 3.2], "y120": [3.5, 3.6, 3.8],
                          "rn60": [2.8, 2.8, 2.9], "rn120": [3.0, 3.0, 3.1]}, index=idx)
    daily["tp60"], daily["tp120"] = daily.y60 - daily.rn60, daily.y120 - daily.rn120
    f = forward_panel(daily, pd.DataFrame(index=idx))
    assert np.allclose(f.fwd_nominal, f.fwd_expected + f.fwd_tp)
