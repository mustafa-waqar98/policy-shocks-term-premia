import numpy as np
import pandas as pd

from tpshock.transmission.lp import event_changes, local_projections


def test_lp_recovers_known_response():
    rng = np.random.default_rng(0)
    days = pd.bdate_range("2000-01-03", periods=6000)
    events = days[50:-100:30]
    shocks = pd.DataFrame({"MP": rng.normal(0, 0.05, len(events))}, index=events)
    # outcome jumps by 0.5 x shock on the day and stays there
    jumps = pd.Series(0.0, index=days)
    jumps[events] = 0.5 * shocks.MP.to_numpy()
    y = jumps.cumsum() + np.cumsum(rng.normal(0, 0.002, len(days)))
    irf = local_projections(pd.DataFrame({"y": y}), shocks, ["MP"], [0, 5, 20], scale=1.0, units=1.0)
    assert np.allclose(irf.coef, 0.5, atol=0.05)


def test_event_changes_use_previous_close():
    days = pd.bdate_range("2020-01-01", periods=10)
    daily = pd.DataFrame({"x": np.arange(10.0)}, index=days)
    shocks = pd.DataFrame({"s": [1.0]}, index=[days[4] + pd.Timedelta(hours=14)])
    chg = event_changes(daily, shocks, h=2)
    assert chg.x.iloc[0] == 3.0  # x[6] - x[3]
