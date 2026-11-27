import pandas as pd

from tpshock.shocks.identify import scheduled_flag


def test_calendar_overrides_event_label():
    times = pd.DatetimeIndex(["2008-10-29 14:15", "2008-11-25 08:15", "2020-03-03 10:00", "1991-01-08 10:00"])
    label = pd.Series([True, True, True, False], index=times)  # what the JK labels say
    flag = scheduled_flag(times, label)
    assert flag.tolist() == [True, False, False, False]  # 1991 keeps the JK label
