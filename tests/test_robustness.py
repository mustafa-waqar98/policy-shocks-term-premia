import numpy as np
import pandas as pd

from tpshock.transmission.robustness import make_variants


def fake_shocks():
    idx = pd.to_datetime(["2007-06-28", "2008-09-16", "2020-03-15", "2020-06-10", "2024-12-18"])
    return pd.DataFrame({"target": np.arange(5.0), "path": np.arange(5.0),
                         "scheduled": [True, True, False, True, True]}, index=idx)


def test_every_check_keeps_a_subset_of_meetings():
    shocks = fake_shocks()
    for name, sample in make_variants(shocks).items():
        assert sample.index.isin(shocks.index).all(), name


def test_there_is_a_check_without_2020():
    variants = make_variants(fake_shocks())
    ex2020 = [v for name, v in variants.items() if "2020" in name]
    assert ex2020, "add a robustness check that drops the 2020 meetings"
    assert not (ex2020[0].index.year == 2020).any()
    assert len(ex2020[0]) == 3
