import numpy as np
import pandas as pd

from tpshock.data.fred import tbill_cc


def test_tbill_discount_to_continuous(tmp_path):
    # a 5% discount rate on a 91-day bill: price 0.98736, continuously compounded yield ~5.10%
    pd.DataFrame({"observation_date": ["2020-01-02"], "DTB3": [5.0]}).to_csv(tmp_path / "DTB3.csv", index=False)
    y = tbill_cc("DTB3", dest_dir=tmp_path).iloc[0]
    price = 1 - 0.05 * 91 / 360
    assert np.isclose(y, -np.log(price) * 365 / 91 * 100)
    assert 5.05 < y < 5.15
