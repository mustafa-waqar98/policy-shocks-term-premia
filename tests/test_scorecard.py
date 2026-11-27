import numpy as np
import pandas as pd

from tpshock.transmission.scorecard import HORIZON, score, tally


def _fred(days, y2, dfii5, dfii10):
    return pd.DataFrame({"DGS2": y2, "DFII5": dfii5, "DFII10": dfii10}, index=days)


def test_hit_miss_and_open_windows():
    days = pd.bdate_range("2026-01-01", periods=80)
    n = len(days)
    y2 = np.full(n, 4.00)
    real5, real10 = np.full(n, 2.00), np.full(n, 2.50)
    # meeting A at day 10: 2y up 5bp, real forward rises afterwards -> hit
    y2[10:] += 0.05
    real10[15:] += 0.05
    # meeting B at day 40: 2y down 1bp, real forward rises -> miss, flagged under 3bp
    y2[40:] -= 0.01
    real10[45:] += 0.05
    # meeting C at day 70: window still open
    card = score(_fred(days, y2, real5, real10),
                 [str(days[10].date()), str(days[40].date()), str(days[70].date()), "2027-01-27"],
                 today=days[-1])
    assert list(card.result[:2]) == ["hit", "miss"]
    assert card.big_move.iloc[0] and not card.big_move.iloc[1]
    assert card.result.iloc[2].startswith("open")
    assert card.result.iloc[3] == "upcoming"
    assert card.window_end.iloc[0] == str(days[10 + HORIZON].date())
    t = tally(card)
    assert (t["scored"], t["hits"], t["scored_big"], t["hits_big"]) == (2, 1, 1, 1)


def test_no_2y_move_is_a_miss():
    days = pd.bdate_range("2026-01-01", periods=40)
    n = len(days)
    real10 = np.full(n, 2.5)
    real10[12:] += 0.1
    card = score(_fred(days, np.full(n, 4.0), np.full(n, 2.0), real10), [str(days[10].date())],
                 today=days[-1])
    assert card.result.iloc[0] == "miss (2y unchanged)"
