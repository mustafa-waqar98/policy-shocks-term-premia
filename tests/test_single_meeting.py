import numpy as np

from tpshock.transmission.single_meeting import compare


def test_no_edge_when_meetings_look_like_control():
    rng = np.random.default_rng(0)
    n = 3000
    hit = rng.random(n) < 0.6
    is_fomc = np.zeros(n, bool)
    is_fomc[::30] = True
    clean = ~is_fomc
    r = compare(hit, np.ones(n, bool), is_fomc, clean)
    assert abs(r["fomc_rate"] - r["control_rate"]) < 0.15
    assert r["p_two_sided"] > 0.05


def test_detects_a_real_edge():
    rng = np.random.default_rng(1)
    n = 3000
    is_fomc = np.zeros(n, bool)
    is_fomc[::20] = True
    hit = np.where(is_fomc, rng.random(n) < 0.9, rng.random(n) < 0.5)
    r = compare(hit, np.ones(n, bool), is_fomc, ~is_fomc)
    assert r["fomc_rate"] > r["control_rate"]
    assert r["p_two_sided"] < 0.01
