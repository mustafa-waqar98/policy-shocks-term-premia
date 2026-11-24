import numpy as np
import pandas as pd
import pytest

from tpshock import config
from tpshock.shocks.identify import build_shocks, median_rotation, poor_mans, target_path


def fake_surprises(T=300, seed=0):
    rng = np.random.default_rng(seed)
    target, path, info = rng.normal(0, 0.05, (3, T))
    df = pd.DataFrame({
        "MP1": target + rng.normal(0, 0.005, T),
        "FF4": target + 0.5 * path + rng.normal(0, 0.005, T),
        "ED2": 0.7 * target + 0.8 * path + info + rng.normal(0, 0.005, T),
        "ED3": 0.5 * target + 0.9 * path + info + rng.normal(0, 0.005, T),
        "ED4": 0.4 * target + 1.0 * path + info + rng.normal(0, 0.005, T),
        "SP500": -3 * (target + path) + 4 * info + rng.normal(0, 0.2, T),
    }, index=pd.date_range("1995-01-01", periods=T, freq="45D"))
    return df


def test_shocks_add_up():
    s = build_shocks(fake_surprises())
    assert np.allclose(s.MP_pm + s.CBI_pm, s.pc1)
    assert np.allclose(s.MP_median + s.CBI_median, s.pc1)


def test_median_shocks_are_orthogonal():
    s = fake_surprises()
    u = median_rotation(s.ED4, s.SP500)
    # orthogonal in the uncentred sense, as in JK (surprises have mean zero by design)
    assert abs((u.MP_median * u.CBI_median).sum()) < 1e-10


def test_poor_mans_signs():
    pc1 = pd.Series([0.1, 0.1, -0.1, -0.1])
    sp = pd.Series([-1.0, 1.0, 1.0, -1.0])
    u = poor_mans(pc1, sp)
    assert list(u.MP_pm) == [0.1, 0, -0.1, 0]
    assert list(u.CBI_pm) == [0, 0.1, 0, -0.1]


def test_path_factor_does_not_move_current_rate():
    s = fake_surprises()
    tp = target_path(s)
    assert abs(np.corrcoef(tp.path, s.MP1)[0, 1]) < 1e-8
    assert abs(np.corrcoef(tp.target, tp.path)[0, 1]) < 1e-8
    assert np.polyfit(tp.target, s.MP1 - s.MP1.mean(), 1)[0] == pytest.approx(1.0)


@pytest.mark.network
def test_replicates_published_jk_shocks(tmp_path):
    try:
        raw = pd.read_csv(config.JK_SURPRISES_URL, parse_dates=["start"]).set_index("start")
        pub = pd.read_csv(config.JK_SHOCKS_URL, parse_dates=["start"]).set_index("start")
    except Exception:
        pytest.skip("no network")
    ours = build_shocks(raw)
    ours.index, pub.index = ours.index.floor("min"), pub.index.floor("min")
    j = ours.join(pub, rsuffix="_pub", how="inner")
    for c in ["pc1", "MP_pm", "CBI_pm", "MP_median", "CBI_median"]:
        assert (j[c] - j[c + "_pub"]).abs().max() < 1e-6
