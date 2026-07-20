import math

import pytest

from humanoid_vla.stats import format_rate, wilson_ci


def test_wilson_known_value():
    # 18/20 → Wilson 95% CI ≈ (0.699, 0.972)
    low, high = wilson_ci(18, 20)
    assert math.isclose(low, 0.6989, abs_tol=1e-3)
    assert math.isclose(high, 0.9721, abs_tol=1e-3)


def test_wilson_bounds_and_edges():
    low, high = wilson_ci(0, 20)
    assert low == 0.0 and 0.0 < high < 0.25
    low, high = wilson_ci(20, 20)
    assert 0.75 < low < 1.0 and high == 1.0


def test_wilson_narrows_with_n():
    w20 = wilson_ci(10, 20)
    w200 = wilson_ci(100, 200)
    assert (w200[1] - w200[0]) < (w20[1] - w20[0])


def test_wilson_invalid():
    with pytest.raises(ValueError):
        wilson_ci(1, 0)
    with pytest.raises(ValueError):
        wilson_ci(5, 4)


def test_format_rate():
    s = format_rate(18, 20)
    assert s.startswith("18/20 (90.0%")
    assert "CI" in s
