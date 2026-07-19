"""Statistical helpers for evaluation reporting.

Every success rate published from this repo must carry a confidence interval;
the Wilson score interval is the standard choice for binomial proportions at
the episode counts used here (n = 20–200), where the normal approximation is
unreliable near 0 % and 100 %.
"""

from __future__ import annotations

import math


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Args:
        successes: number of successful episodes.
        n: total episodes (must be > 0).
        z: normal quantile (1.96 → 95 % interval).

    Returns:
        (low, high) bounds in [0, 1].
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= successes <= n:
        raise ValueError(f"successes={successes} outside [0, {n}]")

    p = successes / n
    denom = 1.0 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1.0 - p) / n + z * z / (4 * n * n))
    low = (centre - margin) / denom
    high = (centre + margin) / denom
    # Pin exact edges (floating error otherwise yields e.g. 0.9999999999999998).
    if successes == 0:
        low = 0.0
    if successes == n:
        high = 1.0
    return max(0.0, low), min(1.0, high)


def format_rate(successes: int, n: int, z: float = 1.96) -> str:
    """Format ``successes/n`` as e.g. ``'18/20 (90.0%, 95% CI 69.9-97.2%)'``."""
    low, high = wilson_ci(successes, n, z)
    rate = successes / n * 100
    return f"{successes}/{n} ({rate:.1f}%, 95% CI {low * 100:.1f}-{high * 100:.1f}%)"
