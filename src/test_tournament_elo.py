"""Test that calc_elo produces reasonable ratings and doesn't explode."""

import math
import numpy as np
from tournament import calc_elo


# Win matrix from a real 20-player monrad tournament that triggered the bug.
# nan = pair never played. Values are win rates for row player vs column player.
WIN_MATRIX = np.array([
    [  np.nan, 0.139, 0.033, 0.014,   np.nan,   np.nan, 0.005,   np.nan,   np.nan,   np.nan, 0.005,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan],
    [0.861,   np.nan, 0.171, 0.086,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan, 0.025,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan, 0.018],
    [0.967, 0.829,   np.nan, 0.333,   np.nan,   np.nan,   np.nan, 0.131,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan],
    [0.986, 0.914, 0.667,   np.nan, 0.333,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan, 0.217,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan],
    [  np.nan,   np.nan,   np.nan, 0.667,   np.nan, 0.414,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan, 0.277,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan],
    [  np.nan,   np.nan,   np.nan,   np.nan, 0.586,   np.nan, 0.401,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan, 0.333,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan],
    [0.995,   np.nan,   np.nan,   np.nan,   np.nan, 0.599,   np.nan, 0.478,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan, 0.364,   np.nan,   np.nan,   np.nan,   np.nan],
    [  np.nan,   np.nan, 0.869,   np.nan,   np.nan,   np.nan, 0.522,   np.nan, 0.468,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan, 0.367,   np.nan,   np.nan,   np.nan],
    [  np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan, 0.532,   np.nan, 0.452,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan, 0.385,   np.nan,   np.nan],
    [  np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan, 0.548,   np.nan, 0.425,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan, 0.434,   np.nan],
    [0.995,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan, 0.575,   np.nan, 0.412,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan, 0.354],
    [  np.nan, 0.975,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan, 0.588,   np.nan, 0.445,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan],
    [  np.nan,   np.nan,   np.nan, 0.783,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan, 0.555,   np.nan, 0.423,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan],
    [  np.nan,   np.nan,   np.nan,   np.nan, 0.723,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan, 0.577,   np.nan, 0.467,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan],
    [  np.nan,   np.nan,   np.nan,   np.nan,   np.nan, 0.667,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan, 0.533,   np.nan, 0.504,   np.nan,   np.nan,   np.nan,   np.nan],
    [  np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan, 0.636,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan, 0.496,   np.nan, 0.515,   np.nan,   np.nan,   np.nan],
    [  np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan, 0.633,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan, 0.485,   np.nan, 0.517,   np.nan,   np.nan],
    [  np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan, 0.615,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan, 0.483,   np.nan, 0.515,   np.nan],
    [  np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan, 0.566,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan, 0.485,   np.nan, 0.524],
    [  np.nan, 0.982, 0.911,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan, 0.601, 0.570,   np.nan,   np.nan,   np.nan, 0.520, 0.512, 0.485,   np.nan,   np.nan,   np.nan],
])


def test_elo_values_reasonable():
    """ELO ratings should stay within a reasonable range, not explode to ~730k."""
    past_elo = np.zeros(WIN_MATRIX.shape[0])
    elo = calc_elo(past_elo, WIN_MATRIX)

    print("\n--- Corrected ELO Leaderboard ---")
    ranked = sorted(enumerate(elo), key=lambda x: x[1], reverse=True)
    for rank, (idx, rating) in enumerate(ranked, 1):
        print(f"  {rank:2d}. Player {idx:2d}: {rating:7.1f}")
    print(f"  Max ELO: {max(elo):.1f}")
    print(f"  Min ELO: {min(elo):.1f}")
    print(f"  Spread:  {max(elo) - min(elo):.1f}")

    # All ratings must stay below 5000 (the bug produced ~730,000)
    assert max(elo) < 5000, f"ELO exploded: max = {max(elo):.1f}"

    # Weakest player should be anchored at 0
    assert min(elo) == 0, f"Min ELO should be 0, got {min(elo):.1f}"


def test_elo_ordering_roughly_monotonic():
    """Earlier players (weaker networks) should generally have lower ELO."""
    past_elo = np.zeros(WIN_MATRIX.shape[0])
    elo = calc_elo(past_elo, WIN_MATRIX)

    # Players 0, 1, 2 should be among the weakest (bottom half)
    n = len(elo)
    median_elo = sorted(elo)[n // 2]
    for i in [0, 1, 2]:
        assert elo[i] < median_elo, (
            f"Player {i} (early/weak) has ELO {elo[i]:.1f} >= median {median_elo:.1f}"
        )
