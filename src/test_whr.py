"""Unit tests for whr_refit in game_runner.py.

Run with: uv run python -m pytest src/test_whr.py -v
"""
import math
import sys

import numpy as np
import pytest

from game_runner import _ELO_ALPHA, whr_refit


def _expected_rate(r_i, r_j):
    """BT win probability of player with rating r_i vs r_j."""
    x = _ELO_ALPHA * (r_i - r_j)
    return 1.0 / (1.0 + math.exp(-x))


def _build_wld_from_truth(true_elos, n_games_per_pair=100, draw_rate=0.0):
    """Build (wins, draws) matrices implied by a known elo ladder.

    For each ordered pair (i, j) we play n_games_per_pair games. The
    expected i-win rate is the BT formula. We round to integers so the
    matrices are exact counts. Draws are split evenly between sides.
    """
    n = len(true_elos)
    wins = np.zeros((n, n), dtype=np.float64)
    draws = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            ng = n_games_per_pair
            nd = int(round(ng * draw_rate))
            decisive = ng - nd
            p_i = _expected_rate(true_elos[i], true_elos[j])
            i_wins = int(round(decisive * p_i))
            j_wins = decisive - i_wins
            wins[i, j] = i_wins
            wins[j, i] = j_wins
            draws[i, j] = nd
            draws[j, i] = nd
    return wins, draws


def test_linear_chain_identity():
    """Refit recovers a linear elo ladder when given BT-implied game counts."""
    true_elos = np.linspace(0, 900, 10)
    wins, draws = _build_wld_from_truth(true_elos, n_games_per_pair=200)
    out = whr_refit(wins, draws, max_sweeps=500, tol=1e-4)

    # Anchor at 0; compare relative spacing to truth.
    out_rel = out - out[0]
    true_rel = true_elos - true_elos[0]
    np.testing.assert_allclose(out_rel, true_rel, atol=5.0)


def test_saturated_row_tolerance():
    """A row where one agent wins every game produces a finite (large)
    rating, no nan/inf, and the saturated agent ranks above the rest."""
    n = 5
    # All agents play; agent 0 wins all 100 games vs each of agents 1..4.
    # Agents 1..4 are tied 50/50 with each other.
    ng = 100
    wins = np.zeros((n, n), dtype=np.float64)
    for i in range(1, n):
        wins[0, i] = ng  # 0 beats i, all games
        wins[i, 0] = 0
    for i in range(1, n):
        for j in range(i + 1, n):
            wins[i, j] = ng // 2
            wins[j, i] = ng - ng // 2
    out = whr_refit(wins, anchor=1, max_sweeps=500, tol=0.01)
    assert np.all(np.isfinite(out)), f"non-finite ratings: {out}"
    # Agent 0 should rise far above agents 1..4.
    assert out[0] > out[1:].max() + 500, f"saturated winner not separated: {out}"


def test_draws_half_credit():
    """Half-draw convention: 100% draws between two agents = equal ratings."""
    n = 4
    wins = np.zeros((n, n), dtype=np.float64)
    draws = np.zeros((n, n), dtype=np.float64)
    # Agents 0 and 1: all draws.
    draws[0, 1] = draws[1, 0] = 50
    # Agents 0 and 2: agent 0 wins 60/40.
    wins[0, 2] = 60
    wins[2, 0] = 40
    # Agents 1 and 3: agent 1 wins 60/40 (by symmetry, 1 should match 0).
    wins[1, 3] = 60
    wins[3, 1] = 40
    out = whr_refit(wins, draws, max_sweeps=500, tol=1e-4)
    # 0 and 1 tied (all draws between them, same wr against equally weak opponents).
    assert abs(out[0] - out[1]) < 10, f"draws didn't tie 0 and 1: {out}"


def test_anchor_invariance():
    """Rating differences are anchor-independent (only the offset shifts)."""
    n = 8
    true_elos = np.array([0, 50, 200, 350, 500, 650, 800, 900], dtype=float)
    wins, draws = _build_wld_from_truth(true_elos, n_games_per_pair=200)
    out0 = whr_refit(wins, draws, max_sweeps=500, tol=1e-4, anchor=0)
    out5 = whr_refit(wins, draws, max_sweeps=500, tol=1e-4, anchor=5)
    diffs0 = out0 - out0[0]
    diffs5 = out5 - out5[0]
    np.testing.assert_allclose(diffs0, diffs5, atol=2.0)


def test_more_games_tighter_fit():
    """With more games per pair the BT MLE should land closer to the
    truth (less noise in the empirical rates)."""
    rng = np.random.default_rng(0)
    true_elos = np.linspace(0, 600, 6)
    # Few games -> noisy empirical rates -> bigger error.
    def err(ngames):
        wins = np.zeros((6, 6), dtype=np.float64)
        for i in range(6):
            for j in range(i + 1, 6):
                p = _expected_rate(true_elos[i], true_elos[j])
                k = int(rng.binomial(ngames, p))
                wins[i, j] = k
                wins[j, i] = ngames - k
        out = whr_refit(wins, max_sweeps=500, tol=1e-4)
        out_rel = out - out[0]
        return float(np.max(np.abs(out_rel - (true_elos - true_elos[0]))))

    err_few = err(20)
    err_many = err(2000)
    assert err_many <= err_few, (
        f"More games didn't reduce error: 20={err_few:.1f}, 2000={err_many:.1f}"
    )


def test_convergence():
    """A moderate sparse matrix converges within the default max_sweeps."""
    rng = np.random.default_rng(0)
    n = 30
    true_elos = np.linspace(0, 1500, n) + rng.normal(0, 20, n)
    wins, _ = _build_wld_from_truth(true_elos, n_games_per_pair=64)
    # Drop ~70% of pairs to simulate sparsity.
    sparsity = rng.random((n, n)) < 0.3
    np.fill_diagonal(sparsity, False)
    sparsity = sparsity | sparsity.T
    wins = np.where(sparsity, wins, 0.0)
    out = whr_refit(wins, max_sweeps=200, tol=0.1)
    assert np.all(np.isfinite(out))


def test_empty_input():
    """All-zero matrices return zeros without crashing."""
    n = 5
    wins = np.zeros((n, n), dtype=np.float64)
    out = whr_refit(wins, max_sweeps=10)
    assert out.shape == (n,)
    np.testing.assert_array_equal(out, np.zeros(n))


def test_single_pair():
    """One pair of agents with some games gives them ratings related by
    the BT inverse of the win rate. Other agents stay at the anchor."""
    n = 4
    wins = np.zeros((n, n), dtype=np.float64)
    # 60 wins for 1, 40 wins for 0 -> 60% rate for agent 1.
    wins[1, 0] = 60
    wins[0, 1] = 40
    out = whr_refit(wins, max_sweeps=500, tol=1e-4)
    assert np.all(np.isfinite(out))
    # BT inverse: rate = sigmoid(α*Δ) -> 0.6 -> Δ = log(0.6/0.4) / α ≈ 70.4 elo
    expected_diff = math.log(0.6 / 0.4) / _ELO_ALPHA
    assert abs((out[1] - out[0]) - expected_diff) < 5, (
        f"BT-inverse mismatch: {out}, expected Δ ≈ {expected_diff:.1f}"
    )
    # Agents 2 and 3 had no games, stay at anchor.
    assert out[2] == 0 and out[3] == 0


def test_rejects_mismatched_shapes():
    """wins and draws must have matching square shapes."""
    wins = np.zeros((4, 4))
    bad_draws = np.zeros((3, 3))
    with pytest.raises(ValueError):
        whr_refit(wins, bad_draws)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
