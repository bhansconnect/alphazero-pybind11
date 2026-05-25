"""Gumbel AlphaZero math primitives -- numpy reference + unit tests.

These tests pin the contract that the C++ implementation in src/mcts.cc must
match exactly. The reference functions below are the spec; the tests verify
they behave per Danihelka et al. 2022 ("Policy improvement by planning with
Gumbel", ICLR 2022) and DeepMind's mctx reference implementation.

Run: uv run python -m pytest src/test_gumbel.py -v
"""
import math

import numpy as np
import pytest


# ----------------------------------------------------------------------------
# Reference implementations -- the C++ MCTS code must reproduce these exactly.
# ----------------------------------------------------------------------------

def _ref_sample_gumbel(num_actions, rng):
    """Sample iid Gumbel(0) variables: g = -log(-log(u))."""
    u = rng.uniform(low=np.finfo(np.float64).tiny, high=1.0, size=num_actions)
    return -np.log(-np.log(u))


def _ref_effective_m(gumbel_m, num_legal, num_sims):
    """Runtime-effective m. See plan: cap to legal action count AND to sim
    budget (sequential halving needs >=1 visit per considered action)."""
    return max(1, min(gumbel_m, num_legal, num_sims))


def _ref_gumbel_topk(logits, gumbel, k):
    """Gumbel-Top-K trick: sampling without replacement. Returns the indices
    of the top-k actions ranked by (gumbel + logits)."""
    scores = gumbel + logits
    return np.argsort(-scores)[:k]


def _ref_v_mix(raw_value, qvalues, visit_counts, prior_probs):
    """Mixed value approximation (paper Appendix D, mctx _compute_mixed_value).

        v_mix = (raw_value + N_tot * weighted_q) / (N_tot + 1)
        weighted_q = sum_{a: N(a)>0} pi(a) * q(a) / sum_{a: N(a)>0} pi(a)

    When no actions have been visited, v_mix == raw_value.
    """
    visited = visit_counts > 0
    sum_visits = float(visit_counts.sum())
    if not visited.any():
        return float(raw_value)
    sum_probs = float(prior_probs[visited].sum())
    if sum_probs <= 0:
        return float(raw_value)
    weighted_q = float((prior_probs[visited] * qvalues[visited]).sum() / sum_probs)
    return (float(raw_value) + sum_visits * weighted_q) / (sum_visits + 1.0)


def _ref_completed_q(qvalues, visit_counts, v_mix):
    """completedQ[a] = q[a] if N(a)>0 else v_mix.  (paper Eq 10)"""
    return np.where(visit_counts > 0, qvalues, v_mix)


def _ref_sigma(completed_q, max_visit, c_visit, c_scale):
    """Paper Eq 8:  sigma(q_hat) = (c_visit + max_b N(b)) * c_scale * q_hat."""
    return (c_visit + float(max_visit)) * c_scale * completed_q


def _ref_improved_policy(prior_logits, completed_q, max_visit, c_visit, c_scale):
    """Paper Eq 11:  pi' = softmax(logits + sigma(completedQ)).

    Numerically stable softmax (subtract max before exp).
    """
    sigma = _ref_sigma(completed_q, max_visit, c_visit, c_scale)
    z = prior_logits + sigma
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def _ref_seq_halving_phase_plan(m, n):
    """Per-phase plan, paper Figure 1 schedule.

    Returns list of (num_candidates_in_phase, visits_per_candidate). Each
    phase: round-robin all surviving candidates v_per times each. After
    phase, halve survivors (rank by gumbel+logits+sigma(q_hat)). The FINAL
    phase distributes all remaining sims so total visits == n.

    For m=16, n=200: returns [(16,3), (8,6), (4,12), (2,28)] → visit dist
    [49,49,21,21,9,9,9,9,3,3,3,3,3,3,3,3]  (sum = 200 ✓, matches Figure 1).

    Low-budget cases (e.g. m=4, n=4) collapse to a single phase because no
    sims are left after giving each candidate one visit -- halving then has
    no information to act on. The runtime effective_m cap (separate fn)
    ensures n >= m before we get here.
    """
    if m <= 1:
        return [(1, n)]
    log2m = int(math.ceil(math.log2(m)))
    base_v = max(1, n // (log2m * m))   # visits per candidate in phase 0
    phases = []
    sims_used = 0
    num_c = m
    for phase_idx in range(log2m):
        remaining = n - sims_used
        if remaining <= 0:
            break
        is_final = (phase_idx == log2m - 1)
        if is_final:
            v_per = max(1, remaining // num_c)
        else:
            v_per = base_v * (2 ** phase_idx)   # doubles each phase
        # Don't overrun the budget. If we can't afford v_per for every
        # candidate, truncate v_per (and possibly num_c if v_per would be 0).
        if num_c * v_per > remaining:
            v_per = remaining // num_c
            if v_per == 0:
                # Not enough budget to give every candidate a visit; give 1
                # visit to as many as the budget allows, and stop.
                num_c = remaining
                v_per = 1
        phases.append((num_c, v_per))
        sims_used += num_c * v_per
        num_c = max(1, num_c // 2)
    return phases


def _simulate_seq_halving_visit_counts(m, n):
    """Run the paper schedule and return visit counts per candidate.

    Survivors after each phase are ranked by accumulated visits (which is a
    monotone proxy for q in this test -- in a real search, survivors are
    ranked by gumbel + logits + sigma(q_hat)). Returns length-m int array.
    """
    phases = _ref_seq_halving_phase_plan(m, n)
    visits = np.zeros(m, dtype=np.int64)
    # Survivors begin as indices 0..m-1 (treated as the gumbel-top-m
    # ranking; lower index = higher initial score).
    survivors = list(range(m))
    for num_c, v_per in phases:
        active = survivors[:num_c]
        for s in active:
            visits[s] += v_per
        # Halve survivors. In a real search this is rank by score; here we
        # keep the lower-indexed half so the test is deterministic.
        if num_c > 2:
            survivors = active[: max(2, num_c // 2)]
        else:
            survivors = active
    return visits


# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------

def test_sigma_formula():
    """sigma values exactly match (c_visit + max_n) * c_scale * q."""
    q = np.array([0.1, 0.5, 0.9, 0.0])
    sigma = _ref_sigma(q, max_visit=10, c_visit=50.0, c_scale=1.0)
    expected = (50.0 + 10.0) * 1.0 * q
    np.testing.assert_allclose(sigma, expected)
    # Scale linearly with max_visit + c_visit.
    sigma2 = _ref_sigma(q, max_visit=100, c_visit=50.0, c_scale=1.0)
    np.testing.assert_allclose(sigma2, 150.0 * q)
    # c_scale scales linearly.
    sigma3 = _ref_sigma(q, max_visit=10, c_visit=50.0, c_scale=0.5)
    np.testing.assert_allclose(sigma3, 60.0 * 0.5 * q)


def test_v_mix_no_visits_equals_raw():
    """With zero visits everywhere, v_mix reduces to raw_value."""
    q = np.array([0.0, 0.0, 0.0, 0.0])
    N = np.array([0, 0, 0, 0])
    prior = np.array([0.25, 0.25, 0.25, 0.25])
    assert _ref_v_mix(0.7, q, N, prior) == pytest.approx(0.7)


def test_v_mix_one_visit_pulls_toward_q():
    """A single visit to a high-q action pulls v_mix above raw_value."""
    q = np.array([0.0, 0.0, 0.8, 0.0])
    N = np.array([0, 0, 1, 0])
    prior = np.array([0.25, 0.25, 0.25, 0.25])
    # weighted_q = 0.25 * 0.8 / 0.25 = 0.8
    # v_mix = (0.5 + 1 * 0.8) / (1 + 1) = 1.3 / 2 = 0.65
    assert _ref_v_mix(0.5, q, N, prior) == pytest.approx(0.65)


def test_v_mix_many_visits_dominates_raw():
    """With many visits the weighted_q dominates raw_value."""
    q = np.array([0.9, 0.9, 0.9, 0.9])
    N = np.array([100, 100, 100, 100])
    prior = np.array([0.25, 0.25, 0.25, 0.25])
    # weighted_q = 0.9, v_mix ≈ (0.1 + 400 * 0.9) / 401 ≈ 0.898
    v_mix = _ref_v_mix(0.1, q, N, prior)
    assert v_mix == pytest.approx(0.898, abs=0.005)


def test_completed_q_unvisited_uses_v_mix():
    """Visited slots keep their q; unvisited get v_mix."""
    q = np.array([0.3, 0.7, 0.0, 0.5])
    N = np.array([5, 3, 0, 1])
    v_mix = 0.42
    cq = _ref_completed_q(q, N, v_mix)
    np.testing.assert_allclose(cq, [0.3, 0.7, 0.42, 0.5])


def test_improved_policy_no_visits_equals_prior():
    """When N=0 everywhere, sigma adds a constant offset -> softmax washes
    it out, so pi' equals softmax(logits)."""
    logits = np.array([0.1, 0.4, -0.2, 1.0])
    q = np.zeros(4)
    N = np.zeros(4, dtype=np.int64)
    prior = np.exp(logits) / np.exp(logits).sum()
    v_mix = _ref_v_mix(0.0, q, N, prior)
    cq = _ref_completed_q(q, N, v_mix)
    pi_prime = _ref_improved_policy(logits, cq, max_visit=0, c_visit=50.0, c_scale=1.0)
    np.testing.assert_allclose(pi_prime, prior, atol=1e-6)


def test_improved_policy_high_q_concentrates():
    """An action with high q + many visits captures most of the probability."""
    logits = np.array([0.0, 0.0, 0.0, 0.0])  # uniform prior
    q = np.array([0.0, 0.0, 0.9, 0.0])
    N = np.array([1, 1, 50, 1])  # action 2 well-explored, others tried
    prior = np.array([0.25, 0.25, 0.25, 0.25])
    v_mix = _ref_v_mix(0.2, q, N, prior)
    cq = _ref_completed_q(q, N, v_mix)
    pi_prime = _ref_improved_policy(logits, cq, max_visit=50, c_visit=50.0, c_scale=1.0)
    # action 2 should dominate
    assert pi_prime[2] > 0.99


def test_improved_policy_sums_to_one():
    """pi' is a probability distribution."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        n = rng.integers(2, 30)
        logits = rng.normal(0, 1, n)
        q = rng.uniform(0, 1, n)
        N = rng.integers(0, 50, n)
        prior = np.exp(logits) / np.exp(logits).sum()
        v_mix = _ref_v_mix(rng.uniform(), q, N, prior)
        cq = _ref_completed_q(q, N, v_mix)
        pi = _ref_improved_policy(logits, cq, max_visit=int(N.max()),
                                  c_visit=50.0, c_scale=1.0)
        assert pi.sum() == pytest.approx(1.0)
        assert (pi >= 0).all()


def test_seq_halving_phase_plan_paper_figure1():
    """Paper Figure 1: m=16, n=200 -> phases [(16,3), (8,6), (4,12), (2,28)]
    summing to 200 sims, with surviving candidates accruing 3+6+12+28=49."""
    phases = _ref_seq_halving_phase_plan(16, 200)
    assert phases == [(16, 3), (8, 6), (4, 12), (2, 28)]
    assert sum(num_c * v for num_c, v in phases) == 200
    assert sum(v for _, v in phases) == 49


def test_seq_halving_visit_distribution_paper_figure1():
    """Visit count distribution after running the m=16 n=200 schedule:
    8 candidates with N=3, 4 with N=9, 2 with N=21, 2 with N=49."""
    visits = _simulate_seq_halving_visit_counts(m=16, n=200)
    assert visits.sum() == 200
    sorted_desc = np.sort(visits)[::-1].tolist()
    assert sorted_desc == [49, 49, 21, 21, 9, 9, 9, 9, 3, 3, 3, 3, 3, 3, 3, 3]


def test_seq_halving_low_budget_no_crash():
    """m=16, n=8: effective_m caps to 8; schedule finishes without stalling."""
    m_eff = _ref_effective_m(gumbel_m=16, num_legal=20, num_sims=8)
    assert m_eff == 8
    visits = _simulate_seq_halving_visit_counts(m=m_eff, n=8)
    assert visits.sum() == 8
    # All 8 candidates should get at least 1 visit.
    assert (visits[:m_eff] >= 1).all()


def test_seq_halving_m_equals_one():
    """Degenerate m=1: every sim goes to the single candidate."""
    visits = _simulate_seq_halving_visit_counts(m=1, n=10)
    assert visits[0] == 10


def test_seq_halving_phase_plan_totals_match_budget():
    """Total visits in the phase plan equal n for many (m, n) combos."""
    for m in [2, 4, 8, 16, 32]:
        for n in [m, m * 2, m * 4, 50, 200, 500]:
            if n < m:
                continue
            phases = _ref_seq_halving_phase_plan(m, n)
            total = sum(num_c * v for num_c, v in phases)
            # Final phase distributes remainder; may be slightly off when
            # (n - sims_used) is not divisible by num_c. Allow ≤ num_c slack.
            assert total <= n
            assert total >= n - phases[-1][0], f"m={m} n={n} phases={phases}"


def test_effective_m_cap_by_num_legal():
    """num_legal < gumbel_m: cap to num_legal."""
    assert _ref_effective_m(gumbel_m=16, num_legal=5, num_sims=32) == 5
    assert _ref_effective_m(gumbel_m=16, num_legal=8, num_sims=240) == 8


def test_effective_m_cap_by_num_sims():
    """num_sims < gumbel_m: cap to num_sims (need >=1 visit per candidate)."""
    assert _ref_effective_m(gumbel_m=16, num_legal=50, num_sims=8) == 8
    assert _ref_effective_m(gumbel_m=16, num_legal=362, num_sims=4) == 4


def test_effective_m_no_cap_when_budget_large():
    """When both caps are loose, return gumbel_m."""
    assert _ref_effective_m(gumbel_m=16, num_legal=50, num_sims=240) == 16


def test_effective_m_minimum_one():
    """Pathological inputs (zero) don't return zero -- minimum is 1."""
    assert _ref_effective_m(gumbel_m=0, num_legal=10, num_sims=10) == 1
    assert _ref_effective_m(gumbel_m=16, num_legal=0, num_sims=10) == 1


def test_gumbel_topk_distribution():
    """10k Gumbel-Top-K samples should produce empirical top-1 frequencies
    that match softmax(logits) within ~2%."""
    rng = np.random.default_rng(0)
    probs = np.array([0.5, 0.3, 0.15, 0.04, 0.01])
    logits = np.log(probs)
    counts = np.zeros(5, dtype=np.int64)
    n_trials = 20000
    for _ in range(n_trials):
        g = _ref_sample_gumbel(5, rng)
        top1 = _ref_gumbel_topk(logits, g, 1)[0]
        counts[top1] += 1
    empirical = counts / n_trials
    # Within ~2% of target probs.
    np.testing.assert_allclose(empirical, probs, atol=0.02)


def test_gumbel_topk_without_replacement():
    """The k returned indices are distinct."""
    rng = np.random.default_rng(0)
    logits = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    for _ in range(50):
        g = _ref_sample_gumbel(5, rng)
        idx = _ref_gumbel_topk(logits, g, 3)
        assert len(set(idx.tolist())) == 3


def test_sample_gumbel_finite_and_centered():
    """Samples are finite. Mean ~= Euler-Mascheroni constant (~0.5772)."""
    rng = np.random.default_rng(0)
    samples = _ref_sample_gumbel(100000, rng)
    assert np.isfinite(samples).all()
    # Gumbel(0) has mean = euler-mascheroni ~ 0.5772, variance pi^2/6.
    assert abs(samples.mean() - 0.5772) < 0.02


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
