"""Tests for mcts_analysis.py -- statistical functions, calc_temp, imports, PIO gap, ECE, VIR."""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import TrainConfig, GAME_REGISTRY
from mcts_analysis import (
    jensen_shannon_divergence,
    total_variation,
    hellinger_distance,
    top_k_agreement,
    kl_divergence,
    policy_entropy,
    calc_temp,
    calc_temp_selfplay,
    get_available_games,
    entry_label,
    entry_sort_key,
    compute_scaling_report,
)


# --- Statistical function tests (pure math, no GPU) ---


def test_jsd_identical_distributions():
    """JSD of identical distributions is 0."""
    p = np.array([0.5, 0.3, 0.2])
    assert jensen_shannon_divergence(p, p) == pytest.approx(0.0, abs=1e-10)


def test_jsd_orthogonal_distributions():
    """JSD of non-overlapping distributions is ln(2)."""
    p = np.array([1.0, 0.0])
    q = np.array([0.0, 1.0])
    assert jensen_shannon_divergence(p, q) == pytest.approx(math.log(2), abs=1e-6)


def test_jsd_symmetric():
    """JSD(p, q) == JSD(q, p)."""
    p = np.array([0.7, 0.2, 0.1])
    q = np.array([0.3, 0.4, 0.3])
    assert jensen_shannon_divergence(p, q) == pytest.approx(jensen_shannon_divergence(q, p))


def test_total_variation_identical():
    """TV of identical distributions is 0."""
    p = np.array([0.5, 0.3, 0.2])
    assert total_variation(p, p) == pytest.approx(0.0)


def test_total_variation_orthogonal():
    """TV of non-overlapping distributions is 1."""
    p = np.array([1.0, 0.0])
    q = np.array([0.0, 1.0])
    assert total_variation(p, q) == pytest.approx(1.0)


def test_hellinger_identical():
    """Hellinger of identical distributions is 0."""
    p = np.array([0.5, 0.3, 0.2])
    assert hellinger_distance(p, p) == pytest.approx(0.0, abs=1e-10)


def test_hellinger_bounded():
    """Hellinger distance is in [0, 1]."""
    p = np.array([0.9, 0.05, 0.05])
    q = np.array([0.1, 0.5, 0.4])
    h = hellinger_distance(p, q)
    assert 0.0 <= h <= 1.0


def test_top_k_agreement_identical():
    """Top-k agreement of identical distributions is 1.0."""
    p = np.array([0.5, 0.3, 0.2])
    assert top_k_agreement(p, p, 1) == 1.0
    assert top_k_agreement(p, p, 2) == 1.0


def test_top_k_agreement_disjoint():
    """Top-k agreement of very different distributions can be 0."""
    p = np.array([0.5, 0.3, 0.1, 0.1])
    q = np.array([0.1, 0.1, 0.3, 0.5])
    assert top_k_agreement(p, q, 1) == 0.0  # top-1: p=0, q=3


# --- calc_temp tests ---


def test_calc_temp_turn_zero():
    """At turn 0, temp equals eval_temp."""
    config = TrainConfig(eval_temp=0.5, final_temp=0.2, temp_decay_half_life=10)
    assert calc_temp(config, 0) == pytest.approx(0.5)


def test_calc_temp_large_turn():
    """At very large turn, temp approaches final_temp."""
    config = TrainConfig(eval_temp=0.5, final_temp=0.2, temp_decay_half_life=10)
    assert calc_temp(config, 1000) == pytest.approx(0.2, abs=0.01)


def test_calc_temp_half_life():
    """At turn = half_life, temp is halfway between eval and final."""
    config = TrainConfig(eval_temp=1.0, final_temp=0.0, temp_decay_half_life=10)
    assert calc_temp(config, 10) == pytest.approx(0.5, abs=0.01)


def test_calc_temp_selfplay_turn_zero():
    """At turn 0, selfplay temp equals self_play_temp."""
    config = TrainConfig(self_play_temp=1.0, final_temp=0.2, temp_decay_half_life=10)
    assert calc_temp_selfplay(config, 0) == pytest.approx(1.0)


def test_calc_temp_selfplay_large_turn():
    """At very large turn, selfplay temp approaches final_temp."""
    config = TrainConfig(self_play_temp=1.0, final_temp=0.2, temp_decay_half_life=10)
    assert calc_temp_selfplay(config, 1000) == pytest.approx(0.2, abs=0.01)


def test_calc_temp_selfplay_differs_from_eval():
    """Selfplay temp uses self_play_temp, not eval_temp."""
    config = TrainConfig(self_play_temp=1.0, eval_temp=0.5, final_temp=0.2, temp_decay_half_life=10)
    assert calc_temp_selfplay(config, 0) > calc_temp(config, 0)


# --- Import / smoke tests ---


def test_mcts_analysis_imports():
    """Module imports without error."""
    import mcts_analysis  # noqa: F401


def test_game_registry_used():
    """All GAME_REGISTRY games are selectable (no hardcoded Star Gambit only)."""
    available = get_available_games()
    assert "connect4" in available
    assert "star_gambit_skirmish" in available
    # Should have all registry games
    for game in GAME_REGISTRY:
        assert game in available


# --- Entry helper tests ---


def test_entry_label_base():
    """Base mode entries show just the VC number."""
    assert entry_label((200, "base")) == "200"
    assert entry_label((1, "base")) == "1"


def test_entry_label_selfplay():
    """Selfplay mode entries show VC with 'sp' suffix."""
    assert entry_label((200, "selfplay")) == "200sp"
    assert entry_label((1, "selfplay")) == "1sp"


def test_entry_sort_key_base_before_selfplay():
    """At the same VC, base sorts before selfplay."""
    assert entry_sort_key((200, "base")) < entry_sort_key((200, "selfplay"))


def test_entry_sort_key_lower_vc_first():
    """Lower VC sorts before higher VC regardless of mode."""
    assert entry_sort_key((100, "selfplay")) < entry_sort_key((200, "base"))


def test_entry_sort_key_full_ordering():
    """Full sort produces expected order."""
    entries = [(200, "selfplay"), (100, "base"), (200, "base"), (50, "selfplay")]
    sorted_entries = sorted(entries, key=entry_sort_key)
    assert sorted_entries == [(50, "selfplay"), (100, "base"), (200, "base"), (200, "selfplay")]


def test_entry_sort_key_duplicate_entries():
    """Identical entries sort equally."""
    assert entry_sort_key((400, "base")) == entry_sort_key((400, "base"))


# --- Anchor reference tests ---


def test_anchor_selfplay_is_reference(monkeypatch):
    """When anchor is selfplay, metrics use the anchor entry as reference, not base_ref.

    Regression test: previously base_ref (base-mode tree at anchor VC) was always
    the comparison reference, so (vc, "base") always got JSD=0 instead of the anchor.
    """
    import mcts_analysis
    monkeypatch.setattr(mcts_analysis, "ANALYSIS_GAMES", 4)

    config = TrainConfig(game="connect4")
    Game = config.Game

    entries = [(10, "base"), (10, "selfplay")]
    anchor = (10, "selfplay")

    metrics, snapshots = mcts_analysis.run_analysis(
        config, Game, network_path=None, entries=entries, anchor=anchor,
        use_playout=False, cache_size=0, tree_reuse=False,
    )

    jsd_means = metrics["jsd_means"]
    regret_means = metrics["regret_means"]

    # The anchor entry must have JSD=0 — it IS the reference (JSD(p,p)=0 exactly)
    assert anchor in jsd_means
    assert jsd_means[anchor] == pytest.approx(0.0, abs=1e-10)

    # The anchor entry must have regret=0 — V(pi_anchor) - V(pi_anchor) = 0
    assert anchor in regret_means
    assert regret_means[anchor] == pytest.approx(0.0, abs=1e-10)

    # The base entry at the same VC must have JSD > 0
    # (base policy differs from selfplay due to Dirichlet noise)
    base_entry = (10, "base")
    assert base_entry in jsd_means
    assert jsd_means[base_entry] > 0.0


def test_anchor_base_is_reference(monkeypatch):
    """When anchor is base, the anchor base entry gets JSD=0 and regret=0."""
    import mcts_analysis
    monkeypatch.setattr(mcts_analysis, "ANALYSIS_GAMES", 4)

    config = TrainConfig(game="connect4")
    Game = config.Game

    entries = [(10, "base"), (10, "selfplay")]
    anchor = (10, "base")

    metrics, snapshots = mcts_analysis.run_analysis(
        config, Game, network_path=None, entries=entries, anchor=anchor,
        use_playout=False, cache_size=0, tree_reuse=False,
    )

    jsd_means = metrics["jsd_means"]
    regret_means = metrics["regret_means"]

    # Base anchor must have JSD=0 and regret=0
    assert jsd_means[anchor] == pytest.approx(0.0, abs=1e-10)
    assert regret_means[anchor] == pytest.approx(0.0, abs=1e-10)

    # Selfplay entry should diverge from the base anchor
    sp_entry = (10, "selfplay")
    assert sp_entry in jsd_means
    assert jsd_means[sp_entry] > 0.0


# --- kl_divergence tests ---


def test_kl_divergence_identical():
    """KL divergence of identical distributions is ~0 (epsilon smoothing causes tiny offset)."""
    p = np.array([0.5, 0.3, 0.2])
    assert kl_divergence(p, p) == pytest.approx(0.0, abs=1e-8)


def test_kl_divergence_nonnegative():
    """KL divergence is always >= 0."""
    p = np.array([0.7, 0.2, 0.1])
    q = np.array([0.3, 0.4, 0.3])
    assert kl_divergence(p, q) >= 0.0


def test_kl_divergence_asymmetric():
    """KL(p||q) != KL(q||p) in general."""
    p = np.array([0.8, 0.15, 0.05])
    q = np.array([0.2, 0.3, 0.5])
    assert kl_divergence(p, q) != pytest.approx(kl_divergence(q, p), abs=1e-4)


def test_kl_divergence_zeros_in_p():
    """Zeros in p are skipped (0 * log(0/q) = 0 by convention)."""
    p = np.array([0.0, 1.0])
    q = np.array([0.5, 0.5])
    result = kl_divergence(p, q)
    # KL = 1.0 * log(1.0/0.5) = log(2)
    assert result == pytest.approx(math.log(2), abs=1e-6)


# --- policy_entropy tests ---


def test_policy_entropy_uniform():
    """Entropy of uniform distribution = log(n)."""
    n = 5
    p = np.full(n, 1.0 / n)
    assert policy_entropy(p) == pytest.approx(math.log(n), abs=1e-6)


def test_policy_entropy_deterministic():
    """Entropy of deterministic distribution is 0."""
    p = np.array([0.0, 0.0, 1.0, 0.0])
    assert policy_entropy(p) == pytest.approx(0.0, abs=1e-10)


def test_policy_entropy_nonnegative():
    """Entropy is always >= 0."""
    p = np.array([0.6, 0.3, 0.1])
    assert policy_entropy(p) >= 0.0


# --- PIO integration tests ---


def test_pio_metrics_present(monkeypatch):
    """All PIO metric keys exist in returned metrics dict with correct types."""
    import mcts_analysis
    monkeypatch.setattr(mcts_analysis, "ANALYSIS_GAMES", 4)

    config = TrainConfig(game="connect4")
    Game = config.Game

    entries = [(1, "base"), (10, "base"), (25, "base")]
    anchor = (25, "base")

    metrics, snapshots = mcts_analysis.run_analysis(
        config, Game, network_path=None, entries=entries, anchor=anchor,
        use_playout=False, cache_size=0, tree_reuse=False,
    )

    pio_metric_names = ["pio_kl", "pio_top1_flip", "pio_entropy_raw", "pio_entropy_mcts",
                        "pio_entropy_reduction", "pio_value_correction",
                        "pio_value_sign_flip", "pio_correction_quality"]

    for name in pio_metric_names:
        assert f"{name}_means" in metrics, f"missing {name}_means"
        assert f"{name}_all" in metrics, f"missing {name}_all"
        means = metrics[f"{name}_means"]
        all_vals = metrics[f"{name}_all"]
        assert isinstance(means, dict)
        assert isinstance(all_vals, dict)
        # vc=1 should NOT be in PIO metrics (it IS the baseline)
        assert (1, "base") not in means
        # vc=10 and vc=25 should be present
        for entry in [(10, "base"), (25, "base")]:
            assert entry in means, f"{entry} missing from {name}_means"
            assert isinstance(means[entry], float)
            assert entry in all_vals
            assert isinstance(all_vals[entry], np.ndarray)

    # KL should be >= 0
    for entry in [(10, "base"), (25, "base")]:
        assert metrics["pio_kl_means"][entry] >= 0.0

    # Top-1 flip rate should be in [0, 1]
    for entry in [(10, "base"), (25, "base")]:
        assert 0.0 <= metrics["pio_top1_flip_means"][entry] <= 1.0

    # Marginal KL should exist
    assert "pio_marginal_kl_means" in metrics
    assert "pio_marginal_kl_all" in metrics


def test_pio_vc1_injected_when_absent(monkeypatch):
    """When entries don't include vc=1, PIO metrics still work, and entries is NOT modified."""
    import mcts_analysis
    monkeypatch.setattr(mcts_analysis, "ANALYSIS_GAMES", 4)

    config = TrainConfig(game="connect4")
    Game = config.Game

    entries = [(10, "base"), (25, "base")]
    anchor = (25, "base")

    metrics, snapshots = mcts_analysis.run_analysis(
        config, Game, network_path=None, entries=entries, anchor=anchor,
        use_playout=False, cache_size=0, tree_reuse=False,
    )

    # entries in metrics should NOT include vc=1 (not modified)
    assert (1, "base") not in metrics["entries"]
    assert metrics["entries"] == entries

    # PIO metrics should still be computed for vc=10, vc=25
    assert (10, "base") in metrics["pio_kl_means"]
    assert (25, "base") in metrics["pio_kl_means"]


def test_pio_vc1_not_duplicated_when_present(monkeypatch):
    """When entries include vc=1, PIO data point count matches total positions (no double-counting)."""
    import mcts_analysis
    monkeypatch.setattr(mcts_analysis, "ANALYSIS_GAMES", 4)

    config = TrainConfig(game="connect4")
    Game = config.Game

    entries = [(1, "base"), (10, "base"), (25, "base")]
    anchor = (25, "base")

    metrics, snapshots = mcts_analysis.run_analysis(
        config, Game, network_path=None, entries=entries, anchor=anchor,
        use_playout=False, cache_size=0, tree_reuse=False,
    )

    total_positions = metrics["total_positions"]
    # PIO data points for vc=10 should equal total positions (one per position)
    n_pio = len(metrics["pio_kl_all"][(10, "base")])
    assert n_pio == total_positions
    # Same for vc=25
    n_pio_25 = len(metrics["pio_kl_all"][(25, "base")])
    assert n_pio_25 == total_positions


# --- compute_scaling_report tests ---


def _make_entries(vcs):
    """Helper: create base-mode entries from a list of visit counts."""
    return [(vc, "base") for vc in vcs]


def test_elo_per_doubling_uniform():
    """Power-of-2 VCs with uniform Elo spacing -> exact 100.0/doubling, R2=1.0."""
    entries = _make_entries([1, 2, 4, 8])
    anchor = (8, "base")
    elo = np.array([0.0, 100.0, 200.0, 300.0])

    result = compute_scaling_report(entries, anchor, elo=elo, metrics=None)

    assert "elo_per_doubling" in result
    epd = result["elo_per_doubling"]
    assert len(epd) == 3
    for vc1, vc2, val in epd:
        assert val == pytest.approx(100.0, abs=1e-6)

    assert result["elo_slope"] == pytest.approx(100.0, abs=1e-6)
    assert result["elo_r2"] == pytest.approx(1.0, abs=1e-6)


def test_elo_per_doubling_nonuniform():
    """Non-power-of-2 VCs use correct log2 denominator."""
    entries = _make_entries([10, 30, 90])
    anchor = (90, "base")
    elo = np.array([0.0, 100.0, 200.0])

    result = compute_scaling_report(entries, anchor, elo=elo, metrics=None)

    epd = result["elo_per_doubling"]
    assert len(epd) == 2
    # 10->30: log2(3) ~= 1.585, so elo/doubling = 100/1.585 ~= 63.09
    assert epd[0][2] == pytest.approx(100.0 / math.log2(3), abs=0.1)
    # 30->90: same ratio
    assert epd[1][2] == pytest.approx(100.0 / math.log2(3), abs=0.1)


def test_elo_per_doubling_none_when_no_elo():
    """elo=None -> elo_per_doubling key absent from result."""
    entries = _make_entries([1, 10, 100])
    anchor = (100, "base")
    metrics = {"pio_top1_flip_means": {(10, "base"): 0.3, (100, "base"): 0.5}}

    result = compute_scaling_report(entries, anchor, elo=None, metrics=metrics)

    assert "elo_per_doubling" not in result
    assert "elo_slope" not in result


def test_policy_improvement_basic():
    """Policy improvement = pio_top1_flip (ascending with VC)."""
    entries = _make_entries([1, 10, 100])
    anchor = (100, "base")
    metrics = {
        "pio_top1_flip_means": {(10, "base"): 0.3, (100, "base"): 0.5},
    }

    result = compute_scaling_report(entries, anchor, elo=None, metrics=metrics)

    assert "policy_improvement" in result
    assert result["policy_improvement"][(10, "base")] == pytest.approx(0.3)
    assert result["policy_improvement"][(100, "base")] == pytest.approx(0.5)
    # vc=1 should NOT be present (pio_top1_flip only exists for vc > 1)
    assert (1, "base") not in result["policy_improvement"]


def test_capacity_score_vc1_excluded():
    """vc=1 never appears in capacity_score."""
    entries = _make_entries([1, 10, 100])
    anchor = (100, "base")
    metrics = {
        "pio_top1_flip_means": {(10, "base"): 0.3, (100, "base"): 0.5},
        "pio_correction_quality_means": {(10, "base"): 0.8, (100, "base"): 0.9},
    }

    result = compute_scaling_report(entries, anchor, elo=None, metrics=metrics)

    assert "capacity_score" in result
    assert (1, "base") not in result["capacity_score"]
    assert (10, "base") in result["capacity_score"]


def test_capacity_score_computation():
    """Capacity score = pio_top1_flip * correction_quality."""
    entries = _make_entries([1, 50])
    anchor = (50, "base")
    metrics = {
        "pio_top1_flip_means": {(50, "base"): 0.4},
        "pio_correction_quality_means": {(50, "base"): 0.75},
    }

    result = compute_scaling_report(entries, anchor, elo=None, metrics=metrics)

    # capacity = 0.4 * 0.75 = 0.3
    assert result["capacity_score"][(50, "base")] == pytest.approx(0.3)

    # With different values
    metrics["pio_top1_flip_means"][(50, "base")] = 0.0
    result = compute_scaling_report(entries, anchor, elo=None, metrics=metrics)
    # 0.0 * 0.75 = 0.0
    assert result["capacity_score"][(50, "base")] == pytest.approx(0.0)


def test_mcts_utilization_boundaries():
    """Utilization = 0.0 at vc=1 and 1.0 at anchor."""
    entries = _make_entries([1, 10, 100])
    anchor = (100, "base")
    metrics = {
        "reward_means": {(1, "base"): 0.2, (10, "base"): 0.5, (100, "base"): 0.8},
    }

    result = compute_scaling_report(entries, anchor, elo=None, metrics=metrics)

    assert "mcts_utilization" in result
    mu = result["mcts_utilization"]
    assert mu[(1, "base")] == pytest.approx(0.0)
    assert mu[(100, "base")] == pytest.approx(1.0)
    # Intermediate: (0.5 - 0.2) / (0.8 - 0.2) = 0.5
    assert mu[(10, "base")] == pytest.approx(0.5)


def test_mcts_utilization_no_division_by_zero():
    """Equal rewards (anchor ≈ vc=1) -> mcts_utilization absent."""
    entries = _make_entries([1, 10, 100])
    anchor = (100, "base")
    metrics = {
        "reward_means": {(1, "base"): 0.5, (10, "base"): 0.5, (100, "base"): 0.5},
    }

    result = compute_scaling_report(entries, anchor, elo=None, metrics=metrics)

    assert "mcts_utilization" not in result


def test_scaling_report_empty_inputs():
    """elo=None, metrics=None -> empty dict."""
    entries = _make_entries([1, 10, 100])
    anchor = (100, "base")

    result = compute_scaling_report(entries, anchor, elo=None, metrics=None)

    assert result == {}


def test_scaling_report_ece_passthrough():
    """ECE and value accuracy gain are passed through from metrics."""
    entries = _make_entries([1, 10, 100])
    anchor = (100, "base")
    metrics = {
        "pio_top1_flip_means": {(10, "base"): 0.3, (100, "base"): 0.5},
        "value_ece": {(1, "base"): 0.15, (10, "base"): 0.10, (100, "base"): 0.05},
        "pio_value_accuracy_gain_means": {(10, "base"): 0.6, (100, "base"): 0.7},
    }

    result = compute_scaling_report(entries, anchor, elo=None, metrics=metrics)

    assert "value_ece" in result
    assert result["value_ece"][(10, "base")] == pytest.approx(0.10)
    assert result["value_ece"][(100, "base")] == pytest.approx(0.05)
    assert "value_accuracy_gain" in result
    assert result["value_accuracy_gain"][(10, "base")] == pytest.approx(0.6)
    assert result["value_accuracy_gain"][(100, "base")] == pytest.approx(0.7)


# --- Position snapshot tests ---


def test_snapshots_returned(monkeypatch):
    """run_analysis returns position snapshots alongside metrics."""
    import mcts_analysis
    monkeypatch.setattr(mcts_analysis, "ANALYSIS_GAMES", 4)

    config = TrainConfig(game="connect4")
    Game = config.Game

    entries = [(1, "base"), (10, "base")]
    anchor = (10, "base")

    metrics, snapshots = mcts_analysis.run_analysis(
        config, Game, network_path=None, entries=entries, anchor=anchor,
        use_playout=False, cache_size=0, tree_reuse=False,
    )

    # Snapshots are one per turn (not per sub-move), so <= total_positions
    assert 0 < len(snapshots) <= metrics["total_positions"]

    # Each snapshot has the required keys
    snap = snapshots[0]
    assert "gs" in snap
    assert "player" in snap
    assert "values" in snap
    assert isinstance(snap["player"], int)
    assert isinstance(snap["values"], dict)


# --- Value ECE tests ---


def test_value_ece_in_metrics(monkeypatch):
    """ECE dict present with values in [0, 1] for all entries."""
    import mcts_analysis
    monkeypatch.setattr(mcts_analysis, "ANALYSIS_GAMES", 4)

    config = TrainConfig(game="connect4")
    Game = config.Game

    entries = [(1, "base"), (10, "base"), (25, "base")]
    anchor = (25, "base")

    metrics, _ = mcts_analysis.run_analysis(
        config, Game, network_path=None, entries=entries, anchor=anchor,
        use_playout=False, cache_size=0, tree_reuse=False,
    )

    assert "value_ece" in metrics
    value_ece = metrics["value_ece"]
    assert isinstance(value_ece, dict)
    # At least some entries should have ECE computed (need >= 10 positions)
    for entry, ece in value_ece.items():
        assert 0.0 <= ece <= 1.0, f"ECE for {entry} = {ece} out of [0, 1]"

    # Calibration data should also be present
    assert "value_calibration_data" in metrics


# --- Value Accuracy Gain (VIR) tests ---


def test_value_accuracy_gain_in_metrics(monkeypatch):
    """Value accuracy gain present for vc > 1 entries."""
    import mcts_analysis
    monkeypatch.setattr(mcts_analysis, "ANALYSIS_GAMES", 4)

    config = TrainConfig(game="connect4")
    Game = config.Game

    entries = [(1, "base"), (10, "base"), (25, "base")]
    anchor = (25, "base")

    metrics, _ = mcts_analysis.run_analysis(
        config, Game, network_path=None, entries=entries, anchor=anchor,
        use_playout=False, cache_size=0, tree_reuse=False,
    )

    assert "pio_value_accuracy_gain_means" in metrics
    vag_means = metrics["pio_value_accuracy_gain_means"]
    assert isinstance(vag_means, dict)
    # vc=1 should NOT be in VIR (it IS the baseline)
    assert (1, "base") not in vag_means
    # vc=10 and vc=25 should be present
    for entry in [(10, "base"), (25, "base")]:
        assert entry in vag_means, f"{entry} missing from pio_value_accuracy_gain_means"
        assert 0.0 <= vag_means[entry] <= 1.0
