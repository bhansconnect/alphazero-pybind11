"""Tests for mcts_analysis.py -- statistical functions, calc_temp, imports."""

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
    calc_temp,
    calc_temp_selfplay,
    get_available_games,
    entry_label,
    entry_sort_key,
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

    metrics = mcts_analysis.run_analysis(
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

    metrics = mcts_analysis.run_analysis(
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
