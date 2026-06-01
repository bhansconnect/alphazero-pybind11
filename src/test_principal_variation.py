"""Unit tests for MCTS::principal_variation (C++ method exposed to Python).

Verifies the walk follows visit-argmax for PUCT, gumbel_final_action at the
root for Gumbel, stops at unvisited children, and respects the depth cap.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import alphazero


def _make_mcts(Game, gumbel_enabled=False):
    return alphazero.MCTS(
        1.25,                       # cpuct
        Game.NUM_PLAYERS(),
        Game.NUM_MOVES(),
        0.0,                        # epsilon (Dirichlet)
        1.0,                        # root_policy_temp
        0.25,                       # fpu_reduction
        Game().relative_values(),
        False,                      # root_fpu_zero
        False,                      # shaped_dirichlet
        bool(gumbel_enabled),
        16,                         # gumbel_m
        50.0,                       # gumbel_c_visit
        1.0,                        # gumbel_c_scale
        False,                      # gumbel_full
    )


def _uniform_sims(mcts, gs, n, seed=None):
    """Run n sims with uniform policy / value (no NN required)."""
    np_players = gs.num_players()
    uniform_v = np.full(np_players + 1, 1.0 / (np_players + 1))
    for i in range(n):
        leaf = mcts.find_leaf(gs)
        if leaf.scores() is not None:
            v = np.array(leaf.scores())
            pi = np.zeros(gs.num_moves())
        else:
            v = uniform_v
            pi = np.ones(gs.num_moves()) / gs.num_moves()
        mcts.process_result(gs, v, pi, i == 0)


# ---------------------------------------------------------------------------
# Empty / boundary
# ---------------------------------------------------------------------------


def test_pv_depth_zero_returns_empty():
    Game = alphazero.Connect4GS
    mcts = _make_mcts(Game)
    assert list(mcts.principal_variation(0)) == []


def test_pv_no_sims_returns_empty():
    Game = alphazero.Connect4GS
    mcts = _make_mcts(Game)
    # No sims run -> root has no children visited
    assert list(mcts.principal_variation(5)) == []


# ---------------------------------------------------------------------------
# PUCT mode -- argmax(visits) at every level
# ---------------------------------------------------------------------------


def test_pv_root_is_visit_argmax_under_puct():
    Game = alphazero.Connect4GS
    gs = Game()
    mcts = _make_mcts(Game)
    _uniform_sims(mcts, gs, 100)
    pv = list(mcts.principal_variation(5))
    assert len(pv) >= 1
    counts = np.array(mcts.counts())
    assert pv[0] == int(np.argmax(counts))


def test_pv_respects_depth_cap():
    Game = alphazero.Connect4GS
    gs = Game()
    mcts = _make_mcts(Game)
    _uniform_sims(mcts, gs, 200)
    # At 200 sims tree is deep enough to honor depth=3
    pv_short = list(mcts.principal_variation(3))
    pv_long = list(mcts.principal_variation(10))
    assert len(pv_short) <= 3
    # Longer-or-equal request returns at least as much
    assert len(pv_long) >= len(pv_short)


def test_pv_stops_early_at_unvisited_nodes():
    """At very low sim count the tree barely descends past root; PV must
    not return more moves than the tree actually contains."""
    Game = alphazero.Connect4GS
    gs = Game()
    mcts = _make_mcts(Game)
    _uniform_sims(mcts, gs, 3)  # tiny budget
    pv = list(mcts.principal_variation(10))
    # Tree depth is at most 3 here (and likely less) due to back-propagation
    assert len(pv) <= 3


# ---------------------------------------------------------------------------
# Gumbel mode -- gumbel_final_action at root, argmax(visits) deeper
# ---------------------------------------------------------------------------


def test_pv_root_uses_gumbel_final_action_under_gumbel():
    Game = alphazero.Connect4GS
    gs = Game()
    mcts = _make_mcts(Game, gumbel_enabled=True)
    mcts.set_gumbel_num_sims(64)
    _uniform_sims(mcts, gs, 64)
    pv = list(mcts.principal_variation(3))
    assert len(pv) >= 1
    # Root must follow Sequential Halving's pick, not raw visit-argmax,
    # because the two can disagree at small budgets.
    assert pv[0] == int(mcts.gumbel_final_action())


def test_pv_returns_uint32_move_ids():
    Game = alphazero.Connect4GS
    gs = Game()
    mcts = _make_mcts(Game)
    _uniform_sims(mcts, gs, 30)
    pv = mcts.principal_variation(4)
    # Method returns a Vector<uint32_t>; each entry should be a valid move id
    for m in pv:
        assert 0 <= int(m) < gs.num_moves()


def test_pv_first_move_valid_in_starting_position():
    Game = alphazero.Connect4GS
    gs = Game()
    mcts = _make_mcts(Game)
    _uniform_sims(mcts, gs, 50)
    pv = list(mcts.principal_variation(1))
    assert len(pv) == 1
    valids = np.array(gs.valid_moves())
    assert valids[pv[0]] == 1
