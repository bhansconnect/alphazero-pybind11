"""Tests for play.py -- batch sizing, cache-aware batching, calibration, MCTS search."""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import alphazero
from play import (
    _compute_batch_size,
    _run_one_batch,
    run_mcts_search,
    create_mcts,
    calibrate_timed_batch,
    PlayContext,
    PlayerConfig,
    _update_caches,
    _get_cache,
    _effective_greedy,
    _check_resign,
    _lcb_action,
    _greedy_ai_action,
    _apply_player_gumbel,
    handle_config_command,
    get_ai_probs,
    DEFAULT_GUMBEL_C_SCALE,
    DEFAULT_GUMBEL_M,
    DEFAULT_GUMBEL_C_VISIT,
    DEFAULT_TEMPERATURE,
    DEFAULT_NODE_LIMIT,
)
from cache_utils import create_cache


# --- _compute_batch_size tests ---


def test_compute_batch_size_sqrt_scaling():
    """Batch size equals int(sqrt(budget)) for various budgets."""
    for budget in [4, 16, 25, 100, 400, 10000]:
        assert _compute_batch_size(budget) == int(math.sqrt(budget))


def test_compute_batch_size_minimum_one():
    """Budget=0 and budget=1 both return 1."""
    assert _compute_batch_size(0) == 1
    assert _compute_batch_size(1) == 1


def test_compute_batch_size_no_hard_cap():
    """Large budgets produce large batch sizes (no cap at 64)."""
    assert _compute_batch_size(10000) == 100
    assert _compute_batch_size(40000) == 200
    assert _compute_batch_size(1000000) == 1000


# --- _run_one_batch tests ---


def test_run_one_batch_returns_actual_sims():
    """_run_one_batch returns actual simulation count, not nominal batch_size."""
    gs = alphazero.Connect4GS()
    mcts = create_mcts(type(gs))
    batch_size = 4

    actual = _run_one_batch(gs, None, mcts, batch_size, "random", None, 0)

    # With random eval and no cache, all leaves are processed inline
    # actual should equal the number of attempts (all are immediate)
    assert actual >= batch_size
    assert isinstance(actual, int)


def test_run_one_batch_cache_hits_extra_sims():
    """With a warm cache, actual_sims > len(pending_gpu) due to cache hits."""
    gs = alphazero.Connect4GS()
    cache = create_cache(type(gs), 1000)
    mcts = create_mcts(type(gs))
    batch_size = 8

    # With random eval (no agent), all are processed immediately
    actual = _run_one_batch(gs, None, mcts, batch_size, "random", cache, 0)

    # All sims are processed inline (random eval), so actual >= batch_size
    assert actual >= batch_size


# --- run_mcts_search tests ---


def test_mcts_search_node_limit():
    """Basic correctness with auto batch and node limit."""
    gs = alphazero.Connect4GS()
    mcts = create_mcts(type(gs))
    node_limit = 50

    counts, sims, wld = run_mcts_search(
        gs, None, mcts, node_limit=node_limit, eval_type="random",
        max_batch_size=0,
    )

    assert counts.sum() > 0
    assert sims >= node_limit  # may overshoot slightly due to batching
    assert len(wld) == gs.num_players() + 1


def test_mcts_search_sequential_unchanged():
    """batch_size=1 path still works correctly."""
    gs = alphazero.Connect4GS()
    mcts = create_mcts(type(gs))
    node_limit = 20

    counts, sims, wld = run_mcts_search(
        gs, None, mcts, node_limit=node_limit, eval_type="random",
        max_batch_size=1,
    )

    assert sims == node_limit  # sequential is exact
    assert counts.sum() > 0


def test_mcts_search_timed_auto():
    """Timed mode completes and produces results."""
    gs = alphazero.Connect4GS()
    mcts = create_mcts(type(gs))

    counts, sims, wld = run_mcts_search(
        gs, None, mcts, time_limit=0.1, eval_type="random",
        max_batch_size=0,
    )

    assert counts.sum() > 0
    assert sims > 0


# --- calibrate_timed_batch tests ---


def test_calibrate_timed_batch():
    """Calibration returns a power of 2, >= 2."""
    gs = alphazero.Connect4GS()

    result = calibrate_timed_batch(gs, None, time_limit=1.0,
                                    eval_type="random", cache=None)

    assert result >= 2
    # Must be a power of 2
    assert (result & (result - 1)) == 0, f"{result} is not a power of 2"


# --- Per-player cache split tests ---


def test_update_caches_shared_when_no_time():
    """No think_time -> shared cache, no per-player caches."""
    gs = alphazero.Connect4GS()
    ctx = PlayContext(gs, type(gs), cache_size=1000)
    ctx.players[0].is_ai = True
    ctx.players[1].is_ai = True
    _update_caches(ctx)

    assert ctx.cache is not None
    assert ctx.player_caches == [None, None]
    assert _get_cache(ctx, 0) is ctx.cache
    assert _get_cache(ctx, 1) is ctx.cache


def test_update_caches_split_when_timed():
    """With think_time -> per-player caches, no shared cache."""
    gs = alphazero.Connect4GS()
    ctx = PlayContext(gs, type(gs), cache_size=1000)
    ctx.players[0].is_ai = True
    ctx.players[0].think_time = 2.0
    ctx.players[1].is_ai = True
    _update_caches(ctx)

    assert ctx.cache is None
    assert ctx.player_caches[0] is not None
    assert ctx.player_caches[1] is not None
    assert ctx.player_caches[0] is not ctx.player_caches[1]
    assert _get_cache(ctx, 0) is ctx.player_caches[0]
    assert _get_cache(ctx, 1) is ctx.player_caches[1]
    # Each gets half the total cache size
    assert ctx.player_caches[0].max_size() == 500
    assert ctx.player_caches[1].max_size() == 500


def test_update_caches_disabled():
    """cache_size=0 -> all caches are None."""
    gs = alphazero.Connect4GS()
    ctx = PlayContext(gs, type(gs), cache_size=0)
    ctx.players[0].think_time = 2.0
    _update_caches(ctx)

    assert ctx.cache is None
    assert ctx.player_caches == [None, None]
    assert _get_cache(ctx, 0) is None
    assert _get_cache(ctx, 1) is None


# ---------------------------------------------------------------------------
# PlayerConfig defaults (pins the native-algo flip and other new fields)
# ---------------------------------------------------------------------------


def test_player_config_defaults():
    """The diff flipped algo_override from 'puct' to None and added several
    new fields. Pin them so an accidental revert is caught."""
    p = PlayerConfig()
    # Native-algo flip:
    assert p.algo_override is None
    # Variance / noise:
    assert p.epsilon == 0.0
    assert p.prune_frac == 0.02
    # LCB:
    assert p.lcb_enabled is False
    assert p.lcb_z == 2.0
    # Gumbel G2 opening-moves opt-in:
    assert p.gumbel_opening_moves == 0
    assert p.gumbel_c_scale == DEFAULT_GUMBEL_C_SCALE
    # Resign:
    assert p.resign_enabled is False
    assert p.resign_threshold == -0.95
    assert p.resign_consecutive == 3
    assert p._resign_streak == 0
    # PV scratch:
    assert p._last_pv == []


def test_player_config_str_includes_resign_only_when_enabled():
    p = PlayerConfig()
    p.is_ai = True
    p.network_path = "/x/y/foo.pt"
    p.eval_type = "network"
    p.gumbel_enabled = True
    s = str(p)
    assert "resign=" not in s
    p.resign_enabled = True
    p.resign_threshold = -0.9
    p.resign_consecutive = 3
    assert "resign=-0.9/3" in str(p)


def test_player_config_str_shows_eps_only_when_nonzero():
    p = PlayerConfig()
    p.is_ai = True
    p.network_path = "/x/y/foo.pt"
    p.eval_type = "network"
    assert "eps=" not in str(p)
    p.epsilon = 0.25
    assert "eps=0.25" in str(p)


def test_player_config_str_shows_cscale_only_when_off_default():
    p = PlayerConfig()
    p.is_ai = True
    p.network_path = "/x/y/foo.pt"
    p.eval_type = "network"
    p.gumbel_enabled = True
    assert "cscale=" not in str(p)
    p.gumbel_c_scale = 2.0
    assert "cscale=2.0" in str(p)


# ---------------------------------------------------------------------------
# _effective_greedy (G2 opening-moves override)
# ---------------------------------------------------------------------------


def test_effective_greedy_passes_through_without_gumbel():
    p = PlayerConfig()
    p.greedy = True
    assert _effective_greedy(p, move_number=0) is True
    p.greedy = False
    assert _effective_greedy(p, move_number=7) is False


def test_effective_greedy_passes_through_when_gumbel_opening_zero():
    p = PlayerConfig()
    p.gumbel_enabled = True
    p.gumbel_opening_moves = 0
    p.greedy = True
    assert _effective_greedy(p, move_number=0) is True


def test_effective_greedy_g2_forces_non_greedy_in_opening():
    p = PlayerConfig()
    p.gumbel_enabled = True
    p.gumbel_opening_moves = 6
    p.greedy = True
    # Within opening window -> override to False
    assert _effective_greedy(p, move_number=0) is False
    assert _effective_greedy(p, move_number=5) is False
    # At the boundary, opening_moves is exclusive (< not <=)
    assert _effective_greedy(p, move_number=6) is True
    assert _effective_greedy(p, move_number=20) is True


def test_effective_greedy_g2_only_active_when_gumbel():
    p = PlayerConfig()
    p.gumbel_enabled = False  # PUCT
    p.gumbel_opening_moves = 6
    p.greedy = True
    # G2 only fires for Gumbel; PUCT ignores opening_moves.
    assert _effective_greedy(p, move_number=0) is True


# ---------------------------------------------------------------------------
# _check_resign (per-player streak state machine)
# ---------------------------------------------------------------------------


def test_check_resign_disabled_keeps_streak_zero():
    p = PlayerConfig()
    p.resign_enabled = False
    assert _check_resign(p, [0.0, 1.0, 0.0]) is False
    assert p._resign_streak == 0


def test_check_resign_none_wld_resets_streak():
    p = PlayerConfig()
    p.resign_enabled = True
    p._resign_streak = 2
    assert _check_resign(p, None) is False
    assert p._resign_streak == 0


def test_check_resign_increments_streak_then_triggers():
    p = PlayerConfig()
    p.resign_enabled = True
    p.resign_threshold = -0.9
    p.resign_consecutive = 3
    losing = [0.0, 1.0, 0.0]  # V = W - L = -1
    assert _check_resign(p, losing) is False
    assert p._resign_streak == 1
    assert _check_resign(p, losing) is False
    assert p._resign_streak == 2
    assert _check_resign(p, losing) is True
    assert p._resign_streak == 3


def test_check_resign_resets_when_v_recovers():
    p = PlayerConfig()
    p.resign_enabled = True
    p.resign_threshold = -0.9
    p.resign_consecutive = 3
    _check_resign(p, [0.0, 1.0, 0.0])  # streak 1
    _check_resign(p, [0.5, 0.5, 0.0])  # V = 0 > threshold -> reset
    assert p._resign_streak == 0
    # Need three more in a row to trigger
    assert _check_resign(p, [0.0, 1.0, 0.0]) is False  # 1
    assert _check_resign(p, [0.0, 1.0, 0.0]) is False  # 2
    assert _check_resign(p, [0.0, 1.0, 0.0]) is True   # 3


def test_check_resign_v_just_above_threshold_does_not_trigger():
    p = PlayerConfig()
    p.resign_enabled = True
    p.resign_threshold = -0.5
    p.resign_consecutive = 1
    # V = -0.4 > -0.5
    assert _check_resign(p, [0.3, 0.7, 0.0]) is False


def test_check_resign_v_at_threshold_triggers():
    p = PlayerConfig()
    p.resign_enabled = True
    p.resign_threshold = -0.5
    p.resign_consecutive = 1
    # V = -0.5 == threshold (<=)
    assert _check_resign(p, [0.25, 0.75, 0.0]) is True


def test_check_resign_malformed_wld_resets_streak():
    p = PlayerConfig()
    p.resign_enabled = True
    p._resign_streak = 5
    # Indexing past end raises IndexError; helper should swallow and reset.
    assert _check_resign(p, [0.0]) is False
    assert p._resign_streak == 0


# ---------------------------------------------------------------------------
# _lcb_action (greedy LCB selection)
# ---------------------------------------------------------------------------


def _mcts_with_random_sims(num_sims, game_class=alphazero.Connect4GS):
    """Run uniform-policy MCTS sims and return (pcfg, valids)."""
    gs = game_class()
    pcfg = PlayerConfig()
    pcfg.is_ai = True
    pcfg.mcts = create_mcts(game_class)
    np_rng = np.random.default_rng(seed=42)
    for i in range(num_sims):
        leaf = pcfg.mcts.find_leaf(gs)
        if leaf.scores() is not None:
            v = np.array(leaf.scores())
            pi = np.zeros(gs.num_moves())
        else:
            # Slightly non-uniform policy so Q values diverge
            pi = np_rng.dirichlet(np.ones(gs.num_moves()))
            v = np.full(gs.num_players() + 1, 1.0 / (gs.num_players() + 1))
        pcfg.mcts.process_result(gs, v, pi, i == 0)
    valids = np.array(gs.valid_moves())
    return pcfg, valids


def test_lcb_action_returns_none_when_no_mcts():
    p = PlayerConfig()
    p.mcts = None
    assert _lcb_action(p, np.array([1] * 7)) is None


def test_lcb_action_with_z_zero_equals_argmax_q():
    pcfg, valids = _mcts_with_random_sims(64)
    pcfg.lcb_z = 0.0
    a = _lcb_action(pcfg, valids)
    q = np.array(pcfg.mcts.root_q_values())
    n = np.array(pcfg.mcts.counts())
    expected = int(np.argmax(np.where((n > 0) & (valids != 0), q, -np.inf)))
    assert a == expected


def test_lcb_action_returns_visited_move_only():
    pcfg, valids = _mcts_with_random_sims(64)
    pcfg.lcb_z = 2.0
    a = _lcb_action(pcfg, valids)
    n = np.array(pcfg.mcts.counts())
    # The returned move must be both visited and valid.
    assert a is not None
    assert n[a] > 0
    assert valids[a] != 0


def test_lcb_action_returns_none_when_no_visited_valid_move():
    pcfg, valids = _mcts_with_random_sims(64)
    # Force valids to mask every visited move
    n = np.array(pcfg.mcts.counts())
    fake_valids = np.where(n > 0, 0, 1).astype(np.uint8)
    # Pathological: even if there's any visited move whose valid is now 0,
    # AND any valid move whose visits are 0, _lcb_action returns None when
    # the intersection is empty.
    if (n == 0).any():
        # Construct a clean degenerate case
        result = _lcb_action(pcfg, fake_valids)
        assert result is None


# ---------------------------------------------------------------------------
# _greedy_ai_action (Gumbel final-action / LCB / argmax priority)
# ---------------------------------------------------------------------------


def test_greedy_ai_action_falls_back_to_argmax_probs_for_non_ai():
    p = PlayerConfig()
    p.is_ai = False
    probs = np.array([0.1, 0.7, 0.2])
    valids = np.array([1, 1, 1])
    assert _greedy_ai_action(p, probs, valids, sims=0) == 1


def test_greedy_ai_action_uses_lcb_when_enabled():
    pcfg, valids = _mcts_with_random_sims(64)
    pcfg.lcb_enabled = True
    pcfg.lcb_z = 0.0  # LCB collapses to argmax(Q) on visited cells
    # build a counts-based probs that argmax-disagrees with LCB
    counts = np.array(pcfg.mcts.counts()).astype(float)
    probs = counts / counts.sum() if counts.sum() else np.ones_like(counts) / len(counts)
    action = _greedy_ai_action(pcfg, probs, valids, sims=64)
    q = np.array(pcfg.mcts.root_q_values())
    n = np.array(pcfg.mcts.counts())
    lcb_pick = int(np.argmax(np.where((n > 0) & (valids != 0), q, -np.inf)))
    # The LCB pick must drive the result, not raw argmax(probs).
    assert action == lcb_pick


def test_greedy_ai_action_uses_gumbel_final_action_when_enabled():
    """When Gumbel is enabled, _greedy_ai_action calls gumbel_final_action."""
    # Build a Gumbel-enabled MCTS with enough sims for a meaningful pick
    Game = alphazero.Connect4GS
    gs = Game()
    pcfg = PlayerConfig()
    pcfg.is_ai = True
    pcfg.gumbel_enabled = True
    pcfg.mcts = create_mcts(Game, gumbel_enabled=True)
    pcfg.mcts.set_gumbel_num_sims(48)
    for i in range(48):
        leaf = pcfg.mcts.find_leaf(gs)
        v = np.full(3, 1.0/3.0)
        pi = np.ones(gs.num_moves()) / gs.num_moves()
        pcfg.mcts.process_result(gs, v, pi, i == 0)
    probs = np.ones(gs.num_moves()) / gs.num_moves()
    valids = np.array(gs.valid_moves())
    action = _greedy_ai_action(pcfg, probs, valids, sims=48)
    assert action == int(pcfg.mcts.gumbel_final_action())


# ---------------------------------------------------------------------------
# _apply_player_gumbel (algo_override + auto-greedy on Gumbel entry)
# ---------------------------------------------------------------------------


def test_apply_player_gumbel_auto_sets_greedy_when_entering_gumbel():
    """Transitioning from non-Gumbel to Gumbel auto-flips greedy=True so the
    paper-faithful G1 (gumbel_final_action) path is used by default."""
    p = PlayerConfig()
    p.greedy = False
    p.gumbel_enabled = False
    p._yaml_gumbel = {
        "gumbel_enabled": True, "gumbel_m": 8, "gumbel_c_visit": 50.0,
        "gumbel_c_scale": 1.0, "gumbel_full": False,
    }
    p.algo_override = None  # auto -> follow yaml
    _apply_player_gumbel(p)
    assert p.gumbel_enabled is True
    assert p.greedy is True
    # MCTS reset so next search rebuilds with new params
    assert p.mcts is None


def test_apply_player_gumbel_preserves_greedy_when_already_gumbel():
    """Re-applying when already in Gumbel must not clobber a user override."""
    p = PlayerConfig()
    p.gumbel_enabled = True   # already Gumbel
    p.greedy = False           # user explicitly turned greedy off
    p._yaml_gumbel = {
        "gumbel_enabled": True, "gumbel_m": 8, "gumbel_c_visit": 50.0,
        "gumbel_c_scale": 1.0, "gumbel_full": False,
    }
    _apply_player_gumbel(p)
    assert p.greedy is False


def test_apply_player_gumbel_explicit_puct_override():
    p = PlayerConfig()
    p._yaml_gumbel = {
        "gumbel_enabled": True, "gumbel_m": 8, "gumbel_c_visit": 50.0,
        "gumbel_c_scale": 1.0, "gumbel_full": False,
    }
    p.algo_override = "puct"
    _apply_player_gumbel(p)
    assert p.gumbel_enabled is False


def test_apply_player_gumbel_inherits_yaml_params():
    """algo_override only flips the on/off bit; m/c_visit/c_scale/full come
    from the network's yaml so a manual ':gumbel' uses the same hyperparams
    the network was trained with."""
    p = PlayerConfig()
    p._yaml_gumbel = {
        "gumbel_enabled": False, "gumbel_m": 12,
        "gumbel_c_visit": 25.0, "gumbel_c_scale": 0.5, "gumbel_full": True,
    }
    p.algo_override = "gumbel"  # flip on
    _apply_player_gumbel(p)
    assert p.gumbel_enabled is True
    assert p.gumbel_m == 12
    assert p.gumbel_c_visit == 25.0
    assert p.gumbel_c_scale == 0.5
    assert p.gumbel_full is True


# ---------------------------------------------------------------------------
# handle_config_command (interactive command parsing + side effects)
# ---------------------------------------------------------------------------


def _ctx_with_one_ai():
    Game = alphazero.Connect4GS
    ctx = PlayContext(Game(), Game, cache_size=0)
    ctx.players[0].is_ai = True
    ctx.players[1].is_ai = True
    return ctx


def test_handle_lcb_command_enables_and_forces_greedy(capsys):
    ctx = _ctx_with_one_ai()
    ctx.players[0].greedy = False
    handle_config_command(["lcb", "0", "on"], ctx, "connect4", "data")
    assert ctx.players[0].lcb_enabled is True
    assert ctx.players[0].greedy is True
    out = capsys.readouterr().out.lower()
    assert "lcb" in out and "greedy" in out


def test_handle_lcb_command_numeric_value_sets_z_and_enables():
    ctx = _ctx_with_one_ai()
    handle_config_command(["lcb", "0", "1.5"], ctx, "connect4", "data")
    assert ctx.players[0].lcb_enabled is True
    assert ctx.players[0].lcb_z == 1.5


def test_handle_lcb_off_does_not_revert_greedy():
    ctx = _ctx_with_one_ai()
    handle_config_command(["lcb", "0", "on"], ctx, "connect4", "data")
    assert ctx.players[0].greedy is True
    handle_config_command(["lcb", "0", "off"], ctx, "connect4", "data")
    assert ctx.players[0].lcb_enabled is False
    # greedy is not auto-reverted (documented behavior)
    assert ctx.players[0].greedy is True


def test_handle_lcb_invalid_value_returns_without_change():
    ctx = _ctx_with_one_ai()
    before = ctx.players[0].lcb_enabled
    result = handle_config_command(
        ["lcb", "0", "garbage"], ctx, "connect4", "data")
    assert result == "config"
    assert ctx.players[0].lcb_enabled == before


def test_handle_epsilon_command_resets_mcts():
    ctx = _ctx_with_one_ai()
    ctx.players[0].mcts = "sentinel"
    handle_config_command(["epsilon", "0", "0.25"], ctx, "connect4", "data")
    assert ctx.players[0].epsilon == 0.25
    assert ctx.players[0].mcts is None


def test_handle_epsilon_out_of_range():
    ctx = _ctx_with_one_ai()
    handle_config_command(["epsilon", "0", "2.0"], ctx, "connect4", "data")
    # Out-of-range rejected; epsilon stays at default 0.0
    assert ctx.players[0].epsilon == 0.0


def test_handle_prune_off_means_zero():
    ctx = _ctx_with_one_ai()
    handle_config_command(["prune", "0", "off"], ctx, "connect4", "data")
    assert ctx.players[0].prune_frac == 0.0


def test_handle_prune_numeric():
    ctx = _ctx_with_one_ai()
    handle_config_command(["prune", "0", "0.05"], ctx, "connect4", "data")
    assert ctx.players[0].prune_frac == 0.05


def test_handle_prune_rejects_out_of_range():
    ctx = _ctx_with_one_ai()
    before = ctx.players[0].prune_frac
    handle_config_command(["prune", "0", "1.5"], ctx, "connect4", "data")
    assert ctx.players[0].prune_frac == before


def test_handle_cscale_command_resets_mcts():
    ctx = _ctx_with_one_ai()
    ctx.players[0].mcts = "sentinel"
    handle_config_command(["cscale", "0", "2.0"], ctx, "connect4", "data")
    assert ctx.players[0].gumbel_c_scale == 2.0
    assert ctx.players[0].mcts is None


def test_handle_cscale_rejects_zero_or_negative():
    ctx = _ctx_with_one_ai()
    before = ctx.players[0].gumbel_c_scale
    handle_config_command(["cscale", "0", "0"], ctx, "connect4", "data")
    assert ctx.players[0].gumbel_c_scale == before
    handle_config_command(["cscale", "0", "-1"], ctx, "connect4", "data")
    assert ctx.players[0].gumbel_c_scale == before


def test_handle_resign_numeric_enables_and_sets_threshold():
    ctx = _ctx_with_one_ai()
    handle_config_command(["resign", "0", "-0.9"], ctx, "connect4", "data")
    assert ctx.players[0].resign_enabled is True
    assert ctx.players[0].resign_threshold == -0.9


def test_handle_resign_on_off_toggles_enabled():
    ctx = _ctx_with_one_ai()
    handle_config_command(["resign", "0", "on"], ctx, "connect4", "data")
    assert ctx.players[0].resign_enabled is True
    handle_config_command(["resign", "0", "off"], ctx, "connect4", "data")
    assert ctx.players[0].resign_enabled is False


def test_handle_resign_out_of_range_rejected():
    ctx = _ctx_with_one_ai()
    handle_config_command(["resign", "0", "-2.0"], ctx, "connect4", "data")
    # Rejected; resign stays disabled
    assert ctx.players[0].resign_enabled is False


def test_handle_gumbel_opening_command():
    ctx = _ctx_with_one_ai()
    handle_config_command(["gumbel-opening", "0", "6"], ctx, "connect4", "data")
    assert ctx.players[0].gumbel_opening_moves == 6
    handle_config_command(["gumbel-opening", "0", "off"], ctx, "connect4", "data")
    assert ctx.players[0].gumbel_opening_moves == 0


def test_handle_algo_command_sets_override_and_rebuilds_mcts():
    ctx = _ctx_with_one_ai()
    ctx.players[0]._yaml_gumbel = {
        "gumbel_enabled": False, "gumbel_m": DEFAULT_GUMBEL_M,
        "gumbel_c_visit": DEFAULT_GUMBEL_C_VISIT,
        "gumbel_c_scale": DEFAULT_GUMBEL_C_SCALE, "gumbel_full": False,
    }
    ctx.players[0].mcts = "sentinel"
    handle_config_command(["algo", "0", "gumbel"], ctx, "connect4", "data")
    assert ctx.players[0].algo_override == "gumbel"
    assert ctx.players[0].gumbel_enabled is True
    assert ctx.players[0].mcts is None


def test_handle_algo_auto_resets_to_yaml():
    ctx = _ctx_with_one_ai()
    ctx.players[0]._yaml_gumbel = {
        "gumbel_enabled": True, "gumbel_m": DEFAULT_GUMBEL_M,
        "gumbel_c_visit": DEFAULT_GUMBEL_C_VISIT,
        "gumbel_c_scale": DEFAULT_GUMBEL_C_SCALE, "gumbel_full": False,
    }
    ctx.players[0].algo_override = "puct"
    _apply_player_gumbel(ctx.players[0])
    assert ctx.players[0].gumbel_enabled is False
    handle_config_command(["algo", "0", "auto"], ctx, "connect4", "data")
    assert ctx.players[0].algo_override is None
    assert ctx.players[0].gumbel_enabled is True


# ---------------------------------------------------------------------------
# Visit-count pruning degenerate case (get_ai_probs safeguard)
# ---------------------------------------------------------------------------


def test_get_ai_probs_visit_pruning_falls_back_when_all_below_floor():
    """When prune_frac is so high that the floor exceeds every visit count,
    get_ai_probs must keep the pre-prune distribution rather than producing
    NaN/zero probs that np.random.choice would reject."""
    Game = alphazero.Connect4GS
    gs = Game()
    ctx = PlayContext(gs, Game, cache_size=0)
    pcfg = ctx.players[0]
    pcfg.is_ai = True
    pcfg.network = None  # random eval path -> uniform pi
    pcfg.eval_type = "random"
    pcfg.node_limit = 7  # spread across 7 columns => uniform-ish counts
    pcfg.think_time = None
    pcfg.batch_size = 1
    pcfg.greedy = False
    pcfg.temperature = 0.5
    # prune_frac=0.5 with 7 sims => floor=3.5; nearly every move drops out
    # -> safeguard must restore the original distribution.
    pcfg.prune_frac = 0.5
    valids = np.array(gs.valid_moves())
    probs, source, sims, wld = get_ai_probs(ctx, 0, valids)
    assert np.isfinite(probs).all()
    assert probs.sum() == pytest.approx(1.0, abs=1e-5)
    # Some probability mass must survive on at least one valid move.
    assert (probs[valids == 1] > 0).any()
