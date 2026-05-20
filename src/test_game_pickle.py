"""Pickle round-trip tests for every game state class in the bindings.

Verifies that `pickle.dumps(gs)` + `pickle.loads(...)` preserves all
observable state (canonical form, current player, current turn, valid
moves, terminal scores). One test per game; uses a shared helper.

These tests are the contract for the C++ to_bytes/from_bytes implementations.
"""

import pickle

import alphazero
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _play_some_moves(gs, num_moves: int):
    """Play `num_moves` deterministic moves (or until game ends).

    Picks the first valid move at each step so the test is reproducible
    across all games, regardless of move-space layout.
    """
    for _ in range(num_moves):
        if gs.scores() is not None:
            break
        valid = np.array(gs.valid_moves())
        valid_idx = np.flatnonzero(valid)
        if len(valid_idx) == 0:
            break
        gs.play_move(int(valid_idx[0]))


def _assert_state_equal(gs_a, gs_b, label: str = ""):
    """Verify that two game states agree on everything an MCTS or network
    would observe: canonical tensor, current player, current turn, valid
    moves, terminal score (if any)."""
    can_a = np.array(gs_a.canonicalized())
    can_b = np.array(gs_b.canonicalized())
    assert np.array_equal(can_a, can_b), f"{label}: canonical mismatch"
    assert gs_a.current_player() == gs_b.current_player(), \
        f"{label}: current_player mismatch"
    assert gs_a.current_turn() == gs_b.current_turn(), \
        f"{label}: current_turn mismatch"
    valid_a = np.array(gs_a.valid_moves())
    valid_b = np.array(gs_b.valid_moves())
    assert np.array_equal(valid_a, valid_b), f"{label}: valid_moves mismatch"
    scores_a = gs_a.scores()
    scores_b = gs_b.scores()
    if scores_a is None:
        assert scores_b is None, f"{label}: one terminal, one not"
    else:
        assert scores_b is not None, f"{label}: one terminal, one not"
        assert np.array_equal(np.array(scores_a), np.array(scores_b)), \
            f"{label}: scores mismatch"


def _round_trip_check(make_gs, *, moves_to_play: int):
    """Construct a fresh game, play some moves, pickle, unpickle, verify."""
    gs = make_gs()
    _play_some_moves(gs, moves_to_play)
    blob = pickle.dumps(gs)
    gs2 = pickle.loads(blob)
    _assert_state_equal(gs, gs2,
                        label=f"{type(gs).__name__}@{moves_to_play}moves")


# ---------------------------------------------------------------------------
# Test parametrization: every game class in the bindings.
# ---------------------------------------------------------------------------

# (label, factory) for each game. Factories avoid eager class lookups so
# pytest doesn't fail to collect when a binding goes missing.
ALL_GAMES = [
    ("Connect4GS",        lambda: alphazero.Connect4GS()),
    ("OnitamaGS",         lambda: alphazero.OnitamaGS()),
    ("BrandubhGS",        lambda: alphazero.BrandubhGS()),
    ("OpenTaflGS",        lambda: alphazero.OpenTaflGS()),
    ("TawlbwrddGS",       lambda: alphazero.TawlbwrddGS()),
    ("PhotosynthesisGS2", lambda: alphazero.PhotosynthesisGS2()),
    ("PhotosynthesisGS3", lambda: alphazero.PhotosynthesisGS3()),
    ("PhotosynthesisGS4", lambda: alphazero.PhotosynthesisGS4()),
    ("StarGambitSkirmishGS", lambda: alphazero.StarGambitSkirmishGS()),
    ("StarGambitShowdownGS", lambda: alphazero.StarGambitShowdownGS()),
    ("StarGambitClashGS",    lambda: alphazero.StarGambitClashGS()),
    ("StarGambitBattleGS",   lambda: alphazero.StarGambitBattleGS()),
    ("StarGambitUnifiedGS-skirmish",
     lambda: alphazero.StarGambitUnifiedGS(pinned_variant=0)),
    ("StarGambitUnifiedGS-showdown",
     lambda: alphazero.StarGambitUnifiedGS(pinned_variant=1)),
    ("StarGambitUnifiedGS-clash",
     lambda: alphazero.StarGambitUnifiedGS(pinned_variant=2)),
    ("StarGambitUnifiedGS-battle",
     lambda: alphazero.StarGambitUnifiedGS(pinned_variant=3)),
]


@pytest.mark.parametrize("label,make_gs", ALL_GAMES, ids=[g[0] for g in ALL_GAMES])
def test_round_trip_initial_state(label, make_gs):
    """Fresh game state pickle/unpickle preserves all observables."""
    _round_trip_check(make_gs, moves_to_play=0)


@pytest.mark.parametrize("label,make_gs", ALL_GAMES, ids=[g[0] for g in ALL_GAMES])
def test_round_trip_after_short_play(label, make_gs):
    """Pickle after a few moves preserves everything."""
    _round_trip_check(make_gs, moves_to_play=4)


@pytest.mark.parametrize("label,make_gs", ALL_GAMES, ids=[g[0] for g in ALL_GAMES])
def test_round_trip_after_long_play(label, make_gs):
    """Pickle further into the game (catches state that only changes later)."""
    _round_trip_check(make_gs, moves_to_play=20)


@pytest.mark.parametrize("label,make_gs", ALL_GAMES, ids=[g[0] for g in ALL_GAMES])
def test_pickle_unpickle_chain(label, make_gs):
    """Multiple pickle round-trips don't drift — pickle, unpickle, play,
    pickle again, unpickle, verify still equal."""
    gs = make_gs()
    _play_some_moves(gs, 3)
    gs = pickle.loads(pickle.dumps(gs))
    _play_some_moves(gs, 3)
    gs = pickle.loads(pickle.dumps(gs))
    _play_some_moves(gs, 3)
    _assert_state_equal(gs, pickle.loads(pickle.dumps(gs)),
                        label=f"{label}-chain")


# ---------------------------------------------------------------------------
# Special: continued-play equivalence — after pickle round-trip, playing
# the same next moves on both copies must produce identical states. Catches
# state members that are observable only after subsequent play (e.g.,
# repetition counters, sun_phase transitions).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("label,make_gs", ALL_GAMES, ids=[g[0] for g in ALL_GAMES])
def test_round_trip_continues_identically(label, make_gs):
    """After pickle, future moves produce identical states."""
    gs_orig = make_gs()
    _play_some_moves(gs_orig, 5)

    gs_restored = pickle.loads(pickle.dumps(gs_orig))

    # Play the same valid-move sequence on both for 5 more moves.
    for _ in range(5):
        if gs_orig.scores() is not None:
            break
        valid = np.array(gs_orig.valid_moves())
        valid_idx = np.flatnonzero(valid)
        if len(valid_idx) == 0:
            break
        m = int(valid_idx[0])
        gs_orig.play_move(m)
        gs_restored.play_move(m)

    _assert_state_equal(gs_orig, gs_restored, label=f"{label}-continue")
