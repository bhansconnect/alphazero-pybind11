"""Unit tests for tafl_ui.py: move encode/decode and parse/format round-trips."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import alphazero
from tafl_ui import TaflUI
from game_ui import get_game_ui


def _ui():
    return TaflUI(7, 7)


def test_encode_decode_roundtrip():
    ui = _ui()
    M = alphazero.BrandubhGS.NUM_MOVES()
    assert M == 7 * 7 * (7 + 7)
    for a in range(M):
        fr, to = ui._decode(a)
        if fr == to:
            # Self-moves are not encodable (and never legal).
            assert ui._encode(fr, to) is None
        else:
            assert ui._encode(fr, to) == a


def test_parse_move_roundtrip_on_valid_moves():
    ui = _ui()
    gs = alphazero.BrandubhGS()
    valids = gs.valid_moves()
    valid_actions = [i for i in range(len(valids)) if valids[i]]
    assert valid_actions, "fresh board should have legal moves"
    for a in valid_actions:
        short = ui.format_move_short(gs, a)
        assert ui.parse_move(gs, short, valids) == a
        # The dashed form parses identically.
        fr, to = ui._decode(a)
        dashed = f"{ui._sq_to_alg(*fr)}-{ui._sq_to_alg(*to)}"
        assert ui.parse_move(gs, dashed, valids) == a


def test_parse_move_rejects_diagonal_and_out_of_range():
    ui = _ui()
    gs = alphazero.BrandubhGS()
    valids = gs.valid_moves()
    assert ui.parse_move(gs, "a1b2", valids) is None   # diagonal: not encodable
    assert ui.parse_move(gs, "z9z9", valids) is None   # off-board files/ranks
    assert ui.parse_move(gs, "a1a1", valids) is None    # self-move


def test_integer_fallback():
    ui = _ui()
    gs = alphazero.BrandubhGS()
    valids = gs.valid_moves()
    a = next(i for i in range(len(valids)) if valids[i])
    assert ui.parse_move(gs, str(a), valids) == a


def test_algebraic_orientation():
    ui = _ui()
    assert ui._sq_to_alg(6, 0) == "a1"   # bottom-left
    assert ui._sq_to_alg(0, 0) == "a7"   # top-left
    assert ui._sq_to_alg(0, 6) == "g7"   # top-right


def test_registry_returns_tafl_ui():
    for name in ("brandubh", "open_tafl", "tawlbwrdd"):
        ui = get_game_ui(name)
        assert isinstance(ui, TaflUI)
    assert get_game_ui("open_tafl").W == 11


def test_build_action_menu_groups_by_origin():
    ui = _ui()
    gs = alphazero.BrandubhGS()
    valids = gs.valid_moves()
    entries = ui.build_action_menu(gs, None, valids)
    headers = [e for e in entries if e[0] == "header"]
    actions = [e for e in entries if e[0] == "action"]
    assert headers and actions
    assert all(h[1].startswith(("From ", "@ From", "O From", "X From")) for h in headers)
    # Every valid action appears exactly once.
    assert sorted(a[1] for a in actions) == [i for i in range(len(valids)) if valids[i]]
