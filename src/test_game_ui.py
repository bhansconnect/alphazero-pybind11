"""Tests for game_ui.py and star_gambit_ui.py."""

import io
import contextlib
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import alphazero
from game_ui import GameUI, get_game_ui
from star_gambit_ui import StarGambitUI
from star_gambit_play import GameConfig, SKIRMISH


# --- Generic GameUI ---


def test_generic_parse_move_valid():
    """Generic UI parses valid integer action IDs."""
    ui = GameUI()
    gs = alphazero.Connect4GS()
    valids = gs.valid_moves()
    # Column 0 should be valid in initial Connect4 state
    assert ui.parse_move(gs, "0", valids) == 0


def test_generic_parse_move_invalid_string():
    """Generic UI returns None for non-integer input."""
    ui = GameUI()
    gs = alphazero.Connect4GS()
    valids = gs.valid_moves()
    assert ui.parse_move(gs, "abc", valids) is None
    assert ui.parse_move(gs, "", valids) is None


def test_generic_parse_move_out_of_range():
    """Generic UI returns None for out-of-range action."""
    ui = GameUI()
    gs = alphazero.Connect4GS()
    valids = gs.valid_moves()
    assert ui.parse_move(gs, "9999", valids) is None
    assert ui.parse_move(gs, "-1", valids) is None


def test_generic_parse_move_invalid_action():
    """Generic UI returns None for in-range but not-valid action."""
    ui = GameUI()
    gs = alphazero.Connect4GS()
    valids = gs.valid_moves()
    # Find an invalid action if any
    invalid = [i for i in range(len(valids)) if not valids[i]]
    if invalid:
        assert ui.parse_move(gs, str(invalid[0]), valids) is None


def test_generic_format_move():
    """Generic UI formats moves as string of action ID."""
    ui = GameUI()
    gs = alphazero.Connect4GS()
    assert ui.format_move(gs, 0) == "0"
    assert ui.format_move(gs, 42) == "42"


def test_generic_display_board():
    """Generic UI returns non-empty string for board display."""
    ui = GameUI()
    gs = alphazero.Connect4GS()
    board = ui.display_board(gs)
    assert isinstance(board, str)
    assert len(board) > 0


def test_generic_valid_move_descriptions():
    """Generic UI lists all valid moves with descriptions."""
    ui = GameUI()
    gs = alphazero.Connect4GS()
    valids = gs.valid_moves()
    descs = ui.get_valid_move_descriptions(gs, valids)
    num_valid = sum(1 for v in valids if v)
    assert len(descs) == num_valid
    for action_id, desc in descs:
        assert valids[action_id]


def test_generic_display_actions_menu():
    """Generic UI display_actions_menu prints top-N summary."""
    ui = GameUI()
    gs = alphazero.Connect4GS()
    valids = np.array(gs.valid_moves(), dtype=np.float32)
    probs = np.zeros(len(valids))
    valid_idx = np.where(valids > 0)[0]
    probs[valid_idx] = 1.0 / len(valid_idx)
    # Should not raise
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        ui.display_actions_menu(gs, probs, valids)
    output = f.getvalue()
    assert "Top moves:" in output


# --- GameUI Registry ---


def test_get_game_ui_connect4():
    """Connect4 falls back to generic GameUI."""
    ui = get_game_ui("connect4")
    assert isinstance(ui, GameUI)


def test_get_game_ui_star_gambit():
    """Star Gambit returns StarGambitUI."""
    ui = get_game_ui("star_gambit_skirmish")
    assert isinstance(ui, StarGambitUI)


def test_get_game_ui_unknown():
    """Unknown game falls back to generic GameUI."""
    ui = get_game_ui("nonexistent_game")
    assert isinstance(ui, GameUI)
    assert not isinstance(ui, StarGambitUI)


# --- StarGambitUI ---


def test_sg_ui_display_board():
    """Star Gambit UI returns non-empty board with unit info."""
    ui = StarGambitUI(SKIRMISH)
    gs = alphazero.StarGambitSkirmishGS()
    board = ui.display_board(gs)
    assert isinstance(board, str)
    assert len(board) > 0


def test_sg_ui_format_move_end_turn():
    """Star Gambit UI formats end-turn action."""
    ui = StarGambitUI(SKIRMISH)
    gs = alphazero.StarGambitSkirmishGS()
    # End turn is the last action
    desc = ui.format_move(gs, SKIRMISH.end_turn_offset)
    assert isinstance(desc, str)
    assert len(desc) > 0


def test_sg_ui_valid_move_descriptions():
    """Star Gambit UI describes all valid moves."""
    ui = StarGambitUI(SKIRMISH)
    gs = alphazero.StarGambitSkirmishGS()
    valids = gs.valid_moves()
    descs = ui.get_valid_move_descriptions(gs, valids)
    num_valid = sum(1 for v in valids if v)
    assert len(descs) == num_valid
    for action_id, desc in descs:
        assert isinstance(desc, str)
        assert len(desc) > 0


def test_sg_ui_display_actions_menu():
    """Star Gambit UI display_actions_menu shows grouped categories."""
    ui = StarGambitUI(SKIRMISH)
    gs = alphazero.StarGambitSkirmishGS()
    valids = np.array(gs.valid_moves(), dtype=np.float32)
    probs = np.zeros(len(valids))
    valid_idx = np.where(valids > 0)[0]
    probs[valid_idx] = 1.0 / len(valid_idx)
    # Capture stdout
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        ui.display_actions_menu(gs, probs, valids)
    output = f.getvalue()
    assert len(output) > 0  # Should produce non-empty output


# --- Variant selection ---


def test_generic_select_variant_returns_none():
    """Generic GameUI has no variant selection."""
    ui = GameUI()
    assert ui.select_variant() is None


def test_sg_ui_select_variant_skirmish(monkeypatch):
    """Star Gambit variant selection returns correct game name."""
    ui = StarGambitUI(SKIRMISH)
    monkeypatch.setattr("builtins.input", lambda _: "1")
    assert ui.select_variant() == "star_gambit_skirmish"


def test_sg_ui_select_variant_clash(monkeypatch):
    """Star Gambit variant selection returns clash for choice 2."""
    ui = StarGambitUI(SKIRMISH)
    monkeypatch.setattr("builtins.input", lambda _: "2")
    assert ui.select_variant() == "star_gambit_clash"


def test_sg_ui_select_variant_battle(monkeypatch):
    """Star Gambit variant selection returns battle for choice 3."""
    ui = StarGambitUI(SKIRMISH)
    monkeypatch.setattr("builtins.input", lambda _: "3")
    assert ui.select_variant() == "star_gambit_battle"


def test_sg_ui_select_variant_default(monkeypatch):
    """Star Gambit variant selection defaults to skirmish."""
    ui = StarGambitUI(SKIRMISH)
    monkeypatch.setattr("builtins.input", lambda _: "")
    assert ui.select_variant() == "star_gambit_skirmish"
