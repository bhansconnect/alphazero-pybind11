"""Tests for per-variant temp_decay_half_life plumbing.

Covers:
  - PlayParams binding round-trip for temp_decay_half_life_by_variant
  - base_params() builds the per-variant vector in canonical
    UNIFIED_VARIANT_NAMES order when config holds a dict
  - base_params() falls back to scalar when config holds an int
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import alphazero
from config import TrainConfig
from game_runner import base_params, UNIFIED_VARIANT_NAMES


def test_play_params_per_variant_roundtrip():
    params = alphazero.PlayParams()
    assert list(params.temp_decay_half_life_by_variant) == []
    params.temp_decay_half_life_by_variant = [3.0, 4.0, 6.0, 12.0]
    assert list(params.temp_decay_half_life_by_variant) == [3.0, 4.0, 6.0, 12.0]


def test_base_params_dispatches_dict_to_vector():
    cfg = TrainConfig(
        game="star_gambit_unified",
        temp_decay_half_life={
            "skirmish": 3, "showdown": 4, "clash": 6, "battle": 12,
        },
    )
    params = base_params(cfg, start_temp=1.25, bs=64, cb=2)
    # Scalar zeroed so C++ knows to use the vector.
    assert params.temp_decay_half_life == 0.0
    # Order must match UNIFIED_VARIANT_NAMES (variant_id indexing).
    assert UNIFIED_VARIANT_NAMES == ["skirmish", "showdown", "clash", "battle"]
    assert list(params.temp_decay_half_life_by_variant) == [3.0, 4.0, 6.0, 12.0]


def test_base_params_scalar_when_int():
    cfg = TrainConfig(game="connect4", temp_decay_half_life=10)
    params = base_params(cfg, start_temp=1.0, bs=64, cb=2)
    assert params.temp_decay_half_life == 10.0
    assert list(params.temp_decay_half_life_by_variant) == []


def test_base_params_scalar_when_zero():
    """0 (decay disabled) still routes through the scalar path."""
    cfg = TrainConfig(game="connect4", temp_decay_half_life=0)
    params = base_params(cfg, start_temp=1.0, bs=64, cb=2)
    assert params.temp_decay_half_life == 0.0
    assert list(params.temp_decay_half_life_by_variant) == []


def test_base_params_dict_order_independent_of_yaml_order():
    """YAML key order doesn't matter — vector is ordered by variant_id."""
    cfg = TrainConfig(
        game="star_gambit_unified",
        temp_decay_half_life={
            "battle": 12, "skirmish": 3, "clash": 6, "showdown": 4,
        },
    )
    params = base_params(cfg, start_temp=1.25, bs=64, cb=2)
    assert list(params.temp_decay_half_life_by_variant) == [3.0, 4.0, 6.0, 12.0]
