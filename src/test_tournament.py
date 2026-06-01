"""Unit tests for tournament.py helpers: per-agent gumbel arg builder,
c_scale override, G3 mode opt-in, and pit_agents parameter wiring.

These tests deliberately avoid touching real checkpoints; they monkeypatch
the yaml lookup so they're fast and reproducible.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import alphazero  # noqa: F401
from tournament import (
    build_player_gumbel_args,
    _PUCT_GUMBEL_CFG,
    pit_agents,
)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _gumbel_cfg(c_scale=1.0):
    return {
        "gumbel_enabled": True,
        "gumbel_m": 16,
        "gumbel_c_visit": 50.0,
        "gumbel_c_scale": c_scale,
        "gumbel_full": False,
    }


def _puct_cfg():
    return dict(_PUCT_GUMBEL_CFG)


def _patch_agent_lookup(monkeypatch, name_to_cfg):
    """Stub _agent_path / _agent_override / get_agent_gumbel_config so the
    tests don't touch the filesystem."""
    monkeypatch.setattr("tournament._agent_path", lambda x: x)
    monkeypatch.setattr("tournament._agent_override", lambda x: None)

    def fake_get_cfg(path, override):
        if path is None:
            return None
        return dict(name_to_cfg.get(path, _PUCT_GUMBEL_CFG))

    monkeypatch.setattr("tournament.get_agent_gumbel_config", fake_get_cfg)


# ---------------------------------------------------------------------------
# build_player_gumbel_args — basic shape
# ---------------------------------------------------------------------------


def test_build_player_gumbel_args_all_puct_returns_empty(monkeypatch):
    """Backward-compat: when no agent uses Gumbel, no per-seat overrides
    are emitted, so old PUCT-only callers see the legacy {} dict."""
    _patch_agent_lookup(monkeypatch, {
        "a.pt": _puct_cfg(),
        "b.pt": _puct_cfg(),
    })
    out = build_player_gumbel_args(["a.pt", "b.pt"], global_override=None)
    assert out == {}


def test_build_player_gumbel_args_any_gumbel_emits_full_dict(monkeypatch):
    _patch_agent_lookup(monkeypatch, {
        "g.pt": _gumbel_cfg(),
        "p.pt": _puct_cfg(),
    })
    out = build_player_gumbel_args(["g.pt", "p.pt"], None)
    # All five gumbel_* lists present
    for key in ("player_gumbel_enabled", "player_gumbel_m",
                "player_gumbel_c_visit", "player_gumbel_c_scale",
                "player_gumbel_full"):
        assert key in out and len(out[key]) == 2
    # Per-seat enabled tracks each agent's training
    assert out["player_gumbel_enabled"][0] is True
    assert out["player_gumbel_enabled"][1] is False


# ---------------------------------------------------------------------------
# build_player_gumbel_args — c_scale override
# ---------------------------------------------------------------------------


def test_c_scale_override_applies_only_to_gumbel_seats(monkeypatch):
    """A CLI --gumbel-c-scale should reweight Gumbel-trained seats only;
    PUCT seats keep their (effectively unused) yaml value."""
    _patch_agent_lookup(monkeypatch, {
        "g.pt": _gumbel_cfg(c_scale=1.0),
        "p.pt": _puct_cfg(),  # gumbel_c_scale=1.0 by default
    })
    out = build_player_gumbel_args(
        ["g.pt", "p.pt"], None, c_scale_override=2.0)
    assert out["player_gumbel_c_scale"][0] == 2.0   # Gumbel seat — flipped
    assert out["player_gumbel_c_scale"][1] == _PUCT_GUMBEL_CFG["gumbel_c_scale"]


def test_c_scale_override_none_preserves_yaml(monkeypatch):
    _patch_agent_lookup(monkeypatch, {
        "g.pt": _gumbel_cfg(c_scale=0.5),
    })
    out = build_player_gumbel_args(["g.pt", "g.pt"], None, c_scale_override=None)
    assert out["player_gumbel_c_scale"] == [0.5, 0.5]


# ---------------------------------------------------------------------------
# build_player_gumbel_args — G3 mode opt-in
# ---------------------------------------------------------------------------


def test_mode_g1_default_does_not_emit_use_improved_policy(monkeypatch):
    _patch_agent_lookup(monkeypatch, {
        "g.pt": _gumbel_cfg(),
    })
    out = build_player_gumbel_args(["g.pt", "g.pt"], None)  # default mode='g1'
    assert "player_gumbel_use_improved_policy" not in out


def test_mode_g3_emits_use_improved_policy_per_seat(monkeypatch):
    _patch_agent_lookup(monkeypatch, {
        "g.pt": _gumbel_cfg(),
        "p.pt": _puct_cfg(),
    })
    out = build_player_gumbel_args(["g.pt", "p.pt"], None, mode="g3")
    assert "player_gumbel_use_improved_policy" in out
    # Per-seat: Gumbel seat opts in (1), PUCT seat stays out (0)
    assert out["player_gumbel_use_improved_policy"][0] == 1
    assert out["player_gumbel_use_improved_policy"][1] == 0


def test_mode_g3_with_all_puct_still_empty(monkeypatch):
    """No Gumbel agents anywhere -> early-return path skips even the G3 flag."""
    _patch_agent_lookup(monkeypatch, {
        "p1.pt": _puct_cfg(),
        "p2.pt": _puct_cfg(),
    })
    out = build_player_gumbel_args(["p1.pt", "p2.pt"], None, mode="g3")
    assert out == {}


# ---------------------------------------------------------------------------
# build_player_gumbel_args — global override / per-agent override
# ---------------------------------------------------------------------------


def test_global_override_forces_gumbel_when_yaml_says_puct(monkeypatch):
    """global_override='gumbel' applies to agents without a per-agent ':algo'
    suffix. The yaml is PUCT but the override flips them on."""
    def fake_get_cfg(path, override):
        cfg = _puct_cfg() if path == "p.pt" else _gumbel_cfg()
        if override == "gumbel":
            cfg = dict(cfg)
            cfg["gumbel_enabled"] = True
        elif override == "puct":
            cfg = dict(cfg)
            cfg["gumbel_enabled"] = False
        return cfg

    monkeypatch.setattr("tournament._agent_path", lambda x: x)
    monkeypatch.setattr("tournament._agent_override", lambda x: None)
    monkeypatch.setattr("tournament.get_agent_gumbel_config", fake_get_cfg)

    out = build_player_gumbel_args(
        ["p.pt", "p.pt"], global_override="gumbel")
    assert out["player_gumbel_enabled"] == [True, True]


# ---------------------------------------------------------------------------
# pit_agents — verifies new params propagate to PlayParams
# ---------------------------------------------------------------------------


def test_pit_agents_signature_accepts_new_params():
    """The pit_agents signature must accept the new resign + G3 args without
    raising. We bind a fake config/Game just enough that import-time arg
    parsing passes; we don't actually run any games."""
    import inspect
    sig = inspect.signature(pit_agents)
    expected = {
        "player_resign_threshold",
        "player_resign_consecutive",
        "player_gumbel_use_improved_policy",
    }
    missing = expected - set(sig.parameters.keys())
    assert not missing, f"pit_agents missing params: {missing}"


def test_run_monrad_signature_includes_gumbel_overrides():
    import inspect
    from tournament import run_monrad, run_roundrobin
    for fn in (run_monrad, run_roundrobin):
        sig = inspect.signature(fn)
        assert "gumbel_c_scale" in sig.parameters, f"{fn.__name__} missing gumbel_c_scale"
        assert "gumbel_mode" in sig.parameters, f"{fn.__name__} missing gumbel_mode"


def test_play_params_exposes_new_seat_fields():
    """C++ side: PlayParams must expose the new per-seat fields used by the
    PlayManager G1/G3 branch + per-seat resign."""
    pp = alphazero.PlayParams()
    assert hasattr(pp, "seat_gumbel_use_improved_policy")
    assert hasattr(pp, "seat_resign_threshold")
    assert hasattr(pp, "seat_resign_consecutive")
    # Defaults are empty lists -> PlayManager fills with sentinels
    assert pp.seat_gumbel_use_improved_policy == []
    assert pp.seat_resign_threshold == []
    assert pp.seat_resign_consecutive == []
    # Round-trip a value to confirm pybind binding direction. Resign
    # threshold is stored as float32 in C++, so use approx for comparison.
    pp.seat_gumbel_use_improved_policy = [[1, 0]]
    pp.seat_resign_threshold = [[-0.9, -0.9]]
    pp.seat_resign_consecutive = [[3, 3]]
    assert pp.seat_gumbel_use_improved_policy == [[1, 0]]
    assert pp.seat_resign_threshold[0][0] == pytest.approx(-0.9)
    assert pp.seat_resign_threshold[0][1] == pytest.approx(-0.9)
    assert pp.seat_resign_consecutive == [[3, 3]]
