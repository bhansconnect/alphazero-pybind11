"""Unit tests for neural_net.py -- spatial vs flat policy head selection.

Covers the generalized spatial policy head: games that expose a POLICY_SHAPE
(tafl, star gambit) get a spatial conv head under spatial_policy="auto"; other
games fall back to the flat FC head. Also covers the tri-state resolution and
checkpoint back-compat with the deprecated star_gambit_spatial bool.
"""

import io
import os
import sys
from dataclasses import asdict

import pytest
import torch
import zstandard as zstd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import alphazero
from neural_net import NNArgs, NNArch, NNWrapper


def _small_args(**kwargs):
    """A tiny network config; only the head selection matters for these tests."""
    base = dict(num_channels=8, depth=1, kernel_size=3, dense_net=False,
                head_channels=8)
    base.update(kwargs)
    return NNArgs(**base)


def _forward(game, args, batch=2):
    net = NNArch(game, args).cpu().eval()
    c, h, w = game.CANONICAL_SHAPE()
    x = torch.zeros(batch, c, h, w)
    with torch.no_grad():
        v, pi = net(x)
    return net, v, pi


# ---------------------------------------------------------------------------
# Spatial head: tafl (zero global actions, pure spatial conv head)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("game_name,expected_moves", [
    ("BrandubhGS", 7 * 7 * (7 + 7)),       # 686
    ("OpenTaflGS", 11 * 11 * (11 + 11)),   # 2662
    ("TawlbwrddGS", 11 * 11 * (11 + 11)),  # 2662
])
def test_tafl_spatial_head_no_globals(game_name, expected_moves):
    game = getattr(alphazero, game_name)
    net, _, pi = _forward(game, _small_args())
    assert net.spatial_policy is True
    assert net.num_global_actions == 0
    assert hasattr(net, "pi_conv2")
    assert not hasattr(net, "pi_global")
    assert pi.shape == (2, expected_moves)
    assert pi.shape[1] == game.NUM_MOVES()


# ---------------------------------------------------------------------------
# Spatial head: star gambit (spatial block + global actions)
# ---------------------------------------------------------------------------


def test_star_gambit_spatial_head_with_globals():
    game = alphazero.StarGambitSkirmishGS
    net, _, pi = _forward(game, _small_args())
    assert net.spatial_policy is True
    assert net.num_global_actions == 19  # 18 deploy + 1 end_turn
    assert hasattr(net, "pi_global")
    # Layout invariant: spatial block + globals tile the whole action space.
    _, h, w = game.CANONICAL_SHAPE()
    assert net.policy_channels * h * w + net.num_global_actions == game.NUM_MOVES()
    assert pi.shape == (2, game.NUM_MOVES())


# ---------------------------------------------------------------------------
# Flat-head fallback for non-spatial games
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("game_name", ["Connect4GS", "OnitamaGS"])
@pytest.mark.parametrize("spatial_policy", ["auto", "off"])
def test_non_spatial_games_use_flat_head(game_name, spatial_policy):
    game = getattr(alphazero, game_name)
    net, _, pi = _forward(game, _small_args(spatial_policy=spatial_policy))
    assert net.spatial_policy is False
    assert not hasattr(net, "pi_conv2")
    assert pi.shape == (2, game.NUM_MOVES())


# ---------------------------------------------------------------------------
# Tri-state resolution
# ---------------------------------------------------------------------------


def test_spatial_on_unsupported_game_raises():
    with pytest.raises(ValueError, match="does not expose POLICY_SHAPE"):
        NNArch(alphazero.Connect4GS, _small_args(spatial_policy="on"))


def test_spatial_off_forces_flat_head_on_supported_game():
    game = alphazero.BrandubhGS
    net, _, pi = _forward(game, _small_args(spatial_policy="off"))
    assert net.spatial_policy is False
    assert not hasattr(net, "pi_conv2")
    assert pi.shape == (2, game.NUM_MOVES())


def test_auto_with_pi_fc_layers_falls_back_to_flat():
    # The spatial head has no FC stack; auto + pi_fc_layers>0 -> flat (no error).
    game = alphazero.BrandubhGS
    net, _, pi = _forward(game, _small_args(pi_fc_layers=1))
    assert net.spatial_policy is False
    assert pi.shape == (2, game.NUM_MOVES())


def test_on_with_pi_fc_layers_raises():
    with pytest.raises(ValueError, match="pi_fc_layers not supported"):
        NNArch(alphazero.BrandubhGS, _small_args(spatial_policy="on", pi_fc_layers=1))


def test_legacy_star_gambit_spatial_bool_maps_on():
    args = _small_args(star_gambit_spatial=True)
    assert args.spatial_policy == "on"


# ---------------------------------------------------------------------------
# Checkpoint round-trip / back-compat
# ---------------------------------------------------------------------------


def _write_checkpoint(path, game, args_dict, state_dict, opt_state):
    """Write a checkpoint in the same format as NNWrapper.save_checkpoint."""
    buffer = io.BytesIO()
    torch.save(
        {
            "state_dict": state_dict,
            "opt_state": opt_state,
            "args": args_dict,
            "game": game,
            "version": "5.0",
        },
        buffer,
    )
    with open(path, "wb") as f:
        f.write(zstd.ZstdCompressor(level=1).compress(buffer.getvalue()))


def test_checkpoint_roundtrip_preserves_spatial_head(tmp_path):
    game = alphazero.BrandubhGS
    src = NNWrapper(game, _small_args())
    src.save_checkpoint(folder=str(tmp_path), filename="c.pt")
    reloaded = NNWrapper.load_checkpoint(game, folder=str(tmp_path), filename="c.pt")
    assert reloaded.args.spatial_policy == "auto"
    assert reloaded.nnet.spatial_policy is True


def test_legacy_flat_tafl_checkpoint_stays_flat(tmp_path):
    """A pre-spatial_policy flat tafl checkpoint must NOT become spatial."""
    game = alphazero.BrandubhGS
    src = NNWrapper(game, _small_args(spatial_policy="off"))
    args_dict = asdict(src.args)
    args_dict.pop("spatial_policy")  # emulate an old checkpoint
    # star_gambit_spatial stays False -> old code used the flat head.
    path = tmp_path / "legacy_flat.pt"
    _write_checkpoint(path, game, args_dict, src.nnet.state_dict(),
                      src.optimizer.state_dict())
    reloaded = NNWrapper.load_checkpoint(game, folder=str(tmp_path),
                                         filename="legacy_flat.pt")
    assert reloaded.args.spatial_policy == "off"
    assert reloaded.nnet.spatial_policy is False


def test_legacy_star_gambit_checkpoint_loads_spatial(tmp_path):
    """A pre-spatial_policy star-gambit checkpoint (star_gambit_spatial=True)
    must reconstruct the spatial head."""
    game = alphazero.StarGambitSkirmishGS
    src = NNWrapper(game, _small_args(spatial_policy="on"))
    args_dict = asdict(src.args)
    args_dict.pop("spatial_policy")  # emulate an old checkpoint
    args_dict["star_gambit_spatial"] = True  # legacy spatial marker
    path = tmp_path / "legacy_sg.pt"
    _write_checkpoint(path, game, args_dict, src.nnet.state_dict(),
                      src.optimizer.state_dict())
    reloaded = NNWrapper.load_checkpoint(game, folder=str(tmp_path),
                                         filename="legacy_sg.pt")
    assert reloaded.args.spatial_policy == "on"
    assert reloaded.nnet.spatial_policy is True
