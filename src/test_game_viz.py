"""Unit tests for game_viz.py shared visualization helpers."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from game_viz import (
    quartile_masks, draw_square_heatmap, draw_na, value_calibration_figure,
)


# ---------------------------------------------------------------------------
# quartile_masks
# ---------------------------------------------------------------------------


def test_quartile_masks_partition():
    progress = np.linspace(0, 1, 400)
    masks = quartile_masks(progress)
    assert len(masks) == 4
    bool_masks = [m for _, m in masks]
    # Disjoint and covering.
    stacked = np.stack(bool_masks)
    assert (stacked.sum(axis=0) == 1).all()
    assert stacked.sum() == 400
    # Each non-empty and ordered earliest->latest by mean progress.
    means = [progress[m].mean() for m in bool_masks]
    assert all(m.any() for m in bool_masks)
    assert means == sorted(means)


def test_quartile_masks_degenerate_all_equal():
    progress = np.full(40, 0.5)
    masks = quartile_masks(progress)
    bool_masks = [m for _, m in masks]
    # Even with identical values, every bin must be non-empty and disjoint.
    assert all(m.any() for m in bool_masks)
    stacked = np.stack(bool_masks)
    assert (stacked.sum(axis=0) == 1).all()
    assert stacked.sum() == 40


def test_quartile_masks_empty():
    masks = quartile_masks(np.zeros(0))
    assert len(masks) == 4
    assert all(m.shape == (0,) and not m.any() for _, m in masks)


# ---------------------------------------------------------------------------
# draw_square_heatmap / draw_na
# ---------------------------------------------------------------------------


def test_draw_square_heatmap_valid_grid():
    fig, ax = plt.subplots()
    grid = np.arange(49, dtype=float).reshape(7, 7)
    im = draw_square_heatmap(ax, grid, special=[(0, 0), (3, 3), (6, 6)])
    assert im is not None
    plt.close(fig)


def test_draw_square_heatmap_all_nan_draws_na():
    fig, ax = plt.subplots()
    grid = np.full((7, 7), np.nan)
    im = draw_square_heatmap(ax, grid)
    assert im is None  # falls back to N/A panel without raising
    plt.close(fig)


def test_draw_na_runs():
    fig, ax = plt.subplots()
    draw_na(ax)
    plt.close(fig)


# ---------------------------------------------------------------------------
# value_calibration_figure
# ---------------------------------------------------------------------------


def test_value_calibration_single_bucket():
    rng = np.random.default_rng(0)
    vp = rng.uniform(0, 1, 500)
    va = (rng.uniform(0, 1, 500) < vp).astype(float)
    fig = value_calibration_figure({"overall": {"v_pred": vp, "v_actual": va}}, 3)
    assert fig is not None
    plt.close(fig)


def test_value_calibration_empty_returns_none():
    assert value_calibration_figure({}, 0) is None


# ---------------------------------------------------------------------------
# generate_tafl_visualizations smoke test
# ---------------------------------------------------------------------------


def test_generate_tafl_visualizations_smoke(tmp_path):
    aim = pytest.importorskip("aim")  # driver no-ops without aim
    import torch
    from game_runner import save_compressed, generate_tafl_visualizations
    from config import TrainConfig

    N, H, W, C = 400, 7, 7, 7
    M = H * W * (W + H)
    rng = np.random.default_rng(0)

    # Board occupancy: king fixed at the throne; defender/attacker fill
    # decreases across samples so capture-progress spreads across quartiles.
    c = np.zeros((N, C, H, W), dtype=np.float32)
    c[:, 0, H // 2, W // 2] = 1.0
    for i in range(N):
        frac = 1.0 - i / N
        for ch in (1, 2):
            mask = rng.random((H, W)) < (0.3 * frac)
            mask[H // 2, W // 2] = False
            c[i, ch][mask] = 1.0

    pi = rng.random((N, M)).astype(np.float32)
    pi /= pi.sum(axis=1, keepdims=True)
    v = rng.random((N, 3)).astype(np.float32)
    v /= v.sum(axis=1, keepdims=True)

    hist = tmp_path / "history"
    hist.mkdir()
    save_compressed(torch.tensor(c), str(hist / f"0000-exp-canonical-{N}.ptz"))
    save_compressed(torch.tensor(v), str(hist / f"0000-exp-v-{N}.ptz"))
    save_compressed(torch.tensor(pi), str(hist / f"0000-exp-pi-{N}.ptz"))

    paths = {"history": str(hist), "checkpoint": str(tmp_path / "ckpt")}
    config = TrainConfig(game="brandubh")
    config.validate()

    class _Run:
        def __init__(self):
            self.names = []

        def track(self, image, name=None, **kw):
            self.names.append(name)

    run = _Run()
    generate_tafl_visualizations(config, paths, "exp", 0, run, 0)
    # No checkpoint on disk -> value calibration is skipped; the four
    # distribution figures must still be emitted.
    for expected in ("tafl_move_from_by_phase", "tafl_move_from_by_piece",
                     "tafl_piece_counts_by_phase", "tafl_move_geometry_by_phase"):
        assert expected in run.names


def test_analyze_iteration_overall_bucket_for_non_unified(tmp_path):
    """_analyze_iteration_variants returns a single 'overall' bucket (with
    per-sample arrays) for non-unified games like brandubh."""
    import torch
    from game_runner import save_compressed, _analyze_iteration_variants
    from config import TrainConfig
    import neural_net

    N, H, W, C = 64, 7, 7, 7
    M = H * W * (W + H)
    rng = np.random.default_rng(1)
    c = (rng.random((N, C, H, W)) < 0.2).astype(np.float32)
    pi = rng.random((N, M)).astype(np.float32)
    pi /= pi.sum(axis=1, keepdims=True)
    v = rng.random((N, 3)).astype(np.float32)
    v /= v.sum(axis=1, keepdims=True)

    hist = tmp_path / "history"; hist.mkdir()
    ckpt = tmp_path / "ckpt"; ckpt.mkdir()
    save_compressed(torch.tensor(c), str(hist / f"0000-exp-canonical-{N}.ptz"))
    save_compressed(torch.tensor(v), str(hist / f"0000-exp-v-{N}.ptz"))
    save_compressed(torch.tensor(pi), str(hist / f"0000-exp-pi-{N}.ptz"))

    config = TrainConfig(game="brandubh")
    config.validate()
    # Trained checkpoint for iteration 0 is named 0001-<exp>.pt.
    net = neural_net.NNWrapper(
        config.Game,
        neural_net.NNArgs(num_channels=8, depth=1, kernel_size=3,
                          dense_net=False, head_channels=8))
    net.save_checkpoint(str(ckpt), "0001-exp.pt")

    paths = {"history": str(hist), "checkpoint": str(ckpt)}
    result = _analyze_iteration_variants(config, paths, "exp", 0)
    assert set(result.keys()) == {"overall"}
    ov = result["overall"]
    for key in ("pi_loss", "v_loss", "entropy", "top1", "net_top1", "net_at_mcts",
                "top1_gap", "top1_agree", "v_pred", "v_actual"):
        assert ov[key].shape == (N,)
    # top1_gap is the signed identity top1 - net_at_mcts.
    np.testing.assert_allclose(ov["top1_gap"], ov["top1"] - ov["net_at_mcts"], rtol=1e-5, atol=1e-6)
    # Agreement is a 0/1 indicator.
    assert set(np.unique(ov["top1_agree"])).issubset({0.0, 1.0})
