"""Focused unit tests for frozen_eval helpers.

End-to-end behavior (snapshot generation, MCTS-based evaluation) requires
a full game environment and a checkpoint — those paths are smoke-tested via
the live training loop with flags enabled, not here.
"""

import os

import numpy as np
import pytest
import torch

import frozen_eval


# ---------------------------------------------------------------------------
# Path / utility helpers
# ---------------------------------------------------------------------------

def test_anchor_dir_format(tmp_path):
    paths = {"experiment": str(tmp_path)}
    out = frozen_eval.frozen_set_dir(paths, 50)
    assert out.endswith(os.path.join("frozen_eval", "anchor_0050"))


def test_anchor_dir_zero_padding(tmp_path):
    paths = {"experiment": str(tmp_path)}
    assert frozen_eval.frozen_set_dir(paths, 1).endswith("anchor_0001")
    assert frozen_eval.frozen_set_dir(paths, 1234).endswith("anchor_1234")


def test_snapshot_exists_false_when_dir_missing(tmp_path):
    paths = {"experiment": str(tmp_path)}
    assert not frozen_eval.snapshot_exists(paths, 50)


def test_snapshot_exists_true_when_file_present(tmp_path):
    paths = {"experiment": str(tmp_path)}
    fdir = frozen_eval.frozen_set_dir(paths, 50)
    os.makedirs(fdir, exist_ok=True)
    with open(os.path.join(fdir, "states.pkl"), "wb") as f:
        f.write(b"nonempty")  # snapshot_exists rejects empty files
    assert frozen_eval.snapshot_exists(paths, 50)


def test_snapshot_exists_false_when_file_is_empty(tmp_path):
    """Regression: a previous crash mid-pickle could leave an empty file behind.
    snapshot_exists must reject it so ensure_snapshot re-generates."""
    paths = {"experiment": str(tmp_path)}
    fdir = frozen_eval.frozen_set_dir(paths, 50)
    os.makedirs(fdir, exist_ok=True)
    open(os.path.join(fdir, "states.pkl"), "wb").close()  # empty
    assert not frozen_eval.snapshot_exists(paths, 50)


# ---------------------------------------------------------------------------
# Variant proportional sampling
# ---------------------------------------------------------------------------

class _FakeConfig:
    def __init__(self, game, fractions=None):
        self.game = game
        self.variant_fractions = fractions or {}


def test_variant_targets_non_unified():
    cfg = _FakeConfig("connect4")
    targets = frozen_eval._variant_targets(cfg, 1024)
    assert targets == {None: 1024}


def test_variant_targets_unified_proportional():
    cfg = _FakeConfig(
        "star_gambit_unified",
        {"skirmish": 0.4, "showdown": 0.3, "clash": 0.2, "battle": 0.1},
    )
    targets = frozen_eval._variant_targets(cfg, 1000)
    # Ordering by name -> variant id: skirmish=0, showdown=1, clash=2, battle=3
    assert sum(targets.values()) == 1000
    # Largest fraction should get most positions
    assert targets[0] >= targets[1] >= targets[2] >= targets[3]
    # Roughly proportional (small rounding tolerance)
    assert abs(targets[0] - 400) <= 1
    assert abs(targets[3] - 100) <= 1


def test_variant_targets_rounds_correctly():
    cfg = _FakeConfig(
        "star_gambit_unified",
        {"skirmish": 1.0/3, "showdown": 1.0/3, "clash": 1.0/3},
    )
    targets = frozen_eval._variant_targets(cfg, 100)
    assert sum(targets.values()) == 100


# ---------------------------------------------------------------------------
# KL helper
# ---------------------------------------------------------------------------

def test_kl_zero_when_equal():
    p = np.array([0.5, 0.5])
    assert frozen_eval._kl(p, p) == pytest.approx(0.0, abs=1e-8)


def test_kl_positive_when_different():
    p = np.array([0.9, 0.1])
    q = np.array([0.1, 0.9])
    assert frozen_eval._kl(p, q) > 0


def test_kl_handles_zero_in_p():
    # KL is defined to skip terms where p == 0 (by convention 0 * log(0/q) = 0)
    p = np.array([1.0, 0.0])
    q = np.array([0.5, 0.5])
    expected = 1.0 * np.log(1.0 / 0.5)  # only first term contributes
    assert frozen_eval._kl(p, q) == pytest.approx(expected, abs=1e-5)


# ---------------------------------------------------------------------------
# Effective rank
# ---------------------------------------------------------------------------

def _participation_ratio(matrix: torch.Tensor) -> float:
    """Same formula as NNWrapper.effective_rank, for verification."""
    feats = matrix - matrix.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(feats)
    s2 = s ** 2
    denom = (s2 * s2).sum()
    if denom.item() == 0:
        return 1.0
    return float(((s2.sum() ** 2) / denom).item())


def test_participation_ratio_low_for_rank_1():
    # Rank-1 matrix: all rows are scaled copies of one direction.
    # Effective rank should be ~1.
    n, d = 32, 8
    direction = torch.randn(d)
    scales = torch.randn(n, 1)
    m = scales * direction
    pr = _participation_ratio(m)
    assert pr < 1.5


def test_participation_ratio_high_for_isotropic():
    # Random gaussian: effective rank should be close to min(n, d).
    n, d = 64, 16
    torch.manual_seed(0)
    m = torch.randn(n, d)
    pr = _participation_ratio(m)
    # Should be close to d (full-rank random matrix has roughly uniform spectrum)
    assert pr > d * 0.5
    assert pr <= d  # bounded above by feature dim


def test_participation_ratio_bounded():
    """Participation ratio is bounded in [1, min(batch, dim)]."""
    for n, d in [(8, 16), (32, 4), (64, 64)]:
        torch.manual_seed(d)
        m = torch.randn(n, d)
        pr = _participation_ratio(m)
        assert pr >= 1.0
        assert pr <= min(n, d) + 1e-6


# ---------------------------------------------------------------------------
# Aim logging helper (per-variant + variant-averaged 'mean' trace)
# ---------------------------------------------------------------------------

class _CapturingRun:
    """Captures run.track(...) calls for inspection."""
    def __init__(self):
        self.tracks = []
    def track(self, value, *, name, epoch, step, context):
        self.tracks.append({
            "value": value, "name": name, "epoch": epoch,
            "step": step, "context": dict(context),
        })


def test_log_metrics_emits_per_variant_and_mean():
    """Per-variant traces (variant context key set) + variant-averaged
    aggregate trace (no variant key — matches the convention used by
    other aggregate metrics in this repo)."""
    run = _CapturingRun()
    metrics = {
        "skirmish": {"kl_mcts_net": 0.4, "value_mae": 0.2},
        "battle":   {"kl_mcts_net": 0.6, "value_mae": 0.1},
    }
    frozen_eval.log_metrics_to_aim(run, metrics, anchor_iter=50,
                                    epoch=100, step=12345)

    # 2 variants * 2 metrics + 2 mean = 6 tracks
    assert len(run.tracks) == 6

    skirmish_tracks = [t for t in run.tracks
                       if t["context"].get("variant") == "skirmish"]
    battle_tracks = [t for t in run.tracks
                     if t["context"].get("variant") == "battle"]
    mean_tracks = [t for t in run.tracks if "variant" not in t["context"]]
    assert len(skirmish_tracks) == 2
    assert len(battle_tracks) == 2
    assert len(mean_tracks) == 2

    for t in run.tracks:
        assert t["context"]["anchor_iter"] == "0050"
        assert t["epoch"] == 100
        assert t["step"] == 12345

    mean_kl = next(t for t in mean_tracks if t["name"] == "frozen_eval/kl_mcts_net")
    assert mean_kl["value"] == pytest.approx((0.4 + 0.6) / 2)
    mean_val_mae = next(t for t in mean_tracks if t["name"] == "frozen_eval/value_mae")
    assert mean_val_mae["value"] == pytest.approx((0.2 + 0.1) / 2)


def test_log_metrics_empty_dict_no_calls():
    run = _CapturingRun()
    frozen_eval.log_metrics_to_aim(run, {}, anchor_iter=10,
                                    epoch=0, step=0)
    assert run.tracks == []


def test_log_metrics_single_variant_still_emits_mean():
    """Non-unified games (single 'default' variant): mean trace still emitted
    (no variant key), so anchors render consistently."""
    run = _CapturingRun()
    metrics = {"default": {"kl_mcts_net": 0.5}}
    frozen_eval.log_metrics_to_aim(run, metrics, anchor_iter=10,
                                    epoch=5, step=5)
    assert len(run.tracks) == 2  # per-variant + mean
    mean = next(t for t in run.tracks if "variant" not in t["context"])
    assert mean["value"] == 0.5
