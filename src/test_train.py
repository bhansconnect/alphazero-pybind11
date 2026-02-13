"""Integration tests for the training pipeline via the config system.

These tests exercise the full self-play -> train -> gate pipeline.
They are slower (~1-5 min) and require inference capability.

All data is written to pytest's tmp_path (auto-cleaned) to avoid
polluting the project's data/ or .aim/ directories.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import load_config

CONFIGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs")

# Minimal overrides for fast testing
FAST_OVERRIDES = {
    "selfplay_mcts_depth": "8",
    "fast_mcts_depth": "4",
    "compare_mcts_depth": "4",
    "self_play_batch_size": "4",
    "self_play_chunks": "1",
    "self_play_concurrent_batch_mult": "1",
    "past_compare_batch_size": "4",
    "gate_compare_batch_size": "4",
    "max_cache_size": "0",
    "hist_size": "1000",
}


def test_e2e_training_connect4(tmp_path):
    """Full training loop for Connect4: create net -> self-play -> train -> gate."""
    import game_runner

    config = load_config(
        os.path.join(CONFIGS_DIR, "connect4.yaml"),
        {**FAST_OVERRIDES, "iterations": "2"},
    )
    experiment_dir = str(tmp_path / "data" / "connect4" / "test")
    aim_repo = str(tmp_path / "aim")
    game_runner.main(config, experiment_dir, aim_repo=aim_repo)

    # Verify outputs
    checkpoint_dir = tmp_path / "data" / "connect4" / "test" / "checkpoint"
    assert checkpoint_dir.exists()
    pt_files = list(checkpoint_dir.glob("*.pt"))
    assert len(pt_files) >= 2, f"Expected at least 2 checkpoints, got {len(pt_files)}"


def test_e2e_training_star_gambit(tmp_path):
    """Full training loop for Star Gambit Skirmish."""
    import game_runner

    config = load_config(
        os.path.join(CONFIGS_DIR, "star_gambit_skirmish.yaml"),
        {**FAST_OVERRIDES, "iterations": "2"},
    )
    experiment_dir = str(tmp_path / "data" / "star_gambit_skirmish" / "test")
    aim_repo = str(tmp_path / "aim")
    game_runner.main(config, experiment_dir, aim_repo=aim_repo)

    checkpoint_dir = (
        tmp_path / "data" / "star_gambit_skirmish" / "test" / "checkpoint"
    )
    assert checkpoint_dir.exists()
    pt_files = list(checkpoint_dir.glob("*.pt"))
    assert len(pt_files) >= 2


def test_config_saved_to_experiment_dir(tmp_path):
    """Verify config.yaml saved at start of training."""
    import game_runner

    config = load_config(
        os.path.join(CONFIGS_DIR, "connect4.yaml"),
        {**FAST_OVERRIDES, "iterations": "1"},
    )
    experiment_dir = str(tmp_path / "data" / "connect4" / "test")
    aim_repo = str(tmp_path / "aim")
    game_runner.main(config, experiment_dir, aim_repo=aim_repo)

    config_path = tmp_path / "data" / "connect4" / "test" / "config.yaml"
    assert config_path.exists()
    reloaded = load_config(str(config_path), {})
    assert reloaded.game == "connect4"


def test_bootstrap_same_arch(tmp_path):
    """Bootstrap from existing experiment with same architecture.

    Full E2E: source training -> bootstrap -> verify artifacts -> continue training.
    """
    import game_runner

    aim_repo = str(tmp_path / "aim")

    # --- Phase 1: Create source experiment (2 iterations) ---
    source_config = load_config(
        os.path.join(CONFIGS_DIR, "connect4.yaml"),
        {**FAST_OVERRIDES, "iterations": "2"},
    )
    source_dir = str(tmp_path / "data" / "connect4" / "source")
    game_runner.main(source_config, source_dir, aim_repo=aim_repo)

    # Verify source produced expected artifacts
    source_ckpt = tmp_path / "data" / "connect4" / "source" / "checkpoint"
    assert source_ckpt.exists()
    source_pts = sorted(source_ckpt.glob("*.pt"))
    assert len(source_pts) >= 2, f"Source needs >= 2 checkpoints, got {len(source_pts)}"
    source_elo = tmp_path / "data" / "connect4" / "source" / "elo.csv"
    assert source_elo.exists(), "Source must have elo.csv"

    # --- Phase 2: Bootstrap new experiment from source ---
    bootstrap_config = load_config(
        os.path.join(CONFIGS_DIR, "connect4.yaml"),
        {
            **FAST_OVERRIDES,
            "iterations": "4",  # Will run iterations 2 & 3 after bootstrap at iter 2
        },
    )
    new_dir = str(tmp_path / "data" / "connect4" / "bootstrapped")
    game_runner.main(bootstrap_config, new_dir, aim_repo=aim_repo, bootstrap_from=source_dir)

    # --- Phase 3: Verify bootstrap artifacts ---
    new_ckpt = tmp_path / "data" / "connect4" / "bootstrapped" / "checkpoint"
    assert new_ckpt.exists()
    new_pts = list(new_ckpt.glob("*.pt"))
    # Should have: initial + copied source + newly trained ones
    assert len(new_pts) >= 3, f"Expected >= 3 checkpoints, got {len(new_pts)}"

    new_config_path = tmp_path / "data" / "connect4" / "bootstrapped" / "config.yaml"
    assert new_config_path.exists()

    # elo.csv should exist with data from source + new iterations
    new_elo = tmp_path / "data" / "connect4" / "bootstrapped" / "elo.csv"
    assert new_elo.exists()
