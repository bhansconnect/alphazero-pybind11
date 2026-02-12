"""Unit tests for config.py -- Config loading, YAML, CLI overrides, paths, registry."""

import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import alphazero
from config import TrainConfig, GAME_REGISTRY, load_config


# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------


def test_default_config():
    """TrainConfig() has all expected defaults."""
    config = TrainConfig()
    assert config.game == "connect4"
    assert config.depth == 4
    assert config.channels == 12
    assert config.kernel_size == 5
    assert config.dense_net is True
    assert config.star_gambit_spatial is False
    assert config.cpuct == 1.25
    assert config.fpu_reduction == 0.25
    assert config.selfplay_mcts_depth == 100
    assert config.fast_mcts_depth == 25
    assert config.compare_mcts_depth == 50
    assert config.self_play_batch_size == 256
    assert config.train_batch_size == 1024
    assert config.iterations == 200
    assert config.start == 0
    assert config.bootstrap_from == ""


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------


CONFIGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs")


def test_yaml_loading_connect4():
    """Loading connect4.yaml sets game to connect4, preserves other defaults."""
    config = load_config(os.path.join(CONFIGS_DIR, "connect4.yaml"), {})
    assert config.game == "connect4"
    assert config.depth == 4  # default preserved
    assert config.channels == 12  # default preserved


def test_yaml_loading_star_gambit():
    """Star Gambit YAML overrides specified fields, preserves others."""
    config = load_config(
        os.path.join(CONFIGS_DIR, "star_gambit_skirmish.yaml"), {}
    )
    assert config.game == "star_gambit_skirmish"
    assert config.depth == 4
    assert config.channels == 16  # overridden
    assert config.kernel_size == 3  # overridden
    assert config.star_gambit_spatial is True  # overridden
    assert config.selfplay_mcts_depth == 300  # overridden
    assert config.gating_panel_size == 1  # default preserved


# ---------------------------------------------------------------------------
# CLI overrides
# ---------------------------------------------------------------------------


def test_cli_overrides():
    """CLI --key val overrides both defaults and YAML."""
    config = load_config(
        os.path.join(CONFIGS_DIR, "connect4.yaml"),
        {"depth": "8", "cpuct": "2.0"},
    )
    assert config.depth == 8
    assert config.cpuct == 2.0


def test_bool_cli_override_false():
    """Boolean CLI overrides handle false."""
    config = load_config(
        os.path.join(CONFIGS_DIR, "connect4.yaml"), {"dense_net": "false"}
    )
    assert config.dense_net is False


def test_bool_cli_override_true():
    """Boolean CLI overrides handle true."""
    config = load_config(
        os.path.join(CONFIGS_DIR, "connect4.yaml"), {"dense_net": "true"}
    )
    assert config.dense_net is True


# ---------------------------------------------------------------------------
# Unknown keys
# ---------------------------------------------------------------------------


def test_unknown_yaml_key_raises():
    """Unknown key in YAML raises ValueError."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("game: connect4\nbogus_key: 42\n")
        f.flush()
        try:
            with pytest.raises(ValueError, match="Unknown config key in YAML"):
                load_config(f.name, {})
        finally:
            os.unlink(f.name)


def test_unknown_cli_key_raises():
    """Unknown --key in CLI raises ValueError."""
    with pytest.raises(ValueError, match="Unknown config key"):
        load_config(os.path.join(CONFIGS_DIR, "connect4.yaml"), {"nonexistent": "5"})


# ---------------------------------------------------------------------------
# Game registry
# ---------------------------------------------------------------------------


def test_game_registry_classes_exist():
    """All registry entries resolve to valid alphazero classes."""
    for name, cls_name in GAME_REGISTRY.items():
        assert hasattr(alphazero, cls_name), f"Missing class: {cls_name}"


def test_game_property():
    """Config.Game resolves to the correct alphazero class."""
    config = TrainConfig(game="connect4")
    assert config.Game == alphazero.Connect4GS

    config2 = TrainConfig(game="star_gambit_skirmish")
    assert config2.Game == alphazero.StarGambitSkirmishGS


# ---------------------------------------------------------------------------
# Derived properties
# ---------------------------------------------------------------------------


def test_network_name():
    """network_name returns densenet or resnet."""
    config = TrainConfig(dense_net=True)
    assert config.network_name == "densenet"
    config.dense_net = False
    assert config.network_name == "resnet"


def test_auto_experiment_name():
    """auto_experiment_name formats correctly."""
    config = TrainConfig(
        dense_net=True, depth=4, channels=12, kernel_size=5, selfplay_mcts_depth=100
    )
    assert config.auto_experiment_name == "densenet-4d-12c-5k-100sims"

    config.dense_net = False
    assert config.auto_experiment_name == "resnet-4d-12c-5k-100sims"


def test_resolved_cache_shards():
    """resolved_cache_shards defaults to cpu_count."""
    config = TrainConfig(cache_shards=-1)
    assert config.resolved_cache_shards == os.cpu_count()

    config.cache_shards = 8
    assert config.resolved_cache_shards == 8


# ---------------------------------------------------------------------------
# Experiment directory resolution
# ---------------------------------------------------------------------------


def test_experiment_dir_new(tmp_path):
    """New experiment creates directory based on auto name."""
    config = TrainConfig()
    result = config.resolve_experiment_dir(base=str(tmp_path))
    expected = os.path.join(str(tmp_path), "connect4", "densenet-4d-12c-5k-100sims")
    assert result == expected


def test_experiment_dir_auto_suffix(tmp_path):
    """Auto-suffix -01 when experiment dir already exists."""
    base = tmp_path / "connect4" / "densenet-4d-12c-5k-100sims"
    base.mkdir(parents=True)
    config = TrainConfig()
    result = config.resolve_experiment_dir(base=str(tmp_path))
    assert result.endswith("-01")


def test_experiment_dir_auto_suffix_02(tmp_path):
    """Auto-suffix -02 when -01 also exists."""
    base_dir = tmp_path / "connect4"
    (base_dir / "densenet-4d-12c-5k-100sims").mkdir(parents=True)
    (base_dir / "densenet-4d-12c-5k-100sims-01").mkdir(parents=True)
    config = TrainConfig()
    result = config.resolve_experiment_dir(base=str(tmp_path))
    assert result.endswith("-02")


def test_experiment_dir_explicit_name(tmp_path):
    """Explicit experiment name used as-is."""
    config = TrainConfig()
    result = config.resolve_experiment_dir(
        base=str(tmp_path), explicit_name="my-run"
    )
    assert result.endswith("my-run")
    assert "connect4" in result


def test_experiment_dir_resume_exact(tmp_path):
    """Resume (start>0) returns exact path, no auto-suffix."""
    config = TrainConfig(start=50)
    result = config.resolve_experiment_dir(base=str(tmp_path))
    assert result.endswith("densenet-4d-12c-5k-100sims")


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def test_resolve_paths():
    """resolve_paths returns correct subdirectory structure."""
    config = TrainConfig()
    paths = config.resolve_paths("/data/connect4/exp1")
    assert paths["checkpoint"] == "/data/connect4/exp1/checkpoint"
    assert paths["history"] == "/data/connect4/exp1/history"
    assert paths["tmp_history"] == "/data/connect4/exp1/tmp_history"
    assert paths["reservoir"] == "/data/connect4/exp1/reservoir"
    assert paths["experiment"] == "/data/connect4/exp1"


# ---------------------------------------------------------------------------
# Save and reload
# ---------------------------------------------------------------------------


def test_config_save_and_reload(tmp_path):
    """Save resolved config as YAML, reload, verify identical."""
    config = load_config(
        os.path.join(CONFIGS_DIR, "star_gambit_skirmish.yaml"),
        {"iterations": "50"},
    )
    save_path = str(tmp_path / "config.yaml")
    config.save(save_path)

    reloaded = load_config(save_path, {})
    assert reloaded.game == config.game
    assert reloaded.iterations == 50
    assert reloaded.depth == config.depth
    assert reloaded.channels == config.channels
    assert reloaded.star_gambit_spatial == config.star_gambit_spatial
