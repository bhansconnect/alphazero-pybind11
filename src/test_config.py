"""Unit tests for config.py -- Config loading, YAML, CLI overrides, paths, registry."""

import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import alphazero
from config import TrainConfig, GAME_REGISTRY, load_config, find_latest_checkpoint


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
    assert config.selfplay_mcts_visits == 100
    assert config.fast_mcts_visits == 25
    assert config.compare_mcts_visits == 50
    assert config.self_play_batch_size == 1024
    assert config.train_batch_size == 1024
    assert config.iterations == 200
    assert config.lr_schedule == "constant"
    assert config.lr_steps == []
    assert config.lr_drop_factor == 0.3
    assert config.lr_patience == 5
    assert config.lr_min_iter == 50
    assert config.lr_min_between_drops == 30
    assert config.lr_max_drops == 3
    assert config.bootstrap_compare_past == 5


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
    assert config.selfplay_mcts_visits == 120  # overridden
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


def test_unknown_yaml_key_warns(capsys):
    """Unknown key in YAML prints warning and is skipped."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("game: connect4\nbogus_key: 42\n")
        f.flush()
        try:
            config = load_config(f.name, {})
            assert config.game == "connect4"
            captured = capsys.readouterr()
            assert "ignoring unknown config key in YAML: bogus_key" in captured.out
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
        dense_net=True, depth=4, channels=12, kernel_size=5, selfplay_mcts_visits=100
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


# ---------------------------------------------------------------------------
# LR schedule config
# ---------------------------------------------------------------------------


def test_lr_schedule_step_from_yaml(tmp_path):
    """Step LR schedule loads lr_steps from YAML."""
    yaml_content = (
        "game: connect4\n"
        "lr_schedule: step\n"
        "lr_steps:\n"
        "  - [0, 0.01]\n"
        "  - [250, 0.003]\n"
        "  - [750, 0.001]\n"
    )
    yaml_path = str(tmp_path / "step.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    config = load_config(yaml_path, {})
    assert config.lr_schedule == "step"
    assert config.lr_steps == [[0, 0.01], [250, 0.003], [750, 0.001]]


def test_lr_schedule_adaptive_from_yaml(tmp_path):
    """Adaptive LR schedule fields load from YAML."""
    yaml_content = (
        "game: connect4\n"
        "lr: 0.01\n"
        "lr_schedule: adaptive\n"
        "lr_drop_factor: 0.5\n"
        "lr_patience: 10\n"
        "lr_min_iter: 100\n"
        "lr_min_between_drops: 50\n"
        "lr_max_drops: 2\n"
    )
    yaml_path = str(tmp_path / "adaptive.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    config = load_config(yaml_path, {})
    assert config.lr_schedule == "adaptive"
    assert config.lr_drop_factor == 0.5
    assert config.lr_patience == 10
    assert config.lr_min_iter == 100
    assert config.lr_min_between_drops == 50
    assert config.lr_max_drops == 2


def test_lr_schedule_save_reload(tmp_path):
    """LR schedule fields survive save/reload cycle."""
    config = TrainConfig(
        lr_schedule="step",
        lr_steps=[[0, 0.01], [100, 0.003]],
    )
    save_path = str(tmp_path / "config.yaml")
    config.save(save_path)

    reloaded = load_config(save_path, {})
    assert reloaded.lr_schedule == "step"
    assert reloaded.lr_steps == [[0, 0.01], [100, 0.003]]


# ---------------------------------------------------------------------------
# Arg parsing (train.py parse_args)
# ---------------------------------------------------------------------------


from unittest.mock import patch
from train import parse_args


def _parse(argv):
    """Helper: call parse_args with given argv (excluding 'train.py')."""
    with patch("sys.argv", ["train.py"] + argv):
        return parse_args()


def test_parse_config_positional():
    """Positional config path is extracted."""
    args, overrides = _parse(["configs/connect4.yaml"])
    assert args.config == "configs/connect4.yaml"
    assert args.resume is None
    assert args.bootstrap is None
    assert overrides == {}


def test_parse_config_with_overrides():
    """Config + overrides parse correctly."""
    args, overrides = _parse(["configs/connect4.yaml", "--depth", "8", "--cpuct", "2.0"])
    assert args.config == "configs/connect4.yaml"
    assert overrides == {"depth": "8", "cpuct": "2.0"}


def test_parse_resume():
    """--resume is parsed, no config."""
    args, overrides = _parse(["--resume", "data/connect4/exp1"])
    assert args.config is None
    assert args.resume == "data/connect4/exp1"
    assert overrides == {}


def test_parse_resume_with_overrides():
    """--resume + overrides parse correctly."""
    args, overrides = _parse(["--resume", "data/connect4/exp1", "--iterations", "400"])
    assert args.resume == "data/connect4/exp1"
    assert args.config is None
    assert overrides == {"iterations": "400"}


def test_parse_bootstrap():
    """--bootstrap is parsed, no config."""
    args, overrides = _parse(["--bootstrap", "data/connect4/exp1"])
    assert args.config is None
    assert args.bootstrap == "data/connect4/exp1"
    assert overrides == {}


def test_parse_bootstrap_with_overrides():
    """--bootstrap + overrides don't consume values as config."""
    args, overrides = _parse([
        "--bootstrap", "data/connect4/exp1",
        "--selfplay_mcts_visits", "30",
        "--fast-mcts-visits", "1",
    ])
    assert args.config is None
    assert args.bootstrap == "data/connect4/exp1"
    assert overrides == {"selfplay_mcts_visits": "30", "fast_mcts_visits": "1"}


def test_parse_hyphen_underscore_normalization():
    """Override keys normalize hyphens to underscores."""
    args, overrides = _parse([
        "configs/connect4.yaml",
        "--self-play-batch-size", "128",
        "--temp-decay-half-life", "20",
    ])
    assert overrides == {"self_play_batch_size": "128", "temp_decay_half_life": "20"}


def test_parse_bool_flag_no_value():
    """Boolean flag without value defaults to 'true'."""
    args, overrides = _parse(["configs/connect4.yaml", "--dense-net"])
    assert overrides == {"dense_net": "true"}


def test_parse_bootstrap_with_experiment():
    """--bootstrap + --experiment are both parsed."""
    args, overrides = _parse([
        "--bootstrap", "data/connect4/exp1",
        "--experiment", "my-new-run",
        "--channels", "32",
    ])
    assert args.bootstrap == "data/connect4/exp1"
    assert args.experiment == "my-new-run"
    assert args.config is None
    assert overrides == {"channels": "32"}


# ---------------------------------------------------------------------------
# find_latest_checkpoint
# ---------------------------------------------------------------------------


def test_find_latest_checkpoint(tmp_path):
    """find_latest_checkpoint returns highest iteration number from checkpoint files."""
    cp_dir = tmp_path / "checkpoint"
    cp_dir.mkdir()
    (cp_dir / "0005-net.pt").touch()
    (cp_dir / "0010-net.pt").touch()
    (cp_dir / "0003-net.pt").touch()
    assert find_latest_checkpoint(str(cp_dir)) == 10


def test_find_latest_checkpoint_empty(tmp_path):
    """find_latest_checkpoint returns 0 for empty directory."""
    cp_dir = tmp_path / "checkpoint"
    cp_dir.mkdir()
    assert find_latest_checkpoint(str(cp_dir)) == 0


# ---------------------------------------------------------------------------
# load_config warn parameter
# ---------------------------------------------------------------------------


def test_load_config_warn_false_suppresses_warnings(tmp_path, capsys):
    """warn=False suppresses unknown key warnings."""
    yaml_path = str(tmp_path / "stale.yaml")
    with open(yaml_path, "w") as f:
        f.write("game: connect4\nlr_milestone: 100\nbootstrap_from: foo\n")
    load_config(yaml_path, {}, warn=False)
    captured = capsys.readouterr()
    assert captured.out == ""


def test_stale_keys_do_not_corrupt_config(tmp_path):
    """Stale YAML keys are not set as attributes on the config object."""
    yaml_path = str(tmp_path / "stale.yaml")
    with open(yaml_path, "w") as f:
        f.write("game: connect4\nlr_milestone: 100\nbootstrap_from: foo\n")
    config = load_config(yaml_path, {}, warn=False)
    assert config.game == "connect4"
    assert not hasattr(config, "lr_milestone")
    assert not hasattr(config, "bootstrap_from")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_validate_invalid_lr_schedule(tmp_path):
    """Invalid lr_schedule raises ValueError."""
    yaml_path = str(tmp_path / "bad_lr.yaml")
    with open(yaml_path, "w") as f:
        f.write("game: connect4\nlr_schedule: cosine\n")
    with pytest.raises(ValueError, match="Unknown lr_schedule"):
        load_config(yaml_path, {})


def test_validate_gating_panel_size_zero(tmp_path):
    """gating_panel_size: 0 raises ValueError."""
    yaml_path = str(tmp_path / "bad_panel.yaml")
    with open(yaml_path, "w") as f:
        f.write("game: connect4\ngating_panel_size: 0\n")
    with pytest.raises(ValueError, match="gating_panel_size must be >= 1"):
        load_config(yaml_path, {})


def test_validate_iterations_zero(tmp_path):
    """iterations: 0 raises ValueError."""
    yaml_path = str(tmp_path / "bad_iter.yaml")
    with open(yaml_path, "w") as f:
        f.write("game: connect4\niterations: 0\n")
    with pytest.raises(ValueError, match="iterations must be >= 1"):
        load_config(yaml_path, {})


def test_validate_unknown_game(tmp_path):
    """Unknown game raises ValueError."""
    yaml_path = str(tmp_path / "bad_game.yaml")
    with open(yaml_path, "w") as f:
        f.write("game: nonexistent_game\n")
    with pytest.raises(ValueError, match="Unknown game"):
        load_config(yaml_path, {})


def test_list_cli_override_warns(tmp_path, capsys):
    """CLI override of list field prints warning."""
    yaml_path = str(tmp_path / "config.yaml")
    with open(yaml_path, "w") as f:
        f.write("game: connect4\n")
    load_config(yaml_path, {"lr_steps": "[[0, 0.01]]"})
    captured = capsys.readouterr()
    assert "cannot override list field 'lr_steps' via CLI" in captured.out
