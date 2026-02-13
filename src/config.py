"""Training configuration system.

Provides TrainConfig dataclass with YAML loading, CLI overrides,
game registry, experiment directory resolution, and path management.
"""

import glob
import os
import re
from dataclasses import dataclass, field, asdict

import yaml
import alphazero


GAME_REGISTRY = {
    "connect4": "Connect4GS",
    "onitama": "OnitamaGS",
    "brandubh": "BrandubhGS",
    "open_tafl": "OpenTaflGS",
    "tawlbwrdd": "TawlbwrddGS",
    "photosynthesis_2p": "PhotosynthesisGS2",
    "photosynthesis_3p": "PhotosynthesisGS3",
    "photosynthesis_4p": "PhotosynthesisGS4",
    "star_gambit_skirmish": "StarGambitSkirmishGS",
    "star_gambit_clash": "StarGambitClashGS",
    "star_gambit_battle": "StarGambitBattleGS",
}


@dataclass
class TrainConfig:
    # Game
    game: str = "connect4"

    # Network
    depth: int = 4
    channels: int = 12
    kernel_size: int = 5
    dense_net: bool = True
    star_gambit_spatial: bool = False

    # MCTS
    cpuct: float = 1.25
    fpu_reduction: float = 0.25
    selfplay_mcts_depth: int = 100
    fast_mcts_depth: int = 25
    compare_mcts_depth: int = 50

    # Temperature
    self_play_temp: float = 1.0
    eval_temp: float = 0.5
    final_temp: float = 0.2
    temp_decay_half_life: int = 10

    # Self-play
    self_play_batch_size: int = 1024
    self_play_concurrent_batch_mult: int = 2
    self_play_chunks: int = 1

    # Training
    train_batch_size: int = 1024
    train_sample_rate: int = 1
    lr: float = 0.01
    cv: float = 1.5

    # LR schedule: "constant", "step", or "adaptive"
    lr_schedule: str = "constant"
    lr_steps: list = field(default_factory=list)  # for step mode: [[0, 0.01], [250, 0.003]]
    lr_drop_factor: float = 0.3
    lr_patience: int = 5
    lr_min_iter: int = 50
    lr_min_between_drops: int = 30
    lr_max_drops: int = 3

    # Gating
    gating_panel_size: int = 1
    gating_panel_win_rate: float = 0.52
    gating_best_win_rate: float = 0.52

    # History
    hist_size: int = 30_000
    window_size_alpha: float = 0.5
    window_size_beta: float = 0.7
    window_size_scalar: float = 6.0

    # Cache
    max_cache_size: int = 200_000
    cache_shards: int = -1  # -1 = os.cpu_count()

    # Workers
    result_workers: int = 2
    data_workers: int = -1  # -1 = os.cpu_count() - 1

    # Resignation
    resign_percent: float = 0.02
    resign_playthrough_percent: float = 0.20

    # Eval game counts (total games = batch_size * NUM_PLAYERS per position)
    past_compare_batch_size: int = 64
    gate_compare_batch_size: int = 64

    # Iteration control
    iterations: int = 200
    compare_past: int = 20

    # Reservoir
    reservoir_recency_decay: float = 0.99

    # Bootstrap training (only used when architecture differs from source)
    bootstrap_full_passes: int = 5
    bootstrap_window_passes: int = 2
    bootstrap_compare_past: int = 5

    # Inference optimization (autocast + compile on GPU, skipped on CPU)
    autocast: bool = True
    half_storage: bool = True
    torch_compile: bool = True

    def validate(self):
        if self.game not in GAME_REGISTRY:
            raise ValueError(f"Unknown game: {self.game}")
        valid_lr_schedules = {"constant", "step", "adaptive"}
        if self.lr_schedule not in valid_lr_schedules:
            raise ValueError(f"Unknown lr_schedule: {self.lr_schedule}")
        if self.gating_panel_size < 1:
            raise ValueError(f"gating_panel_size must be >= 1, got {self.gating_panel_size}")
        if self.iterations < 1:
            raise ValueError(f"iterations must be >= 1, got {self.iterations}")

    @property
    def network_name(self) -> str:
        return "densenet" if self.dense_net else "resnet"

    @property
    def auto_experiment_name(self) -> str:
        return (
            f"{self.network_name}-{self.depth}d-{self.channels}c"
            f"-{self.kernel_size}k-{self.selfplay_mcts_depth}sims"
        )

    @property
    def Game(self):
        cls_name = GAME_REGISTRY[self.game]
        return getattr(alphazero, cls_name)

    @property
    def resolved_cache_shards(self) -> int:
        return os.cpu_count() if self.cache_shards == -1 else self.cache_shards

    @property
    def resolved_data_workers(self) -> int:
        return os.cpu_count() - 1 if self.data_workers == -1 else self.data_workers

    def resolve_experiment_dir(self, base="data", explicit_name=None) -> str:
        """Resolve experiment directory path.

        - If explicit_name given, use it directly: base/{game}/{explicit_name}/
        - Otherwise auto-derive from config and add integer suffix if collision:
          base/{game}/densenet-4d-16c-3k-300sims/
          base/{game}/densenet-4d-16c-3k-300sims-01/  (if first exists)
        """
        game_dir = os.path.join(base, self.game)

        if explicit_name:
            return os.path.join(game_dir, explicit_name)

        name = self.auto_experiment_name
        path = os.path.join(game_dir, name)

        if not os.path.exists(path):
            return path

        # Auto-suffix
        for i in range(1, 100):
            suffixed = os.path.join(game_dir, f"{name}-{i:02d}")
            if not os.path.exists(suffixed):
                return suffixed

        raise RuntimeError(f"Too many experiment directories for {name}")

    def resolve_paths(self, experiment_dir: str) -> dict:
        return {
            "experiment": experiment_dir,
            "checkpoint": os.path.join(experiment_dir, "checkpoint"),
            "history": os.path.join(experiment_dir, "history"),
            "tmp_history": os.path.join(experiment_dir, "tmp_history"),
            "reservoir": os.path.join(experiment_dir, "reservoir"),
        }

    def save(self, path: str):
        """Save config as YAML."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)


def find_latest_checkpoint(checkpoint_dir):
    """Scan checkpoint dir for highest iteration number, return it (0 if none)."""
    pattern = os.path.join(checkpoint_dir, "*.pt")
    files = glob.glob(pattern)
    if not files:
        return 0
    best = 0
    for f in files:
        basename = os.path.basename(f)
        match = re.match(r"(\d+)-", basename)
        if match:
            best = max(best, int(match.group(1)))
    return best


def load_config(yaml_path: str, cli_overrides: dict, warn=True) -> TrainConfig:
    """Load config from YAML file with CLI overrides.

    yaml_path: path to YAML config file
    cli_overrides: dict of {key: string_value} from CLI --key val pairs
    """
    config = TrainConfig()

    if yaml_path:
        with open(yaml_path) as f:
            yaml_data = yaml.safe_load(f) or {}
        for key, val in yaml_data.items():
            if not hasattr(config, key):
                if warn:
                    print(f"Warning: ignoring unknown config key in YAML: {key}")
                continue
            setattr(config, key, val)

    for key, val in cli_overrides.items():
        if not hasattr(config, key):
            raise ValueError(f"Unknown config key: {key}")
        field_type = type(getattr(config, key))
        if field_type is list:
            if warn:
                print(f"Warning: cannot override list field '{key}' via CLI, use YAML instead")
            continue
        elif field_type is bool:
            setattr(config, key, str(val).lower() in ("true", "1", "yes"))
        else:
            setattr(config, key, field_type(val))

    config.validate()
    return config
