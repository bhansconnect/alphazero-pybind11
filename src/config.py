"""Training configuration system.

Provides TrainConfig dataclass with YAML loading, CLI overrides,
game registry, experiment directory resolution, and path management.
"""

import glob
import os
import re
from dataclasses import dataclass, field, asdict, fields as dc_fields

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
    "star_gambit_showdown": "StarGambitShowdownGS",
    "star_gambit_clash": "StarGambitClashGS",
    "star_gambit_battle": "StarGambitBattleGS",
    "star_gambit_unified": "StarGambitUnifiedGS",
    "star_gambit_unified_skirmish": "StarGambitUnifiedSkirmishGS",
    "star_gambit_unified_showdown": "StarGambitUnifiedShowdownGS",
    "star_gambit_unified_clash": "StarGambitUnifiedClashGS",
    "star_gambit_unified_battle": "StarGambitUnifiedBattleGS",
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
    # Policy head selection: "auto" (default) uses the spatial conv head for any
    # game that exposes a POLICY_SHAPE (tafl, star gambit) and the flat FC head
    # otherwise; "on" forces spatial (errors if unsupported); "off" forces flat.
    spatial_policy: str = "auto"
    # Deprecated alias for spatial_policy, kept so existing YAML configs still
    # load. Normalized into spatial_policy by validate() (True -> "on").
    star_gambit_spatial: bool = False
    head_channels: int = 32
    head_pool: bool = True
    v_head_convs: int = 0
    pi_head_convs: int = 0
    v_fc_layers: int = 1
    pi_fc_layers: int = 0
    # "batch" (default, current behavior) or "layer" (GroupNorm(1,C) — per-sample
    # LayerNorm across (C,H,W); avoids the batch-statistic propagation that
    # contributes to feature collapse in bootstrapped RL targets).
    trunk_norm: str = "batch"
    # AlphaZero standard. KataGo uses 3e-5 (even lower) and the AlphaZero
    # paper specifies 1e-4. star_gambit train-fixes empirically validated
    # the move from 1e-3 -> 1e-4. 1e-3 was the old codebase default and
    # produced visibly worse plasticity / effective rank.
    weight_decay: float = 1e-4
    # "relu" (default) or "crelu" (Concatenated ReLU, preserves negative-direction
    # features and mitigates dying-ReLU plasticity loss; doubles intermediate
    # channel count in blocks).
    trunk_act: str = "relu"
    # Orthogonal regularization strength on trunk Conv2d weights. 0 disables.
    # Penalizes filter correlation directly; promotes high effective rank.
    orth_reg_lambda: float = 0.0

    # MCTS
    cpuct: float = 1.25
    fpu_reduction: float = 0.25
    mcts_root_temp: float = 1.25
    root_fpu_zero: bool = True
    shaped_dirichlet: bool = True
    policy_target_pruning: bool = True
    selfplay_mcts_visits: int = 100
    fast_mcts_visits: int = 25
    compare_mcts_visits: int = 50
    # Playout cap randomization (KataGo). Fraction of self-play moves that
    # use fast_mcts_visits instead of selfplay_mcts_visits. Capped moves
    # don't enter training data. Set 0 to disable (every move is full search
    # and contributes data).
    playout_cap_percent: float = 0.75
    # Gumbel AlphaZero (Danihelka 2022). Replaces PUCT root + Dirichlet noise
    # with Gumbel-Top-K + Sequential Halving and trains on the improved-policy
    # target. Off by default. Capped searches always fall back to PUCT.
    gumbel_enabled: bool = False
    # Max considered actions at root. Auto-capped at runtime to
    # min(num_legal_actions, num_sims_target). Paper uses 16 universally
    # across Go/chess/Atari; ablation showed m in {8,16,32} equivalent.
    gumbel_m: int = 16
    gumbel_c_visit: float = 50.0  # sigma formula constant (paper default)
    gumbel_c_scale: float = 1.0   # sigma formula constant (paper default)
    gumbel_full: bool = False     # also use pi'-matching at non-root nodes
    # What algorithm to use for capped/fast self-play searches:
    #   "auto"   - same as the main search algo (Gumbel if gumbel_enabled, else PUCT)
    #   "puct"   - force PUCT for capped (even when main is Gumbel)
    #   "gumbel" - force Gumbel for capped (Gumbel-everywhere)
    # Connect4 retest showed PUCT-for-capped trains a stronger network than
    # Gumbel-for-capped at low capped visit counts (capped Gumbel introduces
    # ~40% top-1 variance in trajectories, polluting training data).
    fast_search_algo: str = "auto"

    def resolve_fast_search_uses_gumbel(self) -> bool:
        """Return True iff capped self-play searches should run Gumbel."""
        if self.fast_search_algo == "auto":
            return self.gumbel_enabled
        if self.fast_search_algo == "puct":
            return False
        if self.fast_search_algo == "gumbel":
            return True
        raise ValueError(
            f"fast_search_algo must be 'auto'/'puct'/'gumbel', got {self.fast_search_algo!r}"
        )

    # Temperature
    self_play_temp: float = 1.0
    eval_temp: float = 0.5
    final_temp: float = 0.2
    # int = uniform half-life. dict = per-variant; must list every variant
    # named in UNIFIED_VARIANT_NAMES (skirmish, showdown, clash, battle).
    temp_decay_half_life: object = 10

    # Self-play
    self_play_batch_size: int = 1024
    self_play_concurrent_batch_mult: int = 2
    self_play_chunks: int = 1

    # Training
    train_batch_size: int = 1024
    # KataGo's max-train-bucket-per-new-data. Each fresh sample gets 4
    # training passes through the network. star_gambit train-fixes
    # validated this empirically; default was 1 previously.
    train_sample_rate: int = 4
    lr: float = 0.01
    cv: float = 1.5

    # LR schedule: "constant", "step", or "adaptive"
    lr_schedule: str = "constant"
    lr_steps: list = field(default_factory=list)  # for step mode: [[0, 0.01], [250, 0.003]]
    lr_drop_factor: float = 0.3
    lr_patience: int = 8
    lr_min_iter: int = 50
    lr_min_between_drops: int = 30
    lr_max_drops: int = 3

    # Gating
    gating_panel_size: int = 1
    gating_panel_win_rate: float = 0.52
    gating_best_win_rate: float = 0.52

    # History
    hist_size: int = 30_000
    # Window-growth formula: Nwindow = scalar * (1 + beta * (ratio^alpha - 1) / alpha)
    # window_size_unit determines what "scalar" means:
    #   "iterations" (legacy default): scalar in iters; ratio = (iter+1)/scalar
    #   "games":      scalar in games; ratio = total_games/scalar_games.
    #                 total_games is computed exactly from the self-play
    #                 product self_play_batch_size * NUM_PLAYERS *
    #                 self_play_concurrent_batch_mult * self_play_chunks
    #                 (constant per iter, so no history walking).
    # KataGo calibration: their reported numbers are 4.2M games -> 241M
    # samples (~57 samples/game after playout-cap-randomization keeps only
    # ~25% of moves). Their sample-space c=250_000 -> ~4400 games. Their
    # alpha=0.75, beta=0.4. To roughly match KataGo's growth shape on a
    # games basis, set scalar_games=4400, alpha=0.75, beta=0.4.
    # Games-based is preferable when different variants/games have very
    # different lengths, since each game counts equally regardless of how
    # many positions it contributes.
    window_size_unit: str = "iterations"
    window_size_alpha: float = 0.5
    window_size_beta: float = 0.7
    window_size_scalar: float = 6.0
    window_size_scalar_games: int = 4400

    # Cache
    max_cache_size: int = 200_000
    cache_shards: int = -1  # -1 = os.cpu_count()

    # Workers
    result_workers: int = 2
    data_workers: int = -1  # -1 = os.cpu_count() - 1

    # Training I/O
    streaming_active_files: int = 8

    # Resignation
    resign_percent: float = 0.02
    resign_playthrough_percent: float = 0.20

    # Eval game counts (total games = batch_size * NUM_PLAYERS per position)
    past_compare_batch_size: int = 64
    gate_compare_batch_size: int = 64

    # Iteration control
    iterations: int = 200
    # List of historical comparisons to run each iter for elo.
    # Negative entries = relative offset (-20 = "20 iters back").
    # Positive entries = absolute anchor iters (e.g. 50 = "always vs iter 50").
    # Anchors auto-retire (skip future iters) once their win rate saturates
    # past compare_past_saturation_threshold.
    compare_past: list = field(default_factory=lambda: [-20])
    compare_past_saturation_threshold: float = 0.97

    # Reservoir
    reservoir_recency_decay: float = 0.995
    reservoir_n_chunks: int = 100
    reservoir_chunk_size: int = 100_000
    reservoir_chunks_per_update: int = 10
    reservoir_update_interval: int = 10

    # Bootstrap
    bootstrap_window_only: bool = False
    bootstrap_compare_past: int = 5
    bootstrap_epochs: int = 1
    bootstrap_eval_interval: int = 500
    bootstrap_lr: float = 0.01
    bootstrap_lr_drop_factor: float = 0.3
    bootstrap_lr_patience: int = 3
    bootstrap_lr_cooldown: int = 1
    bootstrap_lr_max_drops: int = 3
    bootstrap_convergence_threshold: float = 0.002
    bootstrap_convergence_patience: int = 3

    # Inference optimization (AMP autocast + compile on GPU, skipped on CPU)
    amp_inference: bool = True
    half_storage: bool = True  # only affects history files
    torch_compile: bool = True

    # EMA weight averaging
    ema_averaging: bool = True

    # LR warmup
    lr_warmup_target: int = 15       # iterations to reach full LR
    lr_warmup_floor: float = 0.2     # minimum LR fraction during warmup

    # Compression
    zstd_level: int = 1

    # Parallel file loading
    loader_threads: int = -1  # -1 = os.cpu_count()

    # Multi-variant Star Gambit (only used when game = star_gambit_unified)
    # Fraction of games (or samples for sample_based) per variant; must sum to 1.0.
    # e.g. {"skirmish": 0.25, "showdown": 0.25, "clash": 0.25, "battle": 0.25}
    variant_fractions: dict = field(default_factory=dict)
    # "game_based": use fractions directly as game probabilities.
    # "sample_based": adjust game probs each iteration to hit target sample fractions.
    variant_mixing_mode: str = "game_based"
    # Variant weights for gating evaluation (defaults to variant_fractions if empty).
    gating_variant_weights: dict = field(default_factory=dict)
    # Interval for dedicated per-variant evaluation (0 = disabled).
    variant_eval_interval: int = 0

    # Frozen eval set (Feature 1, see plan i-want-the-vs-refactored-hippo.md)
    # Empty list = disabled. Multiple anchors give complementary windows
    # (e.g., [5, 30, 100, 300] for early/mid/late training signal).
    frozen_eval_anchor_iters: list = field(default_factory=list)
    frozen_eval_positions: int = 1024
    frozen_eval_interval: int = 1
    # Minimum complete games to play during snapshot. Guards against the
    # long-game edge case where pool fills in few games and one game dominates
    # the sample. Default is sane for any game; raise for more source diversity.
    frozen_eval_min_games: int = 20

    # Self-match at multiplied visits (Feature 2)
    # 0 = disabled. 2, 4, 8, ... = play current net at selfplay_mcts_visits vs
    # current net at selfplay_mcts_visits * multiplier.
    selfmatch_visit_multiplier: int = 0
    selfmatch_games: int = 100
    selfmatch_interval: int = 1

    # Effective rank (Feature 3) — participation ratio of penultimate-layer
    # activations. Cheap; default ON.
    effective_rank_enabled: bool = True
    effective_rank_batch_size: int = 512

    def validate(self):
        if self.game not in GAME_REGISTRY:
            raise ValueError(f"Unknown game: {self.game}")
        # Fold the deprecated star_gambit_spatial bool into spatial_policy when
        # the new field was left at its default.
        if self.star_gambit_spatial and self.spatial_policy == "auto":
            self.spatial_policy = "on"
        if self.spatial_policy not in ("auto", "on", "off"):
            raise ValueError(
                f"spatial_policy must be 'auto'/'on'/'off', got {self.spatial_policy!r}"
            )
        valid_lr_schedules = {"constant", "step", "adaptive"}
        if self.lr_schedule not in valid_lr_schedules:
            raise ValueError(f"Unknown lr_schedule: {self.lr_schedule}")
        if self.gating_panel_size < 1:
            raise ValueError(f"gating_panel_size must be >= 1, got {self.gating_panel_size}")
        if self.iterations < 1:
            raise ValueError(f"iterations must be >= 1, got {self.iterations}")
        valid_mixing_modes = {"game_based", "sample_based"}
        if self.variant_mixing_mode not in valid_mixing_modes:
            raise ValueError(f"Unknown variant_mixing_mode: {self.variant_mixing_mode}")
        valid_variants = {"skirmish", "showdown", "clash", "battle"}
        if self.variant_fractions:
            unknown = set(self.variant_fractions) - valid_variants
            if unknown:
                raise ValueError(f"Unknown variant names in variant_fractions: {unknown}")
            if abs(sum(self.variant_fractions.values()) - 1.0) > 1e-4:
                raise ValueError(f"variant_fractions must sum to 1.0, got {sum(self.variant_fractions.values()):.4f}")
        if self.gating_variant_weights:
            unknown = set(self.gating_variant_weights) - valid_variants
            if unknown:
                raise ValueError(f"Unknown variant names in gating_variant_weights: {unknown}")
        if isinstance(self.temp_decay_half_life, dict):
            unknown = set(self.temp_decay_half_life) - valid_variants
            if unknown:
                raise ValueError(f"Unknown variant names in temp_decay_half_life: {unknown}")
            missing = valid_variants - set(self.temp_decay_half_life)
            if missing:
                raise ValueError(
                    f"temp_decay_half_life dict must list every variant "
                    f"({sorted(valid_variants)}); missing: {sorted(missing)}"
                )
            bad = {k: v for k, v in self.temp_decay_half_life.items()
                   if not isinstance(v, (int, float)) or v < 0}
            if bad:
                raise ValueError(f"temp_decay_half_life values must be non-negative numbers: {bad}")
        elif not isinstance(self.temp_decay_half_life, (int, float)):
            raise ValueError(
                f"temp_decay_half_life must be a number or a dict keyed by variant, "
                f"got {type(self.temp_decay_half_life).__name__}"
            )

    @property
    def network_name(self) -> str:
        return "densenet" if self.dense_net else "resnet"

    @property
    def auto_experiment_name(self) -> str:
        return (
            f"{self.network_name}-{self.depth}d-{self.channels}c"
            f"-{self.kernel_size}k-{self.selfplay_mcts_visits}sims"
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

    @property
    def resolved_loader_threads(self) -> int:
        return os.cpu_count() if self.loader_threads == -1 else self.loader_threads

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


# Defaults mirroring src/mcts.h library defaults; used as fallback when an
# experiment's config.yaml is missing or unreadable.
EXPERIMENT_DEFAULT_CPUCT = 1.25
EXPERIMENT_DEFAULT_FPU_REDUCTION = 0.25
EXPERIMENT_DEFAULT_GUMBEL_M = 16
EXPERIMENT_DEFAULT_GUMBEL_C_VISIT = 50.0
EXPERIMENT_DEFAULT_GUMBEL_C_SCALE = 1.0


def _experiment_config_path(checkpoint_path: str) -> str:
    """Return data/<game>/<exp>/config.yaml for a checkpoint at
    data/<game>/<exp>/checkpoint/<iter>-<name>.pt."""
    checkpoint_dir = os.path.dirname(checkpoint_path)
    experiment_dir = os.path.dirname(checkpoint_dir)
    return os.path.join(experiment_dir, "config.yaml")


def load_experiment_config(checkpoint_path: str):
    """Read (cpuct, fpu_reduction) from a checkpoint's experiment config.yaml.
    Falls back to library defaults if missing/unreadable."""
    config_path = _experiment_config_path(checkpoint_path)
    if os.path.isfile(config_path):
        try:
            with open(config_path) as f:
                cfg = yaml.safe_load(f) or {}
            return (
                cfg.get("cpuct", EXPERIMENT_DEFAULT_CPUCT),
                cfg.get("fpu_reduction", EXPERIMENT_DEFAULT_FPU_REDUCTION),
            )
        except Exception:
            pass
    return EXPERIMENT_DEFAULT_CPUCT, EXPERIMENT_DEFAULT_FPU_REDUCTION


def load_experiment_gumbel(checkpoint_path: str) -> dict:
    """Read gumbel_* params from a checkpoint's experiment config.yaml.

    Returns a dict with keys gumbel_enabled / gumbel_m / gumbel_c_visit /
    gumbel_c_scale / gumbel_full. Falls back to library defaults (PUCT) when
    the config is missing or unparseable.
    """
    defaults = {
        "gumbel_enabled": False,
        "gumbel_m": EXPERIMENT_DEFAULT_GUMBEL_M,
        "gumbel_c_visit": EXPERIMENT_DEFAULT_GUMBEL_C_VISIT,
        "gumbel_c_scale": EXPERIMENT_DEFAULT_GUMBEL_C_SCALE,
        "gumbel_full": False,
    }
    config_path = _experiment_config_path(checkpoint_path)
    if not os.path.isfile(config_path):
        return defaults
    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
        return {
            "gumbel_enabled": bool(cfg.get("gumbel_enabled", defaults["gumbel_enabled"])),
            "gumbel_m": int(cfg.get("gumbel_m", defaults["gumbel_m"])),
            "gumbel_c_visit": float(cfg.get("gumbel_c_visit", defaults["gumbel_c_visit"])),
            "gumbel_c_scale": float(cfg.get("gumbel_c_scale", defaults["gumbel_c_scale"])),
            "gumbel_full": bool(cfg.get("gumbel_full", defaults["gumbel_full"])),
        }
    except Exception:
        return defaults


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

    # Declared dataclass-field types (annotation-based). Used to decide whether
    # to coerce YAML-string values for numeric fields. Falling back to
    # type(default) misclassifies dual-typed fields like temp_decay_half_life
    # (declared ``object`` but default ``10`` -> looks like an int).
    declared_types = {f.name: f.type for f in dc_fields(TrainConfig)}

    if yaml_path:
        with open(yaml_path) as f:
            yaml_data = yaml.safe_load(f) or {}
        for key, val in yaml_data.items():
            if not hasattr(config, key):
                if warn:
                    print(f"Warning: ignoring unknown config key in YAML: {key}")
                continue
            # YAML 1.1 (PyYAML default) parses unsigned scientific notation like
            # "1e-4" as a string, not a float. Coerce to the field's declared
            # type when the field is annotated numeric and the YAML value is a
            # string. Skip coercion for fields with a broader annotation
            # (object / Union / etc.) so the dedicated validate() check can
            # produce a friendly error instead of an internal ValueError.
            declared = declared_types.get(key)
            if declared in (int, float) and isinstance(val, str):
                val = declared(val)
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
