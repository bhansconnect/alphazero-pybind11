#!/usr/bin/env python3
"""Interactive Star Gambit player with AI support.

Features:
- Human vs Human, Human vs AI, AI vs AI modes
- MCTS-based AI with configurable network, time/node limits, temperature
- Per-player AI configuration
- Move probability annotations
"""

import alphazero
import numpy as np
import time
import os
import glob
import re
import readline  # Enable line editing (backspace, arrow keys) in input()

try:
    import torch
    import neural_net
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: torch/neural_net not available. AI features disabled.")


def discover_networks(base_path="data/checkpoint"):
    """
    Discover available network checkpoints organized by training run.
    Returns: {run_name: [(iter_num, full_path), ...]} sorted by iter descending (newest first)
    """
    runs = {}
    if not os.path.isdir(base_path):
        return runs

    for run_name in sorted(os.listdir(base_path)):
        run_path = os.path.join(base_path, run_name)
        if not os.path.isdir(run_path):
            continue

        checkpoints = []
        for pt_file in glob.glob(os.path.join(run_path, "*.pt")):
            filename = os.path.basename(pt_file)
            # Extract iteration number from prefix (e.g., "0049" from "0049-name.pt")
            match = re.match(r'^(\d+)-', filename)
            if match:
                iter_num = int(match.group(1))
                checkpoints.append((iter_num, pt_file))

        if checkpoints:
            # Sort by iteration descending (newest first)
            checkpoints.sort(key=lambda x: x[0], reverse=True)
            runs[run_name] = checkpoints

    return runs


def load_network_for_players(ctx, path, players):
    """Load a network and assign it to specified players."""
    if not TORCH_AVAILABLE:
        print("Error: torch not available")
        return False
    try:
        net = neural_net.NNWrapper.load_checkpoint(
            ctx.game_class, os.path.dirname(path), os.path.basename(path)
        )
        for p in players:
            ctx.players[p].network = net
            ctx.players[p].network_path = path
            ctx.players[p].mcts = None
        return True
    except Exception as e:
        print(f"Error loading network: {e}")
        return False


def select_checkpoint_from_run(checkpoints, prompt_prefix=""):
    """
    Select a checkpoint from a list of (iter_num, path) tuples.
    Returns path string, or None for random policy.
    """
    latest_iter, latest_path = checkpoints[0]

    print(f"\nCheckpoints (newest first):")
    print(f"  l. Latest   -> iter {latest_iter:04d} ({os.path.basename(latest_path)})")
    print(f"  r. Random policy (no network)")
    print("  " + "-" * 50)

    # Show first few checkpoints
    show_count = min(10, len(checkpoints))
    for i in range(show_count):
        iter_num, path = checkpoints[i]
        print(f"  {i}. iter {iter_num:04d}")
    if len(checkpoints) > show_count:
        print(f"  ... ({len(checkpoints) - show_count} more, use -N to go back N from latest)")

    print("\nShortcuts: Enter=latest, -N=back N iters, iN=specific iter")

    while True:
        choice = input(f"\n{prompt_prefix}Select checkpoint: ").strip().lower()

        # Enter or 'l' = latest
        if choice in ['', 'l', 'latest']:
            return latest_path

        # 'r' = random
        if choice == 'r':
            return None

        # Negative number = go back N from latest
        if choice.startswith('-'):
            try:
                offset = int(choice)  # negative
                idx = -offset  # convert to positive index
                if 0 <= idx < len(checkpoints):
                    return checkpoints[idx][1]
                print(f"Only {len(checkpoints)} checkpoints available")
            except ValueError:
                print("Invalid offset")
            continue

        # iN or iter:N = specific iteration
        iter_match = re.match(r'^i(?:ter:?)?(\d+)$', choice)
        if iter_match:
            target_iter = int(iter_match.group(1))
            for iter_num, path in checkpoints:
                if iter_num == target_iter:
                    return path
            print(f"Iteration {target_iter} not found")
            continue

        # Numeric index
        try:
            idx = int(choice)
            if 0 <= idx < len(checkpoints):
                return checkpoints[idx][1]
            print(f"Enter 0-{len(checkpoints)-1}")
        except ValueError:
            print("Invalid input")


def select_network_interactive(ctx):
    """
    Interactive network selection menu.
    Returns True if a network was selected, False if both players use random.
    """
    if not TORCH_AVAILABLE:
        print("torch not available - using random policy")
        return False

    runs = discover_networks()

    if not runs:
        print("No checkpoints found in data/checkpoint/")
        print("Using random policy")
        return False

    print("\n=== Network Selection ===")

    # Run selection
    run_names = list(runs.keys())
    if len(run_names) == 1:
        selected_run = run_names[0]
        print(f"Training run: {selected_run}")
    else:
        print("Available training runs:")
        for i, name in enumerate(run_names):
            cpts = runs[name]
            iter_range = f"{cpts[-1][0]:04d}-{cpts[0][0]:04d}"
            print(f"  {i+1}. {name} ({len(cpts)} checkpoints: {iter_range})")

        while True:
            choice = input("\nSelect run (number, or 'r' for random): ").strip().lower()
            if choice == 'r':
                print("Using random policy for both players")
                return False
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(run_names):
                    selected_run = run_names[idx]
                    break
                print(f"Enter 1-{len(run_names)}")
            except ValueError:
                if choice == '':
                    selected_run = run_names[0]
                    break
                print("Invalid input")

    checkpoints = runs[selected_run]

    # Ask if same or different networks
    print(f"\nRun: {selected_run}")
    choice = input("Same network for both players? [y]es / [n]o / [r]andom (default=yes): ").strip().lower()

    if choice == 'r':
        print("Using random policy for both players")
        return False

    if choice == 'n':
        # Different networks per player
        any_loaded = False
        for player in [0, 1]:
            print(f"\n--- Player {player} ---")
            path = select_checkpoint_from_run(checkpoints, f"P{player} ")
            if path is None:
                print(f"Player {player} using random policy")
            elif load_network_for_players(ctx, path, [player]):
                print(f"Player {player} loaded: {os.path.basename(path)}")
                any_loaded = True
        return any_loaded
    else:
        # Same network for both
        path = select_checkpoint_from_run(checkpoints)
        if path is None:
            print("Using random policy for both players")
            return False
        if load_network_for_players(ctx, path, [0, 1]):
            print(f"Loaded for both players: {os.path.basename(path)}")
            return True
        return False


# ANSI color codes
RED = '\033[91m'
BLUE = '\033[94m'
RESET = '\033[0m'

# Game size selection - using Skirmish by default
# 2D grid-based action space layout:
#   Spatial actions: 0 to (BOARD_DIM * BOARD_DIM * 10 - 1)
#     Each grid cell has 10 slots:
#       0: MOVE_FORWARD, 1: MOVE_FORWARD_LEFT, 2: MOVE_FORWARD_RIGHT
#       3: ROTATE_LEFT, 4: ROTATE_RIGHT
#       5: FIRE_FORWARD, 6: FIRE_FORWARD_LEFT, 7: FIRE_FORWARD_RIGHT
#       8: FIRE_REAR_LEFT, 9: FIRE_REAR_RIGHT
#   Deploy actions: SPATIAL_ACTIONS to SPATIAL_ACTIONS + 17 (3 types * 6 facings)
#   End turn: SPATIAL_ACTIONS + 18
#
# For Skirmish/Clash (BOARD_SIDE=5): BOARD_DIM=9, NUM_MOVES = 9*9*10 + 18 + 1 = 829
# For Battle (BOARD_SIDE=6): BOARD_DIM=11, NUM_MOVES = 11*11*10 + 18 + 1 = 1229

# Slot mapping for P1 canonicalization (self-inverse: swaps left/right)
SLOT_MAP = [0, 2, 1, 4, 3, 5, 7, 6, 9, 8]

ACTIONS_PER_POSITION = 10

# Slot meanings (shared across unit types, but not all units use all slots)
SLOT_NAMES = {
    0: 'move_forward',
    1: 'move_forward_left',
    2: 'move_forward_right',
    3: 'rotate_left',
    4: 'rotate_right',
    5: 'fire_forward',
    6: 'fire_forward_left',
    7: 'fire_forward_right',
    8: 'fire_rear_left',
    9: 'fire_rear_right',
}

# Short names for commands
SLOT_SHORT = {
    0: 'f', 1: 'fl', 2: 'fr', 3: 'l', 4: 'r',
    5: 'ff', 6: 'ffl', 7: 'ffr', 8: 'frl', 9: 'frr'
}

# Which slots are valid per unit type (from C++ definitions)
# Fighter: move f/fl/fr (0,1,2), fire forward (5)
# Cruiser: move f/fl/fr/l/r (0,1,2,3,4), fire forward/fl/fr (5,6,7)
# Dread: move fl/fr/l/r (1,2,3,4), fire fl/fr/rl/rr (6,7,8,9)
VALID_SLOTS = {
    0: {0, 1, 2, 5},  # Fighter
    1: {0, 1, 2, 3, 4, 5, 6, 7},  # Cruiser
    2: {1, 2, 3, 4, 6, 7, 8, 9},  # Dreadnought
}


class GameConfig:
    """Action space configuration for a game size (2D grid-based encoding)."""
    def __init__(self, max_fighters, max_cruisers, max_dreads, board_side):
        self.max_fighters = max_fighters
        self.max_cruisers = max_cruisers
        self.max_dreads = max_dreads
        self.board_side = board_side
        self.board_dim = 2 * board_side - 1

        # 2D grid-based action space
        self.spatial_actions = self.board_dim * self.board_dim * ACTIONS_PER_POSITION
        self.deploy_actions = 18  # 3 types * 6 facings

        # Action offsets
        self.deploy_offset = self.spatial_actions
        self.end_turn_offset = self.deploy_offset + self.deploy_actions
        self.num_moves = self.end_turn_offset + 1

    def encode_spatial(self, row, col, slot):
        """Encode (row, col, slot) -> action"""
        return row * self.board_dim * ACTIONS_PER_POSITION + col * ACTIONS_PER_POSITION + slot

    def decode_spatial(self, action):
        """Decode action -> (row, col, slot)"""
        slot = action % ACTIONS_PER_POSITION
        pos = action // ACTIONS_PER_POSITION
        col = pos % self.board_dim
        row = pos // self.board_dim
        return row, col, slot

    def encode_deploy(self, unit_type, facing):
        """Encode deploy action"""
        return self.deploy_offset + unit_type * 6 + facing

    def decode_deploy(self, action):
        """Decode deploy action -> (unit_type, facing)"""
        rel = action - self.deploy_offset
        return rel // 6, rel % 6

    def hex_to_rowcol(self, q, r):
        """Convert hex (q, r) to grid (row, col)."""
        bs = self.board_side - 1
        return q + bs, r + bs

    def rowcol_to_hex(self, row, col):
        """Convert grid (row, col) to hex (q, r)."""
        bs = self.board_side - 1
        return row - bs, col - bs

    def canonicalize_spatial(self, row, col, slot):
        """Canonicalize (row, col, slot) for P1 (180-degree rotation + slot swap)."""
        bd = self.board_dim
        return bd - 1 - row, bd - 1 - col, SLOT_MAP[slot]

    def decanon_spatial(self, row, col, slot):
        """De-canonicalize (row, col, slot) for P1 (same operation - self-inverse)."""
        return self.canonicalize_spatial(row, col, slot)

    def canonicalize_deploy_facing(self, facing):
        """Canonicalize deploy facing for P1 (rotate 180 degrees)."""
        return (facing + 3) % 6

    def decanon_deploy_facing(self, facing):
        """De-canonicalize deploy facing for P1 (same operation - self-inverse)."""
        return (facing + 3) % 6


SKIRMISH = GameConfig(3, 1, 0, board_side=5)
CLASH = GameConfig(3, 2, 1, board_side=5)
BATTLE = GameConfig(4, 3, 2, board_side=6)

# Unified uses 11x11 grid-based encoding (not hex-based)
# Override with correct values matching UnifiedActionSpace in C++
UNIFIED = GameConfig(4, 3, 2, board_side=6)  # Base config
UNIFIED.board_dim = 11  # Grid dimension for unified
UNIFIED.num_hexes = 121  # 11 * 11 grid cells
UNIFIED.hex_actions = 11 * 11 * 10  # 1210 spatial actions
UNIFIED.deploy_offset = 1210
UNIFIED.end_turn_offset = 1228
UNIFIED.num_moves = 1229

# Direction names
DIRECTION_NAMES = ['E', 'NE', 'NW', 'W', 'SW', 'SE']
FIGHTER_MOVE_NAMES = ['f', 'fl', 'fr']
CRUISER_MOVE_NAMES = ['l', 'fl', 'f', 'fr', 'r']
DREAD_MOVE_NAMES = ['l', 'fl', 'fr', 'r']
CRUISER_CANNON_NAMES = ['l', 'f', 'r']
DREAD_CANNON_NAMES = ['rl', 'fl', 'fr', 'rr']

# Detailed direction names for display
FIGHTER_MOVE_DETAIL = ['forward', 'forward-left', 'forward-right']
CRUISER_MOVE_DETAIL = ['rotate-left', 'forward-left', 'forward', 'forward-right', 'rotate-right']
DREAD_MOVE_DETAIL = ['rotate-left', 'forward-left', 'forward-right', 'rotate-right']
CRUISER_CANNON_DETAIL = ['left cannon', 'forward cannon', 'right cannon']
DREAD_CANNON_DETAIL = ['rear-left cannon', 'front-left cannon', 'front-right cannon', 'rear-right cannon']

# Type names for display
TYPE_NAMES = ['Fighter', 'Cruiser', 'Dreadnought', 'Portal']
TYPE_CHARS = ['F', 'C', 'D', 'P']

# Default AI parameters
DEFAULT_CPUCT = 1.25
DEFAULT_FPU_REDUCTION = 0.25
DEFAULT_TEMPERATURE = 0.5
DEFAULT_NODE_LIMIT = 100


class PlayerConfig:
    """Configuration for a single player (human or AI)."""
    def __init__(self):
        self.is_ai = False
        self.network_path = None  # None means random
        self.network = None       # Loaded network wrapper
        self.think_time = None    # Seconds, or None for no time limit
        self.node_limit = DEFAULT_NODE_LIMIT  # Max nodes, or None for no limit
        self.temperature = DEFAULT_TEMPERATURE
        self.mcts = None          # MCTS instance (reset on network change)
        self.show_hints = False   # Show AI analysis on human turns

    def __str__(self):
        if not self.is_ai:
            hints_str = ", hints=on" if self.show_hints else ""
            return f"Human{hints_str}"
        net_str = os.path.basename(self.network_path) if self.network_path else "random"
        time_str = f"{self.think_time}s" if self.think_time else "unlimited"
        node_str = str(self.node_limit) if self.node_limit else "unlimited"
        return f"AI(net={net_str}, time={time_str}, nodes={node_str}, temp={self.temperature})"


class GameContext:
    """Global game context with player configurations."""
    def __init__(self, game, cfg, game_class):
        self.game = game
        self.cfg = cfg
        self.game_class = game_class
        self.players = [PlayerConfig(), PlayerConfig()]
        self.auto_play = False  # Auto-play mode vs manual selection
        self.cpuct = DEFAULT_CPUCT
        self.fpu_reduction = DEFAULT_FPU_REDUCTION


def create_mcts(game_class, cpuct=DEFAULT_CPUCT, fpu_reduction=DEFAULT_FPU_REDUCTION):
    """Create a new MCTS instance for the game."""
    return alphazero.MCTS(
        cpuct,
        game_class.NUM_PLAYERS(),
        game_class.NUM_MOVES(),
        0.25,  # epsilon for Dirichlet noise
        1.4,   # alpha for Dirichlet noise
        fpu_reduction
    )


def apply_temperature(probs, temperature):
    """Apply temperature to probabilities."""
    if temperature == 0:
        # Greedy: all mass on highest prob
        result = np.zeros_like(probs)
        result[np.argmax(probs)] = 1.0
        return result
    elif temperature == 1.0:
        return probs
    else:
        # Apply temperature: p^(1/T) then renormalize
        scaled = np.power(probs + 1e-10, 1.0 / temperature)
        return scaled / scaled.sum()


def run_mcts_search(gs, agent, mcts, time_limit=None, node_limit=None):
    """
    Run MCTS search with time or node limit.
    Returns (visit_counts, num_simulations, wld).
    wld is [win, loss, draw] from current player's perspective.
    """
    if time_limit is None and node_limit is None:
        node_limit = DEFAULT_NODE_LIMIT

    start = time.time()
    sims = 0

    while True:
        # Check termination conditions
        if time_limit is not None and time.time() - start >= time_limit:
            break
        if node_limit is not None and sims >= node_limit:
            break

        # Expand one node
        leaf = mcts.find_leaf(gs)
        if agent is None:
            # Random policy
            v = np.zeros(gs.NUM_PLAYERS() + 1)
            v[-1] = 1.0  # Uniform draw assumption
            pi = np.ones(gs.NUM_MOVES()) / gs.NUM_MOVES()
        else:
            canonical = torch.from_numpy(np.array(leaf.canonicalized()))
            v, pi = agent.predict(canonical)
            v = v.cpu().numpy().flatten()
            pi = pi.cpu().numpy().flatten()

        mcts.process_result(gs, v, pi, sims == 0)  # Add noise on first sim
        sims += 1

    counts = np.array(mcts.counts())
    wld = np.array(mcts.root_value())
    return counts, sims, wld


def get_ai_probs(ctx, player_idx, valids):
    """
    Get AI move probabilities for display.
    Returns (probs, source_str, sims, wld) where probs are temperature-adjusted and normalized to valid moves.
    wld is [win, loss, draw] or None.
    """
    pcfg = ctx.players[player_idx]
    wld = None

    # Create or reuse MCTS
    if pcfg.mcts is None:
        pcfg.mcts = create_mcts(ctx.game_class, ctx.cpuct, ctx.fpu_reduction)

    # Determine if we should run MCTS or just use raw policy
    should_search = (pcfg.think_time is not None and pcfg.think_time > 0) or \
                   (pcfg.node_limit is not None and pcfg.node_limit > 0)

    if should_search and (pcfg.network is not None or TORCH_AVAILABLE):
        # Run MCTS search
        counts, sims, wld = run_mcts_search(
            ctx.game, pcfg.network, pcfg.mcts,
            time_limit=pcfg.think_time,
            node_limit=pcfg.node_limit
        )

        # Convert counts to probabilities
        if counts.sum() > 0:
            probs = counts.astype(float) / counts.sum()
        else:
            probs = np.ones(ctx.game.NUM_MOVES()) / ctx.game.NUM_MOVES()

        source = f"MCTS ({sims} sims)"
    else:
        # Use raw policy network (or uniform for random)
        sims = 0
        if pcfg.network is not None:
            canonical = torch.from_numpy(np.array(ctx.game.canonicalized()))
            _, pi = pcfg.network.predict(canonical)
            probs = pi.cpu().numpy().flatten()
            # Softmax is already applied by network
            source = "policy network"
        else:
            probs = np.ones(ctx.game.NUM_MOVES()) / ctx.game.NUM_MOVES()
            source = "uniform random"

    # Apply temperature
    probs = apply_temperature(probs, pcfg.temperature)

    # Mask invalid moves and renormalize
    probs[valids == 0] = 0
    if probs.sum() > 0:
        probs = probs / probs.sum()
    else:
        # Fallback: uniform over valid
        valid_count = valids.sum()
        if valid_count > 0:
            probs[valids == 1] = 1.0 / valid_count

    return probs, source, sims, wld


def get_ai_move(ctx, player_idx, valids):
    """Get AI move for the specified player. Returns (action, probs, source, sims, wld)."""
    probs, source, sims, wld = get_ai_probs(ctx, player_idx, valids)

    # Sample from distribution
    action = np.random.choice(len(probs), p=probs)

    # Reset MCTS for next move (don't reuse tree)
    ctx.players[player_idx].mcts = None

    return action, probs, source, sims, wld


def color_unit(name, player):
    """Wrap unit name in ANSI color codes based on player."""
    color = RED if player == 0 else BLUE
    return f"{color}{name}{RESET}"


def get_unit_name(unit_type, slot):
    """Get unit name like F1, C2, etc."""
    return f"{TYPE_CHARS[unit_type]}{slot + 1}"


def find_unit_at_position(game, row, col, cfg, current_player):
    """Find unit at grid position. De-canonicalizes for P1. Returns UnitInfo or None."""
    # De-canonicalize the position for P1 to get true coordinates
    if current_player == 1:
        true_row, true_col, _ = cfg.decanon_spatial(row, col, 0)
    else:
        true_row, true_col = row, col

    q, r = cfg.rowcol_to_hex(true_row, true_col)
    units = game.get_units()
    for u in units:
        if u.anchor_q == q and u.anchor_r == r and u.player == current_player:
            return u
    return None


def find_unit_by_type_slot(game, unit_type, slot, current_player):
    """Find unit by type, slot, and player. Returns UnitInfo or None."""
    units = game.get_units()
    for u in units:
        if u.type == unit_type and u.slot == slot and u.player == current_player:
            return u
    return None


def decode_action(action, cfg, game=None):
    """Decode action ID to human-readable description (2D grid-based encoding)."""
    current_player = game.current_player() if game is not None else 0

    # Deploy actions
    if cfg.deploy_offset <= action < cfg.end_turn_offset:
        unit_type, facing = cfg.decode_deploy(action)
        # De-canonicalize facing for P1
        if current_player == 1:
            facing = cfg.decanon_deploy_facing(facing)
        type_chars = ['f', 'c', 'd']
        facing_names = ['e', 'ne', 'nw', 'w', 'sw', 'se']
        return f"d {type_chars[unit_type]} {facing_names[facing]}"

    # End turn
    if action >= cfg.end_turn_offset:
        return "e"

    # Spatial action
    if action < cfg.spatial_actions:
        row, col, slot = cfg.decode_spatial(action)

        # De-canonicalize for display
        if current_player == 1:
            disp_row, disp_col, disp_slot = cfg.decanon_spatial(row, col, slot)
        else:
            disp_row, disp_col, disp_slot = row, col, slot

        # Try to find unit at this position
        unit = None
        if game is not None:
            unit = find_unit_at_position(game, row, col, cfg, current_player)

        # Format unit name
        if unit is not None:
            unit_name = f"{TYPE_CHARS[unit.type].lower()}{unit.slot + 1}"
        else:
            q, r = cfg.rowcol_to_hex(disp_row, disp_col)
            unit_name = f"({q},{r})"

        # Movement slots (0-4) - use de-canonicalized slot for display
        if disp_slot <= 4:
            move_names = {0: 'f', 1: 'fl', 2: 'fr', 3: 'l', 4: 'r'}
            return f"m {unit_name} {move_names.get(disp_slot, '?')}"
        # Fire slots (5-9)
        else:
            fire_names = {5: 'f', 6: 'fl', 7: 'fr', 8: 'rl', 9: 'rr'}
            return f"f {unit_name} {fire_names.get(disp_slot, '?')}"

    return f"?{action}"


def format_action(action, cfg, game):
    """Format action with short form and detailed description, with colored unit names."""
    short = decode_action(action, cfg, game)
    current_player = game.current_player()

    # Deploy action
    if cfg.deploy_offset <= action < cfg.end_turn_offset:
        unit_type, facing = cfg.decode_deploy(action)
        # De-canonicalize facing for P1
        if current_player == 1:
            facing = cfg.decanon_deploy_facing(facing)
        type_name = TYPE_NAMES[unit_type].lower()
        facing_name = DIRECTION_NAMES[facing]
        return f"{short:<12} -  Deploy {type_name} facing {facing_name}"

    # End turn
    if action >= cfg.end_turn_offset:
        return f"{short:<12} -  End turn"

    # Spatial action
    if action < cfg.spatial_actions:
        row, col, slot = cfg.decode_spatial(action)

        # De-canonicalize for display
        if current_player == 1:
            disp_row, disp_col, disp_slot = cfg.decanon_spatial(row, col, slot)
        else:
            disp_row, disp_col, disp_slot = row, col, slot

        unit = find_unit_at_position(game, row, col, cfg, current_player)

        if unit is None:
            q, r = cfg.rowcol_to_hex(disp_row, disp_col)
            return f"{short:<12} -  (no unit at ({q},{r}))"

        name = color_unit(get_unit_name(unit.type, unit.slot), current_player)

        # Movement slots (0-4) - use de-canonicalized slot for display
        if disp_slot <= 4:
            move_detail = {
                0: 'forward', 1: 'forward-left', 2: 'forward-right',
                3: 'rotate-left', 4: 'rotate-right'
            }
            detail = move_detail.get(disp_slot, f'slot{disp_slot}')
            return f"{short:<12} -  {name} {detail}"

        # Fire slots (5-9) - pass canonical action directly to get_fire_info
        else:
            cannon_detail = {
                5: 'forward cannon', 6: 'forward-left cannon', 7: 'forward-right cannon',
                8: 'rear-left cannon', 9: 'rear-right cannon'
            }
            cannon_name = cannon_detail.get(disp_slot, f'cannon{disp_slot}')

            fire_info = game.get_fire_info(action)
            if fire_info.has_target:
                target_name = color_unit(get_unit_name(fire_info.target_type, fire_info.target_slot),
                                         fire_info.target_player)
                return f"{short:<12} -  {name} {cannon_name} -> {target_name} ({fire_info.damage} dmg)"
            return f"{short:<12} -  {name} {cannon_name}"

    return f"{short:<12} -  ???"


def print_unit_lists(game):
    """Print unit lists for both players with full info."""
    units = game.get_units()

    p0_units = [u for u in units if u.player == 0]
    p1_units = [u for u in units if u.player == 1]

    # Sort by type then slot
    p0_units.sort(key=lambda u: (u.type, u.slot))
    p1_units.sort(key=lambda u: (u.type, u.slot))

    print("\nP0 units:")
    if not p0_units:
        print("  (none)")
    for u in p0_units:
        name = color_unit(get_unit_name(u.type, u.slot), 0)
        facing = DIRECTION_NAMES[u.facing]
        moves_word = "move" if u.moves_left == 1 else "moves"
        print(f"  {name}  -  {u.hp}hp  ({u.anchor_q},{u.anchor_r}) facing {facing}  {u.moves_left} {moves_word}")

    print("P1 units:")
    if not p1_units:
        print("  (none)")
    for u in p1_units:
        name = color_unit(get_unit_name(u.type, u.slot), 1)
        facing = DIRECTION_NAMES[u.facing]
        moves_word = "move" if u.moves_left == 1 else "moves"
        print(f"  {name}  -  {u.hp}hp  ({u.anchor_q},{u.anchor_r}) facing {facing}  {u.moves_left} {moves_word}")


def parse_command(cmd, valids, cfg, ctx=None):
    """Parse user command and return action ID or special command string."""
    parts = cmd.strip().split()  # Don't lowercase yet - need original case for paths
    if not parts:
        return None

    cmd_lower = parts[0].lower()

    if cmd_lower == 'q':
        return 'quit'
    if cmd_lower == 'h':
        return 'help'
    if cmd_lower == 's':
        return 'show'
    if cmd_lower == 'v':
        return 'valid'
    if cmd_lower == 'u':
        return 'undo'
    if cmd_lower == 'status':
        return 'status'
    if cmd_lower == 'auto':
        return 'auto'
    if cmd_lower == 'manual':
        return 'manual'

    # AI configuration commands
    if cmd_lower == 'ai' and ctx:
        # ai <0|1|both|none>
        if len(parts) < 2:
            print("Usage: ai <0|1|both|none>")
            return 'config'
        arg = parts[1].lower()
        if arg == 'both':
            ctx.players[0].is_ai = True
            ctx.players[1].is_ai = True
            print("Both players set to AI")
        elif arg == 'none':
            ctx.players[0].is_ai = False
            ctx.players[1].is_ai = False
            print("Both players set to human")
        elif arg in ['0', '1']:
            player = int(arg)
            ctx.players[player].is_ai = True
            ctx.players[1 - player].is_ai = False
            print(f"Player {player} set to AI, Player {1-player} set to human")
        else:
            print("Usage: ai <0|1|both|none>")
        return 'config'

    if cmd_lower == 'time' and ctx:
        # time <0|1> <seconds|off>
        if len(parts) < 3:
            print("Usage: time <0|1> <seconds|off>")
            return 'config'
        try:
            player = int(parts[1])
            if player not in [0, 1]:
                raise ValueError()
        except ValueError:
            print("Player must be 0 or 1")
            return 'config'
        if parts[2].lower() == 'off':
            ctx.players[player].think_time = None
            print(f"Player {player} think time: unlimited")
        else:
            try:
                ctx.players[player].think_time = float(parts[2])
                print(f"Player {player} think time: {ctx.players[player].think_time}s")
            except ValueError:
                print("Time must be a number or 'off'")
        return 'config'

    if cmd_lower == 'nodes' and ctx:
        # nodes <0|1> <count|off>
        if len(parts) < 3:
            print("Usage: nodes <0|1> <count|off>")
            return 'config'
        try:
            player = int(parts[1])
            if player not in [0, 1]:
                raise ValueError()
        except ValueError:
            print("Player must be 0 or 1")
            return 'config'
        if parts[2].lower() == 'off':
            ctx.players[player].node_limit = None
            print(f"Player {player} node limit: unlimited")
        else:
            try:
                ctx.players[player].node_limit = int(parts[2])
                print(f"Player {player} node limit: {ctx.players[player].node_limit}")
            except ValueError:
                print("Node count must be an integer or 'off'")
        return 'config'

    if cmd_lower == 'temp' and ctx:
        # temp <0|1> <value>
        if len(parts) < 3:
            print("Usage: temp <0|1> <value>")
            return 'config'
        try:
            player = int(parts[1])
            if player not in [0, 1]:
                raise ValueError()
            ctx.players[player].temperature = float(parts[2])
            print(f"Player {player} temperature: {ctx.players[player].temperature}")
        except ValueError:
            print("Player must be 0 or 1, value must be a number")
        return 'config'

    if cmd_lower == 'hints' and ctx:
        # hints <0|1> <on|off>
        if len(parts) < 3:
            print("Usage: hints <0|1> <on|off>")
            return 'config'
        try:
            player = int(parts[1])
            if player not in [0, 1]:
                raise ValueError()
        except ValueError:
            print("Player must be 0 or 1")
            return 'config'
        if parts[2].lower() == 'on':
            ctx.players[player].show_hints = True
            print(f"Player {player} AI hints: enabled")
        elif parts[2].lower() == 'off':
            ctx.players[player].show_hints = False
            print(f"Player {player} AI hints: disabled")
        else:
            print("Usage: hints <0|1> <on|off>")
        return 'config'

    if cmd_lower == 'net' and ctx:
        # net (no args) -> interactive selection
        # net <0|1|both> <path|random>
        if len(parts) == 1:
            # Interactive selection
            select_network_interactive(ctx)
            return 'config'

        if len(parts) < 3:
            print("Usage: net  (interactive)")
            print("       net <0|1|both> <path|random>")
            return 'config'

        player_arg = parts[1].lower()
        path_arg = parts[2]  # Keep original case for path

        if player_arg == 'both':
            players = [0, 1]
        elif player_arg in ['0', '1']:
            players = [int(player_arg)]
        else:
            print("Player must be 0, 1, or 'both'")
            return 'config'

        if path_arg.lower() == 'random':
            for p in players:
                ctx.players[p].network = None
                ctx.players[p].network_path = None
                ctx.players[p].mcts = None
            player_str = "Both players" if len(players) == 2 else f"Player {players[0]}"
            print(f"{player_str} using random policy")
        else:
            if load_network_for_players(ctx, path_arg, players):
                player_str = "both players" if len(players) == 2 else f"Player {players[0]}"
                print(f"Loaded for {player_str}: {path_arg}")
        return 'config'

    # Now lowercase for game commands
    parts = [p.lower() for p in parts]

    if parts[0] == 'e':
        # End turn
        if valids[cfg.end_turn_offset] == 1:
            return cfg.end_turn_offset
        print("End turn not valid!")
        return None

    # Get game from context for grid-based action encoding
    game = ctx.game if ctx else None
    current_player = game.current_player() if game else 0

    if parts[0] == 'm' and len(parts) >= 3:
        # Move: m <unit><slot> <direction>
        unit_slot = parts[1]
        direction = parts[2]

        if len(unit_slot) < 2:
            print("Invalid unit. Use f1, f2, c1, d1, etc.")
            return None

        unit_type_char = unit_slot[0]
        try:
            slot = int(unit_slot[1:]) - 1  # Convert to 0-indexed
        except ValueError:
            print("Invalid slot number.")
            return None

        # Map unit type char to type index
        type_map = {'f': 0, 'c': 1, 'd': 2}
        if unit_type_char not in type_map:
            print("Invalid unit type. Use f, c, or d.")
            return None
        unit_type = type_map[unit_type_char]

        if game is None:
            print("Game state not available for move parsing.")
            return None

        unit = find_unit_by_type_slot(game, unit_type, slot, current_player)
        if unit is None:
            print(f"Unit {unit_slot} not found on board.")
            return None

        # Convert unit's true anchor to grid position
        row, col = cfg.hex_to_rowcol(unit.anchor_q, unit.anchor_r)

        # Map direction to movement slot (0-4)
        if unit_type == 0:  # Fighter
            dir_map = {'f': 0, 'fl': 1, 'fr': 2}
        elif unit_type == 1:  # Cruiser
            dir_map = {'f': 0, 'fl': 1, 'fr': 2, 'l': 3, 'r': 4}
        else:  # Dreadnought
            dir_map = {'fl': 1, 'fr': 2, 'l': 3, 'r': 4}

        if direction not in dir_map:
            print(f"Invalid direction '{direction}' for {TYPE_NAMES[unit_type]}.")
            return None

        move_slot = dir_map[direction]

        # Canonicalize for current player
        if current_player == 1:
            row, col, move_slot = cfg.canonicalize_spatial(row, col, move_slot)

        action = cfg.encode_spatial(row, col, move_slot)

        if valids[action] == 1:
            return action
        print(f"Move not valid! (action {action})")
        return None

    if parts[0] == 'f' and len(parts) >= 2:
        # Fire: f <unit><slot> [cannon]
        unit_slot = parts[1]
        cannon = parts[2] if len(parts) >= 3 else None

        if len(unit_slot) < 2:
            print("Invalid unit. Use f1, c1 l, d1 fl, etc.")
            return None

        unit_type_char = unit_slot[0]
        try:
            slot = int(unit_slot[1:]) - 1
        except ValueError:
            print("Invalid slot number.")
            return None

        # Map unit type char to type index
        type_map = {'f': 0, 'c': 1, 'd': 2}
        if unit_type_char not in type_map:
            print("Invalid unit type. Use f, c, or d.")
            return None
        unit_type = type_map[unit_type_char]

        if game is None:
            print("Game state not available for fire parsing.")
            return None

        unit = find_unit_by_type_slot(game, unit_type, slot, current_player)
        if unit is None:
            print(f"Unit {unit_slot} not found on board.")
            return None

        # Convert unit's true anchor to grid position
        row, col = cfg.hex_to_rowcol(unit.anchor_q, unit.anchor_r)

        # Map cannon direction to fire slot (5-9)
        if unit_type == 0:  # Fighter - only forward fire (slot 5)
            if cannon is not None and cannon != 'f':
                print("Fighter only has forward cannon.")
                return None
            fire_slot = 5
        elif unit_type == 1:  # Cruiser - forward, forward-left, forward-right (slots 5,6,7)
            cannon_map = {'l': 6, 'f': 5, 'r': 7, 'fl': 6, 'fr': 7}
            if cannon not in cannon_map:
                print("Invalid cruiser cannon. Use l, f, r (or fl, fr).")
                return None
            fire_slot = cannon_map[cannon]
        else:  # Dreadnought - forward-left, forward-right, rear-left, rear-right (slots 6,7,8,9)
            cannon_map = {'fl': 6, 'fr': 7, 'rl': 8, 'rr': 9}
            if cannon not in cannon_map:
                print("Invalid dreadnought cannon. Use fl, fr, rl, rr.")
                return None
            fire_slot = cannon_map[cannon]

        # Canonicalize for current player
        if current_player == 1:
            row, col, fire_slot = cfg.canonicalize_spatial(row, col, fire_slot)

        action = cfg.encode_spatial(row, col, fire_slot)

        if valids[action] == 1:
            return action
        print(f"Fire not valid! (action {action})")
        return None

    if parts[0] == 'd' and len(parts) >= 3:
        # Deploy: d <type> <facing>
        unit_type = parts[1]
        facing_str = parts[2]

        type_map = {'f': 0, 'c': 1, 'd': 2}
        facing_map = {'e': 0, 'ne': 1, 'nw': 2, 'w': 3, 'sw': 4, 'se': 5}

        if unit_type not in type_map:
            print("Invalid unit type. Use f, c, or d.")
            return None
        if facing_str not in facing_map:
            print("Invalid facing. Use e, ne, nw, w, sw, se.")
            return None

        facing = facing_map[facing_str]
        # Canonicalize facing for P1
        if current_player == 1:
            facing = cfg.canonicalize_deploy_facing(facing)

        action = cfg.deploy_offset + type_map[unit_type] * 6 + facing
        if valids[action] == 1:
            return action
        print(f"Deploy not valid! (action {action})")
        return None

    # Try to parse as raw action number
    try:
        action = int(parts[0])
        if 0 <= action < cfg.num_moves and valids[action] == 1:
            return action
        print(f"Action {action} not valid!")
    except ValueError:
        pass

    return None


def print_actions_menu(valids, cfg, game, probs=None, source=None, wld=None, show_commands=True):
    """Print a menu of available actions with detailed descriptions and optional probabilities."""
    print("\n=== Available Actions ===")

    if source:
        print(f"  (probabilities from {source})")

    if wld is not None:
        print(f"  Win: {wld[0]*100:.1f}%  Loss: {wld[1]*100:.1f}%  Draw: {wld[2]*100:.1f}%")

    def format_with_prob(action):
        base = format_action(action, cfg, game)
        if probs is not None and probs[action] > 0.001:
            return f"{base}  [{probs[action]*100:5.1f}%]"
        elif probs is not None:
            return f"{base}  [  0.0%]"
        return base

    # Group valid actions (keep original order within groups)
    # Spatial: slots 0-4 are moves, slots 5-9 are fires
    moves = []
    fires = []
    deploys = []
    end_turn = False

    for i in range(cfg.num_moves):
        if valids[i] != 1:
            continue
        if i < cfg.spatial_actions:
            # Spatial action - determine if move or fire by slot
            _, _, slot = cfg.decode_spatial(i)
            if slot <= 4:
                moves.append(i)
            else:
                fires.append(i)
        elif i < cfg.end_turn_offset:
            deploys.append(i)
        else:
            end_turn = True

    if moves:
        print("\nMoves:")
        for action in moves:
            print(f"  {format_with_prob(action)}")

    if fires:
        print("\nFire:")
        for action in fires:
            print(f"  {format_with_prob(action)}")

    if deploys:
        print("\nDeploy:")
        for action in deploys:
            print(f"  {format_with_prob(action)}")

    if end_turn:
        print(f"\n  {format_with_prob(cfg.end_turn_offset)}")

    if show_commands:
        print("\nCommands: (q)uit, (h)elp, (s)how board, (u)ndo, (v)alid actions")
        print("AI config: ai, net, time, nodes, temp, hints, auto, manual, status")


def play_random_action(valids):
    """Play a random valid action."""
    valid_indices = [i for i in range(len(valids)) if valids[i] == 1]
    if not valid_indices:
        return None
    return np.random.choice(valid_indices)


def print_status(ctx):
    """Print current configuration status."""
    print("\n=== Configuration Status ===")
    print(f"  Auto-play: {ctx.auto_play}")
    print(f"  CPUCT: {ctx.cpuct}")
    print(f"  FPU reduction: {ctx.fpu_reduction}")
    for i in range(2):
        print(f"  Player {i}: {ctx.players[i]}")


def print_help():
    """Print help message."""
    print("\n=== Game Commands ===")
    print("  m <unit><slot> <dir>  - Move (e.g., m f1 f, m c1 fl)")
    print("    Fighter dirs: f, fl, fr")
    print("    Cruiser dirs: l, fl, f, fr, r")
    print("    Dread dirs: l, fl, fr, r")
    print("  f <unit><slot> [cannon] - Fire (e.g., f f1, f c1 l)")
    print("  d <type> <facing>       - Deploy (e.g., d f se)")
    print("    Types: f, c, d")
    print("    Facings: e, ne, nw, w, sw, se")
    print("  e           - End turn")
    print("  u           - Undo last move")
    print("  s           - Show board")
    print("  v           - Show all valid actions")
    print("  q           - Quit")
    print("  <number>    - Play action by ID")
    print("\n=== AI Configuration ===")
    print("  ai <0|1|both|none>      - Set which players are AI")
    print("  net                     - Interactive network selection")
    print("  net <0|1|both> <path>   - Load network directly")
    print("  time <0|1> <secs|off>  - Set thinking time limit")
    print("  nodes <0|1> <count|off> - Set node expansion limit")
    print("  temp <0|1> <value>     - Set temperature for move selection")
    print("  hints <0|1> <on|off>   - Show AI analysis on human turns")
    print("  auto                   - AI auto-plays when it's AI's turn")
    print("  manual                 - User picks from AI-annotated actions")
    print("  status                 - Show current configuration")


def select_unified_variant():
    """Select a variant when playing with unified (multi-size) network."""
    print("\nSelect variant for unified network:")
    print("  1. Skirmish (3F, 1C, 0D)")
    print("  2. Clash (3F, 2C, 1D)")
    print("  3. Battle (4F, 3C, 2D)")

    variant_input = input("Variant (1/2/3) [1]: ").strip()
    if variant_input == '2':
        return alphazero.StarGambitVariant.CLASH
    elif variant_input == '3':
        return alphazero.StarGambitVariant.BATTLE
    return alphazero.StarGambitVariant.SKIRMISH


def main():
    print("=== Star Gambit Interactive Player ===\n")

    # Game size selection
    print("Select game size:")
    print("  1. Skirmish (3F, 1C, 0D) - single size")
    print("  2. Clash (3F, 2C, 1D) - single size")
    print("  3. Battle (4F, 3C, 2D) - single size")
    print("  4. Unified (multi-size network) - plays any variant")

    size_input = input("Size (1/2/3/4) [1]: ").strip()
    if size_input == '2':
        game = alphazero.StarGambitClashGS()
        game_class = alphazero.StarGambitClashGS
        cfg = CLASH
    elif size_input == '3':
        game = alphazero.StarGambitBattleGS()
        game_class = alphazero.StarGambitBattleGS
        cfg = BATTLE
    elif size_input == '4':
        # Unified mode - select variant then create unified game
        variant = select_unified_variant()
        game = alphazero.StarGambitUnifiedGS(variant)
        game_class = alphazero.StarGambitUnifiedGS
        cfg = UNIFIED
        print(f"Playing {game.get_variant_name()} variant with unified network")
    else:
        game = alphazero.StarGambitSkirmishGS()
        game_class = alphazero.StarGambitSkirmishGS
        cfg = SKIRMISH

    # Create game context
    ctx = GameContext(game, cfg, game_class)

    print(f"\nAction space: {cfg.num_moves} actions")
    print("\nNotation:")
    print("  m f1 f    - move fighter 1 forward")
    print("  m f1 fl   - move fighter 1 forward-left")
    print("  m c1 l    - move cruiser 1 rotate-left")
    print("  f f1      - fire fighter 1")
    print("  f c1 l    - fire cruiser 1 left cannon")
    print("  d f ne    - deploy fighter facing northeast")
    print("  e         - end turn")

    # Mode selection
    print("\nSelect mode:")
    print("  1. Human vs Human")
    print("  2. Human (P0) vs AI (P1)")
    print("  3. AI (P0) vs Human (P1)")
    print("  4. AI vs AI (watch)")

    mode = input("Mode (1/2/3/4) [2]: ").strip()
    if mode == '1':
        ctx.players[0].is_ai = False
        ctx.players[1].is_ai = False
    elif mode == '3':
        ctx.players[0].is_ai = True
        ctx.players[1].is_ai = False
    elif mode == '4':
        ctx.players[0].is_ai = True
        ctx.players[1].is_ai = True
        ctx.auto_play = True
    else:  # Default: mode 2
        ctx.players[0].is_ai = False
        ctx.players[1].is_ai = True

    # Network selection and configuration phase
    if ctx.players[0].is_ai or ctx.players[1].is_ai:
        # Interactive network selection first
        select_network_interactive(ctx)

        print("\n=== AI Configuration ===")
        print("Adjust settings (or 'start'/Enter to begin):")
        print("  net                      - Re-select network")
        print("  net <0|1> <path|random>  - Load specific network")
        print("  nodes <0|1> <count|off>  - Node limit (default: 100)")
        print("  time <0|1> <secs|off>    - Time limit")
        print("  temp <0|1> <value>       - Temperature (default: 0.5)")
        print("  hints <0|1> <on|off>     - AI analysis for human player")
        print("  status                   - Show current config")
        print_status(ctx)

        while True:
            cmd = input("\nConfig> ").strip()
            if cmd.lower() in ['', 'start', 'go', 'begin']:
                break
            # Use empty valids array since we're not playing moves yet
            dummy_valids = np.zeros(cfg.num_moves)
            result = parse_command(cmd, dummy_valids, cfg, ctx)
            if result == 'quit':
                print("Goodbye!")
                return
            if result == 'help':
                print_help()
            elif result == 'status':
                print_status(ctx)
            elif result == 'auto':
                ctx.auto_play = True
                print("Auto-play mode enabled")
            elif result == 'manual':
                ctx.auto_play = False
                print("Manual mode enabled")
            # 'config' result means a config command was processed

        print("\nStarting game...")

    # History stack for undo
    history = []

    while True:
        print("\n" + "=" * 50)
        print(ctx.game)
        print_unit_lists(ctx.game)

        scores = ctx.game.scores()
        if scores is not None:
            scores_arr = np.array(scores)
            if scores_arr[0] == 1:
                print("\nPlayer 0 (Red) wins!")
            elif scores_arr[1] == 1:
                print("\nPlayer 1 (Blue) wins!")
            else:
                print("\nDraw!")
            break

        valids = np.array(ctx.game.valid_moves())
        current = ctx.game.current_player()
        pcfg = ctx.players[current]

        if pcfg.is_ai:
            # AI turn
            if ctx.auto_play:
                action, probs, source, sims, wld = get_ai_move(ctx, current, valids)
                if action is None:
                    print("No valid moves for AI!")
                    break

                # Show full action menu with probabilities
                print(f"\nAI (P{current}) [{source}]")
                print_actions_menu(valids, cfg, ctx.game, probs=probs, source=source, wld=wld, show_commands=False)

                # Pause in AI vs AI mode before revealing the chosen move
                if ctx.players[0].is_ai and ctx.players[1].is_ai:
                    input("Press Enter to see move...")

                # Show chosen action and play it
                print(f"\n>>> Plays: {format_action(action, cfg, ctx.game)}  [{probs[action]*100:.1f}%]")
                history.append(ctx.game.copy())
                ctx.game.play_move(action)
            else:
                # Manual mode: show AI probabilities, let user pick
                probs, source, sims, wld = get_ai_probs(ctx, current, valids)
                print(f"\nAI (P{current}) suggests [{source}]:")
                print_actions_menu(valids, cfg, ctx.game, probs=probs, source=source, wld=wld)

                while True:
                    cmd = input(f"\nPlayer {current} (AI-assisted) move: ").strip()
                    result = parse_command(cmd, valids, cfg, ctx)

                    if result == 'quit':
                        print("Goodbye!")
                        return
                    if result == 'help':
                        print_help()
                        continue
                    if result == 'show':
                        print(ctx.game)
                        continue
                    if result == 'valid':
                        for i in range(cfg.num_moves):
                            if valids[i] == 1:
                                prob_str = f"  [{probs[i]*100:5.1f}%]" if probs is not None else ""
                                print(f"  {i}: {format_action(i, cfg, ctx.game)}{prob_str}")
                        continue
                    if result == 'undo':
                        if history:
                            ctx.game = history.pop()
                            print("Move undone")
                            break
                        else:
                            print("No moves to undo")
                        continue
                    if result == 'status':
                        print_status(ctx)
                        continue
                    if result == 'auto':
                        ctx.auto_play = True
                        print("Auto-play mode enabled")
                        break  # Let AI auto-play now
                    if result == 'manual':
                        ctx.auto_play = False
                        print("Manual mode enabled")
                        continue
                    if result == 'config':
                        continue
                    if result is not None:
                        print(f"Playing: {format_action(result, cfg, ctx.game)}")
                        history.append(ctx.game.copy())
                        ctx.game.play_move(result)
                        ctx.players[current].mcts = None  # Reset MCTS
                        break
        else:
            # Human turn - show AI probabilities only if hints enabled
            probs = None
            source = None
            wld = None
            if pcfg.show_hints and (pcfg.network is not None or (pcfg.node_limit and pcfg.node_limit > 0)):
                probs, source, _, wld = get_ai_probs(ctx, current, valids)

            print_actions_menu(valids, cfg, ctx.game, probs=probs, source=source, wld=wld)

            while True:
                cmd = input(f"\nPlayer {current} move: ").strip()
                result = parse_command(cmd, valids, cfg, ctx)

                if result == 'quit':
                    print("Goodbye!")
                    return
                if result == 'help':
                    print_help()
                    continue
                if result == 'show':
                    print(ctx.game)
                    continue
                if result == 'valid':
                    for i in range(cfg.num_moves):
                        if valids[i] == 1:
                            prob_str = f"  [{probs[i]*100:5.1f}%]" if probs is not None else ""
                            print(f"  {i}: {format_action(i, cfg, ctx.game)}{prob_str}")
                    continue
                if result == 'undo':
                    if history:
                        ctx.game = history.pop()
                        print("Move undone")
                        break
                    else:
                        print("No moves to undo")
                    continue
                if result == 'status':
                    print_status(ctx)
                    continue
                if result == 'auto':
                    ctx.auto_play = True
                    print("Auto-play mode enabled")
                    continue
                if result == 'manual':
                    ctx.auto_play = False
                    print("Manual mode enabled")
                    continue
                if result == 'config':
                    continue
                if result is not None:
                    print(f"Playing: {format_action(result, cfg, ctx.game)}")
                    history.append(ctx.game.copy())
                    ctx.game.play_move(result)
                    break


if __name__ == "__main__":
    main()
