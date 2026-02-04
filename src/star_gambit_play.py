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
import readline  # Enable line editing (backspace, arrow keys) in input()

try:
    import torch
    import neural_net
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: torch/neural_net not available. AI features disabled.")

# ANSI color codes
RED = '\033[91m'
BLUE = '\033[94m'
RESET = '\033[0m'

# Game size selection - using Skirmish by default
# Skirmish: 3F, 1C, 0D, 5-side board, 39 actions
# Clash: 3F, 2C, 1D, 5-side board, 55 actions
# Battle: 4F, 3C, 2D, 6-side board, 75 actions

# Skirmish action space layout (39 total):
#   Fighter moves:  0-8   (3 fighters * 3 directions)
#   Cruiser moves:  9-13  (1 cruiser * 5 directions)
#   Dread moves:    14-13 (0 dreadnoughts)
#   Fighter fire:   14-16 (3 fighters * 1 cannon)
#   Cruiser fire:   17-19 (1 cruiser * 3 cannons)
#   Dread fire:     20-19 (0 dreadnoughts)
#   Deploy:         20-37 (3 types * 6 facings)
#   End turn:       38

class GameConfig:
    """Action space configuration for a game size."""
    def __init__(self, max_fighters, max_cruisers, max_dreads):
        self.max_fighters = max_fighters
        self.max_cruisers = max_cruisers
        self.max_dreads = max_dreads

        # Movement directions per unit type
        self.fighter_dirs = 3  # f, fl, fr
        self.cruiser_dirs = 5  # l, fl, f, fr, r
        self.dread_dirs = 4    # l, fl, fr, r

        # Cannons per unit type
        self.fighter_cannons = 1
        self.cruiser_cannons = 3
        self.dread_cannons = 4

        # Action counts
        self.fighter_move_actions = max_fighters * self.fighter_dirs
        self.cruiser_move_actions = max_cruisers * self.cruiser_dirs
        self.dread_move_actions = max_dreads * self.dread_dirs
        self.fighter_fire_actions = max_fighters * self.fighter_cannons
        self.cruiser_fire_actions = max_cruisers * self.cruiser_cannons
        self.dread_fire_actions = max_dreads * self.dread_cannons
        self.deploy_actions = 18  # 3 types * 6 facings

        # Action offsets
        self.fighter_move_offset = 0
        self.cruiser_move_offset = self.fighter_move_offset + self.fighter_move_actions
        self.dread_move_offset = self.cruiser_move_offset + self.cruiser_move_actions
        self.fighter_fire_offset = self.dread_move_offset + self.dread_move_actions
        self.cruiser_fire_offset = self.fighter_fire_offset + self.fighter_fire_actions
        self.dread_fire_offset = self.cruiser_fire_offset + self.cruiser_fire_actions
        self.deploy_offset = self.dread_fire_offset + self.dread_fire_actions
        self.end_turn_offset = self.deploy_offset + self.deploy_actions
        self.num_moves = self.end_turn_offset + 1


SKIRMISH = GameConfig(3, 1, 0)
CLASH = GameConfig(3, 2, 1)
BATTLE = GameConfig(4, 3, 2)

# Direction names
DIRECTION_NAMES = ['E', 'NE', 'NW', 'W', 'SW', 'SE']
FIGHTER_MOVE_NAMES = ['f', 'fl', 'fr']
CRUISER_MOVE_NAMES = ['l', 'fl', 'f', 'fr', 'r']
DREAD_MOVE_NAMES = ['l', 'fl', 'fr', 'r']
CRUISER_CANNON_NAMES = ['l', 'f', 'r']
DREAD_CANNON_NAMES = ['rr', 'fr', 'fl', 'rl']

# Detailed direction names for display
FIGHTER_MOVE_DETAIL = ['forward', 'forward-left', 'forward-right']
CRUISER_MOVE_DETAIL = ['rotate-left', 'forward-left', 'forward', 'forward-right', 'rotate-right']
DREAD_MOVE_DETAIL = ['rotate-left', 'forward-left', 'forward-right', 'rotate-right']
CRUISER_CANNON_DETAIL = ['left cannon', 'forward cannon', 'right cannon']
DREAD_CANNON_DETAIL = ['rear-right cannon', 'front-right cannon', 'front-left cannon', 'rear-left cannon']

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
    Returns (visit_counts, num_simulations).
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
            canonical = torch.from_numpy(np.array(leaf.canonicalized())).unsqueeze(0)
            v, pi = agent.predict(canonical)
            v = v.cpu().numpy().flatten()
            pi = pi.cpu().numpy().flatten()

        mcts.process_result(gs, v, pi, sims == 0)  # Add noise on first sim
        sims += 1

    counts = np.array(mcts.counts())
    return counts, sims


def get_ai_probs(ctx, player_idx, valids):
    """
    Get AI move probabilities for display.
    Returns (probs, source_str, sims) where probs are temperature-adjusted and normalized to valid moves.
    """
    pcfg = ctx.players[player_idx]

    # Create or reuse MCTS
    if pcfg.mcts is None:
        pcfg.mcts = create_mcts(ctx.game_class, ctx.cpuct, ctx.fpu_reduction)

    # Determine if we should run MCTS or just use raw policy
    should_search = (pcfg.think_time is not None and pcfg.think_time > 0) or \
                   (pcfg.node_limit is not None and pcfg.node_limit > 0)

    if should_search and (pcfg.network is not None or TORCH_AVAILABLE):
        # Run MCTS search
        counts, sims = run_mcts_search(
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
            canonical = torch.from_numpy(np.array(ctx.game.canonicalized())).unsqueeze(0)
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

    return probs, source, sims


def get_ai_move(ctx, player_idx, valids):
    """Get AI move for the specified player. Returns (action, probs, source, sims)."""
    probs, source, sims = get_ai_probs(ctx, player_idx, valids)

    # Sample from distribution
    action = np.random.choice(len(probs), p=probs)

    # Reset MCTS for next move (don't reuse tree)
    ctx.players[player_idx].mcts = None

    return action, probs, source, sims


def color_unit(name, player):
    """Wrap unit name in ANSI color codes based on player."""
    color = RED if player == 0 else BLUE
    return f"{color}{name}{RESET}"


def get_unit_name(unit_type, slot):
    """Get unit name like F1, C2, etc."""
    return f"{TYPE_CHARS[unit_type]}{slot + 1}"


def decode_action(action, cfg):
    """Decode action ID to human-readable description."""
    if action < cfg.cruiser_move_offset:
        # Fighter move
        idx = action - cfg.fighter_move_offset
        slot = idx // cfg.fighter_dirs
        dir_idx = idx % cfg.fighter_dirs
        return f"m f{slot+1} {FIGHTER_MOVE_NAMES[dir_idx]}"
    elif action < cfg.dread_move_offset:
        # Cruiser move
        idx = action - cfg.cruiser_move_offset
        slot = idx // cfg.cruiser_dirs
        dir_idx = idx % cfg.cruiser_dirs
        return f"m c{slot+1} {CRUISER_MOVE_NAMES[dir_idx]}"
    elif action < cfg.fighter_fire_offset:
        # Dreadnought move
        idx = action - cfg.dread_move_offset
        slot = idx // cfg.dread_dirs
        dir_idx = idx % cfg.dread_dirs
        return f"m d{slot+1} {DREAD_MOVE_NAMES[dir_idx]}"
    elif action < cfg.cruiser_fire_offset:
        # Fighter fire
        idx = action - cfg.fighter_fire_offset
        return f"f f{idx+1}"
    elif action < cfg.dread_fire_offset:
        # Cruiser fire
        idx = action - cfg.cruiser_fire_offset
        slot = idx // cfg.cruiser_cannons
        cannon = idx % cfg.cruiser_cannons
        return f"f c{slot+1} {CRUISER_CANNON_NAMES[cannon]}"
    elif action < cfg.deploy_offset:
        # Dreadnought fire
        idx = action - cfg.dread_fire_offset
        slot = idx // cfg.dread_cannons
        cannon = idx % cfg.dread_cannons
        return f"f d{slot+1} {DREAD_CANNON_NAMES[cannon]}"
    elif action < cfg.end_turn_offset:
        # Deploy
        idx = action - cfg.deploy_offset
        unit_type = idx // 6
        facing = idx % 6
        type_chars = ['f', 'c', 'd']
        facing_names = ['e', 'ne', 'nw', 'w', 'sw', 'se']
        return f"d {type_chars[unit_type]} {facing_names[facing]}"
    else:
        return "e"


def format_action(action, cfg, game):
    """Format action with short form and detailed description, with colored unit names."""
    short = decode_action(action, cfg)
    current_player = game.current_player()

    # Build unit lookup from game state
    units = game.get_units()
    unit_lookup = {}  # (player, type, slot) -> UnitInfo
    for u in units:
        unit_lookup[(u.player, u.type, u.slot)] = u

    if action < cfg.cruiser_move_offset:
        # Fighter move
        idx = action - cfg.fighter_move_offset
        slot = idx // cfg.fighter_dirs
        dir_idx = idx % cfg.fighter_dirs
        name = color_unit(get_unit_name(0, slot), current_player)
        detail = FIGHTER_MOVE_DETAIL[dir_idx]
        return f"{short:<10} -  {name} {detail}"

    elif action < cfg.dread_move_offset:
        # Cruiser move
        idx = action - cfg.cruiser_move_offset
        slot = idx // cfg.cruiser_dirs
        dir_idx = idx % cfg.cruiser_dirs
        name = color_unit(get_unit_name(1, slot), current_player)
        detail = CRUISER_MOVE_DETAIL[dir_idx]
        return f"{short:<10} -  {name} {detail}"

    elif action < cfg.fighter_fire_offset:
        # Dreadnought move
        idx = action - cfg.dread_move_offset
        slot = idx // cfg.dread_dirs
        dir_idx = idx % cfg.dread_dirs
        name = color_unit(get_unit_name(2, slot), current_player)
        detail = DREAD_MOVE_DETAIL[dir_idx]
        return f"{short:<10} -  {name} {detail}"

    elif action < cfg.cruiser_fire_offset:
        # Fighter fire
        idx = action - cfg.fighter_fire_offset
        slot = idx
        name = color_unit(get_unit_name(0, slot), current_player)
        fire_info = game.get_fire_info(action)
        if fire_info.has_target:
            target_name = color_unit(get_unit_name(fire_info.target_type, fire_info.target_slot),
                                     fire_info.target_player)
            return f"{short:<10} -  {name} cannon -> {target_name} ({fire_info.damage} dmg)"
        return f"{short:<10} -  {name} cannon"

    elif action < cfg.dread_fire_offset:
        # Cruiser fire
        idx = action - cfg.cruiser_fire_offset
        slot = idx // cfg.cruiser_cannons
        cannon = idx % cfg.cruiser_cannons
        name = color_unit(get_unit_name(1, slot), current_player)
        fire_info = game.get_fire_info(action)
        cannon_name = CRUISER_CANNON_DETAIL[cannon]
        if fire_info.has_target:
            target_name = color_unit(get_unit_name(fire_info.target_type, fire_info.target_slot),
                                     fire_info.target_player)
            return f"{short:<10} -  {name} {cannon_name} -> {target_name} ({fire_info.damage} dmg)"
        return f"{short:<10} -  {name} {cannon_name}"

    elif action < cfg.deploy_offset:
        # Dreadnought fire
        idx = action - cfg.dread_fire_offset
        slot = idx // cfg.dread_cannons
        cannon = idx % cfg.dread_cannons
        name = color_unit(get_unit_name(2, slot), current_player)
        fire_info = game.get_fire_info(action)
        cannon_name = DREAD_CANNON_DETAIL[cannon]
        if fire_info.has_target:
            target_name = color_unit(get_unit_name(fire_info.target_type, fire_info.target_slot),
                                     fire_info.target_player)
            return f"{short:<10} -  {name} {cannon_name} -> {target_name} ({fire_info.damage} dmg)"
        return f"{short:<10} -  {name} {cannon_name}"

    elif action < cfg.end_turn_offset:
        # Deploy
        idx = action - cfg.deploy_offset
        unit_type = idx // 6
        facing = idx % 6
        type_name = TYPE_NAMES[unit_type].lower()
        facing_name = DIRECTION_NAMES[facing]
        return f"{short:<10} -  Deploy {type_name} facing {facing_name}"

    else:
        return f"{short:<10} -  End turn"


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
        # net <0|1> <path|random>
        if len(parts) < 3:
            print("Usage: net <0|1> <path|random>")
            return 'config'
        try:
            player = int(parts[1])
            if player not in [0, 1]:
                raise ValueError()
        except ValueError:
            print("Player must be 0 or 1")
            return 'config'

        path_arg = parts[2]  # Keep original case for path
        if path_arg.lower() == 'random':
            ctx.players[player].network = None
            ctx.players[player].network_path = None
            ctx.players[player].mcts = None
            print(f"Player {player} using random policy")
        else:
            if not TORCH_AVAILABLE:
                print("Error: torch not available, cannot load networks")
                return 'config'
            try:
                # Try loading the network
                net = neural_net.NNWrapper.load_checkpoint(
                    ctx.game_class, os.path.dirname(path_arg), os.path.basename(path_arg)
                )
                ctx.players[player].network = net
                ctx.players[player].network_path = path_arg
                ctx.players[player].mcts = None
                print(f"Player {player} loaded network: {path_arg}")
            except Exception as e:
                print(f"Error loading network: {e}")
        return 'config'

    # Now lowercase for game commands
    parts = [p.lower() for p in parts]

    if parts[0] == 'e':
        # End turn
        if valids[cfg.end_turn_offset] == 1:
            return cfg.end_turn_offset
        print("End turn not valid!")
        return None

    if parts[0] == 'm' and len(parts) >= 3:
        # Move: m <unit><slot> <direction>
        unit_slot = parts[1]
        direction = parts[2]

        if len(unit_slot) < 2:
            print("Invalid unit. Use f1, f2, c1, d1, etc.")
            return None

        unit_type = unit_slot[0]
        try:
            slot = int(unit_slot[1:]) - 1  # Convert to 0-indexed
        except ValueError:
            print("Invalid slot number.")
            return None

        if unit_type == 'f':
            dir_map = {'f': 0, 'fl': 1, 'fr': 2}
            if direction not in dir_map or slot < 0 or slot >= cfg.max_fighters:
                print("Invalid fighter move.")
                return None
            action = cfg.fighter_move_offset + slot * cfg.fighter_dirs + dir_map[direction]
        elif unit_type == 'c':
            dir_map = {'l': 0, 'fl': 1, 'f': 2, 'fr': 3, 'r': 4}
            if direction not in dir_map or slot < 0 or slot >= cfg.max_cruisers:
                print("Invalid cruiser move.")
                return None
            action = cfg.cruiser_move_offset + slot * cfg.cruiser_dirs + dir_map[direction]
        elif unit_type == 'd':
            dir_map = {'l': 0, 'fl': 1, 'fr': 2, 'r': 3}
            if direction not in dir_map or slot < 0 or slot >= cfg.max_dreads:
                print("Invalid dreadnought move.")
                return None
            action = cfg.dread_move_offset + slot * cfg.dread_dirs + dir_map[direction]
        else:
            print("Invalid unit type. Use f, c, or d.")
            return None

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

        unit_type = unit_slot[0]
        try:
            slot = int(unit_slot[1:]) - 1
        except ValueError:
            print("Invalid slot number.")
            return None

        if unit_type == 'f':
            if slot < 0 or slot >= cfg.max_fighters:
                print("Invalid fighter slot.")
                return None
            action = cfg.fighter_fire_offset + slot
        elif unit_type == 'c':
            cannon_map = {'l': 0, 'f': 1, 'r': 2}
            if cannon not in cannon_map or slot < 0 or slot >= cfg.max_cruisers:
                print("Invalid cruiser fire. Specify cannon: l, f, r")
                return None
            action = cfg.cruiser_fire_offset + slot * cfg.cruiser_cannons + cannon_map[cannon]
        elif unit_type == 'd':
            cannon_map = {'rr': 0, 'fr': 1, 'fl': 2, 'rl': 3}
            if cannon not in cannon_map or slot < 0 or slot >= cfg.max_dreads:
                print("Invalid dreadnought fire. Specify cannon: rr, fr, fl, rl")
                return None
            action = cfg.dread_fire_offset + slot * cfg.dread_cannons + cannon_map[cannon]
        else:
            print("Invalid unit type. Use f, c, or d.")
            return None

        if valids[action] == 1:
            return action
        print(f"Fire not valid! (action {action})")
        return None

    if parts[0] == 'd' and len(parts) >= 3:
        # Deploy: d <type> <facing>
        unit_type = parts[1]
        facing = parts[2]

        type_map = {'f': 0, 'c': 1, 'd': 2}
        facing_map = {'e': 0, 'ne': 1, 'nw': 2, 'w': 3, 'sw': 4, 'se': 5}

        if unit_type not in type_map:
            print("Invalid unit type. Use f, c, or d.")
            return None
        if facing not in facing_map:
            print("Invalid facing. Use e, ne, nw, w, sw, se.")
            return None

        action = cfg.deploy_offset + type_map[unit_type] * 6 + facing_map[facing]
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


def print_actions_menu(valids, cfg, game, probs=None, source=None, show_commands=True):
    """Print a menu of available actions with detailed descriptions and optional probabilities."""
    print("\n=== Available Actions ===")

    if source:
        print(f"  (probabilities from {source})")

    def format_with_prob(action):
        base = format_action(action, cfg, game)
        if probs is not None and probs[action] > 0.001:
            return f"{base}  [{probs[action]*100:5.1f}%]"
        elif probs is not None:
            return f"{base}  [  0.0%]"
        return base

    # Group valid actions (keep original order within groups)
    moves = []
    fires = []
    deploys = []
    end_turn = False

    for i in range(cfg.num_moves):
        if valids[i] != 1:
            continue
        if i < cfg.fighter_fire_offset:
            moves.append(i)
        elif i < cfg.deploy_offset:
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
    print("  ai <0|1|both|none>     - Set which players are AI")
    print("  net <0|1> <path|random> - Load network or use random policy")
    print("  time <0|1> <secs|off>  - Set thinking time limit")
    print("  nodes <0|1> <count|off> - Set node expansion limit")
    print("  temp <0|1> <value>     - Set temperature for move selection")
    print("  hints <0|1> <on|off>   - Show AI analysis on human turns")
    print("  auto                   - AI auto-plays when it's AI's turn")
    print("  manual                 - User picks from AI-annotated actions")
    print("  status                 - Show current configuration")


def main():
    print("=== Star Gambit Interactive Player ===\n")

    # Game size selection
    print("Select game size:")
    print("  1. Skirmish (3F, 1C, 0D) - 39 actions")
    print("  2. Clash (3F, 2C, 1D) - 55 actions")
    print("  3. Battle (4F, 3C, 2D) - 75 actions")

    size_input = input("Size (1/2/3) [1]: ").strip()
    if size_input == '2':
        game = alphazero.StarGambitClashGS()
        game_class = alphazero.StarGambitClashGS
        cfg = CLASH
    elif size_input == '3':
        game = alphazero.StarGambitBattleGS()
        game_class = alphazero.StarGambitBattleGS
        cfg = BATTLE
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

    # Configuration phase - allow setting up AI before starting
    if ctx.players[0].is_ai or ctx.players[1].is_ai:
        print("\n=== AI Configuration ===")
        print("Configure AI settings before starting (type 'start' or press Enter to begin):")
        print("  net <0|1> <path|random>  - Load network")
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
                action, probs, source, sims = get_ai_move(ctx, current, valids)
                if action is None:
                    print("No valid moves for AI!")
                    break

                # Show full action menu with probabilities
                print(f"\nAI (P{current}) [{source}]")
                print_actions_menu(valids, cfg, ctx.game, probs=probs, source=source, show_commands=False)

                # Pause in AI vs AI mode before revealing the chosen move
                if ctx.players[0].is_ai and ctx.players[1].is_ai:
                    input("Press Enter to see move...")

                # Show chosen action and play it
                print(f"\n>>> Plays: {format_action(action, cfg, ctx.game)}  [{probs[action]*100:.1f}%]")
                history.append(ctx.game.copy())
                ctx.game.play_move(action)
            else:
                # Manual mode: show AI probabilities, let user pick
                probs, source, sims = get_ai_probs(ctx, current, valids)
                print(f"\nAI (P{current}) suggests [{source}]:")
                print_actions_menu(valids, cfg, ctx.game, probs=probs, source=source)

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
            if pcfg.show_hints and (pcfg.network is not None or (pcfg.node_limit and pcfg.node_limit > 0)):
                probs, source, _ = get_ai_probs(ctx, current, valids)

            print_actions_menu(valids, cfg, ctx.game, probs=probs, source=source)

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
