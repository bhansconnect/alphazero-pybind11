"""Star Gambit game UI for the unified play agent.

Contains all Star Gambit display constants, action encoding/decoding,
input parsing, and the StarGambitUI class that plugs into the generic
play agent framework.
"""

import numpy as np

from game_ui import GameUI

# ---------------------------------------------------------------------------
# ANSI color codes
# ---------------------------------------------------------------------------
RED = '\033[91m'
BLUE = '\033[94m'
RESET = '\033[0m'

# ---------------------------------------------------------------------------
# Action space constants
# ---------------------------------------------------------------------------

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
    0: {0, 1, 2, 5},        # Fighter
    1: {0, 1, 2, 3, 4, 5, 6, 7},  # Cruiser
    2: {1, 2, 3, 4, 6, 7, 8, 9},  # Dreadnought
}

# Direction / move / cannon name tables
DIRECTION_NAMES = ['E', 'NE', 'NW', 'W', 'SW', 'SE']
FIGHTER_MOVE_NAMES = ['f', 'fl', 'fr']
CRUISER_MOVE_NAMES = ['l', 'fl', 'f', 'fr', 'r']
DREAD_MOVE_NAMES = ['l', 'fl', 'fr', 'r']
CRUISER_CANNON_NAMES = ['l', 'f', 'r']
DREAD_CANNON_NAMES = ['rl', 'fl', 'fr', 'rr']

FIGHTER_MOVE_DETAIL = ['forward', 'forward-left', 'forward-right']
CRUISER_MOVE_DETAIL = ['rotate-left', 'forward-left', 'forward', 'forward-right', 'rotate-right']
DREAD_MOVE_DETAIL = ['rotate-left', 'forward-left', 'forward-right', 'rotate-right']
CRUISER_CANNON_DETAIL = ['left cannon', 'forward cannon', 'right cannon']
DREAD_CANNON_DETAIL = ['rear-left cannon', 'front-left cannon', 'front-right cannon', 'rear-right cannon']

# Type names for display
TYPE_NAMES = ['Fighter', 'Cruiser', 'Dreadnought', 'Portal']
TYPE_CHARS = ['F', 'C', 'D', 'P']


# ---------------------------------------------------------------------------
# GameConfig - action space configuration for a game size
# ---------------------------------------------------------------------------

class GameConfig:
    """Action space configuration for a game size (2D grid-based encoding)."""
    def __init__(self, max_fighters, max_cruisers, max_dreads, board_side):
        self.max_fighters = max_fighters
        self.max_cruisers = max_cruisers
        self.max_dreads = max_dreads
        self.board_side = board_side
        self.board_dim = 2 * board_side + 1

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
        bs = self.board_side
        return q + bs, r + bs

    def rowcol_to_hex(self, row, col):
        """Convert grid (row, col) to hex (q, r)."""
        bs = self.board_side
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


# ---------------------------------------------------------------------------
# Display helper functions
# ---------------------------------------------------------------------------

def color_unit(name, player):
    """Wrap unit name in ANSI color codes based on player."""
    color = RED if player == 0 else BLUE
    return f"{color}{name}{RESET}"


def get_unit_name(unit_type, slot):
    """Get unit name like F1, C2, etc."""
    return f"{TYPE_CHARS[unit_type]}{slot + 1}"


def find_unit_at_position(game, row, col, cfg, current_player):
    """Find unit at grid position. De-canonicalizes for P1. Returns UnitInfo or None."""
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
    """Decode action ID to human-readable short command (2D grid-based encoding)."""
    current_player = game.current_player() if game is not None else 0

    # Deploy actions
    if cfg.deploy_offset <= action < cfg.end_turn_offset:
        unit_type, facing = cfg.decode_deploy(action)
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

        if current_player == 1:
            disp_row, disp_col, disp_slot = cfg.decanon_spatial(row, col, slot)
        else:
            disp_row, disp_col, disp_slot = row, col, slot

        unit = None
        if game is not None:
            unit = find_unit_at_position(game, row, col, cfg, current_player)

        if unit is not None:
            unit_name = f"{TYPE_CHARS[unit.type].lower()}{unit.slot + 1}"
        else:
            q, r = cfg.rowcol_to_hex(disp_row, disp_col)
            unit_name = f"({q},{r})"

        if disp_slot <= 4:
            move_names = {0: 'f', 1: 'fl', 2: 'fr', 3: 'l', 4: 'r'}
            return f"m {unit_name} {move_names.get(disp_slot, '?')}"
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

        if current_player == 1:
            disp_row, disp_col, disp_slot = cfg.decanon_spatial(row, col, slot)
        else:
            disp_row, disp_col, disp_slot = row, col, slot

        unit = find_unit_at_position(game, row, col, cfg, current_player)

        if unit is None:
            q, r = cfg.rowcol_to_hex(disp_row, disp_col)
            return f"{short:<12} -  (no unit at ({q},{r}))"

        name = color_unit(get_unit_name(unit.type, unit.slot), current_player)

        if disp_slot <= 4:
            move_detail = {
                0: 'forward', 1: 'forward-left', 2: 'forward-right',
                3: 'rotate-left', 4: 'rotate-right'
            }
            detail = move_detail.get(disp_slot, f'slot{disp_slot}')
            return f"{short:<12} -  {name} {detail}"
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

    # Group valid actions
    moves = []
    fires = []
    deploys = []
    end_turn = False

    for i in range(cfg.num_moves):
        if valids[i] != 1:
            continue
        if i < cfg.spatial_actions:
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


# ---------------------------------------------------------------------------
# Input parsing
# ---------------------------------------------------------------------------

def parse_command(cmd, valids, cfg, ctx=None):
    """Parse user command and return action ID or special command string.

    Returns:
        int: action ID for a valid game move
        str: special command ('quit', 'help', 'show', 'valid', 'undo',
             'status', 'auto', 'manual', 'config')
        None: unrecognized / invalid
    """
    parts = cmd.strip().split()
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
            slot = int(unit_slot[1:]) - 1
        except ValueError:
            print("Invalid slot number.")
            return None

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

        row, col = cfg.hex_to_rowcol(unit.anchor_q, unit.anchor_r)

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

        row, col = cfg.hex_to_rowcol(unit.anchor_q, unit.anchor_r)

        if unit_type == 0:  # Fighter - only forward fire (slot 5)
            if cannon is not None and cannon != 'f':
                print("Fighter only has forward cannon.")
                return None
            fire_slot = 5
        elif unit_type == 1:  # Cruiser
            cannon_map = {'l': 6, 'f': 5, 'r': 7, 'fl': 6, 'fr': 7}
            if cannon not in cannon_map:
                print("Invalid cruiser cannon. Use l, f, r (or fl, fr).")
                return None
            fire_slot = cannon_map[cannon]
        else:  # Dreadnought
            cannon_map = {'fl': 6, 'fr': 7, 'rl': 8, 'rr': 9}
            if cannon not in cannon_map:
                print("Invalid dreadnought cannon. Use fl, fr, rl, rr.")
                return None
            fire_slot = cannon_map[cannon]

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


def print_help():
    """Print Star Gambit game help."""
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


# ---------------------------------------------------------------------------
# StarGambitUI - pluggable GameUI for the unified play agent
# ---------------------------------------------------------------------------

VARIANT_MAP = {
    "1": "star_gambit_skirmish",
    "2": "star_gambit_clash",
    "3": "star_gambit_battle",
}


class StarGambitUI(GameUI):
    """Rich UI for Star Gambit with hex grid, unit selection, move naming."""

    def __init__(self, cfg: GameConfig):
        self.cfg = cfg

    def display_board(self, gs) -> str:
        """Return board string and unit lists."""
        lines = [str(gs)]
        import io
        import contextlib
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            print_unit_lists(gs)
        lines.append(f.getvalue())
        return "\n".join(lines)

    def parse_move(self, gs, input_str, valid_moves) -> int | None:
        """Parse Star Gambit move commands (m f1 f, f c1 l, d f ne, e, etc.)."""
        ctx = type('Ctx', (), {'game': gs, 'players': [None, None]})()

        result = parse_command(input_str, valid_moves, self.cfg, ctx)

        # parse_command returns special strings for non-move commands
        if isinstance(result, str):
            return None
        return result

    def format_move(self, gs, action) -> str:
        """Return human-friendly move description."""
        return format_action(action, self.cfg, gs)

    def get_valid_move_descriptions(self, gs, valid_moves) -> list[tuple[int, str]]:
        """Return (action_id, description) for all valid moves."""
        result = []
        for i in range(len(valid_moves)):
            if valid_moves[i]:
                desc = format_action(i, self.cfg, gs)
                result.append((i, desc))
        return result

    def show_help(self, gs):
        """Print Star Gambit help."""
        print_help()

    def build_action_menu(self, gs, probs, valids, wld=None) -> list[tuple]:
        """Build grouped menu entries for Star Gambit action selection."""
        cfg = self.cfg
        entries = []

        if wld is not None:
            entries.append(("info", f"Win: {wld[0]*100:.1f}%  Loss: {wld[1]*100:.1f}%  Draw: {wld[2]*100:.1f}%"))

        def format_with_prob(action):
            base = format_action(action, cfg, gs)
            if probs is not None and probs[action] > 0.001:
                return f"{base}  [{probs[action]*100:5.1f}%]"
            elif probs is not None:
                return f"{base}  [  0.0%]"
            return base

        # Group valid actions
        moves = []
        fires = []
        deploys = []
        end_turn = None

        for i in range(cfg.num_moves):
            if valids[i] != 1:
                continue
            if i < cfg.spatial_actions:
                _, _, slot = cfg.decode_spatial(i)
                if slot <= 4:
                    moves.append(i)
                else:
                    fires.append(i)
            elif i < cfg.end_turn_offset:
                deploys.append(i)
            else:
                end_turn = i

        if moves:
            entries.append(("header", "Moves:"))
            for action in moves:
                entries.append(("action", action, format_with_prob(action)))

        if fires:
            entries.append(("header", "Fire:"))
            for action in fires:
                entries.append(("action", action, format_with_prob(action)))

        if deploys:
            entries.append(("header", "Deploy:"))
            for action in deploys:
                entries.append(("action", action, format_with_prob(action)))

        if end_turn is not None:
            entries.append(("action", end_turn, format_with_prob(end_turn)))

        return entries

    def display_actions_menu(self, gs, probs, valids, wld=None, top_n=5):
        """Grouped action menu: Moves / Fire / Deploy / End Turn."""
        print_actions_menu(valids, self.cfg, gs, probs=probs, wld=wld)

    def select_variant(self) -> str | None:
        """Offer Skirmish/Clash/Battle selection."""
        print("Select variant:")
        print("  1. Skirmish (3F/1C)")
        print("  2. Clash (3F/2C/1D)")
        print("  3. Battle (4F/3C/2D)")
        choice = input("Variant [1]: ").strip() or "1"
        return VARIANT_MAP.get(choice, "star_gambit_skirmish")
