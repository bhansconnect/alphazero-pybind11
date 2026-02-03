#!/usr/bin/env python3
"""Interactive Star Gambit player for testing and validation."""

import alphazero
import numpy as np

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
DREAD_CANNON_NAMES = ['rl', 'fl', 'fr', 'rr']


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
        return "e (end turn)"


def parse_command(cmd, valids, cfg):
    """Parse user command and return action ID or None."""
    parts = cmd.strip().lower().split()
    if not parts:
        return None

    if parts[0] == 'q':
        return 'quit'
    if parts[0] == 'h':
        return 'help'
    if parts[0] == 's':
        return 'show'
    if parts[0] == 'v':
        return 'valid'

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
            cannon_map = {'rl': 0, 'fl': 1, 'fr': 2, 'rr': 3}
            if cannon not in cannon_map or slot < 0 or slot >= cfg.max_dreads:
                print("Invalid dreadnought fire. Specify cannon: rl, fl, fr, rr")
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


def print_actions_menu(valids, cfg):
    """Print a menu of available actions."""
    print("\n=== Available Actions ===")

    # Group valid actions
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
            print(f"  {decode_action(action, cfg)}")

    if fires:
        print("\nFire:")
        for action in fires:
            print(f"  {decode_action(action, cfg)}")

    if deploys:
        print("\nDeploy:")
        for action in deploys:
            print(f"  {decode_action(action, cfg)}")

    if end_turn:
        print("\nEnd turn: e")

    print("\nCommands: (q)uit, (h)elp, (s)how board, (v)alid actions")


def play_random_action(valids):
    """Play a random valid action."""
    valid_indices = [i for i in range(len(valids)) if valids[i] == 1]
    if not valid_indices:
        return None
    return np.random.choice(valid_indices)


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
        cfg = CLASH
    elif size_input == '3':
        game = alphazero.StarGambitBattleGS()
        cfg = BATTLE
    else:
        game = alphazero.StarGambitSkirmishGS()
        cfg = SKIRMISH

    print(f"\nAction space: {cfg.num_moves} actions")
    print("\nNotation:")
    print("  m f1 f    - move fighter 1 forward")
    print("  m f1 fl   - move fighter 1 forward-left")
    print("  m c1 l    - move cruiser 1 rotate-left")
    print("  f f1      - fire fighter 1")
    print("  f c1 l    - fire cruiser 1 left cannon")
    print("  d f se    - deploy fighter facing southeast")
    print("  e         - end turn")

    # Mode selection
    print("\nSelect mode:")
    print("  1. Human vs Human")
    print("  2. Human (P0) vs Random AI")
    print("  3. Random AI vs Random AI (watch)")

    mode = input("Mode (1/2/3) [2]: ").strip()
    if mode not in ['1', '2', '3']:
        mode = '2'

    human_players = set()
    if mode == '1':
        human_players = {0, 1}
    elif mode == '2':
        human_players = {0}

    while True:
        print("\n" + "=" * 50)
        print(game)

        scores = game.scores()
        if scores is not None:
            scores_arr = np.array(scores)
            if scores_arr[0] == 1:
                print("Player 0 wins!")
            elif scores_arr[1] == 1:
                print("Player 1 wins!")
            else:
                print("Draw!")
            break

        valids = game.valid_moves()
        current = game.current_player()

        if current in human_players:
            print_actions_menu(valids, cfg)

            while True:
                cmd = input(f"\nPlayer {current} move: ").strip()
                result = parse_command(cmd, valids, cfg)

                if result == 'quit':
                    print("Goodbye!")
                    return
                if result == 'help':
                    print("\nCommands:")
                    print("  m <unit><slot> <dir>  - Move (e.g., m f1 f, m c1 fl)")
                    print("    Fighter dirs: f, fl, fr")
                    print("    Cruiser dirs: l, fl, f, fr, r")
                    print("    Dread dirs: l, fl, fr, r")
                    print("  f <unit><slot> [cannon] - Fire (e.g., f f1, f c1 l)")
                    print("  d <type> <facing>       - Deploy (e.g., d f se)")
                    print("    Types: f, c, d")
                    print("    Facings: e, ne, nw, w, sw, se")
                    print("  e           - End turn")
                    print("  s           - Show board")
                    print("  v           - Show all valid actions")
                    print("  q           - Quit")
                    print("  <number>    - Play action by ID")
                    continue
                if result == 'show':
                    print(game)
                    continue
                if result == 'valid':
                    for i in range(cfg.num_moves):
                        if valids[i] == 1:
                            print(f"  {i}: {decode_action(i, cfg)}")
                    continue
                if result is not None:
                    print(f"Playing: {decode_action(result, cfg)}")
                    game.play_move(result)
                    break
        else:
            # AI turn
            action = play_random_action(valids)
            if action is None:
                print("No valid moves for AI!")
                break
            print(f"AI plays: {decode_action(action, cfg)}")
            game.play_move(action)

            if mode == '3':
                input("Press Enter to continue...")


if __name__ == "__main__":
    main()
