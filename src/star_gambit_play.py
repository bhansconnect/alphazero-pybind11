#!/usr/bin/env python3
"""Interactive Star Gambit player for testing and validation."""

import sys
sys.path.insert(0, 'build/src')
import alphazero
import numpy as np

# Action space constants
MOVES_PER_UNIT = 61 * 6  # NUM_HEXES * 6 facings
MAX_TOTAL_UNITS = 10
MOVE_ACTIONS = MAX_TOTAL_UNITS * MOVES_PER_UNIT
MOVE_OFFSET = 0
FIRE_OFFSET = MOVE_ACTIONS
MAX_CANNONS = 4
FIRE_ACTIONS = MAX_TOTAL_UNITS * MAX_CANNONS
DEPLOY_OFFSET = FIRE_OFFSET + FIRE_ACTIONS
DEPLOY_ACTIONS = 3 * 6  # 3 unit types * 6 facings
END_TURN_OFFSET = DEPLOY_OFFSET + DEPLOY_ACTIONS
NUM_MOVES = END_TURN_OFFSET + 1

# Direction names
DIRECTION_NAMES = ['E', 'NE', 'NW', 'W', 'SW', 'SE']
UNIT_TYPES = ['Fighter', 'Cruiser', 'Dreadnought']


def decode_action(action):
    """Decode action ID to human-readable description."""
    if action < FIRE_OFFSET:
        # Move action
        unit_idx = action // MOVES_PER_UNIT
        remainder = action % MOVES_PER_UNIT
        dest_hex = remainder // 6
        facing = remainder % 6
        return f"Move unit {unit_idx} to hex {dest_hex}, facing {DIRECTION_NAMES[facing]}"
    elif action < DEPLOY_OFFSET:
        # Fire action
        idx = action - FIRE_OFFSET
        unit_idx = idx // MAX_CANNONS
        cannon_idx = idx % MAX_CANNONS
        return f"Fire unit {unit_idx} cannon {cannon_idx}"
    elif action < END_TURN_OFFSET:
        # Deploy action
        idx = action - DEPLOY_OFFSET
        unit_type = idx // 6
        facing = idx % 6
        return f"Deploy {UNIT_TYPES[unit_type]} facing {DIRECTION_NAMES[facing]}"
    else:
        return "End turn"


def get_valid_actions_by_type(valids):
    """Group valid actions by type."""
    moves = []
    fires = []
    deploys = []
    end_turn = False

    for i in range(len(valids)):
        if valids[i] != 1:
            continue
        if i < FIRE_OFFSET:
            moves.append(i)
        elif i < DEPLOY_OFFSET:
            fires.append(i)
        elif i < END_TURN_OFFSET:
            deploys.append(i)
        else:
            end_turn = True

    return moves, fires, deploys, end_turn


def print_actions_menu(valids, game):
    """Print a menu of available actions."""
    moves, fires, deploys, end_turn = get_valid_actions_by_type(valids)

    print("\n=== Available Actions ===")

    if deploys:
        print("\nDeploy (d <type> <facing>):")
        print("  Types: f=Fighter, c=Cruiser, r=Dreadnought")
        print("  Facings: 0=E, 1=NE, 2=NW, 3=W, 4=SW, 5=SE")
        for action in deploys:
            idx = action - DEPLOY_OFFSET
            unit_type = idx // 6
            facing = idx % 6
            type_char = 'fcr'[unit_type]
            print(f"    d {type_char} {facing} -> {decode_action(action)}")

    if moves:
        print(f"\nMove (m <unit> <hex> <facing>): {len(moves)} options")
        # Group by unit
        by_unit = {}
        for action in moves:
            unit_idx = action // MOVES_PER_UNIT
            if unit_idx not in by_unit:
                by_unit[unit_idx] = []
            by_unit[unit_idx].append(action)
        for unit_idx, actions in by_unit.items():
            print(f"  Unit {unit_idx}: {len(actions)} moves")
            # Show first few
            for action in actions[:3]:
                remainder = action % MOVES_PER_UNIT
                dest = remainder // 6
                facing = remainder % 6
                print(f"    m {unit_idx} {dest} {facing}")

    if fires:
        print(f"\nFire (f <unit> <cannon>): {len(fires)} options")
        for action in fires:
            idx = action - FIRE_OFFSET
            unit_idx = idx // MAX_CANNONS
            cannon_idx = idx % MAX_CANNONS
            print(f"    f {unit_idx} {cannon_idx}")

    if end_turn:
        print("\nEnd turn (e)")

    print("\nOther: (q)uit, (h)elp, (s)how board, (v)alid all")


def parse_command(cmd, valids):
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
        if valids[END_TURN_OFFSET] == 1:
            return END_TURN_OFFSET
        print("End turn not valid!")
        return None

    if parts[0] == 'd' and len(parts) >= 3:
        # Deploy: d <type> <facing>
        type_map = {'f': 0, 'c': 1, 'r': 2}
        if parts[1] not in type_map:
            print("Invalid unit type. Use f, c, or r.")
            return None
        try:
            unit_type = type_map[parts[1]]
            facing = int(parts[2])
            action = DEPLOY_OFFSET + unit_type * 6 + facing
            if valids[action] == 1:
                return action
            print(f"Deploy action {action} not valid!")
        except ValueError:
            print("Invalid facing number.")
        return None

    if parts[0] == 'm' and len(parts) >= 4:
        # Move: m <unit> <hex> <facing>
        try:
            unit_idx = int(parts[1])
            dest_hex = int(parts[2])
            facing = int(parts[3])
            action = MOVE_OFFSET + unit_idx * MOVES_PER_UNIT + dest_hex * 6 + facing
            if 0 <= action < FIRE_OFFSET and valids[action] == 1:
                return action
            print(f"Move action not valid!")
        except ValueError:
            print("Invalid numbers.")
        return None

    if parts[0] == 'f' and len(parts) >= 3:
        # Fire: f <unit> <cannon>
        try:
            unit_idx = int(parts[1])
            cannon_idx = int(parts[2])
            action = FIRE_OFFSET + unit_idx * MAX_CANNONS + cannon_idx
            if FIRE_OFFSET <= action < DEPLOY_OFFSET and valids[action] == 1:
                return action
            print(f"Fire action not valid!")
        except ValueError:
            print("Invalid numbers.")
        return None

    # Try to parse as raw action number
    try:
        action = int(parts[0])
        if 0 <= action < NUM_MOVES and valids[action] == 1:
            return action
        print(f"Action {action} not valid!")
    except ValueError:
        pass

    return None


def play_random_action(game, valids):
    """Play a random valid action."""
    valid_indices = [i for i in range(len(valids)) if valids[i] == 1]
    if not valid_indices:
        return None
    action = np.random.choice(valid_indices)
    return action


def main():
    print("=== Star Gambit Interactive Player ===")
    print("Commands: d=deploy, m=move, f=fire, e=end turn, q=quit, h=help")
    print()

    game = alphazero.StarGambitGS()

    # Game mode selection
    print("Select mode:")
    print("  1. Human vs Human")
    print("  2. Human (P0) vs Random AI")
    print("  3. Random AI vs Random AI (watch)")

    mode = input("Mode (1/2/3): ").strip()
    if mode not in ['1', '2', '3']:
        mode = '2'

    human_players = set()
    if mode == '1':
        human_players = {0, 1}
    elif mode == '2':
        human_players = {0}
    # mode 3: no human players

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
            print_actions_menu(valids, game)

            while True:
                cmd = input(f"\nPlayer {current} move: ").strip()
                result = parse_command(cmd, valids)

                if result == 'quit':
                    print("Goodbye!")
                    return
                if result == 'help':
                    print("Commands:")
                    print("  d <type> <facing> - Deploy (f=fighter, c=cruiser, r=dreadnought)")
                    print("  m <unit> <hex> <facing> - Move unit")
                    print("  f <unit> <cannon> - Fire cannon")
                    print("  e - End turn")
                    print("  s - Show board")
                    print("  v - Show all valid actions")
                    print("  q - Quit")
                    print("  <number> - Play action by ID")
                    continue
                if result == 'show':
                    print(game)
                    continue
                if result == 'valid':
                    for i in range(len(valids)):
                        if valids[i] == 1:
                            print(f"  {i}: {decode_action(i)}")
                    continue
                if result is not None:
                    print(f"Playing: {decode_action(result)}")
                    game.play_move(result)
                    break
        else:
            # AI turn
            action = play_random_action(game, valids)
            if action is None:
                print("No valid moves for AI!")
                break
            print(f"AI plays: {decode_action(action)}")
            game.play_move(action)

            if mode == '3':
                input("Press Enter to continue...")


if __name__ == "__main__":
    main()
