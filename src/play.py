#!/usr/bin/env python3
"""Unified interactive play agent with pluggable game UI.

Usage:
    python src/play.py configs/connect4.yaml
    python src/play.py configs/star_gambit_skirmish.yaml --think-time 5
    python src/play.py configs/star_gambit_skirmish.yaml --nodes 200
"""

import argparse
import glob
import os
import re
import sys
import time

import numpy as np
import readline  # noqa: F401 - Enable line editing in input()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import alphazero
from config import load_config
from game_ui import get_game_ui

try:
    import torch
    import neural_net

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: torch/neural_net not available. AI features disabled.")


# Default AI parameters
DEFAULT_CPUCT = 1.25
DEFAULT_FPU_REDUCTION = 0.25
DEFAULT_TEMPERATURE = 0.5
DEFAULT_NODE_LIMIT = 100


class PlayerConfig:
    """Configuration for a single player (human or AI)."""

    def __init__(self):
        self.is_ai = False
        self.network_path = None
        self.network = None
        self.eval_type = "random"  # "random", "playout", or "network"
        self.think_time = None
        self.node_limit = DEFAULT_NODE_LIMIT
        self.temperature = DEFAULT_TEMPERATURE
        self.mcts = None
        self.show_hints = False

    def __str__(self):
        if not self.is_ai:
            hints_str = ", hints=on" if self.show_hints else ""
            return f"Human{hints_str}"
        if self.eval_type == "playout":
            net_str = "playout"
        elif self.network_path:
            net_str = os.path.basename(self.network_path)
        else:
            net_str = "random"
        time_str = f"{self.think_time}s" if self.think_time else "unlimited"
        node_str = str(self.node_limit) if self.node_limit else "unlimited"
        return f"AI(net={net_str}, time={time_str}, nodes={node_str}, temp={self.temperature})"


class PlayContext:
    """Game context for play sessions."""

    def __init__(self, game, game_class):
        self.game = game
        self.game_class = game_class
        self.players = [PlayerConfig(), PlayerConfig()]
        self.auto_play = False
        self.cpuct = DEFAULT_CPUCT
        self.fpu_reduction = DEFAULT_FPU_REDUCTION


def create_mcts(game_class, cpuct=DEFAULT_CPUCT, fpu_reduction=DEFAULT_FPU_REDUCTION):
    """Create a new MCTS instance."""
    return alphazero.MCTS(
        cpuct,
        game_class.NUM_PLAYERS(),
        game_class.NUM_MOVES(),
        0.25,
        1.4,
        fpu_reduction,
    )


def apply_temperature(probs, temperature):
    """Apply temperature to probabilities."""
    if temperature == 0:
        result = np.zeros_like(probs)
        result[np.argmax(probs)] = 1.0
        return result
    elif temperature == 1.0:
        return probs
    else:
        scaled = np.power(probs + 1e-10, 1.0 / temperature)
        return scaled / scaled.sum()


def run_mcts_search(gs, agent, mcts, time_limit=None, node_limit=None, eval_type="random"):
    """Run MCTS search. Returns (visit_counts, num_simulations, wld)."""
    if time_limit is None and node_limit is None:
        node_limit = DEFAULT_NODE_LIMIT

    start = time.time()
    sims = 0

    while True:
        if time_limit is not None and time.time() - start >= time_limit:
            break
        if node_limit is not None and sims >= node_limit:
            break

        leaf = mcts.find_leaf(gs)
        if eval_type == "playout":
            v, pi = alphazero.playout_eval(leaf)
            v = np.array(v)
            pi = np.array(pi)
        elif agent is None:
            v = np.full(gs.NUM_PLAYERS() + 1, 1.0 / (gs.NUM_PLAYERS() + 1))
            pi = np.ones(gs.NUM_MOVES()) / gs.NUM_MOVES()
        else:
            canonical = torch.from_numpy(np.array(leaf.canonicalized()))
            v, pi = agent.predict(canonical)
            v = v.cpu().numpy().flatten()
            pi = pi.cpu().numpy().flatten()

        mcts.process_result(gs, v, pi, sims == 0)
        sims += 1

    counts = np.array(mcts.counts())
    wld = np.array(mcts.root_value())
    return counts, sims, wld


def get_ai_probs(ctx, player_idx, valids):
    """Get AI move probabilities. Returns (probs, source_str, sims, wld)."""
    pcfg = ctx.players[player_idx]
    wld = None

    if pcfg.mcts is None:
        pcfg.mcts = create_mcts(ctx.game_class, ctx.cpuct, ctx.fpu_reduction)

    should_search = (pcfg.think_time is not None and pcfg.think_time > 0) or (
        pcfg.node_limit is not None and pcfg.node_limit > 0
    )

    if should_search and (
        pcfg.network is not None or pcfg.eval_type == "playout" or TORCH_AVAILABLE
    ):
        counts, sims, wld = run_mcts_search(
            ctx.game,
            pcfg.network,
            pcfg.mcts,
            time_limit=pcfg.think_time,
            node_limit=pcfg.node_limit,
            eval_type=pcfg.eval_type,
        )
        if counts.sum() > 0:
            probs = counts.astype(float) / counts.sum()
        else:
            probs = np.ones(ctx.game.NUM_MOVES()) / ctx.game.NUM_MOVES()
        source = f"MCTS ({sims} sims)"
    else:
        sims = 0
        if pcfg.eval_type == "playout":
            v, pi = alphazero.playout_eval(ctx.game)
            probs = np.array(pi)
            source = "playout (single rollout)"
        elif pcfg.network is not None:
            canonical = torch.from_numpy(np.array(ctx.game.canonicalized()))
            _, pi = pcfg.network.predict(canonical)
            probs = pi.cpu().numpy().flatten()
            source = "policy network"
        else:
            probs = np.ones(ctx.game.NUM_MOVES()) / ctx.game.NUM_MOVES()
            source = "uniform random"

    probs = apply_temperature(probs, pcfg.temperature)
    probs[valids == 0] = 0
    if probs.sum() > 0:
        probs = probs / probs.sum()
    else:
        valid_count = valids.sum()
        if valid_count > 0:
            probs[valids == 1] = 1.0 / valid_count

    return probs, source, sims, wld


def get_ai_move(ctx, player_idx, valids):
    """Get AI move. Returns (action, probs, source, sims, wld)."""
    probs, source, sims, wld = get_ai_probs(ctx, player_idx, valids)
    action = np.random.choice(len(probs), p=probs)
    ctx.players[player_idx].mcts = None
    return action, probs, source, sims, wld


# ---------------------------------------------------------------------------
# Network discovery and selection
# ---------------------------------------------------------------------------


def discover_checkpoints(game_name, base="data"):
    """Discover checkpoints from experiment directories.

    Returns: {experiment_name: [(iter_num, full_path), ...]}
    Each experiment's checkpoints sorted by iteration descending.
    """
    game_dir = os.path.join(base, game_name)
    if not os.path.isdir(game_dir):
        return {}

    experiments = {}
    for exp_name in sorted(os.listdir(game_dir)):
        checkpoint_dir = os.path.join(game_dir, exp_name, "checkpoint")
        if not os.path.isdir(checkpoint_dir):
            continue

        checkpoints = []
        for pt_file in glob.glob(os.path.join(checkpoint_dir, "*.pt")):
            filename = os.path.basename(pt_file)
            match = re.match(r"^(\d+).*\.pt$", filename)
            if match:
                iter_num = int(match.group(1))
                checkpoints.append((iter_num, pt_file))

        if checkpoints:
            checkpoints.sort(key=lambda x: x[0], reverse=True)
            experiments[exp_name] = checkpoints

    return experiments


def load_network(game_class, path, players, ctx):
    """Load a network and assign to specified players."""
    if path == "playout":
        for p in players:
            ctx.players[p].network = None
            ctx.players[p].network_path = None
            ctx.players[p].eval_type = "playout"
            ctx.players[p].mcts = None
        return True
    if not TORCH_AVAILABLE:
        print("Error: torch not available")
        return False
    try:
        net = neural_net.NNWrapper.load_checkpoint(
            game_class, os.path.dirname(path), os.path.basename(path)
        )
        for p in players:
            ctx.players[p].network = net
            ctx.players[p].network_path = path
            ctx.players[p].eval_type = "network"
            ctx.players[p].mcts = None
        return True
    except Exception as e:
        print(f"Error loading network: {e}")
        return False


def select_checkpoint(checkpoints):
    """Select a checkpoint interactively. Returns path, None (random), or 'playout'."""
    latest_iter, latest_path = checkpoints[0]

    print(f"\nCheckpoints (newest first):")
    print(f"  l. Latest -> iter {latest_iter:04d}")
    print(f"  r. Random policy")
    print(f"  p. Playout policy")
    print("  " + "-" * 40)

    show = min(10, len(checkpoints))
    for i in range(show):
        iter_num, _ = checkpoints[i]
        print(f"  {i}. iter {iter_num:04d}")
    if len(checkpoints) > show:
        print(f"  ... ({len(checkpoints) - show} more)")

    while True:
        choice = input("\nSelect checkpoint (Enter=latest): ").strip().lower()
        if choice in ["", "l"]:
            return latest_path
        if choice == "r":
            return None
        if choice == "p":
            return "playout"
        try:
            idx = int(choice)
            if 0 <= idx < len(checkpoints):
                return checkpoints[idx][1]
            print(f"Enter 0-{len(checkpoints)-1}")
        except ValueError:
            print("Invalid input")


def select_network_interactive(ctx, game_name, base="data"):
    """Interactive network selection. Returns True if a network was loaded."""
    if not TORCH_AVAILABLE:
        print("torch not available - using random policy")
        return False

    experiments = discover_checkpoints(game_name, base)

    if not experiments:
        print(f"No checkpoints found in {base}/{game_name}/*/checkpoint/")
        print("Using random policy")
        return False

    print("\n=== Network Selection ===")

    # Select experiment
    exp_names = list(experiments.keys())
    if len(exp_names) == 1:
        selected = exp_names[0]
        print(f"Experiment: {selected}")
    else:
        print("Available experiments:")
        for i, name in enumerate(exp_names):
            cpts = experiments[name]
            print(
                f"  {i+1}. {name} ({len(cpts)} checkpoints, latest: iter {cpts[0][0]:04d})"
            )
        while True:
            choice = (
                input("\nSelect experiment (Enter=first, r=random): ").strip().lower()
            )
            if choice == "r":
                return False
            if choice in ["", "1"] and len(exp_names) >= 1:
                selected = exp_names[0]
                break
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(exp_names):
                    selected = exp_names[idx]
                    break
            except ValueError:
                pass

    # Same or different networks?
    choice = (
        input(
            "Same network for both players? [y]es / [n]o / [r]andom / [p]layout (default=yes): "
        )
        .strip()
        .lower()
    )

    if choice == "r":
        print("Using random policy")
        return False
    if choice == "p":
        load_network(ctx.game_class, "playout", [0, 1], ctx)
        print("Using playout policy")
        return True

    checkpoints = experiments[selected]

    if choice == "n":
        any_loaded = False
        for player in [0, 1]:
            print(f"\n--- Player {player} ---")
            path = select_checkpoint(checkpoints)
            if path == "playout":
                load_network(ctx.game_class, "playout", [player], ctx)
                print(f"Player {player}: playout")
                any_loaded = True
            elif path is None:
                print(f"Player {player}: random")
            elif load_network(ctx.game_class, path, [player], ctx):
                print(f"Player {player}: {os.path.basename(path)}")
                any_loaded = True
        return any_loaded
    else:
        path = select_checkpoint(checkpoints)
        if path == "playout":
            load_network(ctx.game_class, "playout", [0, 1], ctx)
            print("Using playout policy")
            return True
        if path is None:
            print("Using random policy")
            return False
        if load_network(ctx.game_class, path, [0, 1], ctx):
            print(f"Loaded: {os.path.basename(path)}")
            return True
        return False


# ---------------------------------------------------------------------------
# Display and command handling
# ---------------------------------------------------------------------------


def print_status(ctx):
    """Print current player configuration."""
    for i in range(2):
        print(f"  Player {i}: {ctx.players[i]}")


def format_probs_summary(probs, valids, ui, gs, top_n=5):
    """Format top move probabilities for display."""
    valid_indices = np.where(valids)[0]
    if len(valid_indices) == 0:
        return ""

    sorted_idx = valid_indices[np.argsort(probs[valid_indices])[::-1]]
    lines = []
    for idx in sorted_idx[:top_n]:
        desc = ui.format_move(gs, idx)
        lines.append(f"  {idx:4d}: {desc}  [{probs[idx]*100:5.1f}%]")
    return "\n".join(lines)


def handle_config_command(parts, ctx, game_name, base_dir):
    """Handle AI configuration commands. Returns command string."""
    cmd = parts[0]

    if cmd == "net":
        select_network_interactive(ctx, game_name, base_dir)
        print_status(ctx)
        return "config"

    # Get target player
    player = None
    if len(parts) >= 2:
        try:
            p = int(parts[1])
            if p in [0, 1]:
                player = p
        except ValueError:
            pass

    targets = [player] if player is not None else [0, 1]

    if cmd == "nodes":
        if len(parts) >= (3 if player is not None else 2):
            val = parts[-1]
            for p in targets:
                if val in ["off", "0", "none"]:
                    ctx.players[p].node_limit = None
                else:
                    try:
                        ctx.players[p].node_limit = int(val)
                    except ValueError:
                        print(f"Invalid node count: {val}")
                        return "config"
            print_status(ctx)
    elif cmd == "time":
        if len(parts) >= (3 if player is not None else 2):
            val = parts[-1]
            for p in targets:
                if val in ["off", "0", "none"]:
                    ctx.players[p].think_time = None
                else:
                    try:
                        ctx.players[p].think_time = float(val)
                    except ValueError:
                        print(f"Invalid time: {val}")
                        return "config"
            print_status(ctx)
    elif cmd == "temp":
        if len(parts) >= (3 if player is not None else 2):
            val = parts[-1]
            for p in targets:
                try:
                    ctx.players[p].temperature = float(val)
                except ValueError:
                    print(f"Invalid temperature: {val}")
                    return "config"
            print_status(ctx)
    elif cmd == "hints":
        if len(parts) >= (3 if player is not None else 2):
            val = parts[-1]
            for p in targets:
                ctx.players[p].show_hints = val in ["on", "true", "1", "yes"]
            print_status(ctx)

    return "config"


def parse_meta_command(cmd, ctx, game_name="", base_dir="data"):
    """Parse meta-commands (help, undo, quit, status, config, etc.).

    Returns command string or None if not a meta-command.
    """
    lower = cmd.lower().strip()
    if lower in ["q", "quit", "exit"]:
        return "quit"
    if lower in ["h", "help", "?"]:
        return "help"
    if lower in ["u", "undo"]:
        return "undo"
    if lower in ["s", "status"]:
        return "status"
    if lower in ["v", "valid", "moves"]:
        return "valid"
    if lower == "auto":
        return "auto"
    if lower == "manual":
        return "manual"
    parts = lower.split()
    if parts and parts[0] in ["net", "nodes", "time", "temp", "hints"]:
        return handle_config_command(parts, ctx, game_name, base_dir)
    return None


def print_generic_help():
    """Print generic play help."""
    print("\nCommands:")
    print("  <move>     - Play a move (game-specific syntax or action ID)")
    print("  undo / u   - Undo last move")
    print("  valid / v  - List valid moves")
    print("  help / h   - Show this help")
    print("  status / s - Show player configuration")
    print("  auto       - Enable AI auto-play")
    print("  manual     - Disable AI auto-play")
    print("  quit / q   - Quit game")
    print("\nAI configuration:")
    print("  net                      - Re-select network")
    print("  nodes <0|1> <count|off>  - Node limit")
    print("  time <0|1> <secs|off>    - Time limit")
    print("  temp <0|1> <value>       - Temperature")
    print("  hints <0|1> <on|off>     - AI hints for human player")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Play against AI")
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument(
        "--think-time", type=float, default=None, help="AI think time in seconds"
    )
    parser.add_argument("--nodes", type=int, default=None, help="AI node limit")
    parser.add_argument(
        "--base-dir", default="data", help="Base data directory (default: data)"
    )
    args = parser.parse_args()

    config = load_config(args.config, {})
    ui = get_game_ui(config.game)

    # Offer variant selection for games that support it (e.g., Star Gambit)
    variant = ui.select_variant()
    if variant and variant != config.game:
        variant_yaml = os.path.join(os.path.dirname(args.config), f"{variant}.yaml")
        if os.path.exists(variant_yaml):
            config = load_config(variant_yaml, {})
            ui = get_game_ui(config.game)

    Game = config.Game
    print(f"=== {config.game} Interactive Player ===\n")

    gs = Game()
    ctx = PlayContext(gs, Game)
    ctx.cpuct = config.cpuct
    ctx.fpu_reduction = config.fpu_reduction

    # Apply CLI overrides to AI defaults
    if args.think_time is not None:
        for p in ctx.players:
            p.think_time = args.think_time
            p.node_limit = None
    if args.nodes is not None:
        for p in ctx.players:
            p.node_limit = args.nodes

    # Mode selection
    print("Select mode:")
    print("  1. Human vs Human")
    print("  2. Human (P0) vs AI (P1)")
    print("  3. AI (P0) vs Human (P1)")
    print("  4. AI vs AI (watch)")

    mode = input("Mode (1/2/3/4) [2]: ").strip()
    if mode == "1":
        ctx.players[0].is_ai = False
        ctx.players[1].is_ai = False
    elif mode == "3":
        ctx.players[0].is_ai = True
        ctx.players[1].is_ai = False
    elif mode == "4":
        ctx.players[0].is_ai = True
        ctx.players[1].is_ai = True
        ctx.auto_play = True
    else:  # Default: mode 2
        ctx.players[0].is_ai = False
        ctx.players[1].is_ai = True

    # Network selection and configuration
    if ctx.players[0].is_ai or ctx.players[1].is_ai:
        select_network_interactive(ctx, config.game, args.base_dir)

        print("\n=== AI Configuration ===")
        print("Adjust settings (or Enter to start):")
        print_status(ctx)

        while True:
            cmd = input("\nConfig> ").strip()
            if cmd.lower() in ["", "start", "go"]:
                break
            result = parse_meta_command(cmd, ctx, config.game, args.base_dir)
            if result == "quit":
                return
            if result == "help":
                print_generic_help()
            elif result == "status":
                print_status(ctx)

    print("\nStarting game...")

    # Game loop
    history = []

    while True:
        print("\n" + "=" * 50)
        print(ui.display_board(ctx.game))

        scores = ctx.game.scores()
        if scores is not None:
            scores_arr = np.array(scores)
            winners = np.where(scores_arr == 1)[0]
            if len(winners) > 0:
                print(f"\nPlayer {winners[0]} wins!")
            else:
                print("\nDraw!")
            break

        valids = np.array(ctx.game.valid_moves())
        current = ctx.game.current_player()
        pcfg = ctx.players[current]

        if pcfg.is_ai:
            if ctx.auto_play:
                action, probs, source, sims, wld = get_ai_move(ctx, current, valids)
                if action is None:
                    print("No valid moves!")
                    break

                print(f"\nAI (P{current}) [{source}]")
                ui.display_actions_menu(ctx.game, probs, valids, wld=wld)

                # Pause in AI vs AI to let user watch
                if ctx.players[0].is_ai and ctx.players[1].is_ai:
                    input("Press Enter to see move...")

                print(
                    f"\n>>> Plays: {ui.format_move(ctx.game, action)}  [{probs[action]*100:.1f}%]"
                )
                history.append(ctx.game.copy())
                ctx.game.play_move(action)
            else:
                # Manual mode: show AI suggestions, human picks
                probs, source, sims, wld = get_ai_probs(ctx, current, valids)
                print(f"\nAI (P{current}) suggests [{source}]:")
                ui.display_actions_menu(ctx.game, probs, valids, wld=wld)

                while True:
                    cmd = input(f"\nPlayer {current} (AI-assisted) move: ").strip()
                    if not cmd:
                        continue

                    meta = parse_meta_command(cmd, ctx, config.game, args.base_dir)
                    if meta == "quit":
                        return
                    if meta == "help":
                        print_generic_help()
                        ui.show_help(ctx.game)
                        continue
                    if meta == "undo":
                        if history:
                            ctx.game = history.pop()
                            print("Move undone")
                            break
                        print("No moves to undo")
                        continue
                    if meta == "status":
                        print_status(ctx)
                        continue
                    if meta == "valid":
                        descs = ui.get_valid_move_descriptions(ctx.game, valids)
                        for aid, desc in descs:
                            prob_str = f"  [{probs[aid]*100:5.1f}%]"
                            print(f"  {aid:4d}: {desc}{prob_str}")
                        continue
                    if meta == "auto":
                        ctx.auto_play = True
                        print("Auto-play enabled")
                        break
                    if meta == "config":
                        continue

                    # Try game-specific parse
                    action = ui.parse_move(ctx.game, cmd, valids)
                    if action is not None:
                        print(f"Playing: {ui.format_move(ctx.game, action)}")
                        history.append(ctx.game.copy())
                        ctx.game.play_move(action)
                        pcfg.mcts = None
                        break
                    print(
                        "Invalid move. Type 'help' for commands, 'valid' for valid moves."
                    )
        else:
            # Human turn
            probs = None
            wld = None
            if pcfg.show_hints and (
                pcfg.network is not None or pcfg.eval_type == "playout"
            ):
                probs, source, _, wld = get_ai_probs(ctx, current, valids)
                print(f"\nHints [{source}]:")
                ui.display_actions_menu(ctx.game, probs, valids, wld=wld)

            while True:
                cmd = input(f"\nPlayer {current} move: ").strip()
                if not cmd:
                    continue

                meta = parse_meta_command(cmd, ctx, config.game, args.base_dir)
                if meta == "quit":
                    return
                if meta == "help":
                    print_generic_help()
                    ui.show_help(ctx.game)
                    continue
                if meta == "undo":
                    if history:
                        ctx.game = history.pop()
                        print("Move undone")
                        break
                    print("No moves to undo")
                    continue
                if meta == "status":
                    print_status(ctx)
                    continue
                if meta == "valid":
                    descs = ui.get_valid_move_descriptions(ctx.game, valids)
                    for aid, desc in descs:
                        prob_str = (
                            f"  [{probs[aid]*100:5.1f}%]"
                            if probs is not None
                            else ""
                        )
                        print(f"  {aid:4d}: {desc}{prob_str}")
                    continue
                if meta == "auto":
                    ctx.auto_play = True
                    print("Auto-play enabled")
                    break
                if meta == "config":
                    continue

                # Try game-specific parse
                action = ui.parse_move(ctx.game, cmd, valids)
                if action is not None:
                    print(f"Playing: {ui.format_move(ctx.game, action)}")
                    history.append(ctx.game.copy())
                    ctx.game.play_move(action)
                    break
                print(
                    "Invalid move. Type 'help' for commands, 'valid' for valid moves."
                )


if __name__ == "__main__":
    main()
