"""Interactive network selection for tournament scripts.

Provides shared logic for discovering trained networks, selecting subsets,
and saving/graphing tournament results.
"""

import os
import re
import glob
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np


@dataclass
class RunInfo:
    run_name: str
    iterations: list[int] = field(default_factory=list)  # sorted ascending
    filenames: dict[int, str] = field(default_factory=dict)  # iteration -> filename


def discover_runs(model_path: str) -> dict[str, RunInfo]:
    """Scan model_path for *.pt files and group by run name.

    Expected filename format: {iteration}-{run_name}.pt
    """
    all_pt_files = sorted(glob.glob(os.path.join(model_path, "*.pt")))
    runs: dict[str, RunInfo] = {}
    for pt_file in all_pt_files:
        filename = os.path.basename(pt_file)
        match = re.match(r'^(\d+)-(.+)\.pt$', filename)
        if match:
            iteration = int(match.group(1))
            run_name = match.group(2)
            if run_name not in runs:
                runs[run_name] = RunInfo(run_name=run_name)
            runs[run_name].iterations.append(iteration)
            runs[run_name].filenames[iteration] = pt_file
    # Sort iterations within each run
    for info in runs.values():
        info.iterations.sort()
    return runs


def auto_select(iterations: list[int], n: int) -> list[int]:
    """Select n iterations evenly distributed, always including latest.

    Places n points at 1/n, 2/n, ..., n/n through the index range,
    so the last pick is always the latest iteration.
    Returns sorted list of selected iterations.
    """
    if n >= len(iterations):
        return list(iterations)
    if n <= 0:
        return []
    if n == 1:
        return [iterations[-1]]
    last_idx = len(iterations) - 1
    selected = set()
    for i in range(1, n + 1):
        idx = round(i * last_idx / n)
        selected.add(iterations[idx])
    return sorted(selected)


def parse_manual_iters(input_str: str, available: list[int]) -> list[int]:
    """Parse comma-separated values and ranges like '0, 5, 10-20, 30'.

    Returns sorted list of matching iterations. Warns about missing ones.
    """
    available_set = set(available)
    requested = set()
    for part in input_str.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            bounds = part.split('-', 1)
            try:
                lo, hi = int(bounds[0].strip()), int(bounds[1].strip())
                for it in available:
                    if lo <= it <= hi:
                        requested.add(it)
            except ValueError:
                print(f"  Warning: could not parse range '{part}'")
        else:
            try:
                val = int(part)
                if val in available_set:
                    requested.add(val)
                else:
                    print(f"  Warning: iteration {val} not found")
            except ValueError:
                print(f"  Warning: could not parse '{part}'")
    return sorted(requested)


def interactive_select(
    runs: dict[str, RunInfo], default_mcts: int = 200
) -> tuple[list[str], int, int, int]:
    """Interactive flow for selecting networks. Returns (agent_filenames, mcts_visits, num_random, num_playout)."""
    run_names = sorted(runs.keys())

    # --- Run selection ---
    if len(run_names) == 1:
        selected_run_names = run_names
        info = runs[run_names[0]]
        print(f"Run: {run_names[0]} ({len(info.iterations)} checkpoints, "
              f"iters {info.iterations[0]}-{info.iterations[-1]})")
    else:
        print("Available runs:")
        for i, name in enumerate(run_names):
            info = runs[name]
            print(f"  {i+1}. {name} ({len(info.iterations)} checkpoints, "
                  f"iters {info.iterations[0]}-{info.iterations[-1]})")
        print(f"  a. All runs")

        choice = input("Select run(s) (comma-separated, or 'a' for all) [a]: ").strip().lower()
        if choice == '' or choice == 'a':
            selected_run_names = run_names
        else:
            selected_run_names = []
            for part in choice.split(','):
                part = part.strip()
                try:
                    idx = int(part) - 1
                    if 0 <= idx < len(run_names):
                        selected_run_names.append(run_names[idx])
                    else:
                        print(f"  Warning: ignoring invalid index {part}")
                except ValueError:
                    print(f"  Warning: ignoring '{part}'")
            if not selected_run_names:
                print("No valid runs selected, using all")
                selected_run_names = run_names

    # --- Network selection ---
    print("\nNetwork selection:")
    print("  a. All networks from each run")
    print("  n. Select count per run (auto: latest + evenly spaced)")
    print("  m. Manual (specify iterations per run)")
    sel_mode = input("Selection [a]: ").strip().lower() or 'a'

    nn_agents = []
    if sel_mode == 'n':
        count_str = input("Networks per run [10]: ").strip() or '10'
        try:
            count = int(count_str)
        except ValueError:
            print("Invalid number, using 10")
            count = 10
        for name in selected_run_names:
            info = runs[name]
            selected = auto_select(info.iterations, count)
            labels = [f"{it:04d}" for it in selected]
            print(f"  {name}: {', '.join(labels)}")
            nn_agents.extend(info.filenames[it] for it in selected)
    elif sel_mode == 'm':
        for name in selected_run_names:
            info = runs[name]
            print(f"  {name}: available iters {info.iterations[0]}-{info.iterations[-1]} "
                  f"({len(info.iterations)} total)")
            iters_str = input(f"  Iterations for {name}: ").strip()
            if not iters_str:
                selected = info.iterations
            else:
                selected = parse_manual_iters(iters_str, info.iterations)
            if not selected:
                print(f"  No valid iterations, using all for {name}")
                selected = info.iterations
            labels = [f"{it:04d}" for it in selected]
            print(f"  Selected: {', '.join(labels)}")
            nn_agents.extend(info.filenames[it] for it in selected)
    else:
        for name in selected_run_names:
            info = runs[name]
            nn_agents.extend(info.filenames[it] for it in info.iterations)

    nn_agents.sort()  # Sort by filename (iteration order)

    # --- Random agent ---
    rand_choice = input("\nInclude random agent? (y/n) [n]: ").strip().lower()
    num_random = 1 if rand_choice == 'y' else 0

    # --- Playout agent ---
    playout_choice = input("Include playout agent? (y/n) [n]: ").strip().lower()
    num_playout = 1 if playout_choice == 'y' else 0

    # --- MCTS visits ---
    mcts_str = input(f"\nMCTS visits [{default_mcts}]: ").strip() or str(default_mcts)
    try:
        mcts_visits = int(mcts_str)
    except ValueError:
        print(f"Invalid number, using {default_mcts}")
        mcts_visits = default_mcts

    return nn_agents, mcts_visits, num_random, num_playout


# --- Tournament output utilities ---

def _has_display():
    """Check if a graphical display is available."""
    if os.environ.get('SSH_TTY') and not os.environ.get('DISPLAY'):
        return False
    import matplotlib
    return matplotlib.get_backend().lower() != 'agg'


def create_tournament_dir(base: str, variant: str, fmt: str) -> str:
    """Create and return a timestamped tournament results directory.

    Example: data/tournaments/skirmish_monrad_20260207_143022/
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    dirname = f"{variant}_{fmt}_{timestamp}"
    path = os.path.join(base, "data", "tournaments", dirname)
    os.makedirs(path, exist_ok=True)
    return path


def _parse_agent_info(agent_name: str) -> tuple[str, int] | None:
    """Extract (run_name, iteration) from an agent filename like '0005-myrun.pt'."""
    match = re.match(r'^(\d+)-(.+)\.pt$', str(agent_name))
    if match:
        return match.group(2), int(match.group(1))
    return None


def save_tournament_results(
    output_dir: str,
    agents: list,
    elo: np.ndarray,
    win_matrix: np.ndarray,
    mcts_visits: int,
    num_random: int,
    num_playout: int = 0,
    variant: str = "",
    fmt: str = "",
):
    """Save all tournament outputs: CSVs, summary, and graphs."""
    # --- win_matrix.csv ---
    np.savetxt(
        os.path.join(output_dir, "win_matrix.csv"),
        win_matrix,
        delimiter=",",
        header=",".join([str(a) for a in agents]),
    )

    # --- elo_ratings.csv ---
    rankings = list(np.argsort(elo))
    with open(os.path.join(output_dir, "elo_ratings.csv"), "w") as f:
        f.write("agent,elo\n")
        for i in reversed(rankings):
            f.write(f"{agents[i]},{elo[i]:.1f}\n")

    # --- summary.txt ---
    with open(os.path.join(output_dir, "summary.txt"), "w") as f:
        f.write(f"Tournament: {variant} {fmt}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"MCTS visits: {mcts_visits}\n")
        f.write(f"Random agents: {num_random}\n")
        f.write(f"Playout agents: {num_playout}\n")
        f.write(f"Total agents: {len(agents)}\n\n")
        f.write("Leaderboard:\n")
        for rank, i in enumerate(reversed(rankings)):
            f.write(f"  {rank+1}. {agents[i]}: {elo[i]:.0f}\n")

    # --- Graphs ---
    if not _has_display():
        import matplotlib
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Group agents by run for multi-line plots
    run_data: dict[str, list[tuple[int, int]]] = {}  # run_name -> [(iteration, agent_idx)]
    random_indices = []
    playout_indices = []
    for idx, agent in enumerate(agents):
        info = _parse_agent_info(agent)
        if info:
            run_name, iteration = info
            run_data.setdefault(run_name, []).append((iteration, idx))
        elif agent == "playout":
            playout_indices.append(idx)
        elif agent != "dummy":
            random_indices.append(idx)

    for run_name in run_data:
        run_data[run_name].sort()

    # ELO vs Iteration
    fig, ax = plt.subplots(figsize=(10, 6))
    for run_name, points in run_data.items():
        iters = [p[0] for p in points]
        elos = [elo[p[1]] for p in points]
        ax.plot(iters, elos, 'o-', label=run_name, markersize=4)
    for ri in random_indices:
        ax.axhline(y=elo[ri], linestyle='--', color='gray', alpha=0.7,
                    label=f"random ({elo[ri]:.0f})")
    for pi in playout_indices:
        ax.axhline(y=elo[pi], linestyle='-.', color='orange', alpha=0.7,
                    label=f"playout ({elo[pi]:.0f})")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("ELO Rating")
    ax.set_title("ELO vs Iteration")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "elo_vs_iteration.png"), dpi=150)

    # Points vs Iteration (sum of win rates per agent)
    points_arr = np.nansum(win_matrix, axis=1)
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    for run_name, pts in run_data.items():
        iters = [p[0] for p in pts]
        scores = [points_arr[p[1]] for p in pts]
        ax2.plot(iters, scores, 'o-', label=run_name, markersize=4)
    for ri in random_indices:
        ax2.axhline(y=points_arr[ri], linestyle='--', color='gray', alpha=0.7,
                     label=f"random ({points_arr[ri]:.1f})")
    for pi in playout_indices:
        ax2.axhline(y=points_arr[pi], linestyle='-.', color='orange', alpha=0.7,
                     label=f"playout ({points_arr[pi]:.1f})")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Total Points (sum of win rates)")
    ax2.set_title("Points vs Iteration")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(os.path.join(output_dir, "points_vs_iteration.png"), dpi=150)

    if _has_display():
        plt.show()
    else:
        print("No display detected, skipping interactive display")

    plt.close('all')

    print(f"\nResults saved to {output_dir}/")
