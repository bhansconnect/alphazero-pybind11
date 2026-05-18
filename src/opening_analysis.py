#!/usr/bin/env python3
"""Opening-tree analyzer for AlphaZero training runs.

For each selected checkpoint, runs MCTS from the start position and recursively
expands children while reach probability stays above threshold. The raw tree is
then extracted into a clean hierarchy of openings (top-level lines) and
variations (sibling branches at each opening's first branch point), keyed by
the state hash at the branch point. Across iterations, openings are matched by
identity hash to produce a narrative (still / deepened / diverged / new / dropped).

Usage:
    uv run python src/opening_analysis.py                       # auto-discover
    uv run python src/opening_analysis.py star_gambit_unified   # pre-fill game
    uv run python src/opening_analysis.py --non-interactive     # all defaults
"""

import argparse
import math
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import readline  # noqa: F401 - enables line editing in input()
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import alphazero
from cache_utils import create_cache, print_cache_stats
from config import GAME_REGISTRY, TrainConfig, load_config
from game_ui import GameUI, get_game_ui
from play import (
    _prompt_bool,
    _prompt_value,
    _select_experiment,
    apply_temperature,
    discover_checkpoints,
    resolve_game,
    run_mcts_search,
)

try:
    import torch  # noqa: F401
    import neural_net
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Constants (internal — not user-facing)
# ---------------------------------------------------------------------------

SAFETY_MAX_DEPTH = 200                  # failsafe against runaway recursion
DEFAULT_MIN_REACH_PROB = 0.01           # tree-pruning threshold
DEFAULT_OPENING_THRESHOLD = 0.05        # promotes a root child to a named opening
# Main-line continuation: top1 dominates iff (top1 >= ratio × top2) AND (top1 >= floor).
# This is robust across branching factors — a 0.5 top1 over four near-equal alternatives
# is dominant (ratio ~3), but a 0.4 top1 over a 0.3 second is not (ratio 1.33).
DEFAULT_DOMINANCE_RATIO = 2.0           # top1 must be ≥ this × top2
DEFAULT_MIN_DOMINANCE_PROB = 0.15       # absolute floor so we don't chain on noise
DEFAULT_DISPLAY_CAP = 20                # max openings/variations shown per section
DEFAULT_CACHE_SIZE = 200_000


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ModeConfig:
    """Per-mode parameters: how many sims, what temperature schedule, what noise."""
    name: str                # "selfplay" or "eval"
    sims: int
    start_temp: float
    final_temp: float
    half_life: int
    use_root_noise: bool

    @classmethod
    def selfplay_default(cls, train_config: TrainConfig) -> "ModeConfig":
        return cls(
            name="selfplay",
            sims=train_config.selfplay_mcts_visits,
            start_temp=train_config.self_play_temp,
            final_temp=train_config.final_temp,
            half_life=train_config.temp_decay_half_life,
            use_root_noise=False,
        )

    @classmethod
    def eval_default(cls, train_config: TrainConfig) -> "ModeConfig":
        return cls(
            name="eval",
            sims=train_config.compare_mcts_visits,
            start_temp=train_config.eval_temp,
            final_temp=train_config.final_temp,
            half_life=train_config.temp_decay_half_life,
            use_root_noise=False,
        )


@dataclass
class TreeConfig:
    """Tree-building thresholds."""
    min_reach: float = DEFAULT_MIN_REACH_PROB
    opening_threshold: float = DEFAULT_OPENING_THRESHOLD
    dominance_ratio: float = DEFAULT_DOMINANCE_RATIO
    min_dominance_prob: float = DEFAULT_MIN_DOMINANCE_PROB
    display_cap: int = DEFAULT_DISPLAY_CAP
    full_tree: bool = False
    show_inline_boards: bool = False   # render branch-point board for each opening during the run


# ---------------------------------------------------------------------------
# Tree data structure (layer 1 output)
# ---------------------------------------------------------------------------


@dataclass
class TreeNode:
    """One position in the opening tree."""
    state_hash: int
    depth: int                          # plies from root
    incoming_action: Optional[int]      # action that produced this state from parent (None for root)
    reach_prob: float                   # probability of reaching this position from start under sampling
    sampling_pi: np.ndarray             # post-temperature distribution over actions at this node
    raw_pi: np.ndarray                  # MCTS visit-count distribution (pre-temperature)
    value: np.ndarray                   # MCTS value estimate (WLD or per-player)
    entropy: float                      # entropy of sampling_pi (nats)
    is_terminal: bool
    children: dict = field(default_factory=dict)   # action -> TreeNode
    state: object = None                # GameState (runtime only; not serialized)


# ---------------------------------------------------------------------------
# Layer 1: tree builder
# ---------------------------------------------------------------------------


def temperature_at_depth(depth: int, mode: ModeConfig) -> float:
    """Temperature schedule matching game_runner.py:646-654 (exponential decay)."""
    ln2 = 0.693
    ld = ln2 / max(1, mode.half_life)
    diff = mode.start_temp - mode.final_temp
    return mode.final_temp + diff * math.exp(-ld * depth)


def make_mcts(game_class, train_config: TrainConfig, mode: ModeConfig):
    """Build an MCTS instance configured for the given mode.

    Matches the constructor used in mcts_analysis._make_mcts and play.create_mcts.
    """
    if mode.use_root_noise:
        epsilon = 0.25
        root_temp = train_config.mcts_root_temp
    else:
        epsilon = 0.0
        root_temp = 1.0
    return alphazero.MCTS(
        train_config.cpuct,
        game_class.NUM_PLAYERS(),
        game_class.NUM_MOVES(),
        epsilon,
        root_temp,
        train_config.fpu_reduction,
        game_class().relative_values(),
        train_config.root_fpu_zero,
        train_config.shaped_dirichlet,
    )


def evaluate_node(state, net, game_class, train_config, mode, cache):
    """Run MCTS at `state` and return (raw_pi, value, sims)."""
    mcts = make_mcts(game_class, train_config, mode)
    counts, sims, wld = run_mcts_search(
        state, net, mcts,
        node_limit=mode.sims,
        eval_type="network",
        cache=cache,
        max_batch_size=1,
    )
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        # No visits made (e.g., immediate terminal). Fall back to uniform over valid moves.
        valids = np.asarray(state.valid_moves(), dtype=np.float64)
        if valids.sum() > 0:
            raw_pi = valids / valids.sum()
        else:
            raw_pi = counts  # all zeros
    else:
        raw_pi = counts / total
    return raw_pi, np.asarray(wld, dtype=np.float64), sims


def build_tree(
    start_state,
    net,
    game_class,
    train_config: TrainConfig,
    mode: ModeConfig,
    tree_config: TreeConfig,
    cache,
    progress_fn=None,
    evaluator=None,
    hash_fn=None,
) -> TreeNode:
    """Build an opening tree rooted at `start_state`. Returns the root TreeNode.

    `evaluator` and `hash_fn` are dependency-injection hooks used by tests so
    the tree-building logic can be exercised without the C++ MCTS or hashing
    backends. In production they default to `evaluate_node` and
    `alphazero.hash_game_state`.
    """
    if evaluator is None:
        evaluator = evaluate_node
    if hash_fn is None:
        hash_fn = alphazero.hash_game_state
    node_counter = {"n": 0}

    def _build(state, depth, reach_prob, incoming_action):
        node_counter["n"] += 1
        if progress_fn is not None and node_counter["n"] % 5 == 0:
            progress_fn(node_counter["n"])

        state_hash = hash_fn(state)

        # Terminal: no further expansion.
        if state.scores() is not None:
            return TreeNode(
                state_hash=state_hash, depth=depth,
                incoming_action=incoming_action, reach_prob=reach_prob,
                sampling_pi=np.zeros(0), raw_pi=np.zeros(0),
                value=np.asarray(state.scores(), dtype=np.float64),
                entropy=0.0, is_terminal=True, state=state.copy(),
            )

        # Safety bound (well past any realistic game length).
        if depth >= SAFETY_MAX_DEPTH:
            return TreeNode(
                state_hash=state_hash, depth=depth,
                incoming_action=incoming_action, reach_prob=reach_prob,
                sampling_pi=np.zeros(0), raw_pi=np.zeros(0),
                value=np.zeros(state.num_players() + 1),
                entropy=0.0, is_terminal=False, state=state.copy(),
            )

        raw_pi, value, _sims = evaluator(state, net, game_class, train_config, mode, cache)
        temp = temperature_at_depth(depth, mode)
        sampling_pi = apply_temperature(raw_pi, temp)

        safe = sampling_pi[sampling_pi > 0]
        entropy = float(-np.sum(safe * np.log(safe))) if safe.size > 0 else 0.0

        node = TreeNode(
            state_hash=state_hash, depth=depth,
            incoming_action=incoming_action, reach_prob=reach_prob,
            sampling_pi=sampling_pi, raw_pi=raw_pi,
            value=value, entropy=entropy, is_terminal=False,
            state=state.copy(),
        )

        # Reach-probability is the sole pruning criterion (no top-K, no max-depth).
        for action in range(len(sampling_pi)):
            p_move = float(sampling_pi[action])
            if p_move <= 0.0:
                continue
            child_reach = reach_prob * p_move
            if child_reach < tree_config.min_reach:
                continue
            child_state = state.copy()
            child_state.play_move(int(action))
            node.children[int(action)] = _build(
                child_state, depth + 1, child_reach, int(action),
            )

        return node

    return _build(start_state, depth=0, reach_prob=1.0, incoming_action=None)


def count_tree_nodes(root: TreeNode) -> int:
    n = 1
    for ch in root.children.values():
        n += count_tree_nodes(ch)
    return n


def collect_state_hashes_on_path(root: TreeNode, target_hash: int) -> Optional[list[int]]:
    """Return the path of state hashes from root to the node with target_hash, or None."""
    if root.state_hash == target_hash:
        return [root.state_hash]
    for ch in root.children.values():
        sub = collect_state_hashes_on_path(ch, target_hash)
        if sub is not None:
            return [root.state_hash] + sub
    return None


# ---------------------------------------------------------------------------
# Layer 2: opening extractor
#
# Turns a raw probability tree into a clean human-readable hierarchy of
# openings and variations. Identity is the state hash at each opening's
# branch point (the first node where the policy fans out below
# main_line_threshold). This handles transpositions: two action sequences
# reaching the same state collapse to one identity.
# ---------------------------------------------------------------------------


@dataclass
class Opening:
    """An opening (top-level) or variation (recursive sub-opening).

    A main line is a chain of nodes ending at the first branch point. The
    branch point's state hash is the opening's identity. Sub-variations are
    siblings at the branch point, each recursively an Opening.
    """
    main_line_actions: list      # action indices along the main line
    main_line_nodes: list        # TreeNodes along the main line, ending at branch_point
    branch_point: TreeNode
    identity_hash: int           # = branch_point.state_hash
    terminal_reach: float        # reach_prob at branch_point (absolute, from start)
    depth: int                   # number of plies in the main line
    parent_conditional_prob: float   # P(this opening's first move | at parent's branch point); 1.0 for top-level
    transposition_labels: list = field(default_factory=list)   # alt action sequences reaching identity_hash
    variations: list = field(default_factory=list)             # list[Opening]


def _is_dominant(cur: TreeNode, dominance_ratio: float, min_dominance_prob: float):
    """Return (best_action, is_dominant) using the dominance rule.

    Top action is "dominant" iff its sampling probability is both (a) at least
    `dominance_ratio` × the second-best action's probability, and (b) at least
    `min_dominance_prob` absolute. The ratio rule is robust to branching factor
    (a clear leader stands out regardless of how many alternatives exist); the
    floor avoids chaining on numerical noise.

    Looks at the full policy (`sampling_pi`), not just `children`, so that
    siblings pruned by min_reach still count against dominance — a move the
    policy considers viable but that we pruned for tree-size reasons should
    still prevent us from calling the top move dominant.
    """
    if not cur.children:
        return None, False
    # Sort all action probabilities descending. Ignore zeros.
    probs = [(a, float(p)) for a, p in enumerate(cur.sampling_pi) if p > 0.0]
    if not probs:
        return None, False
    probs.sort(key=lambda kv: -kv[1])
    top_action, top_prob = probs[0]
    # The top action must also be an actual child (else it was pruned and we
    # can't walk down it). If it isn't, the line cannot extend.
    if top_action not in cur.children:
        return None, False
    if top_prob < min_dominance_prob:
        return top_action, False
    if len(probs) == 1:
        # Policy concentrates entirely on one move — fully dominant.
        return top_action, True
    _second_action, second_prob = probs[1]
    if second_prob <= 0.0:
        return top_action, True
    return top_action, (top_prob >= dominance_ratio * second_prob)


def _walk_main_line(start_node: TreeNode, dominance_ratio: float,
                    min_dominance_prob: float):
    """Walk from start_node down while the top child is "dominant."

    Returns (nodes_along_line, branch_point). The branch point is the last
    node walked — either where the policy fans out or where the game ends.
    """
    nodes = [start_node]
    cur = start_node
    while cur.children and not cur.is_terminal:
        best_action, dominant = _is_dominant(cur, dominance_ratio, min_dominance_prob)
        if not dominant or best_action is None:
            break
        cur = cur.children[best_action]
        nodes.append(cur)
    return nodes, cur


def _build_opening_from(start_node: TreeNode, conditional_prob: float,
                       tree_config: TreeConfig) -> Opening:
    """Recursively build an Opening rooted at start_node.

    `conditional_prob` = P(playing the action that leads into start_node | at
    parent's branch point). For top-level openings this is the action's
    sampling_pi at the root. For sub-variations it's the sampling_pi at the
    parent opening's branch point.
    """
    nodes, branch_point = _walk_main_line(
        start_node, tree_config.dominance_ratio, tree_config.min_dominance_prob,
    )
    actions = [n.incoming_action for n in nodes]
    variations: list[Opening] = []
    if not branch_point.is_terminal:
        # Variations are children of the branch point, sorted by reach descending.
        sorted_children = sorted(
            branch_point.children.items(),
            key=lambda kv: -kv[1].reach_prob,
        )
        for action, child in sorted_children:
            cond = (
                float(branch_point.sampling_pi[action])
                if action < len(branch_point.sampling_pi) else 0.0
            )
            variations.append(_build_opening_from(child, cond, tree_config))
    return Opening(
        main_line_actions=actions,
        main_line_nodes=nodes,
        branch_point=branch_point,
        identity_hash=branch_point.state_hash,
        terminal_reach=branch_point.reach_prob,
        depth=len(nodes),
        parent_conditional_prob=conditional_prob,
        variations=variations,
    )


def extract_openings(root: TreeNode, tree_config: TreeConfig):
    """Extract top-level openings from a raw tree.

    Returns (openings, below_threshold_roots) where:
    - openings: list[Opening] sorted by terminal_reach descending
    - below_threshold_roots: list[(action, reach_prob)] for root children
      with reach < opening_threshold (shown in the report's footer for context)
    """
    openings: list[Opening] = []
    below: list[tuple[int, float]] = []
    for action, root_child in root.children.items():
        cond = float(root.sampling_pi[action]) if action < len(root.sampling_pi) else 0.0
        if root_child.reach_prob < tree_config.opening_threshold:
            below.append((action, root_child.reach_prob))
            continue
        openings.append(_build_opening_from(root_child, cond, tree_config))

    # Dedupe transpositions by identity hash. Keep the higher-reach action
    # sequence as canonical, record the other(s) as transposition_labels, and
    # sum reach probabilities.
    by_id: dict[int, Opening] = {}
    for op in openings:
        if op.identity_hash in by_id:
            existing = by_id[op.identity_hash]
            if op.terminal_reach > existing.terminal_reach:
                op.terminal_reach += existing.terminal_reach
                op.transposition_labels = (
                    existing.transposition_labels
                    + [existing.main_line_actions]
                )
                by_id[op.identity_hash] = op
            else:
                existing.terminal_reach += op.terminal_reach
                existing.transposition_labels.append(op.main_line_actions)
        else:
            by_id[op.identity_hash] = op

    openings = sorted(by_id.values(), key=lambda o: -o.terminal_reach)
    below.sort(key=lambda kv: -kv[1])
    return openings, below


def total_variation_count(opening: Opening) -> int:
    """Recursive count of all sub-variations in an opening (excluding the opening itself)."""
    n = len(opening.variations)
    for v in opening.variations:
        n += total_variation_count(v)
    return n


def deepest_main_line(opening: Opening) -> int:
    """Recursively find the deepest main-line depth in this opening's subtree (counting from root)."""
    deepest = opening.branch_point.depth
    for v in opening.variations:
        deepest = max(deepest, deepest_main_line(v))
    return deepest


def all_main_line_state_hashes(opening: Opening) -> set[int]:
    """All state hashes along this opening's main line (used for cross-iter matching)."""
    hashes = {n.state_hash for n in opening.main_line_nodes}
    for v in opening.variations:
        hashes |= all_main_line_state_hashes(v)
    return hashes


# ---------------------------------------------------------------------------
# Layer 3: cross-iteration narrator
#
# Matches openings across iterations by state hash and labels each as
# still / deepened / shallowed / diverged / new / dropped.
# ---------------------------------------------------------------------------


@dataclass
class IterationReport:
    """Everything we computed for one iteration in one mode."""
    iteration: int
    mode_name: str
    root_node: TreeNode
    openings: list                  # list[Opening], sorted by terminal_reach desc
    below_threshold: list           # list[(action, reach_prob)]
    tree_node_count: int

    @property
    def root_entropy(self) -> float:
        return self.root_node.entropy


@dataclass
class OpeningSnapshot:
    """An opening labeled relative to the previous iteration."""
    iteration: int
    opening: Opening
    family_key: int        # = main_line_actions[0]; -1 if empty
    label: str             # one of: first_seen, still, deepened, shallowed, diverged, new, dropped
    matched_prior: Optional["OpeningSnapshot"] = None
    note: str = ""


def _family_key(opening: Opening) -> int:
    return opening.main_line_actions[0] if opening.main_line_actions else -1


def _path_hashes(opening: Opening) -> set[int]:
    """State hashes along the main line (not including sub-variations)."""
    return {n.state_hash for n in opening.main_line_nodes}


class CrossIterClassifier:
    """Streaming cross-iteration classifier.

    Call `classify(report)` for each iteration in order; returns the snapshots
    for that iteration and updates internal state so the next call sees the
    correct "prior" indices. Useful for printing inline as iterations are built.
    """

    def __init__(self):
        self.prev_index: dict = {}        # identity_hash -> OpeningSnapshot
        self.prev_path_hashes: dict = {}  # any main-line state hash -> OpeningSnapshot

    def classify(self, report: "IterationReport") -> list:
        snaps = _classify_one_iter(report, self.prev_index, self.prev_path_hashes)
        # Update state for next iteration: only live (non-dropped) openings carry forward.
        self.prev_index = {}
        self.prev_path_hashes = {}
        for s in snaps:
            if s.label == "dropped":
                continue
            self.prev_index[s.opening.identity_hash] = s
            for h in _path_hashes(s.opening):
                self.prev_path_hashes[h] = s
        return snaps


def classify_across_iterations(reports: list) -> list:
    """For each iteration, label each opening relative to the previous iteration.

    Returns parallel list of OpeningSnapshot lists. Convenience wrapper around
    CrossIterClassifier for batch use (e.g. when reading reports from disk).
    """
    classifier = CrossIterClassifier()
    return [classifier.classify(r) for r in reports]


def _classify_one_iter(report, prev_index, prev_path_hashes):
    """Classify a single iteration's openings against the prior indices.

    Returns the list of OpeningSnapshots for this iteration only. Does NOT
    mutate prev_index/prev_path_hashes — the caller does that based on the
    returned snapshots (see CrossIterClassifier.classify).
    """
    snaps: list[OpeningSnapshot] = []
    used_prior_ids: set[int] = set()

    if not prev_index:
        # First iteration: every opening is "first_seen".
        for op in report.openings:
            snaps.append(OpeningSnapshot(
                iteration=report.iteration, opening=op,
                family_key=_family_key(op),
                label="first_seen", note="initial iteration",
            ))
        return snaps

    cur_path_hashes_by_op: dict[int, set[int]] = {
        id(op): _path_hashes(op) for op in report.openings
    }

    for op in report.openings:
        fkey = _family_key(op)
        op_path = cur_path_hashes_by_op[id(op)]

        # 1. Direct identity match → still / shallowed (by depth comparison)
        if op.identity_hash in prev_index:
            prior = prev_index[op.identity_hash]
            used_prior_ids.add(op.identity_hash)
            if op.depth > prior.opening.depth:
                # Unusual: same branch point but longer line. Treat as still + note.
                label = "still"
                note = f"identity stable; main line {prior.opening.depth}->{op.depth} plies"
            elif op.depth < prior.opening.depth:
                label = "shallowed"
                note = (
                    f"branch point moved earlier "
                    f"({prior.opening.depth} -> {op.depth} plies)"
                )
            else:
                label = "still"
                note = f"main line stable at {op.depth} plies"
            snaps.append(OpeningSnapshot(
                iteration=report.iteration, opening=op,
                family_key=fkey, label=label, matched_prior=prior, note=note,
            ))
            continue

        # 2. Prior identity sits on current main line → deepened.
        deeper_match: Optional[OpeningSnapshot] = None
        for prior_id, prior in prev_index.items():
            if prior_id in used_prior_ids:
                continue
            if prior_id in op_path:
                deeper_match = prior
                used_prior_ids.add(prior_id)
                break
        if deeper_match is not None:
            note = (
                f"main line extended past iter {deeper_match.iteration}'s branch "
                f"point ({deeper_match.opening.depth} -> {op.depth} plies)"
            )
            snaps.append(OpeningSnapshot(
                iteration=report.iteration, opening=op,
                family_key=fkey, label="deepened",
                matched_prior=deeper_match, note=note,
            ))
            continue

        # 3. Current identity sits on a prior opening's main-line path → shallowed.
        if op.identity_hash in prev_path_hashes:
            prior = prev_path_hashes[op.identity_hash]
            used_prior_ids.add(prior.opening.identity_hash)
            note = (
                f"branch point now earlier on iter {prior.iteration}'s line "
                f"({prior.opening.depth} -> {op.depth} plies)"
            )
            snaps.append(OpeningSnapshot(
                iteration=report.iteration, opening=op,
                family_key=fkey, label="shallowed",
                matched_prior=prior, note=note,
            ))
            continue

        # 4. Same family (same first move) → diverged.
        family_match: Optional[OpeningSnapshot] = None
        for prior in prev_index.values():
            if prior.opening.identity_hash in used_prior_ids:
                continue
            if _family_key(prior.opening) == fkey:
                family_match = prior
                break
        if family_match is not None:
            div_ply = 0
            for i, a in enumerate(op.main_line_actions):
                if (i >= len(family_match.opening.main_line_actions)
                        or family_match.opening.main_line_actions[i] != a):
                    div_ply = i
                    break
            else:
                div_ply = len(op.main_line_actions)
            note = (
                f"shares family with iter {family_match.iteration}'s line; "
                f"diverges at ply {div_ply + 1}"
            )
            snaps.append(OpeningSnapshot(
                iteration=report.iteration, opening=op,
                family_key=fkey, label="diverged",
                matched_prior=family_match, note=note,
            ))
            continue

        # 5. No match anywhere → new.
        snaps.append(OpeningSnapshot(
            iteration=report.iteration, opening=op,
            family_key=fkey, label="new",
            note="not seen in prior iteration",
        ))

    # Prior openings that were not claimed → dropped (appended so the user
    # sees what disappeared since last iter).
    for prior_id, prior in prev_index.items():
        if prior_id in used_prior_ids:
            continue
        snaps.append(OpeningSnapshot(
            iteration=report.iteration, opening=prior.opening,
            family_key=prior.family_key,
            label="dropped", matched_prior=prior,
            note=f"present at iter {prior.iteration}, gone now",
        ))

    return snaps


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _opening_name(idx: int) -> str:
    """A→Z, then AA→ZZ, etc."""
    if idx < 26:
        return chr(ord('A') + idx)
    return _opening_name(idx // 26 - 1) + chr(ord('A') + idx % 26)


def _format_action_sequence(actions: list, root_state, ui: GameUI) -> str:
    """Format a sequence of action indices as a human-readable line string.

    Replays from root_state to keep formatting context (whose turn, etc.).
    """
    parts = []
    state = root_state.copy() if root_state is not None else None
    for a in actions:
        if a is None:
            continue
        if state is not None:
            label = ui.format_move(state, int(a))
            state.play_move(int(a))
        else:
            label = f"action#{int(a)}"
        parts.append(label)
    return " / ".join(parts) if parts else "(empty)"


def _render_main_line(opening: Opening, root_node: TreeNode, ui: GameUI, indent: str = "  ") -> list:
    """Render the opening's main line as text lines."""
    lines = []
    parent_node = root_node
    parent_state = root_node.state
    for node in opening.main_line_nodes:
        a = node.incoming_action
        if a is None:
            continue
        label = ui.format_move(parent_state, int(a)) if parent_state is not None else f"action#{int(a)}"
        cond = (
            float(parent_node.sampling_pi[a])
            if a < len(parent_node.sampling_pi) else 0.0
        )
        # ply number is the depth (root has depth 0; nodes start at depth 1)
        ply_num = node.depth
        lines.append(f"{indent}{ply_num:>3d}. {label:<30s}  p={cond:.3f}  reach={node.reach_prob:.3f}")
        parent_node = node
        parent_state = node.state
    if not lines:
        lines.append(f"{indent}(no main-line moves — branch point is the start position)")
    return lines


def _render_variations_summary(opening: Opening, ui: GameUI, indent: str = "  ", max_show: int = 12) -> list:
    """Render a one-line-each summary of the opening's variations."""
    lines = []
    branch_state = opening.branch_point.state
    branch_node = opening.branch_point
    if not opening.variations:
        return lines
    shown = 0
    for var in opening.variations:
        if shown >= max_show:
            remaining = len(opening.variations) - shown
            lines.append(f"{indent}… and {remaining} more variation{'s' if remaining != 1 else ''}")
            break
        if not var.main_line_actions:
            shown += 1
            continue
        first_action = var.main_line_actions[0]
        label = (
            ui.format_move(branch_state, int(first_action))
            if branch_state is not None else f"action#{int(first_action)}"
        )
        cond = (
            float(branch_node.sampling_pi[first_action])
            if first_action < len(branch_node.sampling_pi) else 0.0
        )
        n_sub = len(var.variations)
        sub_descr = f", then {n_sub} sub-variation{'s' if n_sub != 1 else ''}" if n_sub > 0 else ""
        if var.branch_point.is_terminal:
            sub_descr = " (terminal)"
        descriptor = f"main line {var.depth} plies{sub_descr}"
        lines.append(
            f"{indent}· cond={cond:.3f}  reach={var.terminal_reach:.3f}  "
            f"{label:<28s}  → {descriptor}"
        )
        shown += 1
    return lines


def render_iteration_report(report: IterationReport,
                            snapshots: list,
                            ui: GameUI,
                            tree_config: TreeConfig,
                            mode: ModeConfig) -> str:
    """Render the per-iteration text report."""
    out = []
    bar = "=" * 64
    sub_bar = "-" * 64

    out.append(bar)
    out.append(
        f" Iteration {report.iteration:04d} · {report.mode_name} mode · "
        f"{mode.sims} sims/node · min_reach={tree_config.min_reach:.3f}"
    )
    out.append(
        f" opening_threshold={tree_config.opening_threshold:.3f}  "
        f"dominance_ratio={tree_config.dominance_ratio:.2f}  "
        f"min_dominance_prob={tree_config.min_dominance_prob:.3f}"
    )
    out.append(bar)
    out.append("")

    n_openings = len(report.openings)
    total_vars = sum(total_variation_count(op) for op in report.openings)
    deepest = max((op.depth for op in report.openings), default=0)
    out.append(
        f"Summary: {n_openings} opening{'s' if n_openings != 1 else ''} "
        f"({report.root_entropy:.2f} nats root entropy), "
        f"{total_vars} variation{'s' if total_vars != 1 else ''} total, "
        f"deepest opening {deepest} plies."
    )
    out.append(f"Tree size: {report.tree_node_count} nodes above min_reach.")
    out.append("")

    # Index snapshots by opening identity for quick lookup
    snap_by_id = {id(s.opening): s for s in snapshots if s.label != "dropped"}

    if n_openings == 0:
        out.append("(No openings reached the opening threshold. Root policy is too diffuse —")
        out.append(" the network has not yet learned consistent first moves.)")
        out.append("")
    else:
        display_count = min(n_openings, tree_config.display_cap)
        for i in range(display_count):
            op = report.openings[i]
            name = _opening_name(i)
            snap = snap_by_id.get(id(op))
            label_str = f"  [{snap.label}]" if snap else ""

            out.append(sub_bar)
            out.append(
                f" Opening {name}{label_str} · reach {op.terminal_reach:.3f} · "
                f"main line {op.depth} plies · "
                f"{len(op.variations)} variation{'s' if len(op.variations) != 1 else ''}"
            )
            out.append(sub_bar)
            if snap and snap.note:
                out.append(f"({snap.note})")

            if op.transposition_labels:
                for alt in op.transposition_labels:
                    alt_str = _format_action_sequence(alt, report.root_node.state, ui)
                    out.append(f"  Transposition: also reachable via {alt_str}")

            out.append("")
            out.append("Main line:")
            out.extend(_render_main_line(op, report.root_node, ui))
            out.append(
                f"     (branch point — entropy {op.branch_point.entropy:.2f} nats, "
                f"{len(op.branch_point.children)} children explored)"
            )

            out.append("")
            out.append("Position at branch point:")
            if op.branch_point.state is not None:
                board = ui.display_board(op.branch_point.state)
                for line in board.splitlines():
                    out.append(f"  {line}")
            else:
                out.append("  (state unavailable)")

            out.append("")
            if op.variations:
                out.append("Variations at branch point:")
                out.extend(_render_variations_summary(op, ui, max_show=tree_config.display_cap))
            else:
                out.append("(No further variations above min_reach.)")
            out.append("")

        if n_openings > tree_config.display_cap:
            out.append(f"… and {n_openings - tree_config.display_cap} more openings below display cap")
            out.append("")

    # Below-threshold roots
    if report.below_threshold:
        out.append(sub_bar)
        out.append(" Below-threshold root moves (reach below opening_threshold)")
        out.append(sub_bar)
        shown = 0
        root_state = report.root_node.state
        for action, reach in report.below_threshold:
            if shown >= tree_config.display_cap:
                remaining = len(report.below_threshold) - shown
                out.append(f"  … and {remaining} more")
                break
            label = (
                ui.format_move(root_state, int(action))
                if root_state is not None else f"action#{int(action)}"
            )
            out.append(f"  reach {reach:.3f}  {label}")
            shown += 1
        out.append("")

    # Dropped openings (from previous iteration that didn't survive)
    dropped = [s for s in snapshots if s.label == "dropped"]
    if dropped:
        out.append(sub_bar)
        out.append(" Dropped from previous iteration")
        out.append(sub_bar)
        for s in dropped:
            actions_str = _format_action_sequence(
                s.opening.main_line_actions, report.root_node.state, ui
            )
            out.append(
                f"  was at iter {s.matched_prior.iteration if s.matched_prior else '?'}: "
                f"reach {s.opening.terminal_reach:.3f}  ({actions_str})"
            )
        out.append("")

    return "\n".join(out) + "\n"


def render_full_tree(report: IterationReport, ui: GameUI, tree_config: TreeConfig) -> str:
    """Render the full nested tree (for --full-tree mode). One node per line."""
    out = []
    out.append(f"Full tree dump — iteration {report.iteration:04d} mode={report.mode_name}")
    out.append(f"min_reach={tree_config.min_reach}")
    out.append("")

    def recurse(node: TreeNode, indent: int, label_prefix: str):
        a_label = label_prefix
        out.append(
            f"{'  ' * indent}{a_label}  depth={node.depth} reach={node.reach_prob:.3f} "
            f"entropy={node.entropy:.2f}{' (terminal)' if node.is_terminal else ''}"
        )
        if len(node.children) >= 2 and node.state is not None and not node.is_terminal:
            # Render position at branch points
            board = ui.display_board(node.state)
            for line in board.splitlines():
                out.append(f"{'  ' * (indent + 1)}| {line}")
        sorted_children = sorted(node.children.items(), key=lambda kv: -kv[1].reach_prob)
        for action, ch in sorted_children:
            ch_label = (
                ui.format_move(node.state, int(action))
                if node.state is not None else f"action#{int(action)}"
            )
            cond = (
                float(node.sampling_pi[action])
                if action < len(node.sampling_pi) else 0.0
            )
            recurse(ch, indent + 1, f"{ch_label} (p={cond:.3f})")

    recurse(report.root_node, indent=0, label_prefix="(root)")
    return "\n".join(out) + "\n"


def render_summary(reports: list, snapshots_per_iter: list, ui: GameUI,
                  mode_name: str, tree_config: TreeConfig) -> str:
    """Render the cross-iteration summary as markdown."""
    out = []
    out.append(f"# Opening evolution — {mode_name} mode")
    out.append("")
    out.append(
        f"_min_reach={tree_config.min_reach}, "
        f"opening_threshold={tree_config.opening_threshold}, "
        f"dominance_ratio={tree_config.dominance_ratio}, "
        f"min_dominance_prob={tree_config.min_dominance_prob}_"
    )
    out.append("")

    # Story timeline
    out.append("## Story timeline")
    out.append("")
    for report, snaps in zip(reports, snapshots_per_iter):
        n_op = len(report.openings)
        if n_op == 0:
            out.append(
                f"- **Iteration {report.iteration:04d}** — no openings above threshold "
                f"(root entropy {report.root_entropy:.2f} nats; policy too diffuse)."
            )
            continue

        # Count labels
        labels: dict = {}
        for s in snaps:
            if s.opening in report.openings:  # ignore dropped (which are from prior iter)
                labels[s.label] = labels.get(s.label, 0) + 1
        n_new = labels.get("new", 0)
        n_deep = labels.get("deepened", 0)
        n_still = labels.get("still", 0)
        n_div = labels.get("diverged", 0)
        n_shal = labels.get("shallowed", 0)
        n_first = labels.get("first_seen", 0)
        n_drop = sum(1 for s in snaps if s.label == "dropped")

        bits = []
        if n_first > 0:
            bits.append(f"{n_first} first-seen")
        if n_still > 0:
            bits.append(f"{n_still} still")
        if n_deep > 0:
            bits.append(f"{n_deep} deepened")
        if n_shal > 0:
            bits.append(f"{n_shal} shallowed")
        if n_div > 0:
            bits.append(f"{n_div} diverged")
        if n_new > 0:
            bits.append(f"{n_new} new")
        if n_drop > 0:
            bits.append(f"{n_drop} dropped")

        deepest = max((op.depth for op in report.openings), default=0)
        out.append(
            f"- **Iteration {report.iteration:04d}** — {n_op} opening{'s' if n_op != 1 else ''}, "
            f"deepest {deepest} plies, root entropy {report.root_entropy:.2f} nats. "
            f"({', '.join(bits) if bits else 'no changes'})"
        )
    out.append("")

    # Group openings into families (by first action) across all iterations
    families: dict = {}   # family_key -> list of (iter, snap) tuples
    for report, snaps in zip(reports, snapshots_per_iter):
        for s in snaps:
            if s.label == "dropped":
                continue
            key = s.family_key
            families.setdefault(key, []).append((report.iteration, s))

    # Sort families by max reach observed
    sorted_families = sorted(
        families.items(),
        key=lambda kv: -max(s.opening.terminal_reach for _i, s in kv[1]),
    )

    out.append("## Per-family tracker")
    out.append("")
    out.append("Each table tracks one opening family (defined by first move). A family")
    out.append("can hold one or more distinct openings across iterations; rows show how")
    out.append("each opening evolved.")
    out.append("")

    # Need a "first move label" — use the root state from the first iter that has it
    root_state_for_label = None
    for rep in reports:
        if rep.root_node.state is not None:
            root_state_for_label = rep.root_node.state
            break

    for family_key, entries in sorted_families:
        if family_key < 0:
            continue
        first_move_label = (
            ui.format_move(root_state_for_label, int(family_key))
            if root_state_for_label is not None else f"action#{int(family_key)}"
        )
        out.append(f"### Family: {first_move_label} (action #{int(family_key)})")
        out.append("")
        out.append("| Iter | Label | Reach | Depth | Identity hash | Note |")
        out.append("|------|-------|-------|-------|----------------|------|")
        for iteration, snap in entries:
            ident = f"`{snap.opening.identity_hash:016x}`"
            note = snap.note.replace("|", "\\|") if snap.note else ""
            out.append(
                f"| {iteration:04d} | {snap.label} | "
                f"{snap.opening.terminal_reach:.3f} | {snap.opening.depth} | "
                f"{ident} | {note} |"
            )
        out.append("")

        # Show the most recent main line in this family
        last_iter, last_snap = entries[-1]
        actions_str = _format_action_sequence(
            last_snap.opening.main_line_actions, root_state_for_label, ui
        )
        out.append(f"Most recent main line (iter {last_iter:04d}): {actions_str}")
        out.append("")

    # Cross-iter metrics tables
    out.append("## Metrics trajectory")
    out.append("")
    out.append("| Iter | Openings | Root entropy | Deepest main line | Tree nodes |")
    out.append("|------|----------|--------------|-------------------|------------|")
    for report in reports:
        deepest = max((op.depth for op in report.openings), default=0)
        out.append(
            f"| {report.iteration:04d} | {len(report.openings)} | "
            f"{report.root_entropy:.2f} | {deepest} | {report.tree_node_count} |"
        )
    out.append("")

    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# Iteration spec parsing
# ---------------------------------------------------------------------------


def parse_iteration_spec(spec: str, available_iters: list) -> list:
    """Parse an iteration spec string into a sorted list of iteration numbers.

    Accepted forms:
      "1,5,20,50"   - explicit list
      "every:10"    - every Nth (plus latest)
      "latest:5"    - last N
      "all"         - all available
      ""            - same as "every:10"

    `available_iters` is the sorted (ascending) list of iteration numbers in the experiment.
    """
    spec = spec.strip().lower()
    if not spec or spec == "every:10":
        spec = "every:10"
    avail_set = set(available_iters)
    latest = max(available_iters) if available_iters else 0

    if spec == "all":
        return sorted(available_iters)

    if spec.startswith("every:"):
        try:
            n = int(spec.split(":", 1)[1])
            if n <= 0:
                n = 10
        except ValueError:
            n = 10
        picked = [i for i in available_iters if i % n == 0]
        # Always include the latest
        if latest not in picked:
            picked.append(latest)
        # And always include iteration 1 (or the smallest) so we capture the start
        smallest = min(available_iters) if available_iters else 0
        if smallest not in picked:
            picked.insert(0, smallest)
        return sorted(set(picked))

    if spec.startswith("latest:"):
        try:
            n = int(spec.split(":", 1)[1])
            if n <= 0:
                n = 5
        except ValueError:
            n = 5
        return sorted(sorted(available_iters)[-n:])

    # Explicit list
    try:
        picked = []
        for part in spec.split(","):
            part = part.strip()
            if part:
                i = int(part)
                if i in avail_set:
                    picked.append(i)
                else:
                    print(f"  (warning: iteration {i} not found, skipping)")
        return sorted(set(picked))
    except ValueError:
        print(f"  (warning: could not parse spec '{spec}', defaulting to every:10)")
        return parse_iteration_spec("every:10", available_iters)


# ---------------------------------------------------------------------------
# CLI / interactive setup
# ---------------------------------------------------------------------------


def _load_experiment_train_config(experiment_dir: str) -> TrainConfig:
    """Load TrainConfig from the experiment's config.yaml; fall back to defaults."""
    config_path = os.path.join(experiment_dir, "config.yaml")
    if os.path.isfile(config_path):
        try:
            return load_config(config_path, {}, warn=False)
        except Exception as e:
            print(f"  (warning: failed to load {config_path}: {e})")
    return TrainConfig()


def _prompt_mode_config(default: ModeConfig, non_interactive: bool) -> ModeConfig:
    """Prompt for one mode's parameters with defaults shown."""
    if non_interactive:
        return default
    print(f"\n--- {default.name}-mode parameters ---")
    sims = _prompt_value("MCTS sims per node", default.sims, int)
    start = _prompt_value("Start temperature (depth 0)", default.start_temp, float)
    final = _prompt_value("Final temperature", default.final_temp, float)
    half = _prompt_value("Temp decay half-life (plies)", default.half_life, int)
    noise = _prompt_bool("Dirichlet root noise", default.use_root_noise)
    return ModeConfig(
        name=default.name, sims=sims, start_temp=start, final_temp=final,
        half_life=half, use_root_noise=noise,
    )


def _prompt_tree_config(default: TreeConfig, non_interactive: bool) -> TreeConfig:
    """Prompt for tree-building parameters."""
    if non_interactive:
        return default
    print("\n--- Tree-building parameters ---")
    min_reach = _prompt_value(
        "Min reach probability (tree-pruning threshold)",
        default.min_reach, float,
    )
    opening = _prompt_value(
        "Opening threshold (reach to count as a named opening)",
        default.opening_threshold, float,
    )
    print("  (Main-line dominance: top child is dominant when it's at least")
    print("   `ratio` times the second-best AND meets an absolute floor. This")
    print("   handles wide branching — a leader stands out by ratio, not by")
    print("   absolute share, so 4+ candidates with one clear favorite still works.)")
    dom_ratio = _prompt_value(
        "Dominance ratio (top1 / top2)",
        default.dominance_ratio, float,
    )
    min_dom = _prompt_value(
        "Min dominance prob (absolute floor for top1)",
        default.min_dominance_prob, float,
    )
    display = _prompt_value("Display cap per section", default.display_cap, int)
    show_boards = _prompt_bool(
        "Print branch-point board for each opening during the run",
        default.show_inline_boards,
    )
    full = _prompt_bool("Full tree dump (written to disk)", default.full_tree)
    return TreeConfig(
        min_reach=min_reach, opening_threshold=opening,
        dominance_ratio=dom_ratio, min_dominance_prob=min_dom,
        display_cap=display, full_tree=full,
        show_inline_boards=show_boards,
    )


def _prompt_iterations(available_iters: list, non_interactive: bool, default: str = "every:10") -> list:
    if non_interactive:
        return parse_iteration_spec(default, available_iters)
    print(f"\nAvailable iterations: {len(available_iters)} "
          f"(from {min(available_iters):04d} to {max(available_iters):04d})")
    print("Spec: 1,5,20,50 (list) | every:10 (every Nth) | latest:5 (last N) | all")
    raw = input(f"  Iterations to analyze [{default}]: ").strip()
    if not raw:
        raw = default
    return parse_iteration_spec(raw, available_iters)


def interactive_setup(args) -> Optional[dict]:
    """Walk the user through setup, return a dict of resolved choices or None to abort."""
    non_int = args.non_interactive

    # 1. Resolve game (auto-discovery if no positional arg).
    game_name, Game = resolve_game(args.game_or_config, args.base_dir)
    print(f"\nGame: {game_name}")
    ui = get_game_ui(game_name)
    checkpoint_game_name = game_name

    # 2. Variant selection (no-op for games without variants).
    if not non_int:
        variant = ui.select_variant()
        if variant and variant != game_name and variant in GAME_REGISTRY:
            game_name = variant
            Game = getattr(alphazero, GAME_REGISTRY[variant])
            ui = get_game_ui(game_name)
            print(f"Variant: {game_name}")
    else:
        # Non-interactive: pick a deterministic variant for games whose default
        # constructor would otherwise randomise (e.g. star_gambit_unified).
        # We use the *_skirmish subclass for unified games — smallest board,
        # fastest analysis. The user can pre-fill the variant via CLI by
        # passing e.g. `star_gambit_unified_skirmish` directly as game_or_config.
        fallback = game_name + "_skirmish"
        if fallback in GAME_REGISTRY:
            game_name = fallback
            Game = getattr(alphazero, GAME_REGISTRY[fallback])
            ui = get_game_ui(game_name)
            print(f"Variant (non-interactive default): {game_name}")
    # The class used for checkpoint loading must match what was trained; for
    # unified variants, that's the base unified class.
    checkpoint_game_class = getattr(alphazero, GAME_REGISTRY[checkpoint_game_name])

    # 3. Experiment selection.
    experiments = discover_checkpoints(checkpoint_game_name, args.base_dir)
    if not experiments:
        print(f"No checkpoints found under {args.base_dir}/{checkpoint_game_name}/*/checkpoint/")
        return None
    exp_name, checkpoints = _select_experiment(experiments)
    if checkpoints in ("random", "playout"):
        print("This tool requires a real network checkpoint. Aborting.")
        return None

    # `checkpoints` is sorted desc; for spec parsing we want ascending.
    available_iters_asc = sorted(i for i, _p in checkpoints)
    iter_to_path = {i: p for i, p in checkpoints}

    # 4. Load the experiment's TrainConfig (defaults for sim counts, temps, etc.).
    experiment_dir = os.path.join(args.base_dir, checkpoint_game_name, exp_name)
    train_config = _load_experiment_train_config(experiment_dir)

    # 5. Iteration picker.
    iters = _prompt_iterations(available_iters_asc, non_int)
    if not iters:
        print("No iterations selected. Aborting.")
        return None
    print(f"Selected iterations: {iters}")

    # 6. Mode picks.
    if non_int:
        run_selfplay = True
        run_eval = True
    else:
        print("\n--- Analysis modes ---")
        run_selfplay = _prompt_bool("Run selfplay-mode tree?", True)
        run_eval = _prompt_bool("Run eval-mode tree?", True)
    if not (run_selfplay or run_eval):
        print("No modes selected. Aborting.")
        return None

    mode_configs = {}
    if run_selfplay:
        mode_configs["selfplay"] = _prompt_mode_config(
            ModeConfig.selfplay_default(train_config), non_int
        )
    if run_eval:
        mode_configs["eval"] = _prompt_mode_config(
            ModeConfig.eval_default(train_config), non_int
        )

    # 7. Tree-building params.
    tree_config = _prompt_tree_config(TreeConfig(), non_int)

    return {
        "game_name": game_name,                       # variant or base
        "Game": Game,                                 # variant class (or base if no variant)
        "checkpoint_game_name": checkpoint_game_name, # always base, used for paths
        "checkpoint_game_class": checkpoint_game_class,
        "ui": ui,
        "experiment_name": exp_name,
        "experiment_dir": experiment_dir,
        "iter_to_path": iter_to_path,
        "iterations": iters,
        "train_config": train_config,
        "mode_configs": mode_configs,
        "tree_config": tree_config,
        "cache_size": args.cache_size if args.cache_size is not None else DEFAULT_CACHE_SIZE,
    }


# ---------------------------------------------------------------------------
# Main analysis loop
# ---------------------------------------------------------------------------


def analyze_one_iteration(
    iteration: int,
    net,
    Game,
    train_config: TrainConfig,
    mode: ModeConfig,
    tree_config: TreeConfig,
    cache,
    ui: GameUI,
) -> IterationReport:
    """Build tree, extract openings, return an IterationReport."""
    start_state = Game()

    def progress(n_nodes):
        # Print a heartbeat without flooding output.
        if n_nodes % 25 == 0:
            print(f"    {n_nodes} nodes built…", end="\r", flush=True)

    root = build_tree(start_state, net, Game, train_config, mode, tree_config, cache,
                      progress_fn=progress)
    print(" " * 40, end="\r")   # clear the heartbeat
    openings, below = extract_openings(root, tree_config)
    return IterationReport(
        iteration=iteration,
        mode_name=mode.name,
        root_node=root,
        openings=openings,
        below_threshold=below,
        tree_node_count=count_tree_nodes(root),
    )


def _print_inline_openings(report: IterationReport, snaps: list, ui: GameUI,
                          mode_name: str, tree_config: TreeConfig):
    """Print a compact summary of the iteration's openings with cross-iter labels.

    Shown on stdout as the run progresses so the user can watch evolution live.
    If `tree_config.show_inline_boards` is True, also prints the branch-point
    position under each opening (top `display_cap` openings only, to keep
    output bounded).
    """
    indent = "    "
    if not report.openings:
        print(f"{indent}(no openings above threshold)")
    else:
        snap_by_id = {id(s.opening): s for s in snaps if s.label != "dropped"}
        root_state = report.root_node.state
        for i, op in enumerate(report.openings):
            name = _opening_name(i)
            snap = snap_by_id.get(id(op))
            label = snap.label if snap else "?"
            # First-move label, truncated for readability
            first_action = op.main_line_actions[0] if op.main_line_actions else None
            if first_action is not None and root_state is not None:
                move_label = ui.format_move(root_state, int(first_action))
            else:
                move_label = "(empty)"
            move_label = move_label[:42]
            n_var = len(op.variations)
            print(f"{indent}{name} [{label:<10s}] "
                  f"reach={op.terminal_reach:.3f}  depth={op.depth}  "
                  f"{n_var} var   {move_label}")

            # Optional inline board for this opening's branch point.
            if tree_config.show_inline_boards and i < tree_config.display_cap:
                bp_state = op.branch_point.state
                if bp_state is not None:
                    board = ui.display_board(bp_state)
                    for bline in board.splitlines():
                        print(f"{indent}    │ {bline}")
                    print("")

    # Summarise label counts (and dropped from prior iter).
    counts: dict = {}
    for s in snaps:
        counts[s.label] = counts.get(s.label, 0) + 1
    pieces = []
    for key in ("first_seen", "still", "deepened", "shallowed", "diverged", "new", "dropped"):
        if counts.get(key):
            pieces.append(f"{counts[key]} {key}")
    if pieces:
        print(f"{indent}  {' · '.join(pieces)}")


def run_analysis(setup: dict) -> int:
    """Run the full analysis pipeline. Returns 0 on success, nonzero on error."""
    if not TORCH_AVAILABLE:
        print("torch/neural_net not available — cannot load checkpoints.")
        return 1

    game_name = setup["game_name"]
    Game = setup["Game"]
    checkpoint_game_class = setup["checkpoint_game_class"]
    ui = setup["ui"]
    iter_to_path = setup["iter_to_path"]
    iterations = setup["iterations"]
    train_config = setup["train_config"]
    mode_configs = setup["mode_configs"]
    tree_config = setup["tree_config"]
    cache_size = setup["cache_size"]
    exp_dir = setup["experiment_dir"]

    out_dir = os.path.abspath(os.path.join(exp_dir, "openings"))
    os.makedirs(out_dir, exist_ok=True)
    print(f"\nWriting reports to:")
    print(f"  {out_dir}")

    # State for streaming cross-iter classification (one per mode).
    classifiers: dict = {name: CrossIterClassifier() for name in mode_configs}
    reports_by_mode: dict = {name: [] for name in mode_configs}
    snapshots_by_mode: dict = {name: [] for name in mode_configs}
    per_iter_filenames: list = []   # absolute paths for end-of-run summary

    for iteration in iterations:
        path = iter_to_path[iteration]
        print(f"\n=== Iteration {iteration:04d} — loading {os.path.basename(path)} ===")
        net = neural_net.NNWrapper.load_checkpoint(
            checkpoint_game_class, os.path.dirname(path), os.path.basename(path),
        )
        net.enable_inference_optimizations()
        # One cache per iteration (network changes, so old cache entries become stale).
        cache = create_cache(Game, cache_size)

        for mode_name, mode in mode_configs.items():
            print(f"  Building tree ({mode_name}, {mode.sims} sims/node, "
                  f"min_reach={tree_config.min_reach})…")
            report = analyze_one_iteration(
                iteration=iteration, net=net, Game=Game,
                train_config=train_config, mode=mode,
                tree_config=tree_config, cache=cache, ui=ui,
            )
            print(f"    → {len(report.openings)} openings, "
                  f"{report.tree_node_count} tree nodes, "
                  f"root entropy {report.root_entropy:.2f} nats.")

            # Streaming classify so we can show changes inline.
            snaps = classifiers[mode_name].classify(report)
            print(f"\n  Openings ({mode_name}) at iter {iteration:04d}:")
            _print_inline_openings(report, snaps, ui, mode_name, tree_config)

            # Write the per-iteration report immediately so the user can open it
            # mid-run if they want.
            fname = f"iter_{report.iteration:04d}_{mode_name}.txt"
            text = render_iteration_report(report, snaps, ui, tree_config, mode)
            out_path = os.path.join(out_dir, fname)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(text)
            per_iter_filenames.append(fname)
            if tree_config.full_tree:
                full_fname = f"iter_{report.iteration:04d}_{mode_name}_full.txt"
                with open(os.path.join(out_dir, full_fname), "w", encoding="utf-8") as f:
                    f.write(render_full_tree(report, ui, tree_config))
                per_iter_filenames.append(full_fname)

            reports_by_mode[mode_name].append(report)
            snapshots_by_mode[mode_name].append(snaps)

        print_cache_stats(cache, label=f"  Cache (iter {iteration:04d})")
        del net, cache

    # Write per-mode summaries.
    summary_filenames: list = []
    for mode_name in mode_configs:
        reports = reports_by_mode[mode_name]
        snapshots_per_iter = snapshots_by_mode[mode_name]
        summary_text = render_summary(reports, snapshots_per_iter, ui, mode_name, tree_config)
        summary_fname = f"summary_{mode_name}.md"
        with open(os.path.join(out_dir, summary_fname), "w", encoding="utf-8") as f:
            f.write(summary_text)
        summary_filenames.append(summary_fname)

    # End-of-run summary: make it impossible to miss where things went.
    print("")
    print("=" * 74)
    print(" Analysis complete.")
    print("=" * 74)
    print("")
    print(f"Output directory:")
    print(f"  {out_dir}")
    print("")
    print("Cross-iteration narratives (read these first — they tell the story):")
    for fn in summary_filenames:
        print(f"  {os.path.join(out_dir, fn)}")
    print("")
    print(f"Per-iteration detail ({len(per_iter_filenames)} files — each opening's main line,")
    print(f"branch-point board, and variations):")
    # Group per-iter files by iteration for readability.
    by_iter: dict = {}
    for fn in per_iter_filenames:
        m = re.match(r"iter_(\d+)_", fn)
        if m:
            by_iter.setdefault(int(m.group(1)), []).append(fn)
    for it in sorted(by_iter):
        files = "  ".join(by_iter[it])
        print(f"  iter {it:04d}:  {files}")
    print("")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "game_or_config", nargs="?", default=None,
        help="Game name or YAML config path (optional, auto-discovers if omitted)",
    )
    parser.add_argument("--base-dir", default="data", help="Base data directory (default: data)")
    parser.add_argument("--cache-size", type=int, default=None,
                       help=f"Inference cache size (default: {DEFAULT_CACHE_SIZE})")
    parser.add_argument("--non-interactive", action="store_true",
                       help="Accept all defaults without prompting")
    args = parser.parse_args()

    setup = interactive_setup(args)
    if setup is None:
        return 1
    return run_analysis(setup)


if __name__ == "__main__":
    sys.exit(main())
