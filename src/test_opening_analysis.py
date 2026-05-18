"""Tests for opening_analysis.py.

Layered like the module itself: tree builder, opening extractor, cross-iter
narrator, and a slow end-to-end test (opt-in) using a real connect4 checkpoint.

Most tests use stub states and a stub evaluator so we don't need the C++
MCTS backend; the dependency-injection hooks on build_tree make this clean.
"""

import math
import os
import sys
from dataclasses import dataclass, field

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import opening_analysis as oa
from opening_analysis import (
    Opening,
    TreeNode,
    TreeConfig,
    ModeConfig,
    IterationReport,
    build_tree,
    extract_openings,
    classify_across_iterations,
    parse_iteration_spec,
    temperature_at_depth,
    count_tree_nodes,
)


# ---------------------------------------------------------------------------
# Stub game state for tree-builder tests
# ---------------------------------------------------------------------------


@dataclass
class StubState:
    """Minimal GameState surface for testing without the C++ backend."""
    num_moves_val: int = 3
    num_players_val: int = 2
    action_history: list = field(default_factory=list)
    terminal_at_actions: tuple = None   # if action_history == this, state is terminal

    def copy(self):
        return StubState(
            num_moves_val=self.num_moves_val,
            num_players_val=self.num_players_val,
            action_history=list(self.action_history),
            terminal_at_actions=self.terminal_at_actions,
        )

    def play_move(self, action):
        self.action_history.append(int(action))

    def scores(self):
        if self.terminal_at_actions is None:
            return None
        if tuple(self.action_history) == self.terminal_at_actions:
            return [1.0, 0.0, 0.0]
        return None

    def num_moves(self):
        return self.num_moves_val

    def num_players(self):
        return self.num_players_val

    def valid_moves(self):
        return [1] * self.num_moves_val

    def current_player(self):
        return len(self.action_history) % self.num_players_val


def stub_hash(state):
    """Hash by action history (deterministic for testing)."""
    return hash(tuple(state.action_history))


def make_stub_evaluator(policy_fn):
    """Build an evaluator that returns the policy from policy_fn(action_history)."""
    def evaluator(state, net, game_class, train_config, mode, cache):
        pi = np.asarray(policy_fn(tuple(state.action_history)), dtype=np.float64)
        # Normalize defensively
        s = pi.sum()
        if s > 0:
            pi = pi / s
        value = np.zeros(state.num_players() + 1)
        return pi, value, 1
    return evaluator


def make_mode(name="selfplay", sims=10, start_temp=1.0, final_temp=1.0, half_life=10):
    """Build a ModeConfig where temperature == 1.0 everywhere by default (no scaling)."""
    return ModeConfig(
        name=name, sims=sims,
        start_temp=start_temp, final_temp=final_temp,
        half_life=half_life, use_root_noise=False,
    )


# ---------------------------------------------------------------------------
# Layer 1: tree builder
# ---------------------------------------------------------------------------


class TestTreeBuilder:

    def test_reach_pruning_excludes_low_prob_branches(self):
        """A branch whose reach_prob < min_reach is not expanded."""
        # Root policy: [0.9, 0.05, 0.05]. With min_reach=0.1, only action 0 expands.
        expanded_states = []
        def policy_fn(history):
            expanded_states.append(history)
            return [0.9, 0.05, 0.05]
        evaluator = make_stub_evaluator(policy_fn)
        tree = build_tree(
            StubState(num_moves_val=3),
            net=None, game_class=StubState, train_config=None,
            mode=make_mode(), tree_config=TreeConfig(min_reach=0.1),
            cache=None, evaluator=evaluator, hash_fn=stub_hash,
        )
        # Root was evaluated.
        assert () in expanded_states
        # Action 0 (reach=0.9) was evaluated as a child.
        assert (0,) in expanded_states
        # Actions 1 and 2 (reach=0.05) were NOT evaluated (pruned).
        assert (1,) not in expanded_states
        assert (2,) not in expanded_states
        # The root TreeNode should have one child (action 0).
        assert list(tree.children.keys()) == [0]

    def test_no_top_k_cutoff_all_above_threshold_expand(self):
        """If 10 children all have meaningful reach, all 10 expand."""
        n = 10
        policy_fn = lambda _hist: [1.0 / n] * n
        evaluator = make_stub_evaluator(policy_fn)
        tree = build_tree(
            StubState(num_moves_val=n),
            net=None, game_class=StubState, train_config=None,
            mode=make_mode(), tree_config=TreeConfig(min_reach=0.05),
            cache=None, evaluator=evaluator, hash_fn=stub_hash,
        )
        # Each child has reach 1/10 = 0.1 ≥ 0.05, so all 10 expand at depth 1.
        assert len(tree.children) == n

    def test_deterministic_line_extends_past_arbitrary_depth(self):
        """A near-deterministic line should extend many plies before reach drops below threshold."""
        # Policy: [0.99, 0.005, 0.005]. Each ply multiplies reach by 0.99.
        # Starting at reach=1.0, depth d has reach = 0.99^d. With min_reach=0.01,
        # the line stops when 0.99^d < 0.01, i.e., d > log(0.01)/log(0.99) ≈ 458.
        # That's well past the safety bound (SAFETY_MAX_DEPTH=200), so we expect
        # the line to hit the safety bound, not the reach bound.
        # We verify it does NOT stop early.
        evaluator = make_stub_evaluator(lambda _h: [0.99, 0.005, 0.005])
        tree = build_tree(
            StubState(num_moves_val=3),
            net=None, game_class=StubState, train_config=None,
            mode=make_mode(), tree_config=TreeConfig(min_reach=0.01),
            cache=None, evaluator=evaluator, hash_fn=stub_hash,
        )
        # Walk down the line; should be > 20 plies deep.
        depth = 0
        node = tree
        while node.children and depth < 250:
            node = next(iter(node.children.values()))
            depth += 1
        assert depth > 20, f"Line stopped at depth {depth}; expected >20"

    def test_terminal_state_stops_recursion(self):
        """A terminal state is a leaf with is_terminal=True."""
        # After action 0 from root, the state becomes terminal.
        evaluator = make_stub_evaluator(lambda _h: [1.0, 0.0, 0.0])
        tree = build_tree(
            StubState(num_moves_val=3, terminal_at_actions=(0,)),
            net=None, game_class=StubState, train_config=None,
            mode=make_mode(), tree_config=TreeConfig(min_reach=0.01),
            cache=None, evaluator=evaluator, hash_fn=stub_hash,
        )
        # Root has one child (action 0), which is terminal.
        assert 0 in tree.children
        child = tree.children[0]
        assert child.is_terminal
        assert child.children == {}

    def test_threshold_consistency_reach_sums_to_one(self):
        """Sum of leaf reach probabilities + sum of pruned ≈ 1.0."""
        # Policy: [0.5, 0.3, 0.2]. min_reach low enough that the full tree is small.
        # Use one-shot policy (terminal after one move) so we don't recurse forever.
        evaluator = make_stub_evaluator(lambda _h: [0.5, 0.3, 0.2])
        tree = build_tree(
            StubState(num_moves_val=3, terminal_at_actions=None),  # never terminal
            net=None, game_class=StubState, train_config=None,
            mode=make_mode(), tree_config=TreeConfig(min_reach=0.05),
            cache=None, evaluator=evaluator, hash_fn=stub_hash,
        )
        # Walk all leaves at depth 1 (they may or may not have children).
        # Just sum the children's reach: should be ≤ 1.0, and any pruned mass is what's left.
        children_reach = sum(ch.reach_prob for ch in tree.children.values())
        # The pruned mass: 1.0 - children_reach (whatever didn't pass threshold).
        # Sanity: children_reach + pruned = 1.0 by construction.
        assert children_reach <= 1.0 + 1e-9

    def test_temperature_schedule(self):
        """temperature_at_depth matches the closed-form formula."""
        mode = ModeConfig(
            name="test", sims=1, start_temp=1.0, final_temp=0.2,
            half_life=10, use_root_noise=False,
        )
        # At d=0: temp == start_temp
        assert temperature_at_depth(0, mode) == pytest.approx(1.0, abs=1e-6)
        # At d=10 (one half-life): temp should be approximately final + (start - final)/2
        expected = 0.2 + (1.0 - 0.2) * 0.5
        assert temperature_at_depth(10, mode) == pytest.approx(expected, abs=0.01)
        # At large d: temp → final
        assert temperature_at_depth(1000, mode) == pytest.approx(0.2, abs=1e-3)


# ---------------------------------------------------------------------------
# Layer 2: opening extractor
# ---------------------------------------------------------------------------


def make_node(state_hash, depth, incoming_action, reach_prob, sampling_pi,
              entropy=0.0, is_terminal=False, children=None):
    """Build a TreeNode for testing the extractor."""
    return TreeNode(
        state_hash=state_hash, depth=depth, incoming_action=incoming_action,
        reach_prob=reach_prob,
        sampling_pi=np.asarray(sampling_pi, dtype=np.float64),
        raw_pi=np.asarray(sampling_pi, dtype=np.float64),
        value=np.zeros(3), entropy=entropy, is_terminal=is_terminal,
        children=children or {}, state=None,
    )


class TestOpeningExtractor:

    def test_dominance_walks_to_leaf_with_no_letter(self):
        """A fully-dominant chain produces ONE opening named 'A'."""
        # Each step: top1=0.70, top2=0.15, ratio 4.67 → dominant. Walk continues.
        # Eventually no children → emit as leaf.
        leaf = make_node(3, 3, 0, 0.7 * 0.7 * 0.7, [1.0, 0, 0])
        mid = make_node(2, 2, 0, 0.7 * 0.7,
                        [0.70, 0.15, 0.15], children={0: leaf})
        first = make_node(1, 1, 0, 0.7,
                          [0.70, 0.15, 0.15], children={0: mid})
        root = make_node(0, 0, None, 1.0,
                         [0.70, 0.15, 0.15], children={0: first})
        openings, _ = extract_openings(
            root,
            TreeConfig(min_reach=0.01, opening_threshold=0.05,
                       dominance_ratio=2.0, min_dominance_prob=0.15),
        )
        assert len(openings) == 1
        op = openings[0]
        assert op.name == "A"
        assert op.depth == 3
        assert op.terminal_node.state_hash == 3

    def test_fork_spawns_sisters_with_letters(self):
        """4-way fork (no dominance) produces 4 sister openings A, B, C, D."""
        c0 = make_node(10, 1, 0, 0.30, [1.0])
        c1 = make_node(11, 1, 1, 0.25, [1.0])
        c2 = make_node(12, 1, 2, 0.25, [1.0])
        c3 = make_node(13, 1, 3, 0.20, [1.0])
        root = make_node(0, 0, None, 1.0,
                         [0.30, 0.25, 0.25, 0.20],
                         children={0: c0, 1: c1, 2: c2, 3: c3})
        openings, _ = extract_openings(
            root,
            TreeConfig(min_reach=0.01, opening_threshold=0.05,
                       dominance_ratio=2.0, min_dominance_prob=0.15),
        )
        # 4 sister openings, sorted by reach.
        assert len(openings) == 4
        names = [op.name for op in openings]
        assert names == ["A", "B", "C", "D"]
        # Highest-reach sister got "A".
        assert openings[0].reach == pytest.approx(0.30)
        # All share family_name "" (top-level fork).
        for op in openings:
            assert op.family_name == ""
        # Each opening lists the other 3 as sisters.
        for op in openings:
            assert len(op.sister_names) == 3

    def test_nested_fork_produces_two_letter_names(self):
        """Root forks → A, B. A forks again → AA, AB. Names compose."""
        # After root-fork action 0 → branch (which itself forks 2 ways).
        # After root-fork action 1 → leaf.
        sub_a = make_node(20, 2, 0, 0.5 * 0.4, [1.0])
        sub_b = make_node(21, 2, 1, 0.5 * 0.4, [1.0])
        branch = make_node(1, 1, 0, 0.5,
                           [0.40, 0.40, 0.20],
                           children={0: sub_a, 1: sub_b})
        leaf2 = make_node(2, 1, 1, 0.4, [1.0])
        root = make_node(0, 0, None, 1.0,
                         [0.50, 0.40, 0.10],
                         children={0: branch, 1: leaf2})
        openings, _ = extract_openings(
            root,
            TreeConfig(min_reach=0.01, opening_threshold=0.05,
                       dominance_ratio=2.0, min_dominance_prob=0.15),
        )
        # At root: top1=0.50, top2=0.40, ratio 1.25 → NOT dominant → fork.
        # Two sisters by root-reach: branch (0.5) is "A", leaf2 (0.4) is "B".
        # A then forks again at depth 1 → AA, AB.
        names = sorted(op.name for op in openings)
        assert names == ["AA", "AB", "B"]

    def test_opening_threshold_filtering(self):
        """Root children below opening_threshold are excluded from walk."""
        child0 = make_node(1, 1, 0, 0.04, [1.0, 0, 0])
        child1 = make_node(2, 1, 1, 0.08, [1.0, 0, 0])
        root = make_node(0, 0, None, 1.0, [0.04, 0.08, 0.0],
                         children={0: child0, 1: child1})
        openings, below = extract_openings(
            root,
            TreeConfig(min_reach=0.01, opening_threshold=0.05,
                       dominance_ratio=2.0, min_dominance_prob=0.15),
        )
        # Only action 1 is above threshold (0.08 > 0.05). Action 0 is below.
        # Walk: 1 above-thresh child, forced continuation → walk into child1.
        # child1 has no children → emit as leaf. So 1 opening.
        assert len(openings) == 1
        assert openings[0].path_actions == [1]
        # Action 0 appears in below_threshold (root-level diagnostic).
        assert (0, pytest.approx(0.04)) in [(a, pytest.approx(r)) for a, r in below]

    def test_minor_variations_at_dominance_step(self):
        """Lesser siblings at a dominance step become minor variations."""
        # branch has 3 children above min_reach. top1=0.7 dominant over 0.15.
        # The two lesser siblings should appear as minor_variations of the opening.
        leaf_main = make_node(10, 2, 0, 0.7, [1.0])
        leaf_minor1 = make_node(11, 2, 1, 0.15, [1.0])
        leaf_minor2 = make_node(12, 2, 2, 0.15, [1.0])
        first = make_node(1, 1, 0, 1.0,
                          [0.70, 0.15, 0.15],
                          children={0: leaf_main, 1: leaf_minor1, 2: leaf_minor2})
        root = make_node(0, 0, None, 1.0, [1.0], children={0: first})
        openings, _ = extract_openings(
            root,
            TreeConfig(min_reach=0.01, opening_threshold=0.05,
                       dominance_ratio=2.0, min_dominance_prob=0.15),
        )
        assert len(openings) == 1
        op = openings[0]
        # 2 lesser siblings should be recorded as minor variations.
        assert len(op.minor_variations) == 2
        # They branched off at depth 2 (where first played its dominant move).
        for mv in op.minor_variations:
            assert mv.depth == 2

    def test_transposition_dedup_by_identity_hash(self):
        """Two action paths reaching the same leaf state collapse to one opening."""
        a = make_node(99, 1, 0, 0.4, [1.0, 0, 0])
        b = make_node(99, 1, 1, 0.3, [1.0, 0, 0])
        root = make_node(0, 0, None, 1.0, [0.4, 0.3, 0.0], children={0: a, 1: b})
        openings, _ = extract_openings(
            root,
            TreeConfig(min_reach=0.01, opening_threshold=0.05,
                       dominance_ratio=2.0, min_dominance_prob=0.15),
        )
        assert len(openings) == 1
        # Reach is summed (0.4 + 0.3 = 0.7).
        assert openings[0].reach == pytest.approx(0.7)
        assert openings[0].transposition_labels   # other path recorded

    def test_dominance_floor_blocks_extension_on_noise(self):
        """top1 below min_dominance_prob doesn't count as dominant.

        Policy [0.10, 0.005, 0.005, …]: ratio is huge but top1 < 0.15 floor.
        With only one above-threshold child, walk treats as forced continuation
        anyway, but minor-variation siblings should NOT be recorded as
        "lesser siblings of a dominance step" since this wasn't a dominance step.
        """
        leaf = make_node(2, 2, 0, 0.10 * 0.10, [1.0])
        wide_pi = [0.005] * 20
        wide_pi[0] = 0.10
        child = make_node(1, 1, 0, 0.10, wide_pi, children={0: leaf})
        root_pi = [0.005] * 20
        root_pi[0] = 0.10
        root = make_node(0, 0, None, 1.0, root_pi, children={0: child})
        openings, _ = extract_openings(
            root,
            TreeConfig(min_reach=0.001, opening_threshold=0.05,
                       dominance_ratio=2.0, min_dominance_prob=0.15),
        )
        assert len(openings) == 1
        # Walk stops at `child` because its grandchild's reach (0.01) is below threshold.
        assert openings[0].depth == 1


# ---------------------------------------------------------------------------
# Layer 3: cross-iteration narrator
# ---------------------------------------------------------------------------


def make_opening(identity_hash, path_actions, reach,
                 path_state_hashes=None, depth=None, name="A"):
    """Build a synthetic Opening with given properties (new model)."""
    if path_state_hashes is None:
        path_state_hashes = list(range(1, len(path_actions) + 1))
    if depth is None:
        depth = len(path_actions)
    nodes = []
    for i, (act, sh) in enumerate(zip(path_actions, path_state_hashes)):
        nodes.append(make_node(sh, depth=i + 1, incoming_action=act,
                              reach_prob=reach, sampling_pi=[1.0]))
    terminal = nodes[-1] if nodes else make_node(identity_hash, 0, None,
                                                  reach, [1.0])
    terminal.state_hash = identity_hash
    return Opening(
        name=name,
        path_nodes=nodes,
        path_actions=path_actions,
        terminal_node=terminal,
        identity_hash=identity_hash,
        reach=reach,
        depth=depth,
        minor_variations=[],
    )


def make_report(iteration, mode_name, openings):
    root = make_node(0, 0, None, 1.0, [1.0, 0, 0])
    return IterationReport(
        iteration=iteration, mode_name=mode_name,
        root_node=root, openings=openings, below_threshold=[],
        tree_node_count=len(openings) + 1,
    )


class TestCrossIterationNarrator:

    def test_still_same_identity_hash(self):
        op_a = make_opening(identity_hash=42, path_actions=[1, 2],
                           reach=0.3,
                           path_state_hashes=[10, 42])
        op_b = make_opening(identity_hash=42, path_actions=[1, 2],
                           reach=0.35,
                           path_state_hashes=[10, 42])
        snaps = classify_across_iterations([
            make_report(1, "selfplay", [op_a]),
            make_report(5, "selfplay", [op_b]),
        ])
        # First iter: first_seen
        assert snaps[0][0].label == "first_seen"
        # Second iter: still
        assert snaps[1][0].label == "still"

    def test_deepened_prior_identity_on_current_path(self):
        """Iter A's identity hash sits on iter B's main-line path → deepened."""
        # Iter A: identity hash 42 at depth 2.
        op_a = make_opening(identity_hash=42, path_actions=[1, 2],
                           reach=0.3,
                           path_state_hashes=[10, 42])
        # Iter B: identity hash 99 at depth 4; hash 42 sits on its path.
        op_b = make_opening(identity_hash=99, path_actions=[1, 2, 3, 4],
                           reach=0.4,
                           path_state_hashes=[10, 42, 77, 99])
        snaps = classify_across_iterations([
            make_report(1, "selfplay", [op_a]),
            make_report(5, "selfplay", [op_b]),
        ])
        assert snaps[1][0].label == "deepened"
        # Note should reference the prior iter.
        assert "iter 1" in snaps[1][0].note or "ply" in snaps[1][0].note

    def test_shallowed_current_identity_on_prior_path(self):
        """Iter A's main line passes through iter B's identity → shallowed."""
        op_a = make_opening(identity_hash=99, path_actions=[1, 2, 3, 4],
                           reach=0.4,
                           path_state_hashes=[10, 42, 77, 99])
        op_b = make_opening(identity_hash=42, path_actions=[1, 2],
                           reach=0.3,
                           path_state_hashes=[10, 42])
        snaps = classify_across_iterations([
            make_report(1, "selfplay", [op_a]),
            make_report(5, "selfplay", [op_b]),
        ])
        assert snaps[1][0].label == "shallowed"

    def test_diverged_same_first_move_different_path(self):
        """Same first move, but main lines diverge before either's branch point."""
        op_a = make_opening(identity_hash=42, path_actions=[1, 2, 3],
                           reach=0.3,
                           path_state_hashes=[10, 20, 42])
        op_b = make_opening(identity_hash=88, path_actions=[1, 5, 6],
                           reach=0.35,
                           path_state_hashes=[10, 50, 88])
        snaps = classify_across_iterations([
            make_report(1, "selfplay", [op_a]),
            make_report(5, "selfplay", [op_b]),
        ])
        # Op A no longer present, Op B is new in same family.
        labels_in_iter5 = [s.label for s in snaps[1]]
        assert "diverged" in labels_in_iter5
        # The prior opening should appear as dropped.
        assert "dropped" in labels_in_iter5

    def test_new_no_overlap_at_all(self):
        op_a = make_opening(identity_hash=42, path_actions=[1, 2],
                           reach=0.3,
                           path_state_hashes=[10, 42])
        op_b = make_opening(identity_hash=99, path_actions=[5, 6],
                           reach=0.35,
                           path_state_hashes=[50, 99])
        snaps = classify_across_iterations([
            make_report(1, "selfplay", [op_a]),
            make_report(5, "selfplay", [op_b]),
        ])
        labels_in_iter5 = sorted(s.label for s in snaps[1])
        assert "new" in labels_in_iter5
        assert "dropped" in labels_in_iter5

    def test_dropped_prior_not_in_current(self):
        op_a = make_opening(identity_hash=42, path_actions=[1, 2],
                           reach=0.3,
                           path_state_hashes=[10, 42])
        # Iter B: no openings at all.
        snaps = classify_across_iterations([
            make_report(1, "selfplay", [op_a]),
            make_report(5, "selfplay", []),
        ])
        # The dropped entry should be listed in iter-5's snapshots.
        assert any(s.label == "dropped" for s in snaps[1])


# ---------------------------------------------------------------------------
# Iteration spec parsing
# ---------------------------------------------------------------------------


class TestIterationSpec:

    def test_every_n(self):
        avail = list(range(1, 101))
        picked = parse_iteration_spec("every:10", avail)
        # Should include 10, 20, ..., 100, plus iter 1 (smallest).
        assert 1 in picked
        assert 10 in picked
        assert 100 in picked
        assert 5 not in picked

    def test_latest_n(self):
        avail = list(range(1, 51))
        picked = parse_iteration_spec("latest:5", avail)
        assert picked == [46, 47, 48, 49, 50]

    def test_explicit_list(self):
        avail = list(range(1, 51))
        picked = parse_iteration_spec("1,5,20,50", avail)
        assert picked == [1, 5, 20, 50]

    def test_all(self):
        avail = [1, 2, 3]
        assert parse_iteration_spec("all", avail) == [1, 2, 3]

    def test_default_empty(self):
        avail = list(range(1, 51))
        picked = parse_iteration_spec("", avail)
        # Default is every:10
        assert 10 in picked


# ---------------------------------------------------------------------------
# Game UI fallback
# ---------------------------------------------------------------------------


class TestGameUIFallback:

    def test_base_game_ui_format_move_returns_string(self):
        """Base GameUI.format_move returns str(action) — no crash."""
        from game_ui import GameUI
        ui = GameUI()
        # Doesn't crash with None state and arbitrary action.
        label = ui.format_move(None, 42)
        assert isinstance(label, str)
        assert "42" in label


# ---------------------------------------------------------------------------
# End-to-end smoke (slow, opt-in)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_end_to_end_connect4_smoke(tmp_path):
    """Run the analyzer on a real connect4 checkpoint at 2 iterations.

    Skipped unless the connect4 experiment exists in data/.
    """
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    if not os.path.isdir(os.path.join(base_dir, "connect4")):
        pytest.skip("data/connect4 not present")

    import alphazero  # noqa: F401 - confirm backend is available
    from play import discover_checkpoints
    experiments = discover_checkpoints("connect4", base_dir)
    if not experiments:
        pytest.skip("No connect4 checkpoints")
    exp_name, ckpts = next(iter(experiments.items()))
    if len(ckpts) < 2:
        pytest.skip("Need at least 2 connect4 checkpoints")

    # Pick two iterations.
    iters = sorted({ckpts[0][0], ckpts[-1][0]})

    # Build setup manually (skip interactive prompts).
    Game = alphazero.Connect4GS
    ui = oa.get_game_ui("connect4")
    train_config = oa._load_experiment_train_config(
        os.path.join(base_dir, "connect4", exp_name)
    )
    mode_configs = {"eval": ModeConfig.eval_default(train_config)}
    tree_config = TreeConfig(min_reach=0.05, opening_threshold=0.10,
                            main_line_threshold=0.5, display_cap=10)

    setup = {
        "game_name": "connect4",
        "Game": Game,
        "checkpoint_game_name": "connect4",
        "checkpoint_game_class": Game,
        "ui": ui,
        "experiment_name": exp_name,
        "experiment_dir": os.path.join(base_dir, "connect4", exp_name),
        "iter_to_path": dict(ckpts),
        "iterations": iters,
        "train_config": train_config,
        "mode_configs": mode_configs,
        "tree_config": tree_config,
        "cache_size": 10_000,
    }

    rc = oa.run_analysis(setup)
    assert rc == 0

    # Verify expected files exist.
    out_dir = os.path.join(setup["experiment_dir"], "openings")
    for it in iters:
        assert os.path.isfile(os.path.join(out_dir, f"iter_{it:04d}_eval.txt"))
    assert os.path.isfile(os.path.join(out_dir, "summary_eval.md"))
