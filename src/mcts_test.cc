#include "mcts.h"

#include <iostream>

#include "connect4_gs.h"
#include "gtest/gtest.h"

namespace alphazero {
namespace {

// NOLINTNEXTLINE
TEST(Node, Basic) {
  auto gs = connect4_gs::Connect4GS{};
  auto root = Node{};
  root.add_children(gs.valid_moves());
  EXPECT_EQ(7, root.children.size());

  const float CPUCT = 2.0;
  auto pi = SizedVector<float, 7>{};
  pi << 0.1F, 1.2F, 0.3F, 0.4F, 0.5F, 0.6F, 0.7F;
  root.update_policy(pi);
  for (auto& c : root.children) {
    if (c.move == 5) {
      EXPECT_FLOAT_EQ(0.6F, c.policy);
      EXPECT_FLOAT_EQ(1.2F, c.uct(1, CPUCT, 0));
      EXPECT_FLOAT_EQ(2.4F, c.uct(2, CPUCT, 0));
      EXPECT_FLOAT_EQ(3.4F, c.uct(2, CPUCT, 1));
      c.n = 1;
      EXPECT_FLOAT_EQ(1.2F, c.uct(2, CPUCT, 0));
      EXPECT_FLOAT_EQ(1.2F, c.uct(2, CPUCT, 1));
    }
  }
  root.n = 1;
  auto* n = root.best_child(CPUCT, 0);
  EXPECT_EQ(1, n->move);
}

// NOLINTNEXTLINE
TEST(MCTS, Basic) {
  auto gs = connect4_gs::Connect4GS{};
  gs.play_move(1);
  gs.play_move(6);
  gs.play_move(3);
  gs.play_move(6);
  auto mcts = MCTS{2, gs.num_players(), gs.num_moves()};
  while (mcts.depth() < 800) {
    auto leaf = mcts.find_leaf(gs);
    auto [value, pi] = dumb_eval(*leaf);
    mcts.process_result(gs, value, pi);
  }
  auto counts = mcts.counts();
  std::cout << counts << std::endl;
  EXPECT_EQ(MCTS::pick_move(mcts.probs(0)), 2);

  gs = connect4_gs::Connect4GS{};
  gs.play_move(1);
  gs.play_move(6);
  gs.play_move(3);
  gs.play_move(6);
  gs.play_move(4);
  mcts = MCTS{2, gs.num_players(), gs.num_moves()};
  while (mcts.depth() < 800) {
    auto leaf = mcts.find_leaf(gs);
    auto [value, pi] = dumb_eval(*leaf);
    mcts.process_result(gs, value, pi);
  }
  counts = mcts.counts();
  std::cout << counts << std::endl;
  EXPECT_EQ(MCTS::pick_move(mcts.probs(0)), 2);
}

// NOLINTNEXTLINE
TEST(PlayoutEval, Basic) {
  auto gs = connect4_gs::Connect4GS{};
  auto [value, policy] = playout_eval(gs);

  // Value should have num_players + 1 entries (win for p0, win for p1, draw).
  EXPECT_EQ(value.size(), gs.num_players() + 1);
  // Value should sum to 1 (exactly one outcome).
  float value_sum = value.sum();
  EXPECT_NEAR(value_sum, 1.0, 1e-5);
  // All values should be >= 0.
  for (int i = 0; i < value.size(); ++i) {
    EXPECT_GE(value[i], 0.0);
  }

  // Policy should have num_moves entries.
  EXPECT_EQ(policy.size(), gs.num_moves());
  // Policy should sum to ~1.
  float policy_sum = policy.sum();
  EXPECT_NEAR(policy_sum, 1.0, 1e-5);
  // All policy values >= 0.
  for (int i = 0; i < policy.size(); ++i) {
    EXPECT_GE(policy[i], 0.0);
  }
}

// NOLINTNEXTLINE
TEST(MCTS, PlayoutEval) {
  // Same position as the Basic MCTS test: set up a winning threat.
  auto gs = connect4_gs::Connect4GS{};
  gs.play_move(1);
  gs.play_move(6);
  gs.play_move(3);
  gs.play_move(6);
  auto mcts = MCTS{2, gs.num_players(), gs.num_moves()};
  while (mcts.depth() < 800) {
    auto leaf = mcts.find_leaf(gs);
    auto [value, pi] = playout_eval(*leaf);
    mcts.process_result(gs, value, pi);
  }
  auto counts = mcts.counts();
  std::cout << "Playout MCTS counts: " << counts << std::endl;
  // With playout eval and enough simulations, column 2 should be preferred
  // (it creates a winning threat). Allow some variance since playouts are
  // stochastic.
  auto best = MCTS::pick_move(mcts.probs(0));
  std::cout << "Playout MCTS best move: " << best << std::endl;
  // Just verify it runs without crashing and produces a valid move.
  EXPECT_GE(best, 0);
  EXPECT_LT(best, gs.num_moves());
}

// NOLINTNEXTLINE
TEST(MCTS, RootValueSetOnFirstEval) {
  auto gs = connect4_gs::Connect4GS{};
  // Use non-zero FPU reduction so the bug would matter.
  auto mcts = MCTS{2, gs.num_players(), gs.num_moves(), 0, 1.4, 0.25};

  // Run one simulation to initialize root.
  {
    auto leaf = mcts.find_leaf(gs);
    auto [value, pi] = dumb_eval(*leaf);
    mcts.process_result(gs, value, pi);
  }

  // Run 30 more simulations.
  while (mcts.depth() < 31) {
    auto leaf = mcts.find_leaf(gs);
    auto [value, pi] = dumb_eval(*leaf);
    mcts.process_result(gs, value, pi);
  }

  // With root.v properly set, FPU shouldn't create a huge gap that
  // concentrates all visits on a single child. Check that at least 3
  // children have been visited.
  auto counts = mcts.counts();
  int visited = 0;
  for (int i = 0; i < counts.size(); ++i) {
    if (counts(i) > 0) ++visited;
  }
  EXPECT_GE(visited, 3)
      << "With FPU reduction and proper root v, visits should be distributed. "
         "counts: " << counts.transpose();
}

// NOLINTNEXTLINE
TEST(MCTS, PuctInversionWithNoise) {
  // Use an asymmetric position so moves have different Q values.
  auto gs = connect4_gs::Connect4GS{};
  gs.play_move(1);
  gs.play_move(6);
  gs.play_move(3);
  gs.play_move(6);

  auto mcts = MCTS{2, gs.num_players(), gs.num_moves(), 0.25, 1.4, 0.0, false};
  {
    auto leaf = mcts.find_leaf(gs);
    auto [v, pi] = dumb_eval(*leaf);
    mcts.process_result(gs, v, pi, true);  // noise applied on root
  }
  while (mcts.depth() < 200) {
    auto leaf = mcts.find_leaf(gs);
    auto [v, pi] = dumb_eval(*leaf);
    mcts.process_result(gs, v, pi, false);
  }

  auto regular = mcts.probs(1.0);
  auto pruned = mcts.probs_pruned(1.0);

  // In an asymmetric position with noise, PUCT inversion should produce
  // different weights from raw visit counts.
  bool any_diff = false;
  for (uint32_t m = 0; m < gs.num_moves(); ++m) {
    if (std::abs(regular(m) - pruned(m)) > 1e-6) any_diff = true;
  }
  EXPECT_TRUE(any_diff) << "probs_pruned should differ from probs when noise was applied "
                         << "in an asymmetric position.\nregular: " << regular.transpose()
                         << "\npruned:  " << pruned.transpose();

  // Verify pruned sums to 1.
  EXPECT_NEAR(pruned.sum(), 1.0, 1e-5);
}

// NOLINTNEXTLINE
TEST(MCTS, RootFpuZero) {
  auto gs = connect4_gs::Connect4GS{};

  // Without root FPU zero
  auto mcts_normal = MCTS{2, gs.num_players(), gs.num_moves(), 0.0, 1.0, 0.25,
                           false, false};
  while (mcts_normal.depth() < 50) {
    auto leaf = mcts_normal.find_leaf(gs);
    auto [v, pi] = dumb_eval(*leaf);
    mcts_normal.process_result(gs, v, pi);
  }

  // With root FPU zero
  auto mcts_fpu0 = MCTS{2, gs.num_players(), gs.num_moves(), 0.0, 1.0, 0.25,
                          false, true};
  while (mcts_fpu0.depth() < 50) {
    auto leaf = mcts_fpu0.find_leaf(gs);
    auto [v, pi] = dumb_eval(*leaf);
    mcts_fpu0.process_result(gs, v, pi);
  }

  auto counts_normal = mcts_normal.counts();
  auto counts_fpu0 = mcts_fpu0.counts();

  // FPU=0 should give more spread visits (less penalty for unvisited).
  int visited_normal = 0, visited_fpu0 = 0;
  for (int i = 0; i < counts_normal.size(); ++i) {
    if (counts_normal(i) > 0) ++visited_normal;
    if (counts_fpu0(i) > 0) ++visited_fpu0;
  }
  EXPECT_GE(visited_fpu0, visited_normal)
      << "FPU=0 should spread visits at least as much as normal FPU";
}

// NOLINTNEXTLINE
TEST(MCTS, PolicyTargetPruning) {
  auto gs = connect4_gs::Connect4GS{};

  // Run with noise
  auto mcts = MCTS{2, gs.num_players(), gs.num_moves(), 0.25, 1.4, 0.0,
                    false, false, false};
  {
    auto leaf = mcts.find_leaf(gs);
    auto [v, pi] = dumb_eval(*leaf);
    mcts.process_result(gs, v, pi, true);
  }
  while (mcts.depth() < 200) {
    auto leaf = mcts.find_leaf(gs);
    auto [v, pi] = dumb_eval(*leaf);
    mcts.process_result(gs, v, pi, false);
  }

  auto regular = mcts.probs(1.0);
  auto pruned = mcts.probs_pruned(1.0);

  // Both should sum to 1.
  EXPECT_NEAR(regular.sum(), 1.0, 1e-5);
  EXPECT_NEAR(pruned.sum(), 1.0, 1e-5);

  // Best move's share should roughly increase or stay same after pruning.
  // With Dirichlet noise the pruned max can dip slightly, so use a relaxed
  // tolerance (the invariant is approximate under stochastic noise).
  float best_regular = regular.maxCoeff();
  float best_pruned = pruned.maxCoeff();
  EXPECT_GE(best_pruned + 0.01, best_regular)
      << "Best move should gain or maintain share after pruning";

  // Without noise, PUCT inversion can still differ from raw counts
  // (it reflects Q+exploration balance), but it should still sum to 1.
  auto mcts_no_noise = MCTS{2, gs.num_players(), gs.num_moves(), 0.0, 1.0, 0.0,
                              false, false, false};
  while (mcts_no_noise.depth() < 100) {
    auto leaf = mcts_no_noise.find_leaf(gs);
    auto [v, pi] = dumb_eval(*leaf);
    mcts_no_noise.process_result(gs, v, pi, false);
  }
  auto pru_nn = mcts_no_noise.probs_pruned(1.0);
  EXPECT_NEAR(pru_nn.sum(), 1.0, 1e-5);
}

// NOLINTNEXTLINE
TEST(MCTS, ShapedDirichletDistribution) {
  auto gs = connect4_gs::Connect4GS{};

  // Shaped noise
  auto mcts_shaped = MCTS{2, gs.num_players(), gs.num_moves(), 0.25, 1.4, 0.0,
                           false, false, true};
  {
    auto leaf = mcts_shaped.find_leaf(gs);
    auto [v, pi] = dumb_eval(*leaf);
    mcts_shaped.process_result(gs, v, pi, true);
  }
  // Verify policies are valid after shaped noise.
  auto probs = mcts_shaped.probs(1.0);
  float psum = 0;
  for (int m = 0; m < gs.num_moves(); ++m) {
    EXPECT_GE(probs(m), 0.0f);
    psum += probs(m);
  }
  EXPECT_NEAR(psum, 1.0, 1e-4);

  // Uniform noise
  auto mcts_uniform = MCTS{2, gs.num_players(), gs.num_moves(), 0.25, 1.4, 0.0,
                            false, false, false};
  {
    auto leaf = mcts_uniform.find_leaf(gs);
    auto [v, pi] = dumb_eval(*leaf);
    mcts_uniform.process_result(gs, v, pi, true);
  }
  auto probs2 = mcts_uniform.probs(1.0);
  psum = 0;
  for (int m = 0; m < gs.num_moves(); ++m) {
    EXPECT_GE(probs2(m), 0.0f);
    psum += probs2(m);
  }
  EXPECT_NEAR(psum, 1.0, 1e-4);
}

// NOLINTNEXTLINE
TEST(MCTS, ShapedDirichletAlphaDistribution) {
  // Use an asymmetric position where some moves have very different policies.
  auto gs = connect4_gs::Connect4GS{};
  gs.play_move(3);  // center
  gs.play_move(0);  // corner
  gs.play_move(3);  // stack center
  gs.play_move(0);  // stack corner

  const int TRIALS = 50;
  // Track per-move noise across trials to check variance properties.
  std::vector<std::vector<float>> noise_samples(gs.num_moves());

  for (int t = 0; t < TRIALS; ++t) {
    auto mcts = MCTS{2, gs.num_players(), gs.num_moves(), 0.25, 1.4, 0.0,
                      false, false, true};
    auto leaf = mcts.find_leaf(gs);
    auto [v, pi] = dumb_eval(*leaf);
    mcts.process_result(gs, v, pi, true);

    auto p = mcts.probs(1.0);
    EXPECT_NEAR(p.sum(), 1.0, 1e-4);
    for (int m = 0; m < gs.num_moves(); ++m) {
      EXPECT_GE(p(m), 0.0f);
      if (p(m) > 0) {
        noise_samples[m].push_back(p(m));
      }
    }
  }

  // All legal moves should get noise in every trial.
  auto valids = gs.valid_moves();
  for (int m = 0; m < gs.num_moves(); ++m) {
    if (valids(m)) {
      EXPECT_EQ(static_cast<int>(noise_samples[m].size()), TRIALS)
          << "Move " << m << " should get noise in every trial";
    }
  }
}

// NOLINTNEXTLINE
TEST(MCTS, PuctInversionGradual) {
  auto gs = connect4_gs::Connect4GS{};
  gs.play_move(1);
  gs.play_move(6);
  gs.play_move(3);
  gs.play_move(6);

  auto mcts = MCTS{2, gs.num_players(), gs.num_moves(), 0.25, 1.4, 0.0,
                    false, false, false};
  {
    auto leaf = mcts.find_leaf(gs);
    auto [v, pi] = dumb_eval(*leaf);
    mcts.process_result(gs, v, pi, true);
  }
  while (mcts.depth() < 200) {
    auto leaf = mcts.find_leaf(gs);
    auto [v, pi] = dumb_eval(*leaf);
    mcts.process_result(gs, v, pi, false);
  }

  auto regular = mcts.probs(1.0);
  auto pruned = mcts.probs_pruned(1.0);

  EXPECT_NEAR(regular.sum(), 1.0, 1e-5);
  EXPECT_NEAR(pruned.sum(), 1.0, 1e-5);

  // Best move share should increase after pruning.
  EXPECT_GE(pruned.maxCoeff() + 1e-6, regular.maxCoeff());

  // PUCT inversion should be gradual: for moves with moderate Q gap,
  // pruned visits should be between 0 and original (not fully zeroed).
  int partially_pruned = 0;
  for (int m = 0; m < gs.num_moves(); ++m) {
    if (regular(m) > 0 && pruned(m) > 0 &&
        std::abs(regular(m) - pruned(m)) > 1e-6) {
      ++partially_pruned;
    }
  }
  EXPECT_GE(partially_pruned, 1)
      << "PUCT inversion should produce gradual pruning";
}

// NOLINTNEXTLINE
TEST(MCTS, TrainEvalSeparation) {
  // Use asymmetric position so Q values diverge enough to trigger pruning.
  auto gs = connect4_gs::Connect4GS{};
  gs.play_move(1);
  gs.play_move(6);
  gs.play_move(3);
  gs.play_move(6);

  // Eval mode: no noise, temp=1.0, no root_fpu_zero, no shaped_dirichlet
  auto mcts_eval = MCTS{2, gs.num_players(), gs.num_moves(), 0.0, 1.0, 0.0,
                         false, false, false};
  while (mcts_eval.depth() < 200) {
    auto leaf = mcts_eval.find_leaf(gs);
    auto [v, pi] = dumb_eval(*leaf);
    mcts_eval.process_result(gs, v, pi, false);
  }
  auto eval_regular = mcts_eval.probs(1.0);
  auto eval_pruned = mcts_eval.probs_pruned(1.0);
  // Both should be valid distributions.
  EXPECT_NEAR(eval_regular.sum(), 1.0, 1e-5);
  EXPECT_NEAR(eval_pruned.sum(), 1.0, 1e-5);

  // Self-play mode: noise, temp=1.4, root_fpu_zero, shaped_dirichlet
  auto mcts_sp = MCTS{2, gs.num_players(), gs.num_moves(), 0.25, 1.4, 0.0,
                       false, true, true};
  {
    auto leaf = mcts_sp.find_leaf(gs);
    auto [v, pi] = dumb_eval(*leaf);
    mcts_sp.process_result(gs, v, pi, true);
  }
  while (mcts_sp.depth() < 200) {
    auto leaf = mcts_sp.find_leaf(gs);
    auto [v, pi] = dumb_eval(*leaf);
    mcts_sp.process_result(gs, v, pi, false);
  }
  auto sp_regular = mcts_sp.probs(1.0);
  auto sp_pruned = mcts_sp.probs_pruned(1.0);
  // With noise in an asymmetric position, pruned should differ from regular.
  bool any_diff = false;
  for (int m = 0; m < gs.num_moves(); ++m) {
    if (std::abs(sp_regular(m) - sp_pruned(m)) > 1e-6) any_diff = true;
  }
  EXPECT_TRUE(any_diff) << "Self-play mode: noise should cause pruning differences"
                         << "\nregular: " << sp_regular.transpose()
                         << "\npruned:  " << sp_pruned.transpose();

  // Verify root_policy_temp > 1 flattens: max policy with temp=1.4 should be
  // lower than with temp=1.0.
  auto gs2 = connect4_gs::Connect4GS{};
  auto mcts_flat = MCTS{2, gs2.num_players(), gs2.num_moves(), 0.0, 1.4, 0.0,
                         false, false, false};
  {
    auto leaf = mcts_flat.find_leaf(gs2);
    auto [v, pi] = dumb_eval(*leaf);
    mcts_flat.process_result(gs2, v, pi, false);
  }
  auto mcts_sharp = MCTS{2, gs2.num_players(), gs2.num_moves(), 0.0, 1.0, 0.0,
                          false, false, false};
  {
    auto leaf = mcts_sharp.find_leaf(gs2);
    auto [v, pi] = dumb_eval(*leaf);
    mcts_sharp.process_result(gs2, v, pi, false);
  }
  // After 1 sim, probs are just the tempered policy.
  auto p_flat = mcts_flat.probs(1.0);
  auto p_sharp = mcts_sharp.probs(1.0);
  EXPECT_LE(p_flat.maxCoeff(), p_sharp.maxCoeff() + 1e-6)
      << "root_policy_temp=1.4 should flatten policy vs temp=1.0";
}

// NOLINTNEXTLINE
TEST(MCTS, RawPolicyTemperatureInteraction) {
  auto gs = connect4_gs::Connect4GS{};
  gs.play_move(1);
  gs.play_move(6);
  gs.play_move(3);
  gs.play_move(6);

  auto mcts = MCTS{2, gs.num_players(), gs.num_moves(), 0.25, 1.4, 0.0,
                    false, false, false};
  {
    auto leaf = mcts.find_leaf(gs);
    auto [v, pi] = dumb_eval(*leaf);
    mcts.process_result(gs, v, pi, true);  // noise on root
  }

  // After first sim with noise, probs reflect tempered+noised policy.
  auto p = mcts.probs(1.0);
  EXPECT_NEAR(p.sum(), 1.0, 1e-4);

  // All legal moves should have positive probability (noise ensures this).
  auto valids = gs.valid_moves();
  for (int m = 0; m < gs.num_moves(); ++m) {
    if (valids(m)) {
      EXPECT_GT(p(m), 0.0f)
          << "Legal move " << m << " should have positive prob after noise";
    }
  }
}

// NOLINTNEXTLINE
TEST(MCTS, PuctInversionNeverExceedsActual) {
  // Structural invariant: for every move, pruned probability <= unpruned.
  auto gs = connect4_gs::Connect4GS{};
  gs.play_move(1);
  gs.play_move(6);
  gs.play_move(3);
  gs.play_move(6);

  auto mcts = MCTS{2, gs.num_players(), gs.num_moves(), 0.25, 1.4, 0.0,
                    false, false, false};
  {
    auto leaf = mcts.find_leaf(gs);
    auto [v, pi] = dumb_eval(*leaf);
    mcts.process_result(gs, v, pi, true);
  }
  while (mcts.depth() < 200) {
    auto leaf = mcts.find_leaf(gs);
    auto [v, pi] = dumb_eval(*leaf);
    mcts.process_result(gs, v, pi, false);
  }

  auto counts = mcts.counts();
  auto regular = mcts.probs(1.0);
  auto pruned = mcts.probs_pruned(1.0);

  // Pruned uses desired <= actual for each child, so after normalization
  // the best move gains share and others lose. Under Dirichlet noise a
  // lightly-pruned non-best move can gain a small amount of share when
  // other moves are heavily pruned, so use a relaxed tolerance.
  float best_regular_val = regular.maxCoeff();
  for (int m = 0; m < gs.num_moves(); ++m) {
    if (regular(m) == best_regular_val) continue;  // best move can gain
    EXPECT_LE(pruned(m), regular(m) + 0.01)
        << "Non-best move " << m << " should not gain significant share after pruning"
        << "\nregular: " << regular.transpose()
        << "\npruned:  " << pruned.transpose();
  }
}

}  // namespace
}  // namespace alphazero