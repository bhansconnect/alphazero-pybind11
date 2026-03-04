#include "mcts.h"

#include <iostream>
#include <set>
#include <string>

#include "connect4_gs.h"
#include "star_gambit_gs.h"
#include "gtest/gtest.h"

namespace alphazero {
namespace {

// NOLINTNEXTLINE
TEST(Node, Basic) {
  auto gs = connect4_gs::Connect4GS{};
  std::deque<Node> nodes;
  nodes.emplace_back();  // index 0 = root
  auto& root = nodes[0];
  root.v.assign(2, 0.0f);
  root.cached_q.assign(2, 0.0f);
  auto valids = gs.valid_moves();
  for (int i = 0; i < valids.size(); ++i) {
    if (valids(i)) {
      nodes.emplace_back();
      uint32_t child_idx = nodes.size() - 1;
      nodes[child_idx].v.assign(2, 0.0f);
      nodes[child_idx].cached_q.assign(2, 0.0f);
      root.child_indices.push_back(child_idx);
      root.moves.push_back(i);
      root.policies.push_back(0.0f);
      root.edge_n.push_back(0);
      root.edge_n_in_flight.push_back(0);
      nodes[child_idx].ref_count++;
    }
  }
  EXPECT_EQ(7, root.child_indices.size());

  const float CPUCT = 2.0;
  auto pi = SizedVector<float, 7>{};
  pi << 0.1F, 1.2F, 0.3F, 0.4F, 0.5F, 0.6F, 0.7F;
  root.update_policy(pi);
  for (size_t i = 0; i < root.child_indices.size(); ++i) {
    if (root.moves[i] == 5) {
      auto& child = nodes[root.child_indices[i]];
      EXPECT_FLOAT_EQ(0.6F, root.policies[i]);
      // edge_n=0, edge_n_in_flight=0: uses fpu_value (ignores q)
      // uct = fpu + cpuct * policy * sqrt_n / (0 + 0 + 1)
      EXPECT_FLOAT_EQ(1.2F, Node::uct(0, 0, child.cached_q[0], 1, CPUCT, 0, root.policies[i]));
      EXPECT_FLOAT_EQ(2.4F, Node::uct(0, 0, child.cached_q[0], 2, CPUCT, 0, root.policies[i]));
      EXPECT_FLOAT_EQ(3.4F, Node::uct(0, 0, child.cached_q[0], 2, CPUCT, 1, root.policies[i]));
      // edge_n=1: exploitation = cached_q[0] = 0
      // uct = 0.0 + cpuct * 0.6 * 2 / (1 + 0 + 1) = 0.0 + 1.2 = 1.2
      EXPECT_FLOAT_EQ(1.2F, Node::uct(1, 0, child.cached_q[0], 2, CPUCT, 0, root.policies[i]));
      // fpu_value doesn't matter when edge_n > 0
      EXPECT_FLOAT_EQ(1.2F, Node::uct(1, 0, child.cached_q[0], 2, CPUCT, 1, root.policies[i]));
    }
  }
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

  EXPECT_EQ(value.size(), gs.num_players() + 1);
  float value_sum = value.sum();
  EXPECT_NEAR(value_sum, 1.0, 1e-5);
  for (int i = 0; i < value.size(); ++i) {
    EXPECT_GE(value[i], 0.0);
  }

  EXPECT_EQ(policy.size(), gs.num_moves());
  float policy_sum = policy.sum();
  EXPECT_NEAR(policy_sum, 1.0, 1e-5);
  for (int i = 0; i < policy.size(); ++i) {
    EXPECT_GE(policy[i], 0.0);
  }
}

// NOLINTNEXTLINE
TEST(MCTS, PlayoutEval) {
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
  auto best = MCTS::pick_move(mcts.probs(0));
  std::cout << "Playout MCTS best move: " << best << std::endl;
  EXPECT_GE(best, 0);
  EXPECT_LT(best, gs.num_moves());
}

// NOLINTNEXTLINE
TEST(MCTS, RootValueSetOnFirstEval) {
  auto gs = connect4_gs::Connect4GS{};
  auto mcts = MCTS{2, gs.num_players(), gs.num_moves(), 0, 1.4, 0.25};

  {
    auto leaf = mcts.find_leaf(gs);
    auto [value, pi] = dumb_eval(*leaf);
    mcts.process_result(gs, value, pi);
  }

  while (mcts.depth() < 31) {
    auto leaf = mcts.find_leaf(gs);
    auto [value, pi] = dumb_eval(*leaf);
    mcts.process_result(gs, value, pi);
  }

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
  auto gs = connect4_gs::Connect4GS{};
  gs.play_move(1);
  gs.play_move(6);
  gs.play_move(3);
  gs.play_move(6);

  auto mcts = MCTS{2, gs.num_players(), gs.num_moves(), 0.25, 1.4, 0.0, false};
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

  bool any_diff = false;
  for (uint32_t m = 0; m < gs.num_moves(); ++m) {
    if (std::abs(regular(m) - pruned(m)) > 1e-6) any_diff = true;
  }
  EXPECT_TRUE(any_diff) << "probs_pruned should differ from probs when noise was applied "
                         << "in an asymmetric position.\nregular: " << regular.transpose()
                         << "\npruned:  " << pruned.transpose();

  EXPECT_NEAR(pruned.sum(), 1.0, 1e-5);
}

// NOLINTNEXTLINE
TEST(MCTS, RootFpuZero) {
  auto gs = connect4_gs::Connect4GS{};

  auto mcts_normal = MCTS{2, gs.num_players(), gs.num_moves(), 0.0, 1.0, 0.25,
                           false, false};
  while (mcts_normal.depth() < 50) {
    auto leaf = mcts_normal.find_leaf(gs);
    auto [v, pi] = dumb_eval(*leaf);
    mcts_normal.process_result(gs, v, pi);
  }

  auto mcts_fpu0 = MCTS{2, gs.num_players(), gs.num_moves(), 0.0, 1.0, 0.25,
                          false, true};
  while (mcts_fpu0.depth() < 50) {
    auto leaf = mcts_fpu0.find_leaf(gs);
    auto [v, pi] = dumb_eval(*leaf);
    mcts_fpu0.process_result(gs, v, pi);
  }

  auto counts_normal = mcts_normal.counts();
  auto counts_fpu0 = mcts_fpu0.counts();

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

  float best_regular = regular.maxCoeff();
  float best_pruned = pruned.maxCoeff();
  EXPECT_GE(best_pruned + 0.01, best_regular)
      << "Best move should gain or maintain share after pruning";

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

  auto mcts_shaped = MCTS{2, gs.num_players(), gs.num_moves(), 0.25, 1.4, 0.0,
                           false, false, true};
  {
    auto leaf = mcts_shaped.find_leaf(gs);
    auto [v, pi] = dumb_eval(*leaf);
    mcts_shaped.process_result(gs, v, pi, true);
  }
  auto probs = mcts_shaped.probs(1.0);
  float psum = 0;
  for (int m = 0; m < gs.num_moves(); ++m) {
    EXPECT_GE(probs(m), 0.0f);
    psum += probs(m);
  }
  EXPECT_NEAR(psum, 1.0, 1e-4);

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
  auto gs = connect4_gs::Connect4GS{};
  gs.play_move(3);
  gs.play_move(0);
  gs.play_move(3);
  gs.play_move(0);

  const int TRIALS = 50;
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

  EXPECT_GE(pruned.maxCoeff() + 0.1, regular.maxCoeff());

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
  auto gs = connect4_gs::Connect4GS{};
  gs.play_move(1);
  gs.play_move(6);
  gs.play_move(3);
  gs.play_move(6);

  auto mcts_eval = MCTS{2, gs.num_players(), gs.num_moves(), 0.0, 1.0, 0.0,
                         false, false, false};
  while (mcts_eval.depth() < 200) {
    auto leaf = mcts_eval.find_leaf(gs);
    auto [v, pi] = dumb_eval(*leaf);
    mcts_eval.process_result(gs, v, pi, false);
  }
  auto eval_regular = mcts_eval.probs(1.0);
  auto eval_pruned = mcts_eval.probs_pruned(1.0);
  EXPECT_NEAR(eval_regular.sum(), 1.0, 1e-5);
  EXPECT_NEAR(eval_pruned.sum(), 1.0, 1e-5);

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
  bool any_diff = false;
  for (int m = 0; m < gs.num_moves(); ++m) {
    if (std::abs(sp_regular(m) - sp_pruned(m)) > 1e-6) any_diff = true;
  }
  EXPECT_TRUE(any_diff) << "Self-play mode: noise should cause pruning differences"
                         << "\nregular: " << sp_regular.transpose()
                         << "\npruned:  " << sp_pruned.transpose();

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
    mcts.process_result(gs, v, pi, true);
  }

  auto p = mcts.probs(1.0);
  EXPECT_NEAR(p.sum(), 1.0, 1e-4);

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

  float best_regular_val = regular.maxCoeff();
  for (int m = 0; m < gs.num_moves(); ++m) {
    if (regular(m) == best_regular_val) continue;
    EXPECT_LE(pruned(m), regular(m) + 0.01)
        << "Non-best move " << m << " should not gain significant share after pruning"
        << "\nregular: " << regular.transpose()
        << "\npruned:  " << pruned.transpose();
  }
}

// NOLINTNEXTLINE
TEST(MCTS, BatchedBasic) {
  auto gs = connect4_gs::Connect4GS{};
  gs.play_move(1);
  gs.play_move(6);
  gs.play_move(3);
  gs.play_move(6);
  auto mcts = MCTS{2, gs.num_players(), gs.num_moves()};
  constexpr int BATCH_SIZE = 8;
  constexpr int TOTAL_SIMS = 800;
  int sims = 0;
  while (sims < TOTAL_SIMS) {
    int batch = std::min(BATCH_SIZE, TOTAL_SIMS - sims);
    for (int i = 0; i < batch; ++i) {
      auto leaf = mcts.find_leaf_batched(gs);
      auto [value, pi] = dumb_eval(*leaf);
      mcts.process_result_batched(gs, i, value, pi);
    }
    mcts.reset_batch();
    sims += batch;
  }
  auto counts = mcts.counts();
  std::cout << "Batched counts: " << counts << std::endl;
  EXPECT_EQ(MCTS::pick_move(mcts.probs(0)), 2);
}

// NOLINTNEXTLINE
TEST(MCTS, BatchedTerminal) {
  auto gs = connect4_gs::Connect4GS{};
  gs.play_move(3);
  gs.play_move(0);
  gs.play_move(3);
  gs.play_move(0);
  gs.play_move(3);
  gs.play_move(1);

  auto mcts = MCTS{2, gs.num_players(), gs.num_moves()};
  constexpr int BATCH_SIZE = 4;
  int sims = 0;
  while (sims < 100) {
    int batch = std::min(BATCH_SIZE, 100 - sims);
    for (int i = 0; i < batch; ++i) {
      auto leaf = mcts.find_leaf_batched(gs);
      auto [value, pi] = dumb_eval(*leaf);
      mcts.process_result_batched(gs, i, value, pi);
    }
    mcts.reset_batch();
    sims += batch;
  }
  auto best = MCTS::pick_move(mcts.probs(0));
  EXPECT_EQ(best, 3);
}

// NOLINTNEXTLINE
TEST(MCTS, BatchedSingleEquivalent) {
  auto gs = connect4_gs::Connect4GS{};
  gs.play_move(1);
  gs.play_move(6);
  gs.play_move(3);
  gs.play_move(6);

  auto mcts_unbatched = MCTS{2, gs.num_players(), gs.num_moves()};
  while (mcts_unbatched.depth() < 800) {
    auto leaf = mcts_unbatched.find_leaf(gs);
    auto [value, pi] = dumb_eval(*leaf);
    mcts_unbatched.process_result(gs, value, pi);
  }

  auto mcts_batched = MCTS{2, gs.num_players(), gs.num_moves()};
  while (mcts_batched.depth() < 800) {
    auto leaf = mcts_batched.find_leaf_batched(gs);
    auto [value, pi] = dumb_eval(*leaf);
    mcts_batched.process_result_batched(gs, 0, value, pi);
    mcts_batched.reset_batch();
  }

  auto counts_u = mcts_unbatched.counts();
  auto counts_b = mcts_batched.counts();
  auto best_u = MCTS::pick_move(mcts_unbatched.probs(0));
  auto best_b = MCTS::pick_move(mcts_batched.probs(0));
  EXPECT_EQ(best_u, best_b)
      << "Best move differs.\nunbatched counts: " << counts_u.transpose()
      << "\nbatched counts:   " << counts_b.transpose();

  EXPECT_EQ(counts_u.sum(), counts_b.sum());
}

// NOLINTNEXTLINE
TEST(MCTS, WUUCTDiversity) {
  auto gs = connect4_gs::Connect4GS{};
  auto mcts = MCTS{2, gs.num_players(), gs.num_moves()};

  {
    auto leaf = mcts.find_leaf(gs);
    auto [v, pi] = dumb_eval(*leaf);
    mcts.process_result(gs, v, pi);
  }

  std::vector<std::string> leaf_states;
  for (int i = 0; i < 4; ++i) {
    auto leaf = mcts.find_leaf_batched(gs);
    leaf_states.push_back(leaf->dump());
  }
  for (int i = 0; i < 4; ++i) {
    auto leaf_gs = gs.copy();
    auto [v, pi] = dumb_eval(*leaf_gs);
    mcts.process_result_batched(gs, i, v, pi);
  }
  mcts.reset_batch();

  std::set<std::string> unique_states(leaf_states.begin(), leaf_states.end());
  EXPECT_GE(unique_states.size(), 3u)
      << "WU-UCT should diversify leaf selection across children";
}

// --- MCGS Tests ---

// NOLINTNEXTLINE
TEST(MCGS, Basic) {
  // Same position as MCTS::Basic but with mcgs=true.
  auto gs = connect4_gs::Connect4GS{};
  gs.play_move(1);
  gs.play_move(6);
  gs.play_move(3);
  gs.play_move(6);
  auto mcts = MCTS{2, gs.num_players(), gs.num_moves(), 0, 1.0, 0, false,
                    false, false, true};
  while (mcts.depth() < 800) {
    auto leaf = mcts.find_leaf(gs);
    auto [value, pi] = dumb_eval(*leaf);
    mcts.process_result(gs, value, pi);
  }
  auto counts = mcts.counts();
  std::cout << "MCGS counts: " << counts << std::endl;
  EXPECT_EQ(MCTS::pick_move(mcts.probs(0)), 2);
  EXPECT_GT(mcts.tt_size(), 0u);
}

// NOLINTNEXTLINE
TEST(MCGS, TTStats) {
  auto gs = connect4_gs::Connect4GS{};
  gs.play_move(1);
  gs.play_move(6);
  gs.play_move(3);
  gs.play_move(6);

  // With MCGS
  auto mcts_g = MCTS{2, gs.num_players(), gs.num_moves(), 0, 1.0, 0, false,
                      false, false, true};
  while (mcts_g.depth() < 200) {
    auto leaf = mcts_g.find_leaf(gs);
    auto [v, pi] = dumb_eval(*leaf);
    mcts_g.process_result(gs, v, pi);
  }
  EXPECT_GE(mcts_g.tt_hits(), 0u);
  EXPECT_GT(mcts_g.tt_size(), 0u);

  // Without MCGS (explicitly disable)
  auto mcts_n = MCTS{2, gs.num_players(), gs.num_moves(), 0, 1.0, 0, false,
                      false, false, false};
  while (mcts_n.depth() < 200) {
    auto leaf = mcts_n.find_leaf(gs);
    auto [v, pi] = dumb_eval(*leaf);
    mcts_n.process_result(gs, v, pi);
  }
  EXPECT_EQ(mcts_n.tt_hits(), 0u);
  EXPECT_EQ(mcts_n.tt_size(), 0u);
}

// NOLINTNEXTLINE
TEST(MCGS, Consistency) {
  auto gs = connect4_gs::Connect4GS{};
  gs.play_move(1);
  gs.play_move(6);
  gs.play_move(3);
  gs.play_move(6);

  auto mcts_off = MCTS{2, gs.num_players(), gs.num_moves(), 0, 1.0, 0, false,
                        false, false, false};
  while (mcts_off.depth() < 800) {
    auto leaf = mcts_off.find_leaf(gs);
    auto [v, pi] = dumb_eval(*leaf);
    mcts_off.process_result(gs, v, pi);
  }

  auto mcts_on = MCTS{2, gs.num_players(), gs.num_moves(), 0, 1.0, 0, false,
                       false, false, true};
  while (mcts_on.depth() < 800) {
    auto leaf = mcts_on.find_leaf(gs);
    auto [v, pi] = dumb_eval(*leaf);
    mcts_on.process_result(gs, v, pi);
  }

  auto best_off = MCTS::pick_move(mcts_off.probs(0));
  auto best_on = MCTS::pick_move(mcts_on.probs(0));
  EXPECT_EQ(best_off, best_on)
      << "MCGS and non-MCGS should agree on best move";
  EXPECT_EQ(mcts_off.root_n(), 800u);
  EXPECT_EQ(mcts_on.root_n(), 800u);
}

// NOLINTNEXTLINE
TEST(MCGS, TreeReuse) {
  auto gs = connect4_gs::Connect4GS{};
  gs.play_move(1);
  gs.play_move(6);
  gs.play_move(3);
  gs.play_move(6);

  auto mcts = MCTS{2, gs.num_players(), gs.num_moves(), 0, 1.0, 0, false,
                    false, false, true};
  while (mcts.depth() < 200) {
    auto leaf = mcts.find_leaf(gs);
    auto [v, pi] = dumb_eval(*leaf);
    mcts.process_result(gs, v, pi);
  }
  auto tt_before = mcts.tt_size();
  EXPECT_GT(tt_before, 0u);
  auto root_n_before = mcts.root_n();

  // Pick best move and update root
  auto best = MCTS::pick_move(mcts.probs(0));
  gs.play_move(best);
  mcts.update_root(gs, best);

  // Root carries over child's accumulated visits from prior search
  auto root_n_after = mcts.root_n();
  EXPECT_LE(root_n_after, root_n_before);
  // tt_size should be <= tt_before after cleanup (orphaned nodes removed)
  EXPECT_LE(mcts.tt_size(), tt_before);

  // Run more sims — should still work
  auto depth_before = mcts.depth();
  while (mcts.depth() < depth_before + 100) {
    auto leaf = mcts.find_leaf(gs);
    auto [v, pi] = dumb_eval(*leaf);
    mcts.process_result(gs, v, pi);
  }
  // Root_n should have increased by 100
  EXPECT_EQ(mcts.root_n(), root_n_after + 100);
}

// NOLINTNEXTLINE
TEST(MCGS, BatchedBasic) {
  auto gs = connect4_gs::Connect4GS{};
  gs.play_move(1);
  gs.play_move(6);
  gs.play_move(3);
  gs.play_move(6);
  auto mcts = MCTS{2, gs.num_players(), gs.num_moves(), 0, 1.0, 0, false,
                    false, false, true};
  constexpr int BATCH_SIZE = 8;
  constexpr int TOTAL_SIMS = 800;
  int sims = 0;
  while (sims < TOTAL_SIMS) {
    int batch = std::min(BATCH_SIZE, TOTAL_SIMS - sims);
    for (int i = 0; i < batch; ++i) {
      auto leaf = mcts.find_leaf_batched(gs);
      auto [value, pi] = dumb_eval(*leaf);
      mcts.process_result_batched(gs, i, value, pi);
    }
    mcts.reset_batch();
    sims += batch;
  }
  auto counts = mcts.counts();
  std::cout << "MCGS Batched counts: " << counts << std::endl;
  EXPECT_EQ(MCTS::pick_move(mcts.probs(0)), 2);
  EXPECT_GT(mcts.tt_size(), 0u);
}

// NOLINTNEXTLINE
TEST(MCGS, SharedNodeVisits) {
  // With edge-based counts, child_total == root_sims_ (no inflation).
  auto gs = connect4_gs::Connect4GS{};
  gs.play_move(1);
  gs.play_move(6);
  gs.play_move(3);
  gs.play_move(6);
  auto mcts = MCTS{2, gs.num_players(), gs.num_moves(), 0, 1.0, 0, false,
                    false, false, true};
  while (mcts.depth() < 400) {
    auto leaf = mcts.find_leaf(gs);
    auto [v, pi] = dumb_eval(*leaf);
    mcts.process_result(gs, v, pi);
  }

  auto counts = mcts.counts();
  uint32_t child_total = 0;
  for (int i = 0; i < counts.size(); ++i) {
    child_total += counts(i);
  }

  // Edge-based counts: child_total = root_sims - 1 (first sim evaluates root, no edges)
  EXPECT_EQ(child_total + 1, mcts.root_n());
  std::cout << "root_n=" << mcts.root_n() << " child_total=" << child_total
            << " tt_hits=" << mcts.tt_hits() << std::endl;
  // Root N should always be exactly the number of sims
  EXPECT_EQ(mcts.root_n(), 400u);
}

// NOLINTNEXTLINE
TEST(MCGS, AllExistingModes) {
  // Smoke test: MCGS with noise, FPU reduction, root_fpu_zero, shaped Dirichlet.
  auto gs = connect4_gs::Connect4GS{};
  gs.play_move(1);
  gs.play_move(6);
  gs.play_move(3);
  gs.play_move(6);

  auto mcts = MCTS{2, gs.num_players(), gs.num_moves(), 0.25, 1.4, 0.25,
                    false, true, true, true};
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
  auto p = mcts.probs(1.0);
  EXPECT_NEAR(p.sum(), 1.0, 1e-4);
  EXPECT_GT(mcts.tt_size(), 0u);
}

// NOLINTNEXTLINE
TEST(MCTS, FreeListReuse) {
  auto gs = connect4_gs::Connect4GS{};
  gs.play_move(1);
  gs.play_move(6);
  gs.play_move(3);
  gs.play_move(6);
  auto mcts = MCTS{2, gs.num_players(), gs.num_moves()};
  while (mcts.depth() < 200) {
    auto leaf = mcts.find_leaf(gs);
    auto [v, pi] = dumb_eval(*leaf);
    mcts.process_result(gs, v, pi);
  }

  // Pick best move and update root — this should release nodes to free list
  auto best = MCTS::pick_move(mcts.probs(0));
  gs.play_move(best);
  mcts.update_root(gs, best);

  // Run more sims — should reuse freed nodes and still work correctly
  auto depth_before = mcts.depth();
  while (mcts.depth() < depth_before + 200) {
    auto leaf = mcts.find_leaf(gs);
    auto [v, pi] = dumb_eval(*leaf);
    mcts.process_result(gs, v, pi);
  }
  EXPECT_EQ(mcts.depth(), depth_before + 200);

  // Do another update_root
  best = MCTS::pick_move(mcts.probs(0));
  gs.play_move(best);
  mcts.update_root(gs, best);

  auto depth_before2 = mcts.depth();
  while (mcts.depth() < depth_before2 + 100) {
    auto leaf = mcts.find_leaf(gs);
    auto [v, pi] = dumb_eval(*leaf);
    mcts.process_result(gs, v, pi);
  }
  EXPECT_EQ(mcts.depth(), depth_before2 + 100);
}

// NOLINTNEXTLINE
TEST(MCGS, FreeListReuse) {
  auto gs = connect4_gs::Connect4GS{};
  gs.play_move(1);
  gs.play_move(6);
  gs.play_move(3);
  gs.play_move(6);
  auto mcts = MCTS{2, gs.num_players(), gs.num_moves(), 0, 1.0, 0, false,
                    false, false, true};
  while (mcts.depth() < 200) {
    auto leaf = mcts.find_leaf(gs);
    auto [v, pi] = dumb_eval(*leaf);
    mcts.process_result(gs, v, pi);
  }
  auto tt_before = mcts.tt_size();
  EXPECT_GT(tt_before, 0u);

  // Pick best move and update root — should release nodes + sweep TT
  auto best = MCTS::pick_move(mcts.probs(0));
  gs.play_move(best);
  mcts.update_root(gs, best);

  // TT should have been cleaned up
  EXPECT_LE(mcts.tt_size(), tt_before);

  // Run more sims — should reuse freed nodes and TT should still work
  auto depth_before = mcts.depth();
  while (mcts.depth() < depth_before + 200) {
    auto leaf = mcts.find_leaf(gs);
    auto [v, pi] = dumb_eval(*leaf);
    mcts.process_result(gs, v, pi);
  }
  EXPECT_EQ(mcts.depth(), depth_before + 200);
  EXPECT_GT(mcts.tt_size(), 0u);

  // Another round of update_root + search
  best = MCTS::pick_move(mcts.probs(0));
  gs.play_move(best);
  mcts.update_root(gs, best);

  depth_before = mcts.depth();
  while (mcts.depth() < depth_before + 100) {
    auto leaf = mcts.find_leaf(gs);
    auto [v, pi] = dumb_eval(*leaf);
    mcts.process_result(gs, v, pi);
  }
  EXPECT_GT(mcts.root_n(), 0u);
}

// NOLINTNEXTLINE
TEST(MCGS, CycleDetection) {
  // Star Gambit can produce cyclic game states (pieces moving back and forth).
  // With MCGS, transposition table merging can create graph cycles.
  // This test verifies that MCGS completes without infinite loops.
  auto gs = star_gambit_gs::StarGambitSkirmishGS{};
  auto mcts = MCTS{2, gs.num_players(), gs.num_moves(), 0, 1.0, 0, false,
                    false, false, true};
  constexpr int TOTAL_SIMS = 500;
  while (mcts.depth() < TOTAL_SIMS) {
    auto leaf = mcts.find_leaf(gs);
    auto [v, pi] = dumb_eval(*leaf);
    mcts.process_result(gs, v, pi);
  }
  EXPECT_EQ(mcts.root_n(), TOTAL_SIMS);
  EXPECT_GT(mcts.tt_size(), 0u);

  // Also test batched path
  auto gs2 = star_gambit_gs::StarGambitSkirmishGS{};
  auto mcts2 = MCTS{2, gs2.num_players(), gs2.num_moves(), 0, 1.0, 0, false,
                     false, false, true};
  constexpr int BATCH_SIZE = 8;
  int sims = 0;
  while (sims < TOTAL_SIMS) {
    int batch = std::min(BATCH_SIZE, TOTAL_SIMS - sims);
    for (int i = 0; i < batch; ++i) {
      auto leaf = mcts2.find_leaf_batched(gs2);
      auto [v, pi] = dumb_eval(*leaf);
      mcts2.process_result_batched(gs2, i, v, pi);
    }
    mcts2.reset_batch();
    sims += batch;
  }
  EXPECT_EQ(mcts2.root_n(), static_cast<uint32_t>(TOTAL_SIMS));
  EXPECT_GT(mcts2.tt_size(), 0u);
}

// NOLINTNEXTLINE
TEST(MCGS, CycleSafeUpdateRoot) {
  // Regression test: MCGS cycle detection (Case 3 redirect) can create graph
  // cycles (A→B→A). The old update_root force-freed the root node regardless
  // of ref_count, leaving dangling pointers from cycle edges → segfault.
  // This test verifies that multiple rounds of search + update_root complete
  // without crashing on a game that produces cyclic transpositions.
  auto gs = star_gambit_gs::StarGambitSkirmishGS{};
  auto mcts = MCTS{2, gs.num_players(), gs.num_moves(), 0, 1.0, 0, false,
                    false, false, true};

  constexpr int ROUNDS = 6;
  constexpr int SIMS_PER_ROUND = 200;

  for (int round = 0; round < ROUNDS; ++round) {
    if (gs.scores().has_value()) break;

    while (mcts.depth() < static_cast<uint32_t>((round + 1) * SIMS_PER_ROUND)) {
      auto leaf = mcts.find_leaf(gs);
      auto [v, pi] = dumb_eval(*leaf);
      mcts.process_result(gs, v, pi);
    }

    auto best = MCTS::pick_move(mcts.probs(0));
    gs.play_move(best);
    mcts.update_root(gs, best);  // must not crash with cycle edges
  }

  // If we get here without segfault, the fix works.
  EXPECT_GT(mcts.root_n(), 0u);
}

// NOLINTNEXTLINE
TEST(MCGS, CycleSafeUpdateRootBatched) {
  // Same as CycleSafeUpdateRoot but with batched search path.
  auto gs = star_gambit_gs::StarGambitSkirmishGS{};
  auto mcts = MCTS{2, gs.num_players(), gs.num_moves(), 0, 1.0, 0, false,
                    false, false, true};

  constexpr int ROUNDS = 6;
  constexpr int SIMS_PER_ROUND = 200;
  constexpr int BATCH_SIZE = 8;

  for (int round = 0; round < ROUNDS; ++round) {
    if (gs.scores().has_value()) break;

    auto target = static_cast<uint32_t>((round + 1) * SIMS_PER_ROUND);
    while (mcts.depth() < target) {
      int remaining = static_cast<int>(target - mcts.depth());
      int batch = std::min(BATCH_SIZE, remaining);
      for (int i = 0; i < batch; ++i) {
        auto leaf = mcts.find_leaf_batched(gs);
        auto [v, pi] = dumb_eval(*leaf);
        mcts.process_result_batched(gs, i, v, pi);
      }
      mcts.reset_batch();
    }

    auto best = MCTS::pick_move(mcts.probs(0));
    gs.play_move(best);
    mcts.update_root(gs, best);  // must not crash with cycle edges
  }

  EXPECT_GT(mcts.root_n(), 0u);
}

// Helper: create a heavily skewed pi that differs from dumb_eval's uniform
// output. Uses an exponential ramp so that after masking/renormalization in
// process_result, the resulting edge policies are visibly non-uniform.
static Vector<float> make_skewed_pi(uint32_t num_moves) {
  auto pi = Vector<float>{num_moves};
  for (int m = 0; m < static_cast<int>(num_moves); ++m) {
    pi(m) = std::exp(static_cast<float>(m) * 0.01f);
  }
  return pi;
}

// NOLINTNEXTLINE
TEST(MCGS, PolicyCorruptionOnCycleLeaf) {
  // Bug: process_result(_batched) unconditionally calls update_policy() on
  // already-evaluated cycle nodes, overwriting their NN-derived policies
  // with the new pi vector. This test measures REAL corruption: the L2
  // distance between the node's policies before and after calling
  // process_result with a deliberately skewed pi.
  //
  // Both batched and unbatched paths are tested in the same test to ensure
  // at least one path produces cycle leaves (batched is more reliable due
  // to virtual losses pushing exploration into cycle-forming edges).
  auto gs = star_gambit_gs::StarGambitSkirmishGS{};
  constexpr int TOTAL_SIMS = 50000;
  int no_eval_count = 0;
  double total_policy_drift = 0.0;

  // --- Unbatched path ---
  {
    auto mcts = MCTS{2, gs.num_players(), gs.num_moves(), 0, 1.0, 0, false,
                      false, false, true};
    while (mcts.depth() < TOTAL_SIMS) {
      auto leaf = mcts.find_leaf(gs);
      if (!mcts.leaf_needs_eval()) {
        ++no_eval_count;
        auto policies_before = mcts.leaf_node_policies();

        auto bad_v = Vector<float>{gs.num_players() + 1};
        bad_v.setConstant(1.0f / (gs.num_players() + 1));
        auto bad_pi = make_skewed_pi(gs.num_moves());
        mcts.process_result(gs, bad_v, bad_pi);

        auto& policies_after = mcts.leaf_node_policies();
        for (size_t j = 0; j < policies_before.size(); ++j) {
          double diff = policies_before[j] - policies_after[j];
          total_policy_drift += diff * diff;
        }
      } else {
        auto [v, pi] = dumb_eval(*leaf);
        mcts.process_result(gs, v, pi);
      }
    }
    std::cout << "  unbatched: cycle_leaves=" << no_eval_count
              << " policy_drift_L2=" << total_policy_drift
              << " tt_hits=" << mcts.tt_hits() << std::endl;
  }

  // --- Batched path ---
  {
    auto mcts = MCTS{2, gs.num_players(), gs.num_moves(), 0, 1.0, 0, false,
                      false, false, true};
    constexpr int BATCH = 8;
    int sims = 0;
    while (sims < TOTAL_SIMS) {
      int batch = std::min(BATCH, TOTAL_SIMS - sims);
      for (int i = 0; i < batch; ++i) {
        auto leaf = mcts.find_leaf_batched(gs);
        if (!mcts.leaf_needs_eval()) {
          ++no_eval_count;
          auto policies_before = mcts.leaf_node_policies();

          auto bad_v = Vector<float>{gs.num_players() + 1};
          bad_v.setConstant(1.0f / (gs.num_players() + 1));
          auto bad_pi = make_skewed_pi(gs.num_moves());
          mcts.process_result_batched(gs, i, bad_v, bad_pi);

          auto& policies_after = mcts.leaf_node_policies();
          for (size_t j = 0; j < policies_before.size(); ++j) {
            double diff = policies_before[j] - policies_after[j];
            total_policy_drift += diff * diff;
          }
        } else {
          auto [v, pi] = dumb_eval(*leaf);
          mcts.process_result_batched(gs, i, v, pi);
        }
      }
      mcts.reset_batch();
      sims += batch;
    }
    std::cout << "  batched: cycle_leaves=" << no_eval_count
              << " policy_drift_L2=" << total_policy_drift
              << " tt_hits=" << mcts.tt_hits() << std::endl;
  }

  ASSERT_GT(no_eval_count, 0) << "Expected cycle leaves in " << TOTAL_SIMS
                               << " sims on Star Gambit (both paths combined)";
  // Before fix: total_policy_drift > 0 (policies overwritten with skewed pi)
  // After fix: total_policy_drift == 0 (policies untouched)
  EXPECT_DOUBLE_EQ(total_policy_drift, 0.0)
      << no_eval_count << " cycle leaves had their policies corrupted by "
      << "process_result overwriting already-evaluated nodes";
}

// NOLINTNEXTLINE
TEST(MCGS, BackpropLeafBatchedCorrectness) {
  // After the fix: cycle/terminal leaves use backprop_leaf_batched instead
  // of going through the NN eval pipeline. Verify search integrity.
  auto gs = star_gambit_gs::StarGambitSkirmishGS{};
  auto mcts = MCTS{2, gs.num_players(), gs.num_moves(), 0, 1.0, 0, false,
                    false, false, true};
  constexpr int BATCH_SIZE = 8;
  constexpr int TOTAL_SIMS = 50000;

  int nn_evals = 0;
  int short_circuited = 0;
  int sims = 0;
  while (sims < TOTAL_SIMS) {
    int batch = std::min(BATCH_SIZE, TOTAL_SIMS - sims);
    for (int i = 0; i < batch; ++i) {
      auto leaf = mcts.find_leaf_batched(gs);
      if (!mcts.leaf_needs_eval()) {
        mcts.backprop_leaf_batched(i);
        ++short_circuited;
      } else {
        auto [v, pi] = dumb_eval(*leaf);
        mcts.process_result_batched(gs, i, v, pi);
        ++nn_evals;
      }
    }
    mcts.reset_batch();
    sims += batch;
  }
  std::cout << "  nn_evals=" << nn_evals << " short_circuited=" << short_circuited
            << std::endl;
  EXPECT_GT(short_circuited, 0) << "Expected some cycle/terminal leaves to skip NN eval";
  EXPECT_EQ(mcts.root_n(), static_cast<uint32_t>(TOTAL_SIMS));
  auto p = mcts.probs(0);
  EXPECT_NEAR(p.sum(), 1.0, 1e-4);
}

// NOLINTNEXTLINE
TEST(MCGS, BackpropLeafUnbatchedCorrectness) {
  // Same for unbatched path.
  auto gs = star_gambit_gs::StarGambitSkirmishGS{};
  auto mcts = MCTS{2, gs.num_players(), gs.num_moves(), 0, 1.0, 0, false,
                    false, false, true};
  constexpr int SIMS = 50000;

  int nn_evals = 0;
  int short_circuited = 0;
  while (mcts.depth() < SIMS) {
    auto leaf = mcts.find_leaf(gs);
    if (!mcts.leaf_needs_eval()) {
      mcts.backprop_leaf();
      ++short_circuited;
    } else {
      auto [v, pi] = dumb_eval(*leaf);
      mcts.process_result(gs, v, pi);
      ++nn_evals;
    }
  }
  std::cout << "  nn_evals=" << nn_evals << " short_circuited=" << short_circuited
            << std::endl;
  EXPECT_GT(short_circuited, 0) << "Expected some cycle/terminal leaves to skip NN eval";
  EXPECT_EQ(mcts.root_n(), static_cast<uint32_t>(SIMS));
  auto p = mcts.probs(0);
  EXPECT_NEAR(p.sum(), 1.0, 1e-4);
}

}  // namespace
}  // namespace alphazero
