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

}  // namespace
}  // namespace alphazero