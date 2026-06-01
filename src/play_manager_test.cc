#include "play_manager.h"

#include <future>

#include "connect4_gs.h"
#include "gtest/gtest.h"

namespace alphazero {
namespace {

// NOLINTNEXTLINE
TEST(PlayManager, Basic) {
  auto params = PlayParams{};
  params.games_to_play = 32;
  params.concurrent_games = 8;
  params.mcts_visits = {10, 10};
  params.eval_type = {EvalType::RANDOM, EvalType::RANDOM};
  params.history_enabled = true;
  params.playout_cap_randomization = true;
  auto pm = PlayManager{std::make_unique<connect4_gs::Connect4GS>(), params};
  auto play = std::async(std::launch::async, [&] {
    try {
      pm.play();
    } catch (const std::exception& e) {
      FAIL() << "Got an exception: " << e.what() << std::endl;
    }
  });
  play.wait();
}

TEST(PlayManager, MultiThreaded) {
  const auto cores = std::thread::hardware_concurrency();
  const auto workers = cores - 1;

  auto params = PlayParams{};
  params.games_to_play = 32 * workers;
  params.concurrent_games = 8 * workers;
  params.mcts_visits = {10, 10};
  params.eval_type = {EvalType::RANDOM, EvalType::RANDOM};

  auto pm = PlayManager{std::make_unique<connect4_gs::Connect4GS>(), params};
  auto play_workers = std::vector<std::future<void>>{workers};
  for (auto& pw : play_workers) {
    pw = std::async(std::launch::async, [&] {
      try {
        pm.play();
      } catch (const std::exception& e) {
        FAIL() << "Got an exception: " << e.what() << std::endl;
      }
    });
  }
  for (auto& pw : play_workers) {
    pw.wait();
  }
  std::cout << "Scores: " << pm.scores() << std::endl;
}

TEST(PlayManager, StopEarly) {
  auto params = PlayParams{};
  params.games_to_play = 1000000;
  params.concurrent_games = 8;
  params.mcts_visits = {10, 10};
  params.eval_type = {EvalType::RANDOM, EvalType::RANDOM};
  auto pm = PlayManager{std::make_unique<connect4_gs::Connect4GS>(), params};
  auto play = std::async(std::launch::async, [&] { pm.play(); });
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  pm.stop();
  auto status = play.wait_for(std::chrono::seconds(1));
  ASSERT_EQ(status, std::future_status::ready);
  EXPECT_TRUE(pm.stopped());
  EXPECT_EQ(pm.remaining_games(), 0U);
}

TEST(PlayManager, SeatEpsilonWrongOuterDimThrows) {
  auto params = PlayParams{};
  params.games_to_play = 4;
  params.concurrent_games = 2;
  params.mcts_visits = {10, 10};
  params.eval_type = {EvalType::RANDOM, EvalType::RANDOM};
  // Default seat_perms will be 1 perm, but seat_epsilon has 2.
  params.seat_epsilon = {{0.25, 0.0}, {0.0, 0.25}};
  EXPECT_THROW(
      PlayManager(std::make_unique<connect4_gs::Connect4GS>(), params),
      std::runtime_error);
}

TEST(PlayManager, SeatEpsilonWrongInnerDimThrows) {
  auto params = PlayParams{};
  params.games_to_play = 4;
  params.concurrent_games = 2;
  params.mcts_visits = {10, 10};
  params.eval_type = {EvalType::RANDOM, EvalType::RANDOM};
  // 1 perm (default), but inner dim is 3 instead of 2.
  params.seat_epsilon = {{0.25, 0.0, 0.1}};
  EXPECT_THROW(
      PlayManager(std::make_unique<connect4_gs::Connect4GS>(), params),
      std::runtime_error);
}

TEST(PlayManager, SeatVisitsWrongDimThrows) {
  auto params = PlayParams{};
  params.games_to_play = 4;
  params.concurrent_games = 2;
  params.mcts_visits = {10, 10};
  params.eval_type = {EvalType::RANDOM, EvalType::RANDOM};
  // Default seat_perms will be 1 perm, but seat_visits has 3.
  params.seat_visits = {{10, 10}, {10, 10}, {10, 10}};
  EXPECT_THROW(
      PlayManager(std::make_unique<connect4_gs::Connect4GS>(), params),
      std::runtime_error);
}

TEST(PlayManager, SeatOverridesDefaultFill) {
  // Empty seat overrides should not throw (filled from globals).
  auto params = PlayParams{};
  params.games_to_play = 4;
  params.concurrent_games = 2;
  params.mcts_visits = {10, 10};
  params.eval_type = {EvalType::RANDOM, EvalType::RANDOM};
  params.epsilon = 0.25;
  EXPECT_NO_THROW(
      PlayManager(std::make_unique<connect4_gs::Connect4GS>(), params));
}

TEST(PlayManager, PerSeatOverrides) {
  auto params = PlayParams{};
  params.games_to_play = 32;
  params.concurrent_games = 8;
  params.mcts_visits = {10, 10};
  params.eval_type = {EvalType::RANDOM, EvalType::RANDOM};
  params.seat_perms = {{0, 1}, {1, 0}};
  params.seat_epsilon = {{0.25, 0.0}, {0.0, 0.25}};
  params.seat_mcts_root_temp = {{1.25, 1.0}, {1.0, 1.25}};
  params.seat_root_fpu_zero = {{1, 0}, {0, 1}};
  params.seat_visits = {{10, 10}, {10, 10}};

  auto pm = PlayManager{std::make_unique<connect4_gs::Connect4GS>(), params};
  auto play = std::async(std::launch::async, [&] {
    try {
      pm.play();
    } catch (const std::exception& e) {
      FAIL() << "Got an exception: " << e.what() << std::endl;
    }
  });
  play.wait();
  EXPECT_EQ(pm.games_completed(), 32U);
}

TEST(PlayManager, InitOrderPerPermMctsSettings) {
  // Verify that MCTS instances get the correct per-perm settings at
  // construction time.  Before the fix, perm_index was assigned AFTER
  // make_mcts(), so every game got perm 0's settings.
  auto params = PlayParams{};
  params.games_to_play = 100;  // won't run, just need construction
  params.concurrent_games = 4;
  params.mcts_visits = {10, 10};
  params.eval_type = {EvalType::RANDOM, EvalType::RANDOM};
  params.seat_perms = {{0, 1}, {1, 0}};
  // Perm 0: seat 0 gets epsilon=0.25, seat 1 gets 0.0
  // Perm 1: seat 0 gets epsilon=0.0,  seat 1 gets 0.5
  params.seat_epsilon = {{0.25, 0.0}, {0.0, 0.5}};
  params.seat_mcts_root_temp = {{1.25, 1.0}, {1.0, 1.5}};
  params.seat_root_fpu_zero = {{1, 0}, {0, 1}};
  params.seat_visits = {{10, 10}, {10, 10}};

  auto pm = PlayManager{std::make_unique<connect4_gs::Connect4GS>(), params};

  for (uint32_t i = 0; i < params.concurrent_games; ++i) {
    const auto& gd = pm.game_data(i);
    uint8_t pi = i % params.seat_perms.size();
    EXPECT_EQ(gd.perm_index, pi) << "game " << i;
    for (int s = 0; s < 2; ++s) {
      EXPECT_FLOAT_EQ(gd.mcts[s].epsilon(),
                      params.seat_epsilon[pi][s])
          << "game " << i << " seat " << s;
      EXPECT_FLOAT_EQ(gd.mcts[s].root_policy_temp(),
                      params.seat_mcts_root_temp[pi][s])
          << "game " << i << " seat " << s;
      EXPECT_EQ(gd.mcts[s].root_fpu_zero(),
                static_cast<bool>(params.seat_root_fpu_zero[pi][s]))
          << "game " << i << " seat " << s;
    }
  }
}

// -----------------------------------------------------------------------------
// New per-seat fields (added with G1-default refactor + per-seat resign).
// -----------------------------------------------------------------------------

TEST(PlayManager, SeatGumbelUseImprovedPolicyWrongDimThrows) {
  auto params = PlayParams{};
  params.games_to_play = 4;
  params.concurrent_games = 2;
  params.mcts_visits = {10, 10};
  params.eval_type = {EvalType::RANDOM, EvalType::RANDOM};
  // Default seat_perms is 1 perm, but the override has 2.
  params.seat_gumbel_use_improved_policy = {{1, 0}, {0, 1}};
  EXPECT_THROW(
      PlayManager(std::make_unique<connect4_gs::Connect4GS>(), params),
      std::runtime_error);
}

TEST(PlayManager, SeatGumbelUseImprovedPolicyInnerDimThrows) {
  auto params = PlayParams{};
  params.games_to_play = 4;
  params.concurrent_games = 2;
  params.mcts_visits = {10, 10};
  params.eval_type = {EvalType::RANDOM, EvalType::RANDOM};
  // 1 perm (default), but inner dim 3 != num_players (2).
  params.seat_gumbel_use_improved_policy = {{1, 0, 1}};
  EXPECT_THROW(
      PlayManager(std::make_unique<connect4_gs::Connect4GS>(), params),
      std::runtime_error);
}

TEST(PlayManager, SeatResignThresholdWrongDimThrows) {
  auto params = PlayParams{};
  params.games_to_play = 4;
  params.concurrent_games = 2;
  params.mcts_visits = {10, 10};
  params.eval_type = {EvalType::RANDOM, EvalType::RANDOM};
  params.seat_resign_threshold = {{-0.9, -0.9}, {-0.9, -0.9}};  // 2 perms
  EXPECT_THROW(
      PlayManager(std::make_unique<connect4_gs::Connect4GS>(), params),
      std::runtime_error);
}

TEST(PlayManager, SeatResignConsecutiveWrongDimThrows) {
  auto params = PlayParams{};
  params.games_to_play = 4;
  params.concurrent_games = 2;
  params.mcts_visits = {10, 10};
  params.eval_type = {EvalType::RANDOM, EvalType::RANDOM};
  params.seat_resign_consecutive = {{3, 3, 3}};  // 1 perm, inner=3 != 2
  EXPECT_THROW(
      PlayManager(std::make_unique<connect4_gs::Connect4GS>(), params),
      std::runtime_error);
}

TEST(PlayManager, NewSeatFieldDefaultsFromEmpty) {
  // When the new per-seat fields are left empty, PlayManager fills them with
  // the documented sentinels: G3 opt-in = 0 (paper-faithful G1), resign
  // threshold = -2.0 (disabled), resign consecutive = 1.
  auto params = PlayParams{};
  params.games_to_play = 4;
  params.concurrent_games = 2;
  params.mcts_visits = {10, 10};
  params.eval_type = {EvalType::RANDOM, EvalType::RANDOM};
  EXPECT_NO_THROW(
      PlayManager(std::make_unique<connect4_gs::Connect4GS>(), params));
}

TEST(PlayManager, RunsWithG3OptInAndResign) {
  // Sanity check: a full PlayManager run with both new fields exercised
  // completes without throwing. With random eval, the resign threshold
  // (-1.5, below any achievable V) effectively disables resign even when
  // the field is populated.
  auto params = PlayParams{};
  params.games_to_play = 16;
  params.concurrent_games = 4;
  params.mcts_visits = {10, 10};
  params.eval_type = {EvalType::RANDOM, EvalType::RANDOM};
  params.seat_gumbel_use_improved_policy = {{0, 0}};
  params.seat_resign_threshold = {{-1.5f, -1.5f}};  // never fires
  params.seat_resign_consecutive = {{1, 1}};

  auto pm = PlayManager{std::make_unique<connect4_gs::Connect4GS>(), params};
  auto play = std::async(std::launch::async, [&] {
    try {
      pm.play();
    } catch (const std::exception& e) {
      FAIL() << "Got an exception: " << e.what() << std::endl;
    }
  });
  play.wait();
  EXPECT_EQ(pm.games_completed(), 16U);
}

TEST(PlayManager, AggressiveResignTerminatesGames) {
  // Set the resign threshold high enough that every move triggers
  // (threshold > +1 makes V <= threshold always true since V ∈ [-1, 1]).
  // With consecutive=1, the very first move of every game causes the seat
  // to resign, ending the game immediately. We verify all games complete.
  auto params = PlayParams{};
  params.games_to_play = 8;
  params.concurrent_games = 4;
  params.mcts_visits = {10, 10};
  params.eval_type = {EvalType::RANDOM, EvalType::RANDOM};
  params.seat_resign_threshold = {{2.0f, 2.0f}};   // always triggers
  params.seat_resign_consecutive = {{1, 1}};

  auto pm = PlayManager{std::make_unique<connect4_gs::Connect4GS>(), params};
  auto play = std::async(std::launch::async, [&] {
    try {
      pm.play();
    } catch (const std::exception& e) {
      FAIL() << "Got an exception: " << e.what() << std::endl;
    }
  });
  play.wait();
  EXPECT_EQ(pm.games_completed(), 8U);
  // Every game ends via resignation, so the resign_scores accumulator must
  // have counted them all.
  const auto rs = pm.resign_scores();
  float total = 0.0f;
  for (Eigen::Index i = 0; i < rs.size(); ++i) total += rs[i];
  EXPECT_GT(total, 0.0f);
}

}  // namespace
}  // namespace alphazero
