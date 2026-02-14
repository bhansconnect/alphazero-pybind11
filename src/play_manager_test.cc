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

}  // namespace
}  // namespace alphazero
