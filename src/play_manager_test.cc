#include "play_manager.h"

#include <gtest/gtest.h>

#include <future>

#include "connect4_gs.h"

namespace alphazero {
namespace {

// NOLINTNEXTLINE
TEST(PlayManager, Basic) {
  auto params = PlayParams{};
  params.games_to_play = 32;
  params.concurrent_games = 8;
  params.mcts_depth = {10, 10};
  auto pm = PlayManager{std::make_unique<connect4_gs::Connect4GS>(), params};
  auto play = std::async(std::launch::async, [&] {
    try {
      pm.play();
    } catch (const std::exception& e) {
      FAIL() << "Got an exception: " << e.what() << std::endl;
    }
  });
  auto infer_p0 = std::async(std::launch::async, [&] {
    try {
      pm.dumb_inference(0);
    } catch (const std::exception& e) {
      FAIL() << "Got an exception: " << e.what() << std::endl;
    }
  });
  auto infer_p1 = std::async(std::launch::async, [&] {
    try {
      pm.dumb_inference(1);
    } catch (const std::exception& e) {
      FAIL() << "Got an exception: " << e.what() << std::endl;
    }
  });
  play.wait();
  infer_p0.wait();
  infer_p1.wait();
}

TEST(PlayManager, MultiThreaded) {
  const auto cores = std::thread::hardware_concurrency();
  const auto workers = cores - 1;

  auto params = PlayParams{};
  params.games_to_play = 32 * workers;
  params.concurrent_games = 8 * workers;
  params.mcts_depth = {10, 10};

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
  auto infer_p0 = std::async(std::launch::async, [&] {
    try {
      pm.dumb_inference(0);
    } catch (const std::exception& e) {
      FAIL() << "Got an exception: " << e.what() << std::endl;
    }
  });
  auto infer_p1 = std::async(std::launch::async, [&] {
    try {
      pm.dumb_inference(1);
    } catch (const std::exception& e) {
      FAIL() << "Got an exception: " << e.what() << std::endl;
    }
  });
  for (auto& pw : play_workers) {
    pw.wait();
  }
  infer_p0.wait();
  infer_p1.wait();
  std::cout << "Scores: " << pm.scores() << std::endl;
}

}  // namespace
}  // namespace alphazero