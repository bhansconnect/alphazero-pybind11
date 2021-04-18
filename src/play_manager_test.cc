#include "play_manager.h"

#include <future>

#include "connect4_gs.h"
#include "gtest/gtest.h"

namespace alphazero {
namespace {

// NOLINTNEXTLINE
TEST(PlayManager, Basic) {
  auto params = PlayParams{};
  params.base_gs = std::make_unique<connect4_gs::Connect4GS>();
  params.games_to_play = 32;
  params.concurrent_games = 8;
  auto pm = PlayManager{std::move(params)};
  auto play = std::async(std::launch::async, [&] {
    try {
      pm.play();
    } catch (const std::exception& e) {
      FAIL() << "Got an exception: " << e.what() << std::endl;
    }
  });
  auto infer = std::async(std::launch::async, [&] {
    try {
      pm.dumb_inference();
    } catch (const std::exception& e) {
      FAIL() << "Got an exception: " << e.what() << std::endl;
    }
  });
  play.wait();
  infer.wait();
}

TEST(PlayManager, MultiThreaded) {
  const auto cores = std::thread::hardware_concurrency();
  const auto workers = cores - 1;

  auto params = PlayParams{};
  params.base_gs = std::make_unique<connect4_gs::Connect4GS>();
  params.games_to_play = 32 * workers;
  params.concurrent_games = 8 * workers;

  auto pm = PlayManager{std::move(params)};
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
  auto infer = std::async(std::launch::async, [&] {
    try {
      pm.dumb_inference();
    } catch (const std::exception& e) {
      FAIL() << "Got an exception: " << e.what() << std::endl;
    }
  });
  for (auto& pw : play_workers) {
    pw.wait();
  }
  infer.wait();
}

}  // namespace
}  // namespace alphazero