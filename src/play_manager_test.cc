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
  params.games_to_play = 128;
  params.concurrent_games = 16;
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

}  // namespace
}  // namespace alphazero