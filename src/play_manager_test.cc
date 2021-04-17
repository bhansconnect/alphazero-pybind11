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
  params.games_to_play = 10;
  params.concurrent_games = 1;
  auto pm = PlayManager{std::move(params)};
  auto play = std::async(std::launch::async, [&] { pm.play(); });
  auto infer = std::async(std::launch::async, [&] { pm.dumb_inference(); });
  play.wait();
  infer.wait();
}

}  // namespace
}  // namespace alphazero