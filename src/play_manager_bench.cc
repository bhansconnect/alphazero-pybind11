#include <benchmark/benchmark.h>

#include <future>

#include "play_manager.h"
#include "tawlbwrdd_gs.h"

namespace alphazero {
namespace {

void PlayGameSingleThreaded() {
  auto params = PlayParams{};
  params.games_to_play = 64;
  params.concurrent_games = 64;
  params.mcts_depth = {250, 250};
  params.history_enabled = true;
  params.playout_cap_randomization = true;
  auto pm = PlayManager{std::make_unique<tawlbwrdd_gs::TawlbwrddGS>(), params};
  auto play = std::async(std::launch::async, [&] { pm.play(); });
  auto infer_p0 = std::async(std::launch::async, [&] { pm.dumb_inference(0); });
  auto infer_p1 = std::async(std::launch::async, [&] { pm.dumb_inference(1); });
  play.wait();
  infer_p0.wait();
  infer_p1.wait();
}

void PlayGameMultiThreaded() {
  const auto cores = std::thread::hardware_concurrency();
  const auto workers = cores - 1;

  auto params = PlayParams{};
  params.games_to_play = 64;
  params.concurrent_games = 64;
  params.mcts_depth = {250, 250};
  params.history_enabled = true;
  params.playout_cap_randomization = true;
  auto pm = PlayManager{std::make_unique<tawlbwrdd_gs::TawlbwrddGS>(), params};
  auto play_workers = std::vector<std::future<void>>{workers};
  for (auto& pw : play_workers) {
    pw = std::async(std::launch::async, [&] { pm.play(); });
  }
  auto infer_p0 = std::async(std::launch::async, [&] { pm.dumb_inference(0); });
  auto infer_p1 = std::async(std::launch::async, [&] { pm.dumb_inference(1); });
  for (auto& pw : play_workers) {
    pw.wait();
  }
  infer_p0.wait();
  infer_p1.wait();
}

static void BM_PlayGameSingleThreaded(benchmark::State& state) {
  // Perform setup here
  for (auto _ : state) {
    // This code gets timed
    PlayGameSingleThreaded();
  }
}
BENCHMARK(BM_PlayGameSingleThreaded);

static void BM_PlayGameMultiThreaded(benchmark::State& state) {
  // Perform setup here
  for (auto _ : state) {
    // This code gets timed
    PlayGameMultiThreaded();
  }
}
BENCHMARK(BM_PlayGameMultiThreaded);

}  // namespace
}  // namespace alphazero