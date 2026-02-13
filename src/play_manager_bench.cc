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
  params.mcts_visits = {250, 250};
  params.eval_type = {EvalType::RANDOM, EvalType::RANDOM};
  params.history_enabled = true;
  params.playout_cap_randomization = true;
  auto pm = PlayManager{std::make_unique<tawlbwrdd_gs::TawlbwrddGS>(), params};
  auto play = std::async(std::launch::async, [&] { pm.play(); });
  play.wait();
}

void PlayGameMultiThreaded() {
  const auto cores = std::thread::hardware_concurrency();
  const auto workers = cores - 1;

  auto params = PlayParams{};
  params.games_to_play = 64;
  params.concurrent_games = 64;
  params.mcts_visits = {250, 250};
  params.eval_type = {EvalType::RANDOM, EvalType::RANDOM};
  params.history_enabled = true;
  params.playout_cap_randomization = true;
  auto pm = PlayManager{std::make_unique<tawlbwrdd_gs::TawlbwrddGS>(), params};
  auto play_workers = std::vector<std::future<void>>{workers};
  for (auto& pw : play_workers) {
    pw = std::async(std::launch::async, [&] { pm.play(); });
  }
  for (auto& pw : play_workers) {
    pw.wait();
  }
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
