#include <benchmark/benchmark.h>

#include <algorithm>
#include <future>
#include <limits>
#include <memory>

#include "connect4_gs.h"
#include "opentafl_gs.h"
#include "play_manager.h"
#include "star_gambit_gs.h"
#include "tawlbwrdd_gs.h"

namespace alphazero {
namespace {

// Game factory helpers. Each returns a freshly constructed base GameState so
// the streaming-pool benchmarks below can be parameterized over the action
// space of the game (connect4: NUM_MOVES=7, opentafl/tawlbwrdd: NUM_MOVES=2662)
// without duplicating the driver code.
std::unique_ptr<GameState> make_tawlbwrdd() {
  return std::make_unique<tawlbwrdd_gs::TawlbwrddGS>();
}
std::unique_ptr<GameState> make_opentafl() {
  return std::make_unique<opentafl_gs::OpenTaflGS>();
}
std::unique_ptr<GameState> make_connect4() {
  return std::make_unique<connect4_gs::Connect4GS>();
}
// star_gambit_unified: the real production/target game for this investigation.
// UNIFIED_NUM_MOVES=1709 (sparse large action space). Pinned to the Clash
// variant (index 2): the most complex of the three 11x11 variants, giving a
// single reproducible game complexity/length rather than the noisy 4-variant
// random mix. num_moves() is 1709 regardless of variant.
std::unique_ptr<GameState> make_star_gambit() {
  return std::make_unique<star_gambit_gs::StarGambitUnifiedGS>(/*pinned_variant=*/2);
}

PlayParams make_params(uint32_t games) {
  auto params = PlayParams{};
  params.games_to_play = games;
  params.concurrent_games = games;
  params.mcts_visits = {250, 250};
  params.eval_type = {EvalType::RANDOM, EvalType::RANDOM};
  params.history_enabled = true;
  params.playout_cap_randomization = true;
  return params;
}

void PlayGameSingleThreaded() {
  auto params = make_params(64);
  auto pm = PlayManager{std::make_unique<tawlbwrdd_gs::TawlbwrddGS>(), params};
  auto play = std::async(std::launch::async, [&] { pm.play(); });
  play.wait();
}

// Runs `games` self-play games spread across `workers` freshly-spawned worker
// threads. The awaiting_mcts_ queue is sharded to the worker count (as real
// callers should size it -- see config.py's resolved_queue_shards) so this
// benchmark exercises the sharded queue rather than the 1-shard default.
//
// `concurrent` decouples the in-flight game pool (concurrent_games) from the
// total workload (games_to_play): the queue only ever holds `concurrent`
// game indices, so if `concurrent` is small relative to `workers` there may
// not be enough ready work to keep every worker busy. When concurrent==games
// all games are in flight simultaneously (the historical batch behaviour).
//
// `make_game` selects which game (and therefore which action-space size) the
// PlayManager runs, so the same driver can be reused for connect4, opentafl,
// and tawlbwrdd.
template <typename MakeGame>
void PlayGameMultiThreadedPool(unsigned workers, uint32_t games,
                               uint32_t concurrent, MakeGame make_game) {
  auto params = make_params(games);
  params.concurrent_games = concurrent;
  params.queue_shards = static_cast<uint8_t>(
      std::min<unsigned>(workers, std::numeric_limits<uint8_t>::max()));
  auto pm = PlayManager{make_game(), params};
  auto play_workers = std::vector<std::future<void>>{workers};
  for (auto& pw : play_workers) {
    pw = std::async(std::launch::async, [&] { pm.play(); });
  }
  for (auto& pw : play_workers) {
    pw.wait();
  }
}

void PlayGameMultiThreaded(unsigned workers, uint32_t games) {
  PlayGameMultiThreadedPool(workers, games, games, make_tawlbwrdd);
}

static void BM_PlayGameSingleThreaded(benchmark::State& state) {
  // Perform setup here
  for (auto _ : state) {
    // This code gets timed
    PlayGameSingleThreaded();
  }
}
BENCHMARK(BM_PlayGameSingleThreaded)->UseRealTime();

// Sweep worker count for the fixed 64-game workload. arg = worker count.
static void BM_PlayGameMultiThreaded(benchmark::State& state) {
  const auto workers = static_cast<unsigned>(state.range(0));
  for (auto _ : state) {
    PlayGameMultiThreaded(workers, 64);
  }
}
BENCHMARK(BM_PlayGameMultiThreaded)
    ->Arg(1)
    ->Arg(2)
    ->Arg(4)
    ->Arg(8)
    ->Arg(12)
    ->Arg(16)
    ->Arg(20)
    ->Arg(24)
    ->Arg(31)
    ->Arg(32)
    ->UseRealTime();

// Same sweep but 8x the workload (512 games / 512 concurrent) to check how
// much of any scaling shortfall is per-iteration thread create/join overhead
// vs. steady-state contention: if efficiency at a given worker count rises
// materially here, the fixed cost of spawning `workers` threads once per
// google-benchmark iteration was a real contributor.
static void BM_PlayGameMultiThreadedBig(benchmark::State& state) {
  const auto workers = static_cast<unsigned>(state.range(0));
  for (auto _ : state) {
    PlayGameMultiThreaded(workers, 512);
  }
}
BENCHMARK(BM_PlayGameMultiThreadedBig)
    ->Arg(16)
    ->Arg(31)
    ->UseRealTime();

// H2 concurrent_games sweep. Args = {workers, concurrent_games}. The workload
// (games_to_play) equals concurrent_games (batch model, matching the existing
// baseline), so throughput is reported as games/sec and parallel efficiency is
// computed externally as gps(workers,G) / (workers * gps(1,G)). Pin thread
// counts with taskset and select rows with --benchmark_filter.
static void BM_ConcGamesSweep(benchmark::State& state) {
  const auto workers = static_cast<unsigned>(state.range(0));
  const auto games = static_cast<uint32_t>(state.range(1));
  for (auto _ : state) {
    PlayGameMultiThreadedPool(workers, games, games, make_tawlbwrdd);
  }
  state.counters["games_per_sec"] = benchmark::Counter(
      static_cast<double>(state.iterations()) * games,
      benchmark::Counter::kIsRate);
}
BENCHMARK(BM_ConcGamesSweep)
    ->Args({1, 64})->Args({1, 128})->Args({1, 256})
    ->Args({1, 512})->Args({1, 1024})
    ->Args({8, 64})->Args({8, 128})->Args({8, 256})
    ->Args({8, 512})->Args({8, 1024})
    ->Args({16, 64})->Args({16, 128})->Args({16, 256})
    ->Args({16, 512})->Args({16, 1024})
    ->UseRealTime();

// Streaming: a FIXED in-flight pool of `concurrent` games, but 8x that many
// games_to_play so games are continuously replenished (as in real self-play,
// where games_to_play >> concurrent_games). Isolates steady-state behaviour
// from the batch-tail shard imbalance that BM_ConcGamesSweep's
// games_to_play==concurrent model suffers. Args = {workers, concurrent}.
//
// One templated driver, instantiated per game below, so the streaming-pool
// methodology is identical across action-space sizes and only the game (and
// therefore per-simulation memory footprint) changes.
template <typename MakeGame>
void StreamPoolRun(benchmark::State& state, MakeGame make_game) {
  const auto workers = static_cast<unsigned>(state.range(0));
  const auto concurrent = static_cast<uint32_t>(state.range(1));
  const auto games = concurrent * 8;
  for (auto _ : state) {
    PlayGameMultiThreadedPool(workers, games, concurrent, make_game);
  }
  state.counters["games_per_sec"] = benchmark::Counter(
      static_cast<double>(state.iterations()) * games,
      benchmark::Counter::kIsRate);
}

// Shared thread-count sweep applied to every game so the parallel-efficiency
// curves are directly comparable.
#define STREAM_POOL_ARGS \
  Args({1, 64})          \
      ->Args({8, 64})    \
      ->Args({16, 64})   \
      ->Args({20, 64})   \
      ->Args({24, 64})   \
      ->Args({31, 64})   \
      ->Args({32, 64})   \
      ->UseRealTime()

// star_gambit_unified: the PRIMARY target game. UNIFIED_NUM_MOVES=1709 (sparse
// large action space, random-variant mix matching real training config).
static void BM_StreamPoolStarGambit(benchmark::State& state) {
  StreamPoolRun(state, make_star_gambit);
}
BENCHMARK(BM_StreamPoolStarGambit)->STREAM_POOL_ARGS;

// tawlbwrdd: NUM_MOVES=2662, the original sparse large-action-space case.
static void BM_StreamPool(benchmark::State& state) {
  StreamPoolRun(state, make_tawlbwrdd);
}
BENCHMARK(BM_StreamPool)->STREAM_POOL_ARGS;

// opentafl: NUM_MOVES=2662, same 11x11 board / same action-space ceiling as
// tawlbwrdd but a different ruleset -- the matched-action-space control.
static void BM_StreamPoolOpenTafl(benchmark::State& state) {
  StreamPoolRun(state, make_opentafl);
}
BENCHMARK(BM_StreamPoolOpenTafl)->STREAM_POOL_ARGS;

// connect4: NUM_MOVES=7, the extreme dense/tiny-action-space case (~380x
// smaller dense scratch + tiny children arrays per node).
static void BM_StreamPoolConnect4(benchmark::State& state) {
  StreamPoolRun(state, make_connect4);
}
BENCHMARK(BM_StreamPoolConnect4)->STREAM_POOL_ARGS;

}  // namespace
}  // namespace alphazero
