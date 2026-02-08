#include "game_state.h"

#include <algorithm>
#include <random>
#include <thread>

namespace alphazero {

// Core playout logic with an explicit RNG (no thread_local).
static std::tuple<Vector<float>, Vector<float>> playout_eval_impl(
    const GameState& gs, std::default_random_engine& re) {
  // Policy: uniform over legal moves of the original leaf state.
  auto valids = gs.valid_moves();
  auto policy = Vector<float>{gs.num_moves()};
  policy.setZero();
  float sum = valids.sum();
  if (sum > 0.0) {
    policy = valids.cast<float>() / sum;
  }

  // Value: play random moves until terminal.
  auto sim = gs.copy();
  while (!sim->scores().has_value()) {
    auto sim_valids = sim->valid_moves();
    // Collect valid move indices.
    std::vector<uint32_t> valid_indices;
    for (uint32_t i = 0; i < sim->num_moves(); ++i) {
      if (sim_valids[i] != 0) {
        valid_indices.push_back(i);
      }
    }
    if (valid_indices.empty()) {
      break;
    }
    std::uniform_int_distribution<uint32_t> dist{
        0, static_cast<uint32_t>(valid_indices.size() - 1)};
    sim->play_move(valid_indices[dist(re)]);
  }

  auto scores = sim->scores();
  if (scores.has_value()) {
    return {scores.value(), policy};
  }
  // Fallback: uniform value (shouldn't normally happen).
  auto values = Vector<float>{gs.num_players() + 1};
  values.setConstant(1.0 / (gs.num_players() + 1));
  return {values, policy};
}

std::tuple<Vector<float>, Vector<float>> playout_eval(const GameState& gs) {
  thread_local std::default_random_engine re{std::random_device{}()};
  return playout_eval_impl(gs, re);
}

std::tuple<std::vector<Vector<float>>, std::vector<Vector<float>>>
playout_eval_batch(const std::vector<const GameState*>& states) {
  const auto n = states.size();
  std::vector<Vector<float>> values(n);
  std::vector<Vector<float>> policies(n);

  const auto hw = std::thread::hardware_concurrency();
  const auto num_threads =
      std::min(n, static_cast<size_t>(hw > 0 ? hw : 4));

  std::vector<std::thread> threads;
  threads.reserve(num_threads);
  for (size_t t = 0; t < num_threads; ++t) {
    threads.emplace_back([&, t]() {
      // Local RNG per thread — avoids thread_local TLS churn in dylibs.
      std::default_random_engine re{std::random_device{}()};

      const size_t chunk = (n + num_threads - 1) / num_threads;
      const size_t start = t * chunk;
      const size_t end = std::min(start + chunk, n);
      for (size_t i = start; i < end; ++i) {
        std::tie(values[i], policies[i]) =
            playout_eval_impl(*states[i], re);
      }
    });
  }
  for (auto& th : threads) {
    th.join();
  }

  return {std::move(values), std::move(policies)};
}

}  // namespace alphazero
