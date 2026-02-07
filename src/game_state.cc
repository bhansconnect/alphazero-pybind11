#include "game_state.h"

#include <random>

namespace alphazero {

std::tuple<Vector<float>, Vector<float>> playout_eval(const GameState& gs) {
  // Policy: uniform over legal moves of the original leaf state.
  auto valids = gs.valid_moves();
  auto policy = Vector<float>{gs.num_moves()};
  policy.setZero();
  float sum = valids.sum();
  if (sum > 0.0) {
    policy = valids.cast<float>() / sum;
  }

  // Value: play random moves until terminal.
  thread_local std::default_random_engine re{std::random_device{}()};
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

}  // namespace alphazero
