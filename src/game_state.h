#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <tuple>
#include <typeindex>
#include <vector>

#include "absl/hash/hash.h"
#include "dll_export.h"
#include "shapes.h"

namespace alphazero {

struct PlayHistory {
  Tensor<float, 3> canonical;
  Vector<float> v;
  Vector<float> pi;
};

// Rotate absolute value vector to relative (current-player-relative).
// v_rel[i] = v_abs[(player + i) % num_players], draw index unchanged.
inline Vector<float> absolute_to_relative(
    const Vector<float>& v, uint8_t player, uint8_t num_players) {
  if (player == 0) return v;
  Vector<float> out(v.size());
  for (uint8_t i = 0; i < num_players; ++i) {
    out(i) = v((player + i) % num_players);
  }
  out(num_players) = v(num_players);  // draw
  return out;
}

// Rotate relative (current-player-relative) value vector to absolute.
// v_abs[(player + i) % num_players] = v_rel[i], draw index unchanged.
inline Vector<float> relative_to_absolute(
    const Vector<float>& v, uint8_t player, uint8_t num_players) {
  if (player == 0) return v;
  Vector<float> out(v.size());
  for (uint8_t i = 0; i < num_players; ++i) {
    out((player + i) % num_players) = v(i);
  }
  out(num_players) = v(num_players);  // draw
  return out;
}

// GameState is the core class to represent games to be played by AlphaZero. It
// creates the basic api needed by the MCTS. It uses Eigen to represent the
// interface in order to avoid copying to and from Python. Pybind11 already has
// an optimized interface for sharing the underlying data. I would have used
// numpy arrays, but it is easier to test games if you don't have to link
// Pybind11 in all sources.

class WEAKDLLEXPORT GameState {
 public:
  virtual ~GameState() = default;

  [[nodiscard]] virtual std::unique_ptr<GameState> copy() const noexcept = 0;

  // Equality and Hash should only compare things as the neural network sees
  // things. I.E. if the network doesn't know the exact score, don't compare the
  // exact score or hash it. This enables use in the LRU cache correctly.
  [[nodiscard]] virtual bool operator==(
      const GameState& other) const noexcept = 0;
  [[nodiscard]] bool operator!=(const GameState& other) const noexcept {
    return !(*this == other);
  }

  void virtual hash(absl::HashState h) const = 0;

  // Randomize the start state of a game. For most games this does nothing.
  virtual void randomize_start() noexcept {};

  // Returns the current player. Players must be 0 indexed.
  [[nodiscard]] virtual uint8_t current_player() const noexcept = 0;

  // Returns the current turn.
  [[nodiscard]] virtual uint32_t current_turn() const noexcept = 0;

  // Returns the number of possible moves.
  [[nodiscard]] virtual uint32_t num_moves() const noexcept = 0;

  // Returns the number of players.
  [[nodiscard]] virtual uint8_t num_players() const noexcept = 0;

  // Returns a bool for all moves. The value is true if the move is playable
  // from this GameState. All values should be 0 or 1.
  [[nodiscard]] virtual Vector<uint8_t> valid_moves() const noexcept = 0;

  // Plays a move, modifying the current GameState.
  virtual void play_move(uint32_t move) = 0;

  // Returns nullopt if the game isn't over.
  // Returns a one hot encode result of the game.
  // The first num player positions are set to 1 if that player won and 0
  // otherwise. The last position is set to 1 if the game was a draw and 0
  // otherwise.
  [[nodiscard]] virtual std::optional<Vector<float>> scores()
      const noexcept = 0;

  // Returns the canonicalized form of the board, ready for feeding to a NN.
  [[nodiscard]] virtual Tensor<float, 3> canonicalized() const noexcept = 0;

  // Returns true if the game uses player-relative canonical values.
  // When true, NN outputs are rotated to absolute before MCTS backup,
  // and game scores are rotated to relative for training data.
  [[nodiscard]] virtual bool relative_values() const noexcept { return false; }

  // Returns the number of symmetries the game has.
  [[nodiscard]] virtual uint8_t num_symmetries() const noexcept = 0;

  // Returns an list of all symetrical game states (including the base state).
  [[nodiscard]] virtual std::vector<PlayHistory> symmetries(
      const PlayHistory& base) const noexcept = 0;

  // Returns a string representation of the game state.
  [[nodiscard]] virtual std::string dump() const noexcept = 0;

};

namespace detail {
struct GameStateHashRef {
  const GameState& gs;
  template <typename H>
  friend H AbslHashValue(H h, const GameStateHashRef& r) {
    h = H::combine(std::move(h), std::type_index(typeid(r.gs)));
    r.gs.hash(absl::HashState::Create(&h));
    return std::move(h);
  }
};
}  // namespace detail

// Compute a 64-bit hash of a GameState.
inline uint64_t hash_game_state(const GameState& gs) {
  return absl::HashOf(detail::GameStateHashRef{gs});
}

// A sample evaluation function for testing.
// It just returns even probablity.
[[nodiscard]] inline std::tuple<Vector<float>, Vector<float>> dumb_eval(
    const GameState& gs) {
  auto valids = gs.valid_moves();
  auto values = Vector<float>{gs.num_players() + 1};
  values.setConstant(1.0 / (gs.num_players() + 1));
  auto policy = Vector<float>{gs.num_moves()};
  policy.setZero();
  float sum = valids.sum();
  if (sum == 0.0) {
    return {values, policy};
  }
  policy = valids.cast<float>() / sum;
  return {values, policy};
}

// Playout (rollout) evaluation: plays random moves to terminal, returns
// game outcome as value and uniform-over-legal-moves as policy.
[[nodiscard]] std::tuple<Vector<float>, Vector<float>> playout_eval(
    const GameState& gs);

// Batched playout evaluation using C++ threads (one per leaf).
// Avoids Python threading / GIL issues that cause memory corruption.
[[nodiscard]] std::tuple<std::vector<Vector<float>>, std::vector<Vector<float>>>
playout_eval_batch(const std::vector<const GameState*>& states);

}  // namespace alphazero