#pragma once

#include <cstdint>
#include <memory>
#include <optional>

#include "shapes.h"

namespace alphazero {

// GameState is the core class to represent games to be played by AlphaZero. It
// creates the basic api needed by the MCTS. It uses Eigen to represent the
// interface in order to avoid copying to and from Python. Pybind11 already has
// an optimized interface for sharing the underlying data. I would have used
// numpy arrays, but it is easier to test games if you don't have to link
// Pybind11 in all sources.

class GameState {
 public:
  virtual ~GameState() = default;

  [[nodiscard]] virtual std::unique_ptr<GameState> copy() const noexcept = 0;
  [[nodiscard]] virtual bool operator==(const GameState& other) const
      noexcept = 0;
  [[nodiscard]] bool operator!=(const GameState& other) const noexcept {
    return !(*this == other);
  }

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
  [[nodiscard]] virtual std::optional<Vector<float>> scores() const
      noexcept = 0;

  // Returns the canonicalized form of the board, ready for feeding to a NN.
  [[nodiscard]] virtual Tensor<float, 3> canonicalized() const noexcept = 0;

  // Returns a string representation of the game state.
  [[nodiscard]] virtual std::string dump() const noexcept = 0;
};

// A sample evaluation function for testing.
// It just returns even probablity.
[[nodiscard]] std::tuple<Vector<float>, Vector<float>> dumb_eval(
    const GameState& gs) {
  auto valids = gs.valid_moves();
  auto values = Vector<float>{gs.num_players() + 1};
  values.setZero();
  values(gs.num_players()) = 1;
  auto policy = Vector<float>{gs.num_moves()};
  policy.setZero();
  float sum = valids.sum();
  if (sum == 0.0) {
    return {values, policy};
  }
  policy = valids.cast<float>() / sum;
  return {values, policy};
}

}  // namespace alphazero