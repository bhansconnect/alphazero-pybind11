#pragma once

#include "game_state.h"

namespace alphazero::connect4_gs {

constexpr const int WIDTH = 7;
constexpr const int HEIGHT = 6;
constexpr const std::array<int, 3> BOARD_SHAPE = {2, HEIGHT, WIDTH};
constexpr const std::array<int, 3> CANONICAL_SHAPE = {4, HEIGHT, WIDTH};

using BoardTensor =
    SizedTensor<int8_t,
                Eigen::Sizes<BOARD_SHAPE[0], BOARD_SHAPE[1], BOARD_SHAPE[2]>>;

using CanonicalTensor =
    SizedTensor<float, Eigen::Sizes<CANONICAL_SHAPE[0], CANONICAL_SHAPE[1],
                                    CANONICAL_SHAPE[2]>>;

class Connect4GS : public GameState {
 public:
  Connect4GS() { board_.setZero(); }
  Connect4GS(BoardTensor board, int8_t player)
      : board_(board), player_(player) {}
  Connect4GS(BoardTensor&& board, int8_t player)
      : board_(std::move(board)), player_(player) {}

  [[nodiscard]] std::unique_ptr<GameState> copy() const noexcept override;
  [[nodiscard]] bool operator==(const GameState& other) const noexcept override;

  // Returns the current player. Players must be 0 indexed.
  [[nodiscard]] uint8_t current_player() const noexcept override {
    return player_;
  };

  // Returns the number of possible moves.
  [[nodiscard]] int32_t num_moves() const noexcept override { return WIDTH; }

  // Returns the number of players.
  [[nodiscard]] int8_t num_players() const noexcept override { return 2; };

  // Returns a bool for all moves. The value is true if the move is playable
  // from this GameState.
  [[nodiscard]] Vector<uint8_t> valid_moves() const noexcept override;

  // Plays a move, modifying the current GameState.
  void play_move(uint32_t move) override;

  // Returns nullopt if the game isn't over.
  // Returns the score for each player if the game is over.
  // This is: 1 for win, -1 for loss, 0 for draw.
  [[nodiscard]] std::optional<Vector<float>> scores() const noexcept override;

  // Returns the canonicalized form of the board, ready for feeding to a NN.
  [[nodiscard]] Tensor<float, 3> canonicalized() const noexcept override;

  // Returns a string representation of the game state.
  [[nodiscard]] std::string dump() const noexcept override;

 protected:
  // Board contains a layer for each player.
  // A 0 means no piece, a 1 means a piece for that player.
  BoardTensor board_{};
  int8_t player_{0};
};

}  // namespace alphazero::connect4_gs