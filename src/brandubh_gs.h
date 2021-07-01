#pragma once

#include "absl/container/flat_hash_map.h"
#include "game_state.h"

// Base rules from http://aagenielsen.dk/brandubh2_rules_en.pdf
// A small change is made to rule 7. Encirclement is no longer a win for the
// attackers. Instead, they must play the few extra moves to finish capturing
// the defenders. They also might get a win from blocking all possible white
// moves. On this small board, it really should make little to no difference.

namespace alphazero::brandubh_gs {

constexpr const int MAX_TURNS = 150;

constexpr const int KING_LAYER = 0;
constexpr const int DEF_LAYER = 1;
constexpr const int ATK_LAYER = 2;

// Player 0 is the first player (the attackers).
// Player 1 is the second player (the king side defenders).
constexpr const int ATK_PLAYER = 0;
constexpr const int DEF_PLAYER = 1;

constexpr const int WIDTH = 7;
constexpr const int HEIGHT = 7;
constexpr const int NUM_MOVES = WIDTH * HEIGHT * (WIDTH + HEIGHT);
constexpr const int NUM_PLAYERS = 2;
constexpr const std::array<int, 3> BOARD_SHAPE = {3, HEIGHT, WIDTH};
// 3 for board, 2 for current player, 2 for current repetition count.
// May want to add some historical positions.
constexpr const std::array<int, 3> CANONICAL_SHAPE = {7, HEIGHT, WIDTH};

using BoardTensor =
    SizedTensor<int8_t,
                Eigen::Sizes<BOARD_SHAPE[0], BOARD_SHAPE[1], BOARD_SHAPE[2]>>;

using CanonicalTensor =
    SizedTensor<float, Eigen::Sizes<CANONICAL_SHAPE[0], CANONICAL_SHAPE[1],
                                    CANONICAL_SHAPE[2]>>;

struct RepetitionKeyWrapper {
  RepetitionKeyWrapper(const BoardTensor& tensor, uint8_t player)
      : t(tensor), p(player) {}
  RepetitionKeyWrapper(const RepetitionKeyWrapper& rkw) : t(rkw.t), p(rkw.p) {}
  BoardTensor t;
  uint8_t p;
};
template <typename H>
H AbslHashValue(H h, const RepetitionKeyWrapper& rkw) {
  h = H::combine_contiguous(std::move(h), rkw.t.data(), rkw.t.size());
  return H::combine(std::move(h), rkw.p);
}
bool operator==(const RepetitionKeyWrapper& lhs,
                const RepetitionKeyWrapper& rhs) {
  for (auto i = 0; i < BOARD_SHAPE[0]; ++i) {
    for (auto j = 0; j < BOARD_SHAPE[1]; ++j) {
      for (auto k = 0; k < BOARD_SHAPE[2]; ++k) {
        if (lhs.t(i, j, k) != rhs.t(i, j, k)) {
          return false;
        }
      }
    }
  }
  return rhs.p == lhs.p;
}

class BrandubhGS : public GameState {
 public:
  BrandubhGS() {
    board_.setZero();
    // King
    board_(KING_LAYER, 3, 3) = 1;

    // Defenders
    board_(DEF_LAYER, 2, 3) = 1;
    board_(DEF_LAYER, 3, 2) = 1;
    board_(DEF_LAYER, 4, 3) = 1;
    board_(DEF_LAYER, 3, 4) = 1;

    // Attackers
    board_(ATK_LAYER, 1, 3) = 1;
    board_(ATK_LAYER, 0, 3) = 1;
    board_(ATK_LAYER, 3, 1) = 1;
    board_(ATK_LAYER, 3, 0) = 1;
    board_(ATK_LAYER, 5, 3) = 1;
    board_(ATK_LAYER, 6, 3) = 1;
    board_(ATK_LAYER, 3, 5) = 1;
    board_(ATK_LAYER, 3, 6) = 1;

    repetition_counts_[RepetitionKeyWrapper(board_, player_)] = 1;
  }
  BrandubhGS(BoardTensor board, int8_t player, int32_t turn)
      : board_(board), player_(player), turn_(turn) {}
  BrandubhGS(BoardTensor&& board, int8_t player, int32_t turn)
      : board_(std::move(board)), player_(player), turn_(turn) {}
  BrandubhGS(
      BoardTensor board, int8_t player, int32_t turn,
      uint8_t current_repetition_count,
      absl::flat_hash_map<RepetitionKeyWrapper, uint8_t> repetition_counts)
      : board_(board),
        player_(player),
        turn_(turn),
        current_repetition_count_(current_repetition_count) {
    for (const auto& entry : repetition_counts) {
      repetition_counts_[RepetitionKeyWrapper(entry.first)] = entry.second;
    }
  }

  [[nodiscard]] std::unique_ptr<GameState> copy() const noexcept override;
  [[nodiscard]] bool operator==(const GameState& other) const noexcept override;

  void hash(absl::HashState h) const override;

  // Returns the current player. Players must be 0 indexed.
  [[nodiscard]] uint8_t current_player() const noexcept override {
    return player_;
  };

  // Returns the current turn.
  [[nodiscard]] uint32_t current_turn() const noexcept override {
    return turn_;
  }

  // Returns the number of possible moves.
  [[nodiscard]] uint32_t num_moves() const noexcept override {
    return NUM_MOVES;
  }

  // Returns the number of players.
  [[nodiscard]] uint8_t num_players() const noexcept override { return 2; };

  // Returns a bool for all moves. The value is true if the move is playable
  // from this GameState.
  [[nodiscard]] Vector<uint8_t> valid_moves() const noexcept override;

  // Plays a move, modifying the current GameState.
  void play_move(uint32_t move) override;

  // Returns nullopt if the game isn't over.
  // Returns a one hot encode result of the game.
  // The first num player positions are set to 1 if that player won and 0
  // otherwise. The last position is set to 1 if the game was a draw and 0
  // otherwise.
  [[nodiscard]] std::optional<Vector<float>> scores() const noexcept override;

  // Returns the canonicalized form of the board, ready for feeding to a NN.
  [[nodiscard]] Tensor<float, 3> canonicalized() const noexcept override;

  // Returns a string representation of the game state.
  [[nodiscard]] std::string dump() const noexcept override;

 private:
  // Board contains a layer for the king, other white pieces, and black
  // pieces. A 0 means no piece, a 1 means a piece of the respective type.
  BoardTensor board_{};
  int8_t player_{0};
  int32_t turn_{0};
  uint8_t current_repetition_count_{1};
  absl::flat_hash_map<RepetitionKeyWrapper, uint8_t> repetition_counts_{};

  // Repetition count for each board position.
};

}  // namespace alphazero::brandubh_gs