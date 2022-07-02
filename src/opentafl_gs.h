#pragma once

#include <algorithm>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "game_state.h"

// This version is an extension of the fetlar hnefatafl rules.
// Found here: http://aagenielsen.dk/fetlar_rules_en.php
// The modifications come from the OpenTafl computer tafl tournament.
// Found here:
// https://soupbox.manywords.press/2016-opentafl-tafl-open-ai-tournament

namespace alphazero::opentafl_gs {

constexpr const uint16_t DEFAULT_MAX_TURNS = 400;

constexpr const int KING_LAYER = 0;
constexpr const int DEF_LAYER = 1;
constexpr const int ATK_LAYER = 2;

// Player 0 is the first player (the attackers).
// Player 1 is the second player (the king side defenders).
constexpr const int ATK_PLAYER = 0;
constexpr const int DEF_PLAYER = 1;

constexpr const int WIDTH = 11;
constexpr const int HEIGHT = 11;
constexpr const int NUM_MOVES = WIDTH * HEIGHT * (WIDTH + HEIGHT);
constexpr const int NUM_PLAYERS = 2;
constexpr const int NUM_SYMMETRIES = 8;
constexpr const std::array<int, 3> BOARD_SHAPE = {3, HEIGHT, WIDTH};
// 3 for board, 2 for current player, 2 for current repetition count.
// 1 for turn/max_turns.
// May want to add some historical positions.
constexpr const std::array<int, 3> CANONICAL_SHAPE = {8, HEIGHT, WIDTH};

using BoardTensor =
    SizedTensor<int8_t,
                Eigen::Sizes<BOARD_SHAPE[0], BOARD_SHAPE[1], BOARD_SHAPE[2]>>;

using CanonicalTensor =
    SizedTensor<float, Eigen::Sizes<CANONICAL_SHAPE[0], CANONICAL_SHAPE[1],
                                    CANONICAL_SHAPE[2]>>;

struct RepetitionKey {
  BoardTensor t;
  uint8_t p;

  RepetitionKey(BoardTensor tensor, uint8_t player) : t(tensor), p(player) {}
};
struct RepetitionKeyWrapper {
  RepetitionKeyWrapper(const BoardTensor& tensor, uint8_t player) {
    data = std::make_shared<RepetitionKey>(tensor, player);
  }
  RepetitionKeyWrapper(const RepetitionKeyWrapper& rkw) { data = rkw.data; }
  std::shared_ptr<RepetitionKey> data;
};
template <typename H>
H AbslHashValue(H h, const RepetitionKeyWrapper& rkw) {
  h = H::combine_contiguous(std::move(h), rkw.data->t.data(),
                            rkw.data->t.size());
  return H::combine(std::move(h), rkw.data->p);
}
bool operator==(const RepetitionKeyWrapper& lhs,
                const RepetitionKeyWrapper& rhs) {
  for (auto i = 0; i < BOARD_SHAPE[0]; ++i) {
    for (auto j = 0; j < BOARD_SHAPE[1]; ++j) {
      for (auto k = 0; k < BOARD_SHAPE[2]; ++k) {
        if (lhs.data->t(i, j, k) != rhs.data->t(i, j, k)) {
          return false;
        }
      }
    }
  }
  return rhs.data->p == lhs.data->p;
}

class OpenTaflGS : public GameState {
 public:
  OpenTaflGS(uint16_t max_turns = DEFAULT_MAX_TURNS) : max_turns_(max_turns) {
    board_.setZero();
    // King
    board_(KING_LAYER, 5, 5) = 1;

    // Defenders
    board_(DEF_LAYER, 3, 5) = 1;
    board_(DEF_LAYER, 4, 5) = 1;
    board_(DEF_LAYER, 5, 4) = 1;
    board_(DEF_LAYER, 5, 3) = 1;
    board_(DEF_LAYER, 6, 5) = 1;
    board_(DEF_LAYER, 7, 5) = 1;
    board_(DEF_LAYER, 5, 6) = 1;
    board_(DEF_LAYER, 5, 7) = 1;

    board_(DEF_LAYER, 4, 4) = 1;
    board_(DEF_LAYER, 4, 6) = 1;
    board_(DEF_LAYER, 6, 4) = 1;
    board_(DEF_LAYER, 6, 6) = 1;

    // Attackers
    board_(ATK_LAYER, 0, 3) = 1;
    board_(ATK_LAYER, 0, 4) = 1;
    board_(ATK_LAYER, 0, 5) = 1;
    board_(ATK_LAYER, 0, 6) = 1;
    board_(ATK_LAYER, 0, 7) = 1;
    board_(ATK_LAYER, 1, 5) = 1;
    board_(ATK_LAYER, 10, 3) = 1;
    board_(ATK_LAYER, 10, 4) = 1;
    board_(ATK_LAYER, 10, 5) = 1;
    board_(ATK_LAYER, 10, 6) = 1;
    board_(ATK_LAYER, 10, 7) = 1;
    board_(ATK_LAYER, 9, 5) = 1;
    board_(ATK_LAYER, 3, 0) = 1;
    board_(ATK_LAYER, 4, 0) = 1;
    board_(ATK_LAYER, 5, 0) = 1;
    board_(ATK_LAYER, 6, 0) = 1;
    board_(ATK_LAYER, 7, 0) = 1;
    board_(ATK_LAYER, 5, 1) = 1;
    board_(ATK_LAYER, 3, 10) = 1;
    board_(ATK_LAYER, 4, 10) = 1;
    board_(ATK_LAYER, 5, 10) = 1;
    board_(ATK_LAYER, 6, 10) = 1;
    board_(ATK_LAYER, 7, 10) = 1;
    board_(ATK_LAYER, 5, 9) = 1;
  }
  OpenTaflGS(
      BoardTensor board, int8_t player, uint16_t turn, uint16_t max_turns,
      uint8_t current_repetition_count,
      absl::flat_hash_map<const std::shared_ptr<RepetitionKey>, uint8_t>
          repetition_counts,
      std::shared_ptr<absl::flat_hash_set<RepetitionKeyWrapper>> board_intern)
      : board_(board),
        turn_(turn),
        max_turns_(max_turns),
        player_(player),
        current_repetition_count_(current_repetition_count),
        repetition_counts_(repetition_counts),
        board_intern_(board_intern) {
    // Prune old unused keys from global intern pool.
    if (board_intern_) {
      for (auto it = board_intern_->begin(), end = board_intern_->end();
           it != end;) {
        // `erase()` will invalidate `it`, so advance `it` first.
        auto copy_it = it++;
        if (copy_it->data.unique()) {
          board_intern_->erase(copy_it);
        }
      }
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
  [[nodiscard]] bool has_valid_moves() const noexcept;

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

  // Returns the number of symmetries the game has.
  [[nodiscard]] uint8_t num_symmetries() const noexcept override {
    return NUM_SYMMETRIES;
  }

  // Returns an list of all symetrical game states (including the base state).
  [[nodiscard]] std::vector<PlayHistory> symmetries(
      const PlayHistory& base) const noexcept override;

  // Returns a string representation of the game state.
  [[nodiscard]] std::string dump() const noexcept override;

  // Deletes all data that is not necessary for storing as a hash key.
  // This avoids wasting tons of space when caching states.
  void minimize_storage() override;

 private:
  // Board contains a layer for the king, other white pieces, and black
  // pieces. A 0 means no piece, a 1 means a piece of the respective type.
  BoardTensor board_{};
  uint16_t turn_{0};
  uint16_t max_turns_{DEFAULT_MAX_TURNS};
  int8_t player_{0};
  uint8_t current_repetition_count_{1};
  absl::flat_hash_map<const std::shared_ptr<RepetitionKey>, uint8_t>
      repetition_counts_{};
  // This is used to avoid constantly copying board tensors.
  // Instead a shared pointer to this set is copied.
  // It has pointer stability, so we can use pointers to it's elements as if
  // they were a copy of the element.
  std::shared_ptr<absl::flat_hash_set<RepetitionKeyWrapper>> board_intern_ =
      nullptr;

  // Repetition count for each board position.
};

}  // namespace alphazero::opentafl_gs
