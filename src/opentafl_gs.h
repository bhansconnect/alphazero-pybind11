#pragma once

#include "absl/container/flat_hash_map.h"
#include "game_state.h"

// This version is an extension of the fetlar hnefatafl rules.
// Found here: http://aagenielsen.dk/fetlar_rules_en.php
// The modifications come from the OpenTafl computer tafl tournament.
// Found here:
// https://soupbox.manywords.press/2016-opentafl-tafl-open-ai-tournament

namespace alphazero::opentafl_gs {

constexpr const uint16_t DEFAULT_MAX_TURNS = 200;

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

    repetition_counts_[RepetitionKeyWrapper(board_, player_)] = 1;
  }
  OpenTaflGS(
      BoardTensor board, int8_t player, uint16_t turn, uint16_t max_turns,
      uint8_t current_repetition_count,
      absl::flat_hash_map<RepetitionKeyWrapper, uint8_t> repetition_counts)
      : board_(board),
        turn_(turn),
        max_turns_(max_turns),
        player_(player),
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
  absl::flat_hash_map<RepetitionKeyWrapper, uint8_t> repetition_counts_{};

  // Repetition count for each board position.
};
}  // namespace alphazero::opentafl_gs

namespace alphazero::tournament_opentafl_gs {

// 3 for board, 2 for current player, 2 for current repetition count, 3 for
// output of the first game. May want to add some historical positions.
constexpr const std::array<int, 3> CANONICAL_SHAPE = {
    opentafl_gs::CANONICAL_SHAPE[0] + 3, opentafl_gs::HEIGHT,
    opentafl_gs::WIDTH};

using CanonicalTensor =
    SizedTensor<float, Eigen::Sizes<CANONICAL_SHAPE[0], CANONICAL_SHAPE[1],
                                    CANONICAL_SHAPE[2]>>;

// TournamentOpenTaflGS is used for better training and balance.
// With the default OpenTaflGS, the network quickly gets to the point of
// practically always winning as the defender. This can mean that there will
// never be a new best net due to 100% defender winrate (50% winrate best vs new
// network). This hopes to allow training to work well with this imbalance. This
// converts the game to how it would be played in a tournament. Each Game is a
// set of 2 OpenTaflGS games where the players switch sides. The player recieve
// 1 point for a win, 0.5 points for a draw, and 0 points for a loss. The winner
// is the player with the most points. If the players have equal points, the
// winner is the player who won in less moves. If both games are a draw will the
// final results be a draw.
class TournamentOpenTaflGS : public GameState {
 public:
  TournamentOpenTaflGS(uint16_t max_turns = opentafl_gs::DEFAULT_MAX_TURNS)
      : game_(std::make_unique<opentafl_gs::OpenTaflGS>(max_turns)),
        max_turns_(max_turns) {}

  TournamentOpenTaflGS(std::unique_ptr<GameState> game, uint16_t max_turns,
                       const std::optional<uint16_t> first_game_turns,
                       const std::optional<int8_t> first_game_winner)
      : game_(std::move(game)),
        max_turns_(max_turns),
        first_game_turns_(first_game_turns),
        first_game_winner_(first_game_winner) {}

  [[nodiscard]] std::unique_ptr<GameState> copy() const noexcept override {
    return std::make_unique<TournamentOpenTaflGS>(
        game_->copy(), max_turns_, first_game_turns_, first_game_winner_);
  }
  [[nodiscard]] bool operator==(
      const GameState& other) const noexcept override {
    const auto* other_cs = dynamic_cast<const TournamentOpenTaflGS*>(&other);
    if (other_cs == nullptr) {
      return false;
    }
    return game_ == other_cs->game_ &&
           first_game_winner_ == other_cs->first_game_winner_;
  }

  void hash(absl::HashState h) const override {
    absl::HashState::combine(std::move(h), game_, first_game_winner_);
  }

  // Returns the current player. Players must be 0 indexed.
  [[nodiscard]] uint8_t current_player() const noexcept override {
    return (first_game_winner_.has_value() ? ((game_->current_player() + 1) % 2)
                                           : game_->current_player());
  };

  // Returns the current turn.
  [[nodiscard]] uint32_t current_turn() const noexcept override {
    return game_->current_turn();
  }

  // Returns the number of possible moves.
  [[nodiscard]] uint32_t num_moves() const noexcept override {
    return opentafl_gs::NUM_MOVES;
  }

  // Returns the number of players.
  [[nodiscard]] uint8_t num_players() const noexcept override { return 2; };

  // Returns a bool for all moves. The value is true if the move is playable
  // from this GameState.
  [[nodiscard]] Vector<uint8_t> valid_moves() const noexcept override {
    return game_->valid_moves();
  }

  // Plays a move, modifying the current GameState.
  void play_move(uint32_t move) override {
    game_->play_move(move);
    if (!first_game_winner_.has_value()) {
      auto scores = game_->scores();
      if (scores.has_value()) {
        if ((*scores)(0) == 1.0) {
          first_game_winner_ = 0;
        } else if ((*scores)(1) == 1.0) {
          first_game_winner_ = 1;
        } else {
          first_game_winner_ = -1;
        }
        first_game_turns_ = game_->current_turn();
        game_ = std::make_unique<opentafl_gs::OpenTaflGS>(max_turns_);
      }
    }
  }

  // Returns nullopt if the game isn't over.
  // Returns a one hot encode result of the game.
  // The first num player positions are set to 1 if that player won and 0
  // otherwise. The last position is set to 1 if the game was a draw and 0
  // otherwise.
  [[nodiscard]] std::optional<Vector<float>> scores() const noexcept override {
    if (!first_game_winner_.has_value()) {
      return std::nullopt;
    }
    auto scores = game_->scores();
    if (!scores.has_value()) {
      return std::nullopt;
    }
    int8_t second_game_winner;
    if ((*scores)(0) == 1.0) {
      second_game_winner = 0;
    } else if ((*scores)(1) == 1.0) {
      second_game_winner = 1;
    } else {
      second_game_winner = -1;
    }
    int8_t second_game_turns = game_->current_turn();

    uint16_t first_player_turns = max_turns_;
    uint16_t second_player_turns = max_turns_;
    float first_player_points = 0;
    float second_player_points = 0;

    if (*first_game_winner_ == 0) {
      first_player_points += 1;
      first_player_turns = *first_game_turns_;
    } else if (*first_game_winner_ == 1) {
      second_player_points += 1;
      second_player_turns = *first_game_turns_;
    } else {
      first_player_points += 0.5;
      second_player_points += 0.5;
    }
    if (second_game_winner == 0) {
      second_player_points += 1;
      second_player_turns = second_game_turns;
    } else if (second_game_winner == 1) {
      first_player_points += 1;
      first_player_turns = second_game_turns;
    } else {
      first_player_points += 0.5;
      second_player_points += 0.5;
    }

    auto output = SizedVector<float, 3>{};
    output.setZero();
    if (first_player_points > second_player_points) {
      output(0) = 1;
      return output;
    }
    if (first_player_points < second_player_points) {
      output(1) = 1;
      return output;
    }
    if (first_player_turns < second_player_turns) {
      output(0) = 1;
      return output;
    }
    if (first_player_turns > second_player_turns) {
      output(1) = 1;
      return output;
    }

    output(2) = 1;
    return output;
  }

  // Returns the canonicalized form of the board, ready for feeding to a NN.
  [[nodiscard]] Tensor<float, 3> canonicalized() const noexcept override {
    auto game_out = game_->canonicalized();
    auto out = CanonicalTensor{};
    for (auto h = 0; h < opentafl_gs::HEIGHT; ++h) {
      for (auto w = 0; w < opentafl_gs::WIDTH; ++w) {
        out(0, h, w) = 0;
        out(1, h, w) = 0;
        out(2, h, w) = 0;
      }
    }
    if (first_game_winner_.has_value()) {
      if (*first_game_winner_ == -1) {
        for (auto h = 0; h < opentafl_gs::HEIGHT; ++h) {
          for (auto w = 0; w < opentafl_gs::WIDTH; ++w) {
            out(2, h, w) = 1;
          }
        }
      } else {
        for (auto h = 0; h < opentafl_gs::HEIGHT; ++h) {
          for (auto w = 0; w < opentafl_gs::WIDTH; ++w) {
            out(*first_game_winner_, h, w) = 1;
          }
        }
      }
    }
    for (auto x = 0; x < opentafl_gs::CANONICAL_SHAPE[0]; ++x) {
      for (auto h = 0; h < opentafl_gs::HEIGHT; ++h) {
        for (auto w = 0; w < opentafl_gs::WIDTH; ++w) {
          out(x + 3, h, w) = game_out(x, h, w);
        }
      }
    }
    return out;
  }

  // Returns a string representation of the game state.
  [[nodiscard]] std::string dump() const noexcept override {
    auto out = std::string("Game ") +
               (!first_game_winner_.has_value() ? "1" : "2") + '\n';
    out += (first_game_winner_.has_value()
                ? ("First result: " + std::to_string(*first_game_winner_) +
                   " in " + std::to_string(*first_game_turns_))
                : "");
    out += game_->dump();
    return out;
  }

  // Deletes all data that is not necessary for storing as a hash key.
  // This avoids wasting tons of space when caching states.
  void minimize_storage() override { game_->minimize_storage(); }

 private:
  std::unique_ptr<GameState> game_;
  uint16_t max_turns_{opentafl_gs::DEFAULT_MAX_TURNS};
  std::optional<uint16_t> first_game_turns_{std::nullopt};
  std::optional<int8_t> first_game_winner_{std::nullopt};
};

}  // namespace alphazero::tournament_opentafl_gs