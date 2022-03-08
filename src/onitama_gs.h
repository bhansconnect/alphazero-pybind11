#pragma once

#include "game_state.h"

namespace alphazero::onitama_gs {

constexpr const int P0_MASTER_LAYER = 0;
constexpr const int P0_PAWN_LAYER = 1;
constexpr const int P1_MASTER_LAYER = 3;
constexpr const int P1_PAWN_LAYER = 4;

constexpr const int WIDTH = 5;
constexpr const int HEIGHT = 5;
// Move from square 1 to square 2 with card 0 or 1. Also no moves just pass card
// 0 or 1.
constexpr const int NUM_MOVES = 2 * WIDTH * HEIGHT * WIDTH * HEIGHT + 2;
constexpr const int NUM_PLAYERS = 2;
constexpr const int NUM_SYMMETRIES = 8;
constexpr const std::array<int, 3> BOARD_SHAPE = {4, HEIGHT, WIDTH};
constexpr const std::array<int, 3> CANONICAL_SHAPE = {11, HEIGHT, WIDTH};

// TODO define card array.
struct Card {
  std::string name;
  // movement relative to the piece that is valid in height, width format.
  // The should be relative the the player at the top of the board (P0)
  // So 1, 0 would mean moving down 1 square. (- is up) (+ is down)
  // and 0, 1 would mean moving to the right 1 square. (- is left) (+ is right)
  std::vector<std::pair<int8_t, int8_t>> movements;
  int8_t starting_player;
};

constexpr const int NUM_CARDS = 16;
std::array<Card, NUM_CARDS> generate_cards() {
  std::array<Card, NUM_CARDS> cards;
  cards[0].name = "TIGER";
  cards[0].movements = {{2, 0}, {-1, 0}};
  cards[0].starting_player = 0;

  cards[1].name = "FROG";
  cards[1].movements = {{1, 1}, {-1, -1}, {0, 2}};
  cards[1].starting_player = 1;

  cards[2].name = "CRAB";
  cards[2].movements = {{1, 0}, {0, 2}, {0, -2}};
  cards[2].starting_player = 0;

  cards[3].name = "GOOSE";
  cards[3].movements = {{1, 1}, {-1, -1}, {0, 1}, {0, -1}};
  cards[3].starting_player = 0;

  cards[4].name = "MONKEY";
  cards[4].movements = {{1, 1}, {-1, -1}, {1, -1}, {-1, 1}};
  cards[4].starting_player = 0;

  cards[5].name = "HORSE";
  cards[5].movements = {{1, 0}, {-1, 0}, {0, 1}};
  cards[5].starting_player = 1;

  cards[6].name = "CRANE";
  cards[6].movements = {{-1, -1}, {-1, 1}, {1, 0}};
  cards[6].starting_player = 0;

  cards[7].name = "EEL";
  cards[7].movements = {{1, 1}, {-1, 1}, {0, -1}};
  cards[7].starting_player = 0;

  cards[8].name = "DRAGON";
  cards[8].movements = {{1, 2}, {1, -2}, {-1, 1}, {-1, -1}};
  cards[8].starting_player = 1;

  cards[9].name = "RABBIT";
  cards[9].movements = {{-1, 1}, {1, -1}, {0, -2}};
  cards[9].starting_player = 0;

  cards[10].name = "ELEPHANT";
  cards[10].movements = {{1, 1}, {1, -1}, {0, 1}, {0, -1}};
  cards[10].starting_player = 1;

  cards[11].name = "ROOSTER";
  cards[11].movements = {{-1, 1}, {1, -1}, {0, 1}, {0, -1}};
  cards[11].starting_player = 1;

  cards[12].name = "MANTIS";
  cards[12].movements = {{1, 1}, {1, -1}, {-1, 0}};
  cards[12].starting_player = 1;

  cards[13].name = "OX";
  cards[13].movements = {{1, 0}, {-1, 0}, {0, -1}};
  cards[13].starting_player = 0;

  cards[14].name = "BOAR";
  cards[14].movements = {{1, 0}, {0, 1}, {0, -1}};
  cards[14].starting_player = 1;

  cards[15].name = "COBRA";
  cards[15].movements = {{0, 1}, {1, -1}, {-1, -1}};
  cards[15].starting_player = 1;

  return cards;
}

static const std::array<Card, NUM_CARDS> CARDS = generate_cards();

using BoardTensor =
    SizedTensor<int8_t,
                Eigen::Sizes<BOARD_SHAPE[0], BOARD_SHAPE[1], BOARD_SHAPE[2]>>;

using CanonicalTensor =
    SizedTensor<float, Eigen::Sizes<CANONICAL_SHAPE[0], CANONICAL_SHAPE[1],
                                    CANONICAL_SHAPE[2]>>;

class OnitamaGS : public GameState {
 public:
  OnitamaGS() {
    board_.setZero();

    // Masters
    board_(P0_MASTER_LAYER, 0, 2) = 1;
    board_(P1_MASTER_LAYER, 4, 2) = 1;
    // Pawns
    board_(P0_PAWN_LAYER, 0, 0) = 1;
    board_(P0_PAWN_LAYER, 0, 1) = 1;
    board_(P0_PAWN_LAYER, 0, 3) = 1;
    board_(P0_PAWN_LAYER, 0, 4) = 1;
    board_(P1_PAWN_LAYER, 4, 0) = 1;
    board_(P1_PAWN_LAYER, 4, 1) = 1;
    board_(P1_PAWN_LAYER, 4, 3) = 1;
    board_(P1_PAWN_LAYER, 4, 4) = 1;

    // Randomly select 5 cards to play with.
    std::array<int8_t, NUM_CARDS> permutation;
    for (int i = 0; i < NUM_CARDS; ++i) {
      permutation[i] = i;
    }

    std::random_device rd;
    std::mt19937 g(rd());

    std::shuffle(permutation.begin(), permutation.end(), g);

    p0_card0_ = permutation[0];
    p0_card1_ = permutation[1];
    p1_card0_ = permutation[2];
    p1_card1_ = permutation[3];
    waiting_card_ = permutation[4];
  }
  OnitamaGS(BoardTensor board, int8_t player, int8_t p0_card1, int8_t p0_card2,
            int8_t p1_card1, int8_t p1_card2, int8_t waiting_card,
            uint16_t turn)
      : board_(board),
        turn_(turn),
        player_(player),
        p0_card0_(p0_card1),
        p0_card1_(p0_card2),
        p1_card0_(p1_card1),
        p1_card1_(p1_card2),
        waiting_card_(waiting_card) {}
  OnitamaGS(BoardTensor&& board, int8_t player, int8_t p0_card1,
            int8_t p0_card2, int8_t p1_card1, int8_t p1_card2,
            int8_t waiting_card, uint16_t turn)
      : board_(std::move(board)),
        turn_(turn),
        player_(player),
        p0_card0_(p0_card1),
        p0_card1_(p0_card2),
        p1_card0_(p1_card1),
        p1_card1_(p1_card2),
        waiting_card_(waiting_card) {}

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
  [[nodiscard]] uint8_t num_players() const noexcept override {
    return NUM_PLAYERS;
  };

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
  void minimize_storage() override {}

  // Returns a reference the the carders for the specified player.
  [[nodiscard]] std::pair<int8_t*, int8_t*> player_cards(
      int wanted_player) noexcept;

 private:
  // Board contains a layer for each player.
  // A 0 means no piece, a 1 means a piece for that player.
  BoardTensor board_{};
  uint16_t turn_{0};
  int8_t player_;
  int8_t p0_card0_;
  int8_t p0_card1_;
  int8_t p1_card0_;
  int8_t p1_card1_;
  int8_t waiting_card_;
};

}  // namespace alphazero::onitama_gs