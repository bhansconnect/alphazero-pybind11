#pragma once

#include <string_view>

#include "game_state.h"

namespace alphazero::onitama_gs {

// Hopefully this is large enough that it practically never gets hit.
constexpr const uint16_t DEFAULT_MAX_TURNS = 150;

constexpr const int P0_MASTER_LAYER = 0;
constexpr const int P0_PAWN_LAYER = 1;
constexpr const int P1_MASTER_LAYER = 2;
constexpr const int P1_PAWN_LAYER = 3;

constexpr const int PIECE_TYPES = 4;
constexpr const int WIDTH = 5;
constexpr const int HEIGHT = 5;
// Move from square 1 to square 2 with card 0 or 1. Also no moves just pass card
// 0 or 1.
constexpr const int NUM_MOVES = 2 * WIDTH * HEIGHT * WIDTH * HEIGHT + 2;
constexpr const int NUM_PLAYERS = 2;
constexpr const int NUM_SYMMETRIES = 4;  // Only card swapping is symmetric.
constexpr const std::array<int, 3> BOARD_SHAPE = {PIECE_TYPES, HEIGHT, WIDTH};
constexpr const std::array<int, 3> CANONICAL_SHAPE = {16, HEIGHT, WIDTH};

struct Card {
  std::string name;
  // movement relative to the piece that is valid in height, width format.
  // The should be relative the the player at the top of the board (P0)
  // So 1, 0 would mean moving down 1 square. (- is up) (+ is down)
  // and 0, 1 would mean moving to the right 1 square. (- is left) (+ is right)
  std::vector<std::pair<int8_t, int8_t>> movements;
  int8_t starting_player;
};

using CardImage = std::array<std::array<int8_t, 5>, 5>;

struct CardSpec {
  std::string_view name;
  CardImage image;
  int8_t starting_player;
};

constexpr const int NUM_CARDS = 32;
constexpr std::array<CardSpec, NUM_CARDS> CARD_SPECS = {
    CardSpec{.name = "TIGER",
             .image = {{{{0, 0, 1, 0, 0}},
                        {{0, 0, 0, 0, 0}},
                        {{0, 0, 2, 0, 0}},
                        {{0, 0, 1, 0, 0}},
                        {{0, 0, 0, 0, 0}}}},
             .starting_player = 0},
    CardSpec{.name = "FROG",
             .image = {{{{0, 0, 0, 0, 0}},
                        {{0, 1, 0, 0, 0}},
                        {{1, 0, 2, 0, 0}},
                        {{0, 0, 0, 1, 0}},
                        {{0, 0, 0, 0, 0}}}},
             .starting_player = 1},
    CardSpec{.name = "CRAB",
             .image = {{{{0, 0, 0, 0, 0}},
                        {{0, 0, 1, 0, 0}},
                        {{1, 0, 2, 0, 1}},
                        {{0, 0, 0, 0, 0}},
                        {{0, 0, 0, 0, 0}}}},
             .starting_player = 0},
    CardSpec{.name = "ROOSTER",
             .image = {{{{0, 0, 0, 0, 0}},
                        {{0, 0, 0, 1, 0}},
                        {{0, 1, 2, 1, 0}},
                        {{0, 1, 0, 0, 0}},
                        {{0, 0, 0, 0, 0}}}},
             .starting_player = 1},
    CardSpec{.name = "MONKEY",
             .image = {{{{0, 0, 0, 0, 0}},
                        {{0, 1, 0, 1, 0}},
                        {{0, 0, 2, 0, 0}},
                        {{0, 1, 0, 1, 0}},
                        {{0, 0, 0, 0, 0}}}},
             .starting_player = 0},
    CardSpec{.name = "HORSE",
             .image = {{{{0, 0, 0, 0, 0}},
                        {{0, 0, 1, 0, 0}},
                        {{0, 1, 2, 0, 0}},
                        {{0, 0, 1, 0, 0}},
                        {{0, 0, 0, 0, 0}}}},
             .starting_player = 1},
    CardSpec{.name = "CRANE",
             .image = {{{{0, 0, 0, 0, 0}},
                        {{0, 0, 1, 0, 0}},
                        {{0, 0, 2, 0, 0}},
                        {{0, 1, 0, 1, 0}},
                        {{0, 0, 0, 0, 0}}}},
             .starting_player = 0},
    CardSpec{.name = "COBRA",
             .image = {{{{0, 0, 0, 0, 0}},
                        {{0, 0, 0, 1, 0}},
                        {{0, 1, 2, 0, 0}},
                        {{0, 0, 0, 1, 0}},
                        {{0, 0, 0, 0, 0}}}},
             .starting_player = 1},
    CardSpec{.name = "DRAGON",
             .image = {{{{0, 0, 0, 0, 0}},
                        {{1, 0, 0, 0, 1}},
                        {{0, 0, 2, 0, 0}},
                        {{0, 1, 0, 1, 0}},
                        {{0, 0, 0, 0, 0}}}},
             .starting_player = 1},
    CardSpec{.name = "RABBIT",
             .image = {{{{0, 0, 0, 0, 0}},
                        {{0, 0, 0, 1, 0}},
                        {{0, 0, 2, 0, 1}},
                        {{0, 1, 0, 0, 0}},
                        {{0, 0, 0, 0, 0}}}},
             .starting_player = 0},
    CardSpec{.name = "ELEPHANT",
             .image = {{{{0, 0, 0, 0, 0}},
                        {{0, 1, 0, 1, 0}},
                        {{0, 1, 2, 1, 0}},
                        {{0, 0, 0, 0, 0}},
                        {{0, 0, 0, 0, 0}}}},
             .starting_player = 1},
    CardSpec{.name = "GOOSE",
             .image = {{{{0, 0, 0, 0, 0}},
                        {{0, 1, 0, 0, 0}},
                        {{0, 1, 2, 1, 0}},
                        {{0, 0, 0, 1, 0}},
                        {{0, 0, 0, 0, 0}}}},
             .starting_player = 0},
    CardSpec{.name = "MANTIS",
             .image = {{{{0, 0, 0, 0, 0}},
                        {{0, 1, 0, 1, 0}},
                        {{0, 0, 2, 0, 0}},
                        {{0, 0, 1, 0, 0}},
                        {{0, 0, 0, 0, 0}}}},
             .starting_player = 1},
    CardSpec{.name = "OX",
             .image = {{{{0, 0, 0, 0, 0}},
                        {{0, 0, 1, 0, 0}},
                        {{0, 0, 2, 1, 0}},
                        {{0, 0, 1, 0, 0}},
                        {{0, 0, 0, 0, 0}}}},
             .starting_player = 0},
    CardSpec{.name = "BOAR",
             .image = {{{{0, 0, 0, 0, 0}},
                        {{0, 0, 1, 0, 0}},
                        {{0, 1, 2, 1, 0}},
                        {{0, 0, 0, 0, 0}},
                        {{0, 0, 0, 0, 0}}}},
             .starting_player = 1},
    CardSpec{.name = "EEL",
             .image = {{{{0, 0, 0, 0, 0}},
                        {{0, 1, 0, 0, 0}},
                        {{0, 0, 2, 1, 0}},
                        {{0, 1, 0, 0, 0}},
                        {{0, 0, 0, 0, 0}}}},
             .starting_player = 0},
    CardSpec{.name = "SEA SNAKE",
             .image = {{{{0, 0, 0, 0, 0}},
                        {{0, 0, 1, 0, 0}},
                        {{0, 0, 2, 0, 1}},
                        {{0, 1, 0, 0, 0}},
                        {{0, 0, 0, 0, 0}}}},
             .starting_player = 0},
    CardSpec{.name = "OTTER",
             .image = {{{{0, 0, 0, 0, 0}},
                        {{0, 1, 0, 0, 0}},
                        {{0, 0, 2, 0, 1}},
                        {{0, 0, 0, 1, 0}},
                        {{0, 0, 0, 0, 0}}}},
             .starting_player = 1},
    CardSpec{.name = "TANUKI",
             .image = {{{{0, 0, 0, 0, 0}},
                        {{0, 0, 1, 0, 1}},
                        {{0, 0, 2, 0, 0}},
                        {{0, 1, 0, 0, 0}},
                        {{0, 0, 0, 0, 0}}}},
             .starting_player = 0},
    CardSpec{.name = "RAT",
             .image = {{{{0, 0, 0, 0, 0}},
                        {{0, 0, 1, 0, 0}},
                        {{0, 1, 2, 0, 0}},
                        {{0, 0, 0, 1, 0}},
                        {{0, 0, 0, 0, 0}}}},
             .starting_player = 1},
    CardSpec{.name = "PANDA",
             .image = {{{{0, 0, 0, 0, 0}},
                        {{0, 0, 1, 1, 0}},
                        {{0, 0, 2, 0, 0}},
                        {{0, 1, 0, 0, 0}},
                        {{0, 0, 0, 0, 0}}}},
             .starting_player = 1},
    CardSpec{.name = "DOG",
             .image = {{{{0, 0, 0, 0, 0}},
                        {{0, 1, 0, 0, 0}},
                        {{0, 1, 2, 0, 0}},
                        {{0, 1, 0, 0, 0}},
                        {{0, 0, 0, 0, 0}}}},
             .starting_player = 0},
    CardSpec{.name = "KIRIN",
             .image = {{{{0, 1, 0, 1, 0}},
                        {{0, 0, 0, 0, 0}},
                        {{0, 0, 2, 0, 0}},
                        {{0, 0, 0, 0, 0}},
                        {{0, 0, 1, 0, 0}}}},
             .starting_player = 1},
    CardSpec{.name = "PHOENIX",
             .image = {{{{0, 0, 0, 0, 0}},
                        {{0, 1, 0, 1, 0}},
                        {{1, 0, 2, 0, 1}},
                        {{0, 0, 0, 0, 0}},
                        {{0, 0, 0, 0, 0}}}},
             .starting_player = 0},
    CardSpec{.name = "VIPER",
             .image = {{{{0, 0, 0, 0, 0}},
                        {{0, 0, 1, 0, 0}},
                        {{1, 0, 2, 0, 0}},
                        {{0, 0, 0, 1, 0}},
                        {{0, 0, 0, 0, 0}}}},
             .starting_player = 1},
    CardSpec{.name = "SABLE",
             .image = {{{{0, 0, 0, 0, 0}},
                        {{0, 0, 0, 1, 0}},
                        {{1, 0, 2, 0, 0}},
                        {{0, 1, 0, 0, 0}},
                        {{0, 0, 0, 0, 0}}}},
             .starting_player = 0},
    CardSpec{.name = "IGUANA",
             .image = {{{{0, 0, 0, 0, 0}},
                        {{1, 0, 1, 0, 0}},
                        {{0, 0, 2, 0, 0}},
                        {{0, 0, 0, 1, 0}},
                        {{0, 0, 0, 0, 0}}}},
             .starting_player = 1},
    CardSpec{.name = "MOUSE",
             .image = {{{{0, 0, 0, 0, 0}},
                        {{0, 0, 1, 0, 0}},
                        {{0, 0, 2, 1, 0}},
                        {{0, 1, 0, 0, 0}},
                        {{0, 0, 0, 0, 0}}}},
             .starting_player = 0},
    CardSpec{.name = "BEAR",
             .image = {{{{0, 0, 0, 0, 0}},
                        {{0, 1, 1, 0, 0}},
                        {{0, 0, 2, 0, 0}},
                        {{0, 0, 0, 1, 0}},
                        {{0, 0, 0, 0, 0}}}},
             .starting_player = 0},
    CardSpec{.name = "FOX",
             .image = {{{{0, 0, 0, 0, 0}},
                        {{0, 0, 0, 1, 0}},
                        {{0, 0, 2, 1, 0}},
                        {{0, 0, 0, 1, 0}},
                        {{0, 0, 0, 0, 0}}}},
             .starting_player = 1},
    CardSpec{.name = "GIRAFFE",
             .image = {{{{0, 0, 0, 0, 0}},
                        {{1, 0, 0, 0, 1}},
                        {{0, 0, 2, 0, 0}},
                        {{0, 0, 1, 0, 0}},
                        {{0, 0, 0, 0, 0}}}},
             .starting_player = 0},
    CardSpec{.name = "TURTLE",
             .image = {{{{0, 0, 0, 0, 0}},
                        {{0, 0, 0, 0, 0}},
                        {{1, 0, 2, 0, 1}},
                        {{0, 1, 0, 1, 0}},
                        {{0, 0, 0, 0, 0}}}},
             .starting_player = 1},
};

Card transform_spec(CardSpec spec) {
  std::vector<std::pair<int8_t, int8_t>> movements;
  for (int h = 0; h < HEIGHT; ++h) {
    for (int w = 0; w < WIDTH; ++w) {
      if (spec.image[h][w] == 1) {
        movements.emplace_back(HEIGHT / 2 - h, WIDTH / 2 - w);
      }
    }
  }
  return Card{
      .name = std::string{spec.name},
      .movements = movements,
      .starting_player = spec.starting_player,
  };
}

const std::array<Card, NUM_CARDS> generate_cards() {
  std::array<Card, NUM_CARDS> cards;
  for (int i = 0; i < NUM_CARDS; ++i) {
    cards[i] = transform_spec(CARD_SPECS[i]);
  }
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
  OnitamaGS(uint8_t num_cards = 16, uint16_t max_turns = DEFAULT_MAX_TURNS)
      : num_cards_(num_cards), max_turns_(max_turns) {
    assert((num_cards == 8 || num_cards == 16 || num_cards == 24 ||
            num_cards == 32) &&
           "onitama must be played with 8 (simplified) or 16 (full) cards");
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

    randomize_start();
  }
  OnitamaGS(BoardTensor board, int8_t player, int8_t p0_card1, int8_t p0_card2,
            int8_t p1_card1, int8_t p1_card2, int8_t waiting_card,
            uint16_t turn, uint8_t num_cards, uint16_t max_turns)
      : board_(board),
        turn_(turn),
        num_cards_(num_cards),
        max_turns_(max_turns),
        player_(player),
        p0_card0_(p0_card1),
        p0_card1_(p0_card2),
        p1_card0_(p1_card1),
        p1_card1_(p1_card2),
        waiting_card_(waiting_card) {}
  OnitamaGS(BoardTensor&& board, int8_t player, int8_t p0_card1,
            int8_t p0_card2, int8_t p1_card1, int8_t p1_card2,
            int8_t waiting_card, uint16_t turn, uint8_t num_cards,
            uint16_t max_turns)
      : board_(std::move(board)),
        turn_(turn),
        num_cards_(num_cards),
        max_turns_(max_turns),
        player_(player),
        p0_card0_(p0_card1),
        p0_card1_(p0_card2),
        p1_card0_(p1_card1),
        p1_card1_(p1_card2),
        waiting_card_(waiting_card) {}

  void randomize_start() noexcept override {
    // Randomly select 5 cards to play with.
    std::vector<int8_t> permutation;
    permutation.reserve(num_cards_);
    for (int i = 0; i < num_cards_; ++i) {
      permutation.push_back(i);
    }

    std::random_device rd;
    std::mt19937 g(rd());

    std::shuffle(permutation.begin(), permutation.end(), g);

    p0_card0_ = permutation[0];
    p0_card1_ = permutation[1];
    p1_card0_ = permutation[2];
    p1_card1_ = permutation[3];
    waiting_card_ = permutation[4];

    player_ = CARDS[waiting_card_].starting_player;
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
  [[nodiscard]] std::pair<const int8_t*, const int8_t*> player_cards(
      int wanted_player) const noexcept;

  [[nodiscard]] Card p0_card0() { return CARDS[p0_card0_]; }
  [[nodiscard]] Card p0_card1() { return CARDS[p0_card1_]; }
  [[nodiscard]] Card p1_card0() { return CARDS[p1_card0_]; }
  [[nodiscard]] Card p1_card1() { return CARDS[p1_card1_]; }
  [[nodiscard]] Card waiting_card() { return CARDS[waiting_card_]; }

 private:
  // Board contains a layer for each player.
  // A 0 means no piece, a 1 means a piece for that player.
  BoardTensor board_{};
  uint16_t turn_{0};
  uint8_t num_cards_;
  uint16_t max_turns_;
  int8_t player_;
  int8_t p0_card0_;
  int8_t p0_card1_;
  int8_t p1_card0_;
  int8_t p1_card1_;
  int8_t waiting_card_;
};

}  // namespace alphazero::onitama_gs
