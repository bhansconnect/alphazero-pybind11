#include "onitama_gs.h"

namespace alphazero::onitama_gs {

[[nodiscard]] std::unique_ptr<GameState> OnitamaGS::copy() const noexcept {
  return std::make_unique<OnitamaGS>(board_, player_, p0_card0_, p0_card1_,
                                     p1_card0_, p1_card1_, waiting_card_, turn_,
                                     num_cards_, max_turns_);
}

[[nodiscard]] bool OnitamaGS::operator==(
    const GameState& other) const noexcept {
  const auto* other_cs = dynamic_cast<const OnitamaGS*>(&other);
  if (other_cs == nullptr) {
    return false;
  }
  for (auto p = 0; p < PIECE_TYPES; ++p) {
    for (auto h = 0; h < HEIGHT; ++h) {
      for (auto w = 0; w < WIDTH; ++w) {
        if (other_cs->board_(p, h, w) != board_(p, h, w)) {
          return false;
        }
      }
    }
  }
  return (other_cs->player_ == player_ && other_cs->p0_card0_ == p0_card0_ &&
          other_cs->p0_card1_ == p0_card1_ &&
          other_cs->p1_card0_ == p1_card0_ &&
          other_cs->p1_card1_ == p1_card1_ &&
          other_cs->waiting_card_ == waiting_card_);
}

void OnitamaGS::hash(absl::HashState h) const {
  h = absl::HashState::combine_contiguous(std::move(h), board_.data(),
                                          board_.size());
  absl::HashState::combine(std::move(h), player_, p0_card0_, p0_card1_,
                           p1_card0_, p1_card1_, waiting_card_);
}

[[nodiscard]] std::pair<const int8_t*, const int8_t*> OnitamaGS::player_cards(
    int wanted_player) const noexcept {
  if (wanted_player == 0) {
    return std::make_pair(&p0_card0_, &p0_card1_);
  } else {
    return std::make_pair(&p1_card0_, &p1_card1_);
  }
}

[[nodiscard]] std::pair<int8_t*, int8_t*> OnitamaGS::player_cards(
    int wanted_player) noexcept {
  if (wanted_player == 0) {
    return std::make_pair(&p0_card0_, &p0_card1_);
  } else {
    return std::make_pair(&p1_card0_, &p1_card1_);
  }
}

[[nodiscard]] Vector<uint8_t> OnitamaGS::valid_moves() const noexcept {
  auto valids = Vector<uint8_t>{NUM_MOVES};
  valids.setZero();

  bool has_move = false;
  auto [card0, card1] = player_cards(player_);
  int8_t master_layer = (player_ == 0) ? P0_MASTER_LAYER : P1_MASTER_LAYER;
  int8_t pawn_layer = (player_ == 0) ? P0_PAWN_LAYER : P1_PAWN_LAYER;
  int move_mult = (player_ == 0) ? 1 : -1;
  for (int ci = 0; ci < 2; ++ci) {
    const auto& card = (ci == 0) ? CARDS[*card0] : CARDS[*card1];
    for (int from_h = 0; from_h < HEIGHT; ++from_h) {
      for (int from_w = 0; from_w < WIDTH; ++from_w) {
        if (board_(master_layer, from_h, from_w) == 1 ||
            board_(pawn_layer, from_h, from_w) == 1) {
          for (const auto& move : card.movements) {
            int to_h = from_h + move.first * move_mult;
            int to_w = from_w + move.second * move_mult;
            if (to_h < 0 || to_h >= HEIGHT) {
              continue;
            }
            if (to_w < 0 || to_w >= WIDTH) {
              continue;
            }
            if (board_(master_layer, to_h, to_w) == 1 ||
                board_(pawn_layer, to_h, to_w) == 1) {
              continue;
            }
            has_move = true;
            int index = ci * (WIDTH * HEIGHT * WIDTH * HEIGHT) +
                        from_h * (WIDTH * HEIGHT * WIDTH) +
                        from_w * (WIDTH * HEIGHT) + to_h * WIDTH + to_w;
            valids(index) = 1;
          }
        }
      }
    }
  }

  if (!has_move) {
    valids(NUM_MOVES - 2) = 1;
    valids(NUM_MOVES - 1) = 1;
  }
  return valids;
}

void OnitamaGS::play_move(uint32_t move) {
  if (move >= NUM_MOVES) {
    throw std::runtime_error{"Invalid move: You have a bug in your code."};
  }
  auto [card0, card1] = player_cards(player_);
  auto* swap_card = card1;
  if (move < WIDTH * HEIGHT * WIDTH * HEIGHT || move == NUM_MOVES - 2) {
    swap_card = card0;
  }
  std::swap(waiting_card_, *swap_card);
  player_ = (player_ + 1) % 2;
  ++turn_;
  if (move >= NUM_MOVES - 2) {
    // No real move, just swap cards.
    return;
  }

  uint32_t actual_move = move % (WIDTH * HEIGHT * WIDTH * HEIGHT);
  int8_t to_w = actual_move % WIDTH;
  actual_move /= WIDTH;
  int8_t to_h = actual_move % HEIGHT;
  actual_move /= HEIGHT;
  int8_t from_w = actual_move % WIDTH;
  int8_t from_h = actual_move / WIDTH;

  for (int p = 0; p < PIECE_TYPES; ++p) {
    board_(p, to_h, to_w) = board_(p, from_h, from_w);
    board_(p, from_h, from_w) = 0;
  }
}

[[nodiscard]] std::optional<Vector<float>> OnitamaGS::scores() const noexcept {
  auto scores = SizedVector<float, 3>{};
  scores.setZero();
  // P0 has lower thrown.
  if (board_(P0_MASTER_LAYER, 4, 2) == 1) {
    scores(0) = 1;
    return scores;
  }
  // P1 has upper thrown.
  if (board_(P1_MASTER_LAYER, 0, 2) == 1) {
    scores(1) = 1;
    return scores;
  }

  int8_t p0_has_master = 0;
  int8_t p1_has_master = 0;
  for (auto h = 0; h < HEIGHT; ++h) {
    for (auto w = 0; w < WIDTH; ++w) {
      p0_has_master += board_(P0_MASTER_LAYER, h, w);
      p1_has_master += board_(P1_MASTER_LAYER, h, w);
    }
  }
  if (p0_has_master == 0) {
    scores(1) = 1;
    return scores;
  }
  if (p1_has_master == 0) {
    scores(0) = 1;
    return scores;
  }

  if (turn_ >= max_turns_) {
    scores(2) = 1;
    return scores;
  }
  return std::nullopt;
}

[[nodiscard]] Tensor<float, 3> OnitamaGS::canonicalized() const noexcept {
  auto out = CanonicalTensor{};
  for (auto p = 0; p < PIECE_TYPES; ++p) {
    for (auto h = 0; h < HEIGHT; ++h) {
      for (auto w = 0; w < WIDTH; ++w) {
        out(p, h, w) = board_(p, h, w);
      }
    }
  }

  int offset = PIECE_TYPES;
  auto p = player_ + offset;
  auto op = (player_ + 1) % 2 + offset;
  for (auto h = 0; h < HEIGHT; ++h) {
    for (auto w = 0; w < WIDTH; ++w) {
      out(p, h, w) = 1;
      out(op, h, w) = 0;
    }
  }
  offset += 2;

  // Zero card planes before setting specific indices.
  for (auto i = 0; i < 10; ++i) {
    for (auto h = 0; h < HEIGHT; ++h) {
      for (auto w = 0; w < WIDTH; ++w) {
        out(offset + i, h, w) = 0;
      }
    }
  }

  for (const auto& card_index :
       {p0_card0_, p0_card1_, waiting_card_, p1_card0_, p1_card1_}) {
    const auto& card = CARDS[card_index];
    for (const auto& move : card.movements) {
      out(offset, WIDTH / 2 + move.first, HEIGHT / 2 + move.second) = 1;
      // Card from other player perspective.
      out(offset + 5, WIDTH / 2 - move.first, HEIGHT / 2 - move.second) = 1;
    }
    ++offset;
  }

  return out;
}

[[nodiscard]] PlayHistory swap_cards(const PlayHistory& base,
                                     uint8_t player) noexcept {
  static_assert(CANONICAL_SHAPE[0] == 16,
                "this function must be updated when dimensions change");
  PlayHistory mirror;
  mirror.v = base.v;
  mirror.canonical = CanonicalTensor{};
  for (auto p = 0; p < PIECE_TYPES; ++p) {
    for (auto h = 0; h < HEIGHT; ++h) {
      for (auto w = 0; w < WIDTH; ++w) {
        mirror.canonical(p, h, w) = base.canonical(p, h, w);
      }
    }
  }
  int offset = PIECE_TYPES;
  for (auto h = 0; h < HEIGHT; ++h) {
    for (auto w = 0; w < WIDTH; ++w) {
      mirror.canonical(offset, h, w) = base.canonical(offset, h, w);
      mirror.canonical(offset + 1, h, w) = base.canonical(offset + 1, h, w);
    }
  }
  uint8_t current_player = mirror.canonical(offset + 1, 0, 0);

  offset += 2;
  auto second_offset = offset + 5;
  if (player == 0) {
    // swap p0 cards, which is the first 2.
    for (auto h = 0; h < HEIGHT; ++h) {
      for (auto w = 0; w < WIDTH; ++w) {
        mirror.canonical(offset, h, w) = base.canonical(offset + 1, h, w);
        mirror.canonical(offset + 1, h, w) = base.canonical(offset, h, w);
        mirror.canonical(offset + 2, h, w) = base.canonical(offset + 2, h, w);
        mirror.canonical(offset + 3, h, w) = base.canonical(offset + 3, h, w);
        mirror.canonical(offset + 4, h, w) = base.canonical(offset + 4, h, w);
        // Also swap the second versions of the cards
        mirror.canonical(second_offset, h, w) =
            base.canonical(second_offset + 1, h, w);
        mirror.canonical(second_offset + 1, h, w) =
            base.canonical(second_offset, h, w);
        mirror.canonical(second_offset + 2, h, w) =
            base.canonical(second_offset + 2, h, w);
        mirror.canonical(second_offset + 3, h, w) =
            base.canonical(second_offset + 3, h, w);
        mirror.canonical(second_offset + 4, h, w) =
            base.canonical(second_offset + 4, h, w);
      }
    }
  } else {
    // swap p1 cards, which are the last 2.
    for (auto h = 0; h < HEIGHT; ++h) {
      for (auto w = 0; w < WIDTH; ++w) {
        mirror.canonical(offset, h, w) = base.canonical(offset, h, w);
        mirror.canonical(offset + 1, h, w) = base.canonical(offset + 1, h, w);
        mirror.canonical(offset + 2, h, w) = base.canonical(offset + 2, h, w);
        mirror.canonical(offset + 3, h, w) = base.canonical(offset + 4, h, w);
        mirror.canonical(offset + 4, h, w) = base.canonical(offset + 3, h, w);
        // Also swap the second versions of the cards
        mirror.canonical(second_offset, h, w) =
            base.canonical(second_offset, h, w);
        mirror.canonical(second_offset + 1, h, w) =
            base.canonical(second_offset + 1, h, w);
        mirror.canonical(second_offset + 2, h, w) =
            base.canonical(second_offset + 2, h, w);
        mirror.canonical(second_offset + 3, h, w) =
            base.canonical(second_offset + 4, h, w);
        mirror.canonical(second_offset + 4, h, w) =
            base.canonical(second_offset + 3, h, w);
      }
    }
  }
  mirror.pi = Vector<float>{NUM_MOVES};
  if (current_player == player) {
    // We need to update the actions do to swapping the current players cards.
    // swap first and second WIDTH * HEIGHT * WIDTH * HEIGHT move
    auto size = WIDTH * HEIGHT * WIDTH * HEIGHT;
    for (auto i = 0; i < size; ++i) {
      mirror.pi(i) = base.pi(i + size);
      mirror.pi(i + size) = base.pi(i);
    }
    // Also swap last 2 moves.
    mirror.pi(NUM_MOVES - 2) = base.pi(NUM_MOVES - 1);
    mirror.pi(NUM_MOVES - 1) = base.pi(NUM_MOVES - 2);
  } else {
    // No need to update the action space.
    for (auto i = 0; i < NUM_MOVES; ++i) {
      mirror.pi(i) = base.pi(i);
    }
  }
  return mirror;
}

[[nodiscard]] std::vector<PlayHistory> OnitamaGS::symmetries(
    const PlayHistory& base) const noexcept {
  PlayHistory p0_swap = swap_cards(base, 0);
  PlayHistory p1_swap = swap_cards(base, 1);
  PlayHistory both_swap = swap_cards(p0_swap, 1);
  return {base, p0_swap, p1_swap, both_swap};
}

[[nodiscard]] std::string OnitamaGS::dump() const noexcept {
  auto out = "Current Player: " + std::to_string(player_) + '\n';
  out += "Turn: " + std::to_string(turn_) + " / " + std::to_string(max_turns_) +
         '\n';
  out += "Player 0 Cards: " + CARDS[p0_card0_].name + ", " +
         CARDS[p0_card1_].name + '\n';
  out += "Waiting Card: " + CARDS[waiting_card_].name + '\n';
  out += "Player 1 Cards: " + CARDS[p1_card0_].name + ", " +
         CARDS[p1_card1_].name + '\n';
  // TODO: add option to display cards here.
  for (auto h = 0; h < HEIGHT; ++h) {
    for (auto w = 0; w < WIDTH; ++w) {
      if (board_(P0_MASTER_LAYER, h, w) == 1) {
        out += 'X';
      } else if (board_(P0_PAWN_LAYER, h, w) == 1) {
        out += 'x';
      } else if (board_(P1_MASTER_LAYER, h, w) == 1) {
        out += 'O';
      } else if (board_(P1_PAWN_LAYER, h, w) == 1) {
        out += 'o';
      } else {
        out += '.';
      }
    }
    out += '\n';
  }
  out += '\n';
  return out;
}

}  // namespace alphazero::onitama_gs