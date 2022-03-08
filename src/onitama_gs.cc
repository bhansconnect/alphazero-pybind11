#include "onitama_gs.h"

namespace alphazero::onitama_gs {

[[nodiscard]] std::unique_ptr<GameState> OnitamaGS::copy() const noexcept {
  return std::make_unique<OnitamaGS>(board_, player_, p0_card0_, p0_card1_,
                                     p1_card0_, p1_card1_, waiting_card_,
                                     turn_);
}

[[nodiscard]] bool OnitamaGS::operator==(const GameState& other) const
    noexcept {
  const auto* other_cs = dynamic_cast<const OnitamaGS*>(&other);
  if (other_cs == nullptr) {
    return false;
  }
  for (auto p = 0; p < 4; ++p) {
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

  for (int p = 0; p < 4; ++p) {
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
  for (auto p = 0; p < 4; ++p) {
    for (auto h = 0; h < HEIGHT; ++h) {
      for (auto w = 0; w < WIDTH; ++w) {
        out(p, h, w) = board_(p, h, w);
      }
    }
  }
  auto p = player_ + 4;
  auto op = (player_ + 1) % 2 + 4;
  for (auto h = 0; h < HEIGHT; ++h) {
    for (auto w = 0; w < WIDTH; ++w) {
      out(p, h, w) = 1;
      out(op, h, w) = 0;
    }
  }

  // TOOD: Investigate if cards should be viewed with player perspective or if
  // the board should rotated.
  int offset = 6;
  for (const auto& card_index :
       {p0_card0_, p0_card1_, waiting_card_, p1_card0_, p1_card1_}) {
    const auto& card = CARDS[card_index];
    for (const auto& move : card.movements) {
      out(offset, WIDTH / 2 + move.first, HEIGHT / 2 + move.second) = 1;
    }
    ++offset;
  }

  return out;
}

[[nodiscard]] std::vector<PlayHistory> OnitamaGS::symmetries(
    const PlayHistory& base) const noexcept {
  // TODO: Eventually add symmetries.
  std::vector<PlayHistory> syms{base};
  return syms;
}

[[nodiscard]] std::string OnitamaGS::dump() const noexcept {
  auto out = "Current Player: " + std::to_string(player_) + '\n';
  out += "Player 0 Cards: " + CARDS[p0_card0_].name + ", " +
         CARDS[p0_card1_].name + '\n';
  out += "Wainting Card: " + CARDS[waiting_card_].name + '\n';
  out += "Player 1 Cards: " + CARDS[p1_card0_].name + ", " +
         CARDS[p1_card1_].name + '\n';
  // TODO: add option to display cards here.
  for (auto h = 0; h < HEIGHT; ++h) {
    for (auto w = 0; w < WIDTH; ++w) {
      if (board_(P0_MASTER_LAYER, h, w) == 1) {
        out += 'X';
      } else if (board_(P0_MASTER_LAYER, h, w) == 1) {
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