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

  bool has_move = false;
  // TODO for all squares if contains player pieces, add a move for each card
  // applied to that square.
  // cannot move into own pieces. Can move anywhere else on the board.

  if (!has_move) {
    valids[NUM_MOVES - 2] = 1;
    valids[NUM_MOVES - 1] = 1;
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
  // TODO
  auto scores = SizedVector<float, 3>{};
  scores.setZero();
  for (auto p = 0; p < 2; ++p) {
    for (auto h = 0; h < HEIGHT; ++h) {
      auto total = 0;
      for (auto w = 0; w < WIDTH; ++w) {
        if (board_(p, h, w) == 1) {
          ++total;
        } else {
          total = 0;
        }
        if (total == 4) {
          scores(p) = 1;
          return scores;
        }
      }
    }
    for (auto w = 0; w < WIDTH; ++w) {
      auto total = 0;
      for (auto h = 0; h < HEIGHT; ++h) {
        if (board_(p, h, w) == 1) {
          ++total;
        } else {
          total = 0;
        }
        if (total == 4) {
          scores(p) = 1;
          return scores;
        }
      }
    }
    for (auto h = 0; h < HEIGHT - 3; ++h) {
      for (auto w = 0; w < WIDTH - 3; ++w) {
        auto good = true;
        for (auto x = 0; x < 4; ++x) {
          if (board_(p, h + x, w + x) == 0) {
            good = false;
            break;
          }
        }
        if (good) {
          scores(p) = 1;
          return scores;
        }
      }
      for (auto w = WIDTH - 4; w < WIDTH; ++w) {
        auto good = true;
        for (auto x = 0; x < 4; ++x) {
          if (board_(p, h + x, w - x) == 0) {
            good = false;
            break;
          }
        }
        if (good) {
          scores(p) = 1;
          return scores;
        }
      }
    }
  }
  auto valids = valid_moves();
  for (auto w = 0; w < WIDTH; ++w) {
    if (valids(w) == 1) {
      return std::nullopt;
    }
  }
  scores(2) = 1;
  return scores;
}

[[nodiscard]] Tensor<float, 3> OnitamaGS::canonicalized() const noexcept {
  // TODO
  auto out = CanonicalTensor{};
  for (auto p = 0; p < 2; ++p) {
    for (auto h = 0; h < HEIGHT; ++h) {
      for (auto w = 0; w < WIDTH; ++w) {
        out(p, h, w) = board_(p, h, w);
      }
    }
  }
  auto p = player_ + 2;
  auto op = (player_ + 1) % 2 + 2;
  for (auto h = 0; h < HEIGHT; ++h) {
    for (auto w = 0; w < WIDTH; ++w) {
      out(p, h, w) = 1;
      out(op, h, w) = 0;
    }
  }
  return out;
}

[[nodiscard]] std::vector<PlayHistory> OnitamaGS::symmetries(
    const PlayHistory& base) const noexcept {
  // TODO
  std::vector<PlayHistory> syms{base};
  PlayHistory mirror;
  mirror.v = base.v;
  mirror.canonical = CanonicalTensor{};
  for (auto f = 0; f < CANONICAL_SHAPE[0]; ++f) {
    for (auto h = 0; h < HEIGHT; ++h) {
      for (auto w = 0; w < WIDTH; ++w) {
        mirror.canonical(f, h, w) = base.canonical(f, h, (WIDTH - 1) - w);
      }
    }
  }
  mirror.pi = Vector<float>{WIDTH};
  for (auto w = 0; w < WIDTH; ++w) {
    mirror.pi(w) = base.pi((WIDTH - 1) - w);
  }
  syms.push_back(mirror);
  return syms;
}

[[nodiscard]] std::string OnitamaGS::dump() const noexcept {
  auto out = "Current Player: " + std::to_string(player_) + '\n';
  out += "Player 0 Cards: " + CARDS[p0_card0_].name + ", " +
         CARDS[p0_card1_].name + '\n';
  out += "Wainting Card: " + CARDS[waiting_card_].name + '\n';
  out += "Player 1 Cards: " + CARDS[p1_card0_].name + ", " +
         CARDS[p1_card1_].name + '\n';
  // TODO: add cards here.
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