#include "onitami_gs.h"

namespace alphazero::onitami_gs {

[[nodiscard]] std::unique_ptr<GameState> OnitamiGS::copy() const noexcept {
  return std::make_unique<OnitamiGS>(board_, player_, p0_card0_, p0_card1_,
                                     p1_card0_, p1_card1_, waiting_card_,
                                     turn_);
}

[[nodiscard]] bool OnitamiGS::operator==(
    const GameState& other) const noexcept {
  const auto* other_cs = dynamic_cast<const OnitamiGS*>(&other);
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

void OnitamiGS::hash(absl::HashState h) const {
  h = absl::HashState::combine_contiguous(std::move(h), board_.data(),
                                          board_.size());
  absl::HashState::combine(std::move(h), player_, p0_card0_, p0_card1_,
                           p1_card0_, p1_card1_, waiting_card_);
}

[[nodiscard]] Vector<uint8_t> OnitamiGS::valid_moves() const noexcept {
  // TODO
  auto valids = Vector<uint8_t>{WIDTH};
  for (auto w = 0; w < WIDTH; ++w) {
    valids(w) =
        static_cast<uint8_t>(board_(0, 0, w) == 0 && board_(1, 0, w) == 0);
  }
  return valids;
}

void OnitamiGS::play_move(uint32_t move) {
  // TODO
  for (auto h = HEIGHT - 1; h >= 0; --h) {
    if (board_(0, h, move) == 0 && board_(1, h, move) == 0) {
      board_(player_, h, move) = 1;
      player_ = (player_ + 1) % 2;
      ++turn_;
      return;
    }
  }
  throw std::runtime_error{"Invalid move: You have a bug in your code."};
}

[[nodiscard]] std::optional<Vector<float>> OnitamiGS::scores() const noexcept {
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

[[nodiscard]] Tensor<float, 3> OnitamiGS::canonicalized() const noexcept {
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

[[nodiscard]] std::vector<PlayHistory> OnitamiGS::symmetries(
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

[[nodiscard]] std::string OnitamiGS::dump() const noexcept {
  auto out = "Current Player: " + std::to_string(player_) + '\n';
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

}  // namespace alphazero::onitami_gs