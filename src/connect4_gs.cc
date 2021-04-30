#include "connect4_gs.h"

namespace alphazero::connect4_gs {

[[nodiscard]] std::unique_ptr<GameState> Connect4GS::copy() const noexcept {
  return std::make_unique<Connect4GS>(board_, player_, turn_);
}

[[nodiscard]] bool Connect4GS::operator==(const GameState& other) const
    noexcept {
  const auto* other_cs = dynamic_cast<const Connect4GS*>(&other);
  if (other_cs == nullptr) {
    return false;
  }
  for (auto p = 0; p < 2; ++p) {
    for (auto h = 0; h < HEIGHT; ++h) {
      for (auto w = 0; w < WIDTH; ++w) {
        if (other_cs->board_(p, h, w) != board_(p, h, w)) {
          return false;
        }
      }
    }
  }
  return (other_cs->player_ == player_ && other_cs->turn_ == turn_);
}

[[nodiscard]] Vector<uint8_t> Connect4GS::valid_moves() const noexcept {
  auto valids = Vector<uint8_t>{WIDTH};
  for (auto w = 0; w < WIDTH; ++w) {
    valids(w) =
        static_cast<uint8_t>(board_(0, 0, w) == 0 && board_(1, 0, w) == 0);
  }
  return valids;
}

void Connect4GS::play_move(uint32_t move) {
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

[[nodiscard]] std::optional<Vector<float>> Connect4GS::scores() const noexcept {
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

[[nodiscard]] Tensor<float, 3> Connect4GS::canonicalized() const noexcept {
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

[[nodiscard]] std::string Connect4GS::dump() const noexcept {
  auto out = "Current Player: " + std::to_string(player_) + '\n';
  for (auto h = 0; h < HEIGHT; ++h) {
    for (auto w = 0; w < WIDTH; ++w) {
      if (board_(0, h, w) == 1) {
        out += 'X';
      } else if (board_(1, h, w) == 1) {
        out += 'O';
      } else {
        out += '.';
      }
    }
    out += '\n';
  }
  out += '\n';
  return out;
}

}  // namespace alphazero::connect4_gs