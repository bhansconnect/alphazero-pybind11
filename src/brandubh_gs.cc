#include "brandubh_gs.h"

#include "color.h"
#include "tafl_helper.h"

namespace alphazero::brandubh_gs {

[[nodiscard]] std::unique_ptr<GameState> BrandubhGS::copy() const noexcept {
  return std::make_unique<BrandubhGS>(board_, player_, turn_, max_turns_,
                                      current_repetition_count_,
                                      repetition_counts_, board_intern_);
}

[[nodiscard]] bool BrandubhGS::operator==(const GameState& other) const
    noexcept {
  const auto* other_cs = dynamic_cast<const BrandubhGS*>(&other);
  if (other_cs == nullptr) {
    return false;
  }
  for (auto p = 0; p < 3; ++p) {
    for (auto h = 0; h < HEIGHT; ++h) {
      for (auto w = 0; w < WIDTH; ++w) {
        if (other_cs->board_(p, h, w) != board_(p, h, w)) {
          return false;
        }
      }
    }
  }
  return (other_cs->player_ == player_ &&
          other_cs->current_repetition_count_ == current_repetition_count_);
}

void BrandubhGS::hash(absl::HashState h) const {
  h = absl::HashState::combine_contiguous(std::move(h), board_.data(),
                                          board_.size());
  absl::HashState::combine(std::move(h), player_, current_repetition_count_);
}

uint8_t piece_to_player(const BoardTensor& bt, uint8_t target_h,
                        uint8_t target_w) {
  if (bt(ATK_LAYER, target_h, target_w) == 1) {
    return ATK_PLAYER;
  } else if (bt(KING_LAYER, target_h, target_w) == 1 ||
             bt(DEF_LAYER, target_h, target_w) == 1) {
    return DEF_PLAYER;
  }

  throw std::runtime_error{
      "piece to player called on a square without pieces...rip"};
}

bool is_players_piece(const BoardTensor& bt, uint8_t player, int target_h,
                      int target_w) {
  return (player == DEF_PLAYER && (bt(KING_LAYER, target_h, target_w) == 1 ||
                                   bt(DEF_LAYER, target_h, target_w) == 1)) ||
         (player == ATK_PLAYER && bt(ATK_LAYER, target_h, target_w) == 1);
}

bool is_opponent_piece(const BoardTensor& bt, uint8_t player, int target_h,
                       int target_w) {
  return (player == ATK_PLAYER && (bt(KING_LAYER, target_h, target_w) == 1 ||
                                   bt(DEF_LAYER, target_h, target_w) == 1)) ||
         (player == DEF_PLAYER && bt(ATK_LAYER, target_h, target_w) == 1);
}

bool is_valid_square(const BoardTensor& bt, bool is_king, int target_h,
                     int target_w) {
  // Positions off the edge of the board are never valid.
  if (target_w < 0 || target_w >= WIDTH || target_h < 0 || target_h >= HEIGHT) {
    return false;
  }
  // Corners are only valid to the king.
  if ((target_w == 0 && target_h == 0) ||
      (target_w == WIDTH - 1 && target_h == 0) ||
      (target_w == 0 && target_h == HEIGHT - 1) ||
      (target_w == WIDTH - 1 && target_h == HEIGHT - 1)) {
    return is_king;
  }
  // Otherwise is valid if it is empty.
  return bt(0, target_h, target_w) == 0 && bt(1, target_h, target_w) == 0 &&
         bt(2, target_h, target_w) == 0;
}

[[nodiscard]] bool BrandubhGS::has_valid_moves() const noexcept {
  for (auto h = 0; h < HEIGHT; ++h) {
    for (auto w = 0; w < WIDTH; ++w) {
      if (is_players_piece(board_, player_, h, w)) {
        bool is_king = board_(0, h, w) == 1;
        {
          auto target_w = w + 1;
          while (is_valid_square(board_, is_king, h, target_w)) {
            // Special exception for the center. If not king you can move
            // through but not land on.
            if (h == 3 && target_w == 3 && !is_king) {
              ++target_w;
              continue;
            }
            return true;
          }
        }
        {
          auto target_w = w - 1;
          while (is_valid_square(board_, is_king, h, target_w)) {
            // Special exception for the center. If not king you can move
            // through but not land on.
            if (h == 3 && target_w == 3 && !is_king) {
              --target_w;
              continue;
            }
            return true;
          }
        }
        {
          auto target_h = h + 1;
          while (is_valid_square(board_, is_king, target_h, w)) {
            // Special exception for the center. If not king you can move
            // through but not land on.
            if (target_h == 3 && w == 3 && !is_king) {
              ++target_h;
              continue;
            }
            return true;
          }
        }
        {
          auto target_h = h - 1;
          while (is_valid_square(board_, is_king, target_h, w)) {
            // Special exception for the center. If not king you can move
            // through but not land on.
            if (target_h == 3 && w == 3 && !is_king) {
              --target_h;
              continue;
            }
            return true;
          }
        }
      }
    }
  }
  return false;
}

[[nodiscard]] Vector<uint8_t> BrandubhGS::valid_moves() const noexcept {
  auto valids = Vector<uint8_t>{NUM_MOVES};
  valids.setZero();
  for (auto h = 0; h < HEIGHT; ++h) {
    for (auto w = 0; w < WIDTH; ++w) {
      if (is_players_piece(board_, player_, h, w)) {
        bool is_king = board_(0, h, w) == 1;
        {
          auto target_w = w + 1;
          while (is_valid_square(board_, is_king, h, target_w)) {
            // Special exception for the center. If not king you can move
            // through but not land on.
            if (h == 3 && target_w == 3 && !is_king) {
              ++target_w;
              continue;
            }
            valids[(h * WIDTH + w) * (WIDTH + HEIGHT) + target_w + 0] = 1;
            ++target_w;
          }
        }
        {
          auto target_w = w - 1;
          while (is_valid_square(board_, is_king, h, target_w)) {
            // Special exception for the center. If not king you can move
            // through but not land on.
            if (h == 3 && target_w == 3 && !is_king) {
              --target_w;
              continue;
            }
            valids[(h * WIDTH + w) * (WIDTH + HEIGHT) + target_w + 0] = 1;
            --target_w;
          }
        }
        {
          auto target_h = h + 1;
          while (is_valid_square(board_, is_king, target_h, w)) {
            // Special exception for the center. If not king you can move
            // through but not land on.
            if (target_h == 3 && w == 3 && !is_king) {
              ++target_h;
              continue;
            }
            valids[(h * WIDTH + w) * (WIDTH + HEIGHT) + WIDTH + target_h] = 1;
            ++target_h;
          }
        }
        {
          auto target_h = h - 1;
          while (is_valid_square(board_, is_king, target_h, w)) {
            // Special exception for the center. If not king you can move
            // through but not land on.
            if (target_h == 3 && w == 3 && !is_king) {
              --target_h;
              continue;
            }
            valids[(h * WIDTH + w) * (WIDTH + HEIGHT) + WIDTH + target_h] = 1;
            --target_h;
          }
        }
      }
    }
  }
  return valids;
}

bool is_hostile_to(const BoardTensor& bt, uint8_t player, bool is_king,
                   int target_h, int target_w) {
  // Corners are always hostile to all.
  if ((target_w == 0 && target_h == 0) ||
      (target_w == WIDTH - 1 && target_h == 0) ||
      (target_w == 0 && target_h == HEIGHT - 1) ||
      (target_w == WIDTH - 1 && target_h == HEIGHT - 1)) {
    return true;
  }
  if (target_w == 3 && target_h == 3) {
    // To match OpenTafl Brandubh, the throne is also hostile to the king.
    (void)is_king;  // void cast to avoid unused warning.
    // Throne is never hostile to the king.
    // if (is_king) {
    // return false;
    // }
    // The throne is only hostile to other defenders if the king isn't there.
    if (player == DEF_PLAYER) {
      return bt(KING_LAYER, 3, 3) == 0;
    }
    // Throne is otherwise hostile to all.
    return true;
  }
  // Otherwise, opponent pieces decide if a square is hostile.
  return is_opponent_piece(bt, player, target_h, target_w);
}

bool captured(const BoardTensor& bt, int from_h, int from_w, int delta_h,
              int delta_w) {
  auto target_h = from_h + delta_h;
  auto target_w = from_w + delta_w;
  if (target_w < 0 || target_w >= WIDTH || target_h < 0 || target_h >= HEIGHT) {
    return false;
  }
  // For now matching OpenTafl, this is no longer special.
  // Special case: King on throne requires complete surrounding.
  // if (target_w == 3 && target_h == 3 && bt(KING_LAYER, 3, 3) == 1) {
  //   if (bt(ATK_LAYER, 2, 3) == 1 && bt(ATK_LAYER, 4, 3) == 1 &&
  //       bt(ATK_LAYER, 3, 2) == 1 && bt(ATK_LAYER, 3, 4) == 1) {
  //     return true;
  //   }
  //   return false;
  // }

  uint8_t from_player = piece_to_player(bt, from_h, from_w);
  // Only opponent pieces can be captured.
  if (!is_opponent_piece(bt, from_player, target_h, target_w)) {
    return false;
  }

  // Only captured if the final square is hostile to the target player.
  uint8_t target_player = piece_to_player(bt, target_h, target_w);
  auto final_h = target_h + delta_h;
  auto final_w = target_w + delta_w;
  if (final_w < 0 || final_w >= WIDTH || final_h < 0 || final_h >= HEIGHT) {
    return false;
  }
  return is_hostile_to(bt, target_player,
                       bt(KING_LAYER, target_h, target_w) == 1, final_h,
                       final_w);
}

void BrandubhGS::play_move(uint32_t move) {
  if (move < 0 || move >= NUM_MOVES) {
    throw std::runtime_error{"Invalid move: You have a bug in your code."};
  }
  // Initialize board interning if it is turn is 0.
  // We initialize it with the first move because we only want to share it
  // within one game, not across all games.
  if (turn_ == 0) {
    board_intern_ =
        std::make_shared<absl::flat_hash_set<RepetitionKeyWrapper>>();
    auto [key, _inserted_new] = board_intern_->emplace(board_, player_);
    const auto key_ptr = key->data;
    repetition_counts_[key_ptr] = 1;
  }

  // Move specified piece to row/column.
  auto new_loc = move % (WIDTH + HEIGHT);
  auto height_move = new_loc >= WIDTH;
  if (height_move) {
    new_loc -= WIDTH;
  }
  auto piece_loc = (move / (WIDTH + HEIGHT));
  auto piece_w = piece_loc % WIDTH;
  auto piece_h = piece_loc / WIDTH;

  auto new_h = piece_h;
  auto new_w = piece_w;
  if (height_move) {
    new_h = new_loc;
  } else {
    new_w = new_loc;
  }
  board_(0, new_h, new_w) = board_(0, piece_h, piece_w);
  board_(1, new_h, new_w) = board_(1, piece_h, piece_w);
  board_(2, new_h, new_w) = board_(2, piece_h, piece_w);

  board_(0, piece_h, piece_w) = 0;
  board_(1, piece_h, piece_w) = 0;
  board_(2, piece_h, piece_w) = 0;

  // Check if it captures anything.
  if (captured(board_, new_h, new_w, -1, 0)) {
    board_(0, new_h - 1, new_w) = 0;
    board_(1, new_h - 1, new_w) = 0;
    board_(2, new_h - 1, new_w) = 0;
    // All old repetitions are invalid since we no longer have the same number
    // of pieces.
    repetition_counts_.clear();
  }
  if (captured(board_, new_h, new_w, 1, 0)) {
    board_(0, new_h + 1, new_w) = 0;
    board_(1, new_h + 1, new_w) = 0;
    board_(2, new_h + 1, new_w) = 0;
    // All old repetitions are invalid since we no longer have the same number
    // of pieces.
    repetition_counts_.clear();
  }
  if (captured(board_, new_h, new_w, 0, -1)) {
    board_(0, new_h, new_w - 1) = 0;
    board_(1, new_h, new_w - 1) = 0;
    board_(2, new_h, new_w - 1) = 0;
    // All old repetitions are invalid since we no longer have the same number
    // of pieces.
    repetition_counts_.clear();
  }
  if (captured(board_, new_h, new_w, 0, 1)) {
    board_(0, new_h, new_w + 1) = 0;
    board_(1, new_h, new_w + 1) = 0;
    board_(2, new_h, new_w + 1) = 0;
    // All old repetitions are invalid since we no longer have the same number
    // of pieces.
    repetition_counts_.clear();
  }

  player_ = (player_ + 1) % 2;
  ++turn_;

  // Update repetitions.
  auto [key, _inserted_new] = board_intern_->emplace(board_, player_);
  const auto key_ptr = key->data;
  if (!repetition_counts_.contains(key_ptr)) {
    repetition_counts_[key_ptr] = 1;
  } else {
    ++repetition_counts_[key_ptr];
  }
  current_repetition_count_ = repetition_counts_[key_ptr];
}

[[nodiscard]] bool king_exists(const BoardTensor& bt) noexcept {
  for (auto h = 0; h < HEIGHT; ++h) {
    for (auto w = 0; w < WIDTH; ++w) {
      if (bt(KING_LAYER, h, w) == 1) {
        return true;
      }
    }
  }
  return false;
}

[[nodiscard]] std::optional<Vector<float>> BrandubhGS::scores() const noexcept {
  auto scores = SizedVector<float, 3>{};
  scores.setZero();
  // Check if 3 fold repetition.
  if (current_repetition_count_ >= 3) {
    // The player who brought us to this state probably was forced to do so.
    // (E.G. it was required to block a king escape)
    // The opponent brought us to this state and thus wins.
    // auto opponent = (player_ + 1) % 2;
    // scores(opponent) = 1;

    // There seems to be a bug in opentafl currently and this is backwards.
    // The player that enters a 3 fold repeat position loses.
    // So the player whose turn it is, wins.
    // Though it may just be the case that the definition of forced is
    // confounded.
    scores(player_) = 1;
    return scores;
  }
  // Check if the king is on a corner.
  if (board_(KING_LAYER, 0, 0) == 1 || board_(KING_LAYER, HEIGHT - 1, 0) == 1 ||
      board_(KING_LAYER, 0, WIDTH - 1) == 1 ||
      board_(KING_LAYER, HEIGHT - 1, WIDTH - 1) == 1) {
    scores(1) = 1;
    return scores;
  }
  // Check if the king still exists.
  if (!king_exists(board_)) {
    scores(0) = 1;
    return scores;
  }
  // Current player has no moves left.
  if (!has_valid_moves()) {
    auto opponent = (player_ + 1) % 2;
    scores(opponent) = 1;
    return scores;
  }
  // Draw if at/past max turns.
  if (turn_ >= max_turns_) {
    scores(2) = 1;
    return scores;
  }
  return std::nullopt;
}

[[nodiscard]] Tensor<float, 3> BrandubhGS::canonicalized() const noexcept {
  auto out = CanonicalTensor{};

  // Board planes.
  for (auto p = 0; p < 3; ++p) {
    for (auto h = 0; h < HEIGHT; ++h) {
      for (auto w = 0; w < WIDTH; ++w) {
        out(p, h, w) = board_(p, h, w);
      }
    }
  }

  // Current player planes.
  auto p = player_ + 3;
  auto op = (player_ + 1) % 2 + 3;
  for (auto h = 0; h < HEIGHT; ++h) {
    for (auto w = 0; w < WIDTH; ++w) {
      out(p, h, w) = 1;
      out(op, h, w) = 0;
    }
  }

  // Repetition count.
  if (current_repetition_count_ == 0) {
    for (auto h = 0; h < HEIGHT; ++h) {
      for (auto w = 0; w < WIDTH; ++w) {
        out(5, h, w) = 0;
        out(6, h, w) = 0;
      }
    }
  } else if (current_repetition_count_ == 1) {
    for (auto h = 0; h < HEIGHT; ++h) {
      for (auto w = 0; w < WIDTH; ++w) {
        out(5, h, w) = 1;
        out(6, h, w) = 0;
      }
    }
  } else if (current_repetition_count_ == 2) {
    for (auto h = 0; h < HEIGHT; ++h) {
      for (auto w = 0; w < WIDTH; ++w) {
        out(5, h, w) = 0;
        out(6, h, w) = 1;
      }
    }
  } else if (current_repetition_count_ > 2) {
    for (auto h = 0; h < HEIGHT; ++h) {
      for (auto w = 0; w < WIDTH; ++w) {
        out(5, h, w) = 1;
        out(6, h, w) = 1;
      }
    }
  }

  return out;
}

[[nodiscard]] std::vector<PlayHistory> BrandubhGS::symmetries(
    const PlayHistory& base) const noexcept {
  return tafl_helper::eightSym(base);
}

[[nodiscard]] std::string BrandubhGS::dump() const noexcept {
  auto out = "Current Player: " + std::to_string(player_) + '\n';
  out += "Current Turn: " + std::to_string(turn_) + " out of " +
         std::to_string(max_turns_) + '\n';
  out +=
      "Current Repetition Count: " + std::to_string(current_repetition_count_) +
      '\n';
  for (auto h = 0; h < HEIGHT; ++h) {
    for (auto w = 0; w < WIDTH; ++w) {
      if ((h == 0 && w == 0) || (h == HEIGHT - 1 && w == 0) ||
          (h == 0 && w == WIDTH - 1) || (h == HEIGHT - 1 && w == WIDTH - 1) ||
          (h == 3 && w == 3)) {
        out += color::Modifier{color::BG_RED}.dump();
      }
      if (board_(0, h, w) == 1) {
        out += '@';
      } else if (board_(1, h, w) == 1) {
        out += 'O';
      } else if (board_(2, h, w) == 1) {
        out += 'X';
      } else {
        out += '.';
      }
      if ((h == 0 && w == 0) || (h == HEIGHT - 1 && w == 0) ||
          (h == 0 && w == WIDTH - 1) || (h == HEIGHT - 1 && w == WIDTH - 1) ||
          (h == 3 && w == 3)) {
        out += color::Modifier{color::BG_DEFAULT}.dump();
      }
    }
    out += '\n';
  }
  out += '\n';
  return out;
}

void BrandubhGS::minimize_storage() {
  repetition_counts_ = {};
  board_intern_ = nullptr;
}

}  // namespace alphazero::brandubh_gs