#pragma once

#include <iostream>

#include "color.h"
#include "game_state.h"

namespace alphazero::photosynthesis_gs {

constexpr const int WIDTH = 7;
constexpr const int HEIGHT = 7;
constexpr const int NUM_MOVES = 1 + 4 + (WIDTH * HEIGHT + 1) * (WIDTH * HEIGHT);
const std::array<color::Modifier, 7> PLAYER_COLOR = {
    color::Modifier{color::FG_RED},     color::Modifier{color::FG_GREEN},
    color::Modifier{color::FG_BLUE},    color::Modifier{color::FG_YELLOW},
    color::Modifier{color::FG_MAGENTA}, color::Modifier{color::FG_CYAN},
    color::Modifier{color::FG_WHITE},
};
const color::Modifier DEFAULT_COLOR = color::Modifier{color::FG_DEFAULT};

bool on_board(int h, int w) {
  if (h < 0 || h >= HEIGHT || w < 0 || w >= WIDTH) {
    return false;
  }
  if ((h == 0 && w == 0) || (h == 1 && w == 0) || (h == 0 && w == 1) ||
      (h == 1 && w == 1) || (h == 2 && w == 0) || (h == 0 && w == 2) ||
      (h == 6 && w == 6) || (h == 5 && w == 6) || (h == 6 && w == 5) ||
      (h == 5 && w == 5) || (h == 4 && w == 6) || (h == 6 && w == 4)) {
    return false;
  }
  return true;
}

int dist_from_center(int h, int w) {
  return (std::abs(w - 3) + std::abs(w + h - 3 - 3) + std::abs(h - 3)) / 2;
}

template <uint8_t NUM_PLAYERS>
class PhotosynthesisGS : public GameState {
 public:
  static constexpr const std::array<int, 3> BOARD_SHAPE = {NUM_PLAYERS, HEIGHT,
                                                           WIDTH};
  static constexpr const std::array<int, 3> CANONICAL_SHAPE = {
      53 + 94 * NUM_PLAYERS, HEIGHT, WIDTH};
  static constexpr const std::array<uint8_t, 4> MAX_BUYABLE_PLANTS{4, 4, 3, 2};
  static constexpr const std::array<uint8_t, 4> MAX_AVAILABLE_PLANTS{6, 8, 4,
                                                                     2};
  static constexpr const std::array<std::array<uint8_t, 4>, 4> BUY_COSTS{{
      {2, 2, 2, 1},
      {3, 3, 2, 2},
      {4, 3, 3, static_cast<uint8_t>(-1)},
      {5, 4, static_cast<uint8_t>(-1), static_cast<uint8_t>(-1)},
  }};
  static constexpr const uint8_t MAX_SUN = 20;

  using BoardTensor =
      SizedTensor<int8_t,
                  Eigen::Sizes<BOARD_SHAPE[0], BOARD_SHAPE[1], BOARD_SHAPE[2]>>;
  using CanonicalTensor =
      SizedTensor<float, Eigen::Sizes<CANONICAL_SHAPE[0], CANONICAL_SHAPE[1],
                                      CANONICAL_SHAPE[2]>>;

  PhotosynthesisGS() {
    sun_points_.setZero();
    score_.setZero();
    score_tiles_collected_.setZero();
    board_.setZero();
    activated_tiles_.setZero();
    buyable_plants_.setZero();
    available_plants_.setZero();
    for (auto i = 0UL; i < NUM_PLAYERS; ++i) {
      buyable_plants_(i, 0) = MAX_BUYABLE_PLANTS[0];
      buyable_plants_(i, 1) = MAX_BUYABLE_PLANTS[1];
      buyable_plants_(i, 2) = MAX_BUYABLE_PLANTS[2];
      buyable_plants_(i, 3) = MAX_BUYABLE_PLANTS[3];
      available_plants_(i, 0) = 2;
      available_plants_(i, 1) = 2;
      available_plants_(i, 2) = 1;
    }
  }
  PhotosynthesisGS(BoardTensor board,
                   SizedMatrix<uint8_t, HEIGHT, WIDTH> activated_tiles,
                   uint8_t first_player, uint8_t player, uint32_t turn,
                   uint8_t sun_phase,
                   SizedVector<uint8_t, NUM_PLAYERS> sun_points,
                   SizedMatrix<uint8_t, NUM_PLAYERS, 4> buyable_plants,
                   SizedMatrix<uint8_t, NUM_PLAYERS, 4> available_plants,
                   SizedMatrix<uint8_t, NUM_PLAYERS, 4> score_tiles_collected,
                   SizedVector<uint16_t, NUM_PLAYERS> score,
                   std::array<std::vector<uint8_t>, 4> score_tiles)
      : board_(board),
        activated_tiles_(activated_tiles),
        first_player_(first_player),
        player_(player),
        turn_(turn),
        sun_phase_(sun_phase),
        sun_points_(sun_points),
        buyable_plants_(buyable_plants),
        available_plants_(available_plants),
        score_tiles_collected_(score_tiles_collected),
        score_(score) {
    for (auto i = 0; i < 4; ++i) {
      score_tiles_[i].clear();
      for (auto points : score_tiles[i]) {
        score_tiles_[i].push_back(points);
      }
    }
  }
  [[nodiscard]] std::unique_ptr<GameState> copy() const noexcept override {
    return std::make_unique<PhotosynthesisGS<NUM_PLAYERS>>(
        board_, activated_tiles_, first_player_, player_, turn_, sun_phase_,
        sun_points_, buyable_plants_, available_plants_, score_tiles_collected_,
        score_, score_tiles_);
  }

  [[nodiscard]] bool operator==(
      const GameState& other) const noexcept override {
    const auto* other_ps =
        dynamic_cast<const PhotosynthesisGS<NUM_PLAYERS>*>(&other);
    if (other_ps == nullptr) {
      return false;
    }
    for (auto h = 0; h < HEIGHT; ++h) {
      for (auto w = 0; w < WIDTH; ++w) {
        if (other_ps->activated_tiles_(h, w) != activated_tiles_(h, w)) {
          return false;
        }
      }
    }
    for (auto p = 0; p < NUM_PLAYERS; ++p) {
      if (other_ps->sun_points_(p) != sun_points_(p)) {
        return false;
      }
      for (auto i = 0; i < 4; ++i) {
        if (other_ps->buyable_plants_(p, i) != buyable_plants_(p, i) ||
            other_ps->available_plants_(p, i) != available_plants_(p, i) ||
            other_ps->score_tiles_collected_(p, i) !=
                score_tiles_collected_(p, i)) {
          return false;
        }
      }
      for (auto h = 0; h < HEIGHT; ++h) {
        for (auto w = 0; w < WIDTH; ++w) {
          if (other_ps->board_(p, h, w) != board_(p, h, w)) {
            return false;
          }
        }
      }
    }
    return other_ps->first_player_ == first_player_ &&
           other_ps->player_ == player_ && other_ps->sun_phase_ == sun_phase_ &&
           other_ps->score_tiles_ == score_tiles_;
  }

  void hash(absl::HashState h) const override {
    h = absl::HashState::combine_contiguous(std::move(h), board_.data(),
                                            board_.size());
    h = absl::HashState::combine_contiguous(
        std::move(h), activated_tiles_.data(), activated_tiles_.size());
    h = absl::HashState::combine_contiguous(std::move(h), sun_points_.data(),
                                            sun_points_.size());
    h = absl::HashState::combine_contiguous(
        std::move(h), buyable_plants_.data(), buyable_plants_.size());
    h = absl::HashState::combine_contiguous(
        std::move(h), available_plants_.data(), available_plants_.size());
    h = absl::HashState::combine_contiguous(std::move(h),
                                            score_tiles_collected_.data(),
                                            score_tiles_collected_.size());
    absl::HashState::combine(std::move(h), first_player_, player_, sun_phase_,
                             score_tiles_);
  }

  // Returns the current player. Players must be 0 indexed.
  [[nodiscard]] uint8_t current_player() const noexcept override {
    return player_;
  }

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
  }

  // Returns a bool for all moves. The value is true if the move is playable
  // from this GameState. All values should be 0 or 1.
  [[nodiscard]] Vector<uint8_t> valid_moves() const noexcept override {
    auto moves = SizedVector<uint8_t, NUM_MOVES>{};
    moves.setZero();

    // Special case where we are setting up the board.
    // Only valid moves are to place a tree on the edge that isn't occupied.
    if (turn_ < 2 * NUM_PLAYERS) {
      for (auto h = 0; h < HEIGHT; ++h) {
        for (auto w = 0; w < WIDTH; ++w) {
          if (dist_from_center(h, w) == 3) {
            auto empty = true;
            for (auto p = 0; p < NUM_PLAYERS; ++p) {
              if (board_(p, h, w) != 0) {
                empty = false;
                break;
              }
            }
            if (empty) {
              moves(h * WIDTH + w) = 1;
            }
          }
        }
      }
      return moves;
    }

    // board size of growable trees on specified tile.
    auto offset = 0;
    if (sun_points_(player_) > 0) {
      for (auto h = 0; h < HEIGHT; ++h) {
        for (auto w = 0; w < WIDTH; ++w) {
          if (activated_tiles_(h, w) == 0 &&
              (board_(player_, h, w) == 4 ||
               (board_(player_, h, w) > 0 &&
                available_plants_(player_, board_(player_, h, w)) > 0)) &&
              sun_points_(player_) >= board_(player_, h, w)) {
            moves(offset + h * WIDTH + w) = 1;
          }
        }
      }
    }
    offset += WIDTH * HEIGHT;

    // board size squared of plant seed on tile from tile.
    if (available_plants_(player_, 0) > 0 && sun_points_(player_) >= 1) {
      for (auto h = 0; h < HEIGHT; ++h) {
        for (auto w = 0; w < WIDTH; ++w) {
          if (activated_tiles_(h, w) == 0 && board_(player_, h, w) > 1) {
            int seed_range = board_(player_, h, w) - 1;
            const auto current_x = w;
            const auto current_z = h;
            const auto current_y = -current_x - current_z;
            for (auto offset_x = -seed_range; offset_x <= seed_range;
                 ++offset_x) {
              for (auto offset_y =
                       std::max(-seed_range, -offset_x - seed_range);
                   offset_y <= std::min(seed_range, -offset_x + seed_range);
                   ++offset_y) {
                const auto near_z =
                    -(offset_x + current_x) - (offset_y + current_y);
                const auto near_h = near_z;
                const auto near_w = offset_x + current_x;
                if (on_board(near_h, near_w) &&
                    activated_tiles_(near_h, near_w) == 0) {
                  auto empty = true;
                  for (auto p = 0; p < NUM_PLAYERS; ++p) {
                    if (board_(p, near_h, near_w) != 0) {
                      empty = false;
                      break;
                    }
                  }
                  if (empty) {
                    moves(offset + near_h * WIDTH + near_w) = 1;
                  }
                }
              }
            }
          }
          offset += WIDTH * HEIGHT;
        }
      }
    } else {
      offset += (WIDTH * HEIGHT) * (WIDTH * HEIGHT);
    }

    // 4 x buy item.
    for (auto i = 0; i < 4; ++i) {
      if (buyable_plants_(player_, i) > 0 &&
          sun_points_(player_) >=
              BUY_COSTS[i][buyable_plants_(player_, i) - 1]) {
        moves(offset + i) = 1;
      }
    }
    offset += 4;

    // Pass is always valid.
    moves(offset) = 1;

    return moves;
  }

  // Plays a move, modifying the current GameState.
  void play_move(uint32_t move) override {
    // Special case where we are setting up the board.
    // Only valid moves are to place a tree on the edge that isn't occupied.
    if (turn_ < 2 * NUM_PLAYERS) {
      auto h = move / WIDTH;
      auto w = move % WIDTH;
      board_(player_, h, w) = 2;
      player_ = (player_ + 1) % NUM_PLAYERS;
      ++turn_;
      if (turn_ == 2 * NUM_PLAYERS) {
        update_sun_points();
      }
      return;
    }

    ++turn_;
    if (move == NUM_MOVES - 1) {
      // Pass.
    } else if (move >= NUM_MOVES - 5) {
      // Buy something.
      auto bought_item = move - (NUM_MOVES - 5);
      sun_points_(player_) -=
          BUY_COSTS[bought_item][buyable_plants_(player_, bought_item) - 1];
      --buyable_plants_(player_, bought_item);
      ++available_plants_(player_, bought_item);
    } else if (move < WIDTH * HEIGHT) {
      // Grow a plant.
      auto h = move / WIDTH;
      auto w = move % WIDTH;
      sun_points_(player_) -= board_(player_, h, w);
      // special case, max size tree getting cut.
      if (board_(player_, h, w) == 4) {
        auto score_level = 3 - dist_from_center(h, w);
        while (score_level >= 0) {
          if (score_tiles_[score_level].size() == 0) {
            --score_level;
          } else {
            auto points = score_tiles_[score_level].back();
            score_tiles_[score_level].pop_back();
            ++score_tiles_collected_(player_, score_level);
            score_(player_) += points;
            break;
          }
        }
        ++buyable_plants_(player_, board_(player_, h, w) - 1);
        buyable_plants_(player_, board_(player_, h, w) - 1) =
            std::min(buyable_plants_(player_, board_(player_, h, w) - 1),
                     MAX_BUYABLE_PLANTS[board_(player_, h, w) - 1]);
        board_(player_, h, w) = 0;
        activated_tiles_(h, w) = 1;
      } else {
        --available_plants_(player_, board_(player_, h, w));
        ++buyable_plants_(player_, board_(player_, h, w) - 1);
        buyable_plants_(player_, board_(player_, h, w) - 1) =
            std::min(buyable_plants_(player_, board_(player_, h, w) - 1),
                     MAX_BUYABLE_PLANTS[board_(player_, h, w) - 1]);
        ++board_(player_, h, w);
        activated_tiles_(h, w) = 1;
      }
    } else {
      // Plant a seed.
      auto from_move = (move - WIDTH * HEIGHT) / (WIDTH * HEIGHT);
      auto to_move = (move - WIDTH * HEIGHT) % (WIDTH * HEIGHT);
      auto from_h = from_move / WIDTH;
      auto from_w = from_move % WIDTH;
      auto to_h = to_move / WIDTH;
      auto to_w = to_move % WIDTH;
      --sun_points_(player_);
      --available_plants_(player_, 0);
      ++board_(player_, to_h, to_w);
      activated_tiles_(from_h, from_w) = 1;
      activated_tiles_(to_h, to_w) = 1;
    }

    // Check if the player's turn is over, if so move on.
    // If the full set of turns have been played, rotate the sun and update
    // points.
    if (move == NUM_MOVES - 1 || only_pass_valid()) {
      activated_tiles_.setZero();
      if ((player_ + 1) % NUM_PLAYERS == first_player_) {
        // Full sun rotation.
        ++sun_phase_;
        rotate_board();
        update_sun_points();
        first_player_ = (first_player_ + 1) % NUM_PLAYERS;
        player_ = first_player_;
      } else {
        // Just next player turn.
        player_ = (player_ + 1) % NUM_PLAYERS;
      }
    }
  }

  // Returns the canonicalized form of the board, ready for feeding to a NN.
  [[nodiscard]] Tensor<float, 3> canonicalized() const noexcept override {
    constexpr auto SUN_PHASE_COUNT = 18;
    constexpr std::array<int, 4> SCORE_TILE_COUNTS{9, 7, 5, 3};
    constexpr auto PIECE_TYPES = 4;

    auto t = CanonicalTensor{};
    t.setZero();

    auto offset = 0;
    // Current player.
    for (auto h = 0; h < HEIGHT; ++h) {
      for (auto w = 0; w < WIDTH; ++w) {
        t(offset + player_, h, w) = 1;
      }
    }
    offset += NUM_PLAYERS;

    // Current round first player.
    for (auto h = 0; h < HEIGHT; ++h) {
      for (auto w = 0; w < WIDTH; ++w) {
        t(offset + first_player_, h, w) = 1;
      }
    }
    offset += NUM_PLAYERS;

    // Current sun phase.
    for (auto h = 0; h < HEIGHT; ++h) {
      for (auto w = 0; w < WIDTH; ++w) {
        t(offset + sun_phase_, h, w) = 1;
      }
    }
    offset += SUN_PHASE_COUNT;

    // Activated tiles.
    for (auto h = 0; h < HEIGHT; ++h) {
      for (auto w = 0; w < WIDTH; ++w) {
        t(offset, h, w) = activated_tiles_(h, w);
      }
    }
    ++offset;

    // Remaining score tiles.
    for (auto i = 0U; i < SCORE_TILE_COUNTS.size(); ++i) {
      for (auto h = 0; h < HEIGHT; ++h) {
        for (auto w = 0; w < WIDTH; ++w) {
          t(offset + score_tiles_[i].size(), h, w) = 1;
        }
      }
      // + 1 can also be zero.
      offset += SCORE_TILE_COUNTS[i] + 1;
    }

    // Player sun score.
    for (auto p = 0; p < NUM_PLAYERS; ++p) {
      for (auto h = 0; h < HEIGHT; ++h) {
        for (auto w = 0; w < WIDTH; ++w) {
          t(offset + sun_points_[p], h, w) = 1;
        }
      }
      // + 1 can also be zero.
      offset += MAX_SUN + 1;
    }

    // Player pieces.
    for (auto p = 0; p < NUM_PLAYERS; ++p) {
      for (auto h = 0; h < HEIGHT; ++h) {
        for (auto w = 0; w < WIDTH; ++w) {
          if (board_(p, h, w) > 0) {
            t(offset + board_(p, h, w) - 1, h, w) = 1;
          }
        }
      }
      offset += PIECE_TYPES;
    }

    // Player available plants.
    for (auto i = 0U; i < MAX_AVAILABLE_PLANTS.size(); ++i) {
      for (auto p = 0; p < NUM_PLAYERS; ++p) {
        for (auto h = 0; h < HEIGHT; ++h) {
          for (auto w = 0; w < WIDTH; ++w) {
            t(offset + available_plants_(p, i), h, w) = 1;
          }
        }
        // + 1 can also be zero.
        offset += MAX_AVAILABLE_PLANTS[i] + 1;
      }
    }

    // Player buyable plants.
    for (auto i = 0U; i < MAX_BUYABLE_PLANTS.size(); ++i) {
      for (auto p = 0; p < NUM_PLAYERS; ++p) {
        for (auto h = 0; h < HEIGHT; ++h) {
          for (auto w = 0; w < WIDTH; ++w) {
            t(offset + buyable_plants_(p, i), h, w) = 1;
          }
        }
        // + 1 can also be zero.
        offset += MAX_BUYABLE_PLANTS[i] + 1;
      }
    }

    // player score tiles.
    for (auto i = 0U; i < SCORE_TILE_COUNTS.size(); ++i) {
      for (auto p = 0; p < NUM_PLAYERS; ++p) {
        for (auto h = 0; h < HEIGHT; ++h) {
          for (auto w = 0; w < WIDTH; ++w) {
            t(offset + score_tiles_collected_(p, i), h, w) = 1;
          }
        }
        // + 1 can also be zero.
        offset += SCORE_TILE_COUNTS[i] + 1;
      }
    }
    // Use this to figure out the final size and then set constants with it.
    // std::cout << "Offset: " << offset << '\n';

    // This shouldn't crash if everything is set correctly.
    assert(offset == t.dimension(0));

    return t;
  }

  // Returns nullopt if the game isn't over.
  // Returns a one hot encode result of the game.
  // The first num player positions are set to 1 if that player won and 0
  // otherwise. The last position is set to 1 if the game was a draw and 0
  // otherwise.
  [[nodiscard]] std::optional<Vector<float>> scores() const noexcept override {
    if (sun_phase_ < 18) {
      return std::nullopt;
    }
    auto winners = std::vector<uint8_t>{0};
    auto best_score = score_(0) + sun_points_(0) / 3;
    for (auto p = 1; p < NUM_PLAYERS; ++p) {
      auto score = score_(p) + sun_points_(p) / 3;
      if (score > best_score) {
        best_score = score;
        winners.clear();
        winners.push_back(p);
      } else if (score == best_score) {
        winners.push_back(p);
      }
    }
    auto count_plants = [&](uint8_t p) {
      auto count = 0;
      for (auto h = 0; h < HEIGHT; ++h) {
        for (auto w = 0; w < WIDTH; ++w) {
          if (board_(p, h, w) != 0) {
            ++count;
          }
        }
      }
      return count;
    };
    if (winners.size() > 1) {
      best_score = count_plants(winners[0]);
      auto tiebreak_winners = std::vector<uint8_t>{winners[0]};
      for (auto i = 1U; i < winners.size(); ++i) {
        auto score = count_plants(winners[i]);
        if (score > best_score) {
          best_score = score;
          tiebreak_winners.clear();
          tiebreak_winners.push_back(winners[i]);
        } else if (score == best_score) {
          tiebreak_winners.push_back(winners[i]);
        }
      }
      winners = tiebreak_winners;
    }
    auto result = SizedVector<float, NUM_PLAYERS + 1>{};
    result.setZero();
    for (auto w : winners) {
      result(w) = 1.0 / winners.size();
    }
    return result;
  }

  // Returns a string representation of the game state.
  [[nodiscard]] std::string dump() const noexcept override {
    auto out = "Sun Phase: " + std::to_string(sun_phase_) + '\n';
    out += "Current Player: " + std::to_string(player_) + '\n';
    out += "Sun Points: " + std::to_string(sun_points_(player_)) + '\n';
    out += "Buyable Plants: ";
    for (auto i = 0; i < 4; ++i) {
      out += std::to_string(buyable_plants_(player_, i)) + ' ';
    }
    out += '\n';
    out += "Available Plants: ";
    for (auto i = 0; i < 4; ++i) {
      out += std::to_string(available_plants_(player_, i)) + ' ';
    }
    out += '\n';
    out += "Scores: ";
    for (auto p = 0; p < NUM_PLAYERS; ++p) {
      out += PLAYER_COLOR[p].dump();
      out += std::to_string(score_(p));
      out += DEFAULT_COLOR.dump() + ' ';
    }
    out += '\n';
    for (auto h = 0; h < HEIGHT; ++h) {
      for (auto w = 0; w < WIDTH; ++w) {
        auto init = false;
        for (auto p = 0; p < NUM_PLAYERS; ++p) {
          if (board_(p, h, w) != 0) {
            init = true;
            out += PLAYER_COLOR[p].dump();
            out += std::to_string(board_(p, h, w));
            out += DEFAULT_COLOR.dump();
            break;
          }
        }
        if (!init) {
          if ((h <= 1 && w <= 1) || (h >= 5 && w >= 5) || (h == 0 && w == 2) ||
              (h == 2 && w == 0) || (h == 6 && w == 4) || (h == 4 && w == 6)) {
            out += ' ';
          } else {
            out += '.';
          }
        }
      }
      out += '\n';
    }
    out += '\n';
    return out;
  }

 private:
  void update_sun_points() {
    for (auto h = 0; h < HEIGHT; ++h) {
      auto last_large_tree = -10;
      auto last_med_tree = -10;
      auto last_small_tree = -10;
      for (auto w = 0; w < WIDTH; ++w) {
        for (auto p = 0; p < NUM_PLAYERS; ++p) {
          if (board_(p, h, w) == 2) {
            if (std::abs(last_large_tree - w) > 3 &&
                std::abs(last_med_tree - w) > 2 &&
                std::abs(last_small_tree - w) > 1) {
              ++sun_points_(p);
            }
            last_small_tree = w;
          } else if (board_(p, h, w) == 3) {
            if (std::abs(last_large_tree - w) > 3 &&
                std::abs(last_med_tree - w) > 2) {
              sun_points_(p) += 2;
            }
            last_med_tree = w;
          } else if (board_(p, h, w) == 4) {
            if (std::abs(last_large_tree - w) > 3) {
              sun_points_(p) += 3;
            }
            last_large_tree = w;
          }
        }
      }
    }
    for (auto p = 0; p < NUM_PLAYERS; ++p) {
      sun_points_(p) = std::min(sun_points_(p), MAX_SUN);
    }
  }

  void rotate_board() {
    for (auto p = 0; p < NUM_PLAYERS; ++p) {
      auto tmp = board_(p, 0, 3);
      board_(p, 0, 3) = board_(p, 0, 6);
      board_(p, 0, 6) = board_(p, 3, 6);
      board_(p, 3, 6) = board_(p, 6, 3);
      board_(p, 6, 3) = board_(p, 6, 0);
      board_(p, 6, 0) = board_(p, 3, 0);
      board_(p, 3, 0) = tmp;

      tmp = board_(p, 0, 4);
      board_(p, 0, 4) = board_(p, 1, 6);
      board_(p, 1, 6) = board_(p, 4, 5);
      board_(p, 4, 5) = board_(p, 6, 2);
      board_(p, 6, 2) = board_(p, 5, 0);
      board_(p, 5, 0) = board_(p, 2, 1);
      board_(p, 2, 1) = tmp;

      tmp = board_(p, 0, 5);
      board_(p, 0, 5) = board_(p, 2, 6);
      board_(p, 2, 6) = board_(p, 5, 4);
      board_(p, 5, 4) = board_(p, 6, 1);
      board_(p, 6, 1) = board_(p, 4, 0);
      board_(p, 4, 0) = board_(p, 1, 2);
      board_(p, 1, 2) = tmp;

      tmp = board_(p, 1, 3);
      board_(p, 1, 3) = board_(p, 1, 5);
      board_(p, 1, 5) = board_(p, 5, 3);
      board_(p, 5, 3) = board_(p, 3, 5);
      board_(p, 3, 5) = board_(p, 5, 1);
      board_(p, 5, 1) = board_(p, 3, 1);
      board_(p, 3, 1) = tmp;

      tmp = board_(p, 1, 4);
      board_(p, 1, 4) = board_(p, 2, 5);
      board_(p, 2, 5) = board_(p, 4, 4);
      board_(p, 4, 4) = board_(p, 5, 2);
      board_(p, 5, 2) = board_(p, 4, 1);
      board_(p, 4, 1) = board_(p, 2, 2);
      board_(p, 2, 2) = tmp;

      tmp = board_(p, 2, 3);
      board_(p, 2, 3) = board_(p, 2, 4);
      board_(p, 2, 4) = board_(p, 3, 4);
      board_(p, 3, 4) = board_(p, 4, 3);
      board_(p, 4, 3) = board_(p, 4, 2);
      board_(p, 4, 2) = board_(p, 3, 2);
      board_(p, 3, 2) = tmp;
    }
  }

  [[nodiscard]] bool only_pass_valid() const noexcept {
    // Special case where we are setting up the board.
    // Only valid moves are to place a tree on the edge that isn't occupied.
    if (turn_ < 2 * NUM_PLAYERS) {
      return false;
    }

    // board size of growable trees on specified tile.
    if (sun_points_(player_) > 0) {
      for (auto h = 0; h < HEIGHT; ++h) {
        for (auto w = 0; w < WIDTH; ++w) {
          if (activated_tiles_(h, w) == 0 &&
              (board_(player_, h, w) == 4 ||
               (board_(player_, h, w) > 0 &&
                available_plants_(player_, board_(player_, h, w)) > 0)) &&
              sun_points_(player_) >= board_(player_, h, w)) {
            return false;
          }
        }
      }
    }

    // board size squared of plant seed on tile from tile.
    if (available_plants_(player_, 0) > 0 && sun_points_(player_) >= 1) {
      for (auto h = 0; h < HEIGHT; ++h) {
        for (auto w = 0; w < WIDTH; ++w) {
          if (activated_tiles_(h, w) == 0 && board_(player_, h, w) > 1) {
            int seed_range = board_(player_, h, w) - 1;
            const auto current_x = w;
            const auto current_z = h;
            const auto current_y = -current_x - current_z;
            for (auto offset_x = -seed_range; offset_x <= seed_range;
                 ++offset_x) {
              for (auto offset_y =
                       std::max(-seed_range, -offset_x - seed_range);
                   offset_y <= std::min(seed_range, -offset_x + seed_range);
                   ++offset_y) {
                const auto near_z =
                    -(offset_x + current_x) - (offset_y + current_y);
                const auto near_h = near_z;
                const auto near_w = offset_x + current_x;
                if (on_board(near_h, near_w) &&
                    activated_tiles_(near_h, near_w) == 0) {
                  auto empty = true;
                  for (auto p = 0; p < NUM_PLAYERS; ++p) {
                    if (board_(p, near_h, near_w) != 0) {
                      empty = false;
                      break;
                    }
                  }
                  if (empty) {
                    return false;
                  }
                }
              }
            }
          }
        }
      }
    }

    // 4 x buy item.
    for (auto i = 0; i < 4; ++i) {
      if (buyable_plants_(player_, i) > 0 &&
          sun_points_(player_) >=
              BUY_COSTS[i][buyable_plants_(player_, i) - 1]) {
        return false;
      }
    }

    return true;
  }

  // Board contains a layer for each player.
  // A 0 means no piece, 1 through 4 correspond to seed through full tree.
  BoardTensor board_{};
  SizedMatrix<uint8_t, HEIGHT, WIDTH> activated_tiles_{};
  uint8_t first_player_{0};
  uint8_t player_{0};
  uint32_t turn_{0};
  uint8_t sun_phase_{0};
  SizedVector<uint8_t, NUM_PLAYERS> sun_points_{};
  SizedMatrix<uint8_t, NUM_PLAYERS, 4> buyable_plants_{};
  SizedMatrix<uint8_t, NUM_PLAYERS, 4> available_plants_{};
  SizedMatrix<uint8_t, NUM_PLAYERS, 4> score_tiles_collected_{};
  SizedVector<uint16_t, NUM_PLAYERS> score_{};
  std::array<std::vector<uint8_t>, 4> score_tiles_{
      {{12, 12, 12, 12, 13, 13, 13, 14, 14},
       {13, 13, 14, 14, 16, 16, 17},
       {17, 17, 18, 18, 19},
       {20, 21, 22}}};
};

}  // namespace alphazero::photosynthesis_gs