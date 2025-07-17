#include "tak_gs.h"

#include <algorithm>
#include <sstream>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <regex>

namespace alphazero::tak_gs {

template<int SIZE>
TakGS<SIZE>::TakGS(float komi, const std::string& tps_string, bool opening_swap) 
    : board_(SIZE * SIZE),
      player_(0),
      turn_(0),
      p0_stones_(get_board_size_pieces(SIZE)),
      p0_caps_(get_board_size_capstones(SIZE)),
      p1_stones_(get_board_size_pieces(SIZE)),
      p1_caps_(get_board_size_capstones(SIZE)),
      opening_swap_(opening_swap),
      komi_(komi),
      moves_without_placement_(0),
      road_connectivity_cache_(SIZE * SIZE, std::vector<bool>(SIZE * SIZE, false)),
      road_cache_valid_(false),
      compact_board_cache_(SIZE * SIZE, 0),
      compact_board_valid_(false) {
  static_assert(SIZE >= MIN_SIZE && SIZE <= MAX_SIZE);
  
  // If TPS string is provided, parse it
  if (!tps_string.empty()) {
    parse_tps_string(tps_string);
  }
}

template<int SIZE>
TakGS<SIZE>::TakGS(std::vector<Square> board, uint8_t player, uint32_t turn,
             int p0_stones, int p0_caps, int p1_stones, int p1_caps, bool opening_swap,
             float komi, uint32_t moves_without_placement)
    : board_(std::move(board)),
      player_(player),
      turn_(turn),
      p0_stones_(p0_stones),
      p0_caps_(p0_caps),
      p1_stones_(p1_stones),
      p1_caps_(p1_caps),
      opening_swap_(opening_swap),
      komi_(komi),
      moves_without_placement_(moves_without_placement),
      road_connectivity_cache_(SIZE * SIZE, std::vector<bool>(SIZE * SIZE, false)),
      road_cache_valid_(false),
      compact_board_cache_(SIZE * SIZE, 0),
      compact_board_valid_(false) {
  static_assert(SIZE >= MIN_SIZE && SIZE <= MAX_SIZE);
  assert(board_.size() == SIZE * SIZE);
}


template<int SIZE>
std::unique_ptr<GameState> TakGS<SIZE>::copy() const noexcept {
  return std::make_unique<TakGS<SIZE>>(board_, player_, turn_,
                                       p0_stones_, p0_caps_, p1_stones_, p1_caps_, 
                                       opening_swap_, komi_, moves_without_placement_);
}

template<int SIZE>
bool TakGS<SIZE>::operator==(const GameState& other) const noexcept {
  const auto* other_tak = dynamic_cast<const TakGS<SIZE>*>(&other);
  if (!other_tak) return false;
  
  return board_ == other_tak->board_ &&
         player_ == other_tak->player_ &&
         turn_ == other_tak->turn_ &&
         p0_stones_ == other_tak->p0_stones_ &&
         p0_caps_ == other_tak->p0_caps_ &&
         p1_stones_ == other_tak->p1_stones_ &&
         p1_caps_ == other_tak->p1_caps_ &&
         opening_swap_ == other_tak->opening_swap_ &&
         komi_ == other_tak->komi_ &&
         moves_without_placement_ == other_tak->moves_without_placement_;
}

template<int SIZE>
void TakGS<SIZE>::hash(absl::HashState h) const {
  absl::HashState::combine(std::move(h), SIZE, player_, turn_, 
                           p0_stones_, p0_caps_, p1_stones_, p1_caps_, 
                           opening_swap_, komi_, moves_without_placement_);
}

template<int SIZE>
Vector<uint8_t> TakGS<SIZE>::valid_moves() const noexcept {
  Vector<uint8_t> valid(num_moves());
  valid.setZero();
  
  if (opening_swap_ && turn_ == 0) {
    for (int i = 0; i < SIZE * SIZE; ++i) {
      if (board_[i].empty()) {
        valid[encode_placement(i, PieceType::FLAT)] = 1;
      }
    }
    return valid;
  }
  
  int my_stones = (player_ == 0) ? p0_stones_ : p1_stones_;
  int my_caps = (player_ == 0) ? p0_caps_ : p1_caps_;
  
  for (int i = 0; i < SIZE * SIZE; ++i) {
    if (board_[i].empty()) {
      if (my_stones > 0) {
        valid[encode_placement(i, PieceType::FLAT)] = 1;
        if (turn_ >= 1 || !opening_swap_) {
          valid[encode_placement(i, PieceType::WALL)] = 1;
        }
      }
      if (my_caps > 0 && (turn_ >= 1 || !opening_swap_)) {
        valid[encode_placement(i, PieceType::CAP)] = 1;
      }
    }
  }
  
  for (int from_idx = 0; from_idx < SIZE * SIZE; ++from_idx) {
    const Square& sq = board_[from_idx];
    if (sq.empty() || sq.top().owner != player_) continue;
    
    int max_carry = std::min(static_cast<int>(sq.height()), get_carry_limit(SIZE));
    
    auto [row, col] = index_to_square(from_idx);
    const int dirs[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    
    for (int dir = 0; dir < 4; ++dir) {
      int dr = dirs[dir][0];
      int dc = dirs[dir][1];
      
      for (int carry = 1; carry <= max_carry; ++carry) {
        Piece moving_piece = (carry == static_cast<int>(sq.height()) && 
                             sq.stack[0].type != PieceType::FLAT) ? 
                             sq.stack[0] : Piece(player_, PieceType::FLAT);
        
        for (int dist = 1; dist <= SIZE; ++dist) {
          int new_row = row + dr * dist;
          int new_col = col + dc * dist;
          
          if (!is_valid_square(new_row, new_col)) break;
          
          int to_idx = square_to_index(new_row, new_col);
          if (!can_move_onto(board_[to_idx], moving_piece)) break;
          
          auto drop_patterns = get_valid_drop_patterns(carry, dist);
          for (const auto& drops : drop_patterns) {
            valid[encode_movement(from_idx, to_idx, drops)] = 1;
          }
          
          if (!board_[to_idx].empty() && 
              board_[to_idx].top().type == PieceType::WALL &&
              moving_piece.type == PieceType::CAP) {
            break;
          }
          
          if (!board_[to_idx].empty() && 
              board_[to_idx].top().type != PieceType::FLAT) {
            break;
          }
        }
      }
    }
  }
  
  return valid;
}

template<int SIZE>
void TakGS<SIZE>::play_move(uint32_t move) {
  int from_idx, to_idx;
  PieceType place_type;
  int carry_count;
  std::vector<int> drops;
  
  invalidate_road_cache();
  
  decode_move(move, from_idx, to_idx, place_type, carry_count, drops);
  
  if (from_idx == -1) {
    Piece piece((opening_swap_ && turn_ == 0) ? 1 - player_ : player_, place_type);
    board_[to_idx].add(piece);
    
    moves_without_placement_ = 0;
    
    if (!(opening_swap_ && turn_ == 0)) {
      if (place_type == PieceType::CAP) {
        if (player_ == 0) p0_caps_--;
        else p1_caps_--;
      } else {
        if (player_ == 0) p0_stones_--;
        else p1_stones_--;
      }
    }
  } else {
    std::vector<Piece> carry_stack;
    for (int i = 0; i < carry_count; ++i) {
      carry_stack.push_back(board_[from_idx].remove());
    }
    std::reverse(carry_stack.begin(), carry_stack.end());
    
    auto [from_row, from_col] = index_to_square(from_idx);
    auto [to_row, to_col] = index_to_square(to_idx);
    
    int dr = (to_row > from_row) ? 1 : (to_row < from_row) ? -1 : 0;
    int dc = (to_col > from_col) ? 1 : (to_col < from_col) ? -1 : 0;
    
    int curr_row = from_row;
    int curr_col = from_col;
    size_t drop_idx = 0;
    
    while (drop_idx < drops.size()) {
      curr_row += dr;
      curr_col += dc;
      int curr_idx = square_to_index(curr_row, curr_col);
      
      if (curr_idx == to_idx && 
          !board_[curr_idx].empty() && 
          board_[curr_idx].top().type == PieceType::WALL &&
          carry_stack[0].type == PieceType::CAP) {
        board_[curr_idx].stack.back().type = PieceType::FLAT;
        moves_without_placement_ = 0;
      }
      
      int drop_count = drops[drop_idx++];
      for (int i = 0; i < drop_count; ++i) {
        board_[curr_idx].add(carry_stack.back());
        carry_stack.pop_back();
      }
    }
    
    if (drop_idx > 0) {
      moves_without_placement_ = 0;
    }
  }
  
  if (opening_swap_ && turn_ == 0) {
    opening_swap_ = false;
  } else {
    player_ = 1 - player_;
  }
  turn_++;
  
  if (from_idx != -1) {
    moves_without_placement_++;
  }
}

template<int SIZE>
std::optional<Vector<float>> TakGS<SIZE>::scores() const noexcept {
  Vector<float> result(3);
  result.setZero();
  
  bool p0_road = check_road_win(0);
  bool p1_road = check_road_win(1);
  
  if (p0_road && !p1_road) {
    result[0] = 1.0f;
    return result;
  }
  if (p1_road && !p0_road) {
    result[1] = 1.0f;
    return result;
  }
  if (p0_road && p1_road) {
    result[player_] = 1.0f;
    return result;
  }
  
  bool game_ended = is_board_full() || 
                   (p0_stones_ == 0 && p0_caps_ == 0 && 
                    p1_stones_ == 0 && p1_caps_ == 0) ||
                   moves_without_placement_ >= 50;
  
  if (game_ended) {
    int p0_flats = count_flats(0);
    int p1_flats = count_flats(1);
    
    float p0_effective_flats = static_cast<float>(p0_flats) + komi_;
    float p1_effective_flats = static_cast<float>(p1_flats);
    
    if (p0_effective_flats > p1_effective_flats) {
      result[0] = 1.0f;
    } else if (p1_effective_flats > p0_effective_flats) {
      result[1] = 1.0f;
    } else {
      result[2] = 1.0f;
    }
    return result;
  }
  
  return std::nullopt;
}

template<int SIZE>
Tensor<float, 3> TakGS<SIZE>::canonicalized() const noexcept {
  auto shape = canonical_shape();
  Tensor<float, 3> canonical(shape[0], shape[1], shape[2]);
  canonical.setZero();
  
  for (int row = 0; row < SIZE; ++row) {
    for (int col = 0; col < SIZE; ++col) {
      int idx = square_to_index(row, col);
      const Square& sq = board_[idx];
      
      if (!sq.empty()) {
        const Piece& top = sq.top();
        int layer = top.owner * 3 + static_cast<int>(top.type);
        canonical(layer, row, col) = 1.0f;
        
        int height = static_cast<int>(sq.height());
        
        if (height >= 1 && height <= NN_MAX_HEIGHT) {
          canonical(6 + height - 1, row, col) = 1.0f;
        } else if (height > NN_MAX_HEIGHT) {
          canonical(16, row, col) = 1.0f;
          float normalized_height = static_cast<float>(height - NN_MAX_HEIGHT) / 
                                   static_cast<float>(MAX_HEIGHT - NN_MAX_HEIGHT);
          canonical(17, row, col) = std::min(1.0f, normalized_height);
        }
      }
    }
  }
  
  int max_stones = get_board_size_pieces(SIZE);
  int max_caps = get_board_size_capstones(SIZE);
  
  canonical.chip(18, 0).setConstant(static_cast<float>(p0_stones_) / max_stones);
  canonical.chip(19, 0).setConstant(max_caps > 0 ? static_cast<float>(p0_caps_) / max_caps : 0.0f);
  canonical.chip(20, 0).setConstant(static_cast<float>(p1_stones_) / max_stones);
  canonical.chip(21, 0).setConstant(max_caps > 0 ? static_cast<float>(p1_caps_) / max_caps : 0.0f);
  
  return canonical;
}

template<int SIZE>
std::vector<PlayHistory> TakGS<SIZE>::symmetries(const PlayHistory& base) const noexcept {
  std::vector<PlayHistory> result;
  
  for (int rot = 0; rot < 4; ++rot) {
    PlayHistory rotated;
    rotated.canonical = base.canonical;
    rotated.v = base.v;
    rotated.pi = base.pi;
    
    if (rot > 0) {
      auto canonical = canonicalized();
      Tensor<float, 3> rot_canonical(canonical.dimensions());
      
      for (int layer = 0; layer < canonical.dimension(0); ++layer) {
        for (int row = 0; row < SIZE; ++row) {
          for (int col = 0; col < SIZE; ++col) {
            int new_row = row;
            int new_col = col;
            for (int r = 0; r < rot; ++r) {
              int temp = new_row;
              new_row = new_col;
              new_col = SIZE - 1 - temp;
            }
            rot_canonical(layer, new_row, new_col) = canonical(layer, row, col);
          }
        }
      }
      rotated.canonical = rot_canonical;
    }
    
    result.push_back(rotated);
    
    PlayHistory reflected = rotated;
    
    auto ref_canonical = rotated.canonical;
    for (int layer = 0; layer < ref_canonical.dimension(0); ++layer) {
      for (int row = 0; row < SIZE; ++row) {
        for (int col = 0; col < SIZE / 2; ++col) {
          std::swap(ref_canonical(layer, row, col),
                   ref_canonical(layer, row, SIZE - 1 - col));
        }
      }
    }
    reflected.canonical = ref_canonical;
    
    result.push_back(reflected);
  }
  
  return result;
}

template<int SIZE>
std::string TakGS<SIZE>::dump() const noexcept {
  std::ostringstream oss;
  oss << "TPS: " << to_tps() << "\n\n";
  oss << "Tak " << SIZE << "x" << SIZE << "\n";
  oss << "Turn: " << turn_ << ", Player: " << static_cast<int>(player_) << "\n";
  oss << "P0: " << p0_stones_ << " stones, " << p0_caps_ << " caps\n";
  oss << "P1: " << p1_stones_ << " stones, " << p1_caps_ << " caps\n";
  
  for (int row = 0; row < SIZE; ++row) {
    for (int col = 0; col < SIZE; ++col) {
      const Square& sq = board_[square_to_index(row, col)];
      if (sq.empty()) {
        oss << " . ";
      } else {
        const Piece& top = sq.top();
        char c = (top.owner == 0) ? 
                 (top.type == PieceType::FLAT ? 'O' : 
                  top.type == PieceType::WALL ? '|' : 'C') :
                 (top.type == PieceType::FLAT ? 'X' : 
                  top.type == PieceType::WALL ? '/' : 'c');
        oss << " " << c << " ";
      }
    }
    oss << "\n";
  }
  
  return oss.str();
}

template<int SIZE>
bool TakGS<SIZE>::check_road_win(uint8_t player) const noexcept {
  auto is_road_piece = [](const Square& sq, uint8_t p) {
    return !sq.empty() && sq.top().owner == p && 
           (sq.top().type == PieceType::FLAT || sq.top().type == PieceType::CAP);
  };
  
  std::vector<bool> visited(SIZE * SIZE, false);
  std::queue<int> q;
  
  for (int i = 0; i < SIZE; ++i) {
    if (is_road_piece(board_[i], player)) {
      q.push(i);
      visited[i] = true;
    }
  }
  
  while (!q.empty()) {
    int idx = q.front();
    q.pop();
    
    auto [row, col] = index_to_square(idx);
    if (row == SIZE - 1) return true;
    
    const int dirs[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    for (auto [dr, dc] : dirs) {
      int new_row = row + dr;
      int new_col = col + dc;
      
      if (is_valid_square(new_row, new_col)) {
        int new_idx = square_to_index(new_row, new_col);
        if (!visited[new_idx] && is_road_piece(board_[new_idx], player)) {
          visited[new_idx] = true;
          q.push(new_idx);
        }
      }
    }
  }
  
  visited.assign(SIZE * SIZE, false);
  while (!q.empty()) q.pop();
  
  for (int i = 0; i < SIZE; ++i) {
    int idx = square_to_index(0, i);
    if (is_road_piece(board_[idx], player)) {
      q.push(idx);
      visited[idx] = true;
    }
  }
  
  while (!q.empty()) {
    int idx = q.front();
    q.pop();
    
    auto [row, col] = index_to_square(idx);
    if (col == SIZE - 1) return true;
    
    const int dirs[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    for (auto [dr, dc] : dirs) {
      int new_row = row + dr;
      int new_col = col + dc;
      
      if (is_valid_square(new_row, new_col)) {
        int new_idx = square_to_index(new_row, new_col);
        if (!visited[new_idx] && is_road_piece(board_[new_idx], player)) {
          visited[new_idx] = true;
          q.push(new_idx);
        }
      }
    }
  }
  
  return false;
}

template<int SIZE>
int TakGS<SIZE>::count_flats(uint8_t player) const noexcept {
  int count = 0;
  for (const auto& sq : board_) {
    if (!sq.empty() && sq.top().owner == player && sq.top().type == PieceType::FLAT) {
      count++;
    }
  }
  return count;
}

template<int SIZE>
bool TakGS<SIZE>::is_board_full() const noexcept {
  for (const auto& sq : board_) {
    if (sq.empty()) return false;
  }
  return true;
}

template<int SIZE>
bool TakGS<SIZE>::can_move_onto(const Square& sq, const Piece& moving_piece) const noexcept {
  if (sq.empty()) return true;
  
  const Piece& top = sq.top();
  if (top.type == PieceType::CAP) return false;
  if (top.type == PieceType::WALL) {
    return moving_piece.type == PieceType::CAP;
  }
  
  return true;
}

template<int SIZE>
const std::vector<std::vector<int>>& TakGS<SIZE>::get_valid_drop_patterns(int carry, int distance) const noexcept {
  auto key = std::make_pair(carry, distance);
  
  auto it = drop_patterns_cache_.find(key);
  if (it != drop_patterns_cache_.end()) {
    return it->second;
  }
  
  std::vector<std::vector<int>> patterns;
  
  std::function<void(std::vector<int>&, int, int)> generate = 
      [&](std::vector<int>& current, int remaining, int squares_left) {
    if (squares_left == 0) {
      if (remaining == 0) {
        patterns.push_back(current);
      }
      return;
    }
    
    int min_drop = (squares_left == 1) ? remaining : 1;
    int max_drop = remaining - (squares_left - 1);
    
    for (int drop = min_drop; drop <= max_drop; ++drop) {
      current.push_back(drop);
      generate(current, remaining - drop, squares_left - 1);
      current.pop_back();
    }
  };
  
  std::vector<int> current;
  generate(current, carry, distance);
  
  auto result = drop_patterns_cache_.insert({key, std::move(patterns)});
  return result.first->second;
}

template<int SIZE>
void TakGS<SIZE>::decode_move(uint32_t move, int& from_idx, int& to_idx, 
                       PieceType& place_type, int& carry_count, 
                       std::vector<int>& drops) const noexcept {
  uint32_t placement_moves = SIZE * SIZE * 3;
  
  if (move < placement_moves) {
    from_idx = -1;
    to_idx = move / 3;
    place_type = static_cast<PieceType>(move % 3);
    carry_count = 0;
    drops.clear();
  } else {
    move -= placement_moves;
    from_idx = move / (4 * get_carry_limit(SIZE));
    int remainder = move % (4 * get_carry_limit(SIZE));
    int direction = remainder / get_carry_limit(SIZE);
    carry_count = (remainder % get_carry_limit(SIZE)) + 1;
    
    auto [row, col] = index_to_square(from_idx);
    const int dirs[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    
    for (int dist = 1; dist <= SIZE; ++dist) {
      int new_row = row + dirs[direction][0] * dist;
      int new_col = col + dirs[direction][1] * dist;
      
      if (!is_valid_square(new_row, new_col)) break;
      
      to_idx = square_to_index(new_row, new_col);
      auto patterns = get_valid_drop_patterns(carry_count, dist);
      
      if (!patterns.empty()) {
        drops = patterns[0];
        break;
      }
    }
  }
}

template<int SIZE>
uint32_t TakGS<SIZE>::encode_placement(int idx, PieceType type) const noexcept {
  return idx * 3 + static_cast<int>(type);
}

template<int SIZE>
uint32_t TakGS<SIZE>::encode_movement(int from_idx, int to_idx, 
                               const std::vector<int>& drops) const noexcept {
  auto [from_row, from_col] = index_to_square(from_idx);
  auto [to_row, to_col] = index_to_square(to_idx);
  
  int dr = (to_row > from_row) ? 1 : (to_row < from_row) ? -1 : 0;
  int dc = (to_col > from_col) ? 1 : (to_col < from_col) ? -1 : 0;
  
  int direction = 0;
  if (dr == -1) direction = 0;
  else if (dr == 1) direction = 1;
  else if (dc == -1) direction = 2;
  else if (dc == 1) direction = 3;
  
  int carry = std::accumulate(drops.begin(), drops.end(), 0);
  int base = SIZE * SIZE * 3;
  
  return base + from_idx * 4 * get_carry_limit(SIZE) + 
         direction * get_carry_limit(SIZE) + (carry - 1);
}

template<int SIZE>
void TakGS<SIZE>::update_road_cache() const noexcept {
  if (road_cache_valid_) return;
  
  update_compact_board();
  
  // Reset cache
  for (auto& row : road_connectivity_cache_) {
    std::fill(row.begin(), row.end(), false);
  }
  
  // Build adjacency for road pieces using compact representation
  const int dirs[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
  
  for (int idx = 0; idx < SIZE * SIZE; ++idx) {
    auto [row, col] = index_to_square(idx);
    
    for (auto [dr, dc] : dirs) {
      int new_row = row + dr;
      int new_col = col + dc;
      
      if (is_valid_square(new_row, new_col)) {
        int new_idx = square_to_index(new_row, new_col);
        
        // Check if both squares can be part of the same road
        for (uint8_t player = 0; player < 2; ++player) {
          if (Square::is_road_piece_compact(compact_board_cache_[idx], player) && 
              Square::is_road_piece_compact(compact_board_cache_[new_idx], player)) {
            road_connectivity_cache_[idx][new_idx] = true;
            road_connectivity_cache_[new_idx][idx] = true;
          }
        }
      }
    }
  }
  
  road_cache_valid_ = true;
}

template<int SIZE>
void TakGS<SIZE>::update_compact_board() const noexcept {
  if (compact_board_valid_) return;
  
  for (int i = 0; i < SIZE * SIZE; ++i) {
    compact_board_cache_[i] = board_[i].compact_representation();
  }
  
  compact_board_valid_ = true;
}

template<int SIZE>
std::string TakGS<SIZE>::to_tps() const noexcept {
  std::ostringstream oss;
  
  // Build board representation row by row (top to bottom)
  for (int row = SIZE - 1; row >= 0; --row) {
    if (row < SIZE - 1) oss << "/";
    
    bool first_in_row = true;
    int empty_count = 0;
    
    for (int col = 0; col < SIZE; ++col) {
      const Square& sq = board_[square_to_index(row, col)];
      
      if (sq.empty()) {
        empty_count++;
      } else {
        // Output any accumulated empty squares
        if (empty_count > 0) {
          if (!first_in_row) oss << ",";
          if (empty_count == 1) {
            oss << "x";
          } else {
            oss << "x" << empty_count;
          }
          empty_count = 0;
          first_in_row = false;
        }
        
        // Add comma if not first piece in this row
        if (!first_in_row) {
          oss << ",";
        }
        
        // Output the square
        oss << square_to_tps_string(sq);
        first_in_row = false;
      }
    }
    
    // Output any remaining empty squares
    if (empty_count > 0) {
      if (!first_in_row) oss << ",";
      if (empty_count == 1) {
        oss << "x";
      } else {
        oss << "x" << empty_count;
      }
    }
  }
  
  // Add turn and move information
  oss << " " << (player_ + 1) << " " << (turn_ + 1);
  
  return oss.str();
}

template<int SIZE>
std::string TakGS<SIZE>::square_to_tps_string(const Square& sq) const noexcept {
  if (sq.empty()) return "";
  
  std::string result;
  
  // Build stack from bottom to top
  for (const auto& piece : sq.stack) {
    result += std::to_string(piece.owner + 1);  // Convert 0-based to 1-based
    
    if (piece.type == PieceType::WALL) {
      result += "S";
    } else if (piece.type == PieceType::CAP) {
      result += "C";
    }
    // FLAT pieces don't need a suffix
  }
  
  return result;
}

template<int SIZE>
void TakGS<SIZE>::parse_tps_string(const std::string& tps_string) {
  // Remove brackets and "TPS" prefix if present
  std::string cleaned = tps_string;
  if (cleaned.find("[TPS \"") == 0) {
    size_t start = cleaned.find("\"") + 1;
    size_t end = cleaned.find("\"", start);
    if (end != std::string::npos) {
      cleaned = cleaned.substr(start, end - start);
    }
  }
  
  // Split by spaces to get board, player, turn
  std::istringstream iss(cleaned);
  std::string board_str, player_str, turn_str;
  
  if (!(iss >> board_str >> player_str >> turn_str)) {
    throw std::invalid_argument("Invalid TPS format: expected 'board player turn'");
  }
  
  // Parse player (1-based to 0-based)
  int player_num = std::stoi(player_str);
  if (player_num < 1 || player_num > 2) {
    throw std::invalid_argument("Invalid player number: must be 1 or 2");
  }
  player_ = static_cast<uint8_t>(player_num - 1);
  
  // Parse turn (1-based to 0-based)
  int turn_num = std::stoi(turn_str);
  if (turn_num < 1) {
    throw std::invalid_argument("Invalid turn number: must be >= 1");
  }
  turn_ = static_cast<uint32_t>(turn_num - 1);
  
  // Clear the board
  for (auto& sq : board_) {
    sq.stack.clear();
  }
  
  // Parse board state
  std::vector<std::string> rows;
  std::stringstream ss(board_str);
  std::string row;
  
  while (std::getline(ss, row, '/')) {
    rows.push_back(row);
  }
  
  if (static_cast<int>(rows.size()) != SIZE) {
    throw std::invalid_argument("Board has " + std::to_string(rows.size()) + 
                               " rows, expected " + std::to_string(SIZE));
  }
  
  // Process rows from top to bottom (reverse order in our internal representation)
  for (int row_idx = 0; row_idx < SIZE; ++row_idx) {
    int actual_row = SIZE - 1 - row_idx;  // Convert to our coordinate system
    const std::string& row_str = rows[row_idx];
    
    int col = 0;
    for (size_t i = 0; i < row_str.length(); ++i) {
      if (col >= SIZE) {
        throw std::invalid_argument("Too many columns in row " + std::to_string(row_idx));
      }
      
      char c = row_str[i];
      if (c == 'x') {
        // Empty square(s)
        int count = 1;
        if (i + 1 < row_str.length() && std::isdigit(row_str[i + 1])) {
          count = row_str[i + 1] - '0';
          i++; // Skip the digit
        }
        col += count;
      } else if (c == ',') {
        // Comma separator, ignore
        continue;
      } else if (std::isdigit(c)) {
        // Start of a piece/stack
        std::string piece_str;
        while (i < row_str.length() && row_str[i] != ',' && row_str[i] != '/') {
          piece_str += row_str[i];
          i++;
        }
        i--; // Back up one since the loop will increment
        
        // Parse the piece string
        Square& sq = board_[square_to_index(actual_row, col)];
        for (size_t j = 0; j < piece_str.length(); ++j) {
          if (std::isdigit(piece_str[j])) {
            uint8_t owner = piece_str[j] - '1';  // Convert 1-based to 0-based
            PieceType type = PieceType::FLAT;
            
            // Check for piece type modifier
            if (j + 1 < piece_str.length()) {
              if (piece_str[j + 1] == 'S') {
                type = PieceType::WALL;
                j++; // Skip the 'S'
              } else if (piece_str[j + 1] == 'C') {
                type = PieceType::CAP;
                j++; // Skip the 'C'
              }
            }
            
            sq.add(Piece(owner, type));
          }
        }
        col++;
      } else {
        throw std::invalid_argument("Invalid character in TPS string: " + std::string(1, c));
      }
    }
    
    // Validate that we have exactly SIZE columns
    if (col != SIZE) {
      throw std::invalid_argument("Row " + std::to_string(row_idx) + " has " + 
                                 std::to_string(col) + " columns, expected " + 
                                 std::to_string(SIZE));
    }
  }
  
  // Calculate remaining pieces based on what's on the board
  p0_stones_ = get_board_size_pieces(SIZE);
  p0_caps_ = get_board_size_capstones(SIZE);
  p1_stones_ = get_board_size_pieces(SIZE);
  p1_caps_ = get_board_size_capstones(SIZE);
  
  for (const auto& sq : board_) {
    for (const auto& piece : sq.stack) {
      if (piece.owner == 0) {
        if (piece.type == PieceType::CAP) {
          p0_caps_--;
        } else {
          p0_stones_--;
        }
      } else {
        if (piece.type == PieceType::CAP) {
          p1_caps_--;
        } else {
          p1_stones_--;
        }
      }
    }
  }
}

template<int SIZE>
std::pair<int, int> TakGS<SIZE>::parse_ptn_algebraic(const std::string& square) const {
  if (square.length() < 2) {
    throw std::invalid_argument("Invalid square notation: " + square);
  }
  
  char col_char = square[0];
  if (col_char < 'a' || col_char > 'z') {
    throw std::invalid_argument("Invalid column in square notation: " + square);
  }
  
  int col = col_char - 'a';
  if (col >= SIZE) {
    throw std::invalid_argument("Column out of bounds for " + std::to_string(SIZE) + "x" + std::to_string(SIZE) + " board: " + square);
  }
  
  std::string row_str = square.substr(1);
  int row;
  try {
    row = std::stoi(row_str) - 1;  // Convert to 0-based
  } catch (const std::exception&) {
    throw std::invalid_argument("Invalid row in square notation: " + square);
  }
  
  if (row < 0 || row >= SIZE) {
    throw std::invalid_argument("Row out of bounds for " + std::to_string(SIZE) + "x" + std::to_string(SIZE) + " board: " + square);
  }
  
  return {row, col};
}

template<int SIZE>
uint32_t TakGS<SIZE>::ptn_to_move_index(const std::string& ptn_move) const {
  if (ptn_move.empty()) {
    throw std::invalid_argument("Empty PTN move");
  }
  
  // Remove common annotations
  std::string clean_move = ptn_move;
  // Remove trailing annotations
  while (!clean_move.empty() && (clean_move.back() == '\'' || clean_move.back() == '!' || 
                                 clean_move.back() == '?' || clean_move.back() == '*')) {
    clean_move.pop_back();
  }
  
  if (clean_move.empty()) {
    throw std::invalid_argument("Invalid PTN move: " + ptn_move);
  }
  
  // Check for placement moves: [CS]?[a-z][1-9]+
  std::regex placement_regex(R"(^([CS]?)([a-z][1-9]\d*)$)");
  std::smatch placement_match;
  
  if (std::regex_match(clean_move, placement_match, placement_regex)) {
    std::string piece_prefix = placement_match[1].str();
    std::string square_str = placement_match[2].str();
    
    auto [row, col] = parse_ptn_algebraic(square_str);
    int board_idx = square_to_index(row, col);
    
    PieceType piece_type = PieceType::FLAT;
    if (piece_prefix == "S") {
      piece_type = PieceType::WALL;
    } else if (piece_prefix == "C") {
      piece_type = PieceType::CAP;
    }
    
    return encode_placement(board_idx, piece_type);
  }
  
  // Check for movement moves: (\\d*)([a-z][1-9]+)([<>+-])(\\d*)
  std::regex movement_regex(R"(^([1-9]\d*)?([a-z][1-9]\d*)([<>+-])(\d*)$)");
  std::smatch movement_match;
  
  if (std::regex_match(clean_move, movement_match, movement_regex)) {
    std::string count_str = movement_match[1].str();
    std::string from_square = movement_match[2].str();
    std::string direction = movement_match[3].str();
    std::string drops_str = movement_match[4].str();
    
    int carry_count = count_str.empty() ? 1 : std::stoi(count_str);
    auto [from_row, from_col] = parse_ptn_algebraic(from_square);
    
    // Parse direction
    int dr = 0, dc = 0;
    if (direction == "<") {
      dr = 0; dc = -1;  // West
    } else if (direction == ">") {
      dr = 0; dc = 1;   // East
    } else if (direction == "+") {
      dr = -1; dc = 0;  // North
    } else if (direction == "-") {
      dr = 1; dc = 0;   // South
    } else {
      throw std::invalid_argument("Invalid direction: " + direction);
    }
    
    // Parse drop pattern
    std::vector<int> drops;
    if (drops_str.empty()) {
      drops.push_back(carry_count);  // Default: drop all stones
    } else {
      for (char c : drops_str) {
        if (c >= '1' && c <= '9') {
          drops.push_back(c - '0');
        } else {
          throw std::invalid_argument("Invalid drop pattern: " + drops_str);
        }
      }
    }
    
    // Validate carry count matches drop total
    int total_drops = std::accumulate(drops.begin(), drops.end(), 0);
    if (total_drops != carry_count) {
      throw std::invalid_argument("Carry count (" + std::to_string(carry_count) + 
                                 ") doesn't match drop total (" + std::to_string(total_drops) + ")");
    }
    
    // Calculate destination
    int distance = static_cast<int>(drops.size());
    int to_row = from_row + dr * distance;
    int to_col = from_col + dc * distance;
    
    if (!is_valid_square(to_row, to_col)) {
      throw std::invalid_argument("Movement destination out of bounds");
    }
    
    int from_idx = square_to_index(from_row, from_col);
    int to_idx = square_to_index(to_row, to_col);
    
    return encode_movement(from_idx, to_idx, drops);
  }
  
  throw std::invalid_argument("Invalid PTN move format: " + ptn_move);
}

template<int SIZE>
thread_local std::unordered_map<std::pair<int, int>, std::vector<std::vector<int>>, typename TakGS<SIZE>::PairHash> TakGS<SIZE>::drop_patterns_cache_;

// Explicit template instantiations
template class TakGS<4>;
template class TakGS<5>;
template class TakGS<6>;

}  // namespace alphazero::tak_gs