#include "tak_gs.h"

#include <algorithm>
#include <sstream>
#include <numeric>
#include <queue>

namespace alphazero::tak_gs {

template<int SIZE>
TakGS<SIZE>::TakGS(bool opening_swap, float komi) 
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
thread_local std::unordered_map<std::pair<int, int>, std::vector<std::vector<int>>, typename TakGS<SIZE>::PairHash> TakGS<SIZE>::drop_patterns_cache_;

// Explicit template instantiations
template class TakGS<4>;
template class TakGS<5>;
template class TakGS<6>;

}  // namespace alphazero::tak_gs