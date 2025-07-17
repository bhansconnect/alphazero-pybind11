#pragma once

#include <vector>
#include <stack>
#include <cstdint>
#include <unordered_map>

#include "dll_export.h"
#include "game_state.h"

namespace alphazero::tak_gs {

constexpr const int DEFAULT_SIZE = 5;
constexpr const int MIN_SIZE = 4;
constexpr const int MAX_SIZE = 6;

constexpr const int MAX_HEIGHT = 30;
constexpr const int NN_MAX_HEIGHT = 10;

constexpr const int NUM_PLAYERS = 2;
constexpr const int NUM_SYMMETRIES = 8;

enum class PieceType : uint8_t {
  FLAT = 0,
  WALL = 1,
  CAP = 2
};

struct Piece {
  uint8_t owner;
  PieceType type;
  
  Piece() : owner(0), type(PieceType::FLAT) {}
  Piece(uint8_t o, PieceType t) : owner(o), type(t) {}
  
  bool operator==(const Piece& other) const {
    return owner == other.owner && type == other.type;
  }
};

struct Square {
  std::vector<Piece> stack;
  
  Square() { stack.reserve(MAX_HEIGHT); }
  
  bool empty() const { return stack.empty(); }
  size_t height() const { return stack.size(); }
  
  const Piece& top() const { 
    assert(!empty());
    return stack.back(); 
  }
  
  void add(const Piece& piece) {
    assert(stack.size() < MAX_HEIGHT);
    stack.push_back(piece);
  }
  
  Piece remove() {
    assert(!empty());
    Piece p = stack.back();
    stack.pop_back();
    return p;
  }
  
  bool operator==(const Square& other) const {
    return stack == other.stack;
  }
  
  // Compact representation: pack height and top piece info into 8 bits
  // Bits 0-4: height (0-30), Bits 5-6: top piece type, Bit 7: top piece owner
  uint8_t compact_representation() const noexcept {
    if (empty()) return 0;
    
    uint8_t result = static_cast<uint8_t>(std::min(height(), size_t(31)));
    const Piece& top_piece = top();
    result |= (static_cast<uint8_t>(top_piece.type) << 5);
    result |= (top_piece.owner << 7);
    
    return result;
  }
  
  static bool is_road_piece_compact(uint8_t compact, uint8_t player) noexcept {
    if ((compact & 0x1F) == 0) return false; // empty
    uint8_t owner = (compact >> 7) & 1;
    uint8_t type = (compact >> 5) & 3;
    return owner == player && (type == 0 || type == 2); // FLAT or CAP
  }
};

inline int get_board_size_pieces(int size) {
  switch (size) {
    case 4: return 15;
    case 5: return 21;
    case 6: return 30;
    default: assert(false); return 0;
  }
}

inline int get_board_size_capstones(int size) {
  switch (size) {
    case 4: return 0;
    case 5: return 1;
    case 6: return 1;
    default: assert(false); return 0;
  }
}

inline int get_carry_limit(int size) {
  return size;
}

class DLLEXPORT TakGS : public GameState {
 public:
  TakGS(int size = DEFAULT_SIZE, bool opening_swap = true, float komi = 0.0f);
  TakGS(int size, std::vector<Square> board, uint8_t player, uint32_t turn,
        int p0_stones, int p0_caps, int p1_stones, int p1_caps, bool opening_swap, 
        float komi = 0.0f, uint32_t moves_without_placement = 0);
  
  [[nodiscard]] std::unique_ptr<GameState> copy() const noexcept override;
  [[nodiscard]] bool operator==(const GameState& other) const noexcept override;
  
  void hash(absl::HashState h) const override;
  
  [[nodiscard]] uint8_t current_player() const noexcept override {
    return player_;
  }
  
  [[nodiscard]] uint32_t current_turn() const noexcept override {
    return turn_;
  }
  
  [[nodiscard]] uint32_t num_moves() const noexcept override;
  
  [[nodiscard]] uint8_t num_players() const noexcept override {
    return NUM_PLAYERS;
  }
  
  [[nodiscard]] Vector<uint8_t> valid_moves() const noexcept override;
  
  void play_move(uint32_t move) override;
  
  [[nodiscard]] std::optional<Vector<float>> scores() const noexcept override;
  
  [[nodiscard]] Tensor<float, 3> canonicalized() const noexcept override;
  
  [[nodiscard]] uint8_t num_symmetries() const noexcept override {
    return NUM_SYMMETRIES;
  }
  
  [[nodiscard]] std::vector<PlayHistory> symmetries(
      const PlayHistory& base) const noexcept override;
  
  [[nodiscard]] std::string dump() const noexcept override;
  
  void minimize_storage() override {}
  
  [[nodiscard]] int board_size() const noexcept { return size_; }
  [[nodiscard]] std::array<int, 3> board_shape() const noexcept;
  [[nodiscard]] std::array<int, 3> canonical_shape() const noexcept;
  
 private:
  int size_;
  std::vector<Square> board_;
  uint8_t player_;
  uint32_t turn_;
  int p0_stones_;
  int p0_caps_;
  int p1_stones_;
  int p1_caps_;
  bool opening_swap_;
  float komi_;
  uint32_t moves_without_placement_;
  
  [[nodiscard]] int square_to_index(int row, int col) const noexcept {
    return row * size_ + col;
  }
  
  [[nodiscard]] std::pair<int, int> index_to_square(int index) const noexcept {
    return {index / size_, index % size_};
  }
  
  [[nodiscard]] bool is_valid_square(int row, int col) const noexcept {
    return row >= 0 && row < size_ && col >= 0 && col < size_;
  }
  
  [[nodiscard]] bool check_road_win(uint8_t player) const noexcept;
  [[nodiscard]] int count_flats(uint8_t player) const noexcept;
  [[nodiscard]] bool is_board_full() const noexcept;
  
  void decode_move(uint32_t move, int& from_idx, int& to_idx, 
                   PieceType& place_type, int& carry_count, 
                   std::vector<int>& drops) const noexcept;
  
  [[nodiscard]] uint32_t encode_placement(int idx, PieceType type) const noexcept;
  [[nodiscard]] uint32_t encode_movement(int from_idx, int to_idx, 
                                         const std::vector<int>& drops) const noexcept;
  
  [[nodiscard]] bool can_move_onto(const Square& sq, const Piece& moving_piece) const noexcept;
  [[nodiscard]] const std::vector<std::vector<int>>& get_valid_drop_patterns(int carry, int distance) const noexcept;
  
  struct PairHash {
    std::size_t operator()(const std::pair<int, int>& p) const noexcept {
      return std::hash<int>{}(p.first) ^ (std::hash<int>{}(p.second) << 1);
    }
  };
  
  static std::unordered_map<std::pair<int, int>, std::vector<std::vector<int>>, PairHash> drop_patterns_cache_;
  
  // Cached road connectivity for performance optimization
  mutable std::vector<std::vector<bool>> road_connectivity_cache_;
  mutable bool road_cache_valid_;
  
  // Compact board representation for memory efficiency
  mutable std::vector<uint8_t> compact_board_cache_;
  mutable bool compact_board_valid_;
  
  void invalidate_road_cache() const noexcept {
    road_cache_valid_ = false;
    compact_board_valid_ = false;
  }
  
  void update_road_cache() const noexcept;
  void update_compact_board() const noexcept;
};

}  // namespace alphazero::tak_gs