#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

#include "dll_export.h"
#include "game_state.h"

namespace alphazero::star_gambit_gs {

// ============================================================================
// Game Size Configurations
// ============================================================================

struct SkirmishConfig {
  static constexpr int BOARD_SIDE = 5;
  static constexpr int MAX_FIGHTERS = 3;
  static constexpr int MAX_CRUISERS = 1;
  static constexpr int MAX_DREADNOUGHTS = 0;
  static constexpr int STARTING_FIGHTERS = 3;
  static constexpr int STARTING_CRUISERS = 1;
  static constexpr int STARTING_DREADNOUGHTS = 0;
};

struct ClashConfig {
  static constexpr int BOARD_SIDE = 5;
  static constexpr int MAX_FIGHTERS = 3;
  static constexpr int MAX_CRUISERS = 2;
  static constexpr int MAX_DREADNOUGHTS = 1;
  static constexpr int STARTING_FIGHTERS = 3;
  static constexpr int STARTING_CRUISERS = 2;
  static constexpr int STARTING_DREADNOUGHTS = 1;
};

struct BattleConfig {
  static constexpr int BOARD_SIDE = 6;
  static constexpr int MAX_FIGHTERS = 4;
  static constexpr int MAX_CRUISERS = 3;
  static constexpr int MAX_DREADNOUGHTS = 2;
  static constexpr int STARTING_FIGHTERS = 4;
  static constexpr int STARTING_CRUISERS = 3;
  static constexpr int STARTING_DREADNOUGHTS = 2;
};

// ============================================================================
// Constants (shared across all game sizes)
// ============================================================================

constexpr int NUM_PLAYERS = 2;
constexpr int NUM_SYMMETRIES = 2;  // Identity + NW-SE diagonal mirror

// Unit type constants
constexpr int NUM_UNIT_TYPES = 4;

// HP values
constexpr int FIGHTER_HP = 3;
constexpr int CRUISER_HP = 4;
constexpr int DREADNOUGHT_HP = 6;
constexpr int PORTAL_HP = 5;

// Movement values (moves per turn)
constexpr int FIGHTER_MOVES = 2;
constexpr int CRUISER_MOVES = 1;
constexpr int DREADNOUGHT_MOVES = 1;
constexpr int PORTAL_MOVES = 0;

// Maximum turns before game ends in draw
constexpr int MAX_TURNS = 200;

// ============================================================================
// Movement Direction Constants
// ============================================================================

// Fighter movement directions: forward, forward-left, forward-right
constexpr int FIGHTER_MOVE_DIRS = 3;
enum class FighterMove : uint8_t {
  FORWARD = 0,       // Move in facing direction
  FORWARD_LEFT = 1,  // Move in (facing + 1) % 6 direction
  FORWARD_RIGHT = 2  // Move in (facing + 5) % 6 direction
};

// Cruiser movement directions: rotate-left, forward-left, forward, forward-right, rotate-right
constexpr int CRUISER_MOVE_DIRS = 5;
enum class CruiserMove : uint8_t {
  ROTATE_LEFT = 0,    // Anchor stays, facing += 1
  FORWARD_LEFT = 1,   // Move forward, new facing = (facing + 1) % 6
  FORWARD = 2,        // Move forward, facing unchanged
  FORWARD_RIGHT = 3,  // Move forward, new facing = (facing + 5) % 6
  ROTATE_RIGHT = 4    // Anchor stays, facing -= 1
};

// Dreadnought movement directions: rotate-left, forward-left, forward-right, rotate-right
constexpr int DREAD_MOVE_DIRS = 4;
enum class DreadnoughtMove : uint8_t {
  ROTATE_LEFT = 0,    // Anchor stays, facing += 1
  FORWARD_LEFT = 1,   // Move forward-left
  FORWARD_RIGHT = 2,  // Move forward-right
  ROTATE_RIGHT = 3    // Anchor stays, facing -= 1
};

// ============================================================================
// Cannon Constants
// ============================================================================

constexpr int FIGHTER_CANNONS = 1;    // forward only
constexpr int CRUISER_CANNONS = 3;    // left, forward, right
constexpr int DREAD_CANNONS = 4;      // rear-left, front-left, front-right, rear-right

// Maximum cannons per unit (for bitmask sizing)
constexpr int MAX_CANNONS = 4;

// ============================================================================
// Hex Coordinate System
// ============================================================================

struct Hex {
  int8_t q;
  int8_t r;

  bool operator==(const Hex& other) const { return q == other.q && r == other.r; }
  bool operator!=(const Hex& other) const { return !(*this == other); }
};

// ============================================================================
// Hex Lookup Tables (precomputed for O(1) access)
// ============================================================================

template<int BoardSide>
struct HexLookupTables {
  static constexpr int NUM_HEXES = 3 * BoardSide * BoardSide - 3 * BoardSide + 1;
  static constexpr int COORD_RANGE = 2 * BoardSide - 1;

  std::array<Hex, NUM_HEXES> index_to_hex;
  // 2D array indexed by (q + BoardSide-1, r + BoardSide-1) to handle negative coords
  // Invalid entries are set to -1
  std::array<std::array<int, COORD_RANGE>, COORD_RANGE> hex_to_index;

  HexLookupTables() {
    // Initialize hex_to_index with -1 (invalid)
    for (auto& row : hex_to_index) {
      row.fill(-1);
    }

    // Build tables using same iteration order as original functions
    int idx = 0;
    for (int r = -(BoardSide - 1); r < BoardSide; ++r) {
      for (int q = -(BoardSide - 1); q < BoardSide; ++q) {
        int s = -q - r;
        if (std::abs(q) < BoardSide && std::abs(r) < BoardSide &&
            std::abs(s) < BoardSide) {
          index_to_hex[idx] = {static_cast<int8_t>(q), static_cast<int8_t>(r)};
          hex_to_index[q + BoardSide - 1][r + BoardSide - 1] = idx;
          ++idx;
        }
      }
    }
  }
};

// Global lookup table instances (initialized once at program startup)
inline const HexLookupTables<5> HEX_TABLES_5{};
inline const HexLookupTables<6> HEX_TABLES_6{};

// O(1) lookup functions
template<int BoardSide>
inline int hex_to_index_fast(const Hex& h) {
  if constexpr (BoardSide == 5) {
    return HEX_TABLES_5.hex_to_index[h.q + BoardSide - 1][h.r + BoardSide - 1];
  } else {
    return HEX_TABLES_6.hex_to_index[h.q + BoardSide - 1][h.r + BoardSide - 1];
  }
}

template<int BoardSide>
inline Hex index_to_hex_fast(int index) {
  if constexpr (BoardSide == 5) {
    return HEX_TABLES_5.index_to_hex[index];
  } else {
    return HEX_TABLES_6.index_to_hex[index];
  }
}

// 6 hex directions for flat-top hexes
// Direction indices: 0=E, 1=NE, 2=NW, 3=W, 4=SW, 5=SE
constexpr std::array<Hex, 6> HEX_DIRECTIONS = {{
    {1, 0},   // 0: East
    {1, -1},  // 1: Northeast
    {0, -1},  // 2: Northwest
    {-1, 0},  // 3: West
    {-1, 1},  // 4: Southwest
    {0, 1}    // 5: Southeast
}};

// Opposite direction lookup
constexpr std::array<int, 6> OPPOSITE_DIRECTION = {3, 4, 5, 0, 1, 2};

// Direction names for display
constexpr std::array<const char*, 6> DIRECTION_NAMES = {"E", "NE", "NW", "W", "SW", "SE"};

// Rotate direction by steps (positive = counterclockwise, negative = clockwise)
inline int rotate_direction(int dir, int steps) {
  return (dir + steps + 6) % 6;
}

// ============================================================================
// 2D Coordinate Helpers for Observation Space
// ============================================================================

// Convert hex (q, r) to 2D array position (row, col) for observation tensor
template<int BoardSide>
inline std::pair<int, int> hex_to_2d(const Hex& h) {
  return {h.q + BoardSide - 1, h.r + BoardSide - 1};
}

// Check if a 2D position is valid (corresponds to a valid hex)
template<int BoardSide>
inline bool is_valid_2d_pos(int row, int col) {
  if (row < 0 || row >= 2 * BoardSide - 1 || col < 0 || col >= 2 * BoardSide - 1) {
    return false;
  }
  // Convert back to hex and check if valid
  int q = row - (BoardSide - 1);
  int r = col - (BoardSide - 1);
  int s = -q - r;
  return std::abs(q) < BoardSide && std::abs(r) < BoardSide && std::abs(s) < BoardSide;
}

// ============================================================================
// Unit Types and Properties
// ============================================================================

enum class UnitType : uint8_t {
  FIGHTER = 0,
  CRUISER = 1,
  DREADNOUGHT = 2,
  PORTAL = 3
};

inline int get_max_hp(UnitType type) {
  switch (type) {
    case UnitType::FIGHTER: return FIGHTER_HP;
    case UnitType::CRUISER: return CRUISER_HP;
    case UnitType::DREADNOUGHT: return DREADNOUGHT_HP;
    case UnitType::PORTAL: return PORTAL_HP;
  }
  return 0;
}

inline int get_max_moves(UnitType type) {
  switch (type) {
    case UnitType::FIGHTER: return FIGHTER_MOVES;
    case UnitType::CRUISER: return CRUISER_MOVES;
    case UnitType::DREADNOUGHT: return DREADNOUGHT_MOVES;
    case UnitType::PORTAL: return PORTAL_MOVES;
  }
  return 0;
}

inline int get_num_cannons(UnitType type) {
  switch (type) {
    case UnitType::FIGHTER: return FIGHTER_CANNONS;
    case UnitType::CRUISER: return CRUISER_CANNONS;
    case UnitType::DREADNOUGHT: return DREAD_CANNONS;
    case UnitType::PORTAL: return 0;
  }
  return 0;
}

inline int get_num_move_dirs(UnitType type) {
  switch (type) {
    case UnitType::FIGHTER: return FIGHTER_MOVE_DIRS;
    case UnitType::CRUISER: return CRUISER_MOVE_DIRS;
    case UnitType::DREADNOUGHT: return DREAD_MOVE_DIRS;
    case UnitType::PORTAL: return 0;
  }
  return 0;
}

inline int get_unit_size(UnitType type) {
  switch (type) {
    case UnitType::FIGHTER: return 1;
    case UnitType::CRUISER: return 2;
    case UnitType::DREADNOUGHT: return 3;
    case UnitType::PORTAL: return 3;
  }
  return 1;
}

// ============================================================================
// Unit Representation
// ============================================================================

struct Unit {
  UnitType type;
  uint8_t player;        // 0 or 1
  uint8_t slot;          // Slot index within type for this player (0, 1, 2, ...)
  uint8_t hp;            // Current health
  uint8_t facing;        // Direction 0-5
  int8_t anchor_q;       // Anchor hex q coordinate
  int8_t anchor_r;       // Anchor hex r coordinate
  uint8_t moves_left;    // Moves remaining this turn
  uint8_t cannons_fired; // Bitmask of cannons fired this turn

  bool is_alive() const { return hp > 0; }
};

// ============================================================================
// Movement Result
// ============================================================================

struct MoveResult {
  Hex new_anchor;
  int new_facing;
  bool valid;  // False if move would go out of bounds
};

// ============================================================================
// Hex Utility Functions (declarations)
// ============================================================================

Hex hex_add(const Hex& a, const Hex& b);
Hex hex_subtract(const Hex& a, const Hex& b);
Hex hex_scale(const Hex& h, int k);
int hex_distance(const Hex& a, const Hex& b);
Hex hex_neighbor(const Hex& h, int direction);
bool hex_in_bounds(const Hex& h, int board_side);
int hex_to_index(const Hex& h, int board_side);
Hex index_to_hex(int index, int board_side);
int compute_num_hexes(int board_side);

// Get all hexes occupied by a unit given its anchor and facing
std::vector<Hex> get_unit_hexes(UnitType type, const Hex& anchor, int facing);

// Get all hexes of the portal at a given corner (top or bottom)
std::vector<Hex> get_portal_hexes(int player, int board_side);

// Get the deploy hex for a player
Hex get_deploy_hex(int player, int board_side);

// Get valid deploy facings for a unit type
std::vector<int> get_valid_deploy_facings(UnitType type, int player);

// ============================================================================
// Cannon Information
// ============================================================================

struct CannonInfo {
  int direction_offset;  // Add to facing to get absolute direction
  int source_hex_idx;    // Index into unit's hex list (0 = anchor, etc.)
};

std::vector<CannonInfo> get_cannon_info(UnitType type);

// ============================================================================
// Python Binding Support Structs
// ============================================================================

struct UnitInfo {
  int player;       // 0 or 1
  int type;         // 0=Fighter, 1=Cruiser, 2=Dreadnought, 3=Portal
  int slot;         // 0-indexed slot
  int hp;           // Current health
  int anchor_q;     // Hex position q
  int anchor_r;     // Hex position r
  int facing;       // 0=E, 1=NE, 2=NW, 3=W, 4=SW, 5=SE
  int moves_left;   // Moves remaining this turn
};

struct FireInfo {
  bool has_target;      // False if no target in range
  int target_player;    // Target unit's player
  int target_type;      // Target unit's type
  int target_slot;      // Target unit's slot
  int damage;           // 1 (range 2) or 2 (range 1)
};

// Line of sight check
bool has_line_of_sight(const Hex& from, int direction, int distance,
                       const std::vector<Hex>& occupied_hexes);

// ============================================================================
// Spatial Action Space Encoding
// ============================================================================

// Per-position action slots (10 total)
// Movement slots: 0-4 (relative to unit's facing)
// Fire slots: 5-9 (relative to unit's facing)
enum class SpatialAction : uint8_t {
  MOVE_FORWARD = 0,       // All units except dreadnought
  MOVE_FORWARD_LEFT = 1,  // All units
  MOVE_FORWARD_RIGHT = 2, // All units
  ROTATE_LEFT = 3,        // Cruiser, Dreadnought
  ROTATE_RIGHT = 4,       // Cruiser, Dreadnought
  FIRE_FORWARD = 5,       // Fighter, Cruiser
  FIRE_FORWARD_LEFT = 6,  // Cruiser (fl cannon), Dreadnought (fl cannon)
  FIRE_FORWARD_RIGHT = 7, // Cruiser (fr cannon), Dreadnought (fr cannon)
  FIRE_REAR_LEFT = 8,     // Dreadnought (rl cannon)
  FIRE_REAR_RIGHT = 9     // Dreadnought (rr cannon)
};

constexpr int ACTIONS_PER_POSITION = 10;

// Slot map for canonicalization: swaps left/right directions
// Used for both 180° rotation (P1 canonicalization) and NW-SE mirror symmetry
// Self-inverse: SLOT_MAP[SLOT_MAP[x]] == x
constexpr int SLOT_MAP[10] = {0, 2, 1, 4, 3, 5, 7, 6, 9, 8};

// Direction map for NW-axis mirror symmetry: [4, 3, 2, 1, 0, 5]
// Mirror about the NW direction (toward opponent's portal)
// NW stays NW, SE stays SE, NE↔W, E↔SW
constexpr int MIRROR_DIRECTION_MAP[6] = {4, 3, 2, 1, 0, 5};

// Deploy action mirror for dreadnoughts uses a different axis (between NE/NW)
// Swaps 0↔3, 1↔2, 4↔5 to preserve valid dreadnought facings {0,1,2,3}
constexpr int DEPLOY_MIRROR_D[6] = {3, 2, 1, 0, 5, 4};

template<typename Config>
struct ActionSpace {
  // Board dimensions for 2D spatial representation
  static constexpr int BOARD_DIM = 2 * Config::BOARD_SIDE - 1;

  // Hex count for this board size (for reference)
  static constexpr int NUM_HEXES = 3 * Config::BOARD_SIDE * Config::BOARD_SIDE
                                   - 3 * Config::BOARD_SIDE + 1;

  // Spatial actions: full BOARD_DIM x BOARD_DIM x 10 grid
  // Invalid positions (non-hex cells) are simply never marked valid
  static constexpr int SPATIAL_ACTIONS = BOARD_DIM * BOARD_DIM * ACTIONS_PER_POSITION;

  // Deploy actions: 3 unit types * 6 facings (mask invalid facings at runtime)
  static constexpr int DEPLOY_ACTIONS = 3 * 6;

  // End turn
  static constexpr int END_TURN_ACTIONS = 1;

  // Action layout: [spatial_actions | deploy_actions | end_turn]
  static constexpr int SPATIAL_OFFSET = 0;
  static constexpr int DEPLOY_OFFSET = SPATIAL_OFFSET + SPATIAL_ACTIONS;
  static constexpr int END_TURN_OFFSET = DEPLOY_OFFSET + DEPLOY_ACTIONS;

  // Total actions
  static constexpr int NUM_MOVES = END_TURN_OFFSET + END_TURN_ACTIONS;

  // Helper: encode spatial action from (row, col, action_type)
  static constexpr int encode_spatial_action(int row, int col, int action_type) {
    return row * BOARD_DIM * ACTIONS_PER_POSITION + col * ACTIONS_PER_POSITION + action_type;
  }

  // Helper: decode spatial action to (row, col, action_type)
  static constexpr void decode_spatial_action(int action_idx, int& row, int& col, int& action_type) {
    action_type = action_idx % ACTIONS_PER_POSITION;
    int pos = action_idx / ACTIONS_PER_POSITION;
    col = pos % BOARD_DIM;
    row = pos / BOARD_DIM;
  }

  // Helper: check if (row, col) corresponds to a valid hex position
  static bool is_valid_hex_pos(int row, int col) {
    int q = row - (Config::BOARD_SIDE - 1);
    int r = col - (Config::BOARD_SIDE - 1);
    int s = -q - r;
    return std::abs(q) < Config::BOARD_SIDE &&
           std::abs(r) < Config::BOARD_SIDE &&
           std::abs(s) < Config::BOARD_SIDE;
  }

  // Helper: encode deploy action
  static constexpr int encode_deploy(int type_idx, int facing) {
    return DEPLOY_OFFSET + type_idx * 6 + facing;
  }

  // Helper: decode deploy action
  static constexpr void decode_deploy(int action_idx, int& type_idx, int& facing) {
    int rel = action_idx - DEPLOY_OFFSET;
    type_idx = rel / 6;
    facing = rel % 6;
  }

  // Canonical representation shape
  // Observations are canonicalized to current player's perspective

  // Type-based presence: 4 types × 2 (my/opponent) = 8
  static constexpr int TYPE_PRESENCE_CHANNELS = 8;

  // Heading: 6 directions, broadcast to all unit hexes
  static constexpr int HEADING_CHANNELS = 6;

  // HP/moves: single channel each, broadcast to all unit hexes
  static constexpr int HP_CHANNELS = 1;
  static constexpr int MOVES_CHANNELS = 1;

  // Cannon availability: 5 types, broadcast to all unit hexes
  static constexpr int CANNON_CHANNELS = 5;

  // Global: valid_hex(1) + has_acted(1) + repetition(1)
  //         + reserves(6) + portal_hp(2) = 11
  static constexpr int GLOBAL_CHANNELS = 11;

  // Canonical observation channels (32 total):
  //   0:     Valid hex mask
  //   1-4:   My unit types (Fighter, Cruiser, Dreadnought, Portal)
  //   5-8:   Opponent unit types
  //   9-14:  Facing direction (6 dirs, all unit hexes)
  //   15:    HP (normalized, all unit hexes)
  //   16:    Moves remaining (normalized, all unit hexes)
  //   17-21: Cannon availability (5 slots, all unit hexes)
  //   22:    Has taken action (broadcast)
  //   23:    Repetition count (broadcast, 0/0.5/1.0)
  //   24-26: My reserves (Fighter, Cruiser, Dreadnought)
  //   27-29: Opponent reserves
  //   30:    My portal HP (broadcast)
  //   31:    Opponent portal HP (broadcast)
  static constexpr int CANONICAL_CHANNELS = 32;
  static constexpr std::array<int, 3> CANONICAL_SHAPE = {CANONICAL_CHANNELS, BOARD_DIM, BOARD_DIM};
};

// ============================================================================
// Game State Class (Templated)
// ============================================================================

template<typename Config>
class StarGambitGS : public GameState {
 public:
  using AS = ActionSpace<Config>;

  StarGambitGS();

  // Copy constructor
  StarGambitGS(const StarGambitGS& other) = default;

  [[nodiscard]] std::unique_ptr<GameState> copy() const noexcept override;

  [[nodiscard]] bool operator==(const GameState& other) const noexcept override;

  void hash(absl::HashState h) const override;

  [[nodiscard]] uint8_t current_player() const noexcept override {
    return current_player_;
  }

  [[nodiscard]] uint32_t current_turn() const noexcept override {
    return turn_;
  }

  [[nodiscard]] uint32_t num_moves() const noexcept override {
    return AS::NUM_MOVES;
  }

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

  void minimize_storage() override;

  // Helper methods
  [[nodiscard]] bool is_turn_one() const { return turn_ == 1 || turn_ == 2; }
  [[nodiscard]] bool has_taken_action() const { return has_taken_action_; }

  // Find unit by player, type, and slot (returns nullptr if not found/dead)
  [[nodiscard]] Unit* find_unit_by_slot(uint8_t player, UnitType type, uint8_t slot);
  [[nodiscard]] const Unit* find_unit_by_slot(uint8_t player, UnitType type, uint8_t slot) const;

  // Get next available slot for a unit type
  [[nodiscard]] int get_next_slot(uint8_t player, UnitType type) const;

  // Python binding support methods
  [[nodiscard]] std::vector<UnitInfo> get_units() const;
  [[nodiscard]] FireInfo get_fire_info(uint32_t move) const;

 private:
  // Movement computation
  MoveResult compute_fighter_move(const Unit& unit, int direction) const;
  MoveResult compute_cruiser_move(const Unit& unit, int direction) const;
  MoveResult compute_dreadnought_move(const Unit& unit, int direction) const;

  // Get all hexes currently occupied by any unit
  std::vector<Hex> get_all_occupied_hexes() const;

  // Check if a hex is occupied (optionally excluding a specific unit)
  bool is_hex_occupied(const Hex& h, int exclude_unit_idx = -1) const;

  // Find unit at hex (returns -1 if none)
  int find_unit_at_hex(const Hex& h) const;

  // Check if new position would collide (excluding the moving unit's current position)
  bool would_collide(const std::vector<Hex>& new_hexes, int exclude_unit_idx) const;

  // Movement validation
  bool is_fighter_move_valid(const Unit& unit, int direction) const;
  bool is_cruiser_move_valid(const Unit& unit, int direction) const;
  bool is_dreadnought_move_valid(const Unit& unit, int direction) const;

  // Fire validation - checks if cannon has target in range
  bool is_fire_valid(const Unit& unit, int cannon_idx) const;
  bool has_target_in_range(const Unit& unit, int cannon_idx) const;

  // Deploy validation
  bool is_deploy_valid(UnitType type, int facing) const;

  // End turn validation
  bool is_end_turn_valid() const;

  // Execute movement
  void execute_fighter_move(int slot, int direction);
  void execute_cruiser_move(int slot, int direction);
  void execute_dreadnought_move(int slot, int direction);

  // Execute fire (returns target info for display, empty if no hit)
  struct FireResult {
    bool hit;
    int target_unit_idx;
    int damage;
    int range;
  };
  FireResult execute_fire(const Unit& unit, int cannon_idx);

  // Execute deploy
  void execute_deploy(UnitType type, int facing);

  // Execute end turn
  void execute_end_turn();

  // Apply damage to a unit, returns true if destroyed
  bool apply_damage(int unit_idx, int damage);

  // Reset per-turn state for all units of current player
  void reset_turn_state();

  // Check for game end conditions
  void check_game_end();

  // Check for threefold repetition (adds position to history, returns true if draw)
  bool check_repetition();

  // Compute position hash for threefold repetition
  uint64_t compute_position_hash() const;

  // Game state
  std::vector<Unit> units_;
  std::array<std::array<uint8_t, NUM_UNIT_TYPES>, NUM_PLAYERS> reserves_;
  uint8_t current_player_{0};
  uint32_t turn_{1};
  bool has_taken_action_{false};
  bool game_over_{false};
  int8_t winner_{-1};  // -1 = no winner, 0 or 1 = player won, 2 = draw

  // Position history for threefold repetition
  std::vector<uint64_t> position_history_;
};

// ============================================================================
// Type Aliases for Each Game Size
// ============================================================================

using StarGambitSkirmishGS = StarGambitGS<SkirmishConfig>;
using StarGambitClashGS = StarGambitGS<ClashConfig>;
using StarGambitBattleGS = StarGambitGS<BattleConfig>;

// For backward compatibility, keep the old name pointing to Skirmish
using StarGambitGS_Legacy = StarGambitSkirmishGS;

// ============================================================================
// Explicit Template Instantiation Declarations
// ============================================================================

extern template class StarGambitGS<SkirmishConfig>;
extern template class StarGambitGS<ClashConfig>;
extern template class StarGambitGS<BattleConfig>;

}  // namespace alphazero::star_gambit_gs
