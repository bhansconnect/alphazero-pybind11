#include "star_gambit_gs.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <sstream>

#include "color.h"

namespace alphazero::star_gambit_gs {

// =============================================================================
// Hex Utility Functions
// =============================================================================

Hex hex_add(const Hex& a, const Hex& b) {
  return {static_cast<int8_t>(a.q + b.q), static_cast<int8_t>(a.r + b.r)};
}

Hex hex_subtract(const Hex& a, const Hex& b) {
  return {static_cast<int8_t>(a.q - b.q), static_cast<int8_t>(a.r - b.r)};
}

Hex hex_scale(const Hex& h, int k) {
  return {static_cast<int8_t>(h.q * k), static_cast<int8_t>(h.r * k)};
}

int hex_distance(const Hex& a, const Hex& b) {
  int dq = a.q - b.q;
  int dr = a.r - b.r;
  return (std::abs(dq) + std::abs(dr) + std::abs(dq + dr)) / 2;
}

Hex hex_neighbor(const Hex& h, int direction) {
  return hex_add(h, HEX_DIRECTIONS[direction]);
}

bool hex_in_bounds(const Hex& h, int board_side) {
  int s = -h.q - h.r;
  return std::abs(h.q) < board_side && std::abs(h.r) < board_side &&
         std::abs(s) < board_side;
}

int hex_to_index(const Hex& h, int board_side) {
  int index = 0;
  for (int r = -(board_side - 1); r < board_side; ++r) {
    for (int q = -(board_side - 1); q < board_side; ++q) {
      int s = -q - r;
      if (std::abs(q) < board_side && std::abs(r) < board_side &&
          std::abs(s) < board_side) {
        if (q == h.q && r == h.r) {
          return index;
        }
        ++index;
      }
    }
  }
  return -1;
}

Hex index_to_hex(int index, int board_side) {
  int idx = 0;
  for (int r = -(board_side - 1); r < board_side; ++r) {
    for (int q = -(board_side - 1); q < board_side; ++q) {
      int s = -q - r;
      if (std::abs(q) < board_side && std::abs(r) < board_side &&
          std::abs(s) < board_side) {
        if (idx == index) {
          return {static_cast<int8_t>(q), static_cast<int8_t>(r)};
        }
        ++idx;
      }
    }
  }
  return {0, 0};
}

int compute_num_hexes(int board_side) {
  return 3 * board_side * board_side - 3 * board_side + 1;
}

// =============================================================================
// Unit Shape Functions
// =============================================================================

std::vector<Hex> get_unit_hexes(UnitType type, const Hex& anchor, int facing) {
  std::vector<Hex> hexes;
  hexes.push_back(anchor);

  switch (type) {
    case UnitType::FIGHTER:
      break;

    case UnitType::CRUISER: {
      // Anchor is FRONT, rear is computed in opposite direction
      int rear_dir = OPPOSITE_DIRECTION[facing];
      Hex rear = hex_neighbor(anchor, rear_dir);
      hexes.push_back(rear);
      break;
    }

    case UnitType::DREADNOUGHT: {
      int rear_dir = OPPOSITE_DIRECTION[facing];
      // Rear hexes: one at SW (index 1), one at W (index 2) for E-facing
      // Note: SW is rear-right from pilot view, W is rear-center
      Hex rear_sw = hex_neighbor(anchor, rotate_direction(rear_dir, 1));
      Hex rear_w = hex_neighbor(anchor, rear_dir);
      hexes.push_back(rear_sw);   // index 1 (SW for E-facing)
      hexes.push_back(rear_w);    // index 2 (W for E-facing)
      break;
    }

    case UnitType::PORTAL:
      break;
  }

  return hexes;
}

std::vector<Hex> get_portal_hexes(int player, int board_side) {
  std::vector<Hex> hexes;
  int max_coord = board_side - 1;

  if (player == 0) {
    // Player 0 at BOTTOM (positive r)
    Hex corner = {0, static_cast<int8_t>(max_coord)};
    hexes.push_back(corner);
    hexes.push_back({static_cast<int8_t>(1), static_cast<int8_t>(max_coord - 1)});
    hexes.push_back({static_cast<int8_t>(-1), static_cast<int8_t>(max_coord)});
  } else {
    // Player 1 at TOP (negative r)
    Hex corner = {0, static_cast<int8_t>(-max_coord)};
    hexes.push_back(corner);
    hexes.push_back({static_cast<int8_t>(-1), static_cast<int8_t>(-max_coord + 1)});
    hexes.push_back({static_cast<int8_t>(1), static_cast<int8_t>(-max_coord)});
  }

  return hexes;
}

Hex get_deploy_hex(int player, int board_side) {
  int max_coord = board_side - 1;
  if (player == 0) {
    // Player 0 deploys at BOTTOM
    return {0, static_cast<int8_t>(max_coord - 1)};
  } else {
    // Player 1 deploys at TOP
    return {0, static_cast<int8_t>(-max_coord + 1)};
  }
}

// Dreadnought deployment: one rear hex at deploy position, ship extends away
// Returns the direction from deploy hex to anchor hex for the given facing direction
// Returns -1 if the facing is not valid for that player
int get_dreadnought_anchor_dir(int player, int facing) {
  // P0 at bottom: valid facings E(0), NE(1), NW(2), W(3) - ship points toward center (up)
  // P1 at top: valid facings E(0), W(3), SW(4), SE(5) - ship points toward center (down)
  // anchor_dir is the direction from deploy hex to the anchor (front) hex
  static const int P0_ANCHOR_DIRS[6] = {1, 2, 2, 3, -1, -1};  // indexed by facing
  static const int P1_ANCHOR_DIRS[6] = {0, -1, -1, 4, 5, 5};  // indexed by facing

  if (player == 0) {
    return P0_ANCHOR_DIRS[facing];
  } else {
    return P1_ANCHOR_DIRS[facing];
  }
}

std::vector<int> get_valid_deploy_facings(UnitType type, int player) {
  std::vector<int> facings;

  if (type == UnitType::DREADNOUGHT) {
    // Dreadnoughts: one rear hex at deploy square, ship extends away from portal
    // P0 at bottom: E(0), NE(1), NW(2), W(3) - pointing up toward center
    // P1 at top: E(0), W(3), SW(4), SE(5) - pointing down toward center
    if (player == 0) {
      facings = {0, 1, 2, 3};
    } else {
      facings = {0, 3, 4, 5};
    }
  } else {
    // Fighters and Cruisers
    // P0 at bottom: NE(1), NW(2), W(3) - pointing up toward opponent
    // P1 at top: SW(4), SE(5), E(0) - pointing down toward opponent
    if (player == 0) {
      facings = {1, 2, 3};
    } else {
      facings = {4, 5, 0};
    }
  }

  return facings;
}

// =============================================================================
// Cannon Functions
// =============================================================================

std::vector<CannonInfo> get_cannon_info(UnitType type) {
  std::vector<CannonInfo> cannons;

  switch (type) {
    case UnitType::FIGHTER:
      cannons.push_back({0, 0});
      break;

    case UnitType::CRUISER:
      // Cannon order: left (+1), forward (0), right (-1) from anchor/front hex
      cannons.push_back({1, 0});   // Left cannon (counter-clockwise from facing)
      cannons.push_back({0, 0});   // Forward cannon
      cannons.push_back({-1, 0});  // Right cannon (clockwise from facing)
      break;

    case UnitType::DREADNOUGHT:
      // Cannon order: 0=rr, 1=fr, 2=fl, 3=rl (see Python DREAD_CANNON_NAMES)
      // hexes[1] = SW (rear-right from pilot view), hexes[2] = W (rear-left from pilot view)
      // Right cannons fire forward (facing), left cannons fire forward-left (facing+1)
      cannons.push_back({0, 1});   // Cannon 0 (rr): from SW hex, fires forward (facing)
      cannons.push_back({0, 0});   // Cannon 1 (fr): from anchor, fires forward (facing)
      cannons.push_back({1, 0});   // Cannon 2 (fl): from anchor, fires forward-left (facing+1)
      cannons.push_back({1, 2});   // Cannon 3 (rl): from W hex, fires forward-left (facing+1)
      break;

    case UnitType::PORTAL:
      break;
  }

  return cannons;
}

bool has_line_of_sight(const Hex& from, int direction, int distance,
                       const std::vector<Hex>& occupied_hexes) {
  Hex current = from;
  for (int i = 1; i < distance; ++i) {
    current = hex_neighbor(current, direction);
    for (const auto& occ : occupied_hexes) {
      if (occ == current) {
        return false;
      }
    }
  }
  return true;
}

// =============================================================================
// StarGambitGS Template Implementation
// =============================================================================

template<typename Config>
StarGambitGS<Config>::StarGambitGS() {
  // Initialize reserves
  reserves_[0] = {static_cast<uint8_t>(Config::STARTING_FIGHTERS),
                  static_cast<uint8_t>(Config::STARTING_CRUISERS),
                  static_cast<uint8_t>(Config::STARTING_DREADNOUGHTS), 0};
  reserves_[1] = {static_cast<uint8_t>(Config::STARTING_FIGHTERS),
                  static_cast<uint8_t>(Config::STARTING_CRUISERS),
                  static_cast<uint8_t>(Config::STARTING_DREADNOUGHTS), 0};

  // Create portals for both players
  auto p0_portal_hexes = get_portal_hexes(0, Config::BOARD_SIDE);
  auto p1_portal_hexes = get_portal_hexes(1, Config::BOARD_SIDE);

  Unit p0_portal;
  p0_portal.type = UnitType::PORTAL;
  p0_portal.player = 0;
  p0_portal.slot = 0;
  p0_portal.hp = PORTAL_HP;
  p0_portal.facing = 2;  // NW - pointing up toward opponent (P0 at bottom)
  p0_portal.anchor_q = p0_portal_hexes[0].q;
  p0_portal.anchor_r = p0_portal_hexes[0].r;
  p0_portal.moves_left = 0;
  p0_portal.cannons_fired = 0;
  units_.push_back(p0_portal);

  Unit p1_portal;
  p1_portal.type = UnitType::PORTAL;
  p1_portal.player = 1;
  p1_portal.slot = 0;
  p1_portal.hp = PORTAL_HP;
  p1_portal.facing = 5;  // SE - pointing down toward opponent (P1 at top)
  p1_portal.anchor_q = p1_portal_hexes[0].q;
  p1_portal.anchor_r = p1_portal_hexes[0].r;
  p1_portal.moves_left = 0;
  p1_portal.cannons_fired = 0;
  units_.push_back(p1_portal);

  position_history_.push_back(compute_position_hash());
}

template<typename Config>
std::unique_ptr<GameState> StarGambitGS<Config>::copy() const noexcept {
  return std::make_unique<StarGambitGS<Config>>(*this);
}

template<typename Config>
bool StarGambitGS<Config>::operator==(const GameState& other) const noexcept {
  const auto* o = dynamic_cast<const StarGambitGS<Config>*>(&other);
  if (!o) return false;

  if (current_player_ != o->current_player_) return false;
  if (units_.size() != o->units_.size()) return false;
  if (reserves_ != o->reserves_) return false;
  if (has_taken_action_ != o->has_taken_action_) return false;

  for (size_t i = 0; i < units_.size(); ++i) {
    const auto& u1 = units_[i];
    const auto& u2 = o->units_[i];
    if (u1.type != u2.type || u1.player != u2.player || u1.slot != u2.slot ||
        u1.hp != u2.hp || u1.facing != u2.facing ||
        u1.anchor_q != u2.anchor_q || u1.anchor_r != u2.anchor_r ||
        u1.moves_left != u2.moves_left || u1.cannons_fired != u2.cannons_fired) {
      return false;
    }
  }

  return true;
}

template<typename Config>
void StarGambitGS<Config>::hash(absl::HashState h) const {
  h = absl::HashState::combine(std::move(h), current_player_);
  h = absl::HashState::combine(std::move(h), has_taken_action_);

  for (const auto& unit : units_) {
    h = absl::HashState::combine(std::move(h), static_cast<uint8_t>(unit.type),
                                 unit.player, unit.slot, unit.hp, unit.facing,
                                 unit.anchor_q, unit.anchor_r, unit.moves_left,
                                 unit.cannons_fired);
  }

  for (int p = 0; p < NUM_PLAYERS; ++p) {
    for (int t = 0; t < NUM_UNIT_TYPES; ++t) {
      h = absl::HashState::combine(std::move(h), reserves_[p][t]);
    }
  }
}

template<typename Config>
Unit* StarGambitGS<Config>::find_unit_by_slot(uint8_t player, UnitType type, uint8_t slot) {
  for (auto& unit : units_) {
    if (unit.player == player && unit.type == type && unit.slot == slot && unit.is_alive()) {
      return &unit;
    }
  }
  return nullptr;
}

template<typename Config>
const Unit* StarGambitGS<Config>::find_unit_by_slot(uint8_t player, UnitType type, uint8_t slot) const {
  for (const auto& unit : units_) {
    if (unit.player == player && unit.type == type && unit.slot == slot && unit.is_alive()) {
      return &unit;
    }
  }
  return nullptr;
}

template<typename Config>
int StarGambitGS<Config>::get_next_slot(uint8_t player, UnitType type) const {
  int max_slot = -1;
  for (const auto& unit : units_) {
    if (unit.player == player && unit.type == type) {
      max_slot = std::max(max_slot, static_cast<int>(unit.slot));
    }
  }
  return max_slot + 1;
}

template<typename Config>
std::vector<Hex> StarGambitGS<Config>::get_all_occupied_hexes() const {
  std::vector<Hex> occupied;
  for (const auto& unit : units_) {
    if (!unit.is_alive()) continue;

    if (unit.type == UnitType::PORTAL) {
      auto portal_hexes = get_portal_hexes(unit.player, Config::BOARD_SIDE);
      for (const auto& h : portal_hexes) {
        occupied.push_back(h);
      }
    } else {
      auto unit_hexes = get_unit_hexes(unit.type, {unit.anchor_q, unit.anchor_r}, unit.facing);
      for (const auto& h : unit_hexes) {
        occupied.push_back(h);
      }
    }
  }
  return occupied;
}

template<typename Config>
bool StarGambitGS<Config>::is_hex_occupied(const Hex& h, int exclude_unit_idx) const {
  for (size_t i = 0; i < units_.size(); ++i) {
    if (static_cast<int>(i) == exclude_unit_idx) continue;
    const auto& unit = units_[i];
    if (!unit.is_alive()) continue;

    std::vector<Hex> hexes;
    if (unit.type == UnitType::PORTAL) {
      hexes = get_portal_hexes(unit.player, Config::BOARD_SIDE);
    } else {
      hexes = get_unit_hexes(unit.type, {unit.anchor_q, unit.anchor_r}, unit.facing);
    }

    for (const auto& uh : hexes) {
      if (uh == h) return true;
    }
  }
  return false;
}

template<typename Config>
int StarGambitGS<Config>::find_unit_at_hex(const Hex& h) const {
  for (size_t i = 0; i < units_.size(); ++i) {
    const auto& unit = units_[i];
    if (!unit.is_alive()) continue;

    std::vector<Hex> unit_hexes;
    if (unit.type == UnitType::PORTAL) {
      unit_hexes = get_portal_hexes(unit.player, Config::BOARD_SIDE);
    } else {
      unit_hexes = get_unit_hexes(unit.type, {unit.anchor_q, unit.anchor_r}, unit.facing);
    }

    for (const auto& uh : unit_hexes) {
      if (uh == h) {
        return static_cast<int>(i);
      }
    }
  }
  return -1;
}

template<typename Config>
bool StarGambitGS<Config>::would_collide(const std::vector<Hex>& new_hexes, int exclude_unit_idx) const {
  for (const auto& h : new_hexes) {
    if (!hex_in_bounds(h, Config::BOARD_SIDE)) return true;
    if (is_hex_occupied(h, exclude_unit_idx)) return true;
  }
  return false;
}

// =============================================================================
// Movement Computation Functions
// =============================================================================

template<typename Config>
MoveResult StarGambitGS<Config>::compute_fighter_move(const Unit& unit, int direction) const {
  MoveResult result;
  Hex current = {unit.anchor_q, unit.anchor_r};

  // Fighter moves: 0=forward, 1=forward-left, 2=forward-right
  int move_dir;
  switch (direction) {
    case 0:  // Forward
      move_dir = unit.facing;
      break;
    case 1:  // Forward-left (counterclockwise from facing)
      move_dir = rotate_direction(unit.facing, 1);
      break;
    case 2:  // Forward-right (clockwise from facing)
      move_dir = rotate_direction(unit.facing, -1);
      break;
    default:
      result.valid = false;
      return result;
  }

  result.new_anchor = hex_neighbor(current, move_dir);
  result.new_facing = move_dir;  // Face direction of movement
  result.valid = hex_in_bounds(result.new_anchor, Config::BOARD_SIDE);

  return result;
}

template<typename Config>
MoveResult StarGambitGS<Config>::compute_cruiser_move(const Unit& unit, int direction) const {
  MoveResult result;
  // NOTE: For cruisers, anchor IS the front hex (rear is computed in opposite direction)
  Hex current_anchor = {unit.anchor_q, unit.anchor_r};

  // Cruiser moves: 0=rotate-left, 1=fwd-left, 2=forward, 3=fwd-right, 4=rotate-right
  switch (direction) {
    case 0:  // Rotate left: rear stays in place, front pivots
      {
        // Find current rear position
        int rear_dir = OPPOSITE_DIRECTION[unit.facing];
        Hex current_rear = hex_neighbor(current_anchor, rear_dir);
        // New facing
        result.new_facing = rotate_direction(unit.facing, 1);
        // New anchor (front) is in the new facing direction from the stationary rear
        result.new_anchor = hex_neighbor(current_rear, result.new_facing);
        result.valid = true;
      }
      break;
    case 1:  // Forward-left: front moves to forward-left, rear moves to old front
      result.new_facing = rotate_direction(unit.facing, 1);
      result.new_anchor = hex_neighbor(current_anchor, result.new_facing);
      result.valid = true;
      break;
    case 2:  // Forward (straight): front moves forward, rear moves to old front
      result.new_facing = unit.facing;
      result.new_anchor = hex_neighbor(current_anchor, result.new_facing);
      result.valid = true;
      break;
    case 3:  // Forward-right: front moves to forward-right, rear moves to old front
      result.new_facing = rotate_direction(unit.facing, -1);
      result.new_anchor = hex_neighbor(current_anchor, result.new_facing);
      result.valid = true;
      break;
    case 4:  // Rotate right: rear stays in place, front pivots
      {
        // Find current rear position
        int rear_dir = OPPOSITE_DIRECTION[unit.facing];
        Hex current_rear = hex_neighbor(current_anchor, rear_dir);
        // New facing
        result.new_facing = rotate_direction(unit.facing, -1);
        // New anchor (front) is in the new facing direction from the stationary rear
        result.new_anchor = hex_neighbor(current_rear, result.new_facing);
        result.valid = true;
      }
      break;
    default:
      result.valid = false;
      return result;
  }

  // Check all new hexes are in bounds
  auto new_hexes = get_unit_hexes(UnitType::CRUISER, result.new_anchor, result.new_facing);
  for (const auto& h : new_hexes) {
    if (!hex_in_bounds(h, Config::BOARD_SIDE)) {
      result.valid = false;
      break;
    }
  }

  return result;
}

template<typename Config>
MoveResult StarGambitGS<Config>::compute_dreadnought_move(const Unit& unit, int direction) const {
  MoveResult result;
  Hex current_anchor = {unit.anchor_q, unit.anchor_r};

  // Dreadnought moves: 0=left (pivot), 1=fwd-left (slide), 2=fwd-right (slide), 3=right (pivot)
  switch (direction) {
    case 0:  // Left: pivot around rear-left hex (at rear_dir, which is W for E-facing)
      {
        int rear_dir = OPPOSITE_DIRECTION[unit.facing];
        Hex rear_left = hex_neighbor(current_anchor, rear_dir);

        // Anchor rotates counterclockwise around rear_left
        int anchor_dir_from_pivot = OPPOSITE_DIRECTION[rear_dir];
        int new_anchor_dir = rotate_direction(anchor_dir_from_pivot, 1);
        result.new_anchor = hex_neighbor(rear_left, new_anchor_dir);
        result.new_facing = rotate_direction(unit.facing, 1);
        result.valid = true;
      }
      break;
    case 1:  // Forward-left: slide in forward-left direction (same as fl cannon)
      result.new_anchor = hex_neighbor(current_anchor, rotate_direction(unit.facing, 1));
      result.new_facing = unit.facing;  // No rotation
      result.valid = true;
      break;
    case 2:  // Forward-right: slide in forward direction (same as fr cannon)
      result.new_anchor = hex_neighbor(current_anchor, unit.facing);
      result.new_facing = unit.facing;  // No rotation
      result.valid = true;
      break;
    case 3:  // Right: pivot around rear-right hex (at rotate_direction(rear_dir, 1), which is SW for E-facing)
      {
        int rear_dir = OPPOSITE_DIRECTION[unit.facing];
        int rear_right_dir = rotate_direction(rear_dir, 1);
        Hex rear_right = hex_neighbor(current_anchor, rear_right_dir);

        // Anchor rotates clockwise around rear_right
        int anchor_dir_from_pivot = OPPOSITE_DIRECTION[rear_right_dir];
        int new_anchor_dir = rotate_direction(anchor_dir_from_pivot, -1);
        result.new_anchor = hex_neighbor(rear_right, new_anchor_dir);
        result.new_facing = rotate_direction(unit.facing, -1);
        result.valid = true;
      }
      break;
    default:
      result.valid = false;
      return result;
  }

  // Check all new hexes are in bounds
  auto new_hexes = get_unit_hexes(UnitType::DREADNOUGHT, result.new_anchor, result.new_facing);
  for (const auto& h : new_hexes) {
    if (!hex_in_bounds(h, Config::BOARD_SIDE)) {
      result.valid = false;
      break;
    }
  }

  return result;
}

// =============================================================================
// Movement Validation
// =============================================================================

template<typename Config>
bool StarGambitGS<Config>::is_fighter_move_valid(const Unit& unit, int direction) const {
  if (direction < 0 || direction >= FIGHTER_MOVE_DIRS) return false;

  MoveResult move = compute_fighter_move(unit, direction);
  if (!move.valid) return false;

  // Find unit index for exclusion
  int unit_idx = -1;
  for (size_t i = 0; i < units_.size(); ++i) {
    if (&units_[i] == &unit) {
      unit_idx = static_cast<int>(i);
      break;
    }
  }

  // Check collision
  std::vector<Hex> new_hexes = {move.new_anchor};
  return !would_collide(new_hexes, unit_idx);
}

template<typename Config>
bool StarGambitGS<Config>::is_cruiser_move_valid(const Unit& unit, int direction) const {
  if (direction < 0 || direction >= CRUISER_MOVE_DIRS) return false;

  MoveResult move = compute_cruiser_move(unit, direction);
  if (!move.valid) return false;

  int unit_idx = -1;
  for (size_t i = 0; i < units_.size(); ++i) {
    if (&units_[i] == &unit) {
      unit_idx = static_cast<int>(i);
      break;
    }
  }

  auto new_hexes = get_unit_hexes(UnitType::CRUISER, move.new_anchor, move.new_facing);
  return !would_collide(new_hexes, unit_idx);
}

template<typename Config>
bool StarGambitGS<Config>::is_dreadnought_move_valid(const Unit& unit, int direction) const {
  if (direction < 0 || direction >= DREAD_MOVE_DIRS) return false;

  MoveResult move = compute_dreadnought_move(unit, direction);
  if (!move.valid) return false;

  int unit_idx = -1;
  for (size_t i = 0; i < units_.size(); ++i) {
    if (&units_[i] == &unit) {
      unit_idx = static_cast<int>(i);
      break;
    }
  }

  auto new_hexes = get_unit_hexes(UnitType::DREADNOUGHT, move.new_anchor, move.new_facing);
  return !would_collide(new_hexes, unit_idx);
}

// =============================================================================
// Fire Validation
// =============================================================================

template<typename Config>
bool StarGambitGS<Config>::has_target_in_range(const Unit& unit, int cannon_idx) const {
  auto cannons = get_cannon_info(unit.type);
  if (cannon_idx >= static_cast<int>(cannons.size())) return false;

  const auto& cannon = cannons[cannon_idx];

  std::vector<Hex> unit_hexes;
  if (unit.type == UnitType::PORTAL) {
    unit_hexes = get_portal_hexes(unit.player, Config::BOARD_SIDE);
  } else {
    unit_hexes = get_unit_hexes(unit.type, {unit.anchor_q, unit.anchor_r}, unit.facing);
  }

  if (cannon.source_hex_idx >= static_cast<int>(unit_hexes.size())) return false;
  Hex source = unit_hexes[cannon.source_hex_idx];

  int fire_direction = rotate_direction(unit.facing, cannon.direction_offset);
  auto all_occupied = get_all_occupied_hexes();

  // Check range 1 and range 2
  for (int range = 1; range <= 2; ++range) {
    Hex target = source;
    for (int i = 0; i < range; ++i) {
      target = hex_neighbor(target, fire_direction);
    }

    if (!hex_in_bounds(target, Config::BOARD_SIDE)) continue;

    if (!has_line_of_sight(source, fire_direction, range, all_occupied)) {
      break;  // Blocked
    }

    int target_unit_idx = find_unit_at_hex(target);
    if (target_unit_idx >= 0) {
      const auto& target_unit = units_[target_unit_idx];
      // Must be enemy unit
      if (target_unit.player != unit.player) {
        return true;
      }
    }
  }

  return false;
}

template<typename Config>
bool StarGambitGS<Config>::is_fire_valid(const Unit& unit, int cannon_idx) const {
  if (!unit.is_alive()) return false;
  if (unit.player != current_player_) return false;

  int num_cannons = get_num_cannons(unit.type);
  if (cannon_idx < 0 || cannon_idx >= num_cannons) return false;

  if (unit.cannons_fired & (1 << cannon_idx)) return false;

  // NEW: Check if there's a target in range
  return has_target_in_range(unit, cannon_idx);
}

template<typename Config>
bool StarGambitGS<Config>::is_deploy_valid(UnitType type, int facing) const {
  if (type == UnitType::PORTAL) return false;

  int type_idx = static_cast<int>(type);
  if (reserves_[current_player_][type_idx] == 0) return false;

  auto valid_facings = get_valid_deploy_facings(type, current_player_);
  if (std::find(valid_facings.begin(), valid_facings.end(), facing) ==
      valid_facings.end()) {
    return false;
  }

  Hex deploy_hex = get_deploy_hex(current_player_, Config::BOARD_SIDE);

  std::vector<Hex> new_unit_hexes;
  if (type == UnitType::DREADNOUGHT) {
    // For dreadnoughts, one rear hex goes at deploy position, anchor is offset
    int anchor_dir = get_dreadnought_anchor_dir(current_player_, facing);
    Hex anchor = hex_neighbor(deploy_hex, anchor_dir);
    new_unit_hexes = get_unit_hexes(type, anchor, facing);
  } else if (type == UnitType::CRUISER) {
    // For cruisers, rear goes at deploy position, anchor (front) is offset
    // Anchor is one hex in the facing direction from deploy_hex
    Hex anchor = hex_neighbor(deploy_hex, facing);
    new_unit_hexes = get_unit_hexes(type, anchor, facing);
  } else {
    new_unit_hexes = get_unit_hexes(type, deploy_hex, facing);
  }

  auto all_occupied = get_all_occupied_hexes();

  for (const auto& h : new_unit_hexes) {
    if (!hex_in_bounds(h, Config::BOARD_SIDE)) return false;
    if (std::find(all_occupied.begin(), all_occupied.end(), h) !=
        all_occupied.end()) {
      return false;  // Collision with occupied hex
    }
  }

  return true;
}

template<typename Config>
bool StarGambitGS<Config>::is_end_turn_valid() const {
  if (is_turn_one()) {
    return false;
  }
  return has_taken_action_;
}

// =============================================================================
// valid_moves() with hex-based action encoding
// =============================================================================

template<typename Config>
Vector<uint8_t> StarGambitGS<Config>::valid_moves() const noexcept {
  Vector<uint8_t> valids(AS::NUM_MOVES);
  valids.setZero();

  if (game_over_) {
    return valids;
  }

  // Spatial actions for movement and firing with canonicalization
  // P1's positions are rotated 180° in canonical form
  const bool is_p1 = (current_player_ == 1);
  constexpr int BOARD_DIM = AS::BOARD_DIM;

  // Helper to encode action with canonicalization
  auto encode_action = [&](int row, int col, int slot) {
    if (is_p1) {
      // 180° rotation: (row, col) → (BOARD_DIM-1-row, BOARD_DIM-1-col)
      row = BOARD_DIM - 1 - row;
      col = BOARD_DIM - 1 - col;
      // Swap left/right slots
      slot = SLOT_MAP[slot];
    }
    return AS::encode_spatial_action(row, col, slot);
  };

  if (!is_turn_one()) {
    // Iterate through all units of current player
    for (const auto& unit : units_) {
      if (unit.player != current_player_ || !unit.is_alive()) continue;
      if (unit.type == UnitType::PORTAL) continue;

      // Get anchor hex and convert to 2D position
      Hex anchor = {unit.anchor_q, unit.anchor_r};
      auto [row, col] = hex_to_2d<Config::BOARD_SIDE>(anchor);

      // Set valid movement actions based on unit type
      if (unit.moves_left > 0) {
        switch (unit.type) {
          case UnitType::FIGHTER:
            // Fighter: forward, forward-left, forward-right (slots 0, 1, 2)
            if (is_fighter_move_valid(unit, static_cast<int>(FighterMove::FORWARD)))
              valids(encode_action(row, col, static_cast<int>(SpatialAction::MOVE_FORWARD))) = 1;
            if (is_fighter_move_valid(unit, static_cast<int>(FighterMove::FORWARD_LEFT)))
              valids(encode_action(row, col, static_cast<int>(SpatialAction::MOVE_FORWARD_LEFT))) = 1;
            if (is_fighter_move_valid(unit, static_cast<int>(FighterMove::FORWARD_RIGHT)))
              valids(encode_action(row, col, static_cast<int>(SpatialAction::MOVE_FORWARD_RIGHT))) = 1;
            break;

          case UnitType::CRUISER:
            // Cruiser: forward, forward-left, forward-right, rotate-left, rotate-right
            if (is_cruiser_move_valid(unit, static_cast<int>(CruiserMove::FORWARD)))
              valids(encode_action(row, col, static_cast<int>(SpatialAction::MOVE_FORWARD))) = 1;
            if (is_cruiser_move_valid(unit, static_cast<int>(CruiserMove::FORWARD_LEFT)))
              valids(encode_action(row, col, static_cast<int>(SpatialAction::MOVE_FORWARD_LEFT))) = 1;
            if (is_cruiser_move_valid(unit, static_cast<int>(CruiserMove::FORWARD_RIGHT)))
              valids(encode_action(row, col, static_cast<int>(SpatialAction::MOVE_FORWARD_RIGHT))) = 1;
            if (is_cruiser_move_valid(unit, static_cast<int>(CruiserMove::ROTATE_LEFT)))
              valids(encode_action(row, col, static_cast<int>(SpatialAction::ROTATE_LEFT))) = 1;
            if (is_cruiser_move_valid(unit, static_cast<int>(CruiserMove::ROTATE_RIGHT)))
              valids(encode_action(row, col, static_cast<int>(SpatialAction::ROTATE_RIGHT))) = 1;
            break;

          case UnitType::DREADNOUGHT:
            // Dreadnought: forward-left, forward-right, rotate-left, rotate-right (no forward)
            if (is_dreadnought_move_valid(unit, static_cast<int>(DreadnoughtMove::FORWARD_LEFT)))
              valids(encode_action(row, col, static_cast<int>(SpatialAction::MOVE_FORWARD_LEFT))) = 1;
            if (is_dreadnought_move_valid(unit, static_cast<int>(DreadnoughtMove::FORWARD_RIGHT)))
              valids(encode_action(row, col, static_cast<int>(SpatialAction::MOVE_FORWARD_RIGHT))) = 1;
            if (is_dreadnought_move_valid(unit, static_cast<int>(DreadnoughtMove::ROTATE_LEFT)))
              valids(encode_action(row, col, static_cast<int>(SpatialAction::ROTATE_LEFT))) = 1;
            if (is_dreadnought_move_valid(unit, static_cast<int>(DreadnoughtMove::ROTATE_RIGHT)))
              valids(encode_action(row, col, static_cast<int>(SpatialAction::ROTATE_RIGHT))) = 1;
            break;

          default:
            break;
        }
      }

      // Set valid fire actions based on unit type
      switch (unit.type) {
        case UnitType::FIGHTER:
          // Fighter: fire forward (slot 5)
          if (is_fire_valid(unit, 0))  // Cannon 0 = forward
            valids(encode_action(row, col, static_cast<int>(SpatialAction::FIRE_FORWARD))) = 1;
          break;

        case UnitType::CRUISER:
          // Cruiser: fire forward, forward-left, forward-right (slots 5, 6, 7)
          // Cruiser cannons: 0=left(fl), 1=forward(f), 2=right(fr)
          if (is_fire_valid(unit, 1))  // Cannon 1 = forward
            valids(encode_action(row, col, static_cast<int>(SpatialAction::FIRE_FORWARD))) = 1;
          if (is_fire_valid(unit, 0))  // Cannon 0 = left (forward-left)
            valids(encode_action(row, col, static_cast<int>(SpatialAction::FIRE_FORWARD_LEFT))) = 1;
          if (is_fire_valid(unit, 2))  // Cannon 2 = right (forward-right)
            valids(encode_action(row, col, static_cast<int>(SpatialAction::FIRE_FORWARD_RIGHT))) = 1;
          break;

        case UnitType::DREADNOUGHT:
          // Dreadnought: fire forward-left, forward-right, rear-left, rear-right (slots 6-9)
          // Dread cannons: 0=rr, 1=fr, 2=fl, 3=rl
          if (is_fire_valid(unit, 2))  // Cannon 2 = fl (forward-left)
            valids(encode_action(row, col, static_cast<int>(SpatialAction::FIRE_FORWARD_LEFT))) = 1;
          if (is_fire_valid(unit, 1))  // Cannon 1 = fr (forward-right)
            valids(encode_action(row, col, static_cast<int>(SpatialAction::FIRE_FORWARD_RIGHT))) = 1;
          if (is_fire_valid(unit, 3))  // Cannon 3 = rl (rear-left)
            valids(encode_action(row, col, static_cast<int>(SpatialAction::FIRE_REAR_LEFT))) = 1;
          if (is_fire_valid(unit, 0))  // Cannon 0 = rr (rear-right)
            valids(encode_action(row, col, static_cast<int>(SpatialAction::FIRE_REAR_RIGHT))) = 1;
          break;

        default:
          break;
      }
    }
  }

  // Deploy actions: canonicalize facing for P1
  for (int type_idx = 0; type_idx < 3; ++type_idx) {
    UnitType type = static_cast<UnitType>(type_idx);
    for (int facing = 0; facing < 6; ++facing) {
      if (is_deploy_valid(type, facing)) {
        // Canonicalize facing for P1: rotate 180° (+3 mod 6)
        int canonical_facing = is_p1 ? ((facing + 3) % 6) : facing;
        valids(AS::encode_deploy(type_idx, canonical_facing)) = 1;
      }
    }
  }

  // End turn
  if (is_end_turn_valid()) {
    valids(AS::END_TURN_OFFSET) = 1;
  }

  return valids;
}

// =============================================================================
// Execute Movement
// =============================================================================

template<typename Config>
void StarGambitGS<Config>::execute_fighter_move(int slot, int direction) {
  Unit* unit = find_unit_by_slot(current_player_, UnitType::FIGHTER, slot);
  if (!unit) return;

  MoveResult move = compute_fighter_move(*unit, direction);
  if (!move.valid) return;

  unit->anchor_q = move.new_anchor.q;
  unit->anchor_r = move.new_anchor.r;
  unit->facing = static_cast<uint8_t>(move.new_facing);
  unit->moves_left--;
  has_taken_action_ = true;
}

template<typename Config>
void StarGambitGS<Config>::execute_cruiser_move(int slot, int direction) {
  Unit* unit = find_unit_by_slot(current_player_, UnitType::CRUISER, slot);
  if (!unit) return;

  MoveResult move = compute_cruiser_move(*unit, direction);
  if (!move.valid) return;

  unit->anchor_q = move.new_anchor.q;
  unit->anchor_r = move.new_anchor.r;
  unit->facing = static_cast<uint8_t>(move.new_facing);
  unit->moves_left--;
  has_taken_action_ = true;
}

template<typename Config>
void StarGambitGS<Config>::execute_dreadnought_move(int slot, int direction) {
  Unit* unit = find_unit_by_slot(current_player_, UnitType::DREADNOUGHT, slot);
  if (!unit) return;

  MoveResult move = compute_dreadnought_move(*unit, direction);
  if (!move.valid) return;

  unit->anchor_q = move.new_anchor.q;
  unit->anchor_r = move.new_anchor.r;
  unit->facing = static_cast<uint8_t>(move.new_facing);
  unit->moves_left--;
  has_taken_action_ = true;
}

// =============================================================================
// Execute Fire
// =============================================================================

template<typename Config>
typename StarGambitGS<Config>::FireResult StarGambitGS<Config>::execute_fire(
    const Unit& unit, int cannon_idx) {
  FireResult result = {false, -1, 0, 0};

  // Find the unit to modify
  Unit* mutable_unit = nullptr;
  int firing_unit_idx = -1;
  for (size_t i = 0; i < units_.size(); ++i) {
    if (&units_[i] == &unit) {
      mutable_unit = &units_[i];
      firing_unit_idx = static_cast<int>(i);
      break;
    }
  }
  if (!mutable_unit) return result;

  mutable_unit->cannons_fired |= (1 << cannon_idx);
  has_taken_action_ = true;

  auto cannons = get_cannon_info(unit.type);
  if (cannon_idx >= static_cast<int>(cannons.size())) return result;

  const auto& cannon = cannons[cannon_idx];

  std::vector<Hex> unit_hexes;
  if (unit.type == UnitType::PORTAL) {
    unit_hexes = get_portal_hexes(unit.player, Config::BOARD_SIDE);
  } else {
    unit_hexes = get_unit_hexes(unit.type, {unit.anchor_q, unit.anchor_r}, unit.facing);
  }

  if (cannon.source_hex_idx >= static_cast<int>(unit_hexes.size())) return result;
  Hex source = unit_hexes[cannon.source_hex_idx];

  int fire_direction = rotate_direction(unit.facing, cannon.direction_offset);
  auto all_occupied = get_all_occupied_hexes();

  for (int range = 1; range <= 2; ++range) {
    Hex target = source;
    for (int i = 0; i < range; ++i) {
      target = hex_neighbor(target, fire_direction);
    }

    if (!hex_in_bounds(target, Config::BOARD_SIDE)) continue;

    if (!has_line_of_sight(source, fire_direction, range, all_occupied)) {
      break;
    }

    int target_unit_idx = find_unit_at_hex(target);
    if (target_unit_idx >= 0 && target_unit_idx != firing_unit_idx) {
      int damage = (range == 1) ? 2 : 1;
      result.hit = true;
      result.target_unit_idx = target_unit_idx;
      result.damage = damage;
      result.range = range;

      bool destroyed = apply_damage(target_unit_idx, damage);
      if (destroyed) {
        check_game_end();
      }
      return result;
    }
  }

  return result;
}

// =============================================================================
// Execute Deploy
// =============================================================================

template<typename Config>
void StarGambitGS<Config>::execute_deploy(UnitType type, int facing) {
  // Clear position history - reserves changed, old positions can't repeat
  position_history_.clear();

  Hex deploy_hex = get_deploy_hex(current_player_, Config::BOARD_SIDE);

  Hex anchor;
  if (type == UnitType::DREADNOUGHT) {
    // For dreadnoughts, one rear hex goes at deploy position, anchor is offset
    int anchor_dir = get_dreadnought_anchor_dir(current_player_, facing);
    anchor = hex_neighbor(deploy_hex, anchor_dir);
  } else if (type == UnitType::CRUISER) {
    // For cruisers, rear goes at deploy position, anchor (front) is offset
    // Anchor is one hex in the facing direction from deploy_hex
    anchor = hex_neighbor(deploy_hex, facing);
  } else {
    anchor = deploy_hex;
  }

  Unit new_unit;
  new_unit.type = type;
  new_unit.player = current_player_;
  new_unit.slot = static_cast<uint8_t>(get_next_slot(current_player_, type));
  new_unit.hp = static_cast<uint8_t>(get_max_hp(type));
  new_unit.facing = static_cast<uint8_t>(facing);
  new_unit.anchor_q = anchor.q;
  new_unit.anchor_r = anchor.r;
  new_unit.moves_left = 0;
  new_unit.cannons_fired = (1 << get_num_cannons(type)) - 1;

  units_.push_back(new_unit);

  reserves_[current_player_][static_cast<int>(type)]--;

  execute_end_turn();
}

// =============================================================================
// play_move() with hex-based action decoding
// =============================================================================

template<typename Config>
void StarGambitGS<Config>::play_move(uint32_t move) {
  if (move < static_cast<uint32_t>(AS::DEPLOY_OFFSET)) {
    // Spatial action: decode (row, col, action_type) with de-canonicalization
    int row, col, action_type;
    AS::decode_spatial_action(static_cast<int>(move), row, col, action_type);

    // De-canonicalize for P1
    const bool is_p1 = (current_player_ == 1);
    constexpr int BOARD_DIM = AS::BOARD_DIM;
    if (is_p1) {
      // Reverse 180° rotation
      row = BOARD_DIM - 1 - row;
      col = BOARD_DIM - 1 - col;
      // Reverse slot swap (SLOT_MAP is self-inverse)
      action_type = SLOT_MAP[action_type];
    }

    // Convert (row, col) to hex coordinates
    int q = row - (Config::BOARD_SIDE - 1);
    int r = col - (Config::BOARD_SIDE - 1);
    Hex anchor = {static_cast<int8_t>(q), static_cast<int8_t>(r)};

    // Find unit at this anchor hex
    Unit* unit = nullptr;
    for (auto& u : units_) {
      if (u.player == current_player_ && u.is_alive() &&
          u.anchor_q == anchor.q && u.anchor_r == anchor.r &&
          u.type != UnitType::PORTAL) {
        unit = &u;
        break;
      }
    }
    if (!unit) return;  // No unit at this hex

    // Execute action based on action slot and unit type
    SpatialAction action = static_cast<SpatialAction>(action_type);
    switch (action) {
      case SpatialAction::MOVE_FORWARD:
        if (unit->type == UnitType::FIGHTER) {
          execute_fighter_move(unit->slot, static_cast<int>(FighterMove::FORWARD));
        } else if (unit->type == UnitType::CRUISER) {
          execute_cruiser_move(unit->slot, static_cast<int>(CruiserMove::FORWARD));
        }
        // Dreadnought cannot move forward
        break;

      case SpatialAction::MOVE_FORWARD_LEFT:
        if (unit->type == UnitType::FIGHTER) {
          execute_fighter_move(unit->slot, static_cast<int>(FighterMove::FORWARD_LEFT));
        } else if (unit->type == UnitType::CRUISER) {
          execute_cruiser_move(unit->slot, static_cast<int>(CruiserMove::FORWARD_LEFT));
        } else if (unit->type == UnitType::DREADNOUGHT) {
          execute_dreadnought_move(unit->slot, static_cast<int>(DreadnoughtMove::FORWARD_LEFT));
        }
        break;

      case SpatialAction::MOVE_FORWARD_RIGHT:
        if (unit->type == UnitType::FIGHTER) {
          execute_fighter_move(unit->slot, static_cast<int>(FighterMove::FORWARD_RIGHT));
        } else if (unit->type == UnitType::CRUISER) {
          execute_cruiser_move(unit->slot, static_cast<int>(CruiserMove::FORWARD_RIGHT));
        } else if (unit->type == UnitType::DREADNOUGHT) {
          execute_dreadnought_move(unit->slot, static_cast<int>(DreadnoughtMove::FORWARD_RIGHT));
        }
        break;

      case SpatialAction::ROTATE_LEFT:
        if (unit->type == UnitType::CRUISER) {
          execute_cruiser_move(unit->slot, static_cast<int>(CruiserMove::ROTATE_LEFT));
        } else if (unit->type == UnitType::DREADNOUGHT) {
          execute_dreadnought_move(unit->slot, static_cast<int>(DreadnoughtMove::ROTATE_LEFT));
        }
        // Fighter cannot rotate
        break;

      case SpatialAction::ROTATE_RIGHT:
        if (unit->type == UnitType::CRUISER) {
          execute_cruiser_move(unit->slot, static_cast<int>(CruiserMove::ROTATE_RIGHT));
        } else if (unit->type == UnitType::DREADNOUGHT) {
          execute_dreadnought_move(unit->slot, static_cast<int>(DreadnoughtMove::ROTATE_RIGHT));
        }
        // Fighter cannot rotate
        break;

      case SpatialAction::FIRE_FORWARD:
        if (unit->type == UnitType::FIGHTER) {
          execute_fire(*unit, 0);  // Cannon 0 = forward
        } else if (unit->type == UnitType::CRUISER) {
          execute_fire(*unit, 1);  // Cannon 1 = forward
        }
        // Dreadnought cannot fire forward
        break;

      case SpatialAction::FIRE_FORWARD_LEFT:
        if (unit->type == UnitType::CRUISER) {
          execute_fire(*unit, 0);  // Cannon 0 = left (forward-left)
        } else if (unit->type == UnitType::DREADNOUGHT) {
          execute_fire(*unit, 2);  // Cannon 2 = fl (forward-left)
        }
        // Fighter cannot fire forward-left
        break;

      case SpatialAction::FIRE_FORWARD_RIGHT:
        if (unit->type == UnitType::CRUISER) {
          execute_fire(*unit, 2);  // Cannon 2 = right (forward-right)
        } else if (unit->type == UnitType::DREADNOUGHT) {
          execute_fire(*unit, 1);  // Cannon 1 = fr (forward-right)
        }
        // Fighter cannot fire forward-right
        break;

      case SpatialAction::FIRE_REAR_LEFT:
        if (unit->type == UnitType::DREADNOUGHT) {
          execute_fire(*unit, 3);  // Cannon 3 = rl (rear-left)
        }
        // Only dreadnought has rear cannons
        break;

      case SpatialAction::FIRE_REAR_RIGHT:
        if (unit->type == UnitType::DREADNOUGHT) {
          execute_fire(*unit, 0);  // Cannon 0 = rr (rear-right)
        }
        // Only dreadnought has rear cannons
        break;
    }

    // After any spatial action (move or fire), check for mid-turn threefold repetition
    if (check_repetition()) {
      return;  // Game ended due to repetition
    }
  } else if (move < static_cast<uint32_t>(AS::END_TURN_OFFSET)) {
    // Deploy action: de-canonicalize facing for P1
    int type_idx, facing;
    AS::decode_deploy(static_cast<int>(move), type_idx, facing);
    // De-canonicalize facing for P1: rotate 180° (+3 mod 6)
    const bool is_p1 = (current_player_ == 1);
    if (is_p1) {
      facing = (facing + 3) % 6;
    }
    execute_deploy(static_cast<UnitType>(type_idx), facing);
  } else {
    // End turn
    execute_end_turn();
  }
}

// =============================================================================
// Other Methods
// =============================================================================

// Helper: Check for threefold repetition after any action
// Returns true if game ended due to repetition
template<typename Config>
bool StarGambitGS<Config>::check_repetition() {
  position_history_.push_back(compute_position_hash());

  uint64_t current_hash = position_history_.back();
  int count = 0;
  for (auto h : position_history_) {
    if (h == current_hash) count++;
  }
  if (count >= 3) {
    game_over_ = true;
    winner_ = 2;  // Draw
    return true;
  }
  return false;
}

template<typename Config>
void StarGambitGS<Config>::execute_end_turn() {
  // Switch player first - repetition check happens at start of new player's turn
  current_player_ = 1 - current_player_;
  turn_++;
  has_taken_action_ = false;

  if (turn_ > MAX_TURNS) {
    game_over_ = true;
    winner_ = 2;
    return;
  }

  // Check for threefold repetition at start of new player's turn
  if (check_repetition()) {
    return;
  }

  reset_turn_state();

  // Check if current player has no valid moves - they lose
  Vector<uint8_t> valids = valid_moves();
  if (valids.sum() == 0) {
    game_over_ = true;
    winner_ = 1 - current_player_;
    return;
  }
}

template<typename Config>
void StarGambitGS<Config>::reset_turn_state() {
  for (auto& unit : units_) {
    if (unit.player == current_player_ && unit.is_alive()) {
      unit.moves_left = static_cast<uint8_t>(get_max_moves(unit.type));
      unit.cannons_fired = 0;
    }
  }
}

template<typename Config>
bool StarGambitGS<Config>::apply_damage(int unit_idx, int damage) {
  Unit& unit = units_[unit_idx];
  if (damage >= unit.hp) {
    unit.hp = 0;
    return true;
  }
  unit.hp -= static_cast<uint8_t>(damage);
  return false;
}

template<typename Config>
void StarGambitGS<Config>::check_game_end() {
  for (const auto& unit : units_) {
    if (unit.type == UnitType::PORTAL && unit.hp == 0) {
      game_over_ = true;
      winner_ = 1 - unit.player;
      return;
    }
  }

  for (int player = 0; player < 2; ++player) {
    bool has_ships = false;
    for (const auto& unit : units_) {
      if (unit.player == player && unit.is_alive() &&
          unit.type != UnitType::PORTAL) {
        has_ships = true;
        break;
      }
    }
    bool has_reserves = false;
    for (int t = 0; t < 3; ++t) {
      if (reserves_[player][t] > 0) {
        has_reserves = true;
        break;
      }
    }
    if (!has_ships && !has_reserves) {
      game_over_ = true;
      winner_ = 1 - player;
      return;
    }
  }
}

template<typename Config>
std::optional<Vector<float>> StarGambitGS<Config>::scores() const noexcept {
  if (!game_over_) {
    return std::nullopt;
  }

  Vector<float> result(NUM_PLAYERS + 1);
  result.setZero();

  if (winner_ == 2) {
    result(2) = 1.0f;
  } else if (winner_ >= 0 && winner_ < NUM_PLAYERS) {
    result(winner_) = 1.0f;
  }

  return result;
}

template<typename Config>
uint64_t StarGambitGS<Config>::compute_position_hash() const {
  uint64_t hash = 0;
  hash ^= static_cast<uint64_t>(current_player_) * 0x9e3779b97f4a7c15ULL;

  for (const auto& unit : units_) {
    if (!unit.is_alive()) continue;
    uint64_t unit_hash = static_cast<uint64_t>(unit.type) ^
                         (static_cast<uint64_t>(unit.player) << 8) ^
                         (static_cast<uint64_t>(unit.slot) << 12) ^
                         (static_cast<uint64_t>(unit.hp) << 16) ^
                         (static_cast<uint64_t>(unit.facing) << 24) ^
                         (static_cast<uint64_t>(unit.anchor_q + 10) << 32) ^
                         (static_cast<uint64_t>(unit.anchor_r + 10) << 40);
    hash ^= unit_hash * 0x517cc1b727220a95ULL;
  }

  return hash;
}

template<typename Config>
Tensor<float, 3> StarGambitGS<Config>::canonicalized() const noexcept {
  constexpr int BOARD_DIM = AS::BOARD_DIM;
  Tensor<float, 3> tensor(AS::CANONICAL_SHAPE[0], BOARD_DIM, BOARD_DIM);
  tensor.setZero();

  // Canonicalization: always show from current player's perspective
  // When P1 to move: rotate board 180° and swap my/opponent
  const bool is_p1_to_move = (current_player_ == 1);
  const int my_player = current_player_;
  const int opp_player = 1 - current_player_;

  // Helper: get 2D position, rotating 180° for P1
  auto get_pos = [&](const Hex& h) -> std::pair<int, int> {
    if (is_p1_to_move) {
      // Rotate 180°: (q, r) → (-q, -r)
      Hex rotated = {static_cast<int8_t>(-h.q), static_cast<int8_t>(-h.r)};
      return hex_to_2d<Config::BOARD_SIDE>(rotated);
    }
    return hex_to_2d<Config::BOARD_SIDE>(h);
  };

  // Helper: rotate facing by 180° for P1
  auto get_facing = [&](int facing) -> int {
    if (is_p1_to_move) {
      return (facing + 3) % 6;
    }
    return facing;
  };

  // Helper: set value at hex position (with rotation)
  auto set_hex = [&](int channel, const Hex& h, float value = 1.0f) {
    auto [row, col] = get_pos(h);
    tensor(channel, row, col) = value;
  };

  // Helper: broadcast to all valid hexes
  auto broadcast = [&](int channel, float value) {
    for (int idx = 0; idx < AS::NUM_HEXES; ++idx) {
      Hex h = index_to_hex_fast<Config::BOARD_SIDE>(idx);
      auto [row, col] = hex_to_2d<Config::BOARD_SIDE>(h);
      tensor(channel, row, col) = value;
    }
  };

  int ch = 0;

  // ========================================
  // Channel 0: Valid hex mask
  // ========================================
  for (int idx = 0; idx < AS::NUM_HEXES; ++idx) {
    Hex h = index_to_hex_fast<Config::BOARD_SIDE>(idx);
    auto [row, col] = hex_to_2d<Config::BOARD_SIDE>(h);
    tensor(ch, row, col) = 1.0f;
  }
  ch++;  // ch = 1

  // ========================================
  // Channels 1-8: Type-based unit presence (4 types × 2: my/opponent)
  // Layout: My types [1-4], Opponent types [5-8]
  // Type order: Fighter=0, Cruiser=1, Dreadnought=2, Portal=3
  // ========================================
  for (const auto& unit : units_) {
    if (!unit.is_alive()) continue;

    int type_idx = static_cast<int>(unit.type);  // 0-3
    bool is_my_unit = (unit.player == my_player);
    int presence_ch = ch + (is_my_unit ? 0 : 4) + type_idx;

    std::vector<Hex> hexes;
    if (unit.type == UnitType::PORTAL) {
      hexes = get_portal_hexes(unit.player, Config::BOARD_SIDE);
    } else {
      hexes = get_unit_hexes(unit.type, {unit.anchor_q, unit.anchor_r}, unit.facing);
    }

    for (const auto& h : hexes) {
      set_hex(presence_ch, h, 1.0f);
    }
  }
  ch += 8;  // ch = 9

  // ========================================
  // Channels 9-14: Heading (6 directions, generic)
  // One-hot encoding at ALL unit hexes (broadcast to show unit cohesion)
  // ========================================
  for (const auto& unit : units_) {
    if (!unit.is_alive()) continue;
    if (unit.type == UnitType::PORTAL) continue;

    int rotated_facing = get_facing(unit.facing);
    int heading_ch = ch + rotated_facing;

    // Get all hexes for this unit and set heading on each
    std::vector<Hex> hexes = get_unit_hexes(unit.type, {unit.anchor_q, unit.anchor_r}, unit.facing);
    for (const auto& h : hexes) {
      set_hex(heading_ch, h, 1.0f);
    }
  }
  ch += 6;  // ch = 15

  // ========================================
  // Channel 15: HP (generic, normalized)
  // Broadcast to ALL unit hexes, value = hp / max_hp
  // ========================================
  for (const auto& unit : units_) {
    if (!unit.is_alive()) continue;

    float max_hp = static_cast<float>(get_max_hp(unit.type));
    float norm_hp = static_cast<float>(unit.hp) / max_hp;

    // Get all hexes for this unit
    std::vector<Hex> hexes;
    if (unit.type == UnitType::PORTAL) {
      hexes = get_portal_hexes(unit.player, Config::BOARD_SIDE);
    } else {
      hexes = get_unit_hexes(unit.type, {unit.anchor_q, unit.anchor_r}, unit.facing);
    }

    // Set HP on all occupied hexes
    for (const auto& h : hexes) {
      set_hex(ch, h, norm_hp);
    }
  }
  ch++;  // ch = 16

  // ========================================
  // Channel 16: Moves remaining (generic, normalized)
  // Broadcast to ALL unit hexes, value = moves_left / max_moves
  // ========================================
  for (const auto& unit : units_) {
    if (!unit.is_alive()) continue;
    if (unit.type == UnitType::PORTAL) continue;

    float max_moves = static_cast<float>(get_max_moves(unit.type));
    float norm_moves = (max_moves > 0) ? static_cast<float>(unit.moves_left) / max_moves : 0.0f;

    // Get all hexes for this unit and set moves on each
    std::vector<Hex> hexes = get_unit_hexes(unit.type, {unit.anchor_q, unit.anchor_r}, unit.facing);
    for (const auto& h : hexes) {
      set_hex(ch, h, norm_moves);
    }
  }
  ch++;  // ch = 17

  // ========================================
  // Channels 17-21: Cannons available (5 types, generic)
  // Broadcast to ALL unit hexes, 1.0 if unfired, 0.0 if fired or N/A
  // Cannon order: forward, forward-left, forward-right, rear-left, rear-right
  // ========================================
  for (const auto& unit : units_) {
    if (!unit.is_alive()) continue;
    if (unit.type == UnitType::PORTAL) continue;

    // Get all hexes for this unit
    std::vector<Hex> unit_hexes = get_unit_hexes(unit.type, {unit.anchor_q, unit.anchor_r}, unit.facing);
    auto cannons = get_cannon_info(unit.type);

    for (size_t cannon_idx = 0; cannon_idx < cannons.size(); ++cannon_idx) {
      int slot;
      switch (cannons[cannon_idx].direction_offset) {
        case 0: slot = 0; break;   // forward
        case 1: slot = 1; break;   // forward-left
        case -1: slot = 2; break;  // forward-right
        case 2: slot = 3; break;   // rear-left
        case -2: slot = 4; break;  // rear-right
        default: continue;
      }

      int cannon_ch = ch + slot;
      bool fired = (unit.cannons_fired >> cannon_idx) & 1;
      if (!fired) {
        // Broadcast cannon availability to all unit hexes
        for (const auto& h : unit_hexes) {
          set_hex(cannon_ch, h, 1.0f);
        }
      }
    }
  }
  ch += 5;  // ch = 22

  // ========================================
  // Channel 22: Has taken action (broadcast)
  // ========================================
  broadcast(ch, has_taken_action_ ? 1.0f : 0.0f);
  ch++;  // ch = 23

  // ========================================
  // Channel 23: Repetition count (broadcast)
  // Count how many times current position has been seen
  // 0.0 = never seen, 0.5 = seen once, 1.0 = seen twice or more
  // ========================================
  {
    uint64_t current_hash = compute_position_hash();
    int rep_count = 0;
    for (const auto& h : position_history_) {
      if (h == current_hash) rep_count++;
    }
    float rep_value = (rep_count == 0) ? 0.0f : (rep_count == 1) ? 0.5f : 1.0f;
    broadcast(ch, rep_value);
  }
  ch++;  // ch = 24

  // ========================================
  // Channels 24-26: My reserves (3 types, broadcast)
  // ========================================
  {
    float norm_f = (Config::STARTING_FIGHTERS > 0)
        ? static_cast<float>(reserves_[my_player][0]) / static_cast<float>(Config::STARTING_FIGHTERS)
        : 0.0f;
    float norm_c = (Config::STARTING_CRUISERS > 0)
        ? static_cast<float>(reserves_[my_player][1]) / static_cast<float>(Config::STARTING_CRUISERS)
        : 0.0f;
    float norm_d = (Config::STARTING_DREADNOUGHTS > 0)
        ? static_cast<float>(reserves_[my_player][2]) / static_cast<float>(Config::STARTING_DREADNOUGHTS)
        : 0.0f;

    broadcast(ch, norm_f);
    broadcast(ch + 1, norm_c);
    broadcast(ch + 2, norm_d);
    ch += 3;  // ch = 27
  }

  // ========================================
  // Channels 27-29: Opponent reserves (3 types, broadcast)
  // ========================================
  {
    float norm_f = (Config::STARTING_FIGHTERS > 0)
        ? static_cast<float>(reserves_[opp_player][0]) / static_cast<float>(Config::STARTING_FIGHTERS)
        : 0.0f;
    float norm_c = (Config::STARTING_CRUISERS > 0)
        ? static_cast<float>(reserves_[opp_player][1]) / static_cast<float>(Config::STARTING_CRUISERS)
        : 0.0f;
    float norm_d = (Config::STARTING_DREADNOUGHTS > 0)
        ? static_cast<float>(reserves_[opp_player][2]) / static_cast<float>(Config::STARTING_DREADNOUGHTS)
        : 0.0f;

    broadcast(ch, norm_f);
    broadcast(ch + 1, norm_c);
    broadcast(ch + 2, norm_d);
    ch += 3;  // ch = 30
  }

  // ========================================
  // Channel 30: My portal HP (broadcast)
  // ========================================
  {
    const Unit* portal = find_unit_by_slot(my_player, UnitType::PORTAL, 0);
    float norm_hp = 0.0f;
    if (portal && portal->is_alive()) {
      norm_hp = static_cast<float>(portal->hp) / static_cast<float>(PORTAL_HP);
    }
    broadcast(ch, norm_hp);
    ch++;  // ch = 31
  }

  // ========================================
  // Channel 31: Opponent portal HP (broadcast)
  // ========================================
  {
    const Unit* portal = find_unit_by_slot(opp_player, UnitType::PORTAL, 0);
    float norm_hp = 0.0f;
    if (portal && portal->is_alive()) {
      norm_hp = static_cast<float>(portal->hp) / static_cast<float>(PORTAL_HP);
    }
    broadcast(ch, norm_hp);
    ch++;  // ch = 32
  }

  return tensor;
}

template<typename Config>
std::vector<PlayHistory> StarGambitGS<Config>::symmetries(
    const PlayHistory& base) const noexcept {
  // Two symmetries: identity and NW-axis mirror
  // The 180° rotation is already exploited by the perspective canonicalization.
  // The NW-axis mirror reflects across the "forward toward opponent" direction.
  //
  // In hex coordinates: (q, r) → (-q, r+q)
  // In 2D array: (row, col) → (BOARD_DIM-1-row, row+col-(BOARD_SIDE-1))
  // Direction map: NW stays, SE stays, NE↔W, E↔SW
  std::vector<PlayHistory> syms;
  syms.push_back(base);  // Identity

  constexpr int BOARD_DIM = AS::BOARD_DIM;
  constexpr int BOARD_SIDE = Config::BOARD_SIDE;
  constexpr int NUM_CHANNELS = AS::CANONICAL_CHANNELS;

  // Create mirrored version
  PlayHistory mirrored;
  mirrored.v = base.v;  // Value unchanged

  // ========================================
  // Mirror observation tensor
  // NW-axis mirror: (row, col) → (BOARD_DIM-1-row, row+col-(BOARD_SIDE-1))
  // ========================================
  mirrored.canonical.resize(NUM_CHANNELS, BOARD_DIM, BOARD_DIM);
  mirrored.canonical.setZero();

  // First pass: apply position transformation to all channels
  for (int ch = 0; ch < NUM_CHANNELS; ++ch) {
    for (int row = 0; row < BOARD_DIM; ++row) {
      for (int col = 0; col < BOARD_DIM; ++col) {
        int new_row = (BOARD_DIM - 1) - row;
        int new_col = row + col - (BOARD_SIDE - 1);
        // Check bounds (some positions may map outside valid range)
        if (new_col >= 0 && new_col < BOARD_DIM) {
          mirrored.canonical(ch, new_row, new_col) = base.canonical(ch, row, col);
        }
      }
    }
  }

  // Second pass: permute facing channels 9-14 using MIRROR_DIRECTION_MAP
  // Direction d becomes direction MIRROR_DIRECTION_MAP[d]
  {
    Tensor<float, 3> facing_copy(6, BOARD_DIM, BOARD_DIM);
    for (int d = 0; d < 6; ++d) {
      for (int row = 0; row < BOARD_DIM; ++row) {
        for (int col = 0; col < BOARD_DIM; ++col) {
          facing_copy(d, row, col) = mirrored.canonical(9 + d, row, col);
        }
      }
    }
    // Remap: channel 9+d gets content from channel 9+MIRROR_DIRECTION_MAP[d]
    // (MIRROR_DIRECTION_MAP is self-inverse)
    for (int d = 0; d < 6; ++d) {
      int src_d = MIRROR_DIRECTION_MAP[d];
      for (int row = 0; row < BOARD_DIM; ++row) {
        for (int col = 0; col < BOARD_DIM; ++col) {
          mirrored.canonical(9 + d, row, col) = facing_copy(src_d, row, col);
        }
      }
    }
  }

  // Third pass: permute cannon channels 17-21 (swap L/R pairs)
  // Cannon slots: 0=forward, 1=forward-left, 2=forward-right, 3=rear-left, 4=rear-right
  // Mirror swaps L/R: [0, 2, 1, 4, 3]
  {
    constexpr int CANNON_MAP[5] = {0, 2, 1, 4, 3};
    Tensor<float, 3> cannon_copy(5, BOARD_DIM, BOARD_DIM);
    for (int c = 0; c < 5; ++c) {
      for (int row = 0; row < BOARD_DIM; ++row) {
        for (int col = 0; col < BOARD_DIM; ++col) {
          cannon_copy(c, row, col) = mirrored.canonical(17 + c, row, col);
        }
      }
    }
    for (int c = 0; c < 5; ++c) {
      int src_c = CANNON_MAP[c];
      for (int row = 0; row < BOARD_DIM; ++row) {
        for (int col = 0; col < BOARD_DIM; ++col) {
          mirrored.canonical(17 + c, row, col) = cannon_copy(src_c, row, col);
        }
      }
    }
  }

  // ========================================
  // Mirror policy vector
  // ========================================
  mirrored.pi.resize(1, AS::NUM_MOVES);  // Vector is 1xN matrix
  mirrored.pi.setZero();

  // Spatial actions: apply position transformation, swap L/R slots
  // Only transform valid hex positions; invalid positions use identity (copy as-is)
  for (int action = 0; action < AS::SPATIAL_ACTIONS; ++action) {
    int row, col, slot;
    AS::decode_spatial_action(action, row, col, slot);

    // Check if this position corresponds to a valid hex
    if (!AS::is_valid_hex_pos(row, col)) {
      // Invalid hex position - copy as-is (should be 0 probability anyway)
      mirrored.pi(action) = base.pi(action);
      continue;
    }

    // Apply NW-axis mirror position transformation
    int new_row = (BOARD_DIM - 1) - row;
    int new_col = row + col - (BOARD_SIDE - 1);
    // Swap L/R slots
    int new_slot = SLOT_MAP[slot];

    int new_action = AS::encode_spatial_action(new_row, new_col, new_slot);
    mirrored.pi(new_action) = base.pi(action);
  }

  // Deploy actions: use MIRROR_DIRECTION_MAP for F/C, DEPLOY_MIRROR_D for dreadnoughts
  // F/C use same mirror as spatial (NW axis), dreadnoughts need different axis
  for (int d = 0; d < AS::DEPLOY_ACTIONS; ++d) {
    int type_idx = d / 6;
    int facing = d % 6;
    // Dreadnoughts use different axis to preserve valid facings {0,1,2,3}
    int new_facing = (type_idx == 2) ? DEPLOY_MIRROR_D[facing]
                                     : MIRROR_DIRECTION_MAP[facing];
    int new_d = type_idx * 6 + new_facing;
    mirrored.pi(AS::DEPLOY_OFFSET + new_d) = base.pi(AS::DEPLOY_OFFSET + d);
  }

  // End turn: unchanged
  mirrored.pi(AS::END_TURN_OFFSET) = base.pi(AS::END_TURN_OFFSET);

  syms.push_back(mirrored);
  return syms;
}

template<typename Config>
std::string StarGambitGS<Config>::dump() const noexcept {
  std::ostringstream out;

  out << "Turn: " << turn_ << ", Player: " << static_cast<int>(current_player_)
      << (has_taken_action_ ? " (acted)" : "") << "\n";

  out << "P0 reserves: F=" << static_cast<int>(reserves_[0][0])
      << " C=" << static_cast<int>(reserves_[0][1])
      << " D=" << static_cast<int>(reserves_[0][2]) << "\n";
  out << "P1 reserves: F=" << static_cast<int>(reserves_[1][0])
      << " C=" << static_cast<int>(reserves_[1][1])
      << " D=" << static_cast<int>(reserves_[1][2]) << "\n";

  // === Rendering constants ===
  constexpr int HEX_WIDTH = 4;       // " F1 " or "  . "
  constexpr int H_GAP = 2;           // space for E/W arrows
  constexpr int H_STEP = HEX_WIDTH + H_GAP;  // 6 chars per hex
  constexpr int ROW_INDENT_SCALE = H_STEP / 2;  // 3 chars offset per row

  // Unicode arrows by direction index
  const char* ARROWS[6] = {"→", "↗", "↖", "←", "↙", "↘"};

  // === Build hex display map (piece IDs with colors) ===
  struct HexDisplayInfo {
    std::string id;     // "F1", "C2", etc.
    int player;         // for coloring
    bool has_unit;
  };
  std::map<int, HexDisplayInfo> hex_display;

  // === Build cannon arrows set: (source_hex_index, direction) ===
  std::set<std::pair<int, int>> cannon_arrows;

  for (const auto& unit : units_) {
    if (!unit.is_alive()) continue;

    // Build piece ID: type char + slot number
    char type_char;
    switch (unit.type) {
      case UnitType::FIGHTER: type_char = 'F'; break;
      case UnitType::CRUISER: type_char = 'C'; break;
      case UnitType::DREADNOUGHT: type_char = 'D'; break;
      case UnitType::PORTAL: type_char = 'P'; break;
    }
    std::string piece_id = std::string(1, type_char) + std::to_string(unit.slot + 1);

    // Get all hexes occupied by this unit
    std::vector<Hex> hexes;
    if (unit.type == UnitType::PORTAL) {
      hexes = get_portal_hexes(unit.player, Config::BOARD_SIDE);
    } else {
      hexes = get_unit_hexes(unit.type, {unit.anchor_q, unit.anchor_r}, unit.facing);
    }

    // Store display info for each occupied hex
    for (const auto& h : hexes) {
      int idx = hex_to_index_fast<Config::BOARD_SIDE>(h);
      if (idx >= 0) {
        hex_display[idx] = {piece_id, unit.player, true};
      }
    }

    // Get cannon info and compute firing directions
    if (unit.type != UnitType::PORTAL) {
      auto cannons = get_cannon_info(unit.type);
      for (const auto& cannon : cannons) {
        if (cannon.source_hex_idx < static_cast<int>(hexes.size())) {
          Hex source_hex = hexes[cannon.source_hex_idx];
          int fire_dir = rotate_direction(unit.facing, cannon.direction_offset);
          int source_idx = hex_to_index_fast<Config::BOARD_SIDE>(source_hex);
          if (source_idx >= 0) {
            cannon_arrows.insert({source_idx, fire_dir});
          }
        }
      }
    }
  }

  // === Helper to get valid hexes for a row ===
  auto get_row_hexes = [](int r, int board_side) {
    std::vector<Hex> row_hexes;
    for (int q = -(board_side - 1); q < board_side; ++q) {
      int s = -q - r;
      if (std::abs(q) < board_side && std::abs(r) < board_side &&
          std::abs(s) < board_side) {
        row_hexes.push_back({static_cast<int8_t>(q), static_cast<int8_t>(r)});
      }
    }
    return row_hexes;
  };

  // === Render hex content line ===
  auto render_hex_line = [&](int r) {
    std::vector<Hex> row_hexes = get_row_hexes(r, Config::BOARD_SIDE);
    if (row_hexes.empty()) return;

    int indent = std::abs(r) * ROW_INDENT_SCALE;

    // Check if first hex fires W (off the left edge)
    int first_idx = hex_to_index_fast<Config::BOARD_SIDE>(row_hexes[0]);
    bool first_fires_w = cannon_arrows.count({first_idx, 3}) > 0;
    if (first_fires_w && indent >= 2) {
      out << std::string(indent - 2, ' ') << " ←";
    } else if (first_fires_w) {
      out << "←" << std::string(indent > 0 ? indent - 1 : 0, ' ');
    } else {
      out << std::string(indent, ' ');
    }

    for (size_t i = 0; i < row_hexes.size(); ++i) {
      Hex h = row_hexes[i];
      int idx = hex_to_index_fast<Config::BOARD_SIDE>(h);

      // Render hex content (4 chars)
      if (hex_display.count(idx) && hex_display[idx].has_unit) {
        const auto& info = hex_display[idx];
        std::string colored_id;
        if (info.player == 0) {
          colored_id = color::Modifier(color::FG_RED).dump() + info.id +
                       color::Modifier(color::FG_DEFAULT).dump();
        } else {
          colored_id = color::Modifier(color::FG_BLUE).dump() + info.id +
                       color::Modifier(color::FG_DEFAULT).dump();
        }
        // Pad to 4 visible chars (ID is 2 chars)
        out << " " << colored_id << " ";
      } else {
        out << " .  ";  // 4 chars: space, dot, 2 spaces
      }

      // Render gap after this hex (2 chars) - contains E/W arrows
      if (i < row_hexes.size() - 1) {
        Hex next = row_hexes[i + 1];
        int next_idx = hex_to_index_fast<Config::BOARD_SIDE>(next);

        bool arrow_e = cannon_arrows.count({idx, 0}) > 0;       // current fires E
        bool arrow_w = cannon_arrows.count({next_idx, 3}) > 0;  // next fires W

        if (arrow_e && arrow_w) {
          out << "↔ ";  // both directions + space (2 visual chars)
        } else if (arrow_e) {
          out << "→ ";  // arrow + space (2 visual chars)
        } else if (arrow_w) {
          out << " ←";  // space + arrow (2 visual chars)
        } else {
          out << "  ";  // 2 spaces for empty gap
        }
      } else {
        // Last hex in row - check if it fires E (off the edge)
        bool arrow_e = cannon_arrows.count({idx, 0}) > 0;
        if (arrow_e) {
          out << "→";  // arrow pointing off-board
        }
      }
    }
    out << "\n";
  };

  // === Helper to calculate x position for a hex ===
  auto hex_to_x = [&](int q, int r) {
    int indent = std::abs(r) * ROW_INDENT_SCALE;
    auto row_hexes = get_row_hexes(r, Config::BOARD_SIDE);
    if (row_hexes.empty()) return -1;
    int first_q = row_hexes[0].q;
    return indent + (q - first_q) * H_STEP;
  };

  // === Render diagonal edge line between rows r_above and r_below ===
  auto render_diagonal_line = [&](int r_above, int r_below) {
    std::vector<Hex> hexes_above = get_row_hexes(r_above, Config::BOARD_SIDE);
    std::vector<Hex> hexes_below = get_row_hexes(r_below, Config::BOARD_SIDE);

    if (hexes_above.empty() && hexes_below.empty()) return;

    int indent_above = std::abs(r_above) * ROW_INDENT_SCALE;
    int indent_below = std::abs(r_below) * ROW_INDENT_SCALE;

    // Calculate line width needed (add extra space for off-board arrows)
    int width_above = hexes_above.empty() ? 0 :
        indent_above + static_cast<int>(hexes_above.size()) * H_STEP;
    int width_below = hexes_below.empty() ? 0 :
        indent_below + static_cast<int>(hexes_below.size()) * H_STEP;
    int line_width = std::max(width_above, width_below) + H_STEP * 2;  // Extra space for off-board arrows

    // Build the line as a vector of characters (using strings for multi-byte)
    std::vector<std::string> line(line_width, " ");

    // Helper to compute target_x, extrapolating for off-board hexes
    auto compute_target_x = [&](int source_x, int target_q, int source_r, int target_r) {
      int target_x = hex_to_x(target_q, target_r);
      if (target_x >= 0) return target_x;
      // Off-board: extrapolate based on indent change and q change
      int indent_source = std::abs(source_r) * ROW_INDENT_SCALE;
      int indent_target = std::abs(target_r) * ROW_INDENT_SCALE;
      int indent_diff = indent_target - indent_source;
      // Get source hex's q to compute q offset
      auto source_row = get_row_hexes(source_r, Config::BOARD_SIDE);
      if (source_row.empty()) return -1;
      int source_first_q = source_row[0].q;
      int source_q = source_first_q + (source_x - indent_source) / H_STEP;
      int q_diff = target_q - source_q;
      return source_x + indent_diff + q_diff * H_STEP;
    };

    // Place arrows from hexes_above firing down (SE=5, SW=4)
    // Arrow positioned at midpoint between source and target hex centers
    for (const auto& h : hexes_above) {
      int idx = hex_to_index_fast<Config::BOARD_SIDE>(h);

      if (cannon_arrows.count({idx, 5})) {  // SE: target is (q, r+1)
        int source_x = hex_to_x(h.q, r_above);
        if (source_x >= 0) {
          int target_x = compute_target_x(source_x, h.q, r_above, r_below);
          // Midpoint of hex centers: (source + HEX_WIDTH/2 + target + HEX_WIDTH/2) / 2
          int arrow_x = (source_x + target_x) / 2 + HEX_WIDTH / 2;
          if (arrow_x >= 0 && arrow_x < line_width) line[arrow_x] = ARROWS[5];
        }
      }
      if (cannon_arrows.count({idx, 4})) {  // SW: target is (q-1, r+1)
        int source_x = hex_to_x(h.q, r_above);
        if (source_x >= 0) {
          int target_x = compute_target_x(source_x, h.q - 1, r_above, r_below);
          int arrow_x = (source_x + target_x) / 2 + HEX_WIDTH / 2;
          if (arrow_x >= 0 && arrow_x < line_width) line[arrow_x] = ARROWS[4];
        }
      }
    }

    // Place arrows from hexes_below firing up (NE=1, NW=2)
    for (const auto& h : hexes_below) {
      int idx = hex_to_index_fast<Config::BOARD_SIDE>(h);

      if (cannon_arrows.count({idx, 1})) {  // NE: target is (q+1, r-1)
        int source_x = hex_to_x(h.q, r_below);
        if (source_x >= 0) {
          int target_x = compute_target_x(source_x, h.q + 1, r_below, r_above);
          int arrow_x = (source_x + target_x) / 2 + HEX_WIDTH / 2;
          if (arrow_x >= 0 && arrow_x < line_width) line[arrow_x] = ARROWS[1];
        }
      }
      if (cannon_arrows.count({idx, 2})) {  // NW: target is (q, r-1)
        int source_x = hex_to_x(h.q, r_below);
        if (source_x >= 0) {
          int target_x = compute_target_x(source_x, h.q, r_below, r_above);
          int arrow_x = (source_x + target_x) / 2 + HEX_WIDTH / 2;
          if (arrow_x >= 0 && arrow_x < line_width) line[arrow_x] = ARROWS[2];
        }
      }
    }

    // Output the line
    for (const auto& ch : line) {
      out << ch;
    }
    out << "\n";
  };

  // === Render the board ===
  out << "\nBoard:\n";

  int min_r = -(Config::BOARD_SIDE - 1);
  int max_r = Config::BOARD_SIDE - 1;

  for (int r = min_r; r <= max_r; ++r) {
    // Diagonal edge line above this row (between r-1 and r)
    if (r > min_r) {
      render_diagonal_line(r - 1, r);
    }

    // Hex content line for row r
    render_hex_line(r);
  }

  // Final diagonal edge line below last row
  render_diagonal_line(max_r, max_r + 1);

  if (game_over_) {
    out << "\nGame Over! ";
    if (winner_ == 2) {
      out << "Draw";
    } else {
      out << "Player " << winner_ << " wins!";
    }
    out << "\n";
  }

  return out.str();
}

template<typename Config>
void StarGambitGS<Config>::minimize_storage() {
  units_.erase(std::remove_if(units_.begin(), units_.end(),
                              [](const Unit& u) { return !u.is_alive(); }),
               units_.end());
  position_history_.clear();
}

// =============================================================================
// Python Binding Support Methods
// =============================================================================

template<typename Config>
std::vector<UnitInfo> StarGambitGS<Config>::get_units() const {
  std::vector<UnitInfo> result;
  for (const auto& unit : units_) {
    if (!unit.is_alive()) continue;
    UnitInfo info;
    info.player = unit.player;
    info.type = static_cast<int>(unit.type);
    info.slot = unit.slot;
    info.hp = unit.hp;
    info.anchor_q = unit.anchor_q;
    info.anchor_r = unit.anchor_r;
    info.facing = unit.facing;
    info.moves_left = unit.moves_left;
    result.push_back(info);
  }
  return result;
}

template<typename Config>
FireInfo StarGambitGS<Config>::get_fire_info(uint32_t move) const {
  using AS = ActionSpace<Config>;
  FireInfo result = {false, -1, -1, -1, 0};

  // Check if this is a spatial action (not deploy or end turn)
  if (move >= static_cast<uint32_t>(AS::DEPLOY_OFFSET)) {
    return result;  // Not a fire action
  }

  // Decode spatial action with de-canonicalization
  int row, col, action_type;
  AS::decode_spatial_action(static_cast<int>(move), row, col, action_type);

  // De-canonicalize for P1
  const bool is_p1 = (current_player_ == 1);
  constexpr int BOARD_DIM = AS::BOARD_DIM;
  if (is_p1) {
    row = BOARD_DIM - 1 - row;
    col = BOARD_DIM - 1 - col;
    action_type = SLOT_MAP[action_type];
  }

  SpatialAction action = static_cast<SpatialAction>(action_type);

  // Check if this is a fire action
  if (action < SpatialAction::FIRE_FORWARD || action > SpatialAction::FIRE_REAR_RIGHT) {
    return result;  // Not a fire action
  }

  // Convert (row, col) to hex
  int q = row - (Config::BOARD_SIDE - 1);
  int r = col - (Config::BOARD_SIDE - 1);
  Hex anchor = {static_cast<int8_t>(q), static_cast<int8_t>(r)};

  // Find unit at this anchor hex
  const Unit* unit = nullptr;
  for (const auto& u : units_) {
    if (u.player == current_player_ && u.is_alive() &&
        u.anchor_q == anchor.q && u.anchor_r == anchor.r &&
        u.type != UnitType::PORTAL) {
      unit = &u;
      break;
    }
  }

  if (!unit || !unit->is_alive()) {
    return result;
  }

  // Map action slot to cannon index based on unit type
  int cannon_idx = -1;
  switch (action) {
    case SpatialAction::FIRE_FORWARD:
      if (unit->type == UnitType::FIGHTER) cannon_idx = 0;
      else if (unit->type == UnitType::CRUISER) cannon_idx = 1;
      break;
    case SpatialAction::FIRE_FORWARD_LEFT:
      if (unit->type == UnitType::CRUISER) cannon_idx = 0;
      else if (unit->type == UnitType::DREADNOUGHT) cannon_idx = 2;
      break;
    case SpatialAction::FIRE_FORWARD_RIGHT:
      if (unit->type == UnitType::CRUISER) cannon_idx = 2;
      else if (unit->type == UnitType::DREADNOUGHT) cannon_idx = 1;
      break;
    case SpatialAction::FIRE_REAR_LEFT:
      if (unit->type == UnitType::DREADNOUGHT) cannon_idx = 3;
      break;
    case SpatialAction::FIRE_REAR_RIGHT:
      if (unit->type == UnitType::DREADNOUGHT) cannon_idx = 0;
      break;
    default:
      return result;
  }

  if (cannon_idx < 0) {
    return result;
  }

  // Find target (similar to execute_fire but without modifying state)
  auto cannons = get_cannon_info(unit->type);
  if (cannon_idx >= static_cast<int>(cannons.size())) return result;

  const auto& cannon = cannons[cannon_idx];
  std::vector<Hex> unit_hexes = get_unit_hexes(unit->type,
      {unit->anchor_q, unit->anchor_r}, unit->facing);

  if (cannon.source_hex_idx >= static_cast<int>(unit_hexes.size())) return result;
  Hex source = unit_hexes[cannon.source_hex_idx];

  int fire_direction = rotate_direction(unit->facing, cannon.direction_offset);
  auto all_occupied = get_all_occupied_hexes();

  for (int range = 1; range <= 2; ++range) {
    Hex target = source;
    for (int i = 0; i < range; ++i) {
      target = hex_neighbor(target, fire_direction);
    }

    if (!hex_in_bounds(target, Config::BOARD_SIDE)) continue;

    if (!has_line_of_sight(source, fire_direction, range, all_occupied)) {
      break;
    }

    int target_unit_idx = find_unit_at_hex(target);
    if (target_unit_idx >= 0) {
      const auto& target_unit = units_[target_unit_idx];
      if (target_unit.player != unit->player) {
        result.has_target = true;
        result.target_player = target_unit.player;
        result.target_type = static_cast<int>(target_unit.type);
        result.target_slot = target_unit.slot;
        result.damage = (range == 1) ? 2 : 1;
        return result;
      }
    }
  }

  return result;
}

// =============================================================================
// Explicit Template Instantiations
// =============================================================================

template class StarGambitGS<SkirmishConfig>;
template class StarGambitGS<ClashConfig>;
template class StarGambitGS<BattleConfig>;

}  // namespace alphazero::star_gambit_gs
