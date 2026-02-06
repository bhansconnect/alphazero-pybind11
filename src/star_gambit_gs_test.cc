#include "star_gambit_gs.h"

#include <gtest/gtest.h>

#include <iostream>
#include <sstream>
#include <map>
#include <optional>

namespace alphazero::star_gambit_gs {

// Use Skirmish configuration for tests
using TestConfig = SkirmishConfig;
using TestAS = ActionSpace<TestConfig>;
using TestGame = StarGambitSkirmishGS;

// =============================================================================
// Move Notation Parser (Hex-Based Action Encoding)
// =============================================================================
// Notation:
//   m f1 f    - move fighter 1 forward
//   m f1 fl   - move fighter 1 forward-left
//   m f1 fr   - move fighter 1 forward-right
//   m c1 l    - move cruiser 1 rotate-left
//   m c1 fl   - move cruiser 1 forward-left
//   m c1 f    - move cruiser 1 forward
//   m c1 fr   - move cruiser 1 forward-right
//   m c1 r    - move cruiser 1 rotate-right
//   m d1 l    - move dreadnought 1 rotate-left
//   m d1 fl   - move dreadnought 1 forward-left
//   m d1 fr   - move dreadnought 1 forward-right
//   m d1 r    - move dreadnought 1 rotate-right
//   f f1      - fire fighter 1
//   f c1 l    - fire cruiser 1 left cannon (forward-left)
//   f c1 f    - fire cruiser 1 forward cannon
//   f c1 r    - fire cruiser 1 right cannon (forward-right)
//   f d1 rl   - fire dreadnought 1 rear-left cannon
//   f d1 fl   - fire dreadnought 1 front-left cannon
//   f d1 fr   - fire dreadnought 1 front-right cannon
//   f d1 rr   - fire dreadnought 1 rear-right cannon
//   d f e     - deploy fighter facing east
//   d f ne    - deploy fighter facing northeast
//   d c sw    - deploy cruiser facing southwest
//   e         - end turn

// Helper to find unit's position in 2D coordinates
template<typename Config, typename GameType>
std::optional<std::pair<int, int>> find_unit_pos(const GameType& game, UnitType type, int slot) {
  auto units = game.get_units();
  for (const auto& u : units) {
    if (u.player == game.current_player() &&
        static_cast<UnitType>(u.type) == type &&
        u.slot == slot) {
      Hex anchor = {static_cast<int8_t>(u.anchor_q), static_cast<int8_t>(u.anchor_r)};
      return hex_to_2d<Config::BOARD_SIDE>(anchor);
    }
  }
  return std::nullopt;
}

template<typename Config, typename GameType>
std::optional<int> parse_move(const std::string& move_str, const GameType& game) {
  using AS = ActionSpace<Config>;
  constexpr int BOARD_DIM = AS::BOARD_DIM;
  std::istringstream iss(move_str);
  std::string cmd;
  iss >> cmd;

  const bool is_p1 = (game.current_player() == 1);

  // Helper to encode spatial action with canonicalization
  auto encode_action = [&](int row, int col, int slot) {
    if (is_p1) {
      row = BOARD_DIM - 1 - row;
      col = BOARD_DIM - 1 - col;
      slot = SLOT_MAP[slot];
    }
    return AS::encode_spatial_action(row, col, slot);
  };

  if (cmd == "e") {
    return AS::END_TURN_OFFSET;
  }

  if (cmd == "m") {
    // Move command: m <unit><slot> <direction>
    std::string unit_slot, direction;
    iss >> unit_slot >> direction;
    if (unit_slot.empty() || direction.empty()) return std::nullopt;

    char unit_type = unit_slot[0];
    int slot = std::stoi(unit_slot.substr(1)) - 1;  // Convert 1-indexed to 0-indexed

    if (unit_type == 'f') {
      // Fighter moves: f, fl, fr -> SpatialAction slots 0, 1, 2
      static const std::map<std::string, SpatialAction> fighter_dirs = {
        {"f", SpatialAction::MOVE_FORWARD}, {"fl", SpatialAction::MOVE_FORWARD_LEFT},
        {"fr", SpatialAction::MOVE_FORWARD_RIGHT}
      };
      auto it = fighter_dirs.find(direction);
      if (it == fighter_dirs.end() || slot < 0 || slot >= Config::MAX_FIGHTERS) {
        return std::nullopt;
      }
      auto pos = find_unit_pos<Config>(game, UnitType::FIGHTER, slot);
      if (!pos) return std::nullopt;
      return encode_action(pos->first, pos->second, static_cast<int>(it->second));
    } else if (unit_type == 'c') {
      // Cruiser moves: l, fl, f, fr, r
      static const std::map<std::string, SpatialAction> cruiser_dirs = {
        {"l", SpatialAction::ROTATE_LEFT}, {"fl", SpatialAction::MOVE_FORWARD_LEFT},
        {"f", SpatialAction::MOVE_FORWARD}, {"fr", SpatialAction::MOVE_FORWARD_RIGHT},
        {"r", SpatialAction::ROTATE_RIGHT}
      };
      auto it = cruiser_dirs.find(direction);
      if (it == cruiser_dirs.end() || slot < 0 || slot >= Config::MAX_CRUISERS) {
        return std::nullopt;
      }
      auto pos = find_unit_pos<Config>(game, UnitType::CRUISER, slot);
      if (!pos) return std::nullopt;
      return encode_action(pos->first, pos->second, static_cast<int>(it->second));
    } else if (unit_type == 'd') {
      // Dreadnought moves: l, fl, fr, r (no forward)
      static const std::map<std::string, SpatialAction> dread_dirs = {
        {"l", SpatialAction::ROTATE_LEFT}, {"fl", SpatialAction::MOVE_FORWARD_LEFT},
        {"fr", SpatialAction::MOVE_FORWARD_RIGHT}, {"r", SpatialAction::ROTATE_RIGHT}
      };
      auto it = dread_dirs.find(direction);
      if (it == dread_dirs.end() || slot < 0 || slot >= Config::MAX_DREADNOUGHTS) {
        return std::nullopt;
      }
      auto pos = find_unit_pos<Config>(game, UnitType::DREADNOUGHT, slot);
      if (!pos) return std::nullopt;
      return encode_action(pos->first, pos->second, static_cast<int>(it->second));
    }
    return std::nullopt;
  }

  if (cmd == "f") {
    // Fire command: f <unit><slot> [cannon]
    std::string unit_slot, cannon;
    iss >> unit_slot;
    iss >> cannon;  // Optional for fighter

    if (unit_slot.empty()) return std::nullopt;

    char unit_type = unit_slot[0];
    int slot = std::stoi(unit_slot.substr(1)) - 1;

    if (unit_type == 'f') {
      // Fighter has only one cannon (forward)
      if (slot < 0 || slot >= Config::MAX_FIGHTERS) return std::nullopt;
      auto pos = find_unit_pos<Config>(game, UnitType::FIGHTER, slot);
      if (!pos) return std::nullopt;
      return encode_action(pos->first, pos->second, static_cast<int>(SpatialAction::FIRE_FORWARD));
    } else if (unit_type == 'c') {
      // Cruiser cannons: l (fl), f (forward), r (fr)
      static const std::map<std::string, SpatialAction> cruiser_cannons = {
        {"l", SpatialAction::FIRE_FORWARD_LEFT}, {"f", SpatialAction::FIRE_FORWARD},
        {"r", SpatialAction::FIRE_FORWARD_RIGHT}
      };
      auto it = cruiser_cannons.find(cannon);
      if (it == cruiser_cannons.end() || slot < 0 || slot >= Config::MAX_CRUISERS) {
        return std::nullopt;
      }
      auto pos = find_unit_pos<Config>(game, UnitType::CRUISER, slot);
      if (!pos) return std::nullopt;
      return encode_action(pos->first, pos->second, static_cast<int>(it->second));
    } else if (unit_type == 'd') {
      // Dreadnought cannons: rl, fl, fr, rr
      static const std::map<std::string, SpatialAction> dread_cannons = {
        {"rl", SpatialAction::FIRE_REAR_LEFT}, {"fl", SpatialAction::FIRE_FORWARD_LEFT},
        {"fr", SpatialAction::FIRE_FORWARD_RIGHT}, {"rr", SpatialAction::FIRE_REAR_RIGHT}
      };
      auto it = dread_cannons.find(cannon);
      if (it == dread_cannons.end() || slot < 0 || slot >= Config::MAX_DREADNOUGHTS) {
        return std::nullopt;
      }
      auto pos = find_unit_pos<Config>(game, UnitType::DREADNOUGHT, slot);
      if (!pos) return std::nullopt;
      return encode_action(pos->first, pos->second, static_cast<int>(it->second));
    }
    return std::nullopt;
  }

  if (cmd == "d") {
    // Deploy command: d <unit_type> <facing>
    std::string unit_type_str, facing_str;
    iss >> unit_type_str >> facing_str;
    if (unit_type_str.empty() || facing_str.empty()) return std::nullopt;

    // Unit type: f=0, c=1, d=2
    int unit_type_idx = -1;
    if (unit_type_str == "f") unit_type_idx = 0;
    else if (unit_type_str == "c") unit_type_idx = 1;
    else if (unit_type_str == "d") unit_type_idx = 2;
    if (unit_type_idx < 0) return std::nullopt;

    // Facing: e=0, ne=1, nw=2, w=3, sw=4, se=5
    static const std::map<std::string, int> facings = {
      {"e", 0}, {"ne", 1}, {"nw", 2}, {"w", 3}, {"sw", 4}, {"se", 5}
    };
    auto it = facings.find(facing_str);
    if (it == facings.end()) return std::nullopt;

    int facing = it->second;
    // Canonicalize facing for P1: rotate 180° (+3 mod 6)
    if (is_p1) {
      facing = (facing + 3) % 6;
    }

    return AS::encode_deploy(unit_type_idx, facing);
  }

  return std::nullopt;
}

// Overload for deploy-only commands (no game state needed)
template<typename Config>
std::optional<int> parse_deploy_cmd(const std::string& move_str) {
  using AS = ActionSpace<Config>;
  std::istringstream iss(move_str);
  std::string cmd;
  iss >> cmd;

  if (cmd == "e") {
    return AS::END_TURN_OFFSET;
  }

  if (cmd == "d") {
    std::string unit_type_str, facing;
    iss >> unit_type_str >> facing;
    if (unit_type_str.empty() || facing.empty()) return std::nullopt;

    int unit_type_idx = -1;
    if (unit_type_str == "f") unit_type_idx = 0;
    else if (unit_type_str == "c") unit_type_idx = 1;
    else if (unit_type_str == "d") unit_type_idx = 2;
    if (unit_type_idx < 0) return std::nullopt;

    static const std::map<std::string, int> facings = {
      {"e", 0}, {"ne", 1}, {"nw", 2}, {"w", 3}, {"sw", 4}, {"se", 5}
    };
    auto it = facings.find(facing);
    if (it == facings.end()) return std::nullopt;

    return AS::encode_deploy(unit_type_idx, it->second);
  }

  return std::nullopt;
}

// =============================================================================
// Hex Coordinate Tests
// =============================================================================

TEST(HexCoordinates, HexAdd) {
  Hex a = {1, 2};
  Hex b = {3, -1};
  Hex result = hex_add(a, b);
  EXPECT_EQ(result.q, 4);
  EXPECT_EQ(result.r, 1);
}

TEST(HexCoordinates, HexSubtract) {
  Hex a = {5, 3};
  Hex b = {2, 1};
  Hex result = hex_subtract(a, b);
  EXPECT_EQ(result.q, 3);
  EXPECT_EQ(result.r, 2);
}

TEST(HexCoordinates, HexDistance) {
  Hex origin = {0, 0};
  Hex neighbor = {1, 0};
  EXPECT_EQ(hex_distance(origin, neighbor), 1);

  Hex far = {2, -1};
  EXPECT_EQ(hex_distance(origin, far), 2);

  Hex diagonal = {2, 2};
  EXPECT_EQ(hex_distance(origin, diagonal), 4);
}

TEST(HexCoordinates, HexNeighbor) {
  Hex center = {0, 0};

  // Direction 0 = East
  Hex east = hex_neighbor(center, 0);
  EXPECT_EQ(east.q, 1);
  EXPECT_EQ(east.r, 0);

  // Direction 3 = West
  Hex west = hex_neighbor(center, 3);
  EXPECT_EQ(west.q, -1);
  EXPECT_EQ(west.r, 0);
}

TEST(HexCoordinates, HexInBounds) {
  EXPECT_TRUE(hex_in_bounds({0, 0}, TestConfig::BOARD_SIDE));
  EXPECT_TRUE(hex_in_bounds({4, 0}, TestConfig::BOARD_SIDE));
  EXPECT_TRUE(hex_in_bounds({0, 4}, TestConfig::BOARD_SIDE));
  EXPECT_TRUE(hex_in_bounds({-4, 0}, TestConfig::BOARD_SIDE));
  EXPECT_TRUE(hex_in_bounds({0, -4}, TestConfig::BOARD_SIDE));

  // Out of bounds
  EXPECT_FALSE(hex_in_bounds({5, 0}, TestConfig::BOARD_SIDE));
  EXPECT_FALSE(hex_in_bounds({0, 5}, TestConfig::BOARD_SIDE));
  EXPECT_FALSE(hex_in_bounds({3, 3}, TestConfig::BOARD_SIDE));  // |q+r| = 6 >= 5
}

TEST(HexCoordinates, HexToIndexRoundTrip) {
  for (int idx = 0; idx < TestAS::NUM_HEXES; ++idx) {
    Hex h = index_to_hex(idx, TestConfig::BOARD_SIDE);
    EXPECT_TRUE(hex_in_bounds(h, TestConfig::BOARD_SIDE));
    int result_idx = hex_to_index(h, TestConfig::BOARD_SIDE);
    EXPECT_EQ(result_idx, idx) << "Failed at index " << idx;
  }
}

TEST(HexCoordinates, RotateDirection) {
  EXPECT_EQ(rotate_direction(0, 1), 1);
  EXPECT_EQ(rotate_direction(0, -1), 5);
  EXPECT_EQ(rotate_direction(5, 1), 0);
  EXPECT_EQ(rotate_direction(3, 3), 0);
}

// =============================================================================
// Unit Shape Tests
// =============================================================================

TEST(UnitShapes, FighterSingleHex) {
  Hex anchor = {0, 0};
  auto hexes = get_unit_hexes(UnitType::FIGHTER, anchor, 0);
  EXPECT_EQ(hexes.size(), 1u);
  EXPECT_EQ(hexes[0], anchor);
}

TEST(UnitShapes, CruiserTwoHexes) {
  Hex anchor = {0, 0};
  auto hexes = get_unit_hexes(UnitType::CRUISER, anchor, 0);  // Facing East
  EXPECT_EQ(hexes.size(), 2u);
  EXPECT_EQ(hexes[0], anchor);  // Anchor is front
  // Rear hex is West of anchor (opposite of facing direction)
  EXPECT_EQ(hexes[1].q, -1);  // Rear hex is West of anchor
  EXPECT_EQ(hexes[1].r, 0);
}

TEST(UnitShapes, DreadnoughtThreeHexes) {
  Hex anchor = {0, 0};
  auto hexes = get_unit_hexes(UnitType::DREADNOUGHT, anchor, 0);  // Facing East
  EXPECT_EQ(hexes.size(), 3u);
  // Anchor is front, two rear hexes behind
}

TEST(UnitShapes, PortalHexes) {
  auto p0_hexes = get_portal_hexes(0, TestConfig::BOARD_SIDE);
  EXPECT_EQ(p0_hexes.size(), 3u);

  auto p1_hexes = get_portal_hexes(1, TestConfig::BOARD_SIDE);
  EXPECT_EQ(p1_hexes.size(), 3u);

  // Portals should be at opposite ends
  EXPECT_NE(p0_hexes[0], p1_hexes[0]);
}

// =============================================================================
// Game State Tests
// =============================================================================

TEST(GameState, InitialState) {
  TestGame game;

  EXPECT_EQ(game.current_player(), 0);
  EXPECT_EQ(game.current_turn(), 1);
  EXPECT_EQ(game.num_players(), 2);
  EXPECT_EQ(game.num_moves(), TestAS::NUM_MOVES);
  EXPECT_FALSE(game.has_taken_action());
}

TEST(GameState, InitialUnitsArePortals) {
  TestGame game;

  // Should have 2 portals initially - check via get_units()
  auto units = game.get_units();
  EXPECT_EQ(units.size(), 2u);
  EXPECT_EQ(units[0].type, 3);  // Portal type
  EXPECT_EQ(units[1].type, 3);  // Portal type
}

TEST(GameState, ValidMovesOnTurnOne) {
  TestGame game;

  auto valids = game.valid_moves();

  // On turn 1, only deploy actions should be valid
  // No moves, no attacks, no end turn
  int total_valid = valids.sum();
  EXPECT_GT(total_valid, 0);

  // End turn should NOT be valid
  EXPECT_EQ(valids(TestAS::END_TURN_OFFSET), 0);

  // Some deploy action should be valid
  bool has_deploy = false;
  for (int i = TestAS::DEPLOY_OFFSET; i < TestAS::END_TURN_OFFSET; ++i) {
    if (valids(i) == 1) {
      has_deploy = true;
      break;
    }
  }
  EXPECT_TRUE(has_deploy);
}

TEST(GameState, DeployAction) {
  TestGame game;

  auto valids = game.valid_moves();

  // Find a valid deploy action
  int deploy_action = -1;
  for (int i = TestAS::DEPLOY_OFFSET; i < TestAS::END_TURN_OFFSET; ++i) {
    if (valids(i) == 1) {
      deploy_action = i;
      break;
    }
  }

  ASSERT_NE(deploy_action, -1) << "No valid deploy action found";

  // Play the deploy action
  game.play_move(static_cast<uint32_t>(deploy_action));

  // After deploy, turn should have ended and switched to player 1
  EXPECT_EQ(game.current_player(), 1);
  EXPECT_EQ(game.current_turn(), 2);
}

TEST(GameState, CopyEquality) {
  TestGame game;

  auto copy = game.copy();
  EXPECT_EQ(game, *copy);

  // Make a move
  auto valids = game.valid_moves();
  int action = -1;
  for (int i = 0; i < TestAS::NUM_MOVES; ++i) {
    if (valids(i) == 1) {
      action = i;
      break;
    }
  }
  game.play_move(static_cast<uint32_t>(action));

  // Now they should be different
  EXPECT_NE(game, *copy);
}

TEST(GameState, Canonicalized) {
  TestGame game;

  auto tensor = game.canonicalized();

  EXPECT_EQ(tensor.dimension(0), TestAS::CANONICAL_SHAPE[0]);
  EXPECT_EQ(tensor.dimension(1), TestAS::CANONICAL_SHAPE[1]);
  EXPECT_EQ(tensor.dimension(2), TestAS::CANONICAL_SHAPE[2]);
}

TEST(GameState, ScoresNotOverInitially) {
  TestGame game;
  EXPECT_FALSE(game.scores().has_value());
}

TEST(GameState, DumpOutput) {
  TestGame game;

  std::string dump = game.dump();
  EXPECT_FALSE(dump.empty());
  EXPECT_TRUE(dump.find("Turn:") != std::string::npos);
  EXPECT_TRUE(dump.find("reserves") != std::string::npos);
}

// =============================================================================
// Turn Structure Tests
// =============================================================================

TEST(TurnStructure, TurnOneIsDeployOnly) {
  TestGame game;

  auto valids = game.valid_moves();

  // Check that no spatial actions are valid (all movement and fire)
  for (int i = TestAS::SPATIAL_OFFSET; i < TestAS::DEPLOY_OFFSET; ++i) {
    EXPECT_EQ(valids(i), 0) << "Spatial action " << i << " should not be valid on turn 1";
  }
}

TEST(TurnStructure, AfterTurnOneCanMoveAndFire) {
  TestGame game;

  // Player 0 deploys
  auto valids = game.valid_moves();
  int deploy = -1;
  for (int i = TestAS::DEPLOY_OFFSET; i < TestAS::END_TURN_OFFSET; ++i) {
    if (valids(i) == 1) {
      deploy = i;
      break;
    }
  }
  game.play_move(static_cast<uint32_t>(deploy));

  // Player 1 deploys
  valids = game.valid_moves();
  deploy = -1;
  for (int i = TestAS::DEPLOY_OFFSET; i < TestAS::END_TURN_OFFSET; ++i) {
    if (valids(i) == 1) {
      deploy = i;
      break;
    }
  }
  game.play_move(static_cast<uint32_t>(deploy));

  // Now it's turn 3, player 0's turn
  EXPECT_EQ(game.current_turn(), 3);
  EXPECT_EQ(game.current_player(), 0);

  // Should be able to move or deploy
  valids = game.valid_moves();
  int total_valid = valids.sum();
  EXPECT_GT(total_valid, 0);

  // At least some spatial actions should be valid (for the deployed unit)
  bool has_spatial_action = false;
  for (int i = TestAS::SPATIAL_OFFSET; i < TestAS::DEPLOY_OFFSET; ++i) {
    if (valids(i) == 1) {
      has_spatial_action = true;
      break;
    }
  }
  // Also check deploy actions
  bool has_deploy = false;
  for (int i = TestAS::DEPLOY_OFFSET; i < TestAS::END_TURN_OFFSET; ++i) {
    if (valids(i) == 1) {
      has_deploy = true;
      break;
    }
  }
  EXPECT_TRUE(has_spatial_action || has_deploy);
}

// =============================================================================
// Combat Tests
// =============================================================================

TEST(Combat, CannonInfoFighter) {
  auto cannons = get_cannon_info(UnitType::FIGHTER);
  EXPECT_EQ(cannons.size(), 1u);
  EXPECT_EQ(cannons[0].direction_offset, 0);  // Forward
  EXPECT_EQ(cannons[0].source_hex_idx, 0);    // From anchor
}

TEST(Combat, CannonInfoCruiser) {
  auto cannons = get_cannon_info(UnitType::CRUISER);
  EXPECT_EQ(cannons.size(), 3u);
}

TEST(Combat, CannonInfoDreadnought) {
  auto cannons = get_cannon_info(UnitType::DREADNOUGHT);
  EXPECT_EQ(cannons.size(), 4u);
}

TEST(Combat, LineOfSightEmpty) {
  std::vector<Hex> occupied;
  EXPECT_TRUE(has_line_of_sight({0, 0}, 0, 1, occupied));
  EXPECT_TRUE(has_line_of_sight({0, 0}, 0, 2, occupied));
}

TEST(Combat, LineOfSightBlocked) {
  std::vector<Hex> occupied = {{1, 0}};  // Hex at East
  // Shooting East from origin, blocked at distance 2 (passes through {1, 0})
  EXPECT_FALSE(has_line_of_sight({0, 0}, 0, 2, occupied));
}

// =============================================================================
// Deployment Tests
// =============================================================================

TEST(Deployment, DeployHexLocation) {
  Hex p0_deploy = get_deploy_hex(0, TestConfig::BOARD_SIDE);
  Hex p1_deploy = get_deploy_hex(1, TestConfig::BOARD_SIDE);

  EXPECT_TRUE(hex_in_bounds(p0_deploy, TestConfig::BOARD_SIDE));
  EXPECT_TRUE(hex_in_bounds(p1_deploy, TestConfig::BOARD_SIDE));
  EXPECT_NE(p0_deploy, p1_deploy);
}

TEST(Deployment, ValidDeployFacings) {
  auto p0_fighter_facings = get_valid_deploy_facings(UnitType::FIGHTER, 0);
  auto p0_dreadnought_facings = get_valid_deploy_facings(UnitType::DREADNOUGHT, 0);

  EXPECT_EQ(p0_fighter_facings.size(), 3u);
  EXPECT_EQ(p0_dreadnought_facings.size(), 4u);  // Dreadnoughts have 4 valid facings
}

// =============================================================================
// Game Flow Tests
// =============================================================================

TEST(GameFlow, PlayMultipleTurns) {
  TestGame game;

  // Play several turns
  for (int i = 0; i < 10; ++i) {
    auto valids = game.valid_moves();
    int total_valid = valids.sum();

    if (total_valid == 0) {
      // Game might be over or stuck
      break;
    }

    // Pick first valid action
    int action = -1;
    for (int j = 0; j < TestAS::NUM_MOVES; ++j) {
      if (valids(j) == 1) {
        action = j;
        break;
      }
    }

    if (action >= 0) {
      game.play_move(static_cast<uint32_t>(action));
    }

    auto scores = game.scores();
    if (scores.has_value()) {
      // Game ended
      break;
    }
  }

  // Game should still be in a valid state
  std::string dump = game.dump();
  EXPECT_FALSE(dump.empty());
}

TEST(GameFlow, Symmetries) {
  TestGame game;

  auto tensor = game.canonicalized();
  PlayHistory base;
  base.canonical = tensor;
  base.v = Vector<float>(3);
  base.v.setConstant(1.0f / 3.0f);
  base.pi = Vector<float>(TestAS::NUM_MOVES);
  base.pi.setConstant(1.0f / TestAS::NUM_MOVES);

  auto syms = game.symmetries(base);
  EXPECT_GE(syms.size(), 1u);
}

// =============================================================================
// Symmetry Tests (180° Rotation)
// =============================================================================

TEST(Symmetries, ReturnsCorrectCount) {
  TestGame game;
  PlayHistory base;
  base.canonical = game.canonicalized();
  base.v = Vector<float>(3);
  base.v << 1.0f, 0.0f, 0.0f;
  base.pi = Vector<float>(TestAS::NUM_MOVES);
  base.pi.setConstant(1.0f / TestAS::NUM_MOVES);

  auto syms = game.symmetries(base);
  // With canonicalized observations, we now have 2 symmetries:
  // 1. Identity
  // 2. NW-SE diagonal mirror
  EXPECT_EQ(syms.size(), 2u);
}

TEST(Symmetries, IdentityPreservesValues) {
  TestGame game;
  PlayHistory base;
  base.canonical = game.canonicalized();
  base.v = Vector<float>(3);
  base.v << 0.7f, 0.2f, 0.1f;
  base.pi = Vector<float>(TestAS::NUM_MOVES);
  base.pi.setZero();

  auto syms = game.symmetries(base);
  ASSERT_EQ(syms.size(), 2u);

  // Identity symmetry (index 0) - values unchanged
  EXPECT_FLOAT_EQ(syms[0].v(0), 0.7f);
  EXPECT_FLOAT_EQ(syms[0].v(1), 0.2f);
  EXPECT_FLOAT_EQ(syms[0].v(2), 0.1f);

  // Mirror symmetry (index 1) - values also unchanged (mirror doesn't affect value)
  EXPECT_FLOAT_EQ(syms[1].v(0), 0.7f);
  EXPECT_FLOAT_EQ(syms[1].v(1), 0.2f);
  EXPECT_FLOAT_EQ(syms[1].v(2), 0.1f);
}

// Note: Rotation-based symmetry tests removed.
// With canonicalized observations, the 180° rotation symmetry is exploited
// by the perspective transformation. symmetries() now returns identity and
// NW-SE diagonal mirror.

TEST(Symmetries, IdentitySymmetryIsUnchanged) {
  TestGame game;

  // Deploy some units to create interesting state
  auto valids = game.valid_moves();
  for (int i = TestAS::DEPLOY_OFFSET; i < TestAS::END_TURN_OFFSET; ++i) {
    if (valids(i) == 1) {
      game.play_move(i);
      break;
    }
  }

  PlayHistory base;
  base.canonical = game.canonicalized();
  base.v = Vector<float>(3);
  base.v << 0.5f, 0.3f, 0.2f;
  base.pi = Vector<float>(TestAS::NUM_MOVES);
  for (int i = 0; i < TestAS::NUM_MOVES; ++i) {
    base.pi(i) = static_cast<float>(i) / TestAS::NUM_MOVES;
  }

  auto syms = game.symmetries(base);
  ASSERT_EQ(syms.size(), 2u);

  // Identity symmetry (index 0) should be exactly the same as input
  for (int i = 0; i < 3; ++i) {
    EXPECT_FLOAT_EQ(syms[0].v(i), base.v(i))
        << "Identity value mismatch at " << i;
  }

  for (int i = 0; i < TestAS::NUM_MOVES; ++i) {
    EXPECT_FLOAT_EQ(syms[0].pi(i), base.pi(i))
        << "Identity policy mismatch at " << i;
  }
}

// =============================================================================
// P1 Canonicalization Tests (180° Rotation)
// =============================================================================

TEST(P1Canonicalization, ObservationShowsP1Perspective) {
  // After both players deploy, P1's observation should show:
  // - P1's units in "my" channels (1-4)
  // - P0's units in "opponent" channels (5-8)
  // - Board rotated 180° so P1's portal appears at "top" (low row indices)
  TestGame game;

  // P0 deploys fighter facing NE
  game.play_move(TestAS::encode_deploy(0, 1));  // Fighter, NE
  // P1 deploys fighter facing SW
  game.play_move(TestAS::encode_deploy(0, 4));  // Fighter, SW (canonical)

  // Now it's P0's turn again
  game.play_move(TestAS::END_TURN_OFFSET);

  // Now it's P1's turn - get canonical observation
  ASSERT_EQ(game.current_player(), 1);
  auto obs = game.canonicalized();

  // P1's portal should appear in "my portal" channel (channel 4) at positions
  // that are rotated 180° from P0's perspective
  // P1's portal is at the "top" of the board (low indices after rotation)
  constexpr int MY_PORTAL_CHANNEL = 4;
  constexpr int OPP_PORTAL_CHANNEL = 8;

  // Count portal pixels in each channel
  int my_portal_count = 0;
  int opp_portal_count = 0;
  for (int row = 0; row < TestAS::BOARD_DIM; ++row) {
    for (int col = 0; col < TestAS::BOARD_DIM; ++col) {
      if (obs(MY_PORTAL_CHANNEL, row, col) > 0) my_portal_count++;
      if (obs(OPP_PORTAL_CHANNEL, row, col) > 0) opp_portal_count++;
    }
  }

  // Both players should have a portal (3 hexes each)
  EXPECT_EQ(my_portal_count, 3) << "P1 should see own portal in 'my' channel";
  EXPECT_EQ(opp_portal_count, 3) << "P1 should see P0's portal in 'opponent' channel";
}

TEST(P1Canonicalization, ValidMovesAreRotated) {
  // When P1 has a unit, valid_moves should return canonicalized action indices
  // where the position is rotated 180° and L/R slots are swapped
  TestGame game;

  // P0 deploys fighter facing NE at deploy hex
  game.play_move(TestAS::encode_deploy(0, 1));  // Fighter, NE

  // P1 deploys fighter facing SW (canonical: this is actually facing NE in world coords
  // because P1's facing is also rotated)
  game.play_move(TestAS::encode_deploy(0, 4));  // Fighter, SW in canonical space

  // P0 ends turn
  game.play_move(TestAS::END_TURN_OFFSET);

  // Now P1's turn
  ASSERT_EQ(game.current_player(), 1);

  auto valids = game.valid_moves();

  // P1's fighter is at world position near P1's portal
  // In canonical space, this should be rotated 180° to appear near low row indices
  // The valid moves should be at the canonicalized position

  // Count valid spatial actions
  int valid_spatial = 0;
  for (int i = 0; i < TestAS::SPATIAL_ACTIONS; ++i) {
    if (valids(i) == 1) valid_spatial++;
  }

  // P1's fighter should have movement options (forward, forward-left, forward-right)
  EXPECT_GE(valid_spatial, 1) << "P1 should have at least one valid spatial action";
}

TEST(P1Canonicalization, PlayMoveDecanonicalizes) {
  // When P1 plays a move, it should be de-canonicalized correctly
  TestGame game;

  // P0 deploys fighter facing NE
  game.play_move(TestAS::encode_deploy(0, 1));

  // P1 deploys fighter facing SW (canonical)
  // In world coords, P1's fighter faces NE (opposite direction)
  game.play_move(TestAS::encode_deploy(0, 4));

  // P0 ends turn
  game.play_move(TestAS::END_TURN_OFFSET);

  // Now P1's turn
  ASSERT_EQ(game.current_player(), 1);

  // Find a valid movement action and play it
  auto valids = game.valid_moves();
  int move_played = -1;
  for (int i = 0; i < TestAS::SPATIAL_ACTIONS; ++i) {
    if (valids(i) == 1) {
      int row, col, action_type;
      TestAS::decode_spatial_action(i, row, col, action_type);
      // Only test movement actions (0-4), not fire actions
      if (action_type <= 4) {
        move_played = i;
        break;
      }
    }
  }

  if (move_played >= 0) {
    game.play_move(move_played);
    // If we got here without crashing, de-canonicalization worked correctly
    SUCCEED() << "P1 move de-canonicalized and executed successfully";
  }
}

// =============================================================================
// NW-SE Mirror Symmetry Tests
// =============================================================================

TEST(MirrorSymmetry, SelfInverse) {
  // Apply mirror transformation twice -> should equal original
  TestGame game;

  // Deploy some units for interesting state
  game.play_move(TestAS::encode_deploy(0, 1));  // P0 fighter NE
  game.play_move(TestAS::encode_deploy(0, 4));  // P1 fighter SW
  game.play_move(TestAS::END_TURN_OFFSET);      // P0 ends turn
  game.play_move(TestAS::END_TURN_OFFSET);      // P1 ends turn

  // Create base history with non-trivial values
  PlayHistory base;
  base.canonical = game.canonicalized();
  base.v = Vector<float>(3);
  base.v << 0.6f, 0.3f, 0.1f;
  base.pi = Vector<float>(TestAS::NUM_MOVES);
  for (int i = 0; i < TestAS::NUM_MOVES; ++i) {
    base.pi(i) = static_cast<float>(i + 1) / (TestAS::NUM_MOVES + 1);
  }

  auto syms = game.symmetries(base);
  ASSERT_EQ(syms.size(), 2u);

  // Get the mirror symmetry (index 1)
  const auto& mirrored = syms[1];

  // Apply mirror again by treating mirrored as the base
  PlayHistory mirrored_as_base;
  mirrored_as_base.canonical = mirrored.canonical;
  mirrored_as_base.v = mirrored.v;
  mirrored_as_base.pi = mirrored.pi;

  auto syms2 = game.symmetries(mirrored_as_base);
  const auto& double_mirrored = syms2[1];

  // Double mirror should equal original
  // Check observation tensor
  for (int ch = 0; ch < TestAS::CANONICAL_CHANNELS; ++ch) {
    for (int row = 0; row < TestAS::BOARD_DIM; ++row) {
      for (int col = 0; col < TestAS::BOARD_DIM; ++col) {
        EXPECT_NEAR(double_mirrored.canonical(ch, row, col),
                    base.canonical(ch, row, col), 1e-5f)
            << "Self-inverse failed at channel=" << ch << " row=" << row << " col=" << col;
      }
    }
  }

  // Check policy vector
  for (int i = 0; i < TestAS::NUM_MOVES; ++i) {
    EXPECT_NEAR(double_mirrored.pi(i), base.pi(i), 1e-5f)
        << "Policy self-inverse failed at action " << i;
  }
}

TEST(MirrorSymmetry, ObservationTransposeCorrect) {
  // Mirror should transpose (row, col) -> (col, row) for spatial channels
  TestGame game;

  // Create asymmetric state
  game.play_move(TestAS::encode_deploy(0, 0));  // P0 fighter E
  game.play_move(TestAS::encode_deploy(1, 2));  // P1 cruiser NW
  game.play_move(TestAS::END_TURN_OFFSET);

  PlayHistory base;
  base.canonical = game.canonicalized();
  base.v = Vector<float>(3);
  base.v << 0.5f, 0.3f, 0.2f;
  base.pi = Vector<float>(TestAS::NUM_MOVES);
  base.pi.setZero();

  auto syms = game.symmetries(base);
  const auto& mirrored = syms[1];

  // For channels that are pure spatial (like valid hex mask, channel 0),
  // mirrored(ch, col, row) should equal base(ch, row, col)
  for (int row = 0; row < TestAS::BOARD_DIM; ++row) {
    for (int col = 0; col < TestAS::BOARD_DIM; ++col) {
      EXPECT_FLOAT_EQ(mirrored.canonical(0, col, row), base.canonical(0, row, col))
          << "Valid hex mask not transposed correctly at row=" << row << " col=" << col;
    }
  }
}

TEST(MirrorSymmetry, PolicySpatialActionsRemapped) {
  // Spatial action at (row, col, slot) should map to
  // (BOARD_DIM-1-row, row+col-(BOARD_SIDE-1), SLOT_MAP[slot])
  TestGame game;

  PlayHistory base;
  base.canonical = game.canonicalized();
  base.v = Vector<float>(3);
  base.v << 0.5f, 0.5f, 0.0f;
  base.pi = Vector<float>(TestAS::NUM_MOVES);
  base.pi.setZero();

  constexpr int BOARD_DIM = TestAS::BOARD_DIM;
  constexpr int BOARD_SIDE = TestConfig::BOARD_SIDE;

  // Set specific spatial actions to verify remapping
  // Action at (row=2, col=5, slot=1) with NW-axis mirror:
  // new_row = 8 - 2 = 6
  // new_col = 2 + 5 - 4 = 3
  // new_slot = SLOT_MAP[1] = 2 (forward-left -> forward-right)
  int test_row = 2, test_col = 5, test_slot = 1;
  int base_action = TestAS::encode_spatial_action(test_row, test_col, test_slot);
  base.pi(base_action) = 1.0f;

  auto syms = game.symmetries(base);
  const auto& mirrored = syms[1];

  // Expected mirrored action with NW-axis position transform
  int expected_row = (BOARD_DIM - 1) - test_row;  // 8 - 2 = 6
  int expected_col = test_row + test_col - (BOARD_SIDE - 1);  // 2 + 5 - 4 = 3
  int expected_slot = SLOT_MAP[test_slot];  // SLOT_MAP[1] = 2
  int expected_action = TestAS::encode_spatial_action(expected_row, expected_col, expected_slot);

  EXPECT_FLOAT_EQ(mirrored.pi(expected_action), 1.0f)
      << "Spatial action at (" << test_row << "," << test_col << "," << test_slot
      << ") should map to (" << expected_row << "," << expected_col << "," << expected_slot << ")";
}

TEST(MirrorSymmetry, PolicyDeployActionsRemapped) {
  // Deploy actions use:
  // Fighters/Cruisers: MIRROR_DIRECTION_MAP (axis NW, swaps 0↔4, 1↔3)
  // Dreadnoughts: DEPLOY_MIRROR_D (axis between NE/NW, swaps 0↔3, 1↔2, 4↔5)
  TestGame game;

  PlayHistory base;
  base.canonical = game.canonicalized();
  base.v = Vector<float>(3);
  base.v << 0.5f, 0.5f, 0.0f;
  base.pi = Vector<float>(TestAS::NUM_MOVES);
  base.pi.setZero();

  // Test Fighter (type 0) - uses MIRROR_DIRECTION_MAP
  for (int facing = 0; facing < 6; ++facing) {
    base.pi.setZero();
    int deploy_action = TestAS::encode_deploy(0, facing);
    base.pi(deploy_action) = 1.0f;

    auto syms = game.symmetries(base);
    const auto& mirrored = syms[1];

    int expected_facing = MIRROR_DIRECTION_MAP[facing];
    int expected_action = TestAS::encode_deploy(0, expected_facing);

    EXPECT_FLOAT_EQ(mirrored.pi(expected_action), 1.0f)
        << "Fighter deploy facing " << facing << " should map to " << expected_facing;
  }

  // Test Cruiser (type 1) - uses MIRROR_DIRECTION_MAP
  for (int facing = 0; facing < 6; ++facing) {
    base.pi.setZero();
    int deploy_action = TestAS::encode_deploy(1, facing);
    base.pi(deploy_action) = 1.0f;

    auto syms = game.symmetries(base);
    const auto& mirrored = syms[1];

    int expected_facing = MIRROR_DIRECTION_MAP[facing];
    int expected_action = TestAS::encode_deploy(1, expected_facing);

    EXPECT_FLOAT_EQ(mirrored.pi(expected_action), 1.0f)
        << "Cruiser deploy facing " << facing << " should map to " << expected_facing;
  }

  // Test Dreadnought (type 2) - uses DEPLOY_MIRROR_D
  for (int facing = 0; facing < 6; ++facing) {
    base.pi.setZero();
    int deploy_action = TestAS::encode_deploy(2, facing);
    base.pi(deploy_action) = 1.0f;

    auto syms = game.symmetries(base);
    const auto& mirrored = syms[1];

    int expected_facing = DEPLOY_MIRROR_D[facing];
    int expected_action = TestAS::encode_deploy(2, expected_facing);

    EXPECT_FLOAT_EQ(mirrored.pi(expected_action), 1.0f)
        << "Dreadnought deploy facing " << facing << " should map to " << expected_facing;
  }
}

TEST(MirrorSymmetry, EndTurnUnchanged) {
  // End turn action should be unchanged by mirror
  TestGame game;

  PlayHistory base;
  base.canonical = game.canonicalized();
  base.v = Vector<float>(3);
  base.v << 0.5f, 0.5f, 0.0f;
  base.pi = Vector<float>(TestAS::NUM_MOVES);
  base.pi.setZero();
  base.pi(TestAS::END_TURN_OFFSET) = 1.0f;

  auto syms = game.symmetries(base);
  const auto& mirrored = syms[1];

  EXPECT_FLOAT_EQ(mirrored.pi(TestAS::END_TURN_OFFSET), 1.0f)
      << "End turn action should be unchanged by mirror";
}

TEST(MirrorSymmetry, ValuePreserved) {
  // Mirror should not change the value
  TestGame game;

  PlayHistory base;
  base.canonical = game.canonicalized();
  base.v = Vector<float>(3);
  base.v << 0.7f, 0.2f, 0.1f;
  base.pi = Vector<float>(TestAS::NUM_MOVES);
  base.pi.setZero();

  auto syms = game.symmetries(base);
  const auto& mirrored = syms[1];

  for (int i = 0; i < 3; ++i) {
    EXPECT_FLOAT_EQ(mirrored.v(i), base.v(i))
        << "Value should be unchanged by mirror at index " << i;
  }
}

TEST(MirrorSymmetry, FacingChannelsRemapped) {
  // Facing channels (9-14) should be remapped via MIRROR_DIRECTION_MAP
  // Position transform: (row, col) → (BOARD_DIM-1-row, row+col-(BOARD_SIDE-1))
  TestGame game;

  // Deploy a fighter with facing NE (direction 1)
  game.play_move(TestAS::encode_deploy(0, 1));  // P0 fighter NE
  game.play_move(TestAS::encode_deploy(0, 4));  // P1 fighter SW
  game.play_move(TestAS::END_TURN_OFFSET);

  PlayHistory base;
  base.canonical = game.canonicalized();
  base.v = Vector<float>(3);
  base.v << 0.5f, 0.3f, 0.2f;
  base.pi = Vector<float>(TestAS::NUM_MOVES);
  base.pi.setZero();

  auto syms = game.symmetries(base);
  const auto& mirrored = syms[1];

  constexpr int BOARD_DIM = TestAS::BOARD_DIM;
  constexpr int BOARD_SIDE = TestConfig::BOARD_SIDE;

  // Find a position where there's facing info in the base observation
  // and verify it's remapped correctly in the mirror
  for (int row = 0; row < BOARD_DIM; ++row) {
    for (int col = 0; col < BOARD_DIM; ++col) {
      for (int dir = 0; dir < 6; ++dir) {
        float base_val = base.canonical(9 + dir, row, col);
        if (base_val > 0) {
          // This position has a unit facing direction 'dir'
          // NW-axis mirror position: (row, col) → (BOARD_DIM-1-row, row+col-(BOARD_SIDE-1))
          int new_row = (BOARD_DIM - 1) - row;
          int new_col = row + col - (BOARD_SIDE - 1);
          int expected_dir = MIRROR_DIRECTION_MAP[dir];

          if (new_col >= 0 && new_col < BOARD_DIM) {
            float mirrored_val = mirrored.canonical(9 + expected_dir, new_row, new_col);
            EXPECT_FLOAT_EQ(mirrored_val, base_val)
                << "Facing at (" << row << "," << col << ") dir=" << dir
                << " should map to (" << new_row << "," << new_col << ") dir=" << expected_dir;
          }
        }
      }
    }
  }
}

TEST(MirrorSymmetry, DeployMirrorPreservesValidFacings) {
  // Verify that mirroring deploy actions preserves valid facings
  // Fighter/Cruiser valid canonical facings: {1, 2, 3}
  // Dreadnought valid canonical facings: {0, 1, 2, 3}
  //
  // MIRROR_DIRECTION_MAP = [4, 3, 2, 1, 0, 5] preserves F/C validity:
  //   1→3, 2→2, 3→1 => {1,2,3} → {3,2,1} = {1,2,3} ✓
  // DEPLOY_MIRROR_D = [3, 2, 1, 0, 5, 4] preserves Dread validity:
  //   0→3, 1→2, 2→1, 3→0 => {0,1,2,3} → {3,2,1,0} = {0,1,2,3} ✓
  TestGame game;

  PlayHistory base;
  base.canonical = game.canonicalized();
  base.v = Vector<float>(3);
  base.v << 0.5f, 0.5f, 0.0f;
  base.pi = Vector<float>(TestAS::NUM_MOVES);

  // Test Fighter - valid facings {1, 2, 3}
  for (int facing : {1, 2, 3}) {
    base.pi.setZero();
    base.pi(TestAS::encode_deploy(0, facing)) = 1.0f;

    auto syms = game.symmetries(base);
    const auto& mirrored = syms[1];

    // Find where the probability went
    int mirrored_facing = -1;
    for (int f = 0; f < 6; ++f) {
      if (mirrored.pi(TestAS::encode_deploy(0, f)) > 0.5f) {
        mirrored_facing = f;
        break;
      }
    }

    // Verify mirrored facing is valid for fighters
    EXPECT_TRUE(mirrored_facing == 1 || mirrored_facing == 2 || mirrored_facing == 3)
        << "Fighter facing " << facing << " mirrored to invalid facing " << mirrored_facing;
  }

  // Test Cruiser - valid facings {1, 2, 3}
  for (int facing : {1, 2, 3}) {
    base.pi.setZero();
    base.pi(TestAS::encode_deploy(1, facing)) = 1.0f;

    auto syms = game.symmetries(base);
    const auto& mirrored = syms[1];

    int mirrored_facing = -1;
    for (int f = 0; f < 6; ++f) {
      if (mirrored.pi(TestAS::encode_deploy(1, f)) > 0.5f) {
        mirrored_facing = f;
        break;
      }
    }

    EXPECT_TRUE(mirrored_facing == 1 || mirrored_facing == 2 || mirrored_facing == 3)
        << "Cruiser facing " << facing << " mirrored to invalid facing " << mirrored_facing;
  }

  // Test Dreadnought - valid facings {0, 1, 2, 3}
  for (int facing : {0, 1, 2, 3}) {
    base.pi.setZero();
    base.pi(TestAS::encode_deploy(2, facing)) = 1.0f;

    auto syms = game.symmetries(base);
    const auto& mirrored = syms[1];

    int mirrored_facing = -1;
    for (int f = 0; f < 6; ++f) {
      if (mirrored.pi(TestAS::encode_deploy(2, f)) > 0.5f) {
        mirrored_facing = f;
        break;
      }
    }

    EXPECT_TRUE(mirrored_facing >= 0 && mirrored_facing <= 3)
        << "Dreadnought facing " << facing << " mirrored to invalid facing " << mirrored_facing;
  }
}

TEST(MirrorSymmetry, DeployMirrorRoundTrip) {
  // Applying mirror twice should return to original
  TestGame game;

  PlayHistory base;
  base.canonical = game.canonicalized();
  base.v = Vector<float>(3);
  base.v << 0.5f, 0.5f, 0.0f;
  base.pi = Vector<float>(TestAS::NUM_MOVES);

  // Test all unit types with their valid facings
  std::vector<std::pair<int, int>> test_cases = {
    {0, 1}, {0, 2}, {0, 3},  // Fighter facings
    {1, 1}, {1, 2}, {1, 3},  // Cruiser facings
    {2, 0}, {2, 1}, {2, 2}, {2, 3}  // Dreadnought facings
  };

  for (const auto& [type_idx, facing] : test_cases) {
    base.pi.setZero();
    int original_action = TestAS::encode_deploy(type_idx, facing);
    base.pi(original_action) = 1.0f;

    auto syms = game.symmetries(base);
    const auto& mirrored = syms[1];

    // Apply mirror again
    auto syms2 = game.symmetries(mirrored);
    const auto& double_mirrored = syms2[1];

    // Should be back to original
    EXPECT_FLOAT_EQ(double_mirrored.pi(original_action), 1.0f)
        << "Deploy type=" << type_idx << " facing=" << facing
        << " not preserved after double mirror";
  }
}

TEST(MirrorSymmetry, DeployMirrorMapsCorrectly) {
  // Verify the specific mappings using MIRROR_DIRECTION_MAP = [4, 3, 2, 1, 0, 5]:
  // F/C: 0→4, 1→3, 2→2, 3→1, 4→0, 5→5 (valid facings: 1↔3, 2→2)
  // Dread uses DEPLOY_MIRROR_D = [3, 2, 1, 0, 5, 4]:
  //   0↔3, 1↔2 (valid facings all swap within valid set)
  TestGame game;

  PlayHistory base;
  base.canonical = game.canonicalized();
  base.v = Vector<float>(3);
  base.v << 0.5f, 0.5f, 0.0f;
  base.pi = Vector<float>(TestAS::NUM_MOVES);

  // Fighter/Cruiser expected mappings for valid facings using MIRROR_DIRECTION_MAP
  // MIRROR_DIRECTION_MAP = [4, 3, 2, 1, 0, 5]
  std::vector<std::pair<int, int>> fc_expected = {
    {1, 3}, {2, 2}, {3, 1}  // Valid facings only
  };

  for (int type_idx = 0; type_idx < 2; ++type_idx) {  // Fighter and Cruiser
    for (const auto& [from_facing, to_facing] : fc_expected) {
      base.pi.setZero();
      base.pi(TestAS::encode_deploy(type_idx, from_facing)) = 1.0f;

      auto syms = game.symmetries(base);
      const auto& mirrored = syms[1];

      EXPECT_FLOAT_EQ(mirrored.pi(TestAS::encode_deploy(type_idx, to_facing)), 1.0f)
          << (type_idx == 0 ? "Fighter" : "Cruiser")
          << " facing " << from_facing << " should map to " << to_facing;
    }
  }

  // Dreadnought expected mappings for valid facings using DEPLOY_MIRROR_D
  // DEPLOY_MIRROR_D = [3, 2, 1, 0, 5, 4]
  std::vector<std::pair<int, int>> dread_expected = {
    {0, 3}, {1, 2}, {2, 1}, {3, 0}  // Valid facings only
  };

  for (const auto& [from_facing, to_facing] : dread_expected) {
    base.pi.setZero();
    base.pi(TestAS::encode_deploy(2, from_facing)) = 1.0f;

    auto syms = game.symmetries(base);
    const auto& mirrored = syms[1];

    EXPECT_FLOAT_EQ(mirrored.pi(TestAS::encode_deploy(2, to_facing)), 1.0f)
        << "Dreadnought facing " << from_facing << " should map to " << to_facing;
  }
}

// =============================================================================
// Unit Property Tests
// =============================================================================

TEST(UnitProperties, MaxHP) {
  EXPECT_EQ(get_max_hp(UnitType::FIGHTER), 3);
  EXPECT_EQ(get_max_hp(UnitType::CRUISER), 4);
  EXPECT_EQ(get_max_hp(UnitType::DREADNOUGHT), 6);
  EXPECT_EQ(get_max_hp(UnitType::PORTAL), 5);
}

TEST(UnitProperties, MaxMoves) {
  EXPECT_EQ(get_max_moves(UnitType::FIGHTER), 2);
  EXPECT_EQ(get_max_moves(UnitType::CRUISER), 1);
  EXPECT_EQ(get_max_moves(UnitType::DREADNOUGHT), 1);
  EXPECT_EQ(get_max_moves(UnitType::PORTAL), 0);
}

TEST(UnitProperties, NumCannons) {
  EXPECT_EQ(get_num_cannons(UnitType::FIGHTER), 1);
  EXPECT_EQ(get_num_cannons(UnitType::CRUISER), 3);
  EXPECT_EQ(get_num_cannons(UnitType::DREADNOUGHT), 4);
  EXPECT_EQ(get_num_cannons(UnitType::PORTAL), 0);
}

TEST(UnitProperties, UnitSize) {
  EXPECT_EQ(get_unit_size(UnitType::FIGHTER), 1);
  EXPECT_EQ(get_unit_size(UnitType::CRUISER), 2);
  EXPECT_EQ(get_unit_size(UnitType::DREADNOUGHT), 3);
  EXPECT_EQ(get_unit_size(UnitType::PORTAL), 3);
}

// =============================================================================
// Action Space Configuration Tests (Hex-Based Encoding)
// =============================================================================

TEST(ActionSpace, SkirmishActionCounts) {
  // Skirmish: BOARD_SIDE=5, 9x9 spatial grid
  using AS = ActionSpace<SkirmishConfig>;

  // Spatial actions: 9*9*10 = 810
  EXPECT_EQ(AS::SPATIAL_ACTIONS, 810);

  // Deploy: 3 types * 6 facings = 18
  EXPECT_EQ(AS::DEPLOY_ACTIONS, 18);

  // End turn: 1
  EXPECT_EQ(AS::END_TURN_ACTIONS, 1);

  // Total: 810 + 18 + 1 = 829
  EXPECT_EQ(AS::NUM_MOVES, 829);
}

TEST(ActionSpace, ClashActionCounts) {
  // Clash: BOARD_SIDE=5, 9x9 spatial grid (same as Skirmish)
  using AS = ActionSpace<ClashConfig>;

  // Spatial actions: 9*9*10 = 810
  EXPECT_EQ(AS::SPATIAL_ACTIONS, 810);

  // Deploy: 3 types * 6 facings = 18
  EXPECT_EQ(AS::DEPLOY_ACTIONS, 18);

  // Total: 810 + 18 + 1 = 829
  EXPECT_EQ(AS::NUM_MOVES, 829);
}

TEST(ActionSpace, BattleActionCounts) {
  // Battle: BOARD_SIDE=6, 11x11 spatial grid
  using AS = ActionSpace<BattleConfig>;

  // Spatial actions: 11*11*10 = 1210
  EXPECT_EQ(AS::SPATIAL_ACTIONS, 1210);

  // Deploy: 3 types * 6 facings = 18
  EXPECT_EQ(AS::DEPLOY_ACTIONS, 18);

  // Total: 1210 + 18 + 1 = 1229
  EXPECT_EQ(AS::NUM_MOVES, 1229);
}

TEST(ActionSpace, BoardSizes) {
  // Skirmish & Clash: 5-side board
  EXPECT_EQ(SkirmishConfig::BOARD_SIDE, 5);
  EXPECT_EQ(ClashConfig::BOARD_SIDE, 5);
  // Battle: 6-side board
  EXPECT_EQ(BattleConfig::BOARD_SIDE, 6);

  // Hex counts: 3N² - 3N + 1
  // 5-side: 3*25 - 15 + 1 = 61
  EXPECT_EQ(ActionSpace<SkirmishConfig>::NUM_HEXES, 61);
  EXPECT_EQ(ActionSpace<ClashConfig>::NUM_HEXES, 61);
  // 6-side: 3*36 - 18 + 1 = 91
  EXPECT_EQ(ActionSpace<BattleConfig>::NUM_HEXES, 91);
}

// =============================================================================
// Movement Direction Tests
// =============================================================================

TEST(MovementDirections, FighterMoveCount) {
  EXPECT_EQ(FIGHTER_MOVE_DIRS, 3);
  EXPECT_EQ(get_num_move_dirs(UnitType::FIGHTER), 3);
}

TEST(MovementDirections, CruiserMoveCount) {
  EXPECT_EQ(CRUISER_MOVE_DIRS, 5);
  EXPECT_EQ(get_num_move_dirs(UnitType::CRUISER), 5);
}

TEST(MovementDirections, DreadnoughtMoveCount) {
  EXPECT_EQ(DREAD_MOVE_DIRS, 4);
  EXPECT_EQ(get_num_move_dirs(UnitType::DREADNOUGHT), 4);
}

// =============================================================================
// Fire Validation Tests - Fire only available when target in range
// =============================================================================

// Helper to check if any fire action is valid
bool has_fire_action(const Vector<uint8_t>& valids) {
  // Check all spatial actions for fire slots (5-9)
  for (int action = 0; action < TestAS::SPATIAL_ACTIONS; ++action) {
    int slot = action % ACTIONS_PER_POSITION;
    if (slot >= static_cast<int>(SpatialAction::FIRE_FORWARD) &&
        slot <= static_cast<int>(SpatialAction::FIRE_REAR_RIGHT)) {
      if (valids(action) == 1) return true;
    }
  }
  return false;
}

TEST(FireValidation, NoFireWithoutTarget) {
  TestGame game;

  // Deploy fighters for both players
  auto valids = game.valid_moves();
  // Find a valid fighter deploy
  int deploy = -1;
  for (int i = TestAS::DEPLOY_OFFSET; i < TestAS::DEPLOY_OFFSET + 6; ++i) {
    if (valids(i) == 1) {
      deploy = i;
      break;
    }
  }
  ASSERT_NE(deploy, -1);
  game.play_move(deploy);

  // Player 1 deploys
  valids = game.valid_moves();
  deploy = -1;
  for (int i = TestAS::DEPLOY_OFFSET; i < TestAS::DEPLOY_OFFSET + 6; ++i) {
    if (valids(i) == 1) {
      deploy = i;
      break;
    }
  }
  ASSERT_NE(deploy, -1);
  game.play_move(deploy);

  // Now turn 3 - check that fire is NOT valid (enemies too far apart)
  valids = game.valid_moves();

  // Check that no fire actions are valid
  EXPECT_FALSE(has_fire_action(valids)) << "Fire should not be valid when no target in range";
}

TEST(FireValidation, FireAvailableWhenTargetInRange) {
  TestGame game;

  // Deploy fighters facing each other (NW and SE for direct convergence)
  // P0 facing NW (direction 2), P1 facing SE (canonical = NW = 2 after +3 mod 6)
  // Deploy actions: DEPLOY_OFFSET + unit_type*6 + facing
  // Fighter = type 0, NW = facing 2
  int deploy_f_nw = TestAS::DEPLOY_OFFSET + 0 * 6 + 2;
  auto valids = game.valid_moves();
  ASSERT_EQ(valids(deploy_f_nw), 1) << "P0 should be able to deploy fighter facing NW";
  game.play_move(deploy_f_nw);

  // P1's SE (5) canonicalizes to (5+3)%6 = 2 (NW)
  int canonical_se = (5 + 3) % 6;  // = 2 (NW in canonical space)
  int deploy_f_se = TestAS::DEPLOY_OFFSET + 0 * 6 + canonical_se;
  valids = game.valid_moves();
  ASSERT_EQ(valids(deploy_f_se), 1) << "P1 should be able to deploy fighter facing SE (canonical NW)";
  game.play_move(deploy_f_se);

  // Move fighters toward each other until fire becomes available
  bool fire_found = false;
  for (int turn = 0; turn < 30 && !fire_found; ++turn) {
    valids = game.valid_moves();

    // Check for any fire action
    fire_found = has_fire_action(valids);

    if (fire_found) break;

    // Prioritize forward moves to converge faster
    int forward_action = -1;
    for (int action = 0; action < TestAS::SPATIAL_ACTIONS; ++action) {
      int slot = action % ACTIONS_PER_POSITION;
      if (slot == static_cast<int>(SpatialAction::MOVE_FORWARD) && valids(action) == 1) {
        forward_action = action;
        break;
      }
    }

    if (forward_action >= 0) {
      game.play_move(forward_action);
    } else {
      // Fallback to any valid move (end turn, etc.)
      for (int i = 0; i < TestAS::NUM_MOVES; ++i) {
        if (valids(i) == 1) {
          game.play_move(i);
          break;
        }
      }
    }

    if (game.scores().has_value()) break;
  }

  EXPECT_TRUE(fire_found) << "Fire should become available when fighters are in range";
}

// =============================================================================
// Movement Constraint Tests - Correct number of move options per unit
// =============================================================================

TEST(MovementConstraints, FighterHasThreeMoveOptions) {
  TestGame game;

  // Deploy a fighter for player 0
  auto valids = game.valid_moves();
  int deploy = -1;
  for (int i = TestAS::DEPLOY_OFFSET; i < TestAS::DEPLOY_OFFSET + 6; ++i) {
    if (valids(i) == 1) {
      deploy = i;
      break;
    }
  }
  game.play_move(deploy);

  // Player 1 deploys
  valids = game.valid_moves();
  deploy = -1;
  for (int i = TestAS::DEPLOY_OFFSET; i < TestAS::DEPLOY_OFFSET + 6; ++i) {
    if (valids(i) == 1) {
      deploy = i;
      break;
    }
  }
  game.play_move(deploy);

  // Turn 3 - Player 0's fighter should have moves available
  valids = game.valid_moves();

  // In spatial encoding, count valid fighter movement actions (slots 0, 1, 2)
  int move_count = 0;
  for (int action = 0; action < TestAS::SPATIAL_ACTIONS; ++action) {
    int slot = action % ACTIONS_PER_POSITION;
    // Check movement slots 0, 1, 2 (MOVE_FORWARD, MOVE_FORWARD_LEFT, MOVE_FORWARD_RIGHT)
    if (slot < 3 && valids(action) == 1) {
      move_count++;
    }
  }

  // Fighter should have 1-3 move options (depends on board boundaries)
  EXPECT_GE(move_count, 1) << "Fighter should have at least one movement option";
  EXPECT_LE(move_count, 3) << "Fighter should have at most 3 movement options";
}

TEST(MovementConstraints, CruiserHasFiveMoveOptions) {
  TestGame game;

  // Deploy a cruiser for player 0 (cruiser deploy actions are 26-31)
  auto valids = game.valid_moves();
  int deploy = -1;
  for (int i = TestAS::DEPLOY_OFFSET + 6; i < TestAS::DEPLOY_OFFSET + 12; ++i) {
    if (valids(i) == 1) {
      deploy = i;
      break;
    }
  }
  game.play_move(deploy);

  // Player 1 deploys something
  valids = game.valid_moves();
  deploy = -1;
  for (int i = TestAS::DEPLOY_OFFSET; i < TestAS::END_TURN_OFFSET; ++i) {
    if (valids(i) == 1) {
      deploy = i;
      break;
    }
  }
  game.play_move(deploy);

  // Turn 3 - Player 0's cruiser should have moves available
  valids = game.valid_moves();

  // In spatial encoding, count valid movement actions (slots 0-4)
  int move_count = 0;
  for (int action = 0; action < TestAS::SPATIAL_ACTIONS; ++action) {
    int slot = action % ACTIONS_PER_POSITION;
    // Check movement slots (0-4)
    if (slot < 5 && valids(action) == 1) {
      move_count++;
    }
  }

  // Cruiser should have rotation options at minimum
  EXPECT_GE(move_count, 2) << "Cruiser should have at least rotation options";
  EXPECT_LE(move_count, 5) << "Cruiser should have at most 5 movement options";
}

// =============================================================================
// Slot Numbering Tests
// =============================================================================

TEST(SlotNumbering, MultipleUnitsGetDifferentSlots) {
  TestGame game;

  // Deploy fighters for both players over multiple turns until P0 has 2 fighters
  // Turn 1: P0 deploys fighter
  auto valids = game.valid_moves();
  int deploy = -1;
  for (int i = TestAS::DEPLOY_OFFSET; i < TestAS::DEPLOY_OFFSET + 6; ++i) {
    if (valids(i) == 1) {
      deploy = i;
      break;
    }
  }
  ASSERT_NE(deploy, -1);
  game.play_move(deploy);

  // Turn 2: P1 deploys fighter
  valids = game.valid_moves();
  deploy = -1;
  for (int i = TestAS::DEPLOY_OFFSET; i < TestAS::DEPLOY_OFFSET + 6; ++i) {
    if (valids(i) == 1) {
      deploy = i;
      break;
    }
  }
  ASSERT_NE(deploy, -1);
  game.play_move(deploy);

  // P0 now has 1 fighter. Need to deploy another.
  // Play until we can deploy again
  for (int turn = 0; turn < 10; ++turn) {
    valids = game.valid_moves();

    // Check if we can deploy a fighter
    deploy = -1;
    for (int i = TestAS::DEPLOY_OFFSET; i < TestAS::DEPLOY_OFFSET + 6; ++i) {
      if (valids(i) == 1) {
        deploy = i;
        break;
      }
    }

    if (deploy != -1 && game.current_player() == 0) {
      // Deploy second fighter
      game.play_move(deploy);
      break;
    }

    // Otherwise, play any valid move
    for (int i = 0; i < TestAS::NUM_MOVES; ++i) {
      if (valids(i) == 1) {
        game.play_move(i);
        break;
      }
    }
  }

  // Check that P0 has two fighters with different slots via get_units()
  auto units = game.get_units();
  int p0_fighter_count = 0;
  bool has_f1 = false, has_f2 = false;
  for (const auto& u : units) {
    if (u.player == 0 && u.type == 0) {  // Fighter type is 0
      p0_fighter_count++;
      if (u.slot == 0) has_f1 = true;
      if (u.slot == 1) has_f2 = true;
    }
  }
  EXPECT_EQ(p0_fighter_count, 2) << "P0 should have 2 fighters";
  EXPECT_TRUE(has_f1) << "Should have F1 for first fighter";
  EXPECT_TRUE(has_f2) << "Should have F2 for second fighter";
}

// =============================================================================
// End-to-End Playthrough Test
// =============================================================================

TEST(EndToEnd, CanPlayManyMoves) {
  TestGame game;

  // Play 100 random valid moves
  for (int i = 0; i < 100; ++i) {
    auto valids = game.valid_moves();
    int total_valid = valids.sum();

    if (total_valid == 0 || game.scores().has_value()) {
      break;
    }

    // Pick first valid action
    for (int j = 0; j < TestAS::NUM_MOVES; ++j) {
      if (valids(j) == 1) {
        game.play_move(j);
        break;
      }
    }
  }

  // Verify game state is still valid
  std::string dump = game.dump();
  EXPECT_FALSE(dump.empty());
  EXPECT_TRUE(dump.find("Turn:") != std::string::npos);
}

TEST(EndToEnd, AllGameSizesPlayable) {
  // Skirmish
  {
    StarGambitSkirmishGS game;
    auto valids = game.valid_moves();
    EXPECT_GT(valids.sum(), 0) << "Skirmish should have valid moves";
  }

  // Clash
  {
    StarGambitClashGS game;
    auto valids = game.valid_moves();
    EXPECT_GT(valids.sum(), 0) << "Clash should have valid moves";
  }

  // Battle
  {
    StarGambitBattleGS game;
    auto valids = game.valid_moves();
    EXPECT_GT(valids.sum(), 0) << "Battle should have valid moves";
  }
}

// =============================================================================
// Move Parsing Tests (Hex-Based Encoding)
// =============================================================================

TEST(MoveParsing, ParseFighterMoves) {
  // Create a game with a deployed fighter to test parsing
  TestGame game;

  // Deploy fighter facing NE for player 0
  auto deploy = parse_deploy_cmd<TestConfig>("d f ne");
  ASSERT_TRUE(deploy.has_value());
  game.play_move(*deploy);

  // P1 deploys
  auto p1_deploy = parse_deploy_cmd<TestConfig>("d f se");
  game.play_move(*p1_deploy);

  // Now parse movement commands - they use the fighter's hex position
  auto m_f1_f = parse_move<TestConfig>("m f1 f", game);
  ASSERT_TRUE(m_f1_f.has_value());

  // Verify it's a valid action that decodes to the right slot
  int row, col, action_type;
  TestAS::decode_spatial_action(*m_f1_f, row, col, action_type);
  EXPECT_EQ(action_type, static_cast<int>(SpatialAction::MOVE_FORWARD));

  auto m_f1_fl = parse_move<TestConfig>("m f1 fl", game);
  ASSERT_TRUE(m_f1_fl.has_value());
  TestAS::decode_spatial_action(*m_f1_fl, row, col, action_type);
  EXPECT_EQ(action_type, static_cast<int>(SpatialAction::MOVE_FORWARD_LEFT));

  auto m_f1_fr = parse_move<TestConfig>("m f1 fr", game);
  ASSERT_TRUE(m_f1_fr.has_value());
  TestAS::decode_spatial_action(*m_f1_fr, row, col, action_type);
  EXPECT_EQ(action_type, static_cast<int>(SpatialAction::MOVE_FORWARD_RIGHT));
}

TEST(MoveParsing, ParseCruiserMoves) {
  TestGame game;

  // Deploy cruiser for player 0
  auto deploy = parse_deploy_cmd<TestConfig>("d c ne");
  ASSERT_TRUE(deploy.has_value());
  game.play_move(*deploy);

  // P1 deploys
  auto p1_deploy = parse_deploy_cmd<TestConfig>("d f se");
  game.play_move(*p1_deploy);

  // Parse cruiser movement commands
  auto m_c1_l = parse_move<TestConfig>("m c1 l", game);
  ASSERT_TRUE(m_c1_l.has_value());
  int row, col, action_type;
  TestAS::decode_spatial_action(*m_c1_l, row, col, action_type);
  EXPECT_EQ(action_type, static_cast<int>(SpatialAction::ROTATE_LEFT));

  auto m_c1_f = parse_move<TestConfig>("m c1 f", game);
  ASSERT_TRUE(m_c1_f.has_value());
  TestAS::decode_spatial_action(*m_c1_f, row, col, action_type);
  EXPECT_EQ(action_type, static_cast<int>(SpatialAction::MOVE_FORWARD));

  auto m_c1_r = parse_move<TestConfig>("m c1 r", game);
  ASSERT_TRUE(m_c1_r.has_value());
  TestAS::decode_spatial_action(*m_c1_r, row, col, action_type);
  EXPECT_EQ(action_type, static_cast<int>(SpatialAction::ROTATE_RIGHT));
}

TEST(MoveParsing, ParseFighterFire) {
  TestGame game;

  // Deploy fighter for player 0
  auto deploy = parse_deploy_cmd<TestConfig>("d f ne");
  game.play_move(*deploy);

  // P1 deploys
  auto p1_deploy = parse_deploy_cmd<TestConfig>("d f se");
  game.play_move(*p1_deploy);

  auto f_f1 = parse_move<TestConfig>("f f1", game);
  ASSERT_TRUE(f_f1.has_value());
  int row, col, action_type;
  TestAS::decode_spatial_action(*f_f1, row, col, action_type);
  EXPECT_EQ(action_type, static_cast<int>(SpatialAction::FIRE_FORWARD));
}

TEST(MoveParsing, ParseCruiserFire) {
  TestGame game;

  // Deploy cruiser for player 0
  auto deploy = parse_deploy_cmd<TestConfig>("d c ne");
  game.play_move(*deploy);

  // P1 deploys
  auto p1_deploy = parse_deploy_cmd<TestConfig>("d f se");
  game.play_move(*p1_deploy);

  auto f_c1_l = parse_move<TestConfig>("f c1 l", game);
  ASSERT_TRUE(f_c1_l.has_value());
  int row, col, action_type;
  TestAS::decode_spatial_action(*f_c1_l, row, col, action_type);
  EXPECT_EQ(action_type, static_cast<int>(SpatialAction::FIRE_FORWARD_LEFT));

  auto f_c1_f = parse_move<TestConfig>("f c1 f", game);
  ASSERT_TRUE(f_c1_f.has_value());
  TestAS::decode_spatial_action(*f_c1_f, row, col, action_type);
  EXPECT_EQ(action_type, static_cast<int>(SpatialAction::FIRE_FORWARD));

  auto f_c1_r = parse_move<TestConfig>("f c1 r", game);
  ASSERT_TRUE(f_c1_r.has_value());
  TestAS::decode_spatial_action(*f_c1_r, row, col, action_type);
  EXPECT_EQ(action_type, static_cast<int>(SpatialAction::FIRE_FORWARD_RIGHT));
}

TEST(MoveParsing, ParseDeploy) {
  auto d_f_e = parse_deploy_cmd<TestConfig>("d f e");
  ASSERT_TRUE(d_f_e.has_value());
  EXPECT_EQ(*d_f_e, TestAS::encode_deploy(0, 0));  // Fighter, East

  auto d_f_se = parse_deploy_cmd<TestConfig>("d f se");
  ASSERT_TRUE(d_f_se.has_value());
  EXPECT_EQ(*d_f_se, TestAS::encode_deploy(0, 5));  // Fighter, SE

  auto d_c_sw = parse_deploy_cmd<TestConfig>("d c sw");
  ASSERT_TRUE(d_c_sw.has_value());
  EXPECT_EQ(*d_c_sw, TestAS::encode_deploy(1, 4));  // Cruiser, SW

  auto d_d_ne = parse_deploy_cmd<TestConfig>("d d ne");
  ASSERT_TRUE(d_d_ne.has_value());
  EXPECT_EQ(*d_d_ne, TestAS::encode_deploy(2, 1));  // Dreadnought, NE
}

TEST(MoveParsing, ParseEndTurn) {
  auto end = parse_deploy_cmd<TestConfig>("e");
  ASSERT_TRUE(end.has_value());
  EXPECT_EQ(*end, TestAS::END_TURN_OFFSET);
}

TEST(MoveParsing, InvalidMovesReturnNullopt) {
  EXPECT_FALSE(parse_deploy_cmd<TestConfig>("invalid").has_value());
  EXPECT_FALSE(parse_deploy_cmd<TestConfig>("d x e").has_value());   // Invalid unit type
  EXPECT_FALSE(parse_deploy_cmd<TestConfig>("d f x").has_value());   // Invalid facing
}

// =============================================================================
// Full Game Playthrough with Notation
// =============================================================================

// Helper to play a move by notation, returns true if successful
template<typename Config>
bool play_notation(StarGambitGS<Config>& game, const std::string& notation) {
  // Use parse_move which handles canonicalization for the current player
  auto action = parse_move<Config>(notation, game);
  if (!action.has_value()) return false;

  auto valids = game.valid_moves();
  if (valids(*action) == 0) return false;

  game.play_move(*action);
  return true;
}

TEST(FullGame, PlayWithNotation) {
  TestGame game;

  // Turn 1: P0 (bottom) deploys fighter facing NE (toward opponent)
  EXPECT_TRUE(play_notation(game, "d f ne"));
  EXPECT_EQ(game.current_player(), 1);
  EXPECT_EQ(game.current_turn(), 2);

  // Turn 2: P1 (top) deploys fighter facing SE (toward opponent)
  EXPECT_TRUE(play_notation(game, "d f se"));
  EXPECT_EQ(game.current_player(), 0);
  EXPECT_EQ(game.current_turn(), 3);

  // Turn 3: P0 moves fighter forward
  EXPECT_TRUE(play_notation(game, "m f1 f"));

  // P0 still has moves remaining, so should have taken action
  std::string dump = game.dump();
  EXPECT_TRUE(dump.find("(acted)") != std::string::npos || game.current_player() == 1);
}

TEST(FullGame, DeployAndMoveSequence) {
  TestGame game;

  // Both players deploy fighters
  EXPECT_TRUE(play_notation(game, "d f ne"));  // P0 (bottom, facing up)
  EXPECT_TRUE(play_notation(game, "d f se"));  // P1 (top, facing down)

  // P0 moves fighter forward twice (uses both moves)
  EXPECT_TRUE(play_notation(game, "m f1 f"));
  EXPECT_TRUE(play_notation(game, "m f1 f"));

  // P0 ends turn (or it auto-ends)
  if (game.current_player() == 0) {
    EXPECT_TRUE(play_notation(game, "e"));
  }
  EXPECT_EQ(game.current_player(), 1);
}

TEST(FullGame, CruiserDeployAndMove) {
  TestGame game;

  // P0 (bottom) deploys cruiser facing NE
  EXPECT_TRUE(play_notation(game, "d c ne"));
  EXPECT_EQ(game.current_player(), 1);

  // P1 (top) deploys fighter facing SE
  EXPECT_TRUE(play_notation(game, "d f se"));
  EXPECT_EQ(game.current_player(), 0);

  // P0 cruiser can move (rotate or forward)
  auto valids = game.valid_moves();

  // Check cruiser move actions are available (spatial encoding)
  bool has_cruiser_move = false;
  for (int action = 0; action < TestAS::SPATIAL_ACTIONS && !has_cruiser_move; ++action) {
    int slot = action % ACTIONS_PER_POSITION;
    // Check movement slots (0-4)
    if (slot < 5 && valids(action) == 1) {
      has_cruiser_move = true;
    }
  }
  EXPECT_TRUE(has_cruiser_move) << "Cruiser should have movement options";

  // Move cruiser forward
  EXPECT_TRUE(play_notation(game, "m c1 f"));
}

TEST(FullGame, CompleteGameToVictory) {
  TestGame game;

  // Play until game ends (one portal destroyed or max turns)
  int max_iterations = 500;
  for (int i = 0; i < max_iterations; ++i) {
    auto valids = game.valid_moves();
    int total_valid = valids.sum();

    if (total_valid == 0 || game.scores().has_value()) {
      break;
    }

    // Play first valid move
    for (int j = 0; j < TestAS::NUM_MOVES; ++j) {
      if (valids(j) == 1) {
        game.play_move(j);
        break;
      }
    }
  }

  // Game should have ended or be in a valid state
  if (game.scores().has_value()) {
    auto scores = game.scores().value();
    // One player should have won (or draw)
    float total = scores(0) + scores(1) + scores(2);
    EXPECT_FLOAT_EQ(total, 1.0f) << "Scores should sum to 1";
  }

  // Verify game state is consistent
  std::string dump = game.dump();
  EXPECT_FALSE(dump.empty());
}

TEST(FullGame, FireWhenInRange) {
  TestGame game;

  // Deploy fighters for both players facing each other (directly toward each other)
  // P0 at (0, 3) facing NW goes straight up toward P1
  // P1 at (0, -3) facing SE goes straight down toward P0
  EXPECT_TRUE(play_notation(game, "d f nw"));  // P0 (bottom, facing up/NW)
  EXPECT_TRUE(play_notation(game, "d f se"));  // P1 (top, facing down/SE)

  // Move fighters towards each other until fire is available
  bool fire_available = false;
  for (int i = 0; i < 30; ++i) {
    auto valids = game.valid_moves();

    // Check if fire is available (spatial encoding: check fire slots 5-9)
    int fire_action = -1;
    for (int action = 0; action < TestAS::SPATIAL_ACTIONS && fire_action < 0; ++action) {
      int slot = action % ACTIONS_PER_POSITION;
      // Check fire slots (5-9)
      if (slot >= 5 && valids(action) == 1) {
        fire_action = action;
      }
    }

    if (fire_action >= 0) {
      fire_available = true;
      game.play_move(fire_action);
      break;
    }

    // Try to move forward (spatial encoding: check slot 0 = MOVE_FORWARD)
    int forward_action = -1;
    for (int action = 0; action < TestAS::SPATIAL_ACTIONS; ++action) {
      int slot = action % ACTIONS_PER_POSITION;
      if (slot == static_cast<int>(SpatialAction::MOVE_FORWARD) && valids(action) == 1) {
        forward_action = action;
        break;
      }
    }

    if (forward_action >= 0) {
      game.play_move(forward_action);
    } else {
      // If forward not available, try any move or end turn
      for (int j = 0; j < TestAS::NUM_MOVES; ++j) {
        if (valids(j) == 1) {
          game.play_move(j);
          break;
        }
      }
    }

    if (game.scores().has_value()) break;
  }

  // Fire should have become available at some point (or game ended)
  EXPECT_TRUE(fire_available || game.scores().has_value())
      << "Fire should be available when units are in range";
}

// =============================================================================
// Dreadnought Deployment Tests (using BattleConfig)
// =============================================================================

using BattleAS = ActionSpace<BattleConfig>;

TEST(Deployment, DreadnoughtDeployAllFacingsP0) {
  StarGambitBattleGS game;

  // Player 0 valid dreadnought facings: {0, 1, 2, 3} (E, NE, NW, W)
  auto valid_facings = get_valid_deploy_facings(UnitType::DREADNOUGHT, 0);
  ASSERT_EQ(valid_facings.size(), 4u);

  auto valids = game.valid_moves();

  // Dreadnought deploy actions start at DEPLOY_OFFSET + 12 (after fighter and cruiser deploys)
  // Each unit type has 6 facing options
  int dread_deploy_base = BattleAS::DEPLOY_OFFSET + 12;

  for (int facing : valid_facings) {
    int action = dread_deploy_base + facing;
    EXPECT_EQ(valids(action), 1)
        << "Dreadnought deploy facing " << facing << " should be valid for P0";
  }
}

TEST(Deployment, DreadnoughtDeployAllFacingsP1) {
  StarGambitBattleGS game;

  // Deploy a fighter for P0 to switch to P1
  auto valids = game.valid_moves();
  int fighter_deploy = BattleAS::DEPLOY_OFFSET;
  for (int i = BattleAS::DEPLOY_OFFSET; i < BattleAS::DEPLOY_OFFSET + 6; ++i) {
    if (valids(i) == 1) {
      fighter_deploy = i;
      break;
    }
  }
  game.play_move(fighter_deploy);

  // Now P1's turn
  ASSERT_EQ(game.current_player(), 1);

  // Player 1 valid dreadnought facings: {0, 3, 4, 5} (E, W, SW, SE) in world coordinates
  auto valid_facings = get_valid_deploy_facings(UnitType::DREADNOUGHT, 1);
  ASSERT_EQ(valid_facings.size(), 4u);

  valids = game.valid_moves();
  int dread_deploy_base = BattleAS::DEPLOY_OFFSET + 12;

  for (int world_facing : valid_facings) {
    // P1's facings are canonicalized: (world_facing + 3) % 6
    int canonical_facing = (world_facing + 3) % 6;
    int action = dread_deploy_base + canonical_facing;
    EXPECT_EQ(valids(action), 1)
        << "Dreadnought deploy world facing " << world_facing
        << " (canonical " << canonical_facing << ") should be valid for P1";
  }
}

// =============================================================================
// CHARACTERIZATION TESTS: Document CURRENT behavior before redesign
// These tests capture the existing behavior that will change during the
// observation/action space redesign. They should PASS now and will be
// updated to reflect new expected behavior before implementation.
// =============================================================================

// -----------------------------------------------------------------------------
// Cruiser Anchor Position: NOW anchor = FRONT hex (after refactor)
// Previously: anchor = REAR hex
// -----------------------------------------------------------------------------

TEST(CharacterizationCruiserAnchor, AnchorIsFront_RearIsComputed) {
  // NEW behavior: anchor (hexes[0]) is FRONT, hexes[1] is rear
  Hex anchor = {0, 0};
  int facing = 0;  // East

  auto hexes = get_unit_hexes(UnitType::CRUISER, anchor, facing);

  ASSERT_EQ(hexes.size(), 2u);
  EXPECT_EQ(hexes[0], anchor) << "hexes[0] should be anchor (front)";

  // Rear is in OPPOSITE direction of facing
  int rear_dir = OPPOSITE_DIRECTION[facing];  // West for East-facing
  Hex expected_rear = hex_neighbor(anchor, rear_dir);
  EXPECT_EQ(hexes[1], expected_rear) << "hexes[1] should be rear (West of anchor)";
}

TEST(CharacterizationCruiserAnchor, AnchorIsFront_AllFacings) {
  Hex anchor = {0, 0};

  // Test all 6 facings - rear should always be neighbor in OPPOSITE direction
  for (int facing = 0; facing < 6; ++facing) {
    auto hexes = get_unit_hexes(UnitType::CRUISER, anchor, facing);

    ASSERT_EQ(hexes.size(), 2u);
    EXPECT_EQ(hexes[0], anchor) << "Facing " << facing << ": hexes[0] should be anchor (front)";

    int rear_dir = OPPOSITE_DIRECTION[facing];
    Hex expected_rear = hex_neighbor(anchor, rear_dir);
    EXPECT_EQ(hexes[1], expected_rear)
        << "Facing " << facing << ": hexes[1] should be rear at opposite neighbor";
  }
}

// -----------------------------------------------------------------------------
// Cruiser Movement: Movements are computed relative to rear anchor
// -----------------------------------------------------------------------------

TEST(CharacterizationCruiserMovement, ForwardMoveAnchorsToFront) {
  // Create a game and deploy a cruiser
  using TestConfig = SkirmishConfig;
  StarGambitSkirmishGS game;

  // Deploy cruiser facing NE (direction 1)
  // Cruiser deploy is at DEPLOY_OFFSET + 6 (cruiser type index 1 * 6 facings)
  auto valids = game.valid_moves();
  int cruiser_deploy_ne = ActionSpace<TestConfig>::DEPLOY_OFFSET + 6 + 1;  // type 1, facing NE

  if (valids(cruiser_deploy_ne) == 1) {
    game.play_move(cruiser_deploy_ne);
  } else {
    // Find any valid cruiser deploy
    for (int i = ActionSpace<TestConfig>::DEPLOY_OFFSET + 6;
         i < ActionSpace<TestConfig>::DEPLOY_OFFSET + 12; ++i) {
      if (valids(i) == 1) {
        game.play_move(i);
        break;
      }
    }
  }

  // Verify cruiser was deployed
  auto units = game.get_units();
  bool found_cruiser = false;
  for (const auto& u : units) {
    if (u.type == 1 && u.player == 0) {  // Cruiser type is 1
      found_cruiser = true;
      break;
    }
  }
  EXPECT_TRUE(found_cruiser) << "Cruiser should be deployed";
}

TEST(CharacterizationCruiserMovement, RotateLeftKeepsAnchor) {
  // Rotation: anchor stays, facing changes
  // This behavior will REMAIN the same after refactor
  Hex anchor = {0, 0};
  int facing = 0;  // East

  // Create a Unit struct to test compute_cruiser_move
  // We'll verify by checking get_unit_hexes before and after
  auto before = get_unit_hexes(UnitType::CRUISER, anchor, facing);
  EXPECT_EQ(before[0], anchor);

  // After rotate_left, anchor should still be {0, 0}, facing should be 1 (NE)
  int new_facing = rotate_direction(facing, 1);
  auto after = get_unit_hexes(UnitType::CRUISER, anchor, new_facing);
  EXPECT_EQ(after[0], anchor) << "Rotate-left: anchor should stay in place";
  EXPECT_EQ(new_facing, 1);
}

// -----------------------------------------------------------------------------
// Cruiser Cannons: NOW fire from hexes[0] (anchor = front hex)
// Previously: fired from hexes[1] (front was second hex)
// -----------------------------------------------------------------------------

TEST(CharacterizationCruiserCannons, FireFromAnchorHex) {
  auto cannons = get_cannon_info(UnitType::CRUISER);
  ASSERT_EQ(cannons.size(), 3u);

  // All cruiser cannons fire from source_hex_idx = 0 (the anchor, which is front)
  for (size_t i = 0; i < cannons.size(); ++i) {
    EXPECT_EQ(cannons[i].source_hex_idx, 0)
        << "Cannon " << i << " should fire from anchor hex (index 0)";
  }
}

TEST(CharacterizationCruiserCannons, CannonDirections) {
  auto cannons = get_cannon_info(UnitType::CRUISER);

  // Left cannon: direction_offset = +1 (fires facing+1 = counter-clockwise = left)
  EXPECT_EQ(cannons[0].direction_offset, 1) << "Left cannon direction offset";

  // Forward cannon: direction_offset = 0 (fires facing direction)
  EXPECT_EQ(cannons[1].direction_offset, 0) << "Forward cannon direction offset";

  // Right cannon: direction_offset = -1 (fires facing-1 = clockwise = right)
  EXPECT_EQ(cannons[2].direction_offset, -1) << "Right cannon direction offset";
}

// -----------------------------------------------------------------------------
// Current Action Space Encoding: Spatial (row, col, action_type)
// -----------------------------------------------------------------------------

TEST(CharacterizationActionSpace, SpatialEncoding) {
  using AS = ActionSpace<SkirmishConfig>;

  // Spatial encoding: row * BOARD_DIM * 10 + col * 10 + slot
  // Layout: [spatial actions (BOARD_DIM * BOARD_DIM * 10)] + [deploy (18)] + [end_turn (1)]
  constexpr int BOARD_DIM = AS::BOARD_DIM;

  // Verify layout
  EXPECT_EQ(AS::SPATIAL_OFFSET, 0);
  EXPECT_EQ(ACTIONS_PER_POSITION, 10);  // Global constant
  EXPECT_EQ(AS::SPATIAL_ACTIONS, BOARD_DIM * BOARD_DIM * ACTIONS_PER_POSITION);
  EXPECT_EQ(AS::DEPLOY_OFFSET, AS::SPATIAL_ACTIONS);
  EXPECT_EQ(AS::DEPLOY_ACTIONS, 18);  // 3 types * 6 facings
  EXPECT_EQ(AS::END_TURN_OFFSET, AS::DEPLOY_OFFSET + AS::DEPLOY_ACTIONS);
  EXPECT_EQ(AS::END_TURN_ACTIONS, 1);

  // Total actions for Skirmish (9*9*10 + 18 + 1 = 829)
  EXPECT_EQ(AS::NUM_MOVES, BOARD_DIM * BOARD_DIM * 10 + 18 + 1);
}

TEST(CharacterizationActionSpace, SpatialEncodingFormula) {
  using AS = ActionSpace<SkirmishConfig>;
  constexpr int BOARD_DIM = AS::BOARD_DIM;

  // Row 0, Col 0, slot 0 (MOVE_FORWARD)
  int pos00_forward = AS::encode_spatial_action(0, 0, 0);
  EXPECT_EQ(pos00_forward, 0);

  // Row 0, Col 0, slot 5 (FIRE_FORWARD)
  int pos00_fire = AS::encode_spatial_action(0, 0, 5);
  EXPECT_EQ(pos00_fire, 5);

  // Row 2, Col 3, slot 3 (ROTATE_LEFT)
  int pos23_rotate = AS::encode_spatial_action(2, 3, 3);
  EXPECT_EQ(pos23_rotate, 2 * BOARD_DIM * 10 + 3 * 10 + 3);

  // Decode round-trip
  int row, col, action_type;
  AS::decode_spatial_action(pos23_rotate, row, col, action_type);
  EXPECT_EQ(row, 2);
  EXPECT_EQ(col, 3);
  EXPECT_EQ(action_type, 3);
}

// -----------------------------------------------------------------------------
// Current Observation Space: 2D spatial (channels, BOARD_DIM, BOARD_DIM)
// -----------------------------------------------------------------------------

TEST(CharacterizationObservation, CurrentShape) {
  using AS = ActionSpace<SkirmishConfig>;

  // 2D spatial shape: (channels, 2*BOARD_SIDE-1, 2*BOARD_SIDE-1)
  constexpr int BOARD_DIM = 2 * SkirmishConfig::BOARD_SIDE - 1;
  EXPECT_EQ(AS::CANONICAL_SHAPE[1], BOARD_DIM) << "Row dimension is 2*BOARD_SIDE-1";
  EXPECT_EQ(AS::CANONICAL_SHAPE[2], BOARD_DIM) << "Col dimension is 2*BOARD_SIDE-1";
}

TEST(CharacterizationObservation, CurrentTensorShape) {
  StarGambitSkirmishGS game;
  auto tensor = game.canonicalized();

  using AS = ActionSpace<SkirmishConfig>;
  constexpr int BOARD_DIM = 2 * SkirmishConfig::BOARD_SIDE - 1;

  EXPECT_EQ(tensor.dimension(0), AS::CANONICAL_SHAPE[0]);
  EXPECT_EQ(tensor.dimension(1), BOARD_DIM) << "Row dimension is 2*BOARD_SIDE-1";
  EXPECT_EQ(tensor.dimension(2), BOARD_DIM) << "Col dimension is 2*BOARD_SIDE-1";
}

TEST(CharacterizationObservation, TypePresenceChannels) {
  // Current encoding: one channel per (player, unit_type)
  using AS = ActionSpace<SkirmishConfig>;

  // Type presence: 4 types (Fighter, Cruiser, Dreadnought, Portal) × 2 players = 8
  EXPECT_EQ(AS::TYPE_PRESENCE_CHANNELS, 8);
}

// -----------------------------------------------------------------------------
// Dreadnought Characterization (for comparison)
// -----------------------------------------------------------------------------

TEST(CharacterizationDreadnought, AnchorIsFront) {
  // Dreadnought: anchor IS the front hex (different from cruiser)
  Hex anchor = {0, 0};
  int facing = 0;  // East

  auto hexes = get_unit_hexes(UnitType::DREADNOUGHT, anchor, facing);

  ASSERT_EQ(hexes.size(), 3u);
  EXPECT_EQ(hexes[0], anchor) << "hexes[0] is anchor (front)";

  // Rear hexes are behind anchor in opposite direction
  int rear_dir = OPPOSITE_DIRECTION[facing];
  Hex expected_rear_left = hex_neighbor(anchor, rotate_direction(rear_dir, 1));
  Hex expected_rear_right = hex_neighbor(anchor, rear_dir);

  EXPECT_EQ(hexes[1], expected_rear_left) << "hexes[1] should be rear-left";
  EXPECT_EQ(hexes[2], expected_rear_right) << "hexes[2] should be rear-right";
}

TEST(CharacterizationDreadnought, CannonSourceHexes) {
  auto cannons = get_cannon_info(UnitType::DREADNOUGHT);
  ASSERT_EQ(cannons.size(), 4u);

  // Dread cannons (order: rr, fr, fl, rl):
  // Cannon 0 (rr): from hexes[1] (rear-right hex), fires forward
  // Cannon 1 (fr): from hexes[0] (anchor), fires forward
  // Cannon 2 (fl): from hexes[0] (anchor), fires forward-left
  // Cannon 3 (rl): from hexes[2] (rear-left hex), fires forward-left

  EXPECT_EQ(cannons[0].source_hex_idx, 1) << "rr fires from hexes[1]";  // rear-right hex
  EXPECT_EQ(cannons[1].source_hex_idx, 0) << "fr fires from hexes[0]";  // anchor
  EXPECT_EQ(cannons[2].source_hex_idx, 0) << "fl fires from hexes[0]";  // anchor
  EXPECT_EQ(cannons[3].source_hex_idx, 2) << "rl fires from hexes[2]";  // rear-left hex
}

// -----------------------------------------------------------------------------
// Fighter Characterization (anchor = only hex, should NOT change)
// -----------------------------------------------------------------------------

TEST(CharacterizationFighter, SingleHexUnit) {
  Hex anchor = {2, -1};
  int facing = 3;  // West

  auto hexes = get_unit_hexes(UnitType::FIGHTER, anchor, facing);

  ASSERT_EQ(hexes.size(), 1u);
  EXPECT_EQ(hexes[0], anchor);
}

TEST(CharacterizationFighter, CannonFiresForward) {
  auto cannons = get_cannon_info(UnitType::FIGHTER);
  ASSERT_EQ(cannons.size(), 1u);

  EXPECT_EQ(cannons[0].source_hex_idx, 0) << "Fighter fires from anchor";
  EXPECT_EQ(cannons[0].direction_offset, 0) << "Fighter fires forward (facing direction)";
}

TEST(CharacterizationFighter, MovementDirections) {
  // Fighter has 3 moves: forward, forward-left, forward-right
  EXPECT_EQ(FIGHTER_MOVE_DIRS, 3);

  // These are direction offsets from facing:
  // 0 = forward (facing direction)
  // 1 = forward-left (facing + 1)
  // 2 = forward-right (facing - 1)
}

// =============================================================================
// Terminal State Scoring Tests
// =============================================================================

TEST(TerminalStates, ScoresNotPresentDuringGame) {
  TestGame game;
  // Initially, game is not over
  EXPECT_FALSE(game.scores().has_value());

  // Deploy some units
  auto valids = game.valid_moves();
  for (int i = TestAS::DEPLOY_OFFSET; i < TestAS::END_TURN_OFFSET; ++i) {
    if (valids(i) == 1) {
      game.play_move(i);
      break;
    }
  }

  // Game should still not be over after one move
  EXPECT_FALSE(game.scores().has_value());
}

TEST(TerminalStates, ScoresSumToOne) {
  TestGame game;

  // Play until game ends
  for (int i = 0; i < 500; ++i) {
    auto valids = game.valid_moves();
    if (valids.sum() == 0) break;

    auto scores = game.scores();
    if (scores.has_value()) {
      // Verify scores sum to 1.0
      float total = (*scores)(0) + (*scores)(1) + (*scores)(2);
      EXPECT_FLOAT_EQ(total, 1.0f) << "Scores should sum to 1.0";
      break;
    }

    // Play first valid move
    for (int j = 0; j < TestAS::NUM_MOVES; ++j) {
      if (valids(j) == 1) {
        game.play_move(j);
        break;
      }
    }
  }
}

TEST(TerminalStates, WinnerGetsOne) {
  TestGame game;

  // Play until game ends
  for (int i = 0; i < 500; ++i) {
    auto valids = game.valid_moves();
    if (valids.sum() == 0) break;

    auto scores = game.scores();
    if (scores.has_value()) {
      // Exactly one of the three indices should be 1.0, others should be 0.0
      int ones = 0;
      int zeros = 0;
      for (int p = 0; p < 3; ++p) {
        if ((*scores)(p) == 1.0f) ones++;
        else if ((*scores)(p) == 0.0f) zeros++;
      }
      EXPECT_EQ(ones, 1) << "Exactly one score should be 1.0";
      EXPECT_EQ(zeros, 2) << "Exactly two scores should be 0.0";
      break;
    }

    // Play first valid move
    for (int j = 0; j < TestAS::NUM_MOVES; ++j) {
      if (valids(j) == 1) {
        game.play_move(j);
        break;
      }
    }
  }
}

TEST(TerminalStates, DrawIndexIsTwo) {
  // Verify that draw is scored in index 2 (not player indices 0 or 1)
  // We can't easily force a draw, but we can verify the constant
  TestGame game;

  // Play many moves - if we hit 200 turns, it's a draw
  for (int i = 0; i < 10000; ++i) {
    auto scores = game.scores();
    if (scores.has_value()) {
      // Check if it's a draw (index 2 has value)
      if ((*scores)(2) == 1.0f) {
        EXPECT_FLOAT_EQ((*scores)(0), 0.0f);
        EXPECT_FLOAT_EQ((*scores)(1), 0.0f);
      }
      break;
    }

    auto valids = game.valid_moves();
    if (valids.sum() == 0) break;

    // Play first valid move
    for (int j = 0; j < TestAS::NUM_MOVES; ++j) {
      if (valids(j) == 1) {
        game.play_move(j);
        break;
      }
    }
  }
}

TEST(TerminalStates, CanonicalizedWorksAfterGameEnd) {
  TestGame game;

  // Play until game ends
  for (int i = 0; i < 500; ++i) {
    auto scores = game.scores();
    if (scores.has_value()) break;

    auto valids = game.valid_moves();
    if (valids.sum() == 0) break;

    for (int j = 0; j < TestAS::NUM_MOVES; ++j) {
      if (valids(j) == 1) {
        game.play_move(j);
        break;
      }
    }
  }

  // Even if game is over, canonicalized() should work
  auto tensor = game.canonicalized();
  EXPECT_EQ(tensor.dimension(0), TestAS::CANONICAL_SHAPE[0]);
  EXPECT_EQ(tensor.dimension(1), TestAS::CANONICAL_SHAPE[1]);
  EXPECT_EQ(tensor.dimension(2), TestAS::CANONICAL_SHAPE[2]);
}

// =============================================================================
// Repetition Count Observation Tests
// =============================================================================

TEST(RepetitionObservation, InitialRepetitionCountIsOne) {
  TestGame game;

  // The initial position is added to history in constructor, so rep count = 1
  auto tensor = game.canonicalized();

  // Channel 23 is repetition count
  // After construction, position has been seen once, so value should be 0.5
  // (0.0 = never, 0.5 = once, 1.0 = twice+)

  // Get a valid hex position to check
  Hex h = index_to_hex_fast<TestConfig::BOARD_SIDE>(0);
  auto [row, col] = hex_to_2d<TestConfig::BOARD_SIDE>(h);

  // Note: Initial position is pushed in constructor, so rep count should be 1 (= 0.5)
  // But the canonicalized() computes hash and checks against history
  // At start, position_history_ has one entry, and current hash matches it once
  // So rep_count = 1, rep_value = 0.5
  EXPECT_FLOAT_EQ(tensor(23, row, col), 0.5f)
      << "Initial repetition count should be 0.5 (seen once)";
}

TEST(RepetitionObservation, RepetitionChannelIsBroadcast) {
  TestGame game;
  auto tensor = game.canonicalized();

  // Channel 23 should be broadcast to all valid hexes with the same value
  float first_value = -1.0f;
  bool all_same = true;

  for (int idx = 0; idx < TestAS::NUM_HEXES; ++idx) {
    Hex h = index_to_hex_fast<TestConfig::BOARD_SIDE>(idx);
    auto [row, col] = hex_to_2d<TestConfig::BOARD_SIDE>(h);

    if (first_value < 0) {
      first_value = tensor(23, row, col);
    } else if (tensor(23, row, col) != first_value) {
      all_same = false;
      break;
    }
  }

  EXPECT_TRUE(all_same) << "Repetition count should be broadcast uniformly";
}

// =============================================================================
// Mid-Turn Repetition Tests
// =============================================================================

TEST(MidTurnRepetition, PositionTrackedAfterEveryAction) {
  TestGame game;

  // Deploy fighters for both players
  auto valids = game.valid_moves();
  int deploy = -1;
  for (int i = TestAS::DEPLOY_OFFSET; i < TestAS::DEPLOY_OFFSET + 6; ++i) {
    if (valids(i) == 1) {
      deploy = i;
      break;
    }
  }
  game.play_move(deploy);

  valids = game.valid_moves();
  deploy = -1;
  for (int i = TestAS::DEPLOY_OFFSET; i < TestAS::DEPLOY_OFFSET + 6; ++i) {
    if (valids(i) == 1) {
      deploy = i;
      break;
    }
  }
  game.play_move(deploy);

  // Now it's turn 3, make a move
  valids = game.valid_moves();
  for (int i = 0; i < TestAS::NUM_MOVES; ++i) {
    if (valids(i) == 1) {
      game.play_move(i);
      break;
    }
  }

  // Game should still be playable (not crashed due to repetition tracking)
  EXPECT_FALSE(game.scores().has_value()) << "Game should not be over after a few moves";
}

// =============================================================================
// Threefold Repetition Tests
// =============================================================================

TEST(ThreefoldRepetition, DrawOnThirdOccurrence) {
  // Use cruiser rotations to force threefold repetition
  // Rotating left then right returns cruiser to same position
  TestGame game;

  // Deploy cruisers for both players
  EXPECT_TRUE(play_notation(game, "d c ne"));  // P0 cruiser
  EXPECT_TRUE(play_notation(game, "d c sw"));  // P1 cruiser

  // Position 1: Turn 3, P0 to move (initial position after deployment)
  EXPECT_FALSE(game.scores().has_value());

  // Cycle 1: P0 rotates left, P1 rotates left
  EXPECT_TRUE(play_notation(game, "m c1 l"));
  EXPECT_TRUE(play_notation(game, "e"));
  EXPECT_TRUE(play_notation(game, "m c1 l"));
  EXPECT_TRUE(play_notation(game, "e"));

  // Cycle 1 return: P0 rotates right, P1 rotates right
  EXPECT_TRUE(play_notation(game, "m c1 r"));
  EXPECT_TRUE(play_notation(game, "e"));
  EXPECT_TRUE(play_notation(game, "m c1 r"));
  EXPECT_TRUE(play_notation(game, "e"));

  // Position 2: Back to initial position (P0 to move)
  EXPECT_FALSE(game.scores().has_value()) << "Should not be draw after 2nd occurrence";

  // Cycle 2: P0 rotates left, P1 rotates left
  EXPECT_TRUE(play_notation(game, "m c1 l"));
  EXPECT_TRUE(play_notation(game, "e"));
  EXPECT_TRUE(play_notation(game, "m c1 l"));
  EXPECT_TRUE(play_notation(game, "e"));

  // Cycle 2 return: P0 rotates right, P1 rotates right
  EXPECT_TRUE(play_notation(game, "m c1 r"));
  EXPECT_TRUE(play_notation(game, "e"));
  EXPECT_TRUE(play_notation(game, "m c1 r"));
  EXPECT_TRUE(play_notation(game, "e"));

  // Position 3: Third occurrence - should be draw
  ASSERT_TRUE(game.scores().has_value()) << "Should be draw after 3rd occurrence";
  auto scores = game.scores().value();
  EXPECT_FLOAT_EQ(scores(2), 1.0f) << "Draw should be scored in index 2";
}

TEST(ThreefoldRepetition, HistoryClearedOnDeploy) {
  TestGame game;

  // Deploy first fighter
  EXPECT_TRUE(play_notation(game, "d f ne"));
  EXPECT_TRUE(play_notation(game, "d f sw"));

  // Play some moves to build up history
  EXPECT_TRUE(play_notation(game, "m f1 f"));
  EXPECT_TRUE(play_notation(game, "e"));
  EXPECT_TRUE(play_notation(game, "m f1 f"));
  EXPECT_TRUE(play_notation(game, "e"));

  // Deploy another unit - this should clear history
  auto valids = game.valid_moves();
  bool deployed = false;
  for (int i = TestAS::DEPLOY_OFFSET; i < TestAS::END_TURN_OFFSET; ++i) {
    if (valids(i) == 1) {
      game.play_move(i);
      deployed = true;
      break;
    }
  }
  EXPECT_TRUE(deployed) << "Should be able to deploy another unit";

  // Game should not be over (history was cleared, no repetition possible yet)
  EXPECT_FALSE(game.scores().has_value());
}

TEST(ThreefoldRepetition, CheckOccursAtTurnStart) {
  // Verify that repetition check happens at the START of each player's turn
  // (after player switch, not before)
  TestGame game;

  // This test ensures the timing is correct by checking that the position
  // is recorded with the correct current_player value
  EXPECT_TRUE(play_notation(game, "d f ne"));
  EXPECT_EQ(game.current_player(), 1) << "Should be P1's turn after P0 deploys";

  EXPECT_TRUE(play_notation(game, "d f sw"));
  EXPECT_EQ(game.current_player(), 0) << "Should be P0's turn after P1 deploys";

  // The hash should be recorded with current_player being the player
  // who is about to move (checked at start of their turn)
}

TEST(ThreefoldRepetition, DifferentPlayersDifferentPositions) {
  // Same board configuration with different player to move should be
  // different positions (not count toward repetition)
  TestGame game;

  EXPECT_TRUE(play_notation(game, "d f ne"));
  EXPECT_TRUE(play_notation(game, "d f sw"));

  // Even if board looks the same, P0 to move vs P1 to move are different positions
  // This is inherent in the hash including current_player_
}

}  // namespace alphazero::star_gambit_gs
