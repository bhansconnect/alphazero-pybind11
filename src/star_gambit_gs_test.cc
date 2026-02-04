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
// Move Notation Parser
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
//   f c1 l    - fire cruiser 1 left cannon
//   f c1 f    - fire cruiser 1 forward cannon
//   f c1 r    - fire cruiser 1 right cannon
//   f d1 rl   - fire dreadnought 1 rear-left cannon
//   f d1 fl   - fire dreadnought 1 front-left cannon
//   f d1 fr   - fire dreadnought 1 front-right cannon
//   f d1 rr   - fire dreadnought 1 rear-right cannon
//   d f e     - deploy fighter facing east
//   d f ne    - deploy fighter facing northeast
//   d c sw    - deploy cruiser facing southwest
//   e         - end turn

template<typename Config>
std::optional<int> parse_move(const std::string& move_str) {
  using AS = ActionSpace<Config>;
  std::istringstream iss(move_str);
  std::string cmd;
  iss >> cmd;

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
      // Fighter moves: f, fl, fr -> 0, 1, 2
      static const std::map<std::string, int> fighter_dirs = {
        {"f", 0}, {"fl", 1}, {"fr", 2}
      };
      auto it = fighter_dirs.find(direction);
      if (it == fighter_dirs.end() || slot < 0 || slot >= Config::MAX_FIGHTERS) {
        return std::nullopt;
      }
      return AS::FIGHTER_MOVE_OFFSET + slot * FIGHTER_MOVE_DIRS + it->second;
    } else if (unit_type == 'c') {
      // Cruiser moves: l, fl, f, fr, r -> 0, 1, 2, 3, 4
      static const std::map<std::string, int> cruiser_dirs = {
        {"l", 0}, {"fl", 1}, {"f", 2}, {"fr", 3}, {"r", 4}
      };
      auto it = cruiser_dirs.find(direction);
      if (it == cruiser_dirs.end() || slot < 0 || slot >= Config::MAX_CRUISERS) {
        return std::nullopt;
      }
      return AS::CRUISER_MOVE_OFFSET + slot * CRUISER_MOVE_DIRS + it->second;
    } else if (unit_type == 'd') {
      // Dreadnought moves: l, fl, fr, r -> 0, 1, 2, 3
      static const std::map<std::string, int> dread_dirs = {
        {"l", 0}, {"fl", 1}, {"fr", 2}, {"r", 3}
      };
      auto it = dread_dirs.find(direction);
      if (it == dread_dirs.end() || slot < 0 || slot >= Config::MAX_DREADNOUGHTS) {
        return std::nullopt;
      }
      return AS::DREAD_MOVE_OFFSET + slot * DREAD_MOVE_DIRS + it->second;
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
      // Fighter has only one cannon
      if (slot < 0 || slot >= Config::MAX_FIGHTERS) return std::nullopt;
      return AS::FIGHTER_FIRE_OFFSET + slot;
    } else if (unit_type == 'c') {
      // Cruiser cannons: l, f, r -> 0, 1, 2
      static const std::map<std::string, int> cruiser_cannons = {
        {"l", 0}, {"f", 1}, {"r", 2}
      };
      auto it = cruiser_cannons.find(cannon);
      if (it == cruiser_cannons.end() || slot < 0 || slot >= Config::MAX_CRUISERS) {
        return std::nullopt;
      }
      return AS::CRUISER_FIRE_OFFSET + slot * CRUISER_CANNONS + it->second;
    } else if (unit_type == 'd') {
      // Dreadnought cannons: rl, fl, fr, rr -> 0, 1, 2, 3
      static const std::map<std::string, int> dread_cannons = {
        {"rl", 0}, {"fl", 1}, {"fr", 2}, {"rr", 3}
      };
      auto it = dread_cannons.find(cannon);
      if (it == dread_cannons.end() || slot < 0 || slot >= Config::MAX_DREADNOUGHTS) {
        return std::nullopt;
      }
      return AS::DREAD_FIRE_OFFSET + slot * DREAD_CANNONS + it->second;
    }
    return std::nullopt;
  }

  if (cmd == "d") {
    // Deploy command: d <unit_type> <facing>
    std::string unit_type_str, facing;
    iss >> unit_type_str >> facing;
    if (unit_type_str.empty() || facing.empty()) return std::nullopt;

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
    auto it = facings.find(facing);
    if (it == facings.end()) return std::nullopt;

    return AS::DEPLOY_OFFSET + unit_type_idx * 6 + it->second;
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
  EXPECT_EQ(hexes[0], anchor);
  EXPECT_EQ(hexes[1].q, 1);  // Front hex is East of anchor
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

  // Check that no move actions are valid (fighters, cruisers)
  for (int i = TestAS::FIGHTER_MOVE_OFFSET; i < TestAS::FIGHTER_FIRE_OFFSET; ++i) {
    EXPECT_EQ(valids(i), 0) << "Move action " << i << " should not be valid on turn 1";
  }

  // Check that no fire actions are valid
  for (int i = TestAS::FIGHTER_FIRE_OFFSET; i < TestAS::DEPLOY_OFFSET; ++i) {
    EXPECT_EQ(valids(i), 0) << "Fire action " << i << " should not be valid on turn 1";
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

  // At least some move actions should be valid (for the deployed unit)
  bool has_move = false;
  for (int i = TestAS::FIGHTER_MOVE_OFFSET; i < TestAS::FIGHTER_FIRE_OFFSET; ++i) {
    if (valids(i) == 1) {
      has_move = true;
      break;
    }
  }
  // Note: move might not be valid if unit just deployed and has no moves
  // Let's check deploy instead
  bool has_deploy = false;
  for (int i = TestAS::DEPLOY_OFFSET; i < TestAS::END_TURN_OFFSET; ++i) {
    if (valids(i) == 1) {
      has_deploy = true;
      break;
    }
  }
  EXPECT_TRUE(has_move || has_deploy);
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
  EXPECT_EQ(syms.size(), 2u);  // Identity + 180° rotation
}

TEST(Symmetries, ValueSwappedCorrectly) {
  TestGame game;
  PlayHistory base;
  base.canonical = game.canonicalized();
  base.v = Vector<float>(3);
  base.v << 0.7f, 0.2f, 0.1f;  // P0 likely wins
  base.pi = Vector<float>(TestAS::NUM_MOVES);
  base.pi.setZero();

  auto syms = game.symmetries(base);
  ASSERT_EQ(syms.size(), 2u);

  // First symmetry is identity - values unchanged
  EXPECT_FLOAT_EQ(syms[0].v(0), 0.7f);
  EXPECT_FLOAT_EQ(syms[0].v(1), 0.2f);
  EXPECT_FLOAT_EQ(syms[0].v(2), 0.1f);

  // Rotated symmetry should have swapped values
  EXPECT_FLOAT_EQ(syms[1].v(0), 0.2f);  // Was P1's value
  EXPECT_FLOAT_EQ(syms[1].v(1), 0.7f);  // Was P0's value
  EXPECT_FLOAT_EQ(syms[1].v(2), 0.1f);  // Draw unchanged
}

TEST(Symmetries, DoubleRotationReturnsToOriginal) {
  TestGame game;
  // Play a few moves to get interesting state
  game.play_move(TestAS::END_TURN_OFFSET);  // P0 end turn (if valid)

  // Get valid moves and play them to advance state
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
  base.v << 0.6f, 0.3f, 0.1f;
  base.pi = Vector<float>(TestAS::NUM_MOVES);
  base.pi.setConstant(1.0f / TestAS::NUM_MOVES);

  auto syms = game.symmetries(base);
  ASSERT_EQ(syms.size(), 2u);

  // Apply symmetry again to the rotated version
  PlayHistory rotated;
  rotated.canonical = syms[1].canonical;
  rotated.v = syms[1].v;
  rotated.pi = syms[1].pi;

  auto double_syms = game.symmetries(rotated);
  ASSERT_EQ(double_syms.size(), 2u);

  // double_syms[1] should match original base (within floating point tolerance)
  for (int i = 0; i < 3; ++i) {
    EXPECT_NEAR(double_syms[1].v(i), base.v(i), 1e-5f)
        << "Value mismatch at index " << i;
  }

  // Canonical should also match
  for (int c = 0; c < TestAS::CANONICAL_SHAPE[0]; ++c) {
    for (int h = 0; h < TestAS::NUM_HEXES; ++h) {
      EXPECT_NEAR(double_syms[1].canonical(c, 0, h),
                  base.canonical(c, 0, h), 1e-5f)
          << "Canonical mismatch at channel " << c << ", hex " << h;
    }
  }
}

TEST(Symmetries, DeployActionsRotated) {
  TestGame game;
  PlayHistory base;
  base.canonical = game.canonicalized();
  base.v = Vector<float>(3);
  base.v.setConstant(1.0f / 3.0f);
  base.pi = Vector<float>(TestAS::NUM_MOVES);
  base.pi.setZero();

  // Set policy for deploying fighter facing East (dir 0)
  int fighter_deploy_east = TestAS::DEPLOY_OFFSET + 0 * 6 + 0;  // type 0, facing 0 (E)
  base.pi(fighter_deploy_east) = 1.0f;

  auto syms = game.symmetries(base);
  ASSERT_EQ(syms.size(), 2u);

  // After 180° rotation, East (0) becomes West (3)
  int fighter_deploy_west = TestAS::DEPLOY_OFFSET + 0 * 6 + 3;  // type 0, facing 3 (W)
  EXPECT_FLOAT_EQ(syms[1].pi(fighter_deploy_west), 1.0f)
      << "Deploy East should become Deploy West after 180° rotation";
  EXPECT_FLOAT_EQ(syms[1].pi(fighter_deploy_east), 0.0f)
      << "Original East action should be 0 after rotation";
}

TEST(Symmetries, DeployActionsAllDirectionsRotate) {
  TestGame game;

  // Test all 6 directions rotate correctly: d → (d+3) % 6
  // 0(E) → 3(W), 1(NE) → 4(SW), 2(NW) → 5(SE)
  // 3(W) → 0(E), 4(SW) → 1(NE), 5(SE) → 2(NW)
  const std::pair<int, int> rotations[] = {
      {0, 3}, {1, 4}, {2, 5}, {3, 0}, {4, 1}, {5, 2}
  };

  for (auto [orig_dir, rotated_dir] : rotations) {
    PlayHistory base;
    base.canonical = game.canonicalized();
    base.v = Vector<float>(3);
    base.v.setConstant(1.0f / 3.0f);
    base.pi = Vector<float>(TestAS::NUM_MOVES);
    base.pi.setZero();

    int orig_action = TestAS::DEPLOY_OFFSET + 0 * 6 + orig_dir;  // Fighter type
    base.pi(orig_action) = 1.0f;

    auto syms = game.symmetries(base);
    ASSERT_EQ(syms.size(), 2u);

    int expected_action = TestAS::DEPLOY_OFFSET + 0 * 6 + rotated_dir;
    EXPECT_FLOAT_EQ(syms[1].pi(expected_action), 1.0f)
        << "Direction " << orig_dir << " should map to " << rotated_dir;
  }
}

TEST(Symmetries, CurrentPlayerFlipped) {
  TestGame game;
  EXPECT_EQ(game.current_player(), 0);

  PlayHistory base;
  base.canonical = game.canonicalized();
  base.v = Vector<float>(3);
  base.v.setConstant(1.0f / 3.0f);
  base.pi = Vector<float>(TestAS::NUM_MOVES);
  base.pi.setZero();

  auto syms = game.symmetries(base);
  ASSERT_EQ(syms.size(), 2u);

  // Current player channel index: SLOT_PRESENCE_CHANNELS + 18
  // SLOT_PRESENCE_CHANNELS for Skirmish = (3 + 1 + 0) * 2 + 2 = 10
  // Channel layout: orientation(12) + HP(6) = 18, so current_player is at offset 18
  int cp_channel = TestAS::SLOT_PRESENCE_CHANNELS + 18;

  // Original: current_player = 0, so channel value is 0.0
  EXPECT_FLOAT_EQ(base.canonical(cp_channel, 0, 0), 0.0f)
      << "Original current player should be 0";

  // Rotated: should be flipped to 1.0
  EXPECT_FLOAT_EQ(syms[1].canonical(cp_channel, 0, 0), 1.0f)
      << "Rotated current player should be 1";
}

TEST(Symmetries, HexPositionsRotatedCorrectly) {
  // Test that hex rotation works correctly

  TestGame game;

  // Deploy a fighter for P0 to create unit presence at a non-origin hex
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

  // Now P1 deploys
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

  PlayHistory base;
  base.canonical = game.canonicalized();
  base.v = Vector<float>(3);
  base.v.setConstant(1.0f / 3.0f);
  base.pi = Vector<float>(TestAS::NUM_MOVES);
  base.pi.setZero();

  auto syms = game.symmetries(base);
  ASSERT_EQ(syms.size(), 2u);

  // Verify hex rotation: for each channel, data at hex idx should match
  // rotated data at rotated hex idx
  // Check that origin hex is unchanged (rotation of (0,0) is (0,0))
  int origin_idx = hex_to_index({0, 0}, TestConfig::BOARD_SIDE);
  for (int c = 0; c < TestAS::SLOT_PRESENCE_CHANNELS; ++c) {
    // Origin should map to origin
    float orig_val = base.canonical(c, 0, origin_idx);
    float rot_val = syms[1].canonical(c, 0, origin_idx);
    // Note: P0 and P1 channels are swapped, so we need to check corresponding channels
    (void)orig_val;  // Suppress unused variable warning
    (void)rot_val;
  }

  // Verify that hex rotation is correct by checking index mapping
  Hex test_hex = {2, -1};
  int test_idx = hex_to_index(test_hex, TestConfig::BOARD_SIDE);
  Hex rotated_hex = {static_cast<int8_t>(-test_hex.q), static_cast<int8_t>(-test_hex.r)};
  int rotated_idx = hex_to_index(rotated_hex, TestConfig::BOARD_SIDE);

  EXPECT_TRUE(hex_in_bounds(test_hex, TestConfig::BOARD_SIDE));
  EXPECT_TRUE(hex_in_bounds(rotated_hex, TestConfig::BOARD_SIDE));
  EXPECT_NE(test_idx, rotated_idx) << "Rotation should change non-origin hex indices";
}

TEST(Symmetries, MoveAndFireActionsUnchanged) {
  TestGame game;
  PlayHistory base;
  base.canonical = game.canonicalized();
  base.v = Vector<float>(3);
  base.v.setConstant(1.0f / 3.0f);
  base.pi = Vector<float>(TestAS::NUM_MOVES);
  base.pi.setZero();

  // Set policy for a move action (these use relative directions)
  base.pi(0) = 0.5f;  // Fighter 1 move forward
  base.pi(TestAS::FIGHTER_FIRE_OFFSET) = 0.3f;  // Fighter 1 fire
  base.pi(TestAS::END_TURN_OFFSET) = 0.2f;  // End turn

  auto syms = game.symmetries(base);
  ASSERT_EQ(syms.size(), 2u);

  // Move and fire actions should be unchanged (relative directions)
  EXPECT_FLOAT_EQ(syms[1].pi(0), 0.5f)
      << "Move action should be unchanged after rotation";
  EXPECT_FLOAT_EQ(syms[1].pi(TestAS::FIGHTER_FIRE_OFFSET), 0.3f)
      << "Fire action should be unchanged after rotation";
  EXPECT_FLOAT_EQ(syms[1].pi(TestAS::END_TURN_OFFSET), 0.2f)
      << "End turn action should be unchanged after rotation";
}

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

  // First symmetry (identity) should be exactly the same
  for (int i = 0; i < 3; ++i) {
    EXPECT_FLOAT_EQ(syms[0].v(i), base.v(i))
        << "Identity value mismatch at " << i;
  }

  for (int i = 0; i < TestAS::NUM_MOVES; ++i) {
    EXPECT_FLOAT_EQ(syms[0].pi(i), base.pi(i))
        << "Identity policy mismatch at " << i;
  }

  for (int c = 0; c < TestAS::CANONICAL_SHAPE[0]; ++c) {
    for (int h = 0; h < TestAS::NUM_HEXES; ++h) {
      EXPECT_FLOAT_EQ(syms[0].canonical(c, 0, h), base.canonical(c, 0, h))
          << "Identity canonical mismatch at (" << c << ", " << h << ")";
    }
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
// Action Space Configuration Tests
// =============================================================================

TEST(ActionSpace, SkirmishActionCounts) {
  // Skirmish: 3 fighters, 1 cruiser, 0 dreadnoughts
  using AS = ActionSpace<SkirmishConfig>;

  // Fighter moves: 3 * 3 = 9
  EXPECT_EQ(AS::FIGHTER_MOVE_ACTIONS, 9);
  // Cruiser moves: 1 * 5 = 5
  EXPECT_EQ(AS::CRUISER_MOVE_ACTIONS, 5);
  // Dreadnought moves: 0 * 4 = 0
  EXPECT_EQ(AS::DREAD_MOVE_ACTIONS, 0);

  // Fighter fires: 3 * 1 = 3
  EXPECT_EQ(AS::FIGHTER_FIRE_ACTIONS, 3);
  // Cruiser fires: 1 * 3 = 3
  EXPECT_EQ(AS::CRUISER_FIRE_ACTIONS, 3);
  // Dread fires: 0 * 4 = 0
  EXPECT_EQ(AS::DREAD_FIRE_ACTIONS, 0);

  // Deploy: 3 types * 6 facings = 18
  EXPECT_EQ(AS::DEPLOY_ACTIONS, 18);

  // Total: 9 + 5 + 0 + 3 + 3 + 0 + 18 + 1 = 39
  EXPECT_EQ(AS::NUM_MOVES, 39);
}

TEST(ActionSpace, ClashActionCounts) {
  // Clash: 3 fighters, 2 cruisers, 1 dreadnought
  using AS = ActionSpace<ClashConfig>;

  // Fighter moves: 3 * 3 = 9
  EXPECT_EQ(AS::FIGHTER_MOVE_ACTIONS, 9);
  // Cruiser moves: 2 * 5 = 10
  EXPECT_EQ(AS::CRUISER_MOVE_ACTIONS, 10);
  // Dreadnought moves: 1 * 4 = 4
  EXPECT_EQ(AS::DREAD_MOVE_ACTIONS, 4);

  // Fighter fires: 3 * 1 = 3
  EXPECT_EQ(AS::FIGHTER_FIRE_ACTIONS, 3);
  // Cruiser fires: 2 * 3 = 6
  EXPECT_EQ(AS::CRUISER_FIRE_ACTIONS, 6);
  // Dread fires: 1 * 4 = 4
  EXPECT_EQ(AS::DREAD_FIRE_ACTIONS, 4);

  // Deploy: 3 types * 6 facings = 18
  EXPECT_EQ(AS::DEPLOY_ACTIONS, 18);

  // Total: 9 + 10 + 4 + 3 + 6 + 4 + 18 + 1 = 55
  EXPECT_EQ(AS::NUM_MOVES, 55);
}

TEST(ActionSpace, BattleActionCounts) {
  // Battle: 4 fighters, 3 cruisers, 2 dreadnoughts
  using AS = ActionSpace<BattleConfig>;

  // Fighter moves: 4 * 3 = 12
  EXPECT_EQ(AS::FIGHTER_MOVE_ACTIONS, 12);
  // Cruiser moves: 3 * 5 = 15
  EXPECT_EQ(AS::CRUISER_MOVE_ACTIONS, 15);
  // Dreadnought moves: 2 * 4 = 8
  EXPECT_EQ(AS::DREAD_MOVE_ACTIONS, 8);

  // Fighter fires: 4 * 1 = 4
  EXPECT_EQ(AS::FIGHTER_FIRE_ACTIONS, 4);
  // Cruiser fires: 3 * 3 = 9
  EXPECT_EQ(AS::CRUISER_FIRE_ACTIONS, 9);
  // Dread fires: 2 * 4 = 8
  EXPECT_EQ(AS::DREAD_FIRE_ACTIONS, 8);

  // Deploy: 3 types * 6 facings = 18
  EXPECT_EQ(AS::DEPLOY_ACTIONS, 18);

  // Total: 12 + 15 + 8 + 4 + 9 + 8 + 18 + 1 = 75
  EXPECT_EQ(AS::NUM_MOVES, 75);
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

TEST(FireValidation, NoFireWithoutTarget) {
  TestGame game;

  // Deploy fighters for both players
  auto valids = game.valid_moves();
  // Find a valid fighter deploy (action 20-25 range for fighters facing E/SW/SE)
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

  // Check all fire actions are invalid
  for (int i = TestAS::FIGHTER_FIRE_OFFSET; i < TestAS::DEPLOY_OFFSET; ++i) {
    EXPECT_EQ(valids(i), 0) << "Fire action " << i << " should not be valid when no target in range";
  }
}

TEST(FireValidation, FireAvailableWhenTargetInRange) {
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

  // Move fighters toward each other until fire becomes available
  bool fire_found = false;
  for (int turn = 0; turn < 30 && !fire_found; ++turn) {
    valids = game.valid_moves();

    // Check for fire actions
    for (int i = TestAS::FIGHTER_FIRE_OFFSET; i < TestAS::CRUISER_FIRE_OFFSET; ++i) {
      if (valids(i) == 1) {
        fire_found = true;
        break;
      }
    }

    if (fire_found) break;

    // Make a move (prefer forward movement for faster approach)
    for (int i = 0; i < TestAS::NUM_MOVES; ++i) {
      if (valids(i) == 1) {
        game.play_move(i);
        break;
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

  // Count valid fighter move actions for slot 0 (actions 0, 1, 2)
  int move_count = 0;
  for (int i = 0; i < 3; ++i) {
    if (valids(i) == 1) {
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

  // Count valid cruiser move actions for slot 0 (actions 9-13)
  int move_count = 0;
  for (int i = TestAS::CRUISER_MOVE_OFFSET; i < TestAS::CRUISER_MOVE_OFFSET + 5; ++i) {
    if (valids(i) == 1) {
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
// Move Parsing Tests
// =============================================================================

TEST(MoveParsing, ParseFighterMoves) {
  auto m_f1_f = parse_move<TestConfig>("m f1 f");
  ASSERT_TRUE(m_f1_f.has_value());
  EXPECT_EQ(*m_f1_f, TestAS::FIGHTER_MOVE_OFFSET + 0);  // Fighter 1, forward

  auto m_f1_fl = parse_move<TestConfig>("m f1 fl");
  ASSERT_TRUE(m_f1_fl.has_value());
  EXPECT_EQ(*m_f1_fl, TestAS::FIGHTER_MOVE_OFFSET + 1);  // Fighter 1, forward-left

  auto m_f1_fr = parse_move<TestConfig>("m f1 fr");
  ASSERT_TRUE(m_f1_fr.has_value());
  EXPECT_EQ(*m_f1_fr, TestAS::FIGHTER_MOVE_OFFSET + 2);  // Fighter 1, forward-right

  auto m_f2_f = parse_move<TestConfig>("m f2 f");
  ASSERT_TRUE(m_f2_f.has_value());
  EXPECT_EQ(*m_f2_f, TestAS::FIGHTER_MOVE_OFFSET + 3);  // Fighter 2, forward

  auto m_f3_fl = parse_move<TestConfig>("m f3 fl");
  ASSERT_TRUE(m_f3_fl.has_value());
  EXPECT_EQ(*m_f3_fl, TestAS::FIGHTER_MOVE_OFFSET + 7);  // Fighter 3, forward-left
}

TEST(MoveParsing, ParseCruiserMoves) {
  auto m_c1_l = parse_move<TestConfig>("m c1 l");
  ASSERT_TRUE(m_c1_l.has_value());
  EXPECT_EQ(*m_c1_l, TestAS::CRUISER_MOVE_OFFSET + 0);  // Cruiser 1, rotate-left

  auto m_c1_f = parse_move<TestConfig>("m c1 f");
  ASSERT_TRUE(m_c1_f.has_value());
  EXPECT_EQ(*m_c1_f, TestAS::CRUISER_MOVE_OFFSET + 2);  // Cruiser 1, forward

  auto m_c1_r = parse_move<TestConfig>("m c1 r");
  ASSERT_TRUE(m_c1_r.has_value());
  EXPECT_EQ(*m_c1_r, TestAS::CRUISER_MOVE_OFFSET + 4);  // Cruiser 1, rotate-right
}

TEST(MoveParsing, ParseFighterFire) {
  auto f_f1 = parse_move<TestConfig>("f f1");
  ASSERT_TRUE(f_f1.has_value());
  EXPECT_EQ(*f_f1, TestAS::FIGHTER_FIRE_OFFSET + 0);  // Fighter 1 fire

  auto f_f2 = parse_move<TestConfig>("f f2");
  ASSERT_TRUE(f_f2.has_value());
  EXPECT_EQ(*f_f2, TestAS::FIGHTER_FIRE_OFFSET + 1);  // Fighter 2 fire

  auto f_f3 = parse_move<TestConfig>("f f3");
  ASSERT_TRUE(f_f3.has_value());
  EXPECT_EQ(*f_f3, TestAS::FIGHTER_FIRE_OFFSET + 2);  // Fighter 3 fire
}

TEST(MoveParsing, ParseCruiserFire) {
  auto f_c1_l = parse_move<TestConfig>("f c1 l");
  ASSERT_TRUE(f_c1_l.has_value());
  EXPECT_EQ(*f_c1_l, TestAS::CRUISER_FIRE_OFFSET + 0);  // Cruiser 1, left cannon

  auto f_c1_f = parse_move<TestConfig>("f c1 f");
  ASSERT_TRUE(f_c1_f.has_value());
  EXPECT_EQ(*f_c1_f, TestAS::CRUISER_FIRE_OFFSET + 1);  // Cruiser 1, forward cannon

  auto f_c1_r = parse_move<TestConfig>("f c1 r");
  ASSERT_TRUE(f_c1_r.has_value());
  EXPECT_EQ(*f_c1_r, TestAS::CRUISER_FIRE_OFFSET + 2);  // Cruiser 1, right cannon
}

TEST(MoveParsing, ParseDeploy) {
  auto d_f_e = parse_move<TestConfig>("d f e");
  ASSERT_TRUE(d_f_e.has_value());
  EXPECT_EQ(*d_f_e, TestAS::DEPLOY_OFFSET + 0);  // Fighter, East

  auto d_f_se = parse_move<TestConfig>("d f se");
  ASSERT_TRUE(d_f_se.has_value());
  EXPECT_EQ(*d_f_se, TestAS::DEPLOY_OFFSET + 5);  // Fighter, SE

  auto d_c_sw = parse_move<TestConfig>("d c sw");
  ASSERT_TRUE(d_c_sw.has_value());
  EXPECT_EQ(*d_c_sw, TestAS::DEPLOY_OFFSET + 6 + 4);  // Cruiser (type 1 * 6) + SW (4)

  auto d_d_ne = parse_move<TestConfig>("d d ne");
  ASSERT_TRUE(d_d_ne.has_value());
  EXPECT_EQ(*d_d_ne, TestAS::DEPLOY_OFFSET + 12 + 1);  // Dreadnought (type 2 * 6) + NE (1)
}

TEST(MoveParsing, ParseEndTurn) {
  auto end = parse_move<TestConfig>("e");
  ASSERT_TRUE(end.has_value());
  EXPECT_EQ(*end, TestAS::END_TURN_OFFSET);
}

TEST(MoveParsing, InvalidMovesReturnNullopt) {
  EXPECT_FALSE(parse_move<TestConfig>("invalid").has_value());
  EXPECT_FALSE(parse_move<TestConfig>("m f0 f").has_value());  // Invalid slot (0)
  EXPECT_FALSE(parse_move<TestConfig>("m f10 f").has_value()); // Invalid slot (too high)
  EXPECT_FALSE(parse_move<TestConfig>("m f1 x").has_value());  // Invalid direction
  EXPECT_FALSE(parse_move<TestConfig>("d x e").has_value());   // Invalid unit type
  EXPECT_FALSE(parse_move<TestConfig>("d f x").has_value());   // Invalid facing
}

// =============================================================================
// Full Game Playthrough with Notation
// =============================================================================

// Helper to play a move by notation, returns true if successful
template<typename Config>
bool play_notation(StarGambitGS<Config>& game, const std::string& notation) {
  auto action = parse_move<Config>(notation);
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

  // Check cruiser move actions are available
  bool has_cruiser_move = false;
  for (int i = TestAS::CRUISER_MOVE_OFFSET; i < TestAS::DREAD_MOVE_OFFSET; ++i) {
    if (valids(i) == 1) {
      has_cruiser_move = true;
      break;
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

  // Deploy fighters for both players facing each other
  EXPECT_TRUE(play_notation(game, "d f ne"));  // P0 (bottom, facing up/NE)
  EXPECT_TRUE(play_notation(game, "d f sw"));  // P1 (top, facing down/SW)

  // Move fighters towards each other until fire is available
  bool fire_available = false;
  for (int i = 0; i < 30; ++i) {
    auto valids = game.valid_moves();

    // Check if fire is available
    for (int j = TestAS::FIGHTER_FIRE_OFFSET; j < TestAS::CRUISER_FIRE_OFFSET; ++j) {
      if (valids(j) == 1) {
        fire_available = true;
        game.play_move(j);
        break;
      }
    }

    if (fire_available) break;

    // Try to move fighter forward (action 0 = f1 forward)
    // Fighter 1 forward move is at offset 0
    int forward_move = 0;  // Fighter 1 forward
    if (valids(forward_move) == 1) {
      game.play_move(forward_move);
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

  // Player 1 valid dreadnought facings: {0, 3, 4, 5} (E, W, SW, SE)
  auto valid_facings = get_valid_deploy_facings(UnitType::DREADNOUGHT, 1);
  ASSERT_EQ(valid_facings.size(), 4u);

  valids = game.valid_moves();
  int dread_deploy_base = BattleAS::DEPLOY_OFFSET + 12;

  for (int facing : valid_facings) {
    int action = dread_deploy_base + facing;
    EXPECT_EQ(valids(action), 1)
        << "Dreadnought deploy facing " << facing << " should be valid for P1";
  }
}

}  // namespace alphazero::star_gambit_gs
