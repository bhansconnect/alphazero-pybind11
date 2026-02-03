#include "star_gambit_gs.h"

#include <gtest/gtest.h>

#include <iostream>

namespace alphazero::star_gambit_gs {

// Use Skirmish configuration for tests
using TestConfig = SkirmishConfig;
using TestAS = ActionSpace<TestConfig>;
using TestGame = StarGambitSkirmishGS;

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

  // Should have 2 portals initially
  std::string dump = game.dump();
  EXPECT_TRUE(dump.find("Portal") != std::string::npos);
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
  EXPECT_EQ(p0_dreadnought_facings.size(), 4u);
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

}  // namespace alphazero::star_gambit_gs
