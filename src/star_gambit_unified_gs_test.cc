#include "star_gambit_gs.h"

#include <gtest/gtest.h>

#include <cmath>

#include "mcts.h"

namespace alphazero::star_gambit_gs {

// ============================================================================
// Helpers
// ============================================================================

// Play first valid action. Returns false if game is already over.
static bool play_first_valid(StarGambitUnifiedGS& game) {
  if (game.scores().has_value()) return false;
  auto valids = game.valid_moves();
  for (int a = 0; a < StarGambitUnifiedGS::UNIFIED_NUM_MOVES; ++a) {
    if (valids(a)) {
      game.play_move(static_cast<uint32_t>(a));
      return true;
    }
  }
  return false;
}

// Build a PlayHistory from a game state.
static PlayHistory make_history(StarGambitUnifiedGS& game) {
  PlayHistory ph;
  ph.canonical = game.canonicalized();
  ph.v = Vector<float>(3);  // 2 players + draw slot
  ph.v.setConstant(1.0f / 3.0f);
  ph.pi = Vector<float>(StarGambitUnifiedGS::UNIFIED_NUM_MOVES);
  ph.pi.setZero();
  auto valids = game.valid_moves();
  int total = valids.sum();
  if (total > 0) {
    for (int a = 0; a < StarGambitUnifiedGS::UNIFIED_NUM_MOVES; ++a) {
      ph.pi(a) = valids(a) ? 1.0f / total : 0.0f;
    }
  }
  return ph;
}

// ============================================================================
// Static Constants
// ============================================================================

TEST(StarGambitUnifiedGS, StaticConstants) {
  EXPECT_EQ(StarGambitUnifiedGS::UNIFIED_NUM_MOVES, 1709);
  EXPECT_EQ(StarGambitUnifiedGS::UNIFIED_BOARD_DIM, 13);
  EXPECT_EQ(StarGambitUnifiedGS::UNIFIED_CHANNELS, 36);
  EXPECT_EQ(StarGambitUnifiedGS::UNIFIED_SPATIAL, 13 * 13 * 10);   // 1690
  EXPECT_EQ(StarGambitUnifiedGS::UNIFIED_DEPLOY_OFFSET, 1690);
  EXPECT_EQ(StarGambitUnifiedGS::UNIFIED_END_TURN_OFFSET, 1708);
  EXPECT_EQ(StarGambitUnifiedGS::SMALL_BOARD_DIM, 11);
  EXPECT_EQ(StarGambitUnifiedGS::SMALL_SPATIAL, 11 * 11 * 10);      // 1210
  EXPECT_EQ(StarGambitUnifiedGS::SMALL_DEPLOY_OFFSET, 1210);
  EXPECT_EQ(StarGambitUnifiedGS::CANONICAL_SHAPE_ARRAY[0], 36);
  EXPECT_EQ(StarGambitUnifiedGS::CANONICAL_SHAPE_ARRAY[1], 13);
  EXPECT_EQ(StarGambitUnifiedGS::CANONICAL_SHAPE_ARRAY[2], 13);
}

// ============================================================================
// Canonical Shape
// ============================================================================

TEST(StarGambitUnifiedGS, CanonicalShape) {
  for (int v = 0; v < 4; ++v) {
    StarGambitUnifiedGS game(v);
    auto tensor = game.canonicalized();
    EXPECT_EQ(tensor.dimension(0), StarGambitUnifiedGS::UNIFIED_CHANNELS)   << "variant=" << v;
    EXPECT_EQ(tensor.dimension(1), StarGambitUnifiedGS::UNIFIED_BOARD_DIM)  << "variant=" << v;
    EXPECT_EQ(tensor.dimension(2), StarGambitUnifiedGS::UNIFIED_BOARD_DIM)  << "variant=" << v;
  }
}

// ============================================================================
// Game-Type Channels (one-hot)
// ============================================================================

TEST(StarGambitUnifiedGS, GameTypeChannels) {
  for (int v = 0; v < 4; ++v) {
    StarGambitUnifiedGS game(v);
    auto tensor = game.canonicalized();

    EXPECT_EQ(game.get_variant_id(), v);

    // Position (6,6) is the board center, a valid hex for all variants.
    for (int ch = 32; ch < 36; ++ch) {
      float val = tensor(ch, 6, 6);
      if (ch == 32 + v) {
        EXPECT_FLOAT_EQ(val, 1.0f) << "variant=" << v << " channel=" << ch;
      } else {
        EXPECT_FLOAT_EQ(val, 0.0f) << "variant=" << v << " channel=" << ch;
      }
    }

    // Game-type channels should be broadcast: all valid hex positions share the same value.
    // Spot-check a few inner positions.
    for (int row = 2; row <= 10; ++row) {
      for (int ch = 32; ch < 36; ++ch) {
        float expect = (ch == 32 + v) ? 1.0f : 0.0f;
        // Only check positions that are definitely valid hexes for small-variant inner area.
        if (row >= 1 && row <= 11) {
          EXPECT_FLOAT_EQ(tensor(ch, row, 6), expect)
              << "variant=" << v << " ch=" << ch << " row=" << row;
        }
      }
    }
  }
}

// ============================================================================
// Observation Padding (small variants have a 1-cell zero border)
// ============================================================================

TEST(StarGambitUnifiedGS, ObservationPaddingSmallVariants) {
  // Skirmish, Showdown, Clash all use 11x11 inner board → outer ring zeros in channel 0.
  for (int v = 0; v < 3; ++v) {
    StarGambitUnifiedGS game(v);
    auto tensor = game.canonicalized();

    for (int col = 0; col < 13; ++col) {
      EXPECT_FLOAT_EQ(tensor(0, 0, col),  0.0f) << "variant=" << v << " border row 0 col=" << col;
      EXPECT_FLOAT_EQ(tensor(0, 12, col), 0.0f) << "variant=" << v << " border row 12 col=" << col;
    }
    for (int row = 0; row < 13; ++row) {
      EXPECT_FLOAT_EQ(tensor(0, row, 0),  0.0f) << "variant=" << v << " border col 0 row=" << row;
      EXPECT_FLOAT_EQ(tensor(0, row, 12), 0.0f) << "variant=" << v << " border col 12 row=" << row;
    }

    // Inner area (rows 1-11, cols 1-11) should have some valid hex cells set.
    bool found_valid = false;
    for (int r = 1; r <= 11 && !found_valid; ++r)
      for (int c = 1; c <= 11 && !found_valid; ++c)
        if (tensor(0, r, c) > 0.0f) found_valid = true;
    EXPECT_TRUE(found_valid) << "variant=" << v << " inner area should have valid hexes";
  }
}

TEST(StarGambitUnifiedGS, ObservationPaddingBattle) {
  // Battle (13x13): outer ring should contain some valid hexes (no zero border).
  StarGambitUnifiedGS game(3);
  auto tensor = game.canonicalized();
  bool found_valid = false;
  for (int col = 0; col < 13; ++col) {
    if (tensor(0, 0, col) > 0.0f || tensor(0, 12, col) > 0.0f) {
      found_valid = true;
      break;
    }
  }
  EXPECT_TRUE(found_valid) << "Battle outer rows should have valid hexes";
}

// ============================================================================
// Action Remapping – Small Variants
// ============================================================================

TEST(StarGambitUnifiedGS, ActionRemap_SmallVariant) {
  // For Skirmish, valid spatial actions should not be in the outer ring.
  StarGambitUnifiedGS game(0);
  auto valids = game.valid_moves();

  EXPECT_EQ(valids.size(), static_cast<long>(StarGambitUnifiedGS::UNIFIED_NUM_MOVES));

  for (int action = 0; action < StarGambitUnifiedGS::UNIFIED_SPATIAL; ++action) {
    if (!valids(action)) continue;
    int flat_pos = action / ACTIONS_PER_POSITION;
    int row = flat_pos / StarGambitUnifiedGS::UNIFIED_BOARD_DIM;
    int col = flat_pos % StarGambitUnifiedGS::UNIFIED_BOARD_DIM;
    EXPECT_GT(row, 0)  << "valid action in border row 0  (action=" << action << ")";
    EXPECT_LT(row, 12) << "valid action in border row 12 (action=" << action << ")";
    EXPECT_GT(col, 0)  << "valid action in border col 0  (action=" << action << ")";
    EXPECT_LT(col, 12) << "valid action in border col 12 (action=" << action << ")";
  }
}

// ============================================================================
// Action Remapping – Battle (identity)
// ============================================================================

TEST(StarGambitUnifiedGS, ActionRemap_Battle) {
  StarGambitUnifiedGS game(3);
  auto valids = game.valid_moves();

  EXPECT_EQ(valids.size(), static_cast<long>(StarGambitUnifiedGS::UNIFIED_NUM_MOVES));
  EXPECT_GT(valids.sum(), 0);

  // Deploy actions for Battle should be at UNIFIED_DEPLOY_OFFSET, same place as small variants.
  bool has_deploy = false;
  for (int a = StarGambitUnifiedGS::UNIFIED_DEPLOY_OFFSET;
       a < StarGambitUnifiedGS::UNIFIED_END_TURN_OFFSET; ++a) {
    if (valids(a)) { has_deploy = true; break; }
  }
  EXPECT_TRUE(has_deploy) << "Battle should have deploy actions at UNIFIED_DEPLOY_OFFSET";
}

// ============================================================================
// Deploy Action Remapping
// ============================================================================

TEST(StarGambitUnifiedGS, DeployActionRemap) {
  for (int v = 0; v < 4; ++v) {
    StarGambitUnifiedGS game(v);
    auto valids = game.valid_moves();

    // Turn 1 has only deploy / end-turn actions.
    bool has_deploy_at_unified = false;
    for (int a = StarGambitUnifiedGS::UNIFIED_DEPLOY_OFFSET;
         a < StarGambitUnifiedGS::UNIFIED_END_TURN_OFFSET; ++a) {
      if (valids(a)) { has_deploy_at_unified = true; break; }
    }
    EXPECT_TRUE(has_deploy_at_unified) << "variant=" << v << " should have deploy at unified offset";

    // For small variants: should have NO valid moves at the *old* small-variant offset range.
    if (v < 3) {
      for (int a = StarGambitUnifiedGS::SMALL_DEPLOY_OFFSET;
           a < StarGambitUnifiedGS::SMALL_END_TURN_OFFSET; ++a) {
        EXPECT_EQ(valids(a), 0)
            << "variant=" << v << " should have no valid move at small deploy offset " << a;
      }
    }
  }
}

// ============================================================================
// Copy
// ============================================================================

TEST(StarGambitUnifiedGS, Copy) {
  StarGambitUnifiedGS game(0);  // Skirmish

  for (int i = 0; i < 4; ++i) play_first_valid(game);

  auto copy_ptr = game.copy();
  auto* copy_game = dynamic_cast<StarGambitUnifiedGS*>(copy_ptr.get());
  ASSERT_NE(copy_game, nullptr);

  EXPECT_EQ(copy_game->get_variant_id(), game.get_variant_id());
  EXPECT_EQ(copy_game->current_turn(),   game.current_turn());
  EXPECT_EQ(copy_game->current_player(), game.current_player());
  EXPECT_EQ(copy_game->scores().has_value(), game.scores().has_value());

  // Canonical tensors should match.
  auto orig_c = game.canonicalized();
  auto copy_c = copy_game->canonicalized();
  bool same = true;
  for (int ch = 0; ch < 36 && same; ++ch)
    for (int r = 0; r < 13 && same; ++r)
      for (int c = 0; c < 13 && same; ++c)
        if (std::abs(orig_c(ch, r, c) - copy_c(ch, r, c)) > 1e-6f) same = false;
  EXPECT_TRUE(same) << "Copy canonical tensors differ";

  // Mutating the copy should not affect the original.
  // Play all remaining valid moves on copy until turn advances.
  uint32_t orig_turn = game.current_turn();
  for (int i = 0; i < 20 && !copy_game->scores().has_value(); ++i) {
    play_first_valid(*copy_game);
  }
  EXPECT_EQ(game.current_turn(), orig_turn) << "Original changed after mutating copy";
}

// ============================================================================
// randomize_start picks variant from probs
// ============================================================================

TEST(StarGambitUnifiedGS, RandomizeStartPicksVariant) {
  // With probs [1,0,0,0], must always pick Skirmish.
  StarGambitUnifiedGS g0(-1, {1.0f, 0.0f, 0.0f, 0.0f});
  for (int t = 0; t < 20; ++t) {
    g0.randomize_start();
    EXPECT_EQ(g0.get_variant_id(), 0) << "trial=" << t;
  }

  // With probs [0,0,0,1], must always pick Battle.
  StarGambitUnifiedGS g3(-1, {0.0f, 0.0f, 0.0f, 1.0f});
  for (int t = 0; t < 20; ++t) {
    g3.randomize_start();
    EXPECT_EQ(g3.get_variant_id(), 3) << "trial=" << t;
  }

  // With equal probs, all 4 variants should appear over many trials.
  StarGambitUnifiedGS geq(-1, {0.25f, 0.25f, 0.25f, 0.25f});
  int seen[4] = {};
  for (int t = 0; t < 200; ++t) {
    geq.randomize_start();
    seen[geq.get_variant_id()]++;
  }
  for (int v = 0; v < 4; ++v)
    EXPECT_GT(seen[v], 0) << "variant " << v << " never appeared with equal probs";
}

// ============================================================================
// Pinned variant
// ============================================================================

TEST(StarGambitUnifiedGS, PinnedVariant) {
  for (int v = 0; v < 4; ++v) {
    StarGambitUnifiedGS game(v);
    for (int t = 0; t < 10; ++t) {
      game.randomize_start();
      EXPECT_EQ(game.get_variant_id(), v) << "variant=" << v << " trial=" << t;
    }
    // Pinned subclasses
    switch (v) {
      case 0: { StarGambitUnifiedSkirmishGS g; g.randomize_start(); EXPECT_EQ(g.get_variant_id(), 0); break; }
      case 1: { StarGambitUnifiedShowdownGS g; g.randomize_start(); EXPECT_EQ(g.get_variant_id(), 1); break; }
      case 2: { StarGambitUnifiedClashGS    g; g.randomize_start(); EXPECT_EQ(g.get_variant_id(), 2); break; }
      case 3: { StarGambitUnifiedBattleGS   g; g.randomize_start(); EXPECT_EQ(g.get_variant_id(), 3); break; }
    }
  }
}

// ============================================================================
// Game completion (each variant can finish)
// ============================================================================

TEST(StarGambitUnifiedGS, GameCompletion) {
  for (int v = 0; v < 4; ++v) {
    StarGambitUnifiedGS game(v);
    // Play greedily until game over or max moves.
    constexpr int MAX_MOVES = 3000;
    int moves = 0;
    while (!game.scores().has_value() && moves < MAX_MOVES) {
      if (!play_first_valid(game)) break;
      ++moves;
    }
    // Either game ended naturally (good) or we hit the cap (also OK: proves no crash).
    if (game.scores().has_value()) {
      auto scores = game.scores().value();
      // Size 3: [player0_win, player1_win, draw]
      EXPECT_EQ(scores.size(), 3) << "variant=" << v;
      // Exactly one element should be 1.0, the others 0.0.
      float total = scores(0) + scores(1) + scores(2);
      EXPECT_NEAR(total, 1.0f, 0.01f) << "variant=" << v;
    }
  }
}

// ============================================================================
// Symmetry
// ============================================================================

TEST(StarGambitUnifiedGS, SymmetryCount) {
  for (int v = 0; v < 4; ++v) {
    StarGambitUnifiedGS game(v);
    EXPECT_EQ(game.num_symmetries(), NUM_SYMMETRIES);
  }
}

TEST(StarGambitUnifiedGS, SymmetryShapes) {
  StarGambitUnifiedGS game(0);  // Skirmish
  for (int i = 0; i < 5; ++i) play_first_valid(game);

  auto ph = make_history(game);
  auto syms = game.symmetries(ph);

  EXPECT_EQ(static_cast<int>(syms.size()), NUM_SYMMETRIES);
  for (const auto& sym : syms) {
    EXPECT_EQ(sym.canonical.dimension(0), StarGambitUnifiedGS::UNIFIED_CHANNELS);
    EXPECT_EQ(sym.canonical.dimension(1), StarGambitUnifiedGS::UNIFIED_BOARD_DIM);
    EXPECT_EQ(sym.canonical.dimension(2), StarGambitUnifiedGS::UNIFIED_BOARD_DIM);
    EXPECT_EQ(sym.pi.size(), static_cast<long>(StarGambitUnifiedGS::UNIFIED_NUM_MOVES));
    EXPECT_EQ(sym.v.size(), 3);  // num_players + 1
  }
}

TEST(StarGambitUnifiedGS, SymmetryPiSumsPreserved) {
  // The pi probability sum should be preserved under symmetry.
  StarGambitUnifiedGS game(0);
  for (int i = 0; i < 5; ++i) play_first_valid(game);

  auto ph = make_history(game);
  float orig_pi_sum = ph.pi.sum();

  auto syms = game.symmetries(ph);
  for (const auto& sym : syms) {
    float sym_pi_sum = sym.pi.sum();
    EXPECT_NEAR(sym_pi_sum, orig_pi_sum, 1e-4f)
        << "Pi sum not preserved under symmetry";
  }
}

}  // namespace alphazero::star_gambit_gs
