
#include "opentafl_gs.h"

#include "gtest/gtest.h"

namespace alphazero::opentafl_gs {
namespace {

// NOLINTNEXTLINE
TEST(OpenTaflGS, RepetitionCount) {
  auto gs = OpenTaflGS().copy();
  // gs->play_move((0 * WIDTH + 3) * (WIDTH + HEIGHT) + WIDTH + 2);
  // gs->play_move((0 * WIDTH + 4) * (WIDTH + HEIGHT) + WIDTH + 2);
  // gs->play_move((0 * WIDTH + 6) * (WIDTH + HEIGHT) + WIDTH + 2);
  // gs->play_move((0 * WIDTH + 7) * (WIDTH + HEIGHT) + WIDTH + 2);
  // std::cout << gs->dump();
  // gs->play_move((3 * WIDTH + 0) * (WIDTH + HEIGHT) + 2 + 0);
  // gs->play_move((4 * WIDTH + 0) * (WIDTH + HEIGHT) + 2 + 0);
  // gs->play_move((6 * WIDTH + 0) * (WIDTH + HEIGHT) + 2 + 0);
  // gs->play_move((7 * WIDTH + 0) * (WIDTH + HEIGHT) + 2 + 0);
  // std::cout << gs->dump();
  // gs->play_move((3 * WIDTH + 10) * (WIDTH + HEIGHT) + 8 + 0);
  // gs->play_move((4 * WIDTH + 10) * (WIDTH + HEIGHT) + 8 + 0);
  // gs->play_move((6 * WIDTH + 10) * (WIDTH + HEIGHT) + 8 + 0);
  // gs->play_move((7 * WIDTH + 10) * (WIDTH + HEIGHT) + 8 + 0);
  // std::cout << gs->dump();
  // gs->play_move((10 * WIDTH + 3) * (WIDTH + HEIGHT) + WIDTH + 8);
  // gs->play_move((10 * WIDTH + 4) * (WIDTH + HEIGHT) + WIDTH + 8);
  // gs->play_move((10 * WIDTH + 6) * (WIDTH + HEIGHT) + WIDTH + 8);
  // gs->play_move((10 * WIDTH + 7) * (WIDTH + HEIGHT) + WIDTH + 8);
  // std::cout << gs->dump();
  gs->play_move((9 * WIDTH + 5) * (WIDTH + HEIGHT) + WIDTH + 8);
  gs->play_move((7 * WIDTH + 5) * (WIDTH + HEIGHT) + 4 + 0);
  std::cout << gs->dump();
  gs->play_move((8 * WIDTH + 5) * (WIDTH + HEIGHT) + WIDTH + 9);
  gs->play_move((7 * WIDTH + 4) * (WIDTH + HEIGHT) + 5 + 0);
  std::cout << gs->dump();
  gs->play_move((9 * WIDTH + 5) * (WIDTH + HEIGHT) + WIDTH + 8);
  gs->play_move((7 * WIDTH + 5) * (WIDTH + HEIGHT) + 4 + 0);
  std::cout << gs->dump();
  gs = gs->copy();
  gs->play_move((8 * WIDTH + 5) * (WIDTH + HEIGHT) + WIDTH + 9);
  gs->play_move((7 * WIDTH + 4) * (WIDTH + HEIGHT) + 5 + 0);
  std::cout << gs->dump();
  // gs->play_move((8 * WIDTH + 5) * (WIDTH + HEIGHT) + WIDTH + 9);
  // gs->play_move((7 * WIDTH + 4) * (WIDTH + HEIGHT) + 5 + 0);
  // gs->play_move((9 * WIDTH + 5) * (WIDTH + HEIGHT) + WIDTH + 8);
  // gs->play_move((7 * WIDTH + 5) * (WIDTH + HEIGHT) + 4 + 0);
  // std::cout << gs->dump();
  // auto m = gs->valid_moves();
  // for (auto i = 0; i < m.size(); ++i) {
  //   if (m(i) == 1) {
  //     auto new_loc = i % (WIDTH + HEIGHT);
  //     auto height_move = new_loc >= WIDTH;
  //     if (height_move) {
  //       new_loc -= WIDTH;
  //     }
  //     auto piece_loc = (i / (WIDTH + HEIGHT));
  //     auto piece_w = piece_loc % WIDTH;
  //     auto piece_h = piece_loc / WIDTH;

  //     auto new_h = piece_h;
  //     auto new_w = piece_w;
  //     if (height_move) {
  //       new_h = new_loc;
  //     } else {
  //       new_w = new_loc;
  //     }
  //     std::cout << piece_h << ',' << piece_w << " -> " << new_h << ',' <<
  //     new_w
  //               << '\n';
  //   }
  // }
  auto s = gs->scores().value_or(SizedVector<float, 3>{0, 0, 0});

  std::cout << s(0) << ", " << s(1) << ", " << s(2);
  auto expected = SizedVector<float, 3>{1, 0, 0};
  EXPECT_EQ(s, expected);
}

// ---------------------------------------------------------------------------
// Rule-coverage tests. Each exercises one OpenTafl/Fetlar rule through the
// public API only: build a position, play one move, read the board back via
// the canonical encoding (layers 0/1/2 = king/defender/attacker).
//
// Repetition (RepetitionCount above) was verified against the OpenTafl engine
// source (jslater89/OpenTafl): the tournament uses tfr:w (THIRD_REPETITION_
// WINS) and the engine awards the win to getCurrentSide() == the side to move
// after the repeat, which is what scores(player_) implements.
// ---------------------------------------------------------------------------

uint32_t Mv(int fh, int fw, bool height_move, int new_loc) {
  return (fh * WIDTH + fw) * (WIDTH + HEIGHT) +
         (height_move ? WIDTH + new_loc : new_loc);
}

OpenTaflGS MakeGS(const BoardTensor& b, int8_t player, uint16_t turn = 10) {
  auto intern = std::make_shared<absl::flat_hash_set<RepetitionKeyWrapper>>();
  absl::flat_hash_map<const std::shared_ptr<RepetitionKey>, uint8_t> counts;
  return OpenTaflGS(b, player, turn, DEFAULT_MAX_TURNS, 1, counts, intern);
}

// Rule 3: custodial capture between two enemy pieces.
TEST(OpenTaflGS, CaptureBetweenTwoEnemies) {
  BoardTensor b;
  b.setZero();
  b(ATK_LAYER, 3, 2) = 1;   // anvil
  b(DEF_LAYER, 3, 3) = 1;   // victim
  b(ATK_LAYER, 3, 7) = 1;   // hammer (moves)
  b(KING_LAYER, 8, 8) = 1;  // king elsewhere so the game stays in progress
  auto gs = MakeGS(b, ATK_PLAYER);
  gs.play_move(Mv(3, 7, false, 4));  // attacker slides to (3,4)
  auto c = gs.canonicalized();
  EXPECT_EQ(c(DEF_LAYER, 3, 3), 0.0f) << "defender should be captured";
  EXPECT_EQ(c(ATK_LAYER, 3, 4), 1.0f) << "attacker should have moved in";
}

// Rules 3 + 7: capture against a (always-hostile) corner square.
TEST(OpenTaflGS, CaptureAgainstCorner) {
  BoardTensor b;
  b.setZero();
  b(DEF_LAYER, 0, 1) = 1;   // victim next to corner (0,0)
  b(ATK_LAYER, 5, 2) = 1;   // hammer
  b(KING_LAYER, 8, 8) = 1;
  auto gs = MakeGS(b, ATK_PLAYER);
  gs.play_move(Mv(5, 2, true, 0));  // attacker slides up col 2 to (0,2)
  auto c = gs.canonicalized();
  EXPECT_EQ(c(DEF_LAYER, 0, 1), 0.0f) << "defender captured against corner";
}

// Rule 6: the empty throne is hostile to defenders.
TEST(OpenTaflGS, CaptureAgainstEmptyThrone) {
  BoardTensor b;
  b.setZero();
  b(DEF_LAYER, 5, 4) = 1;   // victim next to throne (5,5)
  b(ATK_LAYER, 5, 1) = 1;   // hammer
  b(KING_LAYER, 8, 8) = 1;  // king NOT on throne -> throne empty
  auto gs = MakeGS(b, ATK_PLAYER);
  gs.play_move(Mv(5, 1, false, 3));  // attacker slides to (5,3)
  auto c = gs.canonicalized();
  EXPECT_EQ(c(DEF_LAYER, 5, 4), 0.0f)
      << "defender captured against empty throne";
}

// Rule 6: the throne is NOT hostile to defenders while the king occupies it.
TEST(OpenTaflGS, ThroneNotHostileToDefenderWithKing) {
  BoardTensor b;
  b.setZero();
  b(KING_LAYER, 5, 5) = 1;  // king ON throne
  b(DEF_LAYER, 5, 4) = 1;
  b(ATK_LAYER, 5, 1) = 1;
  auto gs = MakeGS(b, ATK_PLAYER);
  gs.play_move(Mv(5, 1, false, 3));  // attacker to (5,3)
  auto c = gs.canonicalized();
  EXPECT_EQ(c(DEF_LAYER, 5, 4), 1.0f)
      << "occupied throne is not hostile to defenders";
}

// Rule 8: the board edge is NOT a hostile square.
TEST(OpenTaflGS, EdgeIsNotHostile) {
  BoardTensor b;
  b.setZero();
  b(DEF_LAYER, 1, 5) = 1;   // defender one row in from the top edge
  b(ATK_LAYER, 7, 5) = 1;   // hammer below
  b(KING_LAYER, 8, 8) = 1;
  auto gs = MakeGS(b, ATK_PLAYER);
  gs.play_move(Mv(7, 5, true, 2));  // attacker slides up col 5 to (2,5)
  auto c = gs.canonicalized();
  EXPECT_EQ(c(DEF_LAYER, 1, 5), 1.0f)
      << "no capture: the edge behind the defender is not hostile";
}

// Rule 4: a single move may capture in multiple directions at once.
TEST(OpenTaflGS, CaptureInTwoDirectionsAtOnce) {
  BoardTensor b;
  b.setZero();
  b(DEF_LAYER, 3, 2) = 1;
  b(ATK_LAYER, 3, 1) = 1;  // left sandwich
  b(DEF_LAYER, 2, 3) = 1;
  b(ATK_LAYER, 1, 3) = 1;  // up sandwich
  b(ATK_LAYER, 3, 7) = 1;  // hammer
  b(KING_LAYER, 8, 8) = 1;
  auto gs = MakeGS(b, ATK_PLAYER);
  gs.play_move(Mv(3, 7, false, 3));  // attacker lands on (3,3)
  auto c = gs.canonicalized();
  EXPECT_EQ(c(DEF_LAYER, 3, 2), 0.0f) << "left defender captured";
  EXPECT_EQ(c(DEF_LAYER, 2, 3), 0.0f) << "up defender captured";
}

// Rule 9: a piece may safely move between two enemies (capture needs the
// aggressor to close the trap).
TEST(OpenTaflGS, MovingBetweenTwoEnemiesIsSafe) {
  BoardTensor b;
  b.setZero();
  b(ATK_LAYER, 3, 2) = 1;
  b(ATK_LAYER, 3, 4) = 1;   // two attackers flanking (3,3)
  b(DEF_LAYER, 7, 3) = 1;   // defender that moves in
  b(KING_LAYER, 8, 8) = 1;
  auto gs = MakeGS(b, DEF_PLAYER);
  gs.play_move(Mv(7, 3, true, 3));  // defender slides up col 3 to (3,3)
  auto c = gs.canonicalized();
  EXPECT_EQ(c(DEF_LAYER, 3, 3), 1.0f) << "defender that moved in survives";
  EXPECT_EQ(c(ATK_LAYER, 3, 2), 1.0f);
  EXPECT_EQ(c(ATK_LAYER, 3, 4), 1.0f);
}

// Rule 5: non-king pieces pass through the empty throne but cannot land on it.
TEST(OpenTaflGS, NonKingPassesThroughThroneButCannotLand) {
  BoardTensor b;
  b.setZero();
  b(ATK_LAYER, 5, 2) = 1;   // attacker on the throne's row
  b(KING_LAYER, 8, 8) = 1;  // throne empty
  auto gs = MakeGS(b, ATK_PLAYER);
  auto v = gs.valid_moves();
  EXPECT_EQ(v(Mv(5, 2, false, 5)), 0) << "may not land on the throne";
  EXPECT_EQ(v(Mv(5, 2, false, 6)), 1) << "may slide past the empty throne";
}

// Rule 5: the king may re-enter the empty throne.
TEST(OpenTaflGS, KingMayLandOnEmptyThrone) {
  BoardTensor b;
  b.setZero();
  b(KING_LAYER, 5, 2) = 1;  // king on the throne's row, throne empty
  auto gs = MakeGS(b, DEF_PLAYER);
  auto v = gs.valid_moves();
  EXPECT_EQ(v(Mv(5, 2, false, 5)), 1) << "king may re-enter the throne";
}

// Rule 2: fully surrounded defenders lose (encirclement).
TEST(OpenTaflGS, EncirclementIsAttackerWin) {
  BoardTensor b;
  b.setZero();
  for (int x = 3; x <= 7; ++x) {  // solid 5x5 box border of attackers
    b(ATK_LAYER, 3, x) = 1;
    b(ATK_LAYER, 7, x) = 1;
    b(ATK_LAYER, x, 3) = 1;
    b(ATK_LAYER, x, 7) = 1;
  }
  b(KING_LAYER, 5, 5) = 1;  // sealed inside
  auto gs = MakeGS(b, DEF_PLAYER);
  auto s = gs.scores();
  ASSERT_TRUE(s.has_value());
  EXPECT_EQ((*s)(ATK_PLAYER), 1.0f) << "fully surrounded defenders lose";
}

// Rule 2 negative: a wall with a gap is not encirclement.
TEST(OpenTaflGS, NotEncircledGameContinues) {
  BoardTensor b;
  b.setZero();
  for (int x = 3; x <= 7; ++x) {
    b(ATK_LAYER, 3, x) = 1;
    b(ATK_LAYER, 7, x) = 1;
    b(ATK_LAYER, x, 3) = 1;
    b(ATK_LAYER, x, 7) = 1;
  }
  b(ATK_LAYER, 3, 5) = 0;   // gap -> king can reach the edge
  b(KING_LAYER, 5, 5) = 1;
  auto gs = MakeGS(b, DEF_PLAYER);
  EXPECT_FALSE(gs.scores().has_value())
      << "a wall with a gap is not encirclement";
}

// King capture: surrounded on all four cardinal sides.
TEST(OpenTaflGS, KingCapturedOnFourSides) {
  BoardTensor b;
  b.setZero();
  b(KING_LAYER, 3, 3) = 1;
  b(ATK_LAYER, 2, 3) = 1;
  b(ATK_LAYER, 4, 3) = 1;
  b(ATK_LAYER, 3, 2) = 1;
  b(ATK_LAYER, 3, 7) = 1;  // closes the 4th side
  auto gs = MakeGS(b, ATK_PLAYER);
  gs.play_move(Mv(3, 7, false, 4));  // attacker to (3,4)
  auto s = gs.scores();
  ASSERT_TRUE(s.has_value());
  EXPECT_EQ((*s)(ATK_PLAYER), 1.0f) << "king surrounded on 4 sides is captured";
}

// King capture: a king on the board edge cannot be captured.
TEST(OpenTaflGS, KingNotCapturedOnEdge) {
  BoardTensor b;
  b.setZero();
  b(KING_LAYER, 0, 3) = 1;  // king on the top edge
  b(ATK_LAYER, 0, 2) = 1;
  b(ATK_LAYER, 1, 3) = 1;
  b(ATK_LAYER, 0, 7) = 1;
  auto gs = MakeGS(b, ATK_PLAYER);
  gs.play_move(Mv(0, 7, false, 4));  // attacker to (0,4)
  auto c = gs.canonicalized();
  EXPECT_EQ(c(KING_LAYER, 0, 3), 1.0f)
      << "king on the edge cannot be captured";
}

// Rule 1 (2:1 piece ratio) + Rule 2 (attackers move first): starting position.
TEST(OpenTaflGS, StartingPosition) {
  OpenTaflGS gs;
  EXPECT_EQ(gs.current_player(), ATK_PLAYER) << "attackers move first";
  auto c = gs.canonicalized();
  int kings = 0, defs = 0, atks = 0;
  for (int h = 0; h < HEIGHT; ++h) {
    for (int w = 0; w < WIDTH; ++w) {
      if (c(KING_LAYER, h, w) == 1.0f) ++kings;
      if (c(DEF_LAYER, h, w) == 1.0f) ++defs;
      if (c(ATK_LAYER, h, w) == 1.0f) ++atks;
    }
  }
  EXPECT_EQ(kings, 1);
  EXPECT_EQ(defs, 12);
  EXPECT_EQ(atks, 24);  // twice as many attackers as defenders+king
  EXPECT_EQ(c(KING_LAYER, 5, 5), 1.0f) << "king starts on the throne";
}

// Rule 3: pieces slide like a rook -- any vacant distance, blocked by pieces,
// cannot jump.
TEST(OpenTaflGS, RookMovementSlidesAndIsBlocked) {
  BoardTensor b;
  b.setZero();
  b(ATK_LAYER, 3, 0) = 1;   // mover
  b(ATK_LAYER, 3, 5) = 1;   // blocker
  b(KING_LAYER, 8, 8) = 1;
  auto gs = MakeGS(b, ATK_PLAYER);
  auto v = gs.valid_moves();
  EXPECT_EQ(v(Mv(3, 0, false, 4)), 1) << "slides multiple vacant squares";
  EXPECT_EQ(v(Mv(3, 0, false, 5)), 0) << "cannot land on an occupied square";
  EXPECT_EQ(v(Mv(3, 0, false, 6)), 0) << "cannot jump over a piece";
}

// Rule 4 (orthogonal): a diagonal sandwich does not capture.
TEST(OpenTaflGS, DiagonalSandwichDoesNotCapture) {
  BoardTensor b;
  b.setZero();
  b(DEF_LAYER, 3, 3) = 1;
  b(ATK_LAYER, 2, 2) = 1;   // diagonal to the victim
  b(ATK_LAYER, 4, 7) = 1;   // hammer -> (4,4), the other diagonal
  b(KING_LAYER, 8, 8) = 1;
  auto gs = MakeGS(b, ATK_PLAYER);
  gs.play_move(Mv(4, 7, false, 4));  // attacker to (4,4)
  auto c = gs.canonicalized();
  EXPECT_EQ(c(DEF_LAYER, 3, 3), 1.0f) << "diagonal sandwich must not capture";
}

// Rule 4: the king is armed and may take part in captures.
TEST(OpenTaflGS, KingParticipatesInCapture) {
  BoardTensor b;
  b.setZero();
  b(ATK_LAYER, 3, 3) = 1;   // victim attacker
  b(DEF_LAYER, 3, 2) = 1;   // anvil
  b(KING_LAYER, 3, 7) = 1;  // king is the hammer
  auto gs = MakeGS(b, DEF_PLAYER);
  gs.play_move(Mv(3, 7, false, 4));  // king slides to (3,4)
  auto c = gs.canonicalized();
  EXPECT_EQ(c(ATK_LAYER, 3, 3), 0.0f) << "king helps capture the attacker";
}

// Rule 5: restricted corner squares may not be occupied by non-king pieces.
TEST(OpenTaflGS, NonKingCannotLandOnCorner) {
  BoardTensor b;
  b.setZero();
  b(ATK_LAYER, 0, 3) = 1;
  b(KING_LAYER, 8, 8) = 1;
  auto gs = MakeGS(b, ATK_PLAYER);
  auto v = gs.valid_moves();
  EXPECT_EQ(v(Mv(0, 3, false, 0)), 0) << "non-king may not land on a corner";
  EXPECT_EQ(v(Mv(0, 3, false, 1)), 1) << "but may move to a normal edge square";
}

// Rule 5: the throne is hostile to attackers -- when empty...
TEST(OpenTaflGS, ThroneHostileToAttackersWhenEmpty) {
  BoardTensor b;
  b.setZero();
  b(ATK_LAYER, 5, 4) = 1;   // attacker next to throne
  b(DEF_LAYER, 5, 1) = 1;   // hammer
  b(KING_LAYER, 8, 8) = 1;  // throne empty
  auto gs = MakeGS(b, DEF_PLAYER);
  gs.play_move(Mv(5, 1, false, 3));  // defender to (5,3)
  auto c = gs.canonicalized();
  EXPECT_EQ(c(ATK_LAYER, 5, 4), 0.0f)
      << "attacker captured against empty throne";
}

// ...and even while the king occupies it (always hostile to attackers).
TEST(OpenTaflGS, ThroneHostileToAttackersWithKing) {
  BoardTensor b;
  b.setZero();
  b(KING_LAYER, 5, 5) = 1;  // king ON throne
  b(ATK_LAYER, 5, 4) = 1;
  b(DEF_LAYER, 5, 1) = 1;
  auto gs = MakeGS(b, DEF_PLAYER);
  gs.play_move(Mv(5, 1, false, 3));  // defender to (5,3)
  auto c = gs.canonicalized();
  EXPECT_EQ(c(ATK_LAYER, 5, 4), 0.0f)
      << "throne is hostile to attackers even when occupied by the king";
}

// Rule 6: the king wins by reaching a corner.
TEST(OpenTaflGS, KingEscapesToCorner) {
  BoardTensor b;
  b.setZero();
  b(KING_LAYER, 0, 3) = 1;  // king on the top edge
  auto gs = MakeGS(b, DEF_PLAYER);
  gs.play_move(Mv(0, 3, false, 0));  // king slides to corner (0,0)
  auto s = gs.scores();
  ASSERT_TRUE(s.has_value());
  EXPECT_EQ((*s)(DEF_PLAYER), 1.0f) << "king reaching a corner wins";
}

// Rule 7a: next to the throne, the king is captured by 3 attackers + throne.
TEST(OpenTaflGS, KingCapturedNextToThrone) {
  BoardTensor b;
  b.setZero();
  b(KING_LAYER, 5, 4) = 1;  // adjacent to the (empty, hostile) throne (5,5)
  b(ATK_LAYER, 4, 4) = 1;
  b(ATK_LAYER, 6, 4) = 1;
  b(ATK_LAYER, 5, 7) = 1;   // closes the 3rd side at (5,3)
  auto gs = MakeGS(b, ATK_PLAYER);
  gs.play_move(Mv(5, 7, false, 3));  // attacker to (5,3)
  auto s = gs.scores();
  ASSERT_TRUE(s.has_value());
  EXPECT_EQ((*s)(ATK_PLAYER), 1.0f)
      << "king captured by 3 attackers plus the hostile throne";
}

// Rule 7a-iii: a lone king completely surrounded on the edge is a loss for the
// defenders. We don't implement edge king-capture, but the same outcome arises
// via the stalemate rule (the immobile king has no moves) -- result matches.
TEST(OpenTaflGS, LoneSurroundedEdgeKingLoses) {
  BoardTensor b;
  b.setZero();
  b(KING_LAYER, 0, 5) = 1;  // lone king on the top edge
  b(ATK_LAYER, 0, 4) = 1;
  b(ATK_LAYER, 1, 5) = 1;
  b(ATK_LAYER, 0, 8) = 1;
  auto gs = MakeGS(b, ATK_PLAYER);
  gs.play_move(Mv(0, 8, false, 6));  // close the king's last on-board escape
  auto s = gs.scores();
  ASSERT_TRUE(s.has_value());
  EXPECT_EQ((*s)(ATK_PLAYER), 1.0f) << "lone surrounded edge king loses";
}

// Rule 8: a player with no legal moves loses.
TEST(OpenTaflGS, NoLegalMovesLoses) {
  BoardTensor b;
  b.setZero();
  b(ATK_LAYER, 0, 1) = 1;   // the only attacker, boxed in
  b(DEF_LAYER, 0, 2) = 1;   // blocks right (left is corner, up is off-board)
  b(DEF_LAYER, 1, 1) = 1;   // blocks down
  b(KING_LAYER, 5, 5) = 1;
  auto gs = MakeGS(b, ATK_PLAYER);
  auto s = gs.scores();
  ASSERT_TRUE(s.has_value());
  EXPECT_EQ((*s)(DEF_PLAYER), 1.0f) << "no legal moves -> the mover loses";
}

// Rule 9 / impl: reaching the turn cap with no decisive result is a draw.
TEST(OpenTaflGS, MaxTurnsIsADraw) {
  BoardTensor b;
  b.setZero();
  b(KING_LAYER, 5, 5) = 1;
  b(DEF_LAYER, 5, 4) = 1;
  b(ATK_LAYER, 0, 5) = 1;
  b(ATK_LAYER, 10, 5) = 1;
  auto gs = MakeGS(b, ATK_PLAYER, DEFAULT_MAX_TURNS);
  auto s = gs.scores();
  ASSERT_TRUE(s.has_value());
  EXPECT_EQ((*s)(NUM_PLAYERS), 1.0f) << "reaching max turns is a draw";
}

}  // namespace
}  // namespace alphazero::opentafl_gs