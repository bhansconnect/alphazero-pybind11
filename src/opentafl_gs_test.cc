
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

}  // namespace
}  // namespace alphazero::opentafl_gs