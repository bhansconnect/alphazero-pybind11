
#include "brandubh_gs.h"

#include "gtest/gtest.h"

namespace alphazero::brandubh_gs {
namespace {

// NOLINTNEXTLINE
TEST(BrandubhGS, RepetitionCount) {
  auto gs = BrandubhGS().copy();
  gs->play_move((3 * WIDTH + 5) * (WIDTH + HEIGHT) + WIDTH + 4);
  gs->play_move((4 * WIDTH + 5) * (WIDTH + HEIGHT) + WIDTH + 3);
  std::cout << gs->dump();
  gs->play_move((3 * WIDTH + 5) * (WIDTH + HEIGHT) + WIDTH + 4);
  gs->play_move((4 * WIDTH + 5) * (WIDTH + HEIGHT) + WIDTH + 3);
  std::cout << gs->dump();
  gs = gs->copy();
  gs->play_move((3 * WIDTH + 5) * (WIDTH + HEIGHT) + WIDTH + 4);
  gs->play_move((4 * WIDTH + 5) * (WIDTH + HEIGHT) + WIDTH + 3);
  std::cout << gs->dump();
  gs->play_move((3 * WIDTH + 5) * (WIDTH + HEIGHT) + WIDTH + 4);
  gs->play_move((4 * WIDTH + 5) * (WIDTH + HEIGHT) + WIDTH + 3);
  std::cout << gs->dump();
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
}  // namespace alphazero::brandubh_gs