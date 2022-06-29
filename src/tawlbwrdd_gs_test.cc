
#include "tawlbwrdd_gs.h"

#include "gtest/gtest.h"

namespace alphazero::tawlbwrdd_gs {
namespace {

// NOLINTNEXTLINE
TEST(TawlbwrddGS, RepetitionCount) {
  auto gs = TawlbwrddGS().copy();
  std::cout << gs->dump();
  gs->play_move((0 * WIDTH + 4) * (WIDTH + HEIGHT) + 3);
  gs->play_move((2 * WIDTH + 5) * (WIDTH + HEIGHT) + 4);
  std::cout << gs->dump();
  gs->play_move((0 * WIDTH + 3) * (WIDTH + HEIGHT) + 4);
  gs->play_move((2 * WIDTH + 4) * (WIDTH + HEIGHT) + 5);
  std::cout << gs->dump();
  gs = gs->copy();
  gs->play_move((0 * WIDTH + 4) * (WIDTH + HEIGHT) + 3);
  gs->play_move((2 * WIDTH + 5) * (WIDTH + HEIGHT) + 4);
  std::cout << gs->dump();
  gs->play_move((0 * WIDTH + 3) * (WIDTH + HEIGHT) + 4);
  gs->play_move((2 * WIDTH + 4) * (WIDTH + HEIGHT) + 5);
  std::cout << gs->dump();
  auto m = gs->valid_moves();
  for (auto i = 0; i < m.size(); ++i) {
    if (m(i) == 1) {
      auto new_loc = i % (WIDTH + HEIGHT);
      auto height_move = new_loc >= WIDTH;
      if (height_move) {
        new_loc -= WIDTH;
      }
      auto piece_loc = (i / (WIDTH + HEIGHT));
      auto piece_w = piece_loc % WIDTH;
      auto piece_h = piece_loc / WIDTH;

      auto new_h = piece_h;
      auto new_w = piece_w;
      if (height_move) {
        new_h = new_loc;
      } else {
        new_w = new_loc;
      }
      std::cout << piece_h << ',' << piece_w << " -> " << new_h << ',' << new_w
                << '\n';
    }
  }
  auto s = gs->scores().value_or(SizedVector<float, 3>{0, 0, 0});

  std::cout << s(0) << ", " << s(1) << ", " << s(2);
  auto expected = SizedVector<float, 3>{1, 0, 0};
  EXPECT_EQ(s, expected);
}

}  // namespace
}  // namespace alphazero::tawlbwrdd_gs