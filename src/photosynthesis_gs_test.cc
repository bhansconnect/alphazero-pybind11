
#include "photosynthesis_gs.h"

#include "gtest/gtest.h"

namespace alphazero::photosynthesis_gs {
namespace {

// NOLINTNEXTLINE
TEST(PhotosynthesisGS, Equals) {
  auto gs = PhotosynthesisGS<3>();
  gs.play_move(3);
  gs.play_move(6);
  gs.play_move(27);
  gs.play_move(45);
  gs.play_move(42);
  gs.play_move(44);
  std::cout << gs.dump();
  auto m = gs.valid_moves();
  for (auto i = 0; i < m.size(); ++i) {
    if (m(i) == 1) {
      std::cout << i << '\n';
    }
  }
  // EXPECT_TRUE(false);
}

}  // namespace
}  // namespace alphazero::photosynthesis_gs