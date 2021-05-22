
#include "photosynthesis_gs.h"

#include "gtest/gtest.h"

namespace alphazero::photosynthesis_gs {
namespace {

// NOLINTNEXTLINE
TEST(PhotosynthesisGS, Equals) {
  auto gs = PhotosynthesisGS<3>();
  std::cout << gs.dump();
  gs.canonicalized();
  EXPECT_TRUE(false);
}

}  // namespace
}  // namespace alphazero::photosynthesis_gs