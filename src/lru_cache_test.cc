#include "lru_cache.h"

#include "gtest/gtest.h"

namespace alphazero {
namespace {

// NOLINTNEXTLINE
TEST(LRUCache, Basics) {
  auto cache = LRUCache<int, bool>{3};
  cache.insert(1, true);
  cache.insert(2, true);
  cache.insert(3, false);

  EXPECT_EQ(cache.find(3), false);
  EXPECT_EQ(cache.find(2), true);
  EXPECT_EQ(cache.find(1), true);

  // Inserting a different value does nothing. Cache does not update.
  cache.insert(1, false);
  EXPECT_EQ(cache.find(1), true);

  // Insering 4 removes 2 due to it being LRU.
  cache.find(1);
  cache.find(3);
  cache.insert(4, false);
  EXPECT_EQ(cache.find(2), std::nullopt);
  EXPECT_EQ(cache.find(4), false);
  EXPECT_EQ(cache.find(1), true);
  EXPECT_EQ(cache.find(3), false);
}

}  // namespace
}  // namespace alphazero