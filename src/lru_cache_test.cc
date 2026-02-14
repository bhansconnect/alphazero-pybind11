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

// NOLINTNEXTLINE
TEST(LRUCache, FindInto) {
  using V = std::tuple<int, int>;
  auto cache = LRUCache<int, V>{3};
  cache.insert(1, V{10, 20});
  cache.insert(2, V{30, 40});

  int out1 = 0, out2 = 0;
  EXPECT_TRUE(cache.find_into(1, out1, out2));
  EXPECT_EQ(out1, 10);
  EXPECT_EQ(out2, 20);

  EXPECT_TRUE(cache.find_into(2, out1, out2));
  EXPECT_EQ(out1, 30);
  EXPECT_EQ(out2, 40);

  // Miss
  EXPECT_FALSE(cache.find_into(99, out1, out2));
  // out1/out2 unchanged on miss
  EXPECT_EQ(out1, 30);
  EXPECT_EQ(out2, 40);

  // Verify hits/misses tracked
  EXPECT_EQ(cache.hits(), 2);
  EXPECT_EQ(cache.misses(), 1);
}

// NOLINTNEXTLINE
TEST(LRUCache, Evictions) {
  auto cache = LRUCache<int, bool>{3};
  cache.insert(1, true);
  cache.insert(2, true);
  cache.insert(3, true);
  EXPECT_EQ(cache.evictions(), 0);

  cache.insert(4, true);
  EXPECT_EQ(cache.evictions(), 1);

  cache.insert(5, true);
  EXPECT_EQ(cache.evictions(), 2);

  EXPECT_EQ(cache.size(), 3);
}

// NOLINTNEXTLINE
TEST(LRUCache, EvictionsInsertMany) {
  auto cache = LRUCache<int, bool>{3};
  std::vector<int> ks = {1, 2, 3, 4, 5};
  std::vector<bool> vs = {true, true, true, true, true};
  cache.insert_many(ks, vs);

  EXPECT_EQ(cache.evictions(), 2);
  EXPECT_EQ(cache.size(), 3);
}

// NOLINTNEXTLINE
TEST(LRUCache, MaxSize) {
  auto cache = LRUCache<int, bool>{42};
  EXPECT_EQ(cache.max_size(), 42);
}

// NOLINTNEXTLINE
TEST(ShardedLRUCache, InsertManyBatched) {
  auto cache = ShardedLRUCache<int, bool>{6, 3};
  std::vector<int> ks = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<bool> vs = {true, true, true, true, true, true, true, true};
  cache.insert_many(ks, vs);

  // All items should be findable (some evicted due to per-shard capacity)
  size_t found = 0;
  for (int k = 1; k <= 8; ++k) {
    if (cache.find(k).has_value()) ++found;
  }
  // With 6 total capacity across 3 shards (2 each), at most 6 items fit
  EXPECT_LE(found, 6);
  EXPECT_EQ(cache.size(), found);
  // Evictions should have occurred (8 items into 6 capacity)
  EXPECT_GE(cache.evictions(), 2);
}

// NOLINTNEXTLINE
TEST(ShardedLRUCache, FindInto) {
  using V = std::tuple<int, int>;
  auto cache = ShardedLRUCache<int, V>{100, 4};
  cache.insert(1, V{10, 20});
  cache.insert(2, V{30, 40});

  int out1 = 0, out2 = 0;
  EXPECT_TRUE(cache.find_into(1, out1, out2));
  EXPECT_EQ(out1, 10);
  EXPECT_EQ(out2, 20);

  EXPECT_TRUE(cache.find_into(2, out1, out2));
  EXPECT_EQ(out1, 30);
  EXPECT_EQ(out2, 40);

  EXPECT_FALSE(cache.find_into(99, out1, out2));
}

// NOLINTNEXTLINE
TEST(ShardedLRUCache, EvictionsAndMaxSize) {
  auto cache = ShardedLRUCache<int, bool>{6, 3};
  EXPECT_EQ(cache.max_size(), 6);
  EXPECT_EQ(cache.evictions(), 0);

  // Insert more than capacity
  for (int i = 0; i < 20; ++i) {
    cache.insert(i, true);
  }
  EXPECT_GT(cache.evictions(), 0);
  EXPECT_LE(cache.size(), 6);
}

// NOLINTNEXTLINE
TEST(LRUCache, GhostReinserts) {
  // Cache of size 3 with ghost of size 3
  auto cache = LRUCache<int, bool>{3, 3};
  cache.insert(1, true);
  cache.insert(2, true);
  cache.insert(3, true);
  EXPECT_EQ(cache.reinserts(), 0);

  // Evict 1 by inserting 4
  cache.insert(4, true);
  EXPECT_EQ(cache.evictions(), 1);

  // Miss on 1 — should be a reinsert (was in ghost)
  EXPECT_EQ(cache.find(1), std::nullopt);
  EXPECT_EQ(cache.reinserts(), 1);

  // Miss on 1 again — ghost entry consumed, not a reinsert
  EXPECT_EQ(cache.find(1), std::nullopt);
  EXPECT_EQ(cache.reinserts(), 1);

  // Miss on 99 — never cached, not a reinsert
  EXPECT_EQ(cache.find(99), std::nullopt);
  EXPECT_EQ(cache.reinserts(), 1);
}

// NOLINTNEXTLINE
TEST(LRUCache, GhostDisabled) {
  // Ghost size 0 — no reinserts tracked
  auto cache = LRUCache<int, bool>{3, 0};
  cache.insert(1, true);
  cache.insert(2, true);
  cache.insert(3, true);
  cache.insert(4, true);  // evicts 1
  EXPECT_EQ(cache.find(1), std::nullopt);
  EXPECT_EQ(cache.reinserts(), 0);
}

// NOLINTNEXTLINE
TEST(LRUCache, GhostOverflow) {
  // Cache of size 2, ghost of size 2
  auto cache = LRUCache<int, bool>{2, 2};
  cache.insert(1, true);
  cache.insert(2, true);

  // Evict 1, 2, 3 — ghost only holds last 2 (keys 2 and 3)
  cache.insert(3, true);  // evicts 1, ghost: [1]
  cache.insert(4, true);  // evicts 2, ghost: [1, 2]
  cache.insert(5, true);  // evicts 3, ghost: [2, 3] (1 evicted from ghost)

  // Miss on 1 — evicted from ghost, not a reinsert
  EXPECT_EQ(cache.find(1), std::nullopt);
  EXPECT_EQ(cache.reinserts(), 0);

  // Miss on 2 — still in ghost
  EXPECT_EQ(cache.find(2), std::nullopt);
  EXPECT_EQ(cache.reinserts(), 1);
}

// NOLINTNEXTLINE
TEST(LRUCache, GhostFindInto) {
  using V = std::tuple<int, int>;
  auto cache = LRUCache<int, V>{2, 2};
  cache.insert(1, V{10, 20});
  cache.insert(2, V{30, 40});
  cache.insert(3, V{50, 60});  // evicts 1

  int out1 = 0, out2 = 0;
  EXPECT_FALSE(cache.find_into(1, out1, out2));
  EXPECT_EQ(cache.reinserts(), 1);
}

// NOLINTNEXTLINE
TEST(ShardedLRUCache, Reinserts) {
  // Single shard for deterministic behavior
  auto cache = ShardedLRUCache<int, bool>{3, 1, 3};
  cache.insert(1, true);
  cache.insert(2, true);
  cache.insert(3, true);
  cache.insert(4, true);  // evicts 1

  EXPECT_EQ(cache.find(1), std::nullopt);
  EXPECT_EQ(cache.reinserts(), 1);

  // Miss on uncached key — not a reinsert
  EXPECT_EQ(cache.find(99), std::nullopt);
  EXPECT_EQ(cache.reinserts(), 1);
}

}  // namespace
}  // namespace alphazero
