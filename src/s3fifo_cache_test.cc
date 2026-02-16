#include "s3fifo_cache.h"

#include <thread>
#include <vector>

#include "gtest/gtest.h"

namespace alphazero {
namespace {

// Helper: create a simple policy/value array for testing.
std::vector<float> make_policy(uint32_t n, float base) {
  std::vector<float> p(n);
  for (uint32_t i = 0; i < n; ++i) p[i] = base + static_cast<float>(i);
  return p;
}

std::vector<float> make_value(uint32_t n, float base) {
  std::vector<float> v(n);
  for (uint32_t i = 0; i < n; ++i) v[i] = base + static_cast<float>(i) * 0.1f;
  return v;
}

// ---------------------------------------------------------------------------
// Basic operations
// ---------------------------------------------------------------------------

TEST(S3FIFOCache, InsertAndFind) {
  S3FIFOCache cache(10, 5, 3, 2);
  auto pol = make_policy(3, 1.0f);
  auto val = make_value(2, 0.5f);
  cache.insert(42, pol.data(), val.data());

  std::vector<float> pol_out(3), val_out(2);
  EXPECT_TRUE(cache.find(42, pol_out.data(), val_out.data()));
  EXPECT_EQ(pol_out, pol);
  EXPECT_EQ(val_out, val);
}

TEST(S3FIFOCache, FindMiss) {
  S3FIFOCache cache(10, 5, 3, 2);
  std::vector<float> pol_out(3), val_out(2);
  EXPECT_FALSE(cache.find(99, pol_out.data(), val_out.data()));
  EXPECT_EQ(cache.misses(), 1);
}

TEST(S3FIFOCache, DuplicateInsertIsNoop) {
  S3FIFOCache cache(10, 5, 3, 2);
  auto pol1 = make_policy(3, 1.0f);
  auto val1 = make_value(2, 0.5f);
  cache.insert(42, pol1.data(), val1.data());

  // Insert again with different data — should be ignored.
  auto pol2 = make_policy(3, 100.0f);
  auto val2 = make_value(2, 100.0f);
  cache.insert(42, pol2.data(), val2.data());

  std::vector<float> pol_out(3), val_out(2);
  EXPECT_TRUE(cache.find(42, pol_out.data(), val_out.data()));
  EXPECT_EQ(pol_out, pol1);
  EXPECT_EQ(val_out, val1);
  EXPECT_EQ(cache.size(), 1);
}

TEST(S3FIFOCache, StatsTracking) {
  S3FIFOCache cache(10, 5, 3, 2);
  auto pol = make_policy(3, 1.0f);
  auto val = make_value(2, 0.5f);
  cache.insert(1, pol.data(), val.data());

  std::vector<float> po(3), vo(2);
  cache.find(1, po.data(), vo.data());  // hit
  cache.find(1, po.data(), vo.data());  // hit
  cache.find(2, po.data(), vo.data());  // miss
  EXPECT_EQ(cache.hits(), 2);
  EXPECT_EQ(cache.misses(), 1);
  EXPECT_EQ(cache.size(), 1);
}

// ---------------------------------------------------------------------------
// S3-FIFO eviction behavior
// ---------------------------------------------------------------------------

TEST(S3FIFOCache, EvictionKeepsSize) {
  constexpr uint32_t cap = 10;
  S3FIFOCache cache(cap, 5, 2, 1);
  auto pol = make_policy(2, 0);
  auto val = make_value(1, 0);
  for (uint32_t i = 0; i < cap + 5; ++i) {
    cache.insert(i, pol.data(), val.data());
  }
  EXPECT_EQ(cache.size(), cap);
  EXPECT_EQ(cache.evictions(), 5);
}

TEST(S3FIFOCache, ScanResistance) {
  // Insert N items, access one repeatedly, then fill more.
  // The accessed item should survive (promoted S→M), while one-shot items evict.
  constexpr uint32_t cap = 10;
  S3FIFOCache cache(cap, 5, 2, 1);
  auto pol = make_policy(2, 0);
  auto val = make_value(1, 0);
  std::vector<float> po(2), vo(1);

  // Fill cache.
  for (uint32_t i = 0; i < cap; ++i) {
    cache.insert(i, pol.data(), val.data());
  }
  // Access item 0 multiple times (sets freq=1).
  cache.find(0, po.data(), vo.data());
  cache.find(0, po.data(), vo.data());

  // Insert more items to cause eviction.
  for (uint32_t i = cap; i < cap + cap; ++i) {
    cache.insert(i, pol.data(), val.data());
  }

  // Item 0 should have survived (promoted to Main).
  EXPECT_TRUE(cache.find(0, po.data(), vo.data()));
}

TEST(S3FIFOCache, SmallToMainPromotion) {
  // Insert items, access one, trigger eviction from Small.
  // Accessed item promoted to Main, unaccessed evicted.
  constexpr uint32_t cap = 5;
  S3FIFOCache cache(cap, 3, 2, 1);
  auto pol = make_policy(2, 0);
  auto val = make_value(1, 0);
  std::vector<float> po(2), vo(1);

  // Fill cache.
  for (uint32_t i = 0; i < cap; ++i) {
    cache.insert(i, pol.data(), val.data());
  }

  // Access item 0.
  cache.find(0, po.data(), vo.data());

  // Insert one more to trigger eviction. Item 0 (freq=1) should be promoted,
  // and item 1 (freq=0, next oldest in S) should be evicted.
  cache.insert(100, pol.data(), val.data());

  EXPECT_TRUE(cache.find(0, po.data(), vo.data()));  // promoted to M
  // Some of the unaccessed items should have been evicted.
  EXPECT_EQ(cache.size(), cap);
}

TEST(S3FIFOCache, GhostAdmitsToMain) {
  // Fill and evict an item so it enters ghost. Re-insert it; should go to Main.
  constexpr uint32_t cap = 3;
  S3FIFOCache cache(cap, 5, 2, 1);
  auto pol = make_policy(2, 0);
  auto val = make_value(1, 0);
  std::vector<float> po(2), vo(1);

  // Fill cache with 0,1,2.
  for (uint32_t i = 0; i < cap; ++i) {
    cache.insert(i, pol.data(), val.data());
  }

  // Insert 3 — evicts 0 to ghost.
  cache.insert(3, pol.data(), val.data());
  EXPECT_FALSE(cache.find(0, po.data(), vo.data()));

  // Re-insert 0 — ghost hit, should go to Main.
  cache.insert(0, pol.data(), val.data());
  EXPECT_TRUE(cache.find(0, po.data(), vo.data()));

  // Item 0 should now be more resilient to eviction (it's in Main, not Small).
  // Insert several more items to flush Small. Item 0 should survive.
  cache.insert(10, pol.data(), val.data());
  cache.insert(11, pol.data(), val.data());
  EXPECT_TRUE(cache.find(0, po.data(), vo.data()));
}

TEST(S3FIFOCache, MainClockSweep) {
  // Fill Main, access some items, trigger M eviction.
  // Accessed items get second chance, unaccessed evicted.
  constexpr uint32_t cap = 5;
  S3FIFOCache cache(cap, 3, 2, 1);
  auto pol = make_policy(2, 0);
  auto val = make_value(1, 0);
  std::vector<float> po(2), vo(1);

  // Fill cache with items 0..4.
  for (uint32_t i = 0; i < cap; ++i) {
    cache.insert(i, pol.data(), val.data());
  }

  // Access all items (they'll get freq=1, promoting to Main when S evicts).
  for (uint32_t i = 0; i < cap; ++i) {
    cache.find(i, po.data(), vo.data());
  }

  // Insert cap more items. All original items promote to M, then M must evict.
  for (uint32_t i = 100; i < 100 + cap; ++i) {
    cache.insert(i, pol.data(), val.data());
  }

  // Size should still be cap.
  EXPECT_EQ(cache.size(), cap);
  EXPECT_GT(cache.evictions(), 0);
}

TEST(S3FIFOCache, EvictionCascade) {
  // Fill S with all-accessed items, trigger eviction. All promote to M,
  // then M evicts. Verify no infinite loop.
  constexpr uint32_t cap = 5;
  S3FIFOCache cache(cap, 0, 2, 1);
  auto pol = make_policy(2, 0);
  auto val = make_value(1, 0);
  std::vector<float> po(2), vo(1);

  // Fill and access everything.
  for (uint32_t i = 0; i < cap; ++i) {
    cache.insert(i, pol.data(), val.data());
    cache.find(i, po.data(), vo.data());
  }

  // Insert one more — must cascade: S items all promote to M, then M evicts.
  cache.insert(99, pol.data(), val.data());
  EXPECT_EQ(cache.size(), cap);
}

// ---------------------------------------------------------------------------
// Ghost tracking
// ---------------------------------------------------------------------------

TEST(S3FIFOCache, GhostReinsertOnFind) {
  // Reinserts counted on find (miss for recently evicted key).
  constexpr uint32_t cap = 3;
  S3FIFOCache cache(cap, 5, 2, 1);
  auto pol = make_policy(2, 0);
  auto val = make_value(1, 0);
  std::vector<float> po(2), vo(1);

  for (uint32_t i = 0; i < cap; ++i) {
    cache.insert(i, pol.data(), val.data());
  }
  cache.insert(3, pol.data(), val.data());  // evicts 0 to ghost

  EXPECT_FALSE(cache.find(0, po.data(), vo.data()));  // ghost hit → reinsert
  EXPECT_EQ(cache.reinserts(), 1);

  // Ghost entry NOT consumed by find — should still count on another find.
  EXPECT_FALSE(cache.find(0, po.data(), vo.data()));
  EXPECT_EQ(cache.reinserts(), 2);
}

TEST(S3FIFOCache, GhostConsumedOnInsert) {
  constexpr uint32_t cap = 3;
  S3FIFOCache cache(cap, 5, 2, 1);
  auto pol = make_policy(2, 0);
  auto val = make_value(1, 0);
  std::vector<float> po(2), vo(1);

  for (uint32_t i = 0; i < cap; ++i) {
    cache.insert(i, pol.data(), val.data());
  }
  cache.insert(3, pol.data(), val.data());  // evicts 0 to ghost

  // Insert 0 back — consumes ghost entry (routes to Main).
  cache.insert(0, pol.data(), val.data());

  // Now a find-miss on 0 won't hit ghost (it was consumed).
  // But 0 is in the cache now, so find returns true.
  EXPECT_TRUE(cache.find(0, po.data(), vo.data()));
}

TEST(S3FIFOCache, GhostOverflow) {
  constexpr uint32_t cap = 2;
  constexpr uint32_t ghost = 2;
  S3FIFOCache cache(cap, ghost, 2, 1);
  auto pol = make_policy(2, 0);
  auto val = make_value(1, 0);
  std::vector<float> po(2), vo(1);

  cache.insert(1, pol.data(), val.data());
  cache.insert(2, pol.data(), val.data());
  cache.insert(3, pol.data(), val.data());  // evicts 1, ghost: [1]
  cache.insert(4, pol.data(), val.data());  // evicts 2, ghost: [1, 2]
  cache.insert(5, pol.data(), val.data());  // evicts 3, ghost: [2, 3] (1 out)

  // 1 was evicted from ghost.
  cache.find(1, po.data(), vo.data());
  EXPECT_EQ(cache.reinserts(), 0);

  // 2 still in ghost.
  cache.find(2, po.data(), vo.data());
  EXPECT_EQ(cache.reinserts(), 1);
}

TEST(S3FIFOCache, GhostDisabled) {
  S3FIFOCache cache(3, 0, 2, 1);
  auto pol = make_policy(2, 0);
  auto val = make_value(1, 0);
  std::vector<float> po(2), vo(1);

  cache.insert(1, pol.data(), val.data());
  cache.insert(2, pol.data(), val.data());
  cache.insert(3, pol.data(), val.data());
  cache.insert(4, pol.data(), val.data());  // evicts 1

  cache.find(1, po.data(), vo.data());
  EXPECT_EQ(cache.reinserts(), 0);
}

// ---------------------------------------------------------------------------
// Flat storage integrity
// ---------------------------------------------------------------------------

TEST(S3FIFOCache, FlatStorageIntegrity) {
  S3FIFOCache cache(10, 5, 4, 3);

  auto pol1 = make_policy(4, 1.0f);
  auto val1 = make_value(3, 0.1f);
  auto pol2 = make_policy(4, 10.0f);
  auto val2 = make_value(3, 1.0f);
  auto pol3 = make_policy(4, 100.0f);
  auto val3 = make_value(3, 10.0f);

  cache.insert(1, pol1.data(), val1.data());
  cache.insert(2, pol2.data(), val2.data());
  cache.insert(3, pol3.data(), val3.data());

  std::vector<float> po(4), vo(3);

  EXPECT_TRUE(cache.find(1, po.data(), vo.data()));
  EXPECT_EQ(po, pol1);
  EXPECT_EQ(vo, val1);

  EXPECT_TRUE(cache.find(2, po.data(), vo.data()));
  EXPECT_EQ(po, pol2);
  EXPECT_EQ(vo, val2);

  EXPECT_TRUE(cache.find(3, po.data(), vo.data()));
  EXPECT_EQ(po, pol3);
  EXPECT_EQ(vo, val3);
}

// ---------------------------------------------------------------------------
// Sharding
// ---------------------------------------------------------------------------

TEST(ShardedS3FIFOCache, InsertManyDistributes) {
  ShardedS3FIFOCache cache(100, 4, 50, 3, 2);

  constexpr uint32_t N = 20;
  std::vector<uint64_t> hashes(N);
  std::vector<std::vector<float>> pols(N), vals(N);
  std::vector<const float*> pol_ptrs(N), val_ptrs(N);
  for (uint32_t i = 0; i < N; ++i) {
    hashes[i] = i;
    pols[i] = make_policy(3, static_cast<float>(i));
    vals[i] = make_value(2, static_cast<float>(i));
    pol_ptrs[i] = pols[i].data();
    val_ptrs[i] = vals[i].data();
  }
  cache.insert_many(hashes.data(), pol_ptrs.data(), val_ptrs.data(), N);
  EXPECT_EQ(cache.size(), N);

  // All items should be found.
  std::vector<float> po(3), vo(2);
  for (uint32_t i = 0; i < N; ++i) {
    EXPECT_TRUE(cache.find(i, po.data(), vo.data()));
    EXPECT_EQ(po, pols[i]);
    EXPECT_EQ(vo, vals[i]);
  }
}

TEST(ShardedS3FIFOCache, StatsAggregate) {
  ShardedS3FIFOCache cache(20, 4, 10, 2, 1);
  auto pol = make_policy(2, 0);
  auto val = make_value(1, 0);
  std::vector<float> po(2), vo(1);

  for (uint32_t i = 0; i < 20; ++i) {
    cache.insert(i, pol.data(), val.data());
  }
  for (uint32_t i = 0; i < 20; ++i) {
    cache.find(i, po.data(), vo.data());
  }
  for (uint32_t i = 100; i < 105; ++i) {
    cache.find(i, po.data(), vo.data());
  }

  EXPECT_EQ(cache.hits(), 20);
  EXPECT_EQ(cache.misses(), 5);
  EXPECT_EQ(cache.size(), 20);
  EXPECT_EQ(cache.max_size(), 20);
}

// ---------------------------------------------------------------------------
// Thread safety
// ---------------------------------------------------------------------------

TEST(S3FIFOCache, ConcurrentFindInsert) {
  S3FIFOCache cache(1000, 500, 4, 2);
  constexpr int kThreads = 4;
  constexpr int kOps = 5000;

  std::vector<std::thread> threads;
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&cache, t]() {
      auto pol = make_policy(4, static_cast<float>(t));
      auto val = make_value(2, static_cast<float>(t));
      std::vector<float> po(4), vo(2);
      for (int i = 0; i < kOps; ++i) {
        uint64_t h = static_cast<uint64_t>(t * kOps + i);
        cache.insert(h, pol.data(), val.data());
        cache.find(h, po.data(), vo.data());
      }
    });
  }
  for (auto& t : threads) t.join();

  EXPECT_LE(cache.size(), 1000);
  EXPECT_EQ(cache.hits() + cache.misses(),
            static_cast<size_t>(kThreads) * kOps);
}

TEST(ShardedS3FIFOCache, ConcurrentInsertMany) {
  ShardedS3FIFOCache cache(1000, 4, 500, 3, 2);
  constexpr int kThreads = 4;
  constexpr int kBatchSize = 100;

  std::vector<std::thread> threads;
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&cache, t]() {
      std::vector<uint64_t> hashes(kBatchSize);
      std::vector<std::vector<float>> pols(kBatchSize), vals(kBatchSize);
      std::vector<const float*> pp(kBatchSize), vp(kBatchSize);
      for (int i = 0; i < kBatchSize; ++i) {
        hashes[i] = static_cast<uint64_t>(t * kBatchSize + i);
        pols[i] = make_policy(3, 0);
        vals[i] = make_value(2, 0);
        pp[i] = pols[i].data();
        vp[i] = vals[i].data();
      }
      cache.insert_many(hashes.data(), pp.data(), vp.data(), kBatchSize);
    });
  }
  for (auto& t : threads) t.join();

  EXPECT_LE(cache.size(), 1000);
}

// ---------------------------------------------------------------------------
// Memory stability
// ---------------------------------------------------------------------------

TEST(S3FIFOCache, NoGrowthAfterFill) {
  constexpr uint32_t cap = 100;
  constexpr uint32_t ghost = 50;
  S3FIFOCache cache(cap, ghost, 4, 2);
  auto pol = make_policy(4, 0);
  auto val = make_value(2, 0);
  std::vector<float> po(4), vo(2);

  // Fill the cache.
  for (uint32_t i = 0; i < cap; ++i) {
    cache.insert(i, pol.data(), val.data());
  }

  // Run thousands of insert/find/evict cycles.
  for (uint32_t i = cap; i < cap + 10000; ++i) {
    cache.insert(i, pol.data(), val.data());
    cache.find(i, po.data(), vo.data());
  }

  // Size should never exceed capacity.
  EXPECT_EQ(cache.size(), cap);
}

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------

TEST(S3FIFOCache, CapacityOne) {
  S3FIFOCache cache(1, 1, 2, 1);
  auto pol = make_policy(2, 1.0f);
  auto val = make_value(1, 0.5f);
  std::vector<float> po(2), vo(1);

  cache.insert(1, pol.data(), val.data());
  EXPECT_TRUE(cache.find(1, po.data(), vo.data()));
  EXPECT_EQ(po, pol);

  // Insert another — evicts first.
  auto pol2 = make_policy(2, 10.0f);
  auto val2 = make_value(1, 5.0f);
  cache.insert(2, pol2.data(), val2.data());
  EXPECT_FALSE(cache.find(1, po.data(), vo.data()));
  EXPECT_TRUE(cache.find(2, po.data(), vo.data()));
  EXPECT_EQ(po, pol2);
  EXPECT_EQ(cache.size(), 1);
}

TEST(S3FIFOCache, CapacityZero) {
  S3FIFOCache cache(0, 0, 2, 1);
  auto pol = make_policy(2, 1.0f);
  auto val = make_value(1, 0.5f);
  std::vector<float> po(2), vo(1);

  cache.insert(1, pol.data(), val.data());
  EXPECT_FALSE(cache.find(1, po.data(), vo.data()));
  EXPECT_EQ(cache.size(), 0);
}

TEST(S3FIFOCache, AllHits) {
  S3FIFOCache cache(5, 3, 2, 1);
  auto pol = make_policy(2, 1.0f);
  auto val = make_value(1, 0.5f);
  cache.insert(1, pol.data(), val.data());

  std::vector<float> po(2), vo(1);
  for (int i = 0; i < 100; ++i) {
    EXPECT_TRUE(cache.find(1, po.data(), vo.data()));
  }
  EXPECT_EQ(cache.hits(), 100);
  EXPECT_EQ(cache.misses(), 0);
}

TEST(S3FIFOCache, AllMisses) {
  S3FIFOCache cache(5, 3, 2, 1);
  std::vector<float> po(2), vo(1);
  for (int i = 0; i < 100; ++i) {
    EXPECT_FALSE(cache.find(static_cast<uint64_t>(i), po.data(), vo.data()));
  }
  EXPECT_EQ(cache.hits(), 0);
  EXPECT_EQ(cache.misses(), 100);
}

TEST(S3FIFOCache, EvictionFromEmptySmall) {
  // All items in Main, new insert must evict from Main directly.
  constexpr uint32_t cap = 3;
  S3FIFOCache cache(cap, cap, 2, 1);
  auto pol = make_policy(2, 0);
  auto val = make_value(1, 0);
  std::vector<float> po(2), vo(1);

  // Fill cache, access all to set freq=1.
  for (uint32_t i = 0; i < cap; ++i) {
    cache.insert(i, pol.data(), val.data());
    cache.find(i, po.data(), vo.data());
  }

  // Evict from S: all have freq=1, so all promote to M. S becomes empty.
  // Then a new insert forces an eviction. S is empty → evict from M.
  for (uint32_t i = cap; i < cap * 3; ++i) {
    cache.insert(i, pol.data(), val.data());
  }

  EXPECT_EQ(cache.size(), cap);
  EXPECT_GT(cache.evictions(), 0);
}

}  // namespace
}  // namespace alphazero
