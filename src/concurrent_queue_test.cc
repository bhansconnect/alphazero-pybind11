
#include "concurrent_queue.h"

#include <algorithm>
#include <atomic>
#include <future>
#include <thread>

#include "gtest/gtest.h"

namespace alphazero {
namespace {

using namespace std::chrono_literals;

// NOLINTNEXTLINE
TEST(ConcurrentQueue, SingleThreaded) {
  auto queue = ConcurrentQueue<int>{};
  for (auto i = 0; i < 5; ++i) {
    queue.push(i);
  }
  for (auto i = 0; i < 5; ++i) {
    EXPECT_EQ(queue.try_pop(), i);
  }
  EXPECT_EQ(queue.try_pop(), std::nullopt);
  for (auto i = 0; i < 5; ++i) {
    queue.push(i);
  }
  for (auto i = 0; i < 5; ++i) {
    EXPECT_EQ(queue.pop(100us), i);
  }
  EXPECT_EQ(queue.pop(100us), std::nullopt);
}

// NOLINTNEXTLINE
TEST(ConcurrentQueue, MultiThreaded) {
  auto max = 100;
  auto count = std::atomic<int>{};
  auto in_queue = ConcurrentQueue<int>{};
  auto out_queue = ConcurrentQueue<int>{};
  auto counter = std::async(std::launch::async, [&] {
    while (count < max) {
      auto data = in_queue.pop(100us);
      if (!data.has_value()) {
        continue;
      }
      out_queue.push(data.value());
      count += data.value();
    }
  });
  auto looper = std::async(std::launch::async, [&] {
    while (count < max) {
      auto data = out_queue.pop(100us);
      if (!data.has_value()) {
        continue;
      }
      in_queue.push(data.value());
    }
  });
  for (auto i = 0; i < 5; ++i) {
    in_queue.push(1);
  }
  counter.wait();
  looper.wait();
  EXPECT_EQ(count, max);
}

// NOLINTNEXTLINE
TEST(ConcurrentQueue, PopUptoFilledImmediate) {
  auto queue = ConcurrentQueue<int>{};
  for (auto i = 0; i < 5; ++i) {
    queue.push(i);
  }
  auto result = queue.pop_upto_filled(5, 100ms);
  EXPECT_EQ(result.size(), 5);
  for (auto i = 0; i < 5; ++i) {
    EXPECT_EQ(result[i], i);
  }
  EXPECT_EQ(queue.try_pop(), std::nullopt);
}

// NOLINTNEXTLINE
TEST(ConcurrentQueue, PopUptoFilledPartialTimeout) {
  auto queue = ConcurrentQueue<int>{};
  for (auto i = 0; i < 3; ++i) {
    queue.push(i);
  }
  auto start = std::chrono::steady_clock::now();
  auto result = queue.pop_upto_filled(10, 50ms);
  auto elapsed = std::chrono::steady_clock::now() - start;
  EXPECT_EQ(result.size(), 3);
  for (auto i = 0; i < 3; ++i) {
    EXPECT_EQ(result[i], i);
  }
  // Should have waited at least ~50ms before returning partial
  EXPECT_GE(elapsed, 40ms);
  EXPECT_EQ(queue.try_pop(), std::nullopt);
}

// NOLINTNEXTLINE
TEST(ConcurrentQueue, PopUptoFilledThreaded) {
  auto queue = ConcurrentQueue<int>{};
  constexpr auto N = 20;
  auto producer = std::async(std::launch::async, [&] {
    for (auto i = 0; i < N; ++i) {
      queue.push(i);
      std::this_thread::sleep_for(500us);
    }
  });
  // Wait for all N items to accumulate
  auto result = queue.pop_upto_filled(N, 500ms);
  producer.wait();
  // Should get all N (or close) since max_wait is generous
  EXPECT_EQ(result.size(), N);
  for (auto i = 0; i < static_cast<int>(result.size()); ++i) {
    EXPECT_EQ(result[i], i);
  }
}

// NOLINTNEXTLINE
TEST(ConcurrentQueue, PollProperlyReleases) {
  auto done = std::atomic<bool>{};
  auto empty_queue = ConcurrentQueue<int>{};
  auto looper = std::async(std::launch::async, [&] {
    while (!done) {
      auto data = empty_queue.pop(100us);
      if (!data.has_value()) {
        continue;
      }
      // This would proces data if it ever got here. That should never happen.
      EXPECT_TRUE(false);
    }
  });
  std::this_thread::sleep_for(100ms);
  done = true;
  looper.wait();
  EXPECT_TRUE(true);
}

// NOLINTNEXTLINE
TEST(ShardedQueue, SingleShardBehavesLikePlainQueue) {
  auto queue = ShardedQueue<int>{1};
  EXPECT_EQ(queue.num_shards(), 1U);
  for (auto i = 0; i < 5; ++i) {
    queue.push(i, i);
  }
  EXPECT_EQ(queue.size(), 5U);
  for (auto i = 0; i < 5; ++i) {
    EXPECT_EQ(queue.pop(0, 100us), i);
  }
  EXPECT_EQ(queue.pop(0, 100us), std::nullopt);
}

// NOLINTNEXTLINE
TEST(ShardedQueue, RoutesByKeyModuloPrefersHomeShard) {
  auto queue = ShardedQueue<int>{4};
  for (auto i = 0; i < 16; ++i) {
    queue.push(i, i);
  }
  EXPECT_EQ(queue.size(), 16U);
  // Everything pushed with key i lands in shard i % 4. pop_upto(s, ...)
  // checks shard s first, so asking for exactly as many items as live
  // there is satisfied without ever touching another shard -- confirming
  // push() routed by key % num_shards, in FIFO order.
  for (uint32_t s = 0; s < 4; ++s) {
    auto items = queue.pop_upto(s, 4, 10ms);
    ASSERT_EQ(items.size(), 4U);
    for (size_t k = 0; k < items.size(); ++k) {
      EXPECT_EQ(items[k], static_cast<int>(s + 4 * k));
    }
  }
}

// NOLINTNEXTLINE
TEST(ShardedQueue, PopScansEveryShardBeforeBlocking) {
  // A pop() call homed on an empty shard must still find an item that only
  // exists in a different shard -- this is the liveness guarantee that lets
  // num_shards and consumer-thread-count differ safely without starving a
  // shard nobody "owns".
  auto queue = ShardedQueue<int>{4};
  queue.push(/*key=*/3, 42);  // lands in shard 3 only
  auto result = queue.pop(/*home=*/0, 100us);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), 42);
}

// NOLINTNEXTLINE
TEST(ShardedQueue, PopUptoScansEveryShardBeforeBlocking) {
  // Regression test for a real bug: a single legacy consumer that only ever
  // calls pop_upto(0, ...) (e.g. code written before eval-queue sharding
  // existed) MUST still be able to drain items that landed in other
  // shards, or those items -- and the games waiting on them -- would be
  // stuck forever. Before the fix, pop_upto(shard, ...) only looked at
  // `shard` and this scenario hung indefinitely.
  auto queue = ShardedQueue<int>{4};
  queue.push(/*key=*/3, 42);  // shard 3 only
  queue.push(/*key=*/1, 7);   // shard 1 only
  auto items = queue.pop_upto(/*home=*/0, 10, 10ms);
  ASSERT_EQ(items.size(), 2U);
  EXPECT_NE(std::find(items.begin(), items.end(), 42), items.end());
  EXPECT_NE(std::find(items.begin(), items.end(), 7), items.end());
}

// NOLINTNEXTLINE
TEST(ShardedQueue, PushManyGroupsByKey) {
  auto queue = ShardedQueue<int>{3};
  auto items = std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7, 8};
  queue.push_many(items, [](int v) { return static_cast<uint64_t>(v); });
  EXPECT_EQ(queue.size(), items.size());
  // 9 items over 3 shards, keys 0..8 -> shard s holds exactly {s, s+3, s+6}.
  // Requesting exactly that many keeps pop_upto from spilling into another
  // shard, so this also confirms push_many's per-item routing.
  for (uint32_t s = 0; s < 3; ++s) {
    auto got = queue.pop_upto(s, 3, 10ms);
    ASSERT_EQ(got.size(), 3U);
    for (auto v : got) {
      EXPECT_EQ(static_cast<uint32_t>(v) % 3, s);
    }
  }
}

// NOLINTNEXTLINE
TEST(ShardedQueue, MultiThreadedNoLostOrDuplicatedItems) {
  // Many producer/consumer threads, more shards than some runs and fewer in
  // others, verifying every item is delivered exactly once regardless of
  // how num_shards relates to the thread count.
  constexpr auto kItems = 2000;
  constexpr auto kShards = 5U;
  constexpr auto kConsumers = 8U;  // doesn't divide evenly into kShards
  auto queue = ShardedQueue<int>{kShards};
  for (auto i = 0; i < kItems; ++i) {
    queue.push(static_cast<uint64_t>(i), i);
  }
  auto seen = std::vector<std::atomic<int>>(kItems);
  for (auto& s : seen) s = 0;
  auto remaining = std::atomic<int>{kItems};
  auto consumers = std::vector<std::future<void>>{};
  for (uint32_t c = 0; c < kConsumers; ++c) {
    consumers.push_back(std::async(std::launch::async, [&, c] {
      while (remaining.load() > 0) {
        auto v = queue.pop(c % kShards, 1ms);
        if (!v.has_value()) continue;
        seen[v.value()].fetch_add(1);
        remaining.fetch_sub(1);
      }
    }));
  }
  for (auto& c : consumers) c.wait();
  for (auto i = 0; i < kItems; ++i) {
    EXPECT_EQ(seen[i].load(), 1) << "item " << i;
  }
}

}  // namespace
}  // namespace alphazero