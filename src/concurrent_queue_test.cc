
#include "concurrent_queue.h"

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

}  // namespace
}  // namespace alphazero