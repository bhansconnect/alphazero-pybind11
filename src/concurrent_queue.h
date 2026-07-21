#pragma once

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <thread>
#include <vector>

namespace alphazero {

// A simple multithread safe queue.
template <typename T>
class ConcurrentQueue {
 public:
  void push(const T& data) noexcept {
    std::unique_lock lock(m_);
    queue_.push(data);
    lock.unlock();
    cv_.notify_one();
  }

  void push_many(const std::vector<T>& data) noexcept {
    std::unique_lock lock(m_);
    for (const auto& d : data) {
      queue_.push(d);
    }
    lock.unlock();
    cv_.notify_all();
  }

  [[nodiscard]] size_t size() const noexcept { return queue_.size(); }

  [[nodiscard]] bool empty() const noexcept {
    std::unique_lock lock(m_);
    return queue_.empty();
  }

  [[nodiscard]] std::optional<T> try_pop() noexcept {
    std::unique_lock lock(m_);
    return try_pop_();
  }

  // Non-blocking: grabs whatever is available right now (no wait), up to n.
  [[nodiscard]] std::vector<T> try_pop_upto(size_t n) noexcept {
    std::unique_lock lock(m_);
    n = std::min(queue_.size(), n);
    std::vector<T> out;
    out.reserve(n);
    for (auto i = 0ul; i < n; ++i) {
      out.push_back(queue_.front());
      queue_.pop();
    }
    return out;
  }

  template <class Rep, class Period>
  [[nodiscard]] std::vector<T> pop_upto(
      size_t n, const std::chrono::duration<Rep, Period>& max_wait) noexcept {
    std::unique_lock lock(m_);
    cv_.wait_for(lock, max_wait, [this] { return !queue_.empty(); });
    auto out = std::vector<T>{};
    n = std::min(queue_.size(), n);
    out.reserve(n);
    for (auto i = 0ul; i < n; ++i) {
      out.push_back(queue_.front());
      queue_.pop();
    }
    return out;
  }

  template <class Rep, class Period>
  [[nodiscard]] std::vector<T> pop_upto_filled(
      size_t n, const std::chrono::duration<Rep, Period>& max_wait) noexcept {
    std::unique_lock lock(m_);
    cv_.wait_for(lock, max_wait, [this, n] { return queue_.size() >= n; });
    auto count = std::min(queue_.size(), n);
    std::vector<T> out;
    out.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      out.push_back(queue_.front());
      queue_.pop();
    }
    return out;
  }

  template <class Rep, class Period>
  [[nodiscard]] std::optional<T> pop(
      const std::chrono::duration<Rep, Period>& max_wait) noexcept {
    std::unique_lock lock(m_);
    cv_.wait_for(lock, max_wait, [this] { return !queue_.empty(); });
    return try_pop_();
  }

 private:
  [[nodiscard]] std::optional<T> try_pop_() noexcept {
    if (queue_.empty()) {
      return std::nullopt;
    }
    auto val = queue_.front();
    queue_.pop();
    return val;
  }

  std::queue<T> queue_;
  std::mutex m_;
  std::condition_variable cv_;
};

// A sharded multithread safe queue: N independent ConcurrentQueue<T>
// instances behind a single logical interface. Splitting the single global
// mutex of ConcurrentQueue into N independent mutexes removes the
// lock/unlock+notify on every push/pop from being a hard serialization point
// once many producer/consumer threads are involved (see PlayManager's
// awaiting_mcts_/awaiting_inference_ queues, which are touched on every MCTS
// simulation step -- even cache hits).
//
// Items are routed to a shard via an explicit key supplied by the caller
// (callers use a stable, deterministic key -- e.g. the game index -- so a
// given item always round-trips through the same shard). Consumers pick a
// "home" shard (e.g. derived from a per-thread id) to check first, but a
// pop() call will scan every shard before blocking, so correctness never
// depends on the number of consumer threads matching the number of shards:
// every shard is guaranteed to be drained eventually even if no thread
// "owns" it.
template <typename T>
class ShardedQueue {
 public:
  explicit ShardedQueue(uint32_t num_shards) noexcept
      : num_shards_(std::max<uint32_t>(1, num_shards)) {
    shards_.reserve(num_shards_);
    for (uint32_t i = 0; i < num_shards_; ++i) {
      shards_.push_back(std::make_unique<ConcurrentQueue<T>>());
    }
  }

  [[nodiscard]] uint32_t num_shards() const noexcept { return num_shards_; }

  [[nodiscard]] size_t size() const noexcept {
    size_t out = 0;
    for (const auto& q : shards_) out += q->size();
    return out;
  }

  [[nodiscard]] bool empty() const noexcept {
    for (const auto& q : shards_) {
      if (!q->empty()) return false;
    }
    return true;
  }

  void push(uint64_t key, const T& data) noexcept {
    shards_[key % num_shards_]->push(data);
  }

  // Pushes many items, each routed by `key_of(item)`.
  template <class KeyFn>
  void push_many(const std::vector<T>& data, KeyFn key_of) noexcept {
    if (num_shards_ == 1) {
      shards_[0]->push_many(data);
      return;
    }
    std::vector<std::vector<T>> grouped(num_shards_);
    for (const auto& d : data) {
      grouped[key_of(d) % num_shards_].push_back(d);
    }
    for (uint32_t s = 0; s < num_shards_; ++s) {
      if (!grouped[s].empty()) shards_[s]->push_many(grouped[s]);
    }
  }

  // `home` is tried FIRST, with the full blocking wait -- so when home has
  // work (the common case whenever num_shards doesn't badly outnumber the
  // consumer threads) this costs exactly one shard's lock, identical to a
  // plain ConcurrentQueue::pop. Only if home comes up completely empty for
  // the whole max_wait do we pay for a cheap non-blocking sweep of the
  // other shards. This keeps num_shards safe to set high (e.g. cpu_count)
  // even when far fewer threads end up consuming -- correctness never
  // depends on num_shards matching the consumer count -- while not paying
  // an O(num_shards) cost on every single call in that mismatched case.
  template <class Rep, class Period>
  [[nodiscard]] std::optional<T> pop(
      uint32_t home, const std::chrono::duration<Rep, Period>& max_wait) noexcept {
    auto v = shards_[home % num_shards_]->pop(max_wait);
    if (v.has_value()) return v;
    for (uint32_t off = 1; off < num_shards_; ++off) {
      auto v2 = shards_[(home + off) % num_shards_]->try_pop();
      if (v2.has_value()) return v2;
    }
    return std::nullopt;
  }

  // Same home-first strategy as pop() above, adapted for "gather up to n".
  template <class Rep, class Period>
  [[nodiscard]] std::vector<T> pop_upto(
      uint32_t home, size_t n,
      const std::chrono::duration<Rep, Period>& max_wait) noexcept {
    if (n == 0) return {};
    auto out = shards_[home % num_shards_]->pop_upto(n, max_wait);
    if (out.size() < n) {
      for (uint32_t off = 1; off < num_shards_ && out.size() < n; ++off) {
        auto s = (home + off) % num_shards_;
        auto part = shards_[s]->try_pop_upto(n - out.size());
        out.insert(out.end(), part.begin(), part.end());
      }
    }
    return out;
  }

 private:
  uint32_t num_shards_;
  std::vector<std::unique_ptr<ConcurrentQueue<T>>> shards_;
};

}  // namespace alphazero
