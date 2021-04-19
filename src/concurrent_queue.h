#pragma once

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <optional>
#include <queue>
#include <thread>

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

  [[nodiscard]] size_t size() const noexcept { return queue_.size(); }

  [[nodiscard]] bool empty() const noexcept {
    std::unique_lock lock(m_);
    return queue_.empty();
  }

  [[nodiscard]] std::optional<T> try_pop() noexcept {
    std::unique_lock lock(m_);
    return try_pop_();
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

}  // namespace alphazero
