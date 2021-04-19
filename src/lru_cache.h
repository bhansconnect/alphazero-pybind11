#pragma once

#include <absl/container/flat_hash_map.h>

#include <iterator>
#include <list>
#include <mutex>
#include <optional>

namespace alphazero {

template <typename K, typename V>
class LRUCache {
 public:
  LRUCache(size_t max_size) : ms_(max_size) {}

  std::optional<V> find(const K k) {
    std::unique_lock l{m_};
    auto it = cache_.find(k);
    if (it == cache_.end()) {
      ++misses_;
      return std::nullopt;
    }
    ++hits_;
    auto lru_iter = it->second;
    auto& [_, v] = *lru_iter;
    if (lru_iter != lru_.begin()) {
      lru_.splice(lru_.begin(), lru_, lru_iter, std::next(lru_iter));
    }
    return v;
  }

  void insert(const K& k, const V& v) {
    std::unique_lock l{m_};
    auto it = cache_.find(k);
    if (it != cache_.end()) {
      return;
    }
    lru_.emplace_front(k, v);
    cache_.emplace(k, lru_.begin());
    if (lru_.size() > ms_) {
      auto& [k, _] = lru_.back();
      cache_.erase(k);
      lru_.pop_back();
    }
  }

  [[nodiscard]] size_t hits() { return hits_; };
  [[nodiscard]] size_t misses() { return misses_; };
  [[nodiscard]] size_t size() { return lru_.size(); };

 private:
  using List = std::list<std::tuple<K, V>>;
  List lru_;
  absl::flat_hash_map<K, typename List::iterator> cache_;
  std::mutex m_;
  size_t ms_;
  size_t hits_ = 0;
  size_t misses_ = 0;
};

}  // namespace alphazero