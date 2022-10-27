#pragma once

#include <iterator>
#include <list>
#include <mutex>
#include <optional>

#include "absl/container/flat_hash_map.h"
#include "absl/hash/hash.h"

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

  void insert_many(const std::vector<K>& ks, const std::vector<V>& vs) {
    std::unique_lock l{m_};
    for (size_t i = 0; i < ks.size(); ++i) {
      const auto k = ks[i];
      const auto it = cache_.find(k);
      if (it != cache_.end()) {
        return;
      }
      const auto v = vs[i];
      lru_.emplace_front(k, v);
      cache_.emplace(k, lru_.begin());
    }
    while (lru_.size() > ms_) {
      auto& [k, _] = lru_.back();
      cache_.erase(k);
      lru_.pop_back();
    }
  }

  [[nodiscard]] size_t hits() const { return hits_; };
  [[nodiscard]] size_t misses() const { return misses_; };
  [[nodiscard]] size_t size() const { return lru_.size(); };

 private:
  using List = std::list<std::tuple<K, V>>;
  List lru_;
  absl::flat_hash_map<K, typename List::iterator> cache_;
  std::mutex m_;
  size_t ms_;
  size_t hits_ = 0;
  size_t misses_ = 0;
};

template <typename K, typename V>
class ShardedLRUCache {
 public:
  ShardedLRUCache(size_t max_size, size_t shards = 1) {
    for (size_t i = 0; i < shards; ++i) {
      caches_.push_back(std::make_unique<LRUCache<K, V>>(max_size / shards));
    }
  }

  std::optional<V> find(const K k) { return get_cache(k)->find(k); }

  void insert(const K& k, const V& v) { get_cache(k)->insert(k, v); }

  void insert_many(const std::vector<K>& ks, const std::vector<V>& vs) {
    for (size_t i = 0; i < ks.size(); ++i) {
      insert(ks[i], vs[i]);
    }
  }

  [[nodiscard]] size_t hits() const {
    size_t hits = 0;
    for (const auto& cache : caches_) {
      hits += cache->hits();
    }
    return hits;
  };
  [[nodiscard]] size_t misses() const {
    size_t misses = 0;
    for (const auto& cache : caches_) {
      misses += cache->misses();
    }
    return misses;
  };
  [[nodiscard]] size_t size() const {
    size_t size = 0;
    for (const auto& cache : caches_) {
      size += cache->size();
    }
    return size;
  };

 private:
  LRUCache<K, V>* get_cache(const K& k) {
    if (caches_.size() == 1) {
      return caches_[0].get();
    }
    return caches_[absl::HashOf(k) % caches_.size()].get();
  }

  std::vector<std::unique_ptr<LRUCache<K, V>>> caches_;
};

}  // namespace alphazero