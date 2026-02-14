#pragma once

#include <deque>
#include <iterator>
#include <list>
#include <mutex>
#include <optional>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/hash/hash.h"

namespace alphazero {

template <typename K, typename V>
class LRUCache {
 public:
  LRUCache(size_t max_size, size_t ghost_size = 0)
      : ms_(max_size), gs_(ghost_size) {}

  std::optional<V> find(const K k) {
    std::unique_lock l{m_};
    auto it = cache_.find(k);
    if (it == cache_.end()) {
      ++misses_;
      if (gs_ > 0) {
        auto h = absl::HashOf(k);
        if (ghost_set_.erase(h)) {
          ++reinserts_;
        }
      }
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

  template <typename V1, typename V2>
  bool find_into(const K& k, V1& out1, V2& out2) {
    std::unique_lock l{m_};
    auto it = cache_.find(k);
    if (it == cache_.end()) {
      ++misses_;
      if (gs_ > 0) {
        auto h = absl::HashOf(k);
        if (ghost_set_.erase(h)) {
          ++reinserts_;
        }
      }
      return false;
    }
    ++hits_;
    auto lru_iter = it->second;
    auto& [_, val] = *lru_iter;
    auto& [v1, v2] = val;
    out1 = v1;
    out2 = v2;
    if (lru_iter != lru_.begin()) {
      lru_.splice(lru_.begin(), lru_, lru_iter, std::next(lru_iter));
    }
    return true;
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
      auto& [ek, _] = lru_.back();
      if (gs_ > 0) {
        ghost_add(absl::HashOf(ek));
      }
      cache_.erase(ek);
      lru_.pop_back();
      ++evictions_;
    }
  }

  void insert_many(const std::vector<K>& ks, const std::vector<V>& vs) {
    std::unique_lock l{m_};
    for (size_t i = 0; i < ks.size(); ++i) {
      const auto k = ks[i];
      const auto it = cache_.find(k);
      if (it != cache_.end()) {
        continue;
      }
      const auto v = vs[i];
      lru_.emplace_front(k, v);
      cache_.emplace(k, lru_.begin());
    }
    while (lru_.size() > ms_) {
      auto& [ek, _] = lru_.back();
      if (gs_ > 0) {
        ghost_add(absl::HashOf(ek));
      }
      cache_.erase(ek);
      lru_.pop_back();
      ++evictions_;
    }
  }

  [[nodiscard]] size_t hits() const { return hits_; };
  [[nodiscard]] size_t misses() const { return misses_; };
  [[nodiscard]] size_t size() const { return lru_.size(); };
  [[nodiscard]] size_t evictions() const { return evictions_; };
  [[nodiscard]] size_t reinserts() const { return reinserts_; };
  [[nodiscard]] size_t max_size() const { return ms_; };

 private:
  void ghost_add(size_t h) {
    ghost_queue_.push_back(h);
    ghost_set_.insert(h);
    while (ghost_queue_.size() > gs_) {
      ghost_set_.erase(ghost_queue_.front());
      ghost_queue_.pop_front();
    }
  }

  using List = std::list<std::tuple<K, V>>;
  List lru_;
  absl::flat_hash_map<K, typename List::iterator> cache_;
  std::mutex m_;
  size_t ms_;
  size_t gs_;
  size_t hits_ = 0;
  size_t misses_ = 0;
  size_t evictions_ = 0;
  size_t reinserts_ = 0;
  absl::flat_hash_set<size_t> ghost_set_;
  std::deque<size_t> ghost_queue_;
};

template <typename K, typename V>
class ShardedLRUCache {
 public:
  ShardedLRUCache(size_t max_size, size_t shards = 1, size_t ghost_size = 0) {
    for (size_t i = 0; i < shards; ++i) {
      caches_.push_back(std::make_unique<LRUCache<K, V>>(
          max_size / shards, ghost_size / shards));
    }
  }

  std::optional<V> find(const K k) { return get_cache(k)->find(k); }

  template <typename V1, typename V2>
  bool find_into(const K& k, V1& out1, V2& out2) {
    return get_cache(k)->find_into(k, out1, out2);
  }

  void insert(const K& k, const V& v) { get_cache(k)->insert(k, v); }

  void insert_many(const std::vector<K>& ks, const std::vector<V>& vs) {
    if (caches_.size() == 1) {
      caches_[0]->insert_many(ks, vs);
      return;
    }
    // Group by shard
    std::vector<std::vector<size_t>> shard_indices(caches_.size());
    for (size_t i = 0; i < ks.size(); ++i) {
      shard_indices[absl::HashOf(ks[i]) % caches_.size()].push_back(i);
    }
    // Batch insert per shard
    for (size_t s = 0; s < caches_.size(); ++s) {
      if (shard_indices[s].empty()) continue;
      std::vector<K> shard_ks;
      std::vector<V> shard_vs;
      shard_ks.reserve(shard_indices[s].size());
      shard_vs.reserve(shard_indices[s].size());
      for (auto idx : shard_indices[s]) {
        shard_ks.push_back(ks[idx]);
        shard_vs.push_back(vs[idx]);
      }
      caches_[s]->insert_many(shard_ks, shard_vs);
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
  [[nodiscard]] size_t evictions() const {
    size_t evictions = 0;
    for (const auto& cache : caches_) {
      evictions += cache->evictions();
    }
    return evictions;
  };
  [[nodiscard]] size_t reinserts() const {
    size_t reinserts = 0;
    for (const auto& cache : caches_) {
      reinserts += cache->reinserts();
    }
    return reinserts;
  };
  [[nodiscard]] size_t max_size() const {
    size_t max_size = 0;
    for (const auto& cache : caches_) {
      max_size += cache->max_size();
    }
    return max_size;
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
