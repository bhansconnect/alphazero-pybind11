#pragma once

#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"

namespace alphazero {

class S3FIFOCache {
 public:
  S3FIFOCache(uint32_t max_size, uint32_t ghost_size, uint32_t num_policy,
              uint32_t num_value)
      : max_size_(max_size),
        ghost_size_(ghost_size),
        num_policy_(num_policy),
        num_value_(num_value),
        policy_data_(static_cast<size_t>(max_size) * num_policy),
        value_data_(static_cast<size_t>(max_size) * num_value),
        hashes_(max_size),
        freq_(max_size, 0),
        s_ring_(max_size),
        m_ring_(max_size),
        ghost_ring_(ghost_size) {
    map_.reserve(max_size);
    ghost_set_.reserve(ghost_size);
  }

  bool find(uint64_t hash, float* policy_out, float* value_out) {
    std::unique_lock l{m_};
    auto it = map_.find(hash);
    if (it == map_.end()) {
      misses_.fetch_add(1, std::memory_order_relaxed);
      if (ghost_size_ > 0 && ghost_set_.contains(hash)) {
        reinserts_.fetch_add(1, std::memory_order_relaxed);
      }
      return false;
    }
    hits_.fetch_add(1, std::memory_order_relaxed);
    uint32_t slot = it->second;
    freq_[slot] = 1;
    std::memcpy(policy_out, &policy_data_[static_cast<size_t>(slot) * num_policy_],
                num_policy_ * sizeof(float));
    std::memcpy(value_out, &value_data_[static_cast<size_t>(slot) * num_value_],
                num_value_ * sizeof(float));
    return true;
  }

  void insert(uint64_t hash, const float* policy, const float* value) {
    std::unique_lock l{m_};
    insert_locked(hash, policy, value);
  }

  void insert_many(const uint64_t* hashes, const float* const* policies,
                   const float* const* values, uint32_t count) {
    std::unique_lock l{m_};
    for (uint32_t i = 0; i < count; ++i) {
      insert_locked(hashes[i], policies[i], values[i]);
    }
  }

  [[nodiscard]] size_t hits() const { return hits_.load(std::memory_order_relaxed); }
  [[nodiscard]] size_t misses() const { return misses_.load(std::memory_order_relaxed); }
  [[nodiscard]] size_t evictions() const { return evictions_.load(std::memory_order_relaxed); }
  [[nodiscard]] size_t reinserts() const { return reinserts_.load(std::memory_order_relaxed); }
  [[nodiscard]] size_t size() const { return map_.size(); }
  [[nodiscard]] size_t max_size() const { return max_size_; }

 private:
  void insert_locked(uint64_t hash, const float* policy, const float* value) {
    if (max_size_ == 0) return;
    if (map_.contains(hash)) return;

    bool is_ghost_hit = false;
    if (ghost_size_ > 0) {
      is_ghost_hit = ghost_set_.erase(hash) > 0;
      // Also remove from ghost ring by marking it invalid (will be cleaned on
      // overwrite). The ring entry stays but the set no longer contains it.
    }

    uint32_t slot = alloc_slot();

    // Write data into slot.
    hashes_[slot] = hash;
    freq_[slot] = 0;
    std::memcpy(&policy_data_[static_cast<size_t>(slot) * num_policy_], policy,
                num_policy_ * sizeof(float));
    std::memcpy(&value_data_[static_cast<size_t>(slot) * num_value_], value,
                num_value_ * sizeof(float));
    map_[hash] = slot;

    if (is_ghost_hit) {
      // Admitted to Main queue.
      m_enqueue(slot);
    } else {
      // Admitted to Small queue.
      s_enqueue(slot);
    }
  }

  uint32_t alloc_slot() {
    if (next_free_ < max_size_) {
      return next_free_++;
    }
    return evict_one();
  }

  uint32_t evict_one() {
    // Try to evict from Small first.
    while (s_size_ > 0) {
      uint32_t slot = s_dequeue();
      if (freq_[slot]) {
        freq_[slot] = 0;
        m_enqueue(slot);  // Promote to Main.
        continue;
      }
      // Evict this slot.
      if (ghost_size_ > 0) {
        ghost_add(hashes_[slot]);
      }
      map_.erase(hashes_[slot]);
      evictions_.fetch_add(1, std::memory_order_relaxed);
      return slot;
    }
    // Small is empty, evict from Main.
    while (true) {
      uint32_t slot = m_dequeue();
      if (freq_[slot]) {
        freq_[slot] = 0;
        m_enqueue(slot);  // Second chance.
        continue;
      }
      map_.erase(hashes_[slot]);
      evictions_.fetch_add(1, std::memory_order_relaxed);
      return slot;
    }
  }

  // Small FIFO ring buffer operations.
  void s_enqueue(uint32_t slot) {
    uint32_t tail = (s_head_ + s_size_) % max_size_;
    s_ring_[tail] = slot;
    ++s_size_;
  }
  uint32_t s_dequeue() {
    uint32_t slot = s_ring_[s_head_];
    s_head_ = (s_head_ + 1) % max_size_;
    --s_size_;
    return slot;
  }

  // Main FIFO ring buffer operations.
  void m_enqueue(uint32_t slot) {
    uint32_t tail = (m_head_ + m_size_) % max_size_;
    m_ring_[tail] = slot;
    ++m_size_;
  }
  uint32_t m_dequeue() {
    uint32_t slot = m_ring_[m_head_];
    m_head_ = (m_head_ + 1) % max_size_;
    --m_size_;
    return slot;
  }

  // Ghost ring buffer.
  void ghost_add(uint64_t hash) {
    if (ghost_size_ == 0) return;
    if (ghost_count_ >= ghost_size_) {
      // Evict oldest ghost entry.
      ghost_set_.erase(ghost_ring_[ghost_head_]);
      ghost_ring_[ghost_head_] = hash;
      ghost_head_ = (ghost_head_ + 1) % ghost_size_;
    } else {
      uint32_t pos = (ghost_head_ + ghost_count_) % ghost_size_;
      ghost_ring_[pos] = hash;
      ++ghost_count_;
    }
    ghost_set_.insert(hash);
  }

  uint32_t max_size_;
  uint32_t ghost_size_;
  uint32_t num_policy_;
  uint32_t num_value_;
  // Flat storage arrays (indexed by slot).
  std::vector<float> policy_data_;
  std::vector<float> value_data_;
  std::vector<uint64_t> hashes_;
  std::vector<uint8_t> freq_;

  // Ring buffers for S and M queues (hold slot indices).
  std::vector<uint32_t> s_ring_;
  std::vector<uint32_t> m_ring_;
  uint32_t s_head_ = 0, s_size_ = 0;
  uint32_t m_head_ = 0, m_size_ = 0;

  // Slot allocation.
  uint32_t next_free_ = 0;

  // Ghost (hash-only ring buffer).
  std::vector<uint64_t> ghost_ring_;
  absl::flat_hash_set<uint64_t> ghost_set_;
  uint32_t ghost_head_ = 0, ghost_count_ = 0;

  // Lookup.
  absl::flat_hash_map<uint64_t, uint32_t> map_;

  // Stats.
  std::atomic<size_t> hits_ = 0;
  std::atomic<size_t> misses_ = 0;
  std::atomic<size_t> evictions_ = 0;
  std::atomic<size_t> reinserts_ = 0;

  std::mutex m_;
};

class ShardedS3FIFOCache {
 public:
  ShardedS3FIFOCache(uint32_t max_size, uint32_t shards, uint32_t ghost_size,
                     uint32_t num_policy, uint32_t num_value)
      : num_shards_(shards) {
    caches_.reserve(shards);
    auto per_shard = max_size / shards;
    auto ghost_per_shard = ghost_size / shards;
    for (uint32_t i = 0; i < shards; ++i) {
      caches_.push_back(
          std::make_unique<S3FIFOCache>(per_shard, ghost_per_shard, num_policy, num_value));
    }
  }

  bool find(uint64_t hash, float* policy_out, float* value_out) {
    return get_shard(hash).find(hash, policy_out, value_out);
  }

  void insert(uint64_t hash, const float* policy, const float* value) {
    get_shard(hash).insert(hash, policy, value);
  }

  void insert_many(const uint64_t* hashes, const float* const* policies,
                   const float* const* values, uint32_t count) {
    if (num_shards_ == 1) {
      caches_[0]->insert_many(hashes, policies, values, count);
      return;
    }
    // Group by shard.
    std::vector<std::vector<uint32_t>> shard_indices(num_shards_);
    for (uint32_t i = 0; i < count; ++i) {
      shard_indices[hashes[i] % num_shards_].push_back(i);
    }
    for (uint32_t s = 0; s < num_shards_; ++s) {
      if (shard_indices[s].empty()) continue;
      std::vector<uint64_t> shard_hashes;
      std::vector<const float*> shard_policies;
      std::vector<const float*> shard_values;
      shard_hashes.reserve(shard_indices[s].size());
      shard_policies.reserve(shard_indices[s].size());
      shard_values.reserve(shard_indices[s].size());
      for (auto idx : shard_indices[s]) {
        shard_hashes.push_back(hashes[idx]);
        shard_policies.push_back(policies[idx]);
        shard_values.push_back(values[idx]);
      }
      caches_[s]->insert_many(shard_hashes.data(), shard_policies.data(),
                              shard_values.data(), shard_hashes.size());
    }
  }

  [[nodiscard]] size_t hits() const {
    size_t out = 0;
    for (const auto& c : caches_) out += c->hits();
    return out;
  }
  [[nodiscard]] size_t misses() const {
    size_t out = 0;
    for (const auto& c : caches_) out += c->misses();
    return out;
  }
  [[nodiscard]] size_t evictions() const {
    size_t out = 0;
    for (const auto& c : caches_) out += c->evictions();
    return out;
  }
  [[nodiscard]] size_t reinserts() const {
    size_t out = 0;
    for (const auto& c : caches_) out += c->reinserts();
    return out;
  }
  [[nodiscard]] size_t size() const {
    size_t out = 0;
    for (const auto& c : caches_) out += c->size();
    return out;
  }
  [[nodiscard]] size_t max_size() const {
    size_t out = 0;
    for (const auto& c : caches_) out += c->max_size();
    return out;
  }

 private:
  S3FIFOCache& get_shard(uint64_t hash) {
    return *caches_[hash % num_shards_];
  }

  uint32_t num_shards_;
  std::vector<std::unique_ptr<S3FIFOCache>> caches_;
};

}  // namespace alphazero
