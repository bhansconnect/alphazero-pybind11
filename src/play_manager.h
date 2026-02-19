#pragma once

#include <atomic>
#include <cstdint>
#include <deque>
#include <limits>
#include <mutex>
#include <optional>
#include <vector>

#include "absl/hash/hash.h"
#include "concurrent_queue.h"
#include "dll_export.h"
#include "game_state.h"
#include "s3fifo_cache.h"
#include "mcts.h"

namespace alphazero {

enum class EvalType : uint8_t { NN = 0, RANDOM = 1, PLAYOUT = 2 };

using Cache = ShardedS3FIFOCache;

using namespace std::chrono_literals;

constexpr const auto MAX_WAIT = 10ms;

struct PendingHistory {
  PlayHistory ph;
  uint8_t player;
};

struct GameData {
  std::unique_ptr<GameState> gs;
  uint64_t leaf_hash = 0;
  std::vector<MCTS> mcts;
  Tensor<float, 3> canonical;
  Vector<float> v;
  Vector<float> pi;
  std::vector<PendingHistory> partial_history;
  std::vector<uint8_t> seat_perm;  // per-game: physical_player → model_group
  uint8_t perm_index = 0;
  bool initialized = false;
  bool capped = false;
  bool playthrough = false;
  double total_avg_leaf_depth = 0;       // full searches only
  double total_search_entropy = 0;       // full searches only
  double fast_total_avg_leaf_depth = 0;  // fast searches only
  double fast_total_search_entropy = 0;  // fast searches only
  double total_valid_moves = 0;          // all moves
  uint32_t move_count = 0;              // all moves
  uint32_t full_move_count = 0;         // full searches only
  uint32_t fast_move_count = 0;         // fast searches only
};

struct PlayParams {
  uint32_t games_to_play;
  uint32_t concurrent_games;
  uint32_t max_batch_size = 1;
  uint32_t max_cache_size = 0;
  uint8_t cache_shards = 1;
  std::vector<uint32_t> mcts_visits{};
  float cpuct = 2.0;
  float start_temp = 1.0;
  float final_temp = 1.0;
  float temp_decay_half_life = 0;
  bool history_enabled = false;
  bool self_play = false;
  bool tree_reuse = true;
  bool add_noise = false;
  float epsilon = 0.25;
  float mcts_root_temp = 1.0;  // test-time default; self-play sets via config
  bool playout_cap_randomization = false;
  uint32_t playout_cap_depth = 25;
  float playout_cap_percent = 0.75;
  float fpu_reduction = 0.0;
  bool root_fpu_zero = false;
  bool shaped_dirichlet = false;
  bool policy_target_pruning = false;
  float resign_percent = 0.0;
  float resign_playthrough_percent = 0.0;
  std::vector<EvalType> eval_type{};  // per player, empty = all NN
  std::vector<uint8_t> model_groups{};  // player → model_group_index (empty = identity)
  std::vector<std::vector<uint8_t>> seat_perms{};  // list of permutations (empty = no rotation)
  std::vector<std::vector<uint32_t>> seat_visits{};  // per-perm, per-seat visit overrides
};

// This is a multithread safe game play manager.
// It enables running of MCTS and preparing games for GPU inference.

class DLLEXPORT PlayManager {
 public:
  PlayManager(std::unique_ptr<GameState> gs, PlayParams p);
  PlayManager(std::unique_ptr<GameState> gs, PlayParams p,
              std::vector<std::shared_ptr<Cache>> external_caches);

  // play will keep playing games until all games are completed.
  void play();

  void update_inferences(uint8_t group,
                         const std::vector<uint32_t>& game_indices,
                         const Eigen::Ref<const Matrix<float>>& v,
                         const Eigen::Ref<const Matrix<float>>& pi);

  [[nodiscard]] const Vector<float> scores() const noexcept { return scores_; }
  [[nodiscard]] const Vector<float> resign_scores() const noexcept {
    return resign_scores_;
  }
  void stop() noexcept { stopped_.store(true, std::memory_order_relaxed); }
  bool stopped() const noexcept { return stopped_.load(std::memory_order_relaxed); }
  [[nodiscard]] uint32_t remaining_games() const noexcept {
    if (stopped_.load(std::memory_order_relaxed)) return 0;
    return params_.games_to_play - games_completed_;
  }
  [[nodiscard]] uint32_t games_completed() const noexcept {
    return games_completed_;
  }
  [[nodiscard]] std::optional<uint32_t> pop_game(uint32_t player) noexcept {
    return awaiting_inference_[player]->pop(MAX_WAIT);
  }
  [[nodiscard]] std::vector<uint32_t> pop_games_upto(uint32_t player,
                                                     size_t n) noexcept {
    return awaiting_inference_[player]->pop_upto(n, MAX_WAIT);
  }
  [[nodiscard]] std::vector<uint32_t> pop_games_filled(uint32_t player,
                                                        size_t n) noexcept {
    return awaiting_inference_[player]->pop_upto_filled(n, MAX_WAIT);
  }
  template <class Rep, class Period>
  [[nodiscard]] std::vector<uint32_t> pop_games_upto_timed(
      uint32_t group, size_t n,
      const std::chrono::duration<Rep, Period>& timeout) noexcept {
    return awaiting_inference_[group]->pop_upto(n, timeout);
  }

  // Model group accessors
  uint8_t num_model_groups() const noexcept { return num_model_groups_; }
  size_t num_seat_perms() const noexcept { return seat_perms_.size(); }
  const Vector<float>& perm_scores(size_t idx) const noexcept {
    return perm_scores_[idx].scores;
  }
  uint32_t perm_games_completed(size_t idx) const noexcept {
    return perm_scores_[idx].games_completed;
  }

  // GPU steal: set by GPU thread to signal batcher to hand off partial batches
  void set_eager(bool e) noexcept { eager_.store(e, std::memory_order_relaxed); }
  bool is_eager() const noexcept { return eager_.load(std::memory_order_relaxed); }
  [[nodiscard]] std::optional<PlayHistory> pop_hist() noexcept {
    return history_.pop(MAX_WAIT);
  }
  [[nodiscard]] std::vector<PlayHistory> pop_hist_upto(size_t n) noexcept {
    return history_.pop_upto(n, MAX_WAIT);
  }
  void push_inference(const uint32_t i) noexcept { awaiting_mcts_.push(i); }
  [[nodiscard]] GameData& game_data(uint32_t i) noexcept { return games_[i]; }
  [[nodiscard]] const PlayParams& params() const noexcept { return params_; }
  uint64_t avg_game_length() const noexcept {
    return static_cast<float>(game_length_) /
           static_cast<float>(games_completed_);
  }
  float avg_leaf_depth() const noexcept {
    if (full_move_count_ == 0) return 0;
    return static_cast<float>(total_avg_leaf_depth_ / static_cast<double>(full_move_count_));
  }
  float avg_search_entropy() const noexcept {
    if (full_move_count_ == 0) return 0;
    return static_cast<float>(total_search_entropy_ / static_cast<double>(full_move_count_));
  }
  float fast_avg_leaf_depth() const noexcept {
    if (fast_move_count_ == 0) return 0;
    return static_cast<float>(fast_total_avg_leaf_depth_ / static_cast<double>(fast_move_count_));
  }
  float fast_avg_search_entropy() const noexcept {
    if (fast_move_count_ == 0) return 0;
    return static_cast<float>(fast_total_search_entropy_ / static_cast<double>(fast_move_count_));
  }
  float avg_moves_per_turn() const noexcept {
    if (game_length_ == 0) return 0;
    return static_cast<float>(total_move_count_) / static_cast<float>(game_length_);
  }
  float avg_valid_moves() const noexcept {
    if (total_move_count_ == 0) return 0;
    return static_cast<float>(total_valid_moves_ / static_cast<double>(total_move_count_));
  }
  size_t awaiting_mcts_count() const noexcept { return awaiting_mcts_.size(); }
  size_t awaiting_inference_count() const noexcept {
    size_t out = 0;
    for (const auto& queue : awaiting_inference_) {
      out += queue->size();
    }
    return out;
  }
  size_t hist_count() const noexcept { return history_.size(); }
  [[nodiscard]] size_t cache_size() const {
    size_t out = 0;
    for (auto& cache : caches_) {
      if (cache) out += cache->size();
    }
    return out;
  };
  [[nodiscard]] size_t cache_hits() const {
    size_t out = 0;
    for (auto& cache : caches_) {
      if (cache) out += cache->hits();
    }
    return out;
  };
  [[nodiscard]] size_t cache_misses() const {
    size_t out = 0;
    for (auto& cache : caches_) {
      if (cache) out += cache->misses();
    }
    return out;
  };
  [[nodiscard]] size_t cache_evictions() const {
    size_t out = 0;
    for (auto& cache : caches_) {
      if (cache) out += cache->evictions();
    }
    return out;
  };
  [[nodiscard]] size_t cache_reinserts() const {
    size_t out = 0;
    for (auto& cache : caches_) {
      if (cache) out += cache->reinserts();
    }
    return out;
  };
  [[nodiscard]] size_t cache_max_size() const {
    size_t out = 0;
    for (auto& cache : caches_) {
      if (cache) out += cache->max_size();
    }
    return out;
  };

 private:
  std::unique_ptr<GameState> base_gs_;
  const PlayParams params_;
  std::vector<GameData> games_;

  std::mutex game_end_mutex_;
  Vector<float> scores_;
  uint32_t games_started_;
  uint64_t game_length_ = 0;
  double total_avg_leaf_depth_ = 0;       // full searches only
  double total_search_entropy_ = 0;       // full searches only
  double fast_total_avg_leaf_depth_ = 0;  // fast searches only
  double fast_total_search_entropy_ = 0;  // fast searches only
  double total_valid_moves_ = 0;
  uint64_t total_move_count_ = 0;
  uint64_t full_move_count_ = 0;
  uint64_t fast_move_count_ = 0;
  std::atomic<uint32_t> games_completed_ = 0;
  std::atomic<bool> stopped_{false};
  Vector<float> resign_scores_;

  ConcurrentQueue<uint32_t> awaiting_mcts_;
  std::vector<std::unique_ptr<ConcurrentQueue<uint32_t>>> awaiting_inference_;
  ConcurrentQueue<PlayHistory> history_;

  std::vector<std::shared_ptr<Cache>> caches_;

  std::vector<uint8_t> model_groups_;       // computed from params (never empty)
  uint8_t num_model_groups_;
  std::vector<uint32_t> mcts_visits_;       // per-model-group
  std::vector<EvalType> eval_types_;        // per-model-group
  std::vector<std::vector<uint8_t>> seat_perms_;  // computed from params (never empty)
  std::vector<std::vector<uint32_t>> seat_visits_;  // per-perm, per-seat visit overrides
  std::atomic<bool> eager_{false};          // GPU steal: true = hand off partial batches

  struct PermScores {
    Vector<float> scores;
    uint32_t games_completed = 0;
  };
  std::vector<PermScores> perm_scores_;
};

}  // namespace alphazero