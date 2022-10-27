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
#include "lru_cache.h"
#include "mcts.h"

namespace alphazero {

using Cache = ShardedLRUCache<GameStateKeyWrapper,
                              std::tuple<Vector<float>, Vector<float>>>;

using namespace std::chrono_literals;

constexpr const auto MAX_WAIT = 10ms;

struct GameData {
  std::unique_ptr<GameState> gs;
  std::shared_ptr<GameState> leaf;
  std::vector<MCTS> mcts;
  Tensor<float, 3> canonical;
  Vector<float> v;
  Vector<float> pi;
  std::vector<PlayHistory> partial_history;
  bool initialized = false;
  bool capped = false;
  bool playthrough = false;
};

struct PlayParams {
  uint32_t games_to_play;
  uint32_t concurrent_games;
  uint32_t max_batch_size = 1;
  uint32_t max_cache_size = 0;
  uint8_t cache_shards = 1;
  std::vector<uint32_t> mcts_depth{};
  float cpuct = 2.0;
  float start_temp = 1.0;
  float final_temp = 1.0;
  float temp_decay_half_life = 0;
  bool history_enabled = false;
  bool self_play = false;
  bool tree_reuse = true;
  bool add_noise = false;
  float epsilon = 0.25;
  float mcts_root_temp = 1.4;
  bool playout_cap_randomization = false;
  uint32_t playout_cap_depth = 25;
  float playout_cap_percent = 0.75;
  float fpu_reduction = 0.0;
  float resign_percent = 0.0;
  float resign_playthrough_percent = 0.0;
};

// This is a multithread safe game play manager.
// It enables running of MCTS and preparing games for GPU inference.

class DLLEXPORT PlayManager {
 public:
  PlayManager(std::unique_ptr<GameState> gs, PlayParams p);

  // play will keep playing games until all games are completed.
  void play();

  void update_inferences(uint8_t player,
                         const std::vector<uint32_t>& game_indices,
                         const Eigen::Ref<const Matrix<float>>& v,
                         const Eigen::Ref<const Matrix<float>>& pi);

  [[nodiscard]] const Vector<float> scores() const noexcept { return scores_; }
  [[nodiscard]] const Vector<float> resign_scores() const noexcept {
    return resign_scores_;
  }
  [[nodiscard]] uint32_t remaining_games() const noexcept {
    return params_.games_to_play - games_completed_;
  }
  [[nodiscard]] uint32_t games_completed() const noexcept {
    return games_completed_;
  }
  void dumb_inference(const uint8_t player);

  [[nodiscard]] std::optional<uint32_t> pop_game(uint32_t player) noexcept {
    return awaiting_inference_[player]->pop(MAX_WAIT);
  }
  [[nodiscard]] std::vector<uint32_t> pop_games_upto(uint32_t player,
                                                     size_t n) noexcept {
    return awaiting_inference_[player]->pop_upto(n, MAX_WAIT);
  }
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
  size_t awaiting_mcts_count() const noexcept { return awaiting_mcts_.size(); }
  size_t awaiting_inference_count() const noexcept {
    auto out = 0;
    for (const auto& queue : awaiting_inference_) {
      out += queue->size();
    }
    return out;
  }
  size_t hist_count() const noexcept { return history_.size(); }
  [[nodiscard]] size_t cache_size() const {
    size_t out = 0;
    for (auto& cache : caches_) {
      out += cache.size();
    }
    return out;
  };
  [[nodiscard]] size_t cache_hits() const {
    size_t out = 0;
    for (auto& cache : caches_) {
      out += cache.hits();
    }
    return out;
  };
  [[nodiscard]] size_t cache_misses() const {
    size_t out = 0;
    for (auto& cache : caches_) {
      out += cache.misses();
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
  std::atomic<uint32_t> games_completed_ = 0;
  Vector<float> resign_scores_;

  ConcurrentQueue<uint32_t> awaiting_mcts_;
  std::vector<std::unique_ptr<ConcurrentQueue<uint32_t>>> awaiting_inference_;
  ConcurrentQueue<PlayHistory> history_;

  std::vector<Cache> caches_;
  // Eventaully contain history, maybe store it in GameData.
};

}  // namespace alphazero