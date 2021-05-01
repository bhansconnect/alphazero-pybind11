#pragma once

#include <absl/hash/hash.h>

#include <atomic>
#include <deque>
#include <limits>
#include <mutex>
#include <optional>
#include <vector>

#include "concurrent_queue.h"
#include "game_state.h"
#include "lru_cache.h"
#include "mcts.h"

namespace alphazero {

struct TensorKeyWrapper {
  TensorKeyWrapper(const Tensor<float, 3>& tensor) : t(tensor) {}
  Tensor<float, 3> t;
};
template <typename H>
H AbslHashValue(H h, const TensorKeyWrapper& t) {
  return H::combine_contiguous(std::move(h), t.t.data(), t.t.size());
}
bool operator==(const TensorKeyWrapper& lhs, const TensorKeyWrapper& rhs) {
  if (lhs.t.dimensions() != rhs.t.dimensions()) {
    return false;
  }
  for (auto i = 0; i < lhs.t.dimension(0); ++i) {
    for (auto j = 0; j < lhs.t.dimension(1); ++j) {
      for (auto k = 0; k < lhs.t.dimension(2); ++k) {
        if (lhs.t(i, j, k) != rhs.t(i, j, k)) {
          return false;
        }
      }
    }
  }
  return true;
}

using Cache =
    LRUCache<TensorKeyWrapper, std::tuple<Vector<float>, Vector<float>>>;

using namespace std::chrono_literals;

constexpr const auto MAX_WAIT = 10ms;

struct PlayHistory {
  Tensor<float, 3> canonical;
  Vector<float> v;
  Vector<float> pi;
};

struct GameData {
  std::unique_ptr<GameState> gs;
  bool initialized = false;
  bool capped = false;
  std::vector<MCTS> mcts;
  Tensor<float, 3> canonical;
  Vector<float> v;
  Vector<float> pi;
  std::vector<PlayHistory> partial_history;
};

struct PlayParams {
  uint32_t games_to_play;
  uint32_t concurrent_games;
  uint32_t max_batch_size = 1;
  uint32_t max_cache_size = 0;
  std::vector<uint32_t> mcts_depth{};
  float cpuct = 2.0;
  float start_temp = 1.0;
  float final_temp = 1.0;
  float temp_decay_half_life = 0;
  bool history_enabled = false;
  bool self_play = false;
  bool add_noise = false;
  float epsilon = 0.25;
  float mcts_root_temp = 1.4;
  bool playout_cap_randomization = false;
  uint32_t playout_cap_depth = 25;
  float playout_cap_percent = 0.75;
};

// This is a multithread safe game play manager.
// It enables running of MCTS and preparing games for GPU inference.

class PlayManager {
 public:
  PlayManager(std::unique_ptr<GameState> gs, PlayParams p);

  // play will keep playing games until all games are completed.
  void play();

  void update_inferences(uint8_t player,
                         const std::vector<u_int32_t>& game_indices,
                         const Eigen::Ref<const Matrix<float>>& v,
                         const Eigen::Ref<const Matrix<float>>& pi);

  [[nodiscard]] const Vector<float> scores() const noexcept { return scores_; }
  [[nodiscard]] uint32_t remaining_games() const noexcept {
    return params_.games_to_play - games_completed_;
  }
  [[nodiscard]] uint32_t games_completed() const noexcept {
    return games_completed_;
  }
  void dumb_inference(const uint8_t player);

  [[nodiscard]] std::optional<uint32_t> pop_game(uint32_t player) noexcept {
    return awaiting_inference_[params_.self_play ? 0 : player]->pop(MAX_WAIT);
  }
  [[nodiscard]] std::vector<uint32_t> pop_games_upto(uint32_t player,
                                                     size_t n) noexcept {
    return awaiting_inference_[params_.self_play ? 0 : player]->pop_upto(
        n, MAX_WAIT);
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
    return awaiting_inference_.size();
  }
  size_t hist_count() const noexcept { return history_.size(); }
  [[nodiscard]] size_t cache_hits() const {
    size_t out = 0;
    for (auto& cache : caches_) {
      out += cache->hits();
    }
    return out;
  };
  [[nodiscard]] size_t cache_misses() const {
    size_t out = 0;
    for (auto& cache : caches_) {
      out += cache->misses();
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

  ConcurrentQueue<uint32_t> awaiting_mcts_;
  std::vector<std::unique_ptr<ConcurrentQueue<uint32_t>>> awaiting_inference_;
  ConcurrentQueue<PlayHistory> history_;

  std::vector<std::unique_ptr<Cache>> caches_;
  // Eventaully contain history, maybe store it in GameData.
};

}  // namespace alphazero