#pragma once

#include <atomic>
#include <deque>
#include <mutex>
#include <optional>
#include <vector>

#include "concurrent_queue.h"
#include "game_state.h"
#include "mcts.h"

namespace alphazero {

using namespace std::chrono_literals;

constexpr const auto MAX_WAIT = 100us;

struct GameData {
  std::unique_ptr<GameState> gs;
  bool initialized;
  std::vector<MCTS> mcts;
  Vector<float> v;
  Vector<float> pi;
  Tensor<float, 3> canonical;
};

struct PlayParams {
  uint32_t games_to_play;
  uint32_t concurrent_games;
  uint32_t max_batch_size = 1;
  uint32_t mcts_depth = 10;
  float cpuct = 2.0;
  bool history_enabled = true;
};

// This is a multithread safe game play manager.
// It enables running of MCTS and preparing games for GPU inference.

class PlayManager {
 public:
  PlayManager(std::unique_ptr<GameState> gs, PlayParams p);

  // play will keep playing games until all games are completed.
  void play();

  // dumb_inference is a random inference function for testing.
  void dumb_inference();

  void update_inferences(const std::vector<u_int32_t>& game_indices,
                         const Eigen::Ref<const Matrix<float>>& v,
                         const Eigen::Ref<const Matrix<float>>& pi);

  [[nodiscard]] const Vector<float> scores() const noexcept { return scores_; }
  [[nodiscard]] uint32_t remaining_games() const noexcept {
    return params_.games_to_play - games_completed_;
  }
  [[nodiscard]] uint32_t games_completed() const noexcept {
    return games_completed_;
  }

  [[nodiscard]] std::optional<uint32_t> pop_game() noexcept {
    return awaiting_inference_.pop(MAX_WAIT);
  }
  [[nodiscard]] std::vector<uint32_t> pop_games_upto(size_t n) noexcept {
    return awaiting_inference_.pop_upto(n, MAX_WAIT);
  }
  void push_inference(const uint32_t i) noexcept { awaiting_mcts_.push(i); }
  [[nodiscard]] GameData& game_data(uint32_t i) noexcept { return games_[i]; }
  [[nodiscard]] const PlayParams& params() const noexcept { return params_; }
  size_t awaiting_mcts_count() noexcept { return awaiting_mcts_.size(); }
  size_t awaiting_inference_count() noexcept {
    return awaiting_inference_.size();
  }

 private:
  std::unique_ptr<GameState> base_gs_;
  const PlayParams params_;
  std::vector<GameData> games_;

  std::atomic<uint32_t> games_completed_ = 0;

  std::mutex game_end_mutex_;
  Vector<float> scores_;
  uint32_t games_started_;

  ConcurrentQueue<uint32_t> awaiting_mcts_;
  ConcurrentQueue<uint32_t> awaiting_inference_;

  // Eventually contain LRU cache.
  // Eventaully contain history, maybe store it in GameData.
};

}  // namespace alphazero