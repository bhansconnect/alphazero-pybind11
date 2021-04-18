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

struct GameData {
  std::unique_ptr<GameState> gs;
  bool initialized;
  std::vector<MCTS> mcts;
  Vector<float> v;
  Vector<float> pi;
  Tensor<float, 3> canonical;
};

struct PlayParams {
  std::unique_ptr<GameState> base_gs;
  uint32_t games_to_play;
  uint32_t concurrent_games;
  uint32_t mcts_depth = 10;
  float cpuct = 2.0;
  bool history_enabled = true;
};

// This is a multithread safe game play manager.
// It enables running of MCTS and preparing games for GPU inference.

class PlayManager {
 public:
  explicit PlayManager(PlayParams p);

  // play will keep playing games until all games are completed.
  void play();

  // dumb_inference is a random inference function for testing.
  void dumb_inference();

 private:
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