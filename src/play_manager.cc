#include "play_manager.h"

namespace alphazero {

using namespace std::chrono_literals;

constexpr const auto MAX_WAIT = 100us;

PlayManager::PlayManager(PlayParams p)
    : params_(std::move(p)), games_started_(params_.concurrent_games) {
  games_.reserve(params_.concurrent_games);
  for (auto i = 0U; i < params_.concurrent_games; ++i) {
    auto gd = GameData{};
    gd.gs = params_.base_gs->copy();
    for (auto j = 0; j < params_.base_gs->num_players(); ++j) {
      gd.mcts.emplace_back(params_.cpuct, params_.base_gs->num_moves());
    }
    games_.push_back(std::move(gd));
    awaiting_mcts_.push(i);
  }
  scores_ = Vector<float>{params_.base_gs->num_players()};
  scores_.setZero();
}

void PlayManager::play() {
  while (games_completed_ < params_.games_to_play) {
    auto i = awaiting_mcts_.pop(MAX_WAIT);
    if (!i.has_value()) {
      continue;
    }
    auto& game = games_[i.value()];
    if (game.initialized) {
      // Process previous results.
      auto& mcts = game.mcts[game.gs->current_player()];
      mcts.process_result(game.v, game.pi);
      if (mcts.depth() >= params_.mcts_depth) {
        // Actually play a move.
        const auto chosen_m = mcts.pick_move(1, game.gs->num_moves());
        for (auto& m : game.mcts) {
          m.update_root(*game.gs, chosen_m);
        }
        game.gs->play_move(chosen_m);
        const auto scores = game.gs->scores();
        if (scores.has_value()) {
          // Game ended, reset.
          ++games_completed_;
          {
            std::unique_lock<std::mutex>{game_end_mutex_};
            scores_ += scores.value();
            // If we have started enough games just loop and complete games.
            if (games_started_ >= params_.games_to_play) {
              continue;
            }
            ++games_started_;
          }
          // Setup next game.
          game.gs = params_.base_gs->copy();
          for (auto& m : game.mcts) {
            m = MCTS{params_.cpuct, params_.base_gs->num_moves()};
          }
        }
      }
    } else {
      game.initialized = true;
    }
    // Find the next leaf to process and put it in the inference queue.
    auto& mcts = game.mcts[game.gs->current_player()];
    auto leaf = mcts.find_leaf(*game.gs);
    game.canonical = leaf->canonicalized();
    awaiting_inference_.push(i.value());
  }
}

void PlayManager::dumb_inference() {
  while (games_completed_ < params_.games_to_play) {
    auto i = awaiting_inference_.pop(MAX_WAIT);
    if (!i.has_value()) {
      continue;
    }
    auto& game = games_[i.value()];
    std::tie(game.v, game.pi) = dumb_eval(*game.gs);
    awaiting_mcts_.push(i.value());
  }
}

}  // namespace alphazero