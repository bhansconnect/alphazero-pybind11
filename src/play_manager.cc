#include "play_manager.h"

namespace alphazero {

PlayManager::PlayManager(std::unique_ptr<GameState> gs, PlayParams p)
    : base_gs_(std::move(gs)),
      params_(p),
      games_started_(params_.concurrent_games),
      cache_(Cache{params_.max_cache_size}) {
  games_.reserve(params_.concurrent_games);
  for (auto i = 0U; i < params_.concurrent_games; ++i) {
    auto gd = GameData{};
    gd.gs = base_gs_->copy();
    for (auto j = 0; j < base_gs_->num_players(); ++j) {
      gd.mcts.emplace_back(params_.cpuct, base_gs_->num_moves());
    }
    gd.v = Vector<float>{base_gs_->num_players()};
    gd.pi = Vector<float>{base_gs_->num_moves()};
    gd.v.setZero();
    gd.pi.setZero();
    games_.push_back(std::move(gd));
    awaiting_mcts_.push(i);
  }
  scores_ = Vector<float>{base_gs_->num_players()};
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
        auto temp = params_.temp;
        if (game.gs->current_turn() >= params_.temp_minimization_turn) {
          temp = 0;
        }
        const auto pi = mcts.probs(temp);
        const auto chosen_m = MCTS::pick_move(pi);
        if (params_.history_enabled) {
          PlayHistory ph;
          ph.canonical = game.canonical;
          ph.pi = pi;
          game.partial_history.push_back(ph);
        }
        for (auto& m : game.mcts) {
          m.update_root(*game.gs, chosen_m);
        }
        game.gs->play_move(chosen_m);
        const auto scores = game.gs->scores();
        if (scores.has_value()) {
          // Dump history.
          if (params_.history_enabled) {
            while (!game.partial_history.empty()) {
              auto ph = game.partial_history.back();
              ph.v = scores.value();
              history_.push(ph);
              game.partial_history.pop_back();
            }
          }
          // Game ended, reset.
          {
            std::unique_lock<std::mutex>{game_end_mutex_};
            scores_ += scores.value();
            ++games_completed_;
            // If we have started enough games just loop and complete games.
            if (games_started_ >= params_.games_to_play) {
              continue;
            }
            ++games_started_;
          }
          // Setup next game.
          game.gs = base_gs_->copy();
          for (auto& m : game.mcts) {
            m = MCTS{params_.cpuct, base_gs_->num_moves()};
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
    if (params_.max_cache_size > 0) {
      auto opt = cache_.find(game.canonical);
      if (opt.has_value()) {
        std::tie(game.v, game.pi) = opt.value();
        awaiting_mcts_.push(i.value());
        continue;
      }
    }
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

void PlayManager::update_inferences(const std::vector<uint32_t>& game_indices,
                                    const Eigen::Ref<const Matrix<float>>& v,
                                    const Eigen::Ref<const Matrix<float>>& pi) {
  for (auto i = 0UL; i < game_indices.size(); ++i) {
    auto& game = games_[game_indices[i]];
    game.v = v.row(i);
    game.pi = pi.row(i);
    if (params_.max_cache_size > 0) {
      cache_.insert(game.canonical, {game.v, game.pi});
    }
    awaiting_mcts_.push(game_indices[i]);
  }
}

}  // namespace alphazero