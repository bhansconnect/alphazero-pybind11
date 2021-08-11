#include "play_manager.h"

#include <cmath>

namespace alphazero {

PlayManager::PlayManager(std::unique_ptr<GameState> gs, PlayParams p)
    : base_gs_(std::move(gs)),
      params_(p),
      games_started_(params_.concurrent_games) {
  games_.reserve(params_.concurrent_games);
  if (params_.mcts_depth.size() != base_gs_->num_players()) {
    throw std::runtime_error{"You must specify an MCTS depth for each player"};
  }
  for (auto i = 0U; i < params_.concurrent_games; ++i) {
    auto gd = GameData{};
    gd.gs = base_gs_->copy();
    for (auto j = 0; j < base_gs_->num_players(); ++j) {
      gd.mcts.emplace_back(params_.cpuct, base_gs_->num_players(),
                           base_gs_->num_moves(), params_.epsilon,
                           params_.mcts_root_temp, params_.fpu_reduction);
    }
    gd.canonical = Tensor<float, 3>{base_gs_->canonicalized()};
    gd.v = Vector<float>{base_gs_->num_players() + 1};
    gd.pi = Vector<float>{base_gs_->num_moves()};
    gd.v.setZero();
    gd.pi.setZero();
    games_.push_back(std::move(gd));
    awaiting_mcts_.push(i);
  }
  for (auto i = 0U; i < base_gs_->num_players(); ++i) {
    caches_.push_back(std::make_unique<Cache>(
        params_.max_cache_size /
        (params_.self_play ? 1 : base_gs_->num_players())));
    awaiting_inference_.push_back(
        std::make_unique<ConcurrentQueue<uint32_t>>());
    if (params_.self_play) {
      break;
    }
  }
  scores_ = Vector<float>{base_gs_->num_players() + 1};
  scores_.setZero();
}

void PlayManager::play() {
  thread_local std::default_random_engine re{std::random_device{}()};
  thread_local std::uniform_real_distribution<float> dist{0.0F, 1.0F};
  while (games_completed_ < params_.games_to_play) {
    auto i = awaiting_mcts_.pop(MAX_WAIT);
    if (!i.has_value()) {
      continue;
    }
    auto& game = games_[i.value()];
    if (game.initialized) {
      // Process previous results.
      auto cp = game.gs->current_player();
      auto& mcts = game.mcts[cp];
      mcts.process_result(*game.gs, game.v, game.pi,
                          params_.add_noise && !game.capped);
      auto goal_depth =
          game.capped ? params_.playout_cap_depth : params_.mcts_depth[cp];
      if (mcts.depth() >= goal_depth) {
        // Actually play a move.
        auto temp = params_.start_temp;
        if (params_.temp_decay_half_life != 0) {
          const auto t = game.gs->current_turn();
          constexpr float ln2 = 0.693;
          const auto lambda = ln2 / params_.temp_decay_half_life;
          temp -= params_.final_temp;
          temp *= std::exp(-lambda * t);
          temp += params_.final_temp;
        }
        const auto pi = mcts.probs(temp);
        const auto chosen_m = MCTS::pick_move(pi);
        if (params_.history_enabled && !game.capped) {
          PlayHistory ph{
              .canonical = Tensor<float, 3>{game.gs->canonicalized()},
              .v = Vector<float>{game.v.size()},
              .pi = Vector<float>{mcts.probs(1.0)},
          };
          ph.v.setZero();
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
            game_length_ += game.gs->current_turn();
            // If we have started enough games just loop and complete games.
            if (games_started_ >= params_.games_to_play) {
              continue;
            }
            ++games_started_;
          }
          // Setup next game.
          game.gs = base_gs_->copy();
          for (auto& m : game.mcts) {
            m = MCTS{params_.cpuct,          base_gs_->num_players(),
                     base_gs_->num_moves(),  params_.epsilon,
                     params_.mcts_root_temp, params_.fpu_reduction};
          }
        }
        // A move has been played, update playout cap.
        game.capped = params_.playout_cap_randomization &&
                      (dist(re) < params_.playout_cap_percent);
        // If not reusing the mcts tree, reset mcts.
        if (!params_.tree_reuse) {
          for (auto& m : game.mcts) {
            m = MCTS{params_.cpuct,          base_gs_->num_players(),
                     base_gs_->num_moves(),  params_.epsilon,
                     params_.mcts_root_temp, params_.fpu_reduction};
          }
        }
      }
    } else {
      game.initialized = true;
      game.capped = params_.playout_cap_randomization &&
                    (dist(re) < params_.playout_cap_percent);
    }
    // Find the next leaf to process and put it in the inference queue.
    auto& mcts = game.mcts[game.gs->current_player()];
    auto leaf = mcts.find_leaf(*game.gs);
    game.canonical = leaf->canonicalized();
    // Minimize the storage of the leaf node. It is only used as a hash key and
    // network input.
    leaf->minimize_storage();
    game.leaf = std::move(leaf);
    if (params_.max_cache_size > 0) {
      auto opt =
          caches_[params_.self_play ? 0 : game.gs->current_player()]->find(
              game.leaf);
      if (opt.has_value()) {
        std::tie(game.v, game.pi) = opt.value();
        awaiting_mcts_.push(i.value());
        continue;
      }
    }
    awaiting_inference_[params_.self_play ? 0 : game.gs->current_player()]
        ->push(i.value());
  }
}

void PlayManager::update_inferences(const uint8_t player,
                                    const std::vector<uint32_t>& game_indices,
                                    const Eigen::Ref<const Matrix<float>>& v,
                                    const Eigen::Ref<const Matrix<float>>& pi) {
  for (auto i = 0UL; i < game_indices.size(); ++i) {
    auto& game = games_[game_indices[i]];
    game.v = v.row(i);
    game.pi = pi.row(i);
    if (params_.max_cache_size > 0) {
      caches_[params_.self_play ? 0 : player]->insert(
          game.leaf, {Vector<float>{game.v}, Vector<float>{game.pi}});
    }
    awaiting_mcts_.push(game_indices[i]);
  }
}

void PlayManager::dumb_inference(const uint8_t player) {
  while (games_completed_ < params_.games_to_play) {
    auto i = awaiting_inference_[params_.self_play ? 0 : player]->pop(MAX_WAIT);
    if (!i.has_value()) {
      continue;
    }
    auto& game = games_[i.value()];
    std::tie(game.v, game.pi) = dumb_eval(*game.gs);
    if (params_.max_cache_size > 0) {
      caches_[params_.self_play ? 0 : player]->insert(
          game.leaf, {Vector<float>{game.v}, Vector<float>{game.pi}});
    }
    awaiting_mcts_.push(i.value());
  }
}

}  // namespace alphazero