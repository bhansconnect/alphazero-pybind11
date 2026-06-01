#include "play_manager.h"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <optional>

#include "tracy_zones.h"

namespace alphazero {

PlayManager::PlayManager(std::unique_ptr<GameState> gs, PlayParams p)
    : base_gs_(std::move(gs)),
      params_(p),
      games_started_(params_.concurrent_games) {
  games_.reserve(params_.concurrent_games);
  if (params_.mcts_visits.size() != base_gs_->num_players()) {
    throw std::runtime_error{"You must specify MCTS visits for each player"};
  }

  // 1. Compute model_groups_
  if (params_.model_groups.empty()) {
    for (auto i = 0U; i < base_gs_->num_players(); ++i)
      model_groups_.push_back(i);
  } else {
    model_groups_ = params_.model_groups;
  }
  num_model_groups_ =
      *std::max_element(model_groups_.begin(), model_groups_.end()) + 1;

  // 2. Convert mcts_visits to per-model-group
  mcts_visits_.resize(num_model_groups_, 0);
  for (auto i = 0U; i < base_gs_->num_players(); ++i)
    mcts_visits_[model_groups_[i]] = params_.mcts_visits[i];

  // 3. Convert eval_type to per-model-group (if set)
  if (!params_.eval_type.empty()) {
    eval_types_.resize(num_model_groups_, EvalType::NN);
    for (auto i = 0U; i < base_gs_->num_players(); ++i)
      eval_types_[model_groups_[i]] = params_.eval_type[i];
  }

  // 4. Compute seat permutations
  if (params_.seat_perms.empty()) {
    seat_perms_.push_back(model_groups_);
  } else {
    seat_perms_ = params_.seat_perms;
  }

  // 5. Normalize seat overrides (validate dimensions or fill from globals)
  const auto num_perms = seat_perms_.size();
  const auto np = static_cast<size_t>(base_gs_->num_players());

  auto validate_2d = [&](const auto& vec, const char* name) {
    if (vec.size() != num_perms) {
      throw std::runtime_error{
          std::string(name) + " outer dimension must match number of seat permutations"};
    }
    for (size_t p = 0; p < num_perms; ++p) {
      if (vec[p].size() != np) {
        throw std::runtime_error{
            std::string(name) + " inner dimension must match number of players"};
      }
    }
  };

  if (params_.seat_visits.empty()) {
    seat_visits_.resize(num_perms);
    for (size_t p = 0; p < num_perms; ++p) {
      seat_visits_[p].resize(np);
      for (size_t s = 0; s < np; ++s) {
        seat_visits_[p][s] = mcts_visits_[seat_perms_[p][s]];
      }
    }
  } else {
    validate_2d(params_.seat_visits, "seat_visits");
    seat_visits_ = params_.seat_visits;
  }

  if (params_.seat_epsilon.empty()) {
    seat_epsilon_.resize(num_perms, std::vector<float>(np, params_.epsilon));
  } else {
    validate_2d(params_.seat_epsilon, "seat_epsilon");
    seat_epsilon_ = params_.seat_epsilon;
  }

  if (params_.seat_mcts_root_temp.empty()) {
    seat_mcts_root_temp_.resize(num_perms,
        std::vector<float>(np, params_.mcts_root_temp));
  } else {
    validate_2d(params_.seat_mcts_root_temp, "seat_mcts_root_temp");
    seat_mcts_root_temp_ = params_.seat_mcts_root_temp;
  }

  if (params_.seat_root_fpu_zero.empty()) {
    seat_root_fpu_zero_.resize(num_perms,
        std::vector<uint8_t>(np, params_.root_fpu_zero ? 1u : 0u));
  } else {
    validate_2d(params_.seat_root_fpu_zero, "seat_root_fpu_zero");
    seat_root_fpu_zero_ = params_.seat_root_fpu_zero;
  }

  if (params_.seat_gumbel_enabled.empty()) {
    seat_gumbel_enabled_.resize(num_perms,
        std::vector<uint8_t>(np, params_.gumbel_enabled ? 1u : 0u));
  } else {
    validate_2d(params_.seat_gumbel_enabled, "seat_gumbel_enabled");
    seat_gumbel_enabled_ = params_.seat_gumbel_enabled;
  }
  if (params_.seat_gumbel_m.empty()) {
    seat_gumbel_m_.resize(num_perms,
        std::vector<uint32_t>(np, params_.gumbel_m));
  } else {
    validate_2d(params_.seat_gumbel_m, "seat_gumbel_m");
    seat_gumbel_m_ = params_.seat_gumbel_m;
  }
  if (params_.seat_gumbel_c_visit.empty()) {
    seat_gumbel_c_visit_.resize(num_perms,
        std::vector<float>(np, params_.gumbel_c_visit));
  } else {
    validate_2d(params_.seat_gumbel_c_visit, "seat_gumbel_c_visit");
    seat_gumbel_c_visit_ = params_.seat_gumbel_c_visit;
  }
  if (params_.seat_gumbel_c_scale.empty()) {
    seat_gumbel_c_scale_.resize(num_perms,
        std::vector<float>(np, params_.gumbel_c_scale));
  } else {
    validate_2d(params_.seat_gumbel_c_scale, "seat_gumbel_c_scale");
    seat_gumbel_c_scale_ = params_.seat_gumbel_c_scale;
  }
  if (params_.seat_gumbel_full.empty()) {
    seat_gumbel_full_.resize(num_perms,
        std::vector<uint8_t>(np, params_.gumbel_full ? 1u : 0u));
  } else {
    validate_2d(params_.seat_gumbel_full, "seat_gumbel_full");
    seat_gumbel_full_ = params_.seat_gumbel_full;
  }
  // G3 opt-in: default false (i.e. paper-faithful G1 acting for all Gumbel
  // seats). Tournaments / self-play can flip individual seats to G3 via
  // seat_gumbel_use_improved_policy.
  if (params_.seat_gumbel_use_improved_policy.empty()) {
    seat_gumbel_use_improved_policy_.resize(num_perms,
        std::vector<uint8_t>(np, 0u));
  } else {
    validate_2d(params_.seat_gumbel_use_improved_policy,
                "seat_gumbel_use_improved_policy");
    seat_gumbel_use_improved_policy_ = params_.seat_gumbel_use_improved_policy;
  }
  // Resign threshold/consecutive default to "disabled" sentinel values when
  // the caller doesn't specify per-seat overrides. -2.0 is the disabled
  // marker (any value in [-1, 1] is a meaningful threshold).
  if (params_.seat_resign_threshold.empty()) {
    seat_resign_threshold_.resize(num_perms, std::vector<float>(np, -2.0f));
  } else {
    validate_2d(params_.seat_resign_threshold, "seat_resign_threshold");
    seat_resign_threshold_ = params_.seat_resign_threshold;
  }
  if (params_.seat_resign_consecutive.empty()) {
    seat_resign_consecutive_.resize(num_perms, std::vector<uint32_t>(np, 1u));
  } else {
    validate_2d(params_.seat_resign_consecutive, "seat_resign_consecutive");
    seat_resign_consecutive_ = params_.seat_resign_consecutive;
  }

  // 6. Create per-model-group queues and caches
  for (auto i = 0U; i < num_model_groups_; ++i) {
    awaiting_inference_.push_back(
        std::make_unique<ConcurrentQueue<uint32_t>>());
  }
  if (params_.max_cache_size > 0) {
    auto per_group = params_.max_cache_size / num_model_groups_;
    auto ghost_per_group = per_group * 9 / 10;
    auto nm = static_cast<uint32_t>(base_gs_->num_moves());
    auto nv = static_cast<uint32_t>(base_gs_->num_players() + 1);
    for (auto i = 0U; i < num_model_groups_; ++i)
      caches_.push_back(
          std::make_shared<Cache>(per_group, params_.cache_shards, ghost_per_group, nm, nv));
  }

  // 7. Init per-perm score tracking
  for (size_t p = 0; p < seat_perms_.size(); ++p) {
    PermScores ps;
    ps.scores = Vector<float>{base_gs_->num_players() + 1};
    ps.scores.setZero();
    perm_scores_.push_back(std::move(ps));
  }

  // 8. Init games with seat permutations
  for (auto i = 0U; i < params_.concurrent_games; ++i) {
    auto gd = GameData{};
    gd.gs = base_gs_->copy();
    gd.gs->randomize_start();
    gd.perm_index = i % seat_perms_.size();
    gd.seat_perm = seat_perms_[gd.perm_index];
    for (auto j = 0; j < base_gs_->num_players(); ++j) {
      gd.mcts.emplace_back(make_mcts(gd.perm_index, j));
    }
    gd.canonical = Tensor<float, 3>{base_gs_->canonicalized()};
    gd.v = Vector<float>{base_gs_->num_players() + 1};
    gd.pi = Vector<float>{base_gs_->num_moves()};
    gd.v.setZero();
    gd.pi.setZero();
    games_.push_back(std::move(gd));
    awaiting_mcts_.push(i);
  }

  scores_ = Vector<float>{base_gs_->num_players() + 1};
  scores_.setZero();
  resign_scores_ = Vector<float>{base_gs_->num_players() + 1};
  resign_scores_.setZero();

  // Init per-variant score tracking if the game supports variants.
  int nvar = base_gs_->num_variants();
  int nperms = static_cast<int>(seat_perms_.size());
  for (int v = 0; v < nvar; ++v) {
    PermScores vs;
    vs.scores = Vector<float>{base_gs_->num_players() + 1};
    vs.scores.setZero();
    variant_scores_.push_back(std::move(vs));

    std::vector<PermScores> vps;
    for (int p = 0; p < nperms; ++p) {
      PermScores ps;
      ps.scores = Vector<float>{base_gs_->num_players() + 1};
      ps.scores.setZero();
      vps.push_back(std::move(ps));
    }
    variant_perm_scores_.push_back(std::move(vps));
  }
  variant_metrics_.resize(nvar);
}

void PlayManager::play() {
  AZ_SET_THREAD_NAME("mcts_worker");
  AZ_ZONE_SCOPED;
  thread_local std::default_random_engine re{std::random_device{}()};
  thread_local std::uniform_real_distribution<float> dist{0.0F, 1.0F};
  while (games_completed_ < params_.games_to_play && !stopped_.load(std::memory_order_relaxed)) {
    std::optional<uint32_t> i = awaiting_mcts_.pop(MAX_WAIT);
    if (!i.has_value()) {
      continue;
    }
    auto& game = games_[i.value()];
    if (game.initialized) {
      // Process previous results.
      const auto cp = game.gs->current_player();
      auto& mcts = game.mcts[cp];
      mcts.process_result(*game.gs, game.v, game.pi,
                          seat_epsilon_[game.perm_index][cp] > 0 && !game.capped);
      auto goal_depth = game.capped ? params_.playout_cap_depth
                                    : seat_visits_[game.perm_index][cp];
      if (mcts.depth() >= goal_depth) {
        // Actually play a move.
        auto temp = params_.start_temp;
        float half_life = params_.temp_decay_half_life;
        if (!params_.temp_decay_half_life_by_variant.empty()) {
          const int vid = game.gs->get_variant_id();
          if (vid >= 0 &&
              vid < static_cast<int>(params_.temp_decay_half_life_by_variant.size())) {
            half_life = params_.temp_decay_half_life_by_variant[vid];
          }
        }
        if (half_life != 0) {
          const auto t = game.gs->current_turn();
          constexpr float ln2 = 0.693;
          const auto lambda = ln2 / half_life;
          temp -= params_.final_temp;
          temp *= std::exp(-lambda * t);
          temp += params_.final_temp;
        }
        std::optional<Vector<float>> resign_score = std::nullopt;
        if (params_.resign_percent > 0 && !game.playthrough) {
          if (base_gs_->num_players() != 2) {
            throw std::runtime_error{"Resigning only works in 2 player games"};
          }
          const auto pred_score = mcts.root_value();
          const auto w = pred_score[0];
          const auto l = pred_score[1];
          const auto d = pred_score[2];
          const auto resign_val = 1.0 - params_.resign_percent;
          // Check resign thresholds.
          auto tmp_score = Vector<float>{base_gs_->num_players() + 1};
          tmp_score.setZero();
          if (w > resign_val) {
            tmp_score[cp] = 1.0;
          } else if (l > resign_val) {
            const auto opponent = (cp + 1) % 2;
            tmp_score[opponent] = 1.0;
          } else if (d > resign_val) {
            tmp_score[base_gs_->num_players()] = 1.0;
          }
          if (tmp_score.sum() > 0) {
            // If we should resign randomly check playthrough chance.
            if (dist(re) < params_.resign_playthrough_percent) {
              game.playthrough = true;
            } else {
              resign_score = std::make_optional(tmp_score);
            }
          }
        }
        // Per-seat opt-in resign (independent of resign_percent). Triggered
        // when the current seat's expected score V = W − L drops to/below
        // its configured threshold for ``seat_resign_consecutive`` consecutive
        // own moves. -2.0 sentinel disables per seat.
        if (!resign_score.has_value() && !game.playthrough) {
          const float seat_thresh =
              seat_resign_threshold_[game.perm_index][cp];
          if (seat_thresh > -2.0f) {
            if (base_gs_->num_players() != 2) {
              throw std::runtime_error{"Per-seat resign only works in 2 player games"};
            }
            if (game.resign_streak.empty()) {
              game.resign_streak.assign(base_gs_->num_players(), 0u);
            }
            const auto pred_score = mcts.root_value();
            const float v_self = pred_score[0] - pred_score[1];
            if (v_self <= seat_thresh) {
              ++game.resign_streak[cp];
            } else {
              game.resign_streak[cp] = 0;
            }
            const uint32_t need =
                std::max(1u, seat_resign_consecutive_[game.perm_index][cp]);
            if (game.resign_streak[cp] >= need) {
              auto tmp_score = Vector<float>{base_gs_->num_players() + 1};
              tmp_score.setZero();
              const auto opponent = (cp + 1) % 2;
              tmp_score[opponent] = 1.0;
              resign_score = std::make_optional(tmp_score);
            }
          }
        }
        uint32_t chosen_m;
        if (mcts.gumbel_enabled() && !game.capped) {
          // Default Gumbel acting is paper-faithful (Danihelka 2022 Eq. 11):
          //   a* = argmax_a [ g(a) + logits(a) + sigma(N) q_hat(a) ]
          // Variance comes from fresh Gumbel perturbations per search, not
          // from sampling. Temperature does not modulate Gumbel selection;
          // control diversity via gumbel_c_scale instead. Empirically this
          // outperforms improved-policy-sampling (G3) on trained networks.
          // The G3 path remains available behind an explicit opt-in flag
          // (seat_gumbel_use_improved_policy).
          const bool use_g3 =
              static_cast<bool>(seat_gumbel_use_improved_policy_
                                    [game.perm_index][cp]);
          if (!use_g3) {
            chosen_m = mcts.gumbel_final_action();
          } else {
            auto pi = mcts.gumbel_improved_policy();
            if (temp != 1.0f && temp > 0.0f) {
              pi = pi.array().pow(1.0f / temp);
              const auto s = pi.sum();
              if (s > 0) pi /= s;
            } else if (temp <= 0.0f) {
              // temp=0 with G3: degenerate, use argmax of pi'
              chosen_m = static_cast<uint32_t>(
                  std::distance(pi.data(),
                                std::max_element(pi.data(),
                                                 pi.data() + pi.size())));
              pi.setZero();
              pi(chosen_m) = 1.0f;
            }
            if (pi.sum() > 0) {
              chosen_m = MCTS::pick_move(pi);
            } else {
              chosen_m = mcts.gumbel_final_action();
            }
          }
        } else {
          const auto pi = mcts.probs(temp);
          chosen_m = MCTS::pick_move(pi);
        }
        if (params_.history_enabled && !game.capped) {
          PlayHistory ph{
              .canonical = Tensor<float, 3>{game.gs->canonicalized()},
              .v = Vector<float>{game.v.size()},
              .pi = Vector<float>{
                  params_.gumbel_enabled
                      ? mcts.gumbel_improved_policy()
                      : ((params_.policy_target_pruning &&
                          seat_epsilon_[game.perm_index][cp] > 0)
                             ? mcts.probs_pruned(1.0)
                             : mcts.probs(1.0))
              },
          };
          ph.v.setZero();
          game.partial_history.push_back(
              PendingHistory{.ph = std::move(ph),
                             .player = game.gs->current_player()});
        }
        if (!game.capped) {
          game.total_avg_leaf_depth += mcts.avg_leaf_depth();
          game.total_search_entropy += mcts.normalized_root_entropy();
          ++game.full_move_count;
        } else {
          game.fast_total_avg_leaf_depth += mcts.avg_leaf_depth();
          game.fast_total_search_entropy += mcts.normalized_root_entropy();
          ++game.fast_move_count;
        }
        game.total_valid_moves += mcts.num_root_children();
        ++game.move_count;
        for (auto& m : game.mcts) {
          m.update_root(*game.gs, chosen_m);
        }
        game.gs->play_move(chosen_m);
        auto scores = game.gs->scores();
        if (!scores.has_value() && resign_score.has_value()) {
          scores = resign_score;
        } else {
          resign_score = std::nullopt;
        }
        if (scores.has_value()) {
          // Dump history.
          if (params_.history_enabled) {
            while (!game.partial_history.empty()) {
              auto& pending = game.partial_history.back();
              if (base_gs_->relative_values()) {
                pending.ph.v = absolute_to_relative(
                    scores.value(), pending.player,
                    base_gs_->num_players());
              } else {
                pending.ph.v = scores.value();
              }
              history_.push(std::move(pending.ph));
              game.partial_history.pop_back();
            }
          }
          // Game ended, reset.
          {
            std::unique_lock<std::mutex> lock{game_end_mutex_};
            scores_ += scores.value();
            perm_scores_[game.perm_index].scores += scores.value();
            ++perm_scores_[game.perm_index].games_completed;
            int vid = game.gs->get_variant_id();
            if (vid >= 0 && vid < static_cast<int>(variant_scores_.size())) {
              variant_scores_[vid].scores += scores.value();
              ++variant_scores_[vid].games_completed;
              variant_perm_scores_[vid][game.perm_index].scores += scores.value();
              ++variant_perm_scores_[vid][game.perm_index].games_completed;
              auto& vm = variant_metrics_[vid];
              vm.game_length += game.gs->current_turn();
              vm.total_avg_leaf_depth += game.total_avg_leaf_depth;
              vm.total_search_entropy += game.total_search_entropy;
              vm.fast_total_avg_leaf_depth += game.fast_total_avg_leaf_depth;
              vm.fast_total_search_entropy += game.fast_total_search_entropy;
              vm.total_valid_moves += game.total_valid_moves;
              vm.total_move_count += game.move_count;
              vm.full_move_count += game.full_move_count;
              vm.fast_move_count += game.fast_move_count;
            }
            if (resign_score.has_value()) {
              resign_scores_ += resign_score.value();
            }
            ++games_completed_;
            game_length_ += game.gs->current_turn();
            total_avg_leaf_depth_ += game.total_avg_leaf_depth;
            total_search_entropy_ += game.total_search_entropy;
            fast_total_avg_leaf_depth_ += game.fast_total_avg_leaf_depth;
            fast_total_search_entropy_ += game.fast_total_search_entropy;
            total_valid_moves_ += game.total_valid_moves;
            total_move_count_ += game.move_count;
            full_move_count_ += game.full_move_count;
            fast_move_count_ += game.fast_move_count;
            game.total_avg_leaf_depth = 0;
            game.total_search_entropy = 0;
            game.fast_total_avg_leaf_depth = 0;
            game.fast_total_search_entropy = 0;
            game.total_valid_moves = 0;
            game.move_count = 0;
            game.full_move_count = 0;
            game.fast_move_count = 0;
            // If we have started enough games just loop and complete games.
            if (games_started_ >= params_.games_to_play) {
              continue;
            }
            // Assign next perm for new game.
            game.perm_index = games_started_ % seat_perms_.size();
            game.seat_perm = seat_perms_[game.perm_index];
            ++games_started_;
          }
          // Setup next game.
          game.gs = base_gs_->copy();
          game.gs->randomize_start();
          for (auto j = 0; j < base_gs_->num_players(); ++j) {
            game.mcts[j] = make_mcts(game.perm_index, j);
          }
        }
        // A move has been played, update playout cap.
        game.capped = params_.playout_cap_randomization &&
                      (dist(re) < params_.playout_cap_percent);
        // Capped self-play searches use PUCT (sims_target=0 disables Gumbel),
        // even when gumbel_enabled. Connect4 retest showed that Gumbel-at-
        // capped-visits (high inherent variance ~40% at low visits) produces
        // noisier game trajectories than PUCT-at-capped, hurting training
        // data quality. v1 design (Gumbel for full + PUCT for capped) beats
        // v2 (Gumbel everywhere) by 217 elo at equal compute on Connect4.
        {
          const auto next_cp_g = game.gs->current_player();
          const auto sims_target = game.capped
              ? (params_.fast_search_uses_gumbel
                     ? params_.playout_cap_depth
                     : 0u)
              : seat_visits_[game.perm_index][next_cp_g];
          game.mcts[next_cp_g].set_gumbel_num_sims(sims_target);
        }
        // If not reusing the mcts tree, reset mcts.
        if (!params_.tree_reuse) {
          for (auto j = 0; j < base_gs_->num_players(); ++j) {
            game.mcts[j] = make_mcts(game.perm_index, j);
          }
        } else {
          // Re-apply root policy temperature and noise on the reused subtree.
          auto next_cp = game.gs->current_player();
          auto& next_mcts = game.mcts[next_cp];
          if (next_mcts.root_n() > 0) {
            next_mcts.apply_root_policy_temp();
            if (seat_epsilon_[game.perm_index][next_cp] > 0 && !game.capped) {
              next_mcts.add_root_noise();
            }
          }
        }
      }
    } else {
      game.initialized = true;
      game.capped = params_.playout_cap_randomization &&
                    (dist(re) < params_.playout_cap_percent);
      // Capped: see note above. Controlled by fast_search_uses_gumbel.
      {
        const auto first_cp = game.gs->current_player();
        const auto sims_target = game.capped
            ? (params_.fast_search_uses_gumbel
                   ? params_.playout_cap_depth
                   : 0u)
            : seat_visits_[game.perm_index][first_cp];
        game.mcts[first_cp].set_gumbel_num_sims(sims_target);
      }
    }
    // Find the next leaf to process and put it in the inference queue.
    const auto cp = game.gs->current_player();
    auto& mcts = game.mcts[cp];
    auto leaf = mcts.find_leaf(*game.gs);

    auto group = game.seat_perm[cp];
    auto et = eval_types_.empty() ? EvalType::NN : eval_types_[group];
    if (et != EvalType::NN) {
      if (et == EvalType::PLAYOUT) {
        std::tie(game.v, game.pi) = playout_eval(*leaf);
      } else {
        std::tie(game.v, game.pi) = dumb_eval(*leaf);
      }
      awaiting_mcts_.push(i.value());
      continue;
    }

    game.canonical = leaf->canonicalized();
    game.leaf_hash = hash_game_state(*leaf);

    if (!caches_.empty() && caches_[group]) {
      if (caches_[group]->find(game.leaf_hash, game.pi.data(), game.v.data())) {
        awaiting_mcts_.push(i.value());
        continue;
      }
    }
    awaiting_inference_[group]->push(i.value());
  }
}

MCTS PlayManager::make_mcts(uint8_t perm_index, int player) const {
  return MCTS{params_.cpuct,
              base_gs_->num_players(),
              base_gs_->num_moves(),
              seat_epsilon_[perm_index][player],
              seat_mcts_root_temp_[perm_index][player],
              params_.fpu_reduction,
              base_gs_->relative_values(),
              static_cast<bool>(seat_root_fpu_zero_[perm_index][player]),
              params_.shaped_dirichlet,
              static_cast<bool>(seat_gumbel_enabled_[perm_index][player]),
              seat_gumbel_m_[perm_index][player],
              seat_gumbel_c_visit_[perm_index][player],
              seat_gumbel_c_scale_[perm_index][player],
              static_cast<bool>(seat_gumbel_full_[perm_index][player])};
}

void PlayManager::update_inferences(const uint8_t group,
                                    const std::vector<uint32_t>& game_indices,
                                    const Eigen::Ref<const Matrix<float>>& v,
                                    const Eigen::Ref<const Matrix<float>>& pi) {
  AZ_ZONE_SCOPED;
  std::vector<uint64_t> hashes;
  std::vector<const float*> policies;
  std::vector<const float*> vals;
  for (auto i = 0UL; i < game_indices.size(); ++i) {
    auto& game = games_[game_indices[i]];
    game.v = v.row(i);
    game.pi = pi.row(i);
    if (!caches_.empty() && caches_[group]) {
      hashes.push_back(game.leaf_hash);
      policies.push_back(game.pi.data());
      vals.push_back(game.v.data());
    }
  }
  if (!caches_.empty() && caches_[group]) {
    caches_[group]->insert_many(hashes.data(), policies.data(), vals.data(),
                                hashes.size());
  }
  awaiting_mcts_.push_many(game_indices);
}

PlayManager::PlayManager(std::unique_ptr<GameState> gs, PlayParams p,
                         std::vector<std::shared_ptr<Cache>> external_caches)
    : PlayManager(std::move(gs), [&]{ p.max_cache_size = 0; return p; }())
{
  caches_ = std::move(external_caches);
}

}  // namespace alphazero