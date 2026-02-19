#include "mcts.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>

#include "pcg/pcg_random.hpp"

namespace alphazero {

constexpr const float NOISE_ALPHA_RATIO = 10.83;
thread_local pcg32 re{pcg_extras::seed_seq_from<std::random_device>{}};

void Node::add_children(const Vector<uint8_t>& valids) noexcept {
  children.reserve(valids.sum());
  for (auto w = 0; w < valids.size(); ++w) {
    if (valids(w) == 1) {
      children.emplace_back(w);
    }
  }
  std::shuffle(children.begin(), children.end(), re);
}

void Node::update_policy(const Vector<float>& pi) noexcept {
  for (auto& c : children) {
    c.policy = pi(c.move);
  }
}

float Node::uct(float sqrt_parent_n, float cpuct,
                float fpu_value) const noexcept {
  return (n == 0 ? fpu_value : q) +
         cpuct * policy * sqrt_parent_n / static_cast<float>(n + 1);
}

Node* Node::best_child(float cpuct, float fpu_reduction) noexcept {
  auto seen_policy = 0.0f;
  for (const auto& c : children) {
    if (c.n > 0) {
      seen_policy += c.policy;
    }
  }
  auto fpu_value = v - fpu_reduction * std::sqrt(seen_policy);
  auto sqrt_n = std::sqrt(static_cast<float>(n));
  auto best_i = 0;
  auto best_uct = children.at(0).uct(sqrt_n, cpuct, fpu_value);
  for (auto i = 1; i < static_cast<int>(children.size()); ++i) {
    auto uct = children.at(i).uct(sqrt_n, cpuct, fpu_value);
    if (uct > best_uct) {
      best_uct = uct;
      best_i = i;
    }
  }
  return &children.at(best_i);
}

void MCTS::update_root(const GameState& gs, uint32_t move) {
  depth_ = 0;
  total_leaf_depth_ = 0;
  if (root_.children.empty()) {
    root_.add_children(gs.valid_moves());
  }
  auto x = std::find_if(root_.children.begin(), root_.children.end(),
                        [move](const Node& n) { return n.move == move; });
  if (x == root_.children.end()) {
    std::cout << gs.dump();
    throw std::runtime_error("ahh, what is this move: " + std::to_string(move));
  }
  Node tmp = *x;
  root_ = tmp;
}

void MCTS::add_root_noise() {
  const auto legal_move_count = root_.children.size();
  auto noise = Vector<float>{num_moves_};
  auto sum = 0.0;

  if (shaped_dirichlet_ && legal_move_count > 1) {
    auto N = static_cast<float>(legal_move_count);
    auto log_sum = 0.0f;
    for (const auto& c : root_.children) {
      log_sum += std::log(std::min(c.policy, 0.01f) + 1e-20f);
    }
    auto log_mean = log_sum / N;

    auto shaped_sum = 0.0f;
    for (const auto& c : root_.children) {
      auto lp = std::log(std::min(c.policy, 0.01f) + 1e-20f);
      shaped_sum += std::max(0.0f, lp - log_mean);
    }

    auto uniform = 1.0f / N;
    for (auto& c : root_.children) {
      auto lp = std::log(std::min(c.policy, 0.01f) + 1e-20f);
      auto shaped = std::max(0.0f, lp - log_mean);
      auto alpha_prop = (shaped_sum > 0)
          ? 0.5f * (shaped / shaped_sum + uniform)
          : uniform;
      alpha_prop = std::max(alpha_prop, 1e-6f);
      auto dist = std::gamma_distribution<float>{NOISE_ALPHA_RATIO * alpha_prop, 1.0f};
      noise(c.move) = dist(re);
      sum += noise(c.move);
    }
  } else {
    auto dist = std::gamma_distribution<float>{
        NOISE_ALPHA_RATIO / static_cast<float>(legal_move_count), 1.0};
    for (auto& c : root_.children) {
      noise(c.move) = dist(re);
      sum += noise(c.move);
    }
  }

  for (auto& c : root_.children) {
    c.policy = c.policy * (1 - epsilon_) + epsilon_ * noise(c.move) / static_cast<float>(sum);
  }
}

void MCTS::apply_root_policy_temp() {
  if (root_policy_temp_ == 1.0f) return;
  float sum = 0.0f;
  for (auto& c : root_.children) {
    c.policy = std::pow(c.policy, 1.0f / root_policy_temp_);
    sum += c.policy;
  }
  if (sum > 0.0f) {
    for (auto& c : root_.children) {
      c.policy /= sum;
    }
  }
}

std::unique_ptr<GameState> MCTS::find_leaf(const GameState& gs) {
  current_ = &root_;
  auto leaf = gs.copy();
  while (current_->n > 0 && !current_->scores.has_value()) {
    path_.push_back(current_);

    float fpu = (current_ == &root_ && root_fpu_zero_) ? 0.0f : fpu_reduction_;
    current_ = current_->best_child(cpuct_, fpu);
    leaf->play_move(current_->move);
  }
  total_leaf_depth_ += path_.size();
  if (current_->n == 0) {
    current_->player = leaf->current_player();
    current_->scores = leaf->scores();
    current_->add_children(leaf->valid_moves());
  }
  return leaf;
}

void MCTS::process_result(const GameState& gs, Vector<float>& value,
                          Vector<float>& pi, bool root_noise_enabled) {
  if (current_->scores.has_value()) {
    value = current_->scores.value();
  } else {
    // Rescale pi based on valid moves.
    auto valids = Vector<float>(gs.num_moves());
    valids.setZero();
    for (auto& c : current_->children) {
      valids(c.move) = 1;
    }
    pi.array() *= valids.array();
    pi /= pi.sum();
    if (current_ == &root_) {
      pi = pi.array().pow(1.0f / root_policy_temp_);
      pi /= pi.sum();
      current_->update_policy(pi);
      if (root_noise_enabled) {
        add_root_noise();
      }
    } else {
      current_->update_policy(pi);
    }
    if (relative_values_) {
      value = relative_to_absolute(value, current_->player, num_players_);
    }
  }

  while (!path_.empty()) {
    auto* parent = path_.back();
    path_.pop_back();
    auto v = value(parent->player);
    // Add draws.
    v += value(num_players_) / num_players_;
    current_->q = (current_->q * static_cast<float>(current_->n) + v) /
                  static_cast<float>(current_->n + 1);
    current_->d =
        (current_->d * static_cast<float>(current_->n) + value(num_players_)) /
        static_cast<float>(current_->n + 1);
    if (current_->n == 0) {
      auto leaf_v =
          value(current_->player) + value(num_players_) / num_players_;
      current_->v = leaf_v;
    }
    ++current_->n;
    current_ = parent;
  }
  // On the first evaluation the root is the leaf and the backprop loop
  // doesn't execute (path_ is empty). Set root_.v so FPU for unvisited
  // children uses the actual evaluation rather than the default of 0.
  if (root_.n == 0) {
    root_.v = value(root_.player) + value(num_players_) / num_players_;
    root_.d = value(num_players_);
  }
  ++depth_;
  ++root_.n;
}

Vector<uint32_t> MCTS::counts() const noexcept {
  auto counts = Vector<uint32_t>{num_moves_};
  counts.setZero();
  for (const auto& c : root_.children) {
    counts(c.move) = c.n;
  }
  return counts;
}

Vector<float> MCTS::root_q_values() const noexcept {
  auto q_values = Vector<float>{num_moves_};
  q_values.setZero();
  for (const auto& c : root_.children) {
    q_values(c.move) = c.q;
  }
  return q_values;
}

Vector<float> MCTS::probs(const float temp) const noexcept {
  auto counts = this->counts();
  auto probs = Vector<float>{num_moves_};

  // When no visits have been made, return the prior policy from the root node.
  // This enables mcts_visits=1 as a "raw policy" mode.
  auto count_sum = counts.cast<float>().sum();
  if (count_sum == 0) {
    probs.setZero();
    for (const auto& c : root_.children) {
      probs(c.move) = c.policy;
    }
    if (temp != 0.0f) {
      probs = probs.array().pow(1.0f / temp);
    }
    probs /= probs.sum();
    return probs;
  }

  if (temp == 0) {
    auto best_moves = std::vector<int>{0};
    auto best_count = counts(0);
    for (auto m = 1; m < num_moves_; ++m) {
      if (counts(m) > best_count) {
        best_count = counts(m);
        best_moves.clear();
        best_moves.push_back(m);
      } else if (counts(m) == best_count) {
        best_moves.push_back(m);
      }
    }
    probs.setZero();
    for (auto m : best_moves) {
      probs(m) = 1.0 / best_moves.size();
    }
    return probs;
  }

  probs = counts.cast<float>();
  probs /= probs.sum();
  probs = probs.array().pow(1 / temp);
  probs /= probs.sum();
  return probs;
}

Vector<float> MCTS::probs_pruned(float temp) const noexcept {
  if (root_.n <= 1) return probs(temp);

  // PUCT inversion: compute the desired visit count for each child based on
  // what PUCT would select given the current Q values and noised policy.
  auto explore_scaling = cpuct_ * std::sqrt(static_cast<float>(root_.n));

  // Find best selection value among visited children.
  auto best_sel = -1e30f;
  for (const auto& c : root_.children) {
    if (c.n == 0) continue;
    auto sel = c.q + explore_scaling * c.policy / static_cast<float>(c.n + 1);
    if (sel > best_sel) best_sel = sel;
  }

  // Compute reduced visit counts via PUCT inversion.
  auto pruned = Vector<float>{num_moves_};
  pruned.setZero();
  for (const auto& c : root_.children) {
    if (c.n == 0) continue;
    auto explore_gap = best_sel - c.q;
    float desired;
    if (explore_gap <= 0) {
      // Q alone beats the best selection value — keep all visits.
      desired = static_cast<float>(c.n);
    } else {
      desired = explore_scaling * c.policy / explore_gap - 1.0f;
    }
    pruned(c.move) = std::min(static_cast<float>(c.n),
                               std::max(0.0f, desired));
  }

  // Handle temp==0 on raw counts before normalization.
  auto total = pruned.sum();
  if (total == 0) return probs(temp);
  if (temp == 0) {
    auto best_val = pruned.maxCoeff();
    auto result = Vector<float>{num_moves_};
    result.setZero();
    auto count = 0;
    for (auto m = 0; m < num_moves_; ++m) {
      if (pruned(m) == best_val) ++count;
    }
    for (auto m = 0; m < num_moves_; ++m) {
      if (pruned(m) == best_val) result(m) = 1.0f / count;
    }
    return result;
  }
  pruned /= total;
  if (temp != 1.0f) {
    pruned = pruned.array().pow(1.0f / temp);
    pruned /= pruned.sum();
  }
  return pruned;
}

uint32_t MCTS::pick_move(const Vector<float>& p) {
  std::uniform_real_distribution<float> dist{0.0F, 1.0F};
  auto choice = dist(re);
  auto sum = 0.0F;
  for (auto m = 0U; m < p.size(); ++m) {
    sum += p(m);
    if (sum > choice) {
      return m;
    }
  }
  // Due to floating point error we didn't pick a move.
  // Pick the last valid move.
  for (auto m = static_cast<int64_t>(p.size() - 1); m >= 0; --m) {
    if (p(m) > 0) {
      return m;
    }
  }
  throw std::runtime_error{"this shouldn't be possible."};
}

float MCTS::normalized_root_entropy() const noexcept {
  auto k = static_cast<float>(root_.children.size());
  if (k <= 1 || root_.n <= 1) return 0.0f;
  auto log_k = std::log(k);
  auto entropy = 0.0f;
  auto total_n = static_cast<float>(root_.n);
  for (const auto& c : root_.children) {
    if (c.n > 0) {
      auto p = static_cast<float>(c.n) / total_n;
      entropy -= p * std::log(p);
    }
  }
  return entropy / log_k;
}

}  // namespace alphazero
