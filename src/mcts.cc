#include "mcts.h"

#include <algorithm>
#include <iostream>
#include <random>

namespace alphazero {

constexpr const float NOISE_ALPHA_RATIO = 10.83;

void Node::add_children(const Vector<uint8_t>& valids) noexcept {
  thread_local std::default_random_engine re{std::random_device{}()};
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
  if (root_.children.empty()) {
    root_.add_children(gs.valid_moves());
  }
  auto x = std::find_if(root_.children.begin(), root_.children.end(),
                        [move](const Node& n) { return n.move == move; });
  if (x == root_.children.end()) {
    std::cout << gs.dump();
    throw std::runtime_error("ahh, what is this move: " + std::to_string(move));
  }
  root_ = std::move(*x);
}

void MCTS::add_root_noise() {
  thread_local std::default_random_engine re{std::random_device{}()};
  const auto legal_move_count = root_.children.size();
  auto dist =
      std::gamma_distribution<float>{NOISE_ALPHA_RATIO / legal_move_count, 1.0};
  auto noise = Vector<float>{num_moves_};
  auto sum = 0.0;
  for (auto& c : root_.children) {
    noise(c.move) = dist(re);
    sum += noise(c.move);
  }
  for (auto& c : root_.children) {
    c.policy = c.policy * (1 - epsilon_) + epsilon_ * noise(c.move) / sum;
  }
}

std::unique_ptr<GameState> MCTS::find_leaf(const GameState& gs) {
  current_ = &root_;
  auto leaf = gs.copy();
  while (current_->n > 0 && !current_->scores.has_value()) {
    path_.push_back(current_);
    current_ = current_->best_child(cpuct_, fpu_reduction_);
    leaf->play_move(current_->move);
  }
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
      pi = pi.array().pow(1.0 / root_policy_temp_);
      pi /= pi.sum();
      current_->update_policy(pi);
      if (root_noise_enabled) {
        add_root_noise();
      }
    } else {
      current_->update_policy(pi);
    }
    // current_->update_fpu(value, num_players_, fpu_reduction_);
  }

  while (!path_.empty()) {
    auto* parent = path_.back();
    path_.pop_back();
    auto v = value(parent->player);
    // Add draws.
    v += value(num_players_) / num_players_;
    // Rescale to be from -1 to 1.
    v = v * 2 - 1;
    current_->q = (current_->q * static_cast<float>(current_->n) + v) /
                  static_cast<float>(current_->n + 1);
    if (current_->n == 0) {
      current_->v = v;
    }
    ++current_->n;
    current_ = parent;
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

Vector<float> MCTS::probs(const float temp) const noexcept {
  auto counts = this->counts();
  auto probs = Vector<float>{num_moves_};

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

uint32_t MCTS::pick_move(const Vector<float>& p) {
  thread_local std::default_random_engine re{std::random_device{}()};
  thread_local std::uniform_real_distribution<float> dist{0.0F, 1.0F};
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

}  // namespace alphazero
