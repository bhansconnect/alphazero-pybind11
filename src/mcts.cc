#include "mcts.h"

#include <algorithm>
#include <random>

namespace alphazero {

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

float Node::uct(float sqrt_parent_n, float cpuct) const noexcept {
  return q + cpuct * policy * sqrt_parent_n / static_cast<float>(n + 1);
}

Node* Node::best_child(float cpuct) noexcept {
  auto sqrt_n = std::sqrt(static_cast<float>(n));
  auto best_i = 0;
  auto best_uct = children.at(0).uct(sqrt_n, cpuct);
  for (auto i = 1; i < static_cast<int>(children.size()); ++i) {
    auto uct = children.at(i).uct(sqrt_n, cpuct);
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
    throw std::runtime_error("ahh, what is this move");
  }
  root_ = std::move(*x);
}

std::unique_ptr<GameState> MCTS::find_leaf(const GameState& gs) {
  current_ = &root_;
  auto leaf = gs.copy();
  while (current_->n > 0 && !current_->scores.has_value()) {
    path_.push_back(current_);
    current_ = current_->best_child(cpuct_);
    leaf->play_move(current_->move);
  }
  if (current_->n == 0) {
    current_->player = leaf->current_player();
    current_->scores = leaf->scores();
    current_->add_children(leaf->valid_moves());
  }
  return leaf;
}

void MCTS::process_result(Vector<float>& value, const Vector<float>& pi) {
  if (current_->scores.has_value()) {
    value = current_->scores.value();
  } else {
    current_->update_policy(pi);
  }
  while (!path_.empty()) {
    auto* parent = path_.back();
    path_.pop_back();
    auto v = value(parent->player);
    current_->q = (current_->q * static_cast<float>(current_->n) + v) /
                  static_cast<float>(current_->n + 1);
    ++current_->n;
    current_ = parent;
  }
  ++depth_;
  ++root_.n;
}

Vector<uint32_t> MCTS::counts() const noexcept {
  auto counts = Vector<uint32_t>{num_moves_};
  for (const auto& c : root_.children) {
    counts(c.move) = c.n;
  }
  return counts;
}

Vector<float> MCTS::probs(float temp) const noexcept {
  auto counts = this->counts();
  auto probs = Vector<float>{num_moves_};
  if (temp == 0) {
    auto best_m = 0;
    auto best_count = counts(0);
    for (auto m = 1; m < num_moves_; ++m) {
      if (counts(m) > best_count) {
        best_count = counts(m);
        best_m = m;
      }
    }
    probs(best_m) = 1;
    return probs;
  }

  auto sum = counts.sum();
  if (sum == 0) {
    auto best_m = 0;
    auto best_count = counts(0);
    for (auto m = 1; m < num_moves_; ++m) {
      if (counts(m) > best_count) {
        best_count = counts(m);
        best_m = m;
      }
    }
    probs(best_m) = 1;
    return probs;
  }
  probs = counts.cast<float>();
  probs.array().pow(1 / temp);
  probs /= sum;
  return probs;
}

std::tuple<Vector<float>, Vector<float>> dumb_eval(const GameState& gs) {
  auto valids = gs.valid_moves();
  auto values = Vector<float>{gs.num_players()};
  values.setZero();
  auto policy = Vector<float>{gs.num_moves()};
  float sum = valids.sum();
  if (sum == 0.0) {
    return {values, policy};
  }
  policy = valids.cast<float>() / sum;
  return {values, policy};
}

}  // namespace alphazero
