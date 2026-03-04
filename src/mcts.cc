#include "mcts.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <random>

#include "pcg/pcg_random.hpp"

namespace alphazero {

static bool node_on_path(uint32_t target,
                         const std::vector<std::pair<uint32_t, size_t>>& path,
                         uint32_t current) {
  if (target == current) return true;
  for (auto& [p_idx, _] : path) {
    if (p_idx == target) return true;
  }
  return false;
}

constexpr const float NOISE_ALPHA_RATIO = 10.83;
thread_local pcg32 re{pcg_extras::seed_seq_from<std::random_device>{}};

void Node::update_policy(const Vector<float>& pi) noexcept {
  for (size_t i = 0; i < child_indices.size(); ++i) {
    policies[i] = pi(moves[i]);
  }
}

float Node::uct(uint32_t edge_n, uint32_t edge_n_in_flight,
                float q, float sqrt_parent_n,
                float cpuct, float fpu_value, float policy) noexcept {
  return (edge_n == 0 ? fpu_value : q) +
         cpuct * policy * sqrt_parent_n /
             static_cast<float>(edge_n + edge_n_in_flight + 1);
}

uint32_t MCTS::allocate_node() {
  uint32_t idx;
  if (!free_list_.empty()) {
    idx = free_list_.back();
    free_list_.pop_back();
    nodes_[idx] = Node{};
  } else {
    nodes_.emplace_back();
    idx = static_cast<uint32_t>(nodes_.size() - 1);
  }
  nodes_[idx].v.assign(num_players_, 0.0f);
  nodes_[idx].cached_q.assign(num_players_, 0.0f);
  return idx;
}

void MCTS::release_node(uint32_t idx) {
  auto& node = nodes_[idx];
  if (node.ref_count == 0) return;
  --node.ref_count;
  if (node.ref_count > 0) return;
  auto children = std::move(node.child_indices);
  node = Node{};
  free_list_.push_back(idx);
  for (auto child_idx : children) {
    release_node(child_idx);
  }
}

void MCTS::sweep_tt() {
  std::vector<uint64_t> to_erase;
  for (auto& [hash, node_idx] : tt_) {
    if (nodes_[node_idx].ref_count == 0) {
      to_erase.push_back(hash);
    }
  }
  for (auto hash : to_erase) {
    tt_.erase(hash);
  }
}

void MCTS::recompute_cached_q(uint32_t node_idx) {
  auto& node = nodes_[node_idx];
  std::vector<float> q_sums(num_players_, 0.0f);
  float d_sum = 0;
  uint32_t n_total = 0;
  for (size_t i = 0; i < node.edge_n.size(); ++i) {
    if (node.edge_n[i] > 0) {
      auto& child = nodes_[node.child_indices[i]];
      for (int32_t p = 0; p < num_players_; ++p) {
        q_sums[p] += node.edge_n[i] * child.cached_q[p];
      }
      d_sum += node.edge_n[i] * child.cached_d;
      n_total += node.edge_n[i];
    }
  }
  auto denom = 1.0f + n_total;
  for (int32_t p = 0; p < num_players_; ++p) {
    node.cached_q[p] = (node.v[p] + q_sums[p]) / denom;
  }
  node.cached_d = (node.v_d + d_sum) / denom;
}

void MCTS::expand_node(uint32_t node_idx, const GameState& gs) {
  auto valids = gs.valid_moves();
  auto count = valids.sum();
  auto& node = nodes_[node_idx];
  node.child_indices.reserve(count);
  node.moves.reserve(count);
  node.policies.reserve(count);

  for (int w = 0; w < valids.size(); ++w) {
    if (valids(w) != 1) continue;
    uint32_t child_idx = allocate_node();
    nodes_[node_idx].child_indices.push_back(child_idx);
    nodes_[node_idx].moves.push_back(static_cast<uint32_t>(w));
    nodes_[node_idx].policies.push_back(0.0f);
    nodes_[child_idx].ref_count++;
  }
  // Shuffle all three arrays in sync
  auto& n = nodes_[node_idx];
  for (size_t i = n.child_indices.size(); i > 1; --i) {
    std::uniform_int_distribution<size_t> dist(0, i - 1);
    auto j = dist(re);
    std::swap(n.child_indices[i - 1], n.child_indices[j]);
    std::swap(n.moves[i - 1], n.moves[j]);
    std::swap(n.policies[i - 1], n.policies[j]);
  }
  // Resize edge arrays after shuffle
  n.edge_n.resize(n.child_indices.size(), 0);
  n.edge_n_in_flight.resize(n.child_indices.size(), 0);
}

size_t MCTS::best_child_idx(uint32_t node_idx, float cpuct,
                             float fpu_reduction) noexcept {
  auto& node = nodes_[node_idx];
  auto seen_policy = 0.0f;
  for (size_t i = 0; i < node.child_indices.size(); ++i) {
    if (node.edge_n[i] > 0) seen_policy += node.policies[i];
  }
  auto fpu_value = node.v[node.player] - fpu_reduction * std::sqrt(seen_policy);
  uint32_t total_n = 0;
  for (size_t i = 0; i < node.edge_n.size(); ++i) {
    total_n += node.edge_n[i] + node.edge_n_in_flight[i];
  }
  auto sqrt_n = std::sqrt(static_cast<float>(total_n + 1));
  size_t best_i = 0;
  auto best_uct = Node::uct(node.edge_n[0], node.edge_n_in_flight[0],
                              nodes_[node.child_indices[0]].cached_q[node.player],
                              sqrt_n, cpuct, fpu_value, node.policies[0]);
  for (size_t i = 1; i < node.child_indices.size(); ++i) {
    auto u = Node::uct(node.edge_n[i], node.edge_n_in_flight[i],
                        nodes_[node.child_indices[i]].cached_q[node.player],
                        sqrt_n, cpuct, fpu_value, node.policies[i]);
    if (u > best_uct) {
      best_uct = u;
      best_i = i;
    }
  }
  return best_i;
}

void MCTS::update_root(const GameState& gs, uint32_t move) {
  depth_ = 0;
  total_leaf_depth_ = 0;
  tt_hits_ = 0;
  auto& root = nodes_[root_idx_];
  if (root.child_indices.empty()) {
    expand_node(root_idx_, gs);
  }
  size_t idx = 0;
  bool found = false;
  for (size_t i = 0; i < nodes_[root_idx_].moves.size(); ++i) {
    if (nodes_[root_idx_].moves[i] == move) {
      idx = i;
      found = true;
      break;
    }
  }
  if (!found) {
    std::cout << gs.dump();
    throw std::runtime_error("ahh, what is this move: " +
                             std::to_string(move));
  }
  auto new_root_idx = nodes_[root_idx_].child_indices[idx];

  // Release non-chosen children via release_node
  for (size_t i = 0; i < nodes_[root_idx_].child_indices.size(); ++i) {
    if (i != idx) {
      release_node(nodes_[root_idx_].child_indices[i]);
    }
  }

  // Free old root — clear edges first to prevent cascading into new subtree
  // via cycle edges, then release properly via ref_count.
  nodes_[root_idx_].child_indices.clear();
  nodes_[root_idx_].moves.clear();
  nodes_[root_idx_].policies.clear();
  nodes_[root_idx_].edge_n.clear();
  nodes_[root_idx_].edge_n_in_flight.clear();
  nodes_[root_idx_].evaluated = false;
  nodes_[root_idx_].scores = std::nullopt;
  nodes_[root_idx_].cached_q.assign(num_players_, 0.0f);
  nodes_[root_idx_].cached_d = 0;
  nodes_[root_idx_].v.assign(num_players_, 0.0f);
  nodes_[root_idx_].v_d = 0;
  release_node(root_idx_);

  root_idx_ = new_root_idx;

  // Recompute root_sims_ from edge counts
  root_sims_ = 0;
  for (auto en : nodes_[root_idx_].edge_n) root_sims_ += en;

  if (mcgs_) {
    sweep_tt();
  }
}

void MCTS::add_root_noise() {
  auto& root = nodes_[root_idx_];
  const auto legal_move_count = root.child_indices.size();
  auto noise = Vector<float>{num_moves_};
  auto sum = 0.0;

  if (shaped_dirichlet_ && legal_move_count > 1) {
    auto N = static_cast<float>(legal_move_count);
    auto log_sum = 0.0f;
    for (size_t i = 0; i < root.child_indices.size(); ++i) {
      log_sum += std::log(std::min(root.policies[i], 0.01f) + 1e-20f);
    }
    auto log_mean = log_sum / N;

    auto shaped_sum = 0.0f;
    for (size_t i = 0; i < root.child_indices.size(); ++i) {
      auto lp = std::log(std::min(root.policies[i], 0.01f) + 1e-20f);
      shaped_sum += std::max(0.0f, lp - log_mean);
    }

    auto uniform = 1.0f / N;
    for (size_t i = 0; i < root.child_indices.size(); ++i) {
      auto lp = std::log(std::min(root.policies[i], 0.01f) + 1e-20f);
      auto shaped = std::max(0.0f, lp - log_mean);
      auto alpha_prop = (shaped_sum > 0)
                            ? 0.5f * (shaped / shaped_sum + uniform)
                            : uniform;
      alpha_prop = std::max(alpha_prop, 1e-6f);
      auto dist = std::gamma_distribution<float>{NOISE_ALPHA_RATIO * alpha_prop,
                                                  1.0f};
      noise(root.moves[i]) = dist(re);
      sum += noise(root.moves[i]);
    }
  } else {
    auto dist = std::gamma_distribution<float>{
        NOISE_ALPHA_RATIO / static_cast<float>(legal_move_count), 1.0};
    for (size_t i = 0; i < root.child_indices.size(); ++i) {
      noise(root.moves[i]) = dist(re);
      sum += noise(root.moves[i]);
    }
  }

  for (size_t i = 0; i < root.child_indices.size(); ++i) {
    root.policies[i] = root.policies[i] * (1 - epsilon_) +
                        epsilon_ * noise(root.moves[i]) / static_cast<float>(sum);
  }
}

void MCTS::apply_root_policy_temp() {
  if (root_policy_temp_ == 1.0f) return;
  auto& root = nodes_[root_idx_];
  float sum = 0.0f;
  for (size_t i = 0; i < root.child_indices.size(); ++i) {
    root.policies[i] = std::pow(root.policies[i], 1.0f / root_policy_temp_);
    sum += root.policies[i];
  }
  if (sum > 0.0f) {
    for (size_t i = 0; i < root.child_indices.size(); ++i) {
      root.policies[i] /= sum;
    }
  }
}

std::unique_ptr<GameState> MCTS::find_leaf(const GameState& gs) {
  current_idx_ = root_idx_;
  path_.clear();
  leaf_hash_ = 0;
  auto leaf = gs.copy();

  for (;;) {
    auto& cur = nodes_[current_idx_];

    // Case 1: terminal
    if (cur.scores.has_value()) {
      break;
    }

    // Case 2: evaluated with children — descend
    if (cur.evaluated && !cur.child_indices.empty()) {
      float fpu =
          (current_idx_ == root_idx_ && root_fpu_zero_) ? 0.0f : fpu_reduction_;
      auto idx = best_child_idx(current_idx_, cpuct_, fpu);
      auto child_idx = nodes_[current_idx_].child_indices[idx];
      if (mcgs_ && node_on_path(child_idx, path_, current_idx_)) break;
      path_.push_back({current_idx_, idx});
      leaf->play_move(nodes_[current_idx_].moves[idx]);
      current_idx_ = child_idx;
      continue;
    }

    // Case 3: unevaluated (no children)
    leaf_hash_ = hash_game_state(*leaf);

    if (mcgs_) {
      auto it = tt_.find(leaf_hash_);
      if (it != tt_.end() && it->second != current_idx_ && !path_.empty()) {
        auto tt_node_idx = it->second;
        bool cycle = node_on_path(tt_node_idx, path_, current_idx_);
        auto parent_idx = path_.back().first;
        auto edge_i = path_.back().second;
        nodes_[parent_idx].child_indices[edge_i] = tt_node_idx;
        ++nodes_[tt_node_idx].ref_count;
        release_node(current_idx_);
        current_idx_ = tt_node_idx;
        ++tt_hits_;
        if (cycle) break;
        continue;
      }
      tt_.emplace(leaf_hash_, current_idx_);
    }

    cur.player = leaf->current_player();
    cur.scores = leaf->scores();
    expand_node(current_idx_, *leaf);
    break;
  }

  total_leaf_depth_ += path_.size();

  // Ensure leaf_hash_ is set for terminal/already-visited cases
  if (leaf_hash_ == 0) {
    leaf_hash_ = hash_game_state(*leaf);
  }

  return leaf;
}

bool MCTS::leaf_needs_eval() const noexcept {
  auto& cur = nodes_[current_idx_];
  return !cur.evaluated && !cur.scores.has_value();
}

const std::vector<float>& MCTS::leaf_node_policies() const noexcept {
  return nodes_[current_idx_].policies;
}

void MCTS::backprop_leaf() {
  auto& cur = nodes_[current_idx_];

  // Terminal first-eval: set v/cached_q from scores
  if (!cur.evaluated && cur.scores.has_value()) {
    auto& scores = cur.scores.value();
    for (int32_t p = 0; p < num_players_; ++p) {
      auto pv = scores(p) + scores(num_players_) / num_players_;
      cur.v[p] = pv;
      cur.cached_q[p] = pv;
    }
    cur.v_d = scores(num_players_);
    cur.evaluated = true;
    cur.cached_d = scores(num_players_);
  }

  // Backprop: increment edge visits, recompute parent Q values
  for (auto it = path_.rbegin(); it != path_.rend(); ++it) {
    auto [parent_idx, edge_i] = *it;
    ++nodes_[parent_idx].edge_n[edge_i];
    recompute_cached_q(parent_idx);
  }

  ++depth_;
  ++root_sims_;
}

void MCTS::backprop_leaf_batched(uint32_t leaf_index) {
  auto& ifl = in_flight_.at(leaf_index);
  current_idx_ = ifl.leaf_idx;
  --nodes_[current_idx_].leaf_in_flight;

  auto& cur = nodes_[current_idx_];

  if (!cur.evaluated && cur.scores.has_value()) {
    auto& scores = cur.scores.value();
    for (int32_t p = 0; p < num_players_; ++p) {
      auto pv = scores(p) + scores(num_players_) / num_players_;
      cur.v[p] = pv;
      cur.cached_q[p] = pv;
    }
    cur.v_d = scores(num_players_);
    cur.evaluated = true;
    cur.cached_d = scores(num_players_);
  }

  for (auto it = ifl.path.rbegin(); it != ifl.path.rend(); ++it) {
    auto [parent_idx, edge_i] = *it;
    --nodes_[parent_idx].edge_n_in_flight[edge_i];
    ++nodes_[parent_idx].edge_n[edge_i];
    recompute_cached_q(parent_idx);
  }

  ++depth_;
  ++root_sims_;
}

void MCTS::process_result(const GameState& gs, Vector<float>& value,
                          Vector<float>& pi, bool root_noise_enabled) {
  auto& cur = nodes_[current_idx_];
  if (cur.scores.has_value()) {
    value = cur.scores.value();
  } else {
    // Rescale pi based on valid moves.
    auto valids = Vector<float>(gs.num_moves());
    valids.setZero();
    for (size_t i = 0; i < cur.child_indices.size(); ++i) {
      valids(cur.moves[i]) = 1;
    }
    pi.array() *= valids.array();
    pi /= pi.sum();
    if (current_idx_ == root_idx_) {
      pi = pi.array().pow(1.0f / root_policy_temp_);
      pi /= pi.sum();
      cur.update_policy(pi);
      if (root_noise_enabled) {
        add_root_noise();
      }
    } else if (!cur.evaluated) {
      cur.update_policy(pi);
    }
    if (relative_values_) {
      value = relative_to_absolute(value, cur.player, num_players_);
    }
  }

  // Set leaf values on first eval
  if (!nodes_[current_idx_].evaluated) {
    for (int32_t p = 0; p < num_players_; ++p) {
      auto pv = value(p) + value(num_players_) / num_players_;
      nodes_[current_idx_].v[p] = pv;
      nodes_[current_idx_].cached_q[p] = pv;
    }
    nodes_[current_idx_].v_d = value(num_players_);
    nodes_[current_idx_].evaluated = true;
    nodes_[current_idx_].cached_d = value(num_players_);
  }

  // Backprop: walk path from leaf toward root
  for (auto it = path_.rbegin(); it != path_.rend(); ++it) {
    auto [parent_idx, edge_i] = *it;
    ++nodes_[parent_idx].edge_n[edge_i];
    recompute_cached_q(parent_idx);
  }

  // Root first-eval
  auto& root = nodes_[root_idx_];
  if (!root.evaluated) {
    for (int32_t p = 0; p < num_players_; ++p) {
      root.v[p] = value(p) + value(num_players_) / num_players_;
      root.cached_q[p] = root.v[p];
    }
    root.v_d = value(num_players_);
    root.evaluated = true;
    root.cached_d = root.v_d;
  }
  ++depth_;
  ++root_sims_;
}

Vector<uint32_t> MCTS::counts() const noexcept {
  auto counts = Vector<uint32_t>{num_moves_};
  counts.setZero();
  auto& root = nodes_[root_idx_];
  for (size_t i = 0; i < root.child_indices.size(); ++i) {
    counts(root.moves[i]) = root.edge_n[i];
  }
  return counts;
}

Vector<float> MCTS::root_q_values() const noexcept {
  auto q_values = Vector<float>{num_moves_};
  q_values.setZero();
  auto& root = nodes_[root_idx_];
  for (size_t i = 0; i < root.child_indices.size(); ++i) {
    q_values(root.moves[i]) = nodes_[root.child_indices[i]].cached_q[root.player];
  }
  return q_values;
}

Vector<float> MCTS::root_value() const {
  auto& root = nodes_[root_idx_];
  float q = 0;
  float d = 0;
  bool found = false;
  for (size_t i = 0; i < root.child_indices.size(); ++i) {
    auto& child = nodes_[root.child_indices[i]];
    auto child_q = child.cached_q[root.player];
    if (root.edge_n[i] > 0 && child_q > q) {
      q = child_q;
      d = child.cached_d;
      found = true;
    }
  }
  if (!found && root.evaluated) {
    q = root.v[root.player];
    d = root.v_d;
  }
  auto w = q - d / num_players_;
  auto l = 1.0 - w - d;
  auto wld = Vector<float>{3};
  wld[0] = w;
  wld[1] = l;
  wld[2] = d;
  return wld;
}

Vector<float> MCTS::probs(const float temp) const noexcept {
  auto counts = this->counts();
  auto probs = Vector<float>{num_moves_};
  auto& root = nodes_[root_idx_];

  auto count_sum = counts.cast<float>().sum();
  if (count_sum == 0) {
    probs.setZero();
    for (size_t i = 0; i < root.child_indices.size(); ++i) {
      probs(root.moves[i]) = root.policies[i];
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
  auto& root = nodes_[root_idx_];
  if (root_sims_ <= 1) return probs(temp);

  auto explore_scaling = cpuct_ * std::sqrt(static_cast<float>(root_sims_));

  auto best_sel = -1e30f;
  for (size_t i = 0; i < root.child_indices.size(); ++i) {
    if (root.edge_n[i] == 0) continue;
    auto child_q = nodes_[root.child_indices[i]].cached_q[root.player];
    auto sel = child_q + explore_scaling * root.policies[i] /
                              static_cast<float>(root.edge_n[i] + 1);
    if (sel > best_sel) best_sel = sel;
  }

  auto pruned = Vector<float>{num_moves_};
  pruned.setZero();
  for (size_t i = 0; i < root.child_indices.size(); ++i) {
    if (root.edge_n[i] == 0) continue;
    auto child_q = nodes_[root.child_indices[i]].cached_q[root.player];
    auto explore_gap = best_sel - child_q;
    float desired;
    if (explore_gap <= 0) {
      desired = static_cast<float>(root.edge_n[i]);
    } else {
      desired = explore_scaling * root.policies[i] / explore_gap - 1.0f;
    }
    pruned(root.moves[i]) =
        std::min(static_cast<float>(root.edge_n[i]), std::max(0.0f, desired));
  }

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
  for (auto m = static_cast<int64_t>(p.size() - 1); m >= 0; --m) {
    if (p(m) > 0) {
      return m;
    }
  }
  throw std::runtime_error{"this shouldn't be possible."};
}

float MCTS::normalized_root_entropy() const noexcept {
  auto& root = nodes_[root_idx_];
  auto k = static_cast<float>(root.child_indices.size());
  if (k <= 1 || root_sims_ <= 1) return 0.0f;
  auto log_k = std::log(k);
  auto entropy = 0.0f;
  auto total_n = static_cast<float>(root_sims_);
  for (size_t i = 0; i < root.child_indices.size(); ++i) {
    auto cn = root.edge_n[i];
    if (cn > 0) {
      auto p = static_cast<float>(cn) / total_n;
      entropy -= p * std::log(p);
    }
  }
  return entropy / log_k;
}

std::unique_ptr<GameState> MCTS::find_leaf_batched(const GameState& gs) {
  InFlightLeaf ifl;
  auto cur_idx = root_idx_;
  auto leaf = gs.copy();
  leaf_hash_ = 0;

  for (;;) {
    auto& cur = nodes_[cur_idx];

    // Case 1: terminal
    if (cur.scores.has_value()) {
      break;
    }

    // Case 2: evaluated (or in-flight) with children — descend
    if ((cur.evaluated || cur.leaf_in_flight > 0) && !cur.child_indices.empty()) {
      float fpu =
          (cur_idx == root_idx_ && root_fpu_zero_) ? 0.0f : fpu_reduction_;
      auto idx = best_child_idx(cur_idx, cpuct_, fpu);
      auto child_idx = nodes_[cur_idx].child_indices[idx];
      if (mcgs_ && node_on_path(child_idx, ifl.path, cur_idx)) break;
      ifl.path.push_back({cur_idx, idx});
      ++nodes_[cur_idx].edge_n_in_flight[idx];
      leaf->play_move(nodes_[cur_idx].moves[idx]);
      cur_idx = child_idx;
      continue;
    }

    // Case 3: unevaluated (no children)
    leaf_hash_ = hash_game_state(*leaf);

    if (mcgs_) {
      auto it = tt_.find(leaf_hash_);
      if (it != tt_.end() && it->second != cur_idx && !ifl.path.empty()) {
        auto tt_node_idx = it->second;
        bool cycle = node_on_path(tt_node_idx, ifl.path, cur_idx);
        auto parent_idx = ifl.path.back().first;
        auto edge_i = ifl.path.back().second;
        nodes_[parent_idx].child_indices[edge_i] = tt_node_idx;
        ++nodes_[tt_node_idx].ref_count;
        release_node(cur_idx);
        cur_idx = tt_node_idx;
        ++tt_hits_;
        if (cycle) break;
        continue;
      }
      tt_.emplace(leaf_hash_, cur_idx);
    }

    nodes_[cur_idx].player = leaf->current_player();
    nodes_[cur_idx].scores = leaf->scores();
    expand_node(cur_idx, *leaf);
    break;
  }

  ++nodes_[cur_idx].leaf_in_flight;
  total_leaf_depth_ += ifl.path.size();

  // Ensure leaf_hash_ is set for terminal/already-visited cases
  if (leaf_hash_ == 0) {
    leaf_hash_ = hash_game_state(*leaf);
  }

  current_idx_ = cur_idx;
  ifl.leaf_idx = cur_idx;
  in_flight_.push_back(std::move(ifl));
  return leaf;
}

void MCTS::process_result_batched(const GameState& gs, uint32_t leaf_index,
                                  Vector<float>& value, Vector<float>& pi,
                                  bool root_noise_enabled) {
  auto& ifl = in_flight_.at(leaf_index);
  current_idx_ = ifl.leaf_idx;
  --nodes_[current_idx_].leaf_in_flight;

  auto& cur = nodes_[current_idx_];
  if (cur.scores.has_value()) {
    value = cur.scores.value();
  } else {
    auto valids = Vector<float>(gs.num_moves());
    valids.setZero();
    for (size_t i = 0; i < cur.child_indices.size(); ++i) {
      valids(cur.moves[i]) = 1;
    }
    pi.array() *= valids.array();
    pi /= pi.sum();
    if (current_idx_ == root_idx_) {
      pi = pi.array().pow(1.0f / root_policy_temp_);
      pi /= pi.sum();
      cur.update_policy(pi);
      if (root_noise_enabled) {
        add_root_noise();
      }
    } else if (!cur.evaluated) {
      cur.update_policy(pi);
    }
    if (relative_values_) {
      value = relative_to_absolute(value, cur.player, num_players_);
    }
  }

  // Set leaf values on first eval
  if (!nodes_[current_idx_].evaluated) {
    for (int32_t p = 0; p < num_players_; ++p) {
      auto pv = value(p) + value(num_players_) / num_players_;
      nodes_[current_idx_].v[p] = pv;
      nodes_[current_idx_].cached_q[p] = pv;
    }
    nodes_[current_idx_].v_d = value(num_players_);
    nodes_[current_idx_].evaluated = true;
    nodes_[current_idx_].cached_d = value(num_players_);
  }

  // Backprop walk
  for (auto it = ifl.path.rbegin(); it != ifl.path.rend(); ++it) {
    auto [parent_idx, edge_i] = *it;
    --nodes_[parent_idx].edge_n_in_flight[edge_i];
    ++nodes_[parent_idx].edge_n[edge_i];
    recompute_cached_q(parent_idx);
  }

  // Root first-eval
  auto& root = nodes_[root_idx_];
  if (!root.evaluated) {
    for (int32_t p = 0; p < num_players_; ++p) {
      root.v[p] = value(p) + value(num_players_) / num_players_;
      root.cached_q[p] = root.v[p];
    }
    root.v_d = value(num_players_);
    root.evaluated = true;
    root.cached_d = root.v_d;
  }
  ++depth_;
  ++root_sims_;
}

uint32_t MCTS::in_flight_count() const noexcept {
  return static_cast<uint32_t>(in_flight_.size());
}

void MCTS::reset_batch() noexcept { in_flight_.clear(); }

}  // namespace alphazero
