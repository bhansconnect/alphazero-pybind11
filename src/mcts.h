#pragma once

#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "dll_export.h"
#include "game_state.h"
#include "shapes.h"

namespace alphazero {

struct DLLEXPORT Node {
  Node() = default;

  std::vector<float> v{};   // [num_players] absolute: v[p] = value(p) + draw_share
  float v_d = 0;            // NN draw estimate (set once on first eval)
  std::vector<float> cached_q{};  // [num_players] recursive absolute Q
  float cached_d = 0;       // recursive draw estimate
  bool evaluated = false;   // replaces n > 0 for "has been backpropagated"
  uint32_t leaf_in_flight = 0;  // batched: in-flight evals targeting this leaf
  int8_t player = 0;
  uint16_t ref_count = 0;
  std::optional<Vector<float>> scores = std::nullopt;

  // Edge data — parallel arrays (per-parent-edge semantics)
  std::vector<uint32_t> child_indices{};  // indices into MCTS::nodes_
  std::vector<uint32_t> moves{};
  std::vector<float> policies{};
  std::vector<uint32_t> edge_n{};
  std::vector<uint32_t> edge_n_in_flight{};

  void update_policy(const Vector<float>& pi) noexcept;
  [[nodiscard]] static float uct(uint32_t edge_n, uint32_t edge_n_in_flight,
                                  float q, float sqrt_parent_n,
                                  float cpuct, float fpu_value,
                                  float policy) noexcept;
};

class DLLEXPORT MCTS {
 public:
  MCTS(float cpuct, uint32_t num_players, uint32_t num_moves, float epsilon = 0,
       float root_policy_temp = 1.0, float fpu_reduction = 0,
       bool relative_values = false, bool root_fpu_zero = false,
       bool shaped_dirichlet = false, bool mcgs = true)
      : cpuct_(cpuct),
        num_players_(num_players),
        num_moves_(num_moves),
        epsilon_(epsilon),
        root_policy_temp_(root_policy_temp),
        fpu_reduction_(fpu_reduction),
        relative_values_(relative_values),
        root_fpu_zero_(root_fpu_zero),
        shaped_dirichlet_(shaped_dirichlet),
        mcgs_(mcgs) {
    nodes_.emplace_back();  // root = index 0
    root_idx_ = 0;
    nodes_[0].ref_count = 1;  // root always has ref_count >= 1
    nodes_[0].v.assign(num_players_, 0.0f);
    nodes_[0].cached_q.assign(num_players_, 0.0f);
  }
  void update_root(const GameState& gs, uint32_t move);
  [[nodiscard]] std::unique_ptr<GameState> find_leaf(const GameState& gs);
  void process_result(const GameState& gs, Vector<float>& value,
                      Vector<float>& pi, bool root_noise_enabled = false);
  void add_root_noise();
  [[nodiscard]] Vector<float> root_value() const;
  [[nodiscard]] Vector<uint32_t> counts() const noexcept;
  [[nodiscard]] Vector<float> root_q_values() const noexcept;
  [[nodiscard]] Vector<float> probs(float temp) const noexcept;
  [[nodiscard]] Vector<float> probs_pruned(float temp) const noexcept;
  [[nodiscard]] uint32_t depth() const noexcept { return depth_; };
  [[nodiscard]] float avg_leaf_depth() const noexcept {
      return depth_ == 0 ? 0.0f : static_cast<float>(total_leaf_depth_) / static_cast<float>(depth_);
  };
  [[nodiscard]] float normalized_root_entropy() const noexcept;
  [[nodiscard]] uint32_t num_root_children() const noexcept {
    return nodes_[root_idx_].child_indices.size();
  }
  [[nodiscard]] uint32_t root_n() const noexcept { return root_sims_; }
  void apply_root_policy_temp();
  [[nodiscard]] float epsilon() const noexcept { return epsilon_; }
  [[nodiscard]] float root_policy_temp() const noexcept { return root_policy_temp_; }
  [[nodiscard]] bool root_fpu_zero() const noexcept { return root_fpu_zero_; }
  [[nodiscard]] uint32_t tt_hits() const noexcept { return tt_hits_; }
  [[nodiscard]] uint32_t tt_size() const noexcept { return tt_.size(); }
  [[nodiscard]] uint64_t leaf_hash() const noexcept { return leaf_hash_; }
  [[nodiscard]] bool leaf_needs_eval() const noexcept;
  [[nodiscard]] const std::vector<float>& leaf_node_policies() const noexcept;
  void backprop_leaf();
  void backprop_leaf_batched(uint32_t leaf_index);

  // Batched WU-UCT
  [[nodiscard]] std::unique_ptr<GameState> find_leaf_batched(const GameState& gs);
  void process_result_batched(const GameState& gs, uint32_t leaf_index,
                              Vector<float>& value, Vector<float>& pi,
                              bool root_noise_enabled = false);
  [[nodiscard]] uint32_t in_flight_count() const noexcept;
  void reset_batch() noexcept;

  [[nodiscard]] static uint32_t pick_move(const Vector<float>& p);

 private:
  struct InFlightLeaf {
    std::vector<std::pair<uint32_t, size_t>> path;  // (parent_idx, edge_idx)
    uint32_t leaf_idx;
  };

  uint32_t allocate_node();
  void expand_node(uint32_t node_idx, const GameState& gs);
  void release_node(uint32_t idx);
  void sweep_tt();
  void recompute_cached_q(uint32_t node_idx);
  [[nodiscard]] size_t best_child_idx(uint32_t node_idx, float cpuct,
                                       float fpu_reduction) noexcept;

  float cpuct_;
  int32_t num_players_;
  int32_t num_moves_;

  uint32_t depth_ = 0;
  uint64_t total_leaf_depth_ = 0;
  std::deque<Node> nodes_;
  std::vector<uint32_t> free_list_;
  uint32_t root_idx_ = 0;
  uint32_t root_sims_ = 0;
  uint32_t current_idx_ = 0;
  std::vector<std::pair<uint32_t, size_t>> path_{};  // (parent_idx, edge_idx)
  uint64_t leaf_hash_ = 0;
  float epsilon_;
  float root_policy_temp_;
  float fpu_reduction_;
  bool relative_values_;
  bool root_fpu_zero_;
  bool shaped_dirichlet_;
  bool mcgs_;
  absl::flat_hash_map<uint64_t, uint32_t> tt_;
  uint32_t tt_hits_ = 0;
  std::vector<InFlightLeaf> in_flight_;
};

}  // namespace alphazero
