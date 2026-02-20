#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "dll_export.h"
#include "game_state.h"
#include "shapes.h"

namespace alphazero {

struct DLLEXPORT Node {
  Node() = default;
  explicit Node(uint32_t m) : move(m){};

  float q = 0;
  float d = 0;
  float v = 0;
  float policy = 0;
  uint32_t move = 0;
  uint32_t n = 0;
  int8_t player = 0;
  std::optional<Vector<float>> scores = std::nullopt;
  std::vector<Node> children{};

  void add_children(const Vector<uint8_t>& valids) noexcept;
  void update_policy(const Vector<float>& pi) noexcept;
  [[nodiscard]] float uct(float sqrt_parent_n, float cpuct,
                          float fpu_value) const noexcept;
  [[nodiscard]] Node* best_child(float cpuct, float fpu_reduction) noexcept;
};

class DLLEXPORT MCTS {
 public:
  MCTS(float cpuct, uint32_t num_players, uint32_t num_moves, float epsilon = 0,
       float root_policy_temp = 1.0, float fpu_reduction = 0,
       bool relative_values = false, bool root_fpu_zero = false,
       bool shaped_dirichlet = false)
      : cpuct_(cpuct),
        num_players_(num_players),
        num_moves_(num_moves),
        current_(&root_),
        epsilon_(epsilon),
        root_policy_temp_(root_policy_temp),
        fpu_reduction_(fpu_reduction),
        relative_values_(relative_values),
        root_fpu_zero_(root_fpu_zero),
        shaped_dirichlet_(shaped_dirichlet) {}
  void update_root(const GameState& gs, uint32_t move);
  [[nodiscard]] std::unique_ptr<GameState> find_leaf(const GameState& gs);
  void process_result(const GameState& gs, Vector<float>& value,
                      Vector<float>& pi, bool root_noise_enabled = false);
  void add_root_noise();
  [[nodiscard]] Vector<float> root_value() const {
    float q = 0;
    float d = 0;
    bool found = false;
    for (const auto& c : root_.children) {
      if (c.n > 0 && c.q > q) {
        q = c.q;
        d = c.d;
        found = true;
      }
    }
    if (!found && root_.n > 0) {
      q = root_.v;
      d = root_.d;
    }
    auto w = q - d / num_players_;
    auto l = 1.0 - w - d;
    auto wld = Vector<float>{3};
    wld[0] = w;
    wld[1] = l;
    wld[2] = d;
    return wld;
  }
  [[nodiscard]] Vector<uint32_t> counts() const noexcept;
  [[nodiscard]] Vector<float> root_q_values() const noexcept;
  [[nodiscard]] Vector<float> probs(float temp) const noexcept;
  [[nodiscard]] Vector<float> probs_pruned(float temp) const noexcept;
  [[nodiscard]] uint32_t depth() const noexcept { return depth_; };
  [[nodiscard]] float avg_leaf_depth() const noexcept {
      return depth_ == 0 ? 0.0f : static_cast<float>(total_leaf_depth_) / static_cast<float>(depth_);
  };
  [[nodiscard]] float normalized_root_entropy() const noexcept;
  [[nodiscard]] uint32_t num_root_children() const noexcept { return root_.children.size(); }
  [[nodiscard]] uint32_t root_n() const noexcept { return root_.n; }
  void apply_root_policy_temp();
  [[nodiscard]] float epsilon() const noexcept { return epsilon_; }
  [[nodiscard]] float root_policy_temp() const noexcept { return root_policy_temp_; }
  [[nodiscard]] bool root_fpu_zero() const noexcept { return root_fpu_zero_; }

  [[nodiscard]] static uint32_t pick_move(const Vector<float>& p);

 private:
  float cpuct_;
  int32_t num_players_;
  int32_t num_moves_;

  uint32_t depth_ = 0;
  uint64_t total_leaf_depth_ = 0;
  Node root_ = Node{};
  Node* current_;
  std::vector<Node*> path_{};
  float epsilon_;
  float root_policy_temp_;
  float fpu_reduction_;
  bool relative_values_;
  bool root_fpu_zero_;
  bool shaped_dirichlet_;
};

}  // namespace alphazero