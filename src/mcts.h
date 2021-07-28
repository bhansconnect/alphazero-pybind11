#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "game_state.h"
#include "shapes.h"

namespace alphazero {

struct Node {
  Node() = default;
  explicit Node(uint32_t m) : move(m){};

  float q = 0;
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

class MCTS {
 public:
  MCTS(float cpuct, uint32_t num_players, uint32_t num_moves, float epsilon = 0,
       float root_policy_temp = 1.4, float fpu_reduction = 0)
      : cpuct_(cpuct),
        num_players_(num_players),
        num_moves_(num_moves),
        current_(&root_),
        epsilon_(epsilon),
        root_policy_temp_(root_policy_temp),
        fpu_reduction_(fpu_reduction) {}
  void update_root(const GameState& gs, uint32_t move);
  [[nodiscard]] std::unique_ptr<GameState> find_leaf(const GameState& gs);
  void process_result(const GameState& gs, Vector<float>& value,
                      Vector<float>& pi, bool root_noise_enabled = false);
  void add_root_noise();
  [[nodiscard]] float root_value() const {
    float q = -1;
    for (const auto& c : root_.children) {
      if (c.n > 0 && c.q > q) {
        q = c.q;
      }
    }
    return (q + 1.0) / 2.0;
  }
  [[nodiscard]] Vector<uint32_t> counts() const noexcept;
  [[nodiscard]] Vector<float> probs(float temp) const noexcept;
  [[nodiscard]] uint32_t depth() const noexcept { return depth_; };

  [[nodiscard]] static uint32_t pick_move(const Vector<float>& p);

 private:
  float cpuct_;
  int32_t num_players_;
  int32_t num_moves_;

  uint32_t depth_ = 0;
  Node root_ = Node{};
  Node* current_;
  std::vector<Node*> path_{};
  float epsilon_;
  float root_policy_temp_;
  float fpu_reduction_;
};

}  // namespace alphazero