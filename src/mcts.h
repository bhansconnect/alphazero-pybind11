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
  uint32_t n_in_flight = 0;  // WU-UCT: pending evaluations through this node
  int8_t player = 0;
  // Terminal game result cached at terminal nodes (nullptr = non-terminal).
  // A unique_ptr (8B) rather than std::optional<Vector<float>> (24B) keeps the
  // Node small: the children arrays (~branching-factor wide) are the dominant
  // DRAM consumer in best_child/add_children/~Node, so every byte shaved off
  // Node reduces steady-state memory bandwidth. Terminal nodes are a minority,
  // so the extra indirection for them is a good trade.
  std::unique_ptr<Vector<float>> scores = nullptr;
  std::vector<Node> children{};

  void add_children(const Vector<uint8_t>& valids) noexcept;
  void update_policy(const Vector<float>& pi) noexcept;
  // Set child priors directly from pi indexed at each child's move, then
  // renormalize to sum 1. children already enumerate exactly the legal moves,
  // so this avoids building a dense num_moves-sized valids mask / masked-pi
  // array. When apply_temp is true each prior is first raised to inv_temp
  // (root-policy temperature); the pre-temp normalization cancels out and is
  // therefore skipped.
  void set_policy_normalized(const Vector<float>& pi, bool apply_temp,
                             float inv_temp) noexcept;
  [[nodiscard]] float uct(float sqrt_parent_n, float cpuct,
                          float fpu_value) const noexcept;
  [[nodiscard]] Node* best_child(float cpuct, float fpu_reduction) noexcept;
};

class DLLEXPORT MCTS {
 public:
  MCTS(float cpuct, uint32_t num_players, uint32_t num_moves, float epsilon = 0,
       float root_policy_temp = 1.0, float fpu_reduction = 0,
       bool relative_values = false, bool root_fpu_zero = false,
       bool shaped_dirichlet = false, bool gumbel_enabled = false,
       uint32_t gumbel_m = 16, float gumbel_c_visit = 50.0f,
       float gumbel_c_scale = 1.0f, bool gumbel_full = false)
      : cpuct_(cpuct),
        num_players_(num_players),
        num_moves_(num_moves),
        current_(&root_),
        epsilon_(epsilon),
        root_policy_temp_(root_policy_temp),
        fpu_reduction_(fpu_reduction),
        relative_values_(relative_values),
        root_fpu_zero_(root_fpu_zero),
        shaped_dirichlet_(shaped_dirichlet),
        gumbel_enabled_(gumbel_enabled),
        gumbel_m_(gumbel_m),
        gumbel_c_visit_(gumbel_c_visit),
        gumbel_c_scale_(gumbel_c_scale),
        gumbel_full_(gumbel_full) {}
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
  // Walk the tree from the root, taking the best (most-visited) child at each
  // level up to ``depth`` plies. For Gumbel-enabled roots the first ply uses
  // gumbel_final_action() instead of argmax(visits); deeper plies use argmax
  // because the Gumbel state is only meaningful at the root. Stops early when
  // a node has no visited children. Returns the move indices played.
  [[nodiscard]] Vector<uint32_t> principal_variation(uint32_t depth) const noexcept;
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

  // Batched WU-UCT
  [[nodiscard]] std::unique_ptr<GameState> find_leaf_batched(const GameState& gs);
  void process_result_batched(const GameState& gs, uint32_t leaf_index,
                              Vector<float>& value, Vector<float>& pi,
                              bool root_noise_enabled = false);
  [[nodiscard]] uint32_t in_flight_count() const noexcept;
  void reset_batch() noexcept;

  [[nodiscard]] static uint32_t pick_move(const Vector<float>& p);

  // ----- Gumbel AlphaZero (Danihelka 2022) --------------------------------
  // Caller must invoke set_gumbel_num_sims(n) BEFORE the first find_leaf of
  // each search when Gumbel is active. Re-init happens lazily inside
  // find_leaf once the root is expanded.
  void set_gumbel_num_sims(uint32_t n) noexcept;
  [[nodiscard]] bool gumbel_enabled() const noexcept { return gumbel_enabled_; }
  // Improved policy target (paper Eq 11), softmax(logits + sigma(completedQ)).
  // Returns a length-num_moves_ vector with mass only on legal moves.
  [[nodiscard]] Vector<float> gumbel_improved_policy() const noexcept;
  // The action A_{n+1} the paper recommends playing: argmax over the final
  // surviving candidates of (g + logits + sigma(q_hat)).
  [[nodiscard]] uint32_t gumbel_final_action() const noexcept;

  // Seed the thread-local RNG used for Dirichlet noise and Gumbel
  // perturbations. Test-only — production callers rely on the random_device
  // seed set at first use.
  static void seed_thread_rng(uint64_t seed) noexcept;

 private:
  struct InFlightLeaf {
    std::vector<Node*> path;
    Node* leaf;
  };
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
  std::vector<InFlightLeaf> in_flight_;

  // Gumbel AlphaZero state ---------------------------------------------------
  bool gumbel_enabled_;
  uint32_t gumbel_m_;
  float gumbel_c_visit_;
  float gumbel_c_scale_;
  bool gumbel_full_;
  // Per-search (reset by update_root and on set_gumbel_num_sims).
  bool gumbel_initialized_ = false;
  uint32_t gumbel_num_sims_target_ = 0;
  uint32_t gumbel_effective_m_ = 0;
  // gumbel_g_[i] = Gumbel(0) sample for root_.children[i] (legal-action
  // index, NOT move id). Length = root_.children.size() once initialized.
  std::vector<float> gumbel_g_;
  // Indices into root_.children of currently-surviving candidates.
  std::vector<size_t> gumbel_survivors_;
  // Phase plan: (num_candidates_in_phase, visits_per_candidate).
  std::vector<std::pair<uint32_t, uint32_t>> gumbel_phases_;
  uint32_t gumbel_phase_idx_ = 0;
  uint32_t gumbel_sims_in_phase_ = 0;

  void init_gumbel_state() noexcept;
  void reset_gumbel_state() noexcept;
  // Decide next root child index (into root_.children) per sequential halving.
  size_t gumbel_next_root_child() noexcept;
  // After phase complete, rank survivors by g+logit+sigma(q_hat) and keep top.
  void gumbel_advance_phase() noexcept;
  // pi'-matching interior selection (paper Eq 14). Returns child index.
  size_t gumbel_interior_select(Node& node) const noexcept;
};

}  // namespace alphazero