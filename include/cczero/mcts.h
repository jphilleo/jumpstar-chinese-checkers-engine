#pragma once

#include "cczero/cczero.h"
#include "cczero/model.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace cczero {

enum class MovegenBackend {
  Reference,
  Fast,
  Bitboard,
};

struct MctsChild {
  MoveEndpoint move;
  int action = 0;
  int child = -1;
  double prior = 0.0;
  int visits = 0;
  double value_sum = 0.0;
};

struct MctsNode {
  State state;
  uint64_t hash = 0;
  bool expanded = false;
  bool root_noise_applied = false;
  TerminalStatus status;
  std::vector<MctsChild> children;
  int visits = 0;
  double value_sum = 0.0;
  double leaf_value = 0.0;
  double eval_bias = 0.0;
  bool pending_eval = false;
};

struct MctsConfig {
  int simulations = 64;
  double cpuct = 1.4;
  bool add_root_noise = false;
  double root_dirichlet_alpha = 0.3;
  double root_noise_fraction = 0.25;
  double temperature = 0.0;
  double draw_leaf_value = 0.25;
  double anti_draw_logit_scale = 0.0;
  double progress_prior_scale = 0.0;
  double home_pressure_scale = 0.0;
  bool transpositions = true;
  bool reuse_tree = false;
  bool adaptive_simulations = false;
  int min_simulations = 0;
  int adaptive_check_interval = 8;
  double adaptive_confidence = 0.92;
  MovegenBackend movegen = MovegenBackend::Bitboard;
  InferenceBackend inference_backend = InferenceBackend::Auto;
  int inference_batch_size = 1;
  bool materialize_root_moves = true;
  bool profile_mcts = false;
};

struct MctsOverrides {
  std::optional<int> simulations;
  std::optional<double> cpuct;
  std::optional<double> temperature;
  std::optional<int> temperature_plies;
  std::optional<double> anti_draw_logit_scale;
  std::optional<double> progress_prior_scale;
  std::optional<double> home_pressure_scale;
  std::optional<MovegenBackend> movegen;
  std::optional<InferenceBackend> inference_backend;
  std::optional<int> inference_batch_size;
};

struct MctsStats {
  int simulations = 0;
  int nodes = 0;
  int evals = 0;
  int root_legal_moves = 0;
  int transposition_hits = 0;
  int inference_batches = 0;
  bool adaptive_stopped = false;
  bool reused_tree = false;
  double elapsed_ms = 0.0;
  double movegen_ms = 0.0;
  double eval_ms = 0.0;
  double policy_ms = 0.0;
  double select_ms = 0.0;
  double backup_ms = 0.0;
};

struct MctsRootMove {
  Move move;
  int visits = 0;
  double prior = 0.0;
  double value = 0.0;
};

struct MctsResult {
  Move move;
  std::vector<MctsRootMove> root_moves;
  MctsStats stats;
  std::vector<MctsNode> tree;
  int selected_child_node = -1;
};

class MctsSearchContext {
 public:
  struct Impl;

  MctsSearchContext();
  ~MctsSearchContext();
  MctsSearchContext(MctsSearchContext&&) noexcept;
  MctsSearchContext& operator=(MctsSearchContext&&) noexcept;
  MctsSearchContext(const MctsSearchContext&) = delete;
  MctsSearchContext& operator=(const MctsSearchContext&) = delete;

  void clear();

 private:
  std::unique_ptr<Impl> impl_;

  friend MctsResult run_mcts_search(const State& state, const Board& board,
                                    const RuleProfile& rules, const PolicyModel& model,
                                    const std::unordered_map<uint64_t, int>& repetition_counts,
                                    const MctsConfig& config, std::mt19937_64& rng,
                                    MctsSearchContext* context);
};

std::string movegen_backend_name(MovegenBackend backend);
MovegenBackend parse_movegen_backend(const std::string& text);
std::vector<Move> legal_moves_with_backend(const State& state, const Board& board,
                                           const RuleProfile& rules, MovegenBackend backend);

int pieces_in_home(const State& state, const Board& board, int player);
int policy_progress_score(const State& state, const Board& board, int player,
                          const Move& move);

MctsConfig default_eval_mcts_config(const State& state, const PolicyModel& model);
MctsResult run_mcts_search(const State& state, const Board& board, const RuleProfile& rules,
                           const PolicyModel& model,
                           const std::unordered_map<uint64_t, int>& repetition_counts,
                           const MctsConfig& config, std::mt19937_64& rng,
                           std::vector<MctsNode>* reusable_tree = nullptr,
                           int* reusable_root = nullptr);
MctsResult run_mcts_search(const State& state, const Board& board, const RuleProfile& rules,
                           const PolicyModel& model,
                           const std::unordered_map<uint64_t, int>& repetition_counts,
                           const MctsConfig& config, std::mt19937_64& rng,
                           MctsSearchContext* context);
Move choose_mcts_move(const State& state, const Board& board, const RuleProfile& rules,
                      const PolicyModel& model,
                      const std::unordered_map<uint64_t, int>& repetition_counts,
                      std::mt19937_64& rng, const MctsOverrides* overrides = nullptr);

}  // namespace cczero
