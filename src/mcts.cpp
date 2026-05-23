#include "cczero/mcts.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace cczero {

std::string movegen_backend_name(MovegenBackend backend) {
  switch (backend) {
    case MovegenBackend::Reference:
      return "reference";
    case MovegenBackend::Fast:
      return "fast";
    case MovegenBackend::Bitboard:
      return "bitboard";
  }
  return "unknown";
}

MovegenBackend parse_movegen_backend(const std::string& text) {
  if (text == "reference" || text == "ref" || text == "trusted") {
    return MovegenBackend::Reference;
  }
  if (text == "fast") {
    return MovegenBackend::Fast;
  }
  if (text == "bitboard" || text == "bitset" || text == "bits") {
    return MovegenBackend::Bitboard;
  }
  throw std::runtime_error("unknown movegen backend: " + text);
}

std::vector<Move> legal_moves_with_backend(const State& state, const Board& board,
                                           const RuleProfile& rules,
                                           MovegenBackend backend) {
  switch (backend) {
    case MovegenBackend::Reference:
      return legal_moves_reference(state, board, rules);
    case MovegenBackend::Fast:
      return legal_moves_fast(state, board, rules);
    case MovegenBackend::Bitboard:
      return legal_moves_bitboard(state, board, rules);
  }
  return legal_moves_reference(state, board, rules);
}

int pieces_in_home(const State& state, const Board& board, int player) {
  int count = 0;
  for (int id : board.home_cell_ids(player)) {
    if (state.cells.at(static_cast<size_t>(id)) == player) {
      ++count;
    }
  }
  return count;
}

namespace {

struct MaterializedMctsMoves {
  Move chosen;
  std::vector<MctsRootMove> root_moves;
};

struct PreparedLeaf {
  int node_index = -1;
  std::vector<std::pair<int, int>> path;
  bool needs_model_eval = false;
  double virtual_value = -1.0;
};

uint64_t mcts_node_hash(MctsNode& node);

std::vector<MoveEndpoint> legal_move_endpoints_with_backend(
    const State& state, const Board& board, const RuleProfile& rules, MovegenBackend backend) {
  if (backend == MovegenBackend::Bitboard) {
    return legal_move_endpoints_bitboard(state, board, rules);
  }
  const std::vector<Move> moves = legal_moves_with_backend(state, board, rules, backend);
  std::vector<MoveEndpoint> endpoints;
  endpoints.reserve(moves.size());
  for (const Move& move : moves) {
    endpoints.push_back(
        MoveEndpoint{move.from, move.to, static_cast<int>(std::max<size_t>(2, move.path.size()))});
  }
  return endpoints;
}

Move endpoint_to_move(const MoveEndpoint& endpoint) {
  return Move{endpoint.from, endpoint.to, {}};
}

int endpoint_action(const MoveEndpoint& endpoint) {
  return endpoint.from * kBoardSize + endpoint.to;
}

bool apply_endpoint(State& state, const MoveEndpoint& endpoint) {
  const int player = state.player_to_move;
  if (endpoint.from < 0 || endpoint.from >= kBoardSize || endpoint.to < 0 ||
      endpoint.to >= kBoardSize) {
    return false;
  }
  if (state.cells[static_cast<size_t>(endpoint.from)] != player ||
      state.cells[static_cast<size_t>(endpoint.to)] != kEmpty) {
    return false;
  }
  state.cells[static_cast<size_t>(endpoint.from)] = kEmpty;
  state.cells[static_cast<size_t>(endpoint.to)] = static_cast<int8_t>(player);
  state.player_to_move = 1 - state.player_to_move;
  ++state.ply;
  return true;
}

MaterializedMctsMoves materialize_mcts_moves(const State& state, const Board& board,
                                             const RuleProfile& rules,
                                             const MoveEndpoint& chosen,
                                             const std::vector<MctsChild>& children) {
  const std::vector<Move> legal = legal_moves_bitboard(state, board, rules);
  std::array<int, kBoardSize * kBoardSize> legal_index;
  legal_index.fill(-1);
  for (size_t i = 0; i < legal.size(); ++i) {
    const Move& move = legal[i];
    legal_index[static_cast<size_t>(move.from * kBoardSize + move.to)] =
        static_cast<int>(i);
  }
  MaterializedMctsMoves materialized;
  materialized.chosen = endpoint_to_move(chosen);
  materialized.root_moves.reserve(children.size());
  const int chosen_index = legal_index[static_cast<size_t>(endpoint_action(chosen))];
  if (chosen_index >= 0) {
    materialized.chosen = legal[static_cast<size_t>(chosen_index)];
  }
  for (const MctsChild& child : children) {
    Move full = endpoint_to_move(child.move);
    const int move_index = legal_index[static_cast<size_t>(child.action)];
    if (move_index >= 0) {
      full = legal[static_cast<size_t>(move_index)];
    }
    const double q = child.visits == 0 ? 0.0 : child.value_sum / child.visits;
    materialized.root_moves.push_back(MctsRootMove{std::move(full), child.visits, child.prior, q});
  }
  return materialized;
}

Move materialize_chosen_move(const State& state, const Board& board, const RuleProfile& rules,
                             const MoveEndpoint& chosen) {
  const int chosen_action = endpoint_action(chosen);
  for (const Move& move : legal_moves_bitboard(state, board, rules)) {
    if (move.from * kBoardSize + move.to == chosen_action) {
      return move;
    }
  }
  return endpoint_to_move(chosen);
}

std::vector<MctsRootMove> endpoint_root_moves(const std::vector<MctsChild>& children) {
  std::vector<MctsRootMove> root_moves;
  root_moves.reserve(children.size());
  for (const MctsChild& child : children) {
    const double q = child.visits == 0 ? 0.0 : child.value_sum / child.visits;
    root_moves.push_back(MctsRootMove{endpoint_to_move(child.move), child.visits, child.prior, q});
  }
  return root_moves;
}

double policy_progress_score_endpoint_with_home(const State& state, const Board& board, int player,
                                                int from, int to, int path_length,
                                                int home_before,
                                                double home_pressure_scale = 1.0) {
  const int distance_gain = goal_distance_for_cell(board, player, from) -
                            goal_distance_for_cell(board, player, to);
  const bool from_goal = board.is_goal(player, from);
  const bool to_goal = board.is_goal(player, to);
  const bool from_home = board.is_home(player, from);
  const bool to_home = board.is_home(player, to);
  const int goal_gain = (to_goal ? 1 : 0) - (from_goal ? 1 : 0);
  const int home_after = home_before - (from_home ? 1 : 0) + (to_home ? 1 : 0);
  const bool leaves_home = from_home && !to_home;
  const bool returns_home = !from_home && to_home;
  const bool home_shuffle = from_home && to_home;

  double score = 95 * distance_gain + 520 * goal_gain;
  double home_score = 0.0;
  if (leaves_home) {
    home_score += 260;
  }
  if (returns_home) {
    home_score -= 420;
  }
  if (state.ply < 80 && home_before >= 7) {
    if (leaves_home) {
      home_score += 760 + 70 * home_before;
    } else if (home_shuffle) {
      home_score -= 900;
    } else if (home_after >= home_before) {
      home_score -= 430 + 45 * home_before;
    }
  } else if (state.ply < 140 && home_before >= 4) {
    if (leaves_home) {
      home_score += 420 + 45 * home_before;
    } else if (home_after >= home_before && goal_gain <= 0) {
      home_score -= 220 + 35 * home_before;
    }
  }
  if (home_after > home_before) {
    home_score -= 1200;
  }
  score += home_pressure_scale * home_score;
  if (!from_goal && to_goal) {
    score += 700;
  }
  if (from_goal && to_goal) {
    score += 80;
  }
  if (distance_gain < 0) {
    score += 220 * distance_gain;
  }
  if (state.ply >= 40 && distance_gain <= 0 && goal_gain <= 0) {
    score -= 180;
  }
  score += path_length * 8;
  return score;
}

int policy_progress_score_endpoint(const State& state, const Board& board, int player,
                                   int from, int to, int path_length) {
  return static_cast<int>(std::lround(policy_progress_score_endpoint_with_home(
      state, board, player, from, to, path_length, pieces_in_home(state, board, player))));
}

int anti_draw_progress_score(const State& state, const Board& board, const RuleProfile& rules,
                             int player, int from, int to) {
  const int remaining = std::max(0, rules.max_plies - state.ply);
  if (state.ply < 90 && remaining > 120) {
    return 0;
  }

  const int distance_gain = goal_distance_for_cell(board, player, from) -
                            goal_distance_for_cell(board, player, to);
  const int goal_gain = (board.is_goal(player, to) ? 1 : 0) -
                        (board.is_goal(player, from) ? 1 : 0);
  const int urgency = state.ply >= 160 || remaining <= 60 ? 2 : 1;
  int score = 0;
  if (goal_gain > 0) {
    score += urgency * 900;
  }
  if (distance_gain > 0) {
    score += urgency * 70 * distance_gain;
  }
  if (distance_gain <= 0 && goal_gain <= 0) {
    score -= urgency * (360 + std::max(0, 90 - remaining) * 6);
  }
  if (distance_gain < 0) {
    score += urgency * 240 * distance_gain;
  }
  if (remaining <= 30 && goal_gain <= 0 && distance_gain <= 1) {
    score -= 900;
  }
  return score;
}

double repetition_arrival_bias(uint64_t state_hash, int ply, const RuleProfile& rules,
                               const std::unordered_map<uint64_t, int>* repetition_counts,
                               int root_ply, bool* terminal_repetition) {
  if (terminal_repetition != nullptr) {
    *terminal_repetition = false;
  }
  if (!rules.repetition_draw || repetition_counts == nullptr || ply <= root_ply) {
    return 0.0;
  }
  const auto found = repetition_counts->find(state_hash);
  if (found == repetition_counts->end()) {
    return 0.0;
  }
  const int arrival_count = found->second + 1;
  if (arrival_count >= rules.repetition_count) {
    if (terminal_repetition != nullptr) {
      *terminal_repetition = true;
    }
    return 0.35;
  }
  return 0.12 * static_cast<double>(found->second);
}

bool prepare_mcts_node(MctsNode& node, const Board& board, const RuleProfile& rules,
                       const MctsConfig& config,
                       const std::unordered_map<uint64_t, int>* repetition_counts,
                       int root_ply, MctsStats* stats) {
  if (node.expanded || node.pending_eval) {
    return false;
  }
  bool terminal_repetition = false;
  const double repetition_bias =
      repetition_arrival_bias(mcts_node_hash(node), node.state.ply, rules, repetition_counts,
                              root_ply, &terminal_repetition);
  node.eval_bias = repetition_bias;
  if (terminal_repetition) {
    node.status = TerminalStatus{true, kInvalid, true, "search_repetition"};
    node.expanded = true;
    node.leaf_value = repetition_bias;
    return false;
  }
  node.status = terminal_status(node.state, board, rules, nullptr);
  if (node.status.terminal) {
    node.expanded = true;
    if (!node.status.draw && node.status.winner != kInvalid) {
      node.leaf_value = node.status.winner == node.state.player_to_move ? 1.0 : -1.0;
    } else if (node.status.draw) {
      node.leaf_value = std::clamp(config.draw_leaf_value, 0.0, 1.0);
    }
    return false;
  }

  std::chrono::steady_clock::time_point movegen_start;
  if (config.profile_mcts) {
    movegen_start = std::chrono::steady_clock::now();
  }
  const std::vector<MoveEndpoint> moves =
      legal_move_endpoints_with_backend(node.state, board, rules, config.movegen);
  if (stats != nullptr && config.profile_mcts) {
    const auto movegen_end = std::chrono::steady_clock::now();
    stats->movegen_ms +=
        std::chrono::duration<double, std::milli>(movegen_end - movegen_start).count();
  }
  if (moves.empty()) {
    node.status = TerminalStatus{true, 1 - node.state.player_to_move, false, "no_legal_moves"};
    node.expanded = true;
    node.leaf_value = -1.0;
    return false;
  }

  node.children.clear();
  node.children.reserve(moves.size());
  for (const MoveEndpoint& move : moves) {
    MctsChild child;
    child.move = move;
    child.action = endpoint_action(move);
    node.children.push_back(child);
  }
  node.pending_eval = true;
  return true;
}

void assign_policy_priors(MctsNode& node, const Board& board, const RuleProfile& rules,
                          const PolicyModel& model, const MctsConfig& config,
                          const float* hidden, const float* policy_state, MctsStats* stats) {
  double max_logit = -std::numeric_limits<double>::infinity();
  const int player = node.state.player_to_move;
  const int home_before = pieces_in_home(node.state, board, player);
  const int remaining = std::max(0, rules.max_plies - node.state.ply);
  const bool anti_draw_active =
      config.anti_draw_logit_scale != 0.0 && !(node.state.ply < 90 && remaining > 120);
  std::chrono::steady_clock::time_point policy_start;
  if (config.profile_mcts) {
    policy_start = std::chrono::steady_clock::now();
  }
  for (MctsChild& child : node.children) {
    const MoveEndpoint& move = child.move;
    const double raw_policy =
        model.kind == ModelKind::PolicyValueMlp
            ? static_cast<double>(
                  policy_state != nullptr
                      ? mlp_policy_logit_action_projected_ptr(model, hidden, policy_state,
                                                              child.action, player)
                      : mlp_policy_logit_action_ptr(model, hidden, child.action, player))
            : static_cast<double>(
                  policy_score_action(model, node.state, player, move.from, move.to));
    double logit = raw_policy;
    if (config.progress_prior_scale != 0.0) {
      logit += config.progress_prior_scale *
               policy_progress_score_endpoint_with_home(
                   node.state, board, player, move.from, move.to, move.path_length, home_before,
                   config.home_pressure_scale);
    }
    if (anti_draw_active) {
      logit += config.anti_draw_logit_scale *
               anti_draw_progress_score(node.state, board, rules, player, move.from, move.to);
    }
    child.prior = logit;
    max_logit = std::max(max_logit, logit);
  }
  if (stats != nullptr && config.profile_mcts) {
    const auto policy_end = std::chrono::steady_clock::now();
    stats->policy_ms +=
        std::chrono::duration<double, std::milli>(policy_end - policy_start).count();
  }

  double denom = 0.0;
  for (MctsChild& child : node.children) {
    child.prior = std::exp(std::clamp(child.prior - max_logit, -60.0, 60.0));
    denom += child.prior;
  }
  if (denom <= 0.0 || !std::isfinite(denom)) {
    denom = static_cast<double>(node.children.size());
    for (MctsChild& child : node.children) {
      child.prior = 1.0 / denom;
    }
  } else {
    for (MctsChild& child : node.children) {
      child.prior /= denom;
    }
  }
}

void finish_prepared_mcts_node(MctsNode& node, const Board& board, const RuleProfile& rules,
                               const PolicyModel& model, const MctsConfig& config,
                               MlpWorkspace& workspace, MctsStats* stats) {
  if (!node.pending_eval) {
    return;
  }
  const float* hidden = nullptr;
  std::chrono::steady_clock::time_point eval_start;
  if (config.profile_mcts) {
    eval_start = std::chrono::steady_clock::now();
  }
  if (model.kind == ModelKind::PolicyValueMlp) {
    mlp_hidden_optimized(model, node.state, node.state.player_to_move, workspace,
                         config.inference_backend);
    hidden = workspace.hidden.data();
    node.leaf_value = mlp_value_from_hidden_ptr(model, hidden);
    if (model.policy_head == PolicyHeadKind::MoveMlp ||
        model.policy_head == PolicyHeadKind::MoveBilinear) {
      workspace.ensure_policy_state(static_cast<size_t>(model.move_hidden_size));
      mlp_policy_state_projection(model, hidden, workspace.policy_state.data());
    }
  } else {
    node.leaf_value =
        std::tanh(static_cast<double>(
                      evaluate_state(node.state, board, rules, node.state.player_to_move)) /
                  1400.0);
  }
  if (stats != nullptr && config.profile_mcts) {
    const auto eval_end = std::chrono::steady_clock::now();
    stats->eval_ms +=
        std::chrono::duration<double, std::milli>(eval_end - eval_start).count();
  }
  node.leaf_value = std::clamp(node.leaf_value + node.eval_bias, -1.0, 1.0);
  assign_policy_priors(
      node, board, rules, model, config, hidden,
      (model.policy_head == PolicyHeadKind::MoveMlp ||
       model.policy_head == PolicyHeadKind::MoveBilinear)
          ? workspace.policy_state.data()
          : nullptr,
      stats);
  node.pending_eval = false;
  node.expanded = true;
}

void finish_prepared_mcts_nodes(std::vector<MctsNode>& tree, const std::vector<int>& node_indices,
                                const Board& board, const RuleProfile& rules,
                                const PolicyModel& model, const MctsConfig& config,
                                MlpWorkspace& workspace,
                                std::vector<const State*>& batch_states,
                                MctsStats* stats) {
  if (node_indices.empty()) {
    return;
  }
  if (model.kind != ModelKind::PolicyValueMlp || node_indices.size() == 1) {
    for (int node_index : node_indices) {
      finish_prepared_mcts_node(tree[static_cast<size_t>(node_index)], board, rules, model,
                                config, workspace, stats);
    }
    return;
  }

  std::chrono::steady_clock::time_point eval_start;
  if (config.profile_mcts) {
    eval_start = std::chrono::steady_clock::now();
  }
  batch_states.clear();
  batch_states.reserve(node_indices.size());
  for (int node_index : node_indices) {
    batch_states.push_back(&tree.at(static_cast<size_t>(node_index)).state);
  }
  mlp_hidden_batch_optimized(model, batch_states, workspace, config.inference_backend);
  if (stats != nullptr) {
    if (config.profile_mcts) {
      const auto eval_end = std::chrono::steady_clock::now();
      stats->eval_ms +=
          std::chrono::duration<double, std::milli>(eval_end - eval_start).count();
    }
    ++stats->inference_batches;
  }

  const size_t hidden_size = static_cast<size_t>(model.hidden_size);
  for (size_t row = 0; row < node_indices.size(); ++row) {
    MctsNode& node = tree[static_cast<size_t>(node_indices[row])];
    node.leaf_value =
        std::clamp(static_cast<double>(workspace.values[row]) + node.eval_bias, -1.0, 1.0);
    float* policy_state = nullptr;
    if (model.policy_head == PolicyHeadKind::MoveMlp ||
        model.policy_head == PolicyHeadKind::MoveBilinear) {
      workspace.ensure_policy_state(node_indices.size() *
                                    static_cast<size_t>(model.move_hidden_size));
      policy_state =
          workspace.policy_state.data() + row * static_cast<size_t>(model.move_hidden_size);
      mlp_policy_state_projection(model, workspace.hidden.data() + row * hidden_size,
                                  policy_state);
    }
    assign_policy_priors(node, board, rules, model, config,
                         workspace.hidden.data() + row * hidden_size, policy_state, stats);
    node.pending_eval = false;
    node.expanded = true;
  }
}

void expand_mcts_node(MctsNode& node, const Board& board, const RuleProfile& rules,
                      const PolicyModel& model, const MctsConfig& config,
                      const std::unordered_map<uint64_t, int>* repetition_counts,
                      int root_ply, MlpWorkspace& workspace, MctsStats* stats) {
  if (!prepare_mcts_node(node, board, rules, config, repetition_counts, root_ply, stats)) {
    return;
  }
  finish_prepared_mcts_node(node, board, rules, model, config, workspace, stats);
}

void add_dirichlet_root_noise(MctsNode& root, const MctsConfig& config, std::mt19937_64& rng) {
  if (!config.add_root_noise || root.children.empty() || root.root_noise_applied) {
    return;
  }
  std::gamma_distribution<double> gamma(std::max(1.0e-6, config.root_dirichlet_alpha), 1.0);
  std::vector<double> noise;
  noise.reserve(root.children.size());
  double total = 0.0;
  for (size_t i = 0; i < root.children.size(); ++i) {
    const double value = gamma(rng);
    noise.push_back(value);
    total += value;
  }
  if (total <= 0.0 || !std::isfinite(total)) {
    return;
  }
  const double frac = std::clamp(config.root_noise_fraction, 0.0, 1.0);
  for (size_t i = 0; i < root.children.size(); ++i) {
    root.children.at(i).prior =
        (1.0 - frac) * root.children.at(i).prior + frac * noise.at(i) / total;
  }
  root.root_noise_applied = true;
}

void apply_repetition_root_prior_penalty(
    MctsNode& root, const State& root_state, const RuleProfile& rules,
    const std::unordered_map<uint64_t, int>& repetition_counts) {
  if (!rules.repetition_draw || repetition_counts.empty() || root.children.empty()) {
    return;
  }
  double total = 0.0;
  for (MctsChild& child : root.children) {
    State next = root_state;
    apply_endpoint(next, child.move);
    const auto found = repetition_counts.find(next.hash());
    if (found != repetition_counts.end()) {
      const int arrival_count = found->second + 1;
      child.prior *= arrival_count >= rules.repetition_count ? 0.01 : 0.12;
    }
    total += child.prior;
  }
  if (total <= 0.0 || !std::isfinite(total)) {
    const double uniform = 1.0 / static_cast<double>(root.children.size());
    for (MctsChild& child : root.children) {
      child.prior = uniform;
    }
    return;
  }
  for (MctsChild& child : root.children) {
    child.prior /= total;
  }
}

int select_mcts_child(const std::vector<MctsNode>& tree, const MctsNode& node, double cpuct,
                      bool skip_pending) {
  int best = 0;
  double best_score = -std::numeric_limits<double>::infinity();
  bool found = false;
  const double parent_sqrt = std::sqrt(static_cast<double>(std::max(1, node.visits)));
  for (size_t i = 0; i < node.children.size(); ++i) {
    const MctsChild& child = node.children[i];
    if (skip_pending && child.child >= 0 && tree[static_cast<size_t>(child.child)].pending_eval) {
      continue;
    }
    const double q = child.visits == 0 ? 0.0 : child.value_sum / child.visits;
    const double u = cpuct * child.prior * parent_sqrt / (1.0 + child.visits);
    const double score = q + u;
    if (score > best_score) {
      best_score = score;
      best = static_cast<int>(i);
      found = true;
    }
  }
  return found ? best : -1;
}

uint64_t mcts_transposition_key(uint64_t state_hash, int ply) {
  return state_hash ^ (static_cast<uint64_t>(ply) * 0x9e3779b97f4a7c15ULL);
}

uint64_t mcts_node_hash(MctsNode& node) {
  if (node.hash == 0) {
    node.hash = node.state.hash();
  }
  return node.hash;
}

int ensure_mcts_child_node(std::vector<MctsNode>& tree, int node_index, int child_index,
                           const MctsConfig& config, MctsStats& stats,
                           std::unordered_map<uint64_t, int>& transpositions) {
  MctsNode& node = tree[static_cast<size_t>(node_index)];
  MctsChild& edge = node.children[static_cast<size_t>(child_index)];
  if (edge.child != -1) {
    return edge.child;
  }
  State next = node.state;
  apply_endpoint(next, edge.move);
  const uint64_t next_state_hash = next.hash();
  const uint64_t next_key = mcts_transposition_key(next_state_hash, next.ply);
  const auto found = config.transpositions ? transpositions.find(next_key) : transpositions.end();
  if (found != transpositions.end()) {
    edge.child = found->second;
    ++stats.transposition_hits;
  } else {
    edge.child = static_cast<int>(tree.size());
    MctsNode child;
    child.state = next;
    child.hash = next_state_hash;
    if (config.transpositions) {
      transpositions.emplace(next_key, edge.child);
    }
    tree.push_back(child);
  }
  return edge.child;
}

double run_mcts_simulation(std::vector<MctsNode>& tree, int node_index, const Board& board,
                           const RuleProfile& rules, const PolicyModel& model,
                           const MctsConfig& config, MctsStats& stats,
                           std::vector<std::pair<int, int>>& path,
                           const std::unordered_map<uint64_t, int>& repetition_counts,
                           std::unordered_map<uint64_t, int>& transpositions, int root_ply,
                           MlpWorkspace& workspace) {
  MctsNode& node = tree[static_cast<size_t>(node_index)];
  const bool was_expanded = node.expanded;
  expand_mcts_node(node, board, rules, model, config, &repetition_counts, root_ply, workspace,
                   &stats);
  if (!was_expanded && node.expanded) {
    ++stats.evals;
    return node.leaf_value;
  }
  if (node.status.terminal || node.children.empty()) {
    return node.leaf_value;
  }

  std::chrono::steady_clock::time_point select_start;
  if (config.profile_mcts) {
    select_start = std::chrono::steady_clock::now();
  }
  const int child_index = select_mcts_child(tree, node, config.cpuct, false);
  if (child_index < 0) {
    return node.leaf_value;
  }
  if (config.profile_mcts) {
    const auto select_end = std::chrono::steady_clock::now();
    stats.select_ms +=
        std::chrono::duration<double, std::milli>(select_end - select_start).count();
  }
  const int child_node =
      ensure_mcts_child_node(tree, node_index, child_index, config, stats, transpositions);

  path.push_back({node_index, child_index});
  return -run_mcts_simulation(tree, child_node, board, rules, model, config, stats, path,
                              repetition_counts, transpositions, root_ply, workspace);
}

int sample_root_child(const std::vector<MctsChild>& children, double temperature,
                      std::mt19937_64& rng) {
  if (children.empty()) {
    return -1;
  }
  if (temperature <= 1.0e-6) {
    int best = 0;
    for (size_t i = 1; i < children.size(); ++i) {
      if (children.at(i).visits > children.at(static_cast<size_t>(best)).visits) {
        best = static_cast<int>(i);
      }
    }
    return best;
  }

  std::vector<double> weights;
  weights.reserve(children.size());
  double total = 0.0;
  const double inv_temp = 1.0 / temperature;
  for (const MctsChild& child : children) {
    const double weight = std::pow(static_cast<double>(std::max(0, child.visits)), inv_temp);
    weights.push_back(weight);
    total += weight;
  }
  if (total <= 0.0 || !std::isfinite(total)) {
    std::uniform_int_distribution<int> dist(0, static_cast<int>(children.size() - 1));
    return dist(rng);
  }
  std::discrete_distribution<int> dist(weights.begin(), weights.end());
  return dist(rng);
}

bool root_visit_confident(const std::vector<MctsChild>& children, double confidence) {
  int total = 0;
  int best = 0;
  for (const MctsChild& child : children) {
    const int visits = std::max(0, child.visits);
    total += visits;
    best = std::max(best, visits);
  }
  if (total <= 0) {
    return false;
  }
  return static_cast<double>(best) / static_cast<double>(total) >= confidence;
}

#ifndef NDEBUG
void validate_mcts_tree_invariants(const std::vector<MctsNode>& tree) {
  for (size_t node_index = 0; node_index < tree.size(); ++node_index) {
    const MctsNode& node = tree[node_index];
    if (node.pending_eval) {
      throw std::logic_error("MCTS invariant failed: pending eval after search");
    }
    if (!node.children.empty()) {
      double prior_sum = 0.0;
      for (const MctsChild& child : node.children) {
        if (!child.move.is_valid()) {
          throw std::logic_error("MCTS invariant failed: invalid child move");
        }
        if (child.action != endpoint_action(child.move)) {
          throw std::logic_error("MCTS invariant failed: stale child action id");
        }
        if (child.child >= static_cast<int>(tree.size())) {
          throw std::logic_error("MCTS invariant failed: child index out of range");
        }
        if (!std::isfinite(child.prior) || child.prior < 0.0) {
          throw std::logic_error("MCTS invariant failed: invalid child prior");
        }
        prior_sum += child.prior;
      }
      if (node.expanded && std::abs(prior_sum - 1.0) > 1.0e-4) {
        throw std::logic_error("MCTS invariant failed: child priors do not sum to one");
      }
    }
  }
}
#endif

void add_virtual_visits(std::vector<MctsNode>& tree, int root_index,
                        const std::vector<std::pair<int, int>>& path) {
  tree[static_cast<size_t>(root_index)].visits += 1;
  for (const auto& [node_index, child_index] : path) {
    MctsNode& node = tree[static_cast<size_t>(node_index)];
    MctsChild& edge = node.children[static_cast<size_t>(child_index)];
    edge.visits += 1;
    if (edge.child >= 0) {
      tree[static_cast<size_t>(edge.child)].visits += 1;
    }
  }
}

void add_value_to_virtual_path(std::vector<MctsNode>& tree, int root_index,
                               const std::vector<std::pair<int, int>>& path, double value) {
  double backed = value;
  tree[static_cast<size_t>(root_index)].value_sum += backed;
  for (const auto& [node_index, child_index] : path) {
    MctsNode& node = tree[static_cast<size_t>(node_index)];
    MctsChild& edge = node.children[static_cast<size_t>(child_index)];
    edge.value_sum += backed;
    backed = -backed;
    if (edge.child >= 0) {
      tree[static_cast<size_t>(edge.child)].value_sum += backed;
    }
  }
}

bool collect_mcts_leaf(std::vector<MctsNode>& tree, int root_index, const Board& board,
                       const RuleProfile& rules, const MctsConfig& config, MctsStats& stats,
                       const std::unordered_map<uint64_t, int>& repetition_counts,
                       std::unordered_map<uint64_t, int>& transpositions, int root_ply,
                       PreparedLeaf& leaf) {
  int node_index = root_index;
  leaf.node_index = -1;
  leaf.needs_model_eval = false;
  leaf.virtual_value = -1.0;
  leaf.path.clear();
  while (true) {
    MctsNode& node = tree[static_cast<size_t>(node_index)];
    if (!node.expanded) {
      leaf.node_index = node_index;
      leaf.needs_model_eval =
          prepare_mcts_node(node, board, rules, config, &repetition_counts, root_ply, &stats);
      ++stats.evals;
      add_virtual_visits(tree, root_index, leaf.path);
      add_value_to_virtual_path(tree, root_index, leaf.path, leaf.virtual_value);
      return true;
    }
    if (node.status.terminal || node.children.empty()) {
      leaf.node_index = node_index;
      leaf.needs_model_eval = false;
      add_virtual_visits(tree, root_index, leaf.path);
      add_value_to_virtual_path(tree, root_index, leaf.path, leaf.virtual_value);
      return true;
    }

    std::chrono::steady_clock::time_point select_start;
    if (config.profile_mcts) {
      select_start = std::chrono::steady_clock::now();
    }
    const int child_index = select_mcts_child(tree, node, config.cpuct, true);
    if (config.profile_mcts) {
      const auto select_end = std::chrono::steady_clock::now();
      stats.select_ms +=
          std::chrono::duration<double, std::milli>(select_end - select_start).count();
    }
    if (child_index < 0) {
      return false;
    }
    const int child_node =
        ensure_mcts_child_node(tree, node_index, child_index, config, stats, transpositions);
    leaf.path.push_back({node_index, child_index});
    node_index = child_node;
  }
}

void run_mcts_batch(std::vector<MctsNode>& tree, int root_index, const Board& board,
                    const RuleProfile& rules, const PolicyModel& model,
                    const MctsConfig& config, MctsStats& stats,
                    const std::unordered_map<uint64_t, int>& repetition_counts,
                    std::unordered_map<uint64_t, int>& transpositions, int root_ply,
                    MlpWorkspace& workspace, int batch_size,
                    std::vector<PreparedLeaf>& leaves, std::vector<int>& eval_nodes,
                    std::vector<const State*>& batch_states) {
  if (leaves.size() < static_cast<size_t>(batch_size)) {
    leaves.resize(static_cast<size_t>(batch_size));
    for (PreparedLeaf& leaf : leaves) {
      leaf.path.reserve(128);
    }
  }
  eval_nodes.clear();

  int leaf_count = 0;
  for (int i = 0; i < batch_size; ++i) {
    PreparedLeaf& leaf = leaves[static_cast<size_t>(i)];
    if (!collect_mcts_leaf(tree, root_index, board, rules, config, stats, repetition_counts,
                           transpositions, root_ply, leaf)) {
      break;
    }
    if (leaf.needs_model_eval) {
      eval_nodes.push_back(leaf.node_index);
    }
    ++leaf_count;
  }

  finish_prepared_mcts_nodes(tree, eval_nodes, board, rules, model, config, workspace,
                             batch_states, &stats);

  std::chrono::steady_clock::time_point backup_start;
  if (config.profile_mcts) {
    backup_start = std::chrono::steady_clock::now();
  }
  for (int i = 0; i < leaf_count; ++i) {
    const PreparedLeaf& leaf = leaves[static_cast<size_t>(i)];
    const double leaf_value = tree[static_cast<size_t>(leaf.node_index)].leaf_value;
    const double value = (leaf.path.size() % 2 == 0) ? leaf_value : -leaf_value;
    add_value_to_virtual_path(tree, root_index, leaf.path, value - leaf.virtual_value);
    ++stats.simulations;
  }
  if (config.profile_mcts) {
    const auto backup_end = std::chrono::steady_clock::now();
    stats.backup_ms +=
        std::chrono::duration<double, std::milli>(backup_end - backup_start).count();
  }
}

}  // namespace

struct MctsSearchContext::Impl {
  MlpWorkspace mlp;
  std::vector<MctsNode> working_tree;
  std::vector<MctsNode> reusable_tree;
  int reusable_root = -1;
  std::unordered_map<uint64_t, int> transpositions;
  std::vector<PreparedLeaf> batch_leaves;
  std::vector<int> batch_eval_nodes;
  std::vector<std::pair<int, int>> single_path;
  std::vector<const State*> batch_states;
};

MctsSearchContext::MctsSearchContext() : impl_(std::make_unique<Impl>()) {}
MctsSearchContext::~MctsSearchContext() = default;
MctsSearchContext::MctsSearchContext(MctsSearchContext&&) noexcept = default;
MctsSearchContext& MctsSearchContext::operator=(MctsSearchContext&&) noexcept = default;

void MctsSearchContext::clear() {
  impl_->working_tree.clear();
  impl_->reusable_tree.clear();
  impl_->reusable_root = -1;
  impl_->transpositions.clear();
  impl_->batch_leaves.clear();
  impl_->batch_eval_nodes.clear();
  impl_->single_path.clear();
  impl_->batch_states.clear();
}

int policy_progress_score(const State& state, const Board& board, int player, const Move& move) {
  return policy_progress_score_endpoint(state, board, player, move.from, move.to,
                                        static_cast<int>(move.path.size()));
}

namespace {

MctsResult run_mcts_search_impl(const State& state, const Board& board,
                                const RuleProfile& rules, const PolicyModel& model,
                                const std::unordered_map<uint64_t, int>& repetition_counts,
                                const MctsConfig& config, std::mt19937_64& rng,
                                MctsSearchContext::Impl* context,
                                std::vector<MctsNode>* reusable_tree, int* reusable_root,
                                bool context_manages_reuse) {
  const auto start = std::chrono::steady_clock::now();
  MctsStats stats;
  MlpWorkspace local_workspace;
  MlpWorkspace& workspace = context == nullptr ? local_workspace : context->mlp;
  std::vector<MctsNode> local_tree;
  std::vector<MctsNode>& tree = context == nullptr ? local_tree : context->working_tree;
  tree.clear();
  tree.reserve(static_cast<size_t>(std::max(256, config.simulations + 16)));
  if (context_manages_reuse && context != nullptr) {
    reusable_tree = &context->reusable_tree;
    reusable_root = &context->reusable_root;
  }
  int root_index = 0;
  if (config.reuse_tree && reusable_tree != nullptr && reusable_root != nullptr &&
      *reusable_root >= 0 &&
      static_cast<size_t>(*reusable_root) < reusable_tree->size()) {
    MctsNode& candidate = reusable_tree->at(static_cast<size_t>(*reusable_root));
    const uint64_t state_hash = state.hash();
    if (mcts_node_hash(candidate) == state_hash && candidate.state.ply == state.ply &&
        candidate.state.player_to_move == state.player_to_move) {
      tree = std::move(*reusable_tree);
      root_index = *reusable_root;
      stats.reused_tree = true;
    }
  }
  if (tree.empty()) {
    MctsNode root;
    root.state = state;
    root.hash = state.hash();
    tree.push_back(root);
    root_index = 0;
  }
  std::unordered_map<uint64_t, int> local_transpositions;
  std::unordered_map<uint64_t, int>& transpositions =
      context == nullptr ? local_transpositions : context->transpositions;
  transpositions.clear();
  if (config.transpositions) {
    transpositions.reserve(static_cast<size_t>(std::max(256, config.simulations + 16)));
    for (size_t index = 0; index < tree.size(); ++index) {
      MctsNode& node = tree.at(index);
      transpositions.emplace(mcts_transposition_key(mcts_node_hash(node), node.state.ply),
                             static_cast<int>(index));
    }
  }
  expand_mcts_node(tree.at(static_cast<size_t>(root_index)), board, rules, model, config,
                   &repetition_counts, state.ply, workspace, &stats);
  add_dirichlet_root_noise(tree.at(static_cast<size_t>(root_index)), config, rng);
  apply_repetition_root_prior_penalty(tree.at(static_cast<size_t>(root_index)), state, rules,
                                      repetition_counts);
  if (tree.at(static_cast<size_t>(root_index)).children.empty()) {
    return MctsResult{};
  }

  stats.root_legal_moves =
      static_cast<int>(tree.at(static_cast<size_t>(root_index)).children.size());
  const int simulations = std::max(1, config.simulations);
  const int min_simulations =
      config.min_simulations > 0 ? std::min(config.min_simulations, simulations) : simulations;
  const int adaptive_interval = std::max(1, config.adaptive_check_interval);
  const int batch_size = std::max(1, config.inference_batch_size);
  std::vector<PreparedLeaf> local_batch_leaves;
  std::vector<int> local_batch_eval_nodes;
  std::vector<std::pair<int, int>> local_single_path;
  std::vector<const State*> local_batch_states;
  std::vector<PreparedLeaf>& batch_leaves =
      context == nullptr ? local_batch_leaves : context->batch_leaves;
  std::vector<int>& batch_eval_nodes =
      context == nullptr ? local_batch_eval_nodes : context->batch_eval_nodes;
  std::vector<std::pair<int, int>>& single_path =
      context == nullptr ? local_single_path : context->single_path;
  std::vector<const State*>& batch_states =
      context == nullptr ? local_batch_states : context->batch_states;
  if (batch_size > 1) {
    batch_leaves.resize(static_cast<size_t>(batch_size));
    for (PreparedLeaf& leaf : batch_leaves) {
      leaf.path.reserve(128);
    }
    batch_eval_nodes.reserve(static_cast<size_t>(batch_size));
  } else {
    single_path.reserve(static_cast<size_t>(std::min(simulations, 256)));
  }
  while (stats.simulations < simulations) {
    if (batch_size <= 1) {
      single_path.clear();
      const double value =
          run_mcts_simulation(tree, root_index, board, rules, model, config, stats, single_path,
                              repetition_counts, transpositions, state.ply, workspace);
      double backed = value;
      ++stats.simulations;
      std::chrono::steady_clock::time_point backup_start;
      if (config.profile_mcts) {
        backup_start = std::chrono::steady_clock::now();
      }
      MctsNode& root = tree[static_cast<size_t>(root_index)];
      root.visits += 1;
      root.value_sum += backed;
      for (auto it = single_path.begin(); it != single_path.end(); ++it) {
        MctsNode& node = tree[static_cast<size_t>(it->first)];
        MctsChild& edge = node.children[static_cast<size_t>(it->second)];
        edge.visits += 1;
        edge.value_sum += backed;
        backed = -backed;
        if (edge.child >= 0) {
          MctsNode& child = tree[static_cast<size_t>(edge.child)];
          child.visits += 1;
          child.value_sum += backed;
        }
      }
      if (config.profile_mcts) {
        const auto backup_end = std::chrono::steady_clock::now();
        stats.backup_ms +=
            std::chrono::duration<double, std::milli>(backup_end - backup_start).count();
      }
    } else {
      const int before_batch = stats.simulations;
      run_mcts_batch(tree, root_index, board, rules, model, config, stats, repetition_counts,
                     transpositions, state.ply, workspace,
                     std::min(batch_size, simulations - stats.simulations), batch_leaves,
                     batch_eval_nodes, batch_states);
      if (stats.simulations == before_batch) {
        break;
      }
    }
    if (config.adaptive_simulations && stats.simulations >= min_simulations &&
        ((stats.simulations - min_simulations) % adaptive_interval == 0) &&
        root_visit_confident(tree.at(static_cast<size_t>(root_index)).children,
                             config.adaptive_confidence)) {
      stats.adaptive_stopped = stats.simulations < simulations;
      break;
    }
  }

  const int sampled_child =
      sample_root_child(tree.at(static_cast<size_t>(root_index)).children, config.temperature, rng);
  int selected_child_index = sampled_child;
  int best_score = std::numeric_limits<int>::min();
  MoveEndpoint best =
      tree.at(static_cast<size_t>(root_index)).children.at(static_cast<size_t>(sampled_child)).move;
  if (config.temperature <= 1.0e-6) {
    int best_visits = -1;
    const std::vector<MctsChild>& root_children =
        tree.at(static_cast<size_t>(root_index)).children;
    const int player = state.player_to_move;
    const int home_before = pieces_in_home(state, board, player);
    for (size_t child_index = 0; child_index < root_children.size(); ++child_index) {
      const MctsChild& child = root_children.at(child_index);
      State next = state;
      apply_endpoint(next, child.move);
      int score = child.visits * 1000 +
                  policy_progress_score_endpoint_with_home(
                      state, board, player, child.move.from, child.move.to,
                      child.move.path_length, home_before);
      const auto repeated = repetition_counts.find(next.hash());
      if (repeated != repetition_counts.end()) {
        score -= 200000 * repeated->second;
      }
      if (child.visits > best_visits || (child.visits == best_visits && score > best_score)) {
        best_visits = child.visits;
        best_score = score;
        best = child.move;
        selected_child_index = static_cast<int>(child_index);
      }
    }
  }

#ifndef NDEBUG
  validate_mcts_tree_invariants(tree);
#endif

  MctsResult result;
  const std::vector<MctsChild>& root_children = tree.at(static_cast<size_t>(root_index)).children;
  if (config.materialize_root_moves) {
    MaterializedMctsMoves materialized =
        materialize_mcts_moves(state, board, rules, best, root_children);
    result.move = std::move(materialized.chosen);
    result.root_moves = std::move(materialized.root_moves);
  } else {
    result.move = materialize_chosen_move(state, board, rules, best);
    result.root_moves = endpoint_root_moves(root_children);
  }
  stats.nodes = static_cast<int>(tree.size());
  const auto end = std::chrono::steady_clock::now();
  stats.elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
  result.stats = stats;
  if (selected_child_index >= 0 && static_cast<size_t>(selected_child_index) < root_children.size()) {
    result.selected_child_node = root_children.at(static_cast<size_t>(selected_child_index)).child;
  }
  if (context_manages_reuse && context != nullptr) {
    if (config.reuse_tree && result.selected_child_node >= 0) {
      context->reusable_root = result.selected_child_node;
      context->reusable_tree = std::move(tree);
    } else {
      context->reusable_root = -1;
      context->reusable_tree.clear();
    }
  } else if (config.reuse_tree) {
    result.tree = std::move(tree);
  }
  return result;
}

}  // namespace

MctsResult run_mcts_search(const State& state, const Board& board, const RuleProfile& rules,
                           const PolicyModel& model,
                           const std::unordered_map<uint64_t, int>& repetition_counts,
                           const MctsConfig& config, std::mt19937_64& rng,
                           std::vector<MctsNode>* reusable_tree, int* reusable_root) {
  return run_mcts_search_impl(state, board, rules, model, repetition_counts, config, rng, nullptr,
                              reusable_tree, reusable_root, false);
}

MctsResult run_mcts_search(const State& state, const Board& board, const RuleProfile& rules,
                           const PolicyModel& model,
                           const std::unordered_map<uint64_t, int>& repetition_counts,
                           const MctsConfig& config, std::mt19937_64& rng,
                           MctsSearchContext* context) {
  return run_mcts_search_impl(state, board, rules, model, repetition_counts, config, rng,
                              context == nullptr ? nullptr : context->impl_.get(), nullptr,
                              nullptr, context != nullptr);
}

MctsConfig default_eval_mcts_config(const State& state, const PolicyModel& model) {
  MctsConfig config;
  config.simulations =
      model.kind == ModelKind::PolicyValueMlp ? (state.ply < 40 ? 16 : 28)
                                              : (state.ply < 40 ? 48 : 80);
  config.cpuct = 1.4;
  config.add_root_noise = false;
  config.temperature = 0.0;
  config.inference_batch_size = model.kind == ModelKind::PolicyValueMlp ? 64 : 1;
  return config;
}

Move choose_mcts_move(const State& state, const Board& board, const RuleProfile& rules,
                      const PolicyModel& model,
                      const std::unordered_map<uint64_t, int>& repetition_counts,
                      std::mt19937_64& rng, const MctsOverrides* overrides) {
  MctsConfig config = default_eval_mcts_config(state, model);
  if (overrides != nullptr) {
    if (overrides->simulations) {
      config.simulations = *overrides->simulations;
    }
    if (overrides->cpuct) {
      config.cpuct = *overrides->cpuct;
    }
    if (overrides->temperature) {
      const int temperature_plies = overrides->temperature_plies.value_or(std::numeric_limits<int>::max());
      config.temperature = state.ply < temperature_plies ? *overrides->temperature : 0.0;
    }
    if (overrides->anti_draw_logit_scale) {
      config.anti_draw_logit_scale = *overrides->anti_draw_logit_scale;
    }
    if (overrides->progress_prior_scale) {
      config.progress_prior_scale = *overrides->progress_prior_scale;
    }
    if (overrides->home_pressure_scale) {
      config.home_pressure_scale = *overrides->home_pressure_scale;
    }
    if (overrides->movegen) {
      config.movegen = *overrides->movegen;
    }
    if (overrides->inference_backend) {
      config.inference_backend = *overrides->inference_backend;
    }
    if (overrides->inference_batch_size) {
      config.inference_batch_size = *overrides->inference_batch_size;
    }
  }
  config.materialize_root_moves = false;
  return run_mcts_search(state, board, rules, model, repetition_counts, config, rng).move;
}

}  // namespace cczero
