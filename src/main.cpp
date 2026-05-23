#include "cczero/cczero.h"
#include "cczero/cli.h"
#include "cczero/cli_utils.h"
#include "cczero/model.h"
#include "cczero/mcts.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cctype>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

struct DatasetRecord {
  int game_id = 0;
  int ply = 0;
  int player = 0;
  cczero::BotKind bot = cczero::BotKind::Random;
  uint64_t hash = 0;
  std::string cells;
  std::vector<cczero::Move> legal_moves;
  cczero::Move chosen;
  int distance_before = 0;
  int distance_after = 0;
  int opponent_distance_before = 0;
  int opponent_distance_after = 0;
  int goal_count_before = 0;
  int goal_count_after = 0;
  int progress_delta = 0;
  int phase = 0;
};

struct MatchSummary {
  cczero::TerminalStatus status;
  int plies = 0;
};

struct SuitePosition {
  std::string id;
  std::string cells;
  int player = 0;
  int ply = 0;
};

struct SuiteTask {
  int game_id = 0;
  int position_index = 0;
  bool swapped = false;
  uint64_t seed = 0;
  cczero::State state;
};

struct SuiteOutput {
  int game_id = 0;
  std::string log;
  MatchSummary summary;
};

struct ScoreRow {
  int games = 0;
  double points = 0.0;
  int wins = 0;
  int draws = 0;
  int losses = 0;
};

using cczero::InferenceBackend;
using cczero::MlpWorkspace;
using cczero::PolicyModel;
using cczero::PolicyHeadKind;
using cczero::ModelKind;
using cczero::accelerate_compiled;
using cczero::MctsConfig;
using cczero::MctsNode;
using cczero::MctsOverrides;
using cczero::MctsResult;
using cczero::MctsRootMove;
using cczero::MctsSearchContext;
using cczero::MctsStats;
using cczero::MovegenBackend;
using cczero::choose_mcts_move;
using cczero::inference_backend_name;
using cczero::legal_moves_with_backend;
using cczero::load_policy_model;
using cczero::mlp_hidden_optimized;
using cczero::mlp_policy_logit_action_projected_ptr;
using cczero::mlp_policy_logit_action_ptr;
using cczero::mlp_policy_state_projection;
using cczero::model_kind_name;
using cczero::policy_head_kind_name;
using cczero::movegen_backend_name;
using cczero::open_output_stream;
using cczero::parse_bot_list;
using cczero::parse_inference_backend;
using cczero::parse_movegen_backend;
using cczero::parse_movegen_backend_list;
using cczero::parse_rule_profile;
using cczero::pieces_in_home;
using cczero::policy_score;
using cczero::policy_progress_score;
using cczero::policy_model_parameter_count;
using cczero::policy_model_storage_bytes;
using cczero::require_arg_value;
using cczero::resolve_inference_backend;
using cczero::run_mcts_search;

struct SelfplayRecord {
  int game_id = 0;
  uint64_t seed = 0;
  int ply = 0;
  int player = 0;
  uint64_t hash = 0;
  std::string cells;
  cczero::Move chosen;
  std::vector<MctsRootMove> root_moves;
  MctsStats stats;
  int distance_before = 0;
  int opponent_distance_before = 0;
  int goal_count_before = 0;
  int opponent_goal_count_before = 0;
  int home_count_before = 0;
  int opponent_home_count_before = 0;
  int goal_blockers_before = 0;
  int opponent_goal_blockers_before = 0;
  int phase = 0;
  bool has_score_margin = false;
  int finish_margin_moves = 0;
  int finish_margin_max_moves = 0;
  bool finish_margin_capped = false;
  double score_margin = 0.0;
};

struct SelfplayGameResult {
  std::vector<SelfplayRecord> records;
  cczero::TerminalStatus status;
  int plies = 0;
  uint64_t nodes = 0;
  uint64_t evals = 0;
  uint64_t simulations = 0;
  uint64_t root_legal_moves = 0;
  uint64_t transposition_hits = 0;
  uint64_t adaptive_stops = 0;
  uint64_t reuse_hits = 0;
  uint64_t inference_batches = 0;
  double search_ms = 0.0;
  double movegen_ms = 0.0;
  double eval_ms = 0.0;
  double policy_ms = 0.0;
};

struct SearchTotals {
  uint64_t nodes = 0;
  uint64_t evals = 0;
  uint64_t simulations = 0;
  uint64_t root_legal_moves = 0;
  uint64_t transposition_hits = 0;
  uint64_t adaptive_stops = 0;
  uint64_t reuse_hits = 0;
  uint64_t inference_batches = 0;
  double search_ms = 0.0;
  double movegen_ms = 0.0;
  double eval_ms = 0.0;
  double policy_ms = 0.0;
};

struct ReanalysisInput {
  int game_id = 0;
  uint64_t record_seed = 0;
  int player = 0;
  int ply = 0;
  int result_value = 0;
  int simulation_budget = 0;
  bool has_score_margin = false;
  int finish_margin_moves = 0;
  int finish_margin_max_moves = 0;
  bool finish_margin_capped = false;
  double score_margin = 0.0;
  cczero::State state;
};

struct FinishMarginInfo {
  bool available = false;
  int moves = 0;
  int max_moves = 0;
  bool capped = false;
  double normalized = 0.0;
};

constexpr int kFinishMarginNormalizeMoves = 80;
constexpr int kFinishMarginMaxMoves = 160;
constexpr const char* kScoreMarginSource = "finish_margin_greedy_cleanup_v1";

struct ReanalysisOutput {
  bool written = false;
  std::string skip_reason;
  std::string json;
  MctsStats stats;
};

struct ReanalysisParsedRecord {
  bool accepted = false;
  ReanalysisInput input;
  int budget = 0;
  std::vector<std::string> budget_reasons;
};

struct ReanalysisBatchResult {
  std::vector<ReanalysisOutput> outputs;
  std::exception_ptr error;
};

struct GateOpponentSpec {
  cczero::BotKind bot = cczero::BotKind::Mcts;
  std::string label;
  std::string model_path;
};

struct GateTask {
  int game_id = 0;
  size_t opponent_index = 0;
  bool candidate_as_p0 = true;
  uint64_t seed = 0;
  int opening_random_plies = 0;
};

struct GateGameResult {
  int game_id = 0;
  size_t opponent_index = 0;
  bool candidate_as_p0 = true;
  double points = 0.0;
  int win = 0;
  int draw = 0;
  int loss = 0;
  cczero::TerminalStatus status;
  int plies = 0;
  int opening_random_plies = 0;
  std::string log_path;
};

struct GateRow {
  GateOpponentSpec opponent;
  int scheduled_games = 0;
  int games = 0;
  double points = 0.0;
  int wins = 0;
  int draws = 0;
  int losses = 0;
  int initial_position_games = 0;
  int seeded_opening_games = 0;
  int scheduled_initial_position_games = 0;
  int scheduled_seeded_opening_games = 0;
  std::map<std::string, int> reasons;
  std::vector<std::string> logs;
};

struct MultiplayerRecord {
  int game_id = 0;
  uint64_t seed = 0;
  int ply = 0;
  int player = 0;
  int phase = 0;
  uint64_t hash = 0;
  std::string cells;
  int chosen_action = 0;
  std::vector<int> actions;
  std::vector<int> visits;
  std::vector<double> priors;
  int requested_simulations = 0;
  int actual_simulations = 0;
};

struct MultiplayerOutcome {
  std::vector<int> placements;
  std::vector<double> scores;
  std::vector<int> winner_seats;
  std::string reason = "unknown";
};

enum class MultiplayerDataPolicy {
  Random,
  Iter60Adapter,
  VectorMcts,
};

MultiplayerDataPolicy parse_multiplayer_data_policy(const std::string& text) {
  if (text == "random") {
    return MultiplayerDataPolicy::Random;
  }
  if (text == "iter60" || text == "iter60-adapter" || text == "adapter" ||
      text == "own-vs-rest" || text == "own_vs_rest") {
    return MultiplayerDataPolicy::Iter60Adapter;
  }
  if (text == "vector-mcts" || text == "mcts" || text == "mp-mcts") {
    return MultiplayerDataPolicy::VectorMcts;
  }
  throw std::runtime_error("unknown multiplayer data policy: " + text);
}

struct NativeMultiplayerModel {
  int feature_size = 0;
  int action_size = 0;
  int hidden_size = 0;
  int blocks = 0;
  int max_players = 0;
  int value_outputs = 1;
  std::vector<float> input_w;
  std::vector<float> input_b;
  std::vector<float> block_w1;
  std::vector<float> block_b1;
  std::vector<float> block_w2;
  std::vector<float> block_b2;
  std::vector<float> policy_w;
  std::vector<float> policy_b;
  std::vector<float> value_w;
  std::vector<float> value_b;
};

enum class StorageFormat {
  Rich,
  Compact,
};

StorageFormat parse_storage_format(const std::string& text) {
  if (text == "rich") {
    return StorageFormat::Rich;
  }
  if (text == "compact") {
    return StorageFormat::Compact;
  }
  throw std::runtime_error("unknown storage format: " + text);
}

struct ReanalysisBudgetConfig {
  bool enabled = false;
  std::string routing_mode = "complexity";
  int low_complexity_simulations = 384;
  int high_complexity_simulations = 1024;
  double low_entropy_threshold = 0.65;
  double low_surprise_threshold = 0.05;
  int opening_simulations = 0;
  int midgame_simulations = 0;
  int conversion_simulations = 0;
  double high_entropy_threshold = 0.95;
  int high_entropy_simulations = 0;
  double high_surprise_threshold = 0.28;
  int high_surprise_simulations = 0;
};

struct ReanalysisBudgetDecision {
  int simulations = 0;
  std::vector<std::string> reasons;
  double entropy = 0.0;
  double surprise = 0.0;
};

constexpr int kDefaultReanalysisStreamingWindowRecords = 8192;

void print_help(std::ostream& out) {
  out << "cczero training-readiness lab\n\n"
      << "Usage:\n"
      << "  cczero match [--rules PROFILE] [--model PATH] [--p0-model PATH] [--p1-model PATH] [--p0 BOT] [--p1 BOT] [--seed N] [--max-plies N] [--opening-random-plies N] [--mcts-movegen BACKEND] [--mcts-inference-backend BACKEND] [--mcts-inference-batch-size N] [--log PATH]\n"
      << "  cczero match-suite --positions PATH [--limit N] [--no-swap] [--workers N] [match options]\n"
      << "  cczero perft [--rules PROFILE] [--depth N] [--fixture NAME]\n"
      << "  cczero validate-movegen [--rules PROFILE] [--positions N] [--plies N] [--seed N]\n"
      << "  cczero benchmark-movegen [--rules PROFILE] [--positions N] [--plies N] [--seed N] [--backends LIST]\n"
      << "  cczero tournament [--rules PROFILE] [--model PATH] [--bots LIST] [--games N] [--seed N] [--max-plies N] [--out PATH]\n"
      << "  cczero dataset [--rules PROFILE] [--model PATH] [--bot BOT] [--opponent BOT] [--games N] [--seed N] [--max-plies N] --out PATH\n"
      << "  cczero multiplayer-dataset --rules mp3|mp4|mp6 --out PATH [--games N] [--seed N] [--max-plies N] [--movegen reference|fast|bitboard] [--policy random|iter60-adapter|vector-mcts] [--model PATH] [--mp-model PATH] [--simulations N] [--temperature X] [--inference-backend auto|portable|accelerate]\n"
      << "  cczero multiplayer-eval --rules mp3|mp4|mp6 --candidate-mp-model PATH --adapter-model PATH --out PATH [--games N] [--candidate-seat N] [--simulations N] [--temperature X]\n"
      << "  cczero selfplay --model PATH --out PATH [--log-dir DIR] [--games N] [--workers N] [--simulations N] [--opening-random-plies N] [--movegen reference|fast|bitboard] [--inference-backend auto|portable|accelerate] [--inference-batch-size N] [--profile-mcts] [--storage-format compact|rich]\n"
      << "  cczero reanalyze --model PATH --data IN --out OUT [--workers N] [--simulations N] [--streaming-window-records N] [--progress-interval-seconds N] [--movegen reference|fast|bitboard] [--inference-backend auto|portable|accelerate] [--inference-batch-size N] [--reuse-tree] [--profile-mcts] [--storage-format compact|rich]\n"
      << "  cczero promotion-gate --candidate-model PATH --champion-model PATH --out-dir DIR [--opponents LIST] [--games N] [--workers N] [--opening-random-plies N] [--initial-position-fraction X] [--mcts-cpuct X]\n"
      << "  cczero position-info --cells COMPACT121 --player 0|1 [--ply N] [--rules PROFILE] [--movegen reference|fast|bitboard]\n"
      << "  cczero best-move (--model PATH | --bot BOT) --cells COMPACT121 --player 0|1 [--ply N] [--simulations N] [--rules PROFILE] [--movegen reference|fast|bitboard] [--inference-backend auto|portable|accelerate] [--inference-batch-size N]\n"
      << "  cczero model-info --model PATH\n"
      << "  cczero help\n\n"
      << "Rule profiles:\n"
      << "  ab       CCZ-121-AB-LG-v2, tightened anti-block terminal scaffolding\n"
      << "  strict   CCZ-121-Strict-LG-v2, goal-fill plus effective blocked-goal wins\n"
      << "  ab-v1    CCZ-121-AB-LG-v1, legacy reproducibility profile\n"
      << "  strict-v1 CCZ-121-Strict-LG-v1, legacy reproducibility profile\n"
      << "  mp3      CCZ-121-MP3-v1, 3-player multiplayer scaffold\n"
      << "  mp4      CCZ-121-MP4-v1, 4-player multiplayer scaffold\n"
      << "  mp6      CCZ-121-MP6-v1, 6-player multiplayer scaffold\n\n"
      << "Bots:\n";
  for (cczero::BotKind bot : cczero::all_bot_kinds()) {
    out << "  " << cczero::bot_name(bot) << "\n";
  }
  out << "  policy        requires --model PATH\n"
      << "  policy-beam   requires --model PATH\n"
      << "  puct-lite     requires --model PATH\n"
      << "  mcts          requires --model PATH\n";
  out << "\nFixtures:\n"
      << "  initial\n"
      << "  hop-chain\n"
      << "  goal-lock\n\n"
      << "Examples:\n"
      << "  cczero validate-movegen --positions 200 --plies 40 --seed 5\n"
      << "  cczero benchmark-movegen --rules strict --positions 500 --plies 80\n"
      << "  cczero perft --fixture initial --depth 3\n"
      << "  cczero tournament --games 4 --max-plies 200 --out build/tournament.jsonl\n"
      << "  cczero dataset --bot beam --opponent pvs --games 10 --out build/bootstrap.jsonl\n"
      << "  cczero selfplay --model build/policy_value_scale_mixed.ccpv --games 2 --out build/selfplay_v1.jsonl\n"
      << "  cczero reanalyze --model build/new.ccpv --data build/selfplay_v1.jsonl --out build/reanalyzed.jsonl\n"
      << "  cczero best-move --model build/new.ccpv --cells $(...) --player 0 --simulations 64\n"
      << "  cczero best-move --bot converter --cells $(...) --player 0\n";
}

bool is_policy_bot(cczero::BotKind bot) {
  return bot == cczero::BotKind::Policy || bot == cczero::BotKind::PolicyBeam ||
         bot == cczero::BotKind::PuctLite || bot == cczero::BotKind::Mcts;
}

cczero::Move choose_policy_move(cczero::BotKind bot, const cczero::State& state,
                                const cczero::Board& board, const cczero::RuleProfile& rules,
                                const PolicyModel& model,
                                const std::unordered_map<uint64_t, int>& repetition_counts) {
  const std::vector<cczero::Move> moves = cczero::legal_moves(state, board, rules);
  if (moves.empty()) {
    return cczero::Move{};
  }

  std::vector<std::pair<float, cczero::Move>> scored;
  scored.reserve(moves.size());
  for (const cczero::Move& move : moves) {
    cczero::State next = state;
    cczero::apply_move(next, move);
    float score = policy_score(model, state, state.player_to_move, move) +
                  static_cast<float>(policy_progress_score(state, board, state.player_to_move,
                                                           move)) *
                      0.04f;
    const auto repeated = repetition_counts.find(next.hash());
    if (repeated != repetition_counts.end()) {
      score -= 200.0f * repeated->second;
      if (rules.repetition_draw && repeated->second + 1 >= rules.repetition_count) {
        score -= 5000.0f;
      }
    }
    scored.push_back({score, move});
  }

  std::sort(scored.begin(), scored.end(), [](const auto& a, const auto& b) {
    if (a.first != b.first) {
      return a.first > b.first;
    }
    if (a.second.from != b.second.from) {
      return a.second.from < b.second.from;
    }
    return a.second.to < b.second.to;
  });

  if (bot == cczero::BotKind::Policy) {
    return scored.front().second;
  }

  const size_t beam =
      std::min<size_t>(bot == cczero::BotKind::PuctLite ? 24 : 12, scored.size());
  int best_score = std::numeric_limits<int>::min();
  cczero::Move best = scored.front().second;
  for (size_t i = 0; i < beam; ++i) {
    cczero::State next = state;
    cczero::apply_move(next, scored.at(i).second);
    const cczero::TerminalStatus immediate = cczero::terminal_status(next, board, rules, nullptr);
    if (immediate.terminal && immediate.winner == state.player_to_move) {
      return scored.at(i).second;
    }
    const int eval = cczero::evaluate_state(next, board, rules, state.player_to_move);
    const int progress =
        policy_progress_score(state, board, state.player_to_move, scored.at(i).second);
    int reply_risk = 0;
    if (bot == cczero::BotKind::PuctLite) {
      std::vector<cczero::Move> replies = cczero::legal_moves(next, board, rules);
      std::vector<std::pair<int, cczero::Move>> reply_scores;
      reply_scores.reserve(replies.size());
      for (const cczero::Move& reply : replies) {
        cczero::State after_reply = next;
        cczero::apply_move(after_reply, reply);
        const int opponent = next.player_to_move;
        const int reply_progress = policy_progress_score(next, board, opponent, reply);
        const int reply_eval = cczero::evaluate_state(after_reply, board, rules, opponent);
        reply_scores.push_back({reply_eval + reply_progress, reply});
      }
      std::sort(reply_scores.begin(), reply_scores.end(), [](const auto& a, const auto& b) {
        if (a.first != b.first) {
          return a.first > b.first;
        }
        if (a.second.from != b.second.from) {
          return a.second.from < b.second.from;
        }
        return a.second.to < b.second.to;
      });
      const size_t reply_beam = std::min<size_t>(6, reply_scores.size());
      for (size_t r = 0; r < reply_beam; ++r) {
        reply_risk = std::max(reply_risk, reply_scores.at(r).first);
      }
    }
    const int policy_weight = bot == cczero::BotKind::PuctLite ? 3 : 6;
    const int combined =
        eval + 2 * progress - reply_risk / 4 + static_cast<int>(policy_weight * scored.at(i).first);
    if (combined > best_score) {
      best_score = combined;
      best = scored.at(i).second;
    }
  }
  return best;
}

int game_phase(const cczero::State& state, const cczero::Board& board, int player) {
  const int in_goal = cczero::pieces_in_goal(state, board, player);
  const int distance = cczero::total_goal_distance(state, board, player);
  if (in_goal >= 7 || distance <= 18) {
    return 2;
  }
  if (state.ply >= 60 || in_goal >= 3 || distance <= 32) {
    return 1;
  }
  return 0;
}

DatasetRecord make_dataset_record(int game_id, const cczero::State& before,
                                  const cczero::State& after, int player,
                                  cczero::BotKind bot, const std::vector<cczero::Move>& legal,
                                  const cczero::Move& move, const cczero::Board& board) {
  const int opponent = 1 - player;
  DatasetRecord record;
  record.game_id = game_id;
  record.ply = before.ply;
  record.player = player;
  record.bot = bot;
  record.hash = before.hash();
  record.cells = cczero::state_to_compact_string(before);
  record.legal_moves = legal;
  record.chosen = move;
  record.distance_before = cczero::total_goal_distance(before, board, player);
  record.distance_after = cczero::total_goal_distance(after, board, player);
  record.opponent_distance_before = cczero::total_goal_distance(before, board, opponent);
  record.opponent_distance_after = cczero::total_goal_distance(after, board, opponent);
  record.goal_count_before = cczero::pieces_in_goal(before, board, player);
  record.goal_count_after = cczero::pieces_in_goal(after, board, player);
  record.progress_delta = record.distance_before - record.distance_after;
  record.phase = game_phase(before, board, player);
  return record;
}

SelfplayRecord make_selfplay_record(int game_id, uint64_t seed, const cczero::State& before,
                                    const cczero::Move& move,
                                    const std::vector<MctsRootMove>& root_moves,
                                    const MctsStats& stats, const cczero::Board& board) {
  const int player = before.player_to_move;
  const int opponent = 1 - player;
  SelfplayRecord record;
  record.game_id = game_id;
  record.seed = seed;
  record.ply = before.ply;
  record.player = player;
  record.hash = before.hash();
  record.cells = cczero::state_to_compact_string(before);
  record.chosen = move;
  record.root_moves = root_moves;
  record.stats = stats;
  record.distance_before = cczero::total_goal_distance(before, board, player);
  record.opponent_distance_before = cczero::total_goal_distance(before, board, opponent);
  record.goal_count_before = cczero::pieces_in_goal(before, board, player);
  record.opponent_goal_count_before = cczero::pieces_in_goal(before, board, opponent);
  record.home_count_before = pieces_in_home(before, board, player);
  record.opponent_home_count_before = pieces_in_home(before, board, opponent);
  record.goal_blockers_before = cczero::goal_blocker_count(before, board, player);
  record.opponent_goal_blockers_before = cczero::goal_blocker_count(before, board, opponent);
  record.phase = game_phase(before, board, player);
  return record;
}

int move_action_id(const cczero::Move& move);

bool cleanup_move_is_better(int progress, int distance_delta, int goal_delta, int home_delta,
                            int after_distance, int action, int best_progress,
                            int best_distance_delta, int best_goal_delta, int best_home_delta,
                            int best_after_distance, int best_action) {
  if (progress != best_progress) {
    return progress > best_progress;
  }
  if (distance_delta != best_distance_delta) {
    return distance_delta > best_distance_delta;
  }
  if (goal_delta != best_goal_delta) {
    return goal_delta > best_goal_delta;
  }
  if (home_delta != best_home_delta) {
    return home_delta > best_home_delta;
  }
  if (after_distance != best_after_distance) {
    return after_distance < best_after_distance;
  }
  return action < best_action;
}

cczero::Move choose_finish_margin_cleanup_move(const cczero::State& state,
                                               const cczero::Board& board,
                                               const cczero::RuleProfile& rules,
                                               MovegenBackend backend, int player,
                                               const std::vector<cczero::Move>& moves) {
  const int before_distance = cczero::total_goal_distance(state, board, player);
  const int before_goal = cczero::pieces_in_goal(state, board, player);
  const int before_home = pieces_in_home(state, board, player);
  cczero::Move best;
  int best_progress = std::numeric_limits<int>::min();
  int best_distance_delta = std::numeric_limits<int>::min();
  int best_goal_delta = std::numeric_limits<int>::min();
  int best_home_delta = std::numeric_limits<int>::min();
  int best_after_distance = std::numeric_limits<int>::max();
  int best_action = std::numeric_limits<int>::max();
  for (const cczero::Move& move : moves) {
    cczero::State after = state;
    after.player_to_move = player;
    if (!cczero::apply_move(after, move)) {
      continue;
    }
    const int after_distance = cczero::total_goal_distance(after, board, player);
    const int distance_delta = before_distance - after_distance;
    const int goal_delta = cczero::pieces_in_goal(after, board, player) - before_goal;
    const int home_delta = before_home - pieces_in_home(after, board, player);
    const int progress = policy_progress_score(state, board, player, move);
    const int action = move_action_id(move);
    if (!best.is_valid() ||
        cleanup_move_is_better(progress, distance_delta, goal_delta, home_delta,
                               after_distance, action, best_progress, best_distance_delta,
                               best_goal_delta, best_home_delta, best_after_distance,
                               best_action)) {
      best = move;
      best_progress = progress;
      best_distance_delta = distance_delta;
      best_goal_delta = goal_delta;
      best_home_delta = home_delta;
      best_after_distance = after_distance;
      best_action = action;
    }
  }
  if (!best.is_valid()) {
    const std::vector<cczero::Move> fallback =
        legal_moves_with_backend(state, board, rules, backend);
    if (!fallback.empty()) {
      return fallback.front();
    }
  }
  return best;
}

FinishMarginInfo finish_margin_from_terminal_state(const cczero::State& terminal_state,
                                                   const cczero::TerminalStatus& status,
                                                   const cczero::Board& board,
                                                   const cczero::RuleProfile& rules,
                                                   MovegenBackend backend) {
  FinishMarginInfo info;
  info.max_moves = kFinishMarginMaxMoves;
  if (!status.terminal || status.draw || status.winner == cczero::kInvalid) {
    return info;
  }
  const int loser = 1 - status.winner;
  cczero::State cleanup = terminal_state;
  cleanup.player_to_move = loser;
  while (info.moves < kFinishMarginMaxMoves &&
         cczero::pieces_in_goal(cleanup, board, loser) < cczero::kPiecesPerPlayer) {
    const std::vector<cczero::Move> moves =
        legal_moves_with_backend(cleanup, board, rules, backend);
    if (moves.empty()) {
      info.capped = true;
      break;
    }
    const cczero::Move move =
        choose_finish_margin_cleanup_move(cleanup, board, rules, backend, loser, moves);
    if (!move.is_valid() || !cczero::apply_move(cleanup, move)) {
      info.capped = true;
      break;
    }
    cleanup.player_to_move = loser;
    ++info.moves;
  }
  if (cczero::pieces_in_goal(cleanup, board, loser) < cczero::kPiecesPerPlayer) {
    info.capped = true;
  }
  info.available = true;
  info.normalized = std::min(info.moves, kFinishMarginNormalizeMoves) /
                    static_cast<double>(kFinishMarginNormalizeMoves);
  return info;
}

void annotate_score_margin(std::vector<SelfplayRecord>& records,
                           const cczero::TerminalStatus& status,
                           const FinishMarginInfo& margin) {
  if (!margin.available || status.draw || status.winner == cczero::kInvalid) {
    return;
  }
  for (SelfplayRecord& record : records) {
    record.has_score_margin = true;
    record.finish_margin_moves = margin.moves;
    record.finish_margin_max_moves = margin.max_moves;
    record.finish_margin_capped = margin.capped;
    const double sign = status.winner == record.player ? 1.0 : -1.0;
    record.score_margin = sign * margin.normalized;
  }
}

void copy_score_margin(SelfplayRecord& record, const ReanalysisInput& input) {
  if (!input.has_score_margin) {
    return;
  }
  record.has_score_margin = true;
  record.finish_margin_moves = input.finish_margin_moves;
  record.finish_margin_max_moves = input.finish_margin_max_moves;
  record.finish_margin_capped = input.finish_margin_capped;
  record.score_margin = input.score_margin;
}

void add_search_stats(SearchTotals& totals, const MctsStats& stats) {
  totals.nodes += static_cast<uint64_t>(std::max(0, stats.nodes));
  totals.evals += static_cast<uint64_t>(std::max(0, stats.evals));
  totals.simulations += static_cast<uint64_t>(std::max(0, stats.simulations));
  totals.root_legal_moves += static_cast<uint64_t>(std::max(0, stats.root_legal_moves));
  totals.transposition_hits += static_cast<uint64_t>(std::max(0, stats.transposition_hits));
  totals.adaptive_stops += stats.adaptive_stopped ? 1ULL : 0ULL;
  totals.reuse_hits += stats.reused_tree ? 1ULL : 0ULL;
  totals.inference_batches += static_cast<uint64_t>(std::max(0, stats.inference_batches));
  totals.search_ms += stats.elapsed_ms;
  totals.movegen_ms += stats.movegen_ms;
  totals.eval_ms += stats.eval_ms;
  totals.policy_ms += stats.policy_ms;
}

void write_hex_hash(std::ostream& out, uint64_t value) {
  constexpr char kHex[] = "0123456789abcdef";
  out << "0x";
  for (int shift = 60; shift >= 0; shift -= 4) {
    out << kHex[(value >> shift) & 0xFULL];
  }
}

void write_path_json(std::ostream& out, const std::vector<int>& path) {
  out << "[";
  for (size_t i = 0; i < path.size(); ++i) {
    if (i != 0) {
      out << ",";
    }
    out << path[i];
  }
  out << "]";
}

void write_move_object(std::ostream& out, const cczero::Move& move) {
  out << "{\"from\":" << move.from << ",\"to\":" << move.to << ",\"path\":";
  write_path_json(out, move.path);
  out << "}";
}

int move_action_id(const cczero::Move& move) {
  return move.from * cczero::kBoardSize + move.to;
}

void write_int_array_json(std::ostream& out, const std::vector<int>& values) {
  out << "[";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      out << ",";
    }
    out << values[i];
  }
  out << "]";
}

void write_double_array_json(std::ostream& out, const std::vector<double>& values) {
  out << "[";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      out << ",";
    }
    out << values[i];
  }
  out << "]";
}

void write_score_margin_fields(std::ostream& out, const SelfplayRecord& record) {
  if (!record.has_score_margin) {
    return;
  }
  out << ",\"finish_margin_moves\":" << record.finish_margin_moves
      << ",\"finish_margin_max_moves\":" << record.finish_margin_max_moves
      << ",\"finish_margin_capped\":" << (record.finish_margin_capped ? "true" : "false")
      << ",\"score_margin\":" << record.score_margin
      << ",\"score_margin_source\":\"" << kScoreMarginSource << "\"";
}

std::optional<double> root_q_target(const std::vector<MctsRootMove>& root_moves) {
  double weighted = 0.0;
  int visit_sum = 0;
  for (const MctsRootMove& root_move : root_moves) {
    if (root_move.visits <= 0) {
      continue;
    }
    weighted += static_cast<double>(root_move.visits) * root_move.value;
    visit_sum += root_move.visits;
  }
  if (visit_sum <= 0) {
    return std::nullopt;
  }
  return std::clamp(weighted / static_cast<double>(visit_sum), -1.0, 1.0);
}

void write_root_q_fields(std::ostream& out, const SelfplayRecord& record) {
  const std::optional<double> root_q = root_q_target(record.root_moves);
  if (!root_q.has_value()) {
    return;
  }
  out << ",\"root_q\":" << root_q.value()
      << ",\"root_q_source\":\"visit_weighted_root_q_v1\"";
}

std::vector<std::pair<int, int>> endpoint_signature(std::vector<cczero::Move> moves) {
  std::vector<std::pair<int, int>> signature;
  signature.reserve(moves.size());
  for (const cczero::Move& move : moves) {
    signature.push_back({move.from, move.to});
  }
  std::sort(signature.begin(), signature.end());
  signature.erase(std::unique(signature.begin(), signature.end()), signature.end());
  return signature;
}

cczero::State fixture_state(const std::string& name, const cczero::Board& board) {
  if (name == "initial") {
    return cczero::State::initial(board);
  }

  cczero::State state = cczero::State::empty();
  if (name == "hop-chain") {
    const int start = board.id_at(0, 0);
    const int blocker_a = board.id_at(1, 0);
    const int blocker_b = board.id_at(3, 0);
    if (start == cczero::kInvalid || blocker_a == cczero::kInvalid ||
        blocker_b == cczero::kInvalid) {
      throw std::runtime_error("hop-chain fixture geometry is unavailable");
    }
    state.cells.at(static_cast<size_t>(start)) = 0;
    state.cells.at(static_cast<size_t>(blocker_a)) = 1;
    state.cells.at(static_cast<size_t>(blocker_b)) = 1;
    return state;
  }

  if (name == "goal-lock") {
    const int in_goal = board.id_at(-1, 5);
    const int neighbor = board.id_at(-2, 5);
    if (in_goal == cczero::kInvalid || neighbor == cczero::kInvalid) {
      throw std::runtime_error("goal-lock fixture geometry is unavailable");
    }
    state.cells.at(static_cast<size_t>(in_goal)) = 0;
    state.cells.at(static_cast<size_t>(neighbor)) = 1;
    return state;
  }

  throw std::runtime_error("unknown fixture: " + name);
}

MatchSummary play_game(int game_id, cczero::BotKind p0, cczero::BotKind p1,
                       cczero::RuleProfile rules, uint64_t seed, int max_plies,
                       std::ostream* log, std::vector<DatasetRecord>* records,
                       const PolicyModel* p0_policy_model, const PolicyModel* p1_policy_model,
                       const MctsOverrides* mcts_overrides = nullptr,
                       int opening_random_plies = 0,
                       const cczero::State* initial_state = nullptr) {
  const cczero::Board& board = cczero::Board::standard();
  rules.max_plies = max_plies;
  cczero::State state = initial_state == nullptr ? cczero::State::initial(board) : *initial_state;
  std::mt19937_64 rng(seed);
  std::unordered_map<uint64_t, int> repetition_counts;
  repetition_counts[state.hash()] = 1;

  for (int opening_ply = 0; opening_ply < opening_random_plies; ++opening_ply) {
    const cczero::TerminalStatus opening_status =
        cczero::terminal_status(state, board, rules, &repetition_counts);
    if (opening_status.terminal) {
      break;
    }
    const std::vector<cczero::Move> opening_moves = cczero::legal_moves(state, board, rules);
    if (opening_moves.empty()) {
      break;
    }
    std::uniform_int_distribution<size_t> dist(0, opening_moves.size() - 1);
    if (!cczero::apply_move(state, opening_moves.at(dist(rng)))) {
      throw std::runtime_error("failed to apply random opening move");
    }
    ++repetition_counts[state.hash()];
  }

  if (log != nullptr) {
    cczero::write_jsonl_game_start(*log, seed, p0, p1, rules, max_plies, &state);
  }

  cczero::TerminalStatus status =
      cczero::terminal_status(state, board, rules, &repetition_counts);
  while (!status.terminal) {
    const int player = state.player_to_move;
    const cczero::BotKind bot = player == 0 ? p0 : p1;
    const PolicyModel* policy_model = player == 0 ? p0_policy_model : p1_policy_model;
    const std::vector<cczero::Move> legal = cczero::legal_moves(state, board, rules);
    if (is_policy_bot(bot) && policy_model == nullptr) {
      throw std::runtime_error("policy bot requires --model PATH");
    }
    const cczero::Move move =
        bot == cczero::BotKind::Mcts
            ? choose_mcts_move(state, board, rules, *policy_model, repetition_counts, rng,
                               mcts_overrides)
            : is_policy_bot(bot)
                  ? choose_policy_move(bot, state, board, rules, *policy_model, repetition_counts)
                  : cczero::choose_move_avoiding_repetition(bot, state, board, rules, rng,
                                                            repetition_counts);
    if (!move.is_valid()) {
      status = cczero::TerminalStatus{true, 1 - player, false, "no_legal_moves"};
      break;
    }

    std::string error;
    if (!cczero::validate_move_witness(state, move, board, rules, &error)) {
      throw std::runtime_error("bot produced invalid move: " + error);
    }

    const cczero::State before = state;
    if (!cczero::apply_move(state, move)) {
      throw std::runtime_error("failed to apply validated move");
    }
    if (records != nullptr) {
      records->push_back(make_dataset_record(game_id, before, state, player, bot, legal, move, board));
    }
    ++repetition_counts[state.hash()];
    if (log != nullptr) {
      cczero::write_jsonl_move(*log, state, move, board, player);
    }
    status = cczero::terminal_status(state, board, rules, &repetition_counts);
  }

  if (log != nullptr) {
    cczero::write_jsonl_game_end(*log, status, state.ply);
  }
  return MatchSummary{status, state.ply};
}

void write_dataset_record(std::ostream& out, const DatasetRecord& record,
                          const cczero::TerminalStatus& result, cczero::BotKind generator,
                          cczero::BotKind opponent, const cczero::RuleProfile& rules) {
  int result_value = 0;
  if (!result.draw && result.winner != cczero::kInvalid) {
    result_value = result.winner == record.player ? 1 : -1;
  }

  out << "{\"type\":\"training_position\",\"game_id\":" << record.game_id
      << ",\"ply\":" << record.ply << ",\"player\":" << record.player
      << ",\"hash\":\"";
  write_hex_hash(out, record.hash);
  out << "\",\"cells\":\""
      << record.cells << "\",\"generator\":\"" << cczero::bot_name(generator)
      << "\",\"opponent\":\"" << cczero::bot_name(opponent) << "\",\"bot\":\""
      << cczero::bot_name(record.bot) << "\",\"rule_profile\":\""
      << cczero::json_escape(rules.name) << "\",\"result\":" << result_value
      << ",\"phase\":" << record.phase
      << ",\"distance_before\":" << record.distance_before
      << ",\"distance_after\":" << record.distance_after
      << ",\"distance_delta\":" << record.progress_delta
      << ",\"opponent_distance_before\":" << record.opponent_distance_before
      << ",\"opponent_distance_after\":" << record.opponent_distance_after
      << ",\"goal_count_before\":" << record.goal_count_before
      << ",\"goal_count_after\":" << record.goal_count_after
      << ",\"goal_delta\":" << (record.goal_count_after - record.goal_count_before)
      << ",\"chosen\":";
  write_move_object(out, record.chosen);
  out << ",\"legal_count\":" << record.legal_moves.size() << ",\"legal\":[";
  for (size_t i = 0; i < record.legal_moves.size(); ++i) {
    if (i != 0) {
      out << ",";
    }
    write_move_object(out, record.legal_moves[i]);
  }
  out << "]}\n";
}

void write_selfplay_record_compact(std::ostream& out, const SelfplayRecord& record,
                                   const cczero::TerminalStatus& result,
                                   const cczero::RuleProfile& rules,
                                   const std::string& model_id) {
  int result_value = 0;
  if (!result.draw && result.winner != cczero::kInvalid) {
    result_value = result.winner == record.player ? 1 : -1;
  }

  std::vector<int> actions;
  std::vector<int> visits;
  std::vector<double> priors;
  actions.reserve(record.root_moves.size());
  visits.reserve(record.root_moves.size());
  priors.reserve(record.root_moves.size());
  int visit_sum = 0;
  for (const MctsRootMove& root_move : record.root_moves) {
    actions.push_back(move_action_id(root_move.move));
    visits.push_back(root_move.visits);
    priors.push_back(root_move.prior);
    visit_sum += root_move.visits;
  }

  out << "{\"type\":\"selfplay_position\",\"schema\":\"cczero.selfplay.compact.v1\""
      << ",\"game_id\":" << record.game_id
      << ",\"seed\":" << record.seed
      << ",\"ply\":" << record.ply
      << ",\"player\":" << record.player
      << ",\"hash\":\"";
  write_hex_hash(out, record.hash);
  out << "\",\"cells\":\"" << record.cells
      << "\",\"model_id\":\"" << cczero::json_escape(model_id)
      << "\",\"rule_profile\":\"" << cczero::json_escape(rules.name)
      << "\",\"result\":" << result_value
      << ",\"phase\":" << record.phase
      << ",\"distance_before\":" << record.distance_before
      << ",\"opponent_distance_before\":" << record.opponent_distance_before
      << ",\"distance_advantage\":"
      << (record.opponent_distance_before - record.distance_before)
      << ",\"goal_count_before\":" << record.goal_count_before
      << ",\"opponent_goal_count_before\":" << record.opponent_goal_count_before
      << ",\"goal_advantage\":"
      << (record.goal_count_before - record.opponent_goal_count_before)
      << ",\"home_count_before\":" << record.home_count_before
      << ",\"opponent_home_count_before\":" << record.opponent_home_count_before
      << ",\"home_advantage\":"
      << (record.opponent_home_count_before - record.home_count_before)
      << ",\"goal_blockers_before\":" << record.goal_blockers_before
      << ",\"opponent_goal_blockers_before\":" << record.opponent_goal_blockers_before
      << ",\"chosen_action\":" << move_action_id(record.chosen)
      << ",\"legal_count\":" << actions.size()
      << ",\"visit_sum\":" << visit_sum
      << ",\"actions\":";
  write_int_array_json(out, actions);
  out << ",\"visits\":";
  write_int_array_json(out, visits);
  out << ",\"priors\":";
  write_double_array_json(out, priors);
  write_root_q_fields(out, record);
  write_score_margin_fields(out, record);
  out << "}\n";
}

void write_selfplay_record(std::ostream& out, const SelfplayRecord& record,
                           const cczero::TerminalStatus& result,
                           const cczero::RuleProfile& rules, const std::string& model_id,
                           const MctsConfig& config, int max_plies) {
  int result_value = 0;
  if (!result.draw && result.winner != cczero::kInvalid) {
    result_value = result.winner == record.player ? 1 : -1;
  }
  int visit_sum = 0;
  for (const MctsRootMove& root_move : record.root_moves) {
    visit_sum += root_move.visits;
  }
  std::ostringstream elapsed_ms;
  elapsed_ms << std::fixed << std::setprecision(3) << record.stats.elapsed_ms;
  out << "{\"type\":\"selfplay_position\",\"schema\":\"cczero.selfplay.v1\""
      << ",\"game_id\":" << record.game_id
      << ",\"seed\":" << record.seed
      << ",\"ply\":" << record.ply
      << ",\"player\":" << record.player
      << ",\"hash\":\"";
  write_hex_hash(out, record.hash);
  out << "\",\"cells\":\"" << record.cells
      << "\",\"model_id\":\"" << cczero::json_escape(model_id)
      << "\",\"rule_profile\":\"" << cczero::json_escape(rules.name)
      << "\",\"result\":" << result_value
      << ",\"phase\":" << record.phase
      << ",\"distance_before\":" << record.distance_before
      << ",\"opponent_distance_before\":" << record.opponent_distance_before
      << ",\"distance_advantage\":"
      << (record.opponent_distance_before - record.distance_before)
      << ",\"goal_count_before\":" << record.goal_count_before
      << ",\"opponent_goal_count_before\":" << record.opponent_goal_count_before
      << ",\"goal_advantage\":"
      << (record.goal_count_before - record.opponent_goal_count_before)
      << ",\"home_count_before\":" << record.home_count_before
      << ",\"opponent_home_count_before\":" << record.opponent_home_count_before
      << ",\"home_advantage\":"
      << (record.opponent_home_count_before - record.home_count_before)
      << ",\"goal_blockers_before\":" << record.goal_blockers_before
      << ",\"opponent_goal_blockers_before\":" << record.opponent_goal_blockers_before
      << ",\"chosen\":";
  write_move_object(out, record.chosen);
  write_score_margin_fields(out, record);
  out << ",\"legal_count\":" << record.root_moves.size()
      << ",\"visit_sum\":" << visit_sum
      << ",\"search\":{\"simulations\":" << config.simulations
      << ",\"cpuct\":" << config.cpuct
      << ",\"root_noise\":" << (config.add_root_noise ? "true" : "false")
      << ",\"root_dirichlet_alpha\":" << config.root_dirichlet_alpha
      << ",\"root_noise_fraction\":" << config.root_noise_fraction
      << ",\"temperature\":" << config.temperature
      << ",\"draw_leaf_value\":" << config.draw_leaf_value
      << ",\"anti_draw_logit_scale\":" << config.anti_draw_logit_scale
      << ",\"progress_prior_scale\":" << config.progress_prior_scale
      << ",\"home_pressure_scale\":" << config.home_pressure_scale
      << ",\"transpositions\":" << (config.transpositions ? "true" : "false")
      << ",\"reuse_tree\":" << (config.reuse_tree ? "true" : "false")
      << ",\"profile_mcts\":" << (config.profile_mcts ? "true" : "false")
      << ",\"adaptive_simulations\":" << (config.adaptive_simulations ? "true" : "false")
      << ",\"min_simulations\":" << config.min_simulations
      << ",\"adaptive_check_interval\":" << config.adaptive_check_interval
      << ",\"adaptive_confidence\":" << config.adaptive_confidence
      << ",\"movegen\":\"" << movegen_backend_name(config.movegen) << "\""
      << ",\"inference_backend\":\"" << inference_backend_name(config.inference_backend) << "\""
      << ",\"inference_resolved_backend\":\""
      << inference_backend_name(resolve_inference_backend(config.inference_backend)) << "\""
      << ",\"inference_batch_size\":" << config.inference_batch_size
      << ",\"materialize_root_moves\":" << (config.materialize_root_moves ? "true" : "false")
      << ",\"max_plies\":" << max_plies
      << "},\"stats\":{\"nodes\":" << record.stats.nodes
      << ",\"evals\":" << record.stats.evals
      << ",\"simulations\":" << record.stats.simulations
      << ",\"elapsed_ms\":" << elapsed_ms.str()
      << ",\"root_legal_moves\":" << record.stats.root_legal_moves
      << ",\"transposition_hits\":" << record.stats.transposition_hits
      << ",\"inference_batches\":" << record.stats.inference_batches
      << ",\"adaptive_stopped\":" << (record.stats.adaptive_stopped ? "true" : "false")
      << ",\"movegen_ms\":" << record.stats.movegen_ms
      << ",\"eval_ms\":" << record.stats.eval_ms
      << ",\"policy_ms\":" << record.stats.policy_ms
      << ",\"select_ms\":" << record.stats.select_ms
      << ",\"backup_ms\":" << record.stats.backup_ms
      << "}";
  write_root_q_fields(out, record);
  out << ",\"legal\":[";
  for (size_t i = 0; i < record.root_moves.size(); ++i) {
    if (i != 0) {
      out << ",";
    }
    const MctsRootMove& root_move = record.root_moves[i];
    out << "{\"from\":" << root_move.move.from << ",\"to\":" << root_move.move.to
        << ",\"path\":";
    write_path_json(out, root_move.move.path);
    out << ",\"visits\":" << root_move.visits
        << ",\"policy_target\":"
        << (visit_sum == 0 ? 0.0 : static_cast<double>(root_move.visits) / visit_sum)
        << ",\"prior\":" << root_move.prior
        << ",\"q\":" << root_move.value
        << "}";
  }
  out << "]}\n";
}

void write_selfplay_record(std::ostream& out, const SelfplayRecord& record,
                           const cczero::TerminalStatus& result,
                           const cczero::RuleProfile& rules, const std::string& model_id,
                           const MctsConfig& config, int max_plies,
                           StorageFormat storage_format) {
  if (storage_format == StorageFormat::Compact) {
    write_selfplay_record_compact(out, record, result, rules, model_id);
    return;
  }
  write_selfplay_record(out, record, result, rules, model_id, config, max_plies);
}

int multiplayer_pieces_in_goal(const cczero::State& state, const cczero::Board& board,
                               const cczero::RuleProfile& rules, int player) {
  int count = 0;
  for (int id : board.goal_cell_ids(rules, player)) {
    if (state.cells.at(static_cast<size_t>(id)) == player) {
      ++count;
    }
  }
  return count;
}

int multiplayer_pieces_in_home(const cczero::State& state, const cczero::Board& board,
                               const cczero::RuleProfile& rules, int player) {
  int count = 0;
  for (int id : board.home_cell_ids(rules, player)) {
    if (state.cells.at(static_cast<size_t>(id)) == player) {
      ++count;
    }
  }
  return count;
}

int multiplayer_total_goal_distance(const cczero::State& state, const cczero::Board& board,
                                    const cczero::RuleProfile& rules, int player) {
  int total = 0;
  for (int id = 0; id < cczero::kBoardSize; ++id) {
    if (state.cells.at(static_cast<size_t>(id)) == player) {
      total += board.goal_distance(rules, player, id);
    }
  }
  return total;
}

int multiplayer_phase(const cczero::State& state, const cczero::Board& board,
                      const cczero::RuleProfile& rules, int player, int max_plies) {
  const int in_goal = multiplayer_pieces_in_goal(state, board, rules, player);
  const int distance = multiplayer_total_goal_distance(state, board, rules, player);
  if (in_goal >= 7 || distance <= 18 || state.ply >= (2 * max_plies) / 3) {
    return 2;
  }
  if (in_goal >= 3 || distance <= 32 || state.ply >= max_plies / 3) {
    return 1;
  }
  return 0;
}

MultiplayerRecord make_multiplayer_record(int game_id, uint64_t seed,
                                          const cczero::State& before,
                                          const cczero::Move& chosen,
                                          const std::vector<cczero::Move>& legal,
                                          const std::vector<int>& visits,
                                          const std::vector<double>& priors,
                                          const cczero::Board& board,
                                          const cczero::RuleProfile& rules,
                                          int max_plies, int requested_simulations,
                                          int actual_simulations) {
  MultiplayerRecord record;
  record.game_id = game_id;
  record.seed = seed;
  record.ply = before.ply;
  record.player = before.player_to_move;
  record.phase = multiplayer_phase(before, board, rules, record.player, max_plies);
  record.hash = before.hash();
  record.cells = cczero::state_to_compact_string(before);
  record.chosen_action = move_action_id(chosen);
  record.requested_simulations = requested_simulations;
  record.actual_simulations = actual_simulations;
  record.actions.reserve(legal.size());
  record.visits.reserve(legal.size());
  record.priors.reserve(legal.size());
  for (const cczero::Move& move : legal) {
    const size_t index = record.actions.size();
    const int action = move_action_id(move);
    record.actions.push_back(action);
    record.visits.push_back(index < visits.size() ? visits.at(index)
                                                  : (action == record.chosen_action ? 1 : 0));
    record.priors.push_back(index < priors.size() ? priors.at(index) : 0.0);
  }
  return record;
}

std::vector<int> one_hot_visits(const std::vector<cczero::Move>& legal,
                                const cczero::Move& chosen) {
  std::vector<int> visits;
  visits.reserve(legal.size());
  const int chosen_action = move_action_id(chosen);
  for (const cczero::Move& move : legal) {
    visits.push_back(move_action_id(move) == chosen_action ? 1 : 0);
  }
  return visits;
}

cczero::Coord rotate_coord_toward_player0_home(cczero::Coord coord, int home_arm) {
  int q = coord.q;
  int r = coord.r;
  for (int step = 0; step < home_arm; ++step) {
    const int s = -q - r;
    const int next_q = -s;
    const int next_r = -q;
    q = next_q;
    r = next_r;
  }
  return cczero::Coord{q, r};
}

using Iter60AdapterRotationMap =
    std::array<std::array<int, cczero::kBoardSize>, cczero::kStarArms>;

const Iter60AdapterRotationMap& iter60_adapter_rotation_map(const cczero::Board& board) {
  static const Iter60AdapterRotationMap maps = [&board]() {
    Iter60AdapterRotationMap built{};
    for (int arm = 0; arm < cczero::kStarArms; ++arm) {
      for (int id = 0; id < cczero::kBoardSize; ++id) {
        const cczero::Coord rotated =
            rotate_coord_toward_player0_home(board.coord(id), arm);
        const int mapped = board.id_at(rotated.q, rotated.r);
        if (mapped == cczero::kInvalid) {
          throw std::runtime_error("multiplayer adapter rotation produced an invalid cell");
        }
        built.at(static_cast<size_t>(arm)).at(static_cast<size_t>(id)) = mapped;
      }
    }
    return built;
  }();
  return maps;
}

int rotate_cell_toward_player0_home(const cczero::Board& board, int id, int home_arm) {
  return iter60_adapter_rotation_map(board)
      .at(static_cast<size_t>(home_arm))
      .at(static_cast<size_t>(id));
}

cczero::State iter60_adapter_state(const cczero::State& state, const cczero::Board& board,
                                   const cczero::RuleProfile& rules, int player) {
  cczero::State adapted = cczero::State::empty();
  adapted.player_to_move = 0;
  adapted.ply = state.ply;
  const int home_arm = rules.home_arm(player);
  for (int id = 0; id < cczero::kBoardSize; ++id) {
    const int occupant = state.cells.at(static_cast<size_t>(id));
    if (occupant == cczero::kEmpty || !rules.valid_player(occupant)) {
      continue;
    }
    const int mapped = rotate_cell_toward_player0_home(board, id, home_arm);
    adapted.cells.at(static_cast<size_t>(mapped)) =
        static_cast<int8_t>(occupant == player ? 0 : 1);
  }
  return adapted;
}

int iter60_adapter_action(const cczero::Move& move, const cczero::Board& board,
                          const cczero::RuleProfile& rules, int player) {
  const int home_arm = rules.home_arm(player);
  const auto& map = iter60_adapter_rotation_map(board).at(static_cast<size_t>(home_arm));
  return map.at(static_cast<size_t>(move.from)) * cczero::kBoardSize +
         map.at(static_cast<size_t>(move.to));
}

std::vector<double> uniform_multiplayer_priors(size_t legal_count) {
  if (legal_count == 0) {
    return {};
  }
  return std::vector<double>(legal_count, 1.0 / static_cast<double>(legal_count));
}

std::vector<double> iter60_adapter_priors(const PolicyModel& model, const cczero::State& state,
                                         const cczero::Board& board,
                                         const cczero::RuleProfile& rules,
                                         const std::vector<cczero::Move>& legal,
                                         double temperature,
                                         MlpWorkspace* workspace = nullptr,
                                         InferenceBackend backend = InferenceBackend::Auto) {
  if (legal.empty()) {
    return {};
  }
  if (temperature <= 0.0 || !std::isfinite(temperature)) {
    throw std::runtime_error("--temperature must be finite and positive");
  }
  const int player = state.player_to_move;
  const cczero::State adapted_state = iter60_adapter_state(state, board, rules, player);
  MlpWorkspace local_workspace;
  const float* hidden = nullptr;
  const float* policy_state = nullptr;
  if (model.kind == ModelKind::PolicyValueMlp) {
    MlpWorkspace& active_workspace = workspace == nullptr ? local_workspace : *workspace;
    mlp_hidden_optimized(model, adapted_state, 0, active_workspace, backend);
    hidden = active_workspace.hidden.data();
    if (model.policy_head == PolicyHeadKind::MoveMlp ||
        model.policy_head == PolicyHeadKind::MoveBilinear) {
      active_workspace.ensure_policy_state(static_cast<size_t>(model.move_hidden_size));
      mlp_policy_state_projection(model, hidden, active_workspace.policy_state.data());
      policy_state = active_workspace.policy_state.data();
    }
  }
  std::vector<double> logits;
  logits.reserve(legal.size());
  double max_logit = -std::numeric_limits<double>::infinity();
  for (const cczero::Move& move : legal) {
    const int action = iter60_adapter_action(move, board, rules, player);
    double raw_logit = 0.0;
    if (model.kind == ModelKind::PolicyValueMlp) {
      raw_logit =
          policy_state == nullptr
              ? static_cast<double>(mlp_policy_logit_action_ptr(model, hidden, action, 0))
              : static_cast<double>(
                    mlp_policy_logit_action_projected_ptr(model, hidden, policy_state,
                                                          action, 0));
    } else {
      raw_logit = static_cast<double>(
          cczero::policy_score_action(model, adapted_state, 0, action / cczero::kBoardSize,
                                      action % cczero::kBoardSize));
    }
    const double logit = raw_logit / temperature;
    logits.push_back(logit);
    max_logit = std::max(max_logit, logit);
  }
  std::vector<double> priors;
  priors.reserve(logits.size());
  double total = 0.0;
  for (double logit : logits) {
    const double prior = std::exp(logit - max_logit);
    priors.push_back(prior);
    total += prior;
  }
  if (total <= 0.0 || !std::isfinite(total)) {
    return uniform_multiplayer_priors(legal.size());
  }
  for (double& prior : priors) {
    prior /= total;
  }
  return priors;
}

std::vector<float> encode_native_mp_features(const cczero::State& state,
                                             const cczero::Board& board,
                                             const cczero::RuleProfile& rules,
                                             int perspective_player);
std::vector<float> native_mp_hidden(const NativeMultiplayerModel& model,
                                    const std::vector<float>& features);
double native_mp_value(const NativeMultiplayerModel& model, const std::vector<float>& hidden);
std::vector<double> native_mp_values(const NativeMultiplayerModel& model,
                                     const std::vector<float>& hidden);
double native_mp_policy_logit(const NativeMultiplayerModel& model,
                              const std::vector<float>& hidden, int action);

std::vector<double> native_mp_priors(const NativeMultiplayerModel& model,
                                     const cczero::State& state,
                                     const cczero::Board& board,
                                     const cczero::RuleProfile& rules,
                                     const std::vector<cczero::Move>& legal,
                                     double temperature) {
  if (legal.empty()) {
    return {};
  }
  if (temperature <= 0.0 || !std::isfinite(temperature)) {
    throw std::runtime_error("--temperature must be finite and positive");
  }
  const std::vector<float> features =
      encode_native_mp_features(state, board, rules, state.player_to_move);
  const std::vector<float> hidden = native_mp_hidden(model, features);
  std::vector<double> logits;
  logits.reserve(legal.size());
  double max_logit = -std::numeric_limits<double>::infinity();
  for (const cczero::Move& move : legal) {
    const double logit =
        native_mp_policy_logit(model, hidden, move_action_id(move)) / temperature;
    logits.push_back(logit);
    max_logit = std::max(max_logit, logit);
  }
  std::vector<double> priors;
  priors.reserve(logits.size());
  double total = 0.0;
  for (double logit : logits) {
    const double prior = std::exp(logit - max_logit);
    priors.push_back(prior);
    total += prior;
  }
  if (total <= 0.0 || !std::isfinite(total)) {
    return uniform_multiplayer_priors(legal.size());
  }
  for (double& prior : priors) {
    prior /= total;
  }
  return priors;
}

std::vector<double> native_mp_value_vector(const NativeMultiplayerModel& model,
                                           const cczero::State& state,
                                           const cczero::Board& board,
                                           const cczero::RuleProfile& rules) {
  std::vector<double> values(static_cast<size_t>(rules.player_count), 0.0);
  if (model.value_outputs >= rules.player_count) {
    const int perspective = state.player_to_move;
    const std::vector<float> features = encode_native_mp_features(state, board, rules, perspective);
    const std::vector<float> hidden = native_mp_hidden(model, features);
    const std::vector<double> relative_values = native_mp_values(model, hidden);
    for (int offset = 0; offset < rules.player_count; ++offset) {
      const int player = (perspective + offset) % rules.player_count;
      values.at(static_cast<size_t>(player)) = relative_values.at(static_cast<size_t>(offset));
    }
    return values;
  }
  for (int player = 0; player < rules.player_count; ++player) {
    const std::vector<float> features = encode_native_mp_features(state, board, rules, player);
    const std::vector<float> hidden = native_mp_hidden(model, features);
    values.at(static_cast<size_t>(player)) = native_mp_value(model, hidden);
  }
  return values;
}

size_t sample_multiplayer_policy_index(const std::vector<double>& priors, std::mt19937_64& rng) {
  if (priors.empty()) {
    throw std::runtime_error("cannot sample from empty multiplayer policy");
  }
  double total = 0.0;
  for (double prior : priors) {
    if (prior > 0.0 && std::isfinite(prior)) {
      total += prior;
    }
  }
  if (total <= 0.0) {
    std::uniform_int_distribution<size_t> dist(0, priors.size() - 1);
    return dist(rng);
  }
  std::uniform_real_distribution<double> dist(0.0, total);
  double sample = dist(rng);
  for (size_t index = 0; index < priors.size(); ++index) {
    const double prior = priors.at(index);
    if (prior <= 0.0 || !std::isfinite(prior)) {
      continue;
    }
    if (sample <= prior) {
      return index;
    }
    sample -= prior;
  }
  return priors.size() - 1;
}

void advance_multiplayer_turn(cczero::State& state, const cczero::RuleProfile& rules) {
  state.player_to_move = (state.player_to_move + 1) % std::max(1, rules.player_count);
  ++state.ply;
}

constexpr int kNativeMpPlayerCountFeatures = 3;
constexpr int kNativeMpPhaseFeatures = 3;
constexpr int kNativeMpFeatureSize =
    cczero::kMaxPlayers * cczero::kBoardSize + cczero::kBoardSize +
    kNativeMpPlayerCountFeatures + cczero::kMaxPlayers + 1 + kNativeMpPhaseFeatures;

void read_exact(std::istream& input, void* data, size_t bytes, const std::string& what) {
  input.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(bytes));
  if (!input) {
    throw std::runtime_error("truncated native multiplayer model section: " + what);
  }
}

void read_float_vector(std::istream& input, std::vector<float>& values, size_t count,
                       const std::string& what) {
  values.resize(count);
  read_exact(input, values.data(), count * sizeof(float), what);
  for (float value : values) {
    if (!std::isfinite(value)) {
      throw std::runtime_error("non-finite native multiplayer model value in: " + what);
    }
  }
}

void read_float_values(std::istream& input, float* values, size_t count,
                       const std::string& what) {
  read_exact(input, values, count * sizeof(float), what);
  for (size_t i = 0; i < count; ++i) {
    if (!std::isfinite(values[i])) {
      throw std::runtime_error("non-finite native multiplayer model value in: " + what);
    }
  }
}

NativeMultiplayerModel load_native_multiplayer_model(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("failed to open native multiplayer model: " + path);
  }
  char magic[16] = {};
  read_exact(input, magic, sizeof(magic), "magic");
  const bool format_v1 = std::memcmp(magic, "CCZMPVv1", 8) == 0;
  const bool format_v2 = std::memcmp(magic, "CCZMPVv2", 8) == 0;
  if (!format_v1 && !format_v2) {
    throw std::runtime_error("invalid native multiplayer model magic: " + path);
  }
  NativeMultiplayerModel model;
  read_exact(input, &model.feature_size, sizeof(model.feature_size), "feature_size");
  read_exact(input, &model.action_size, sizeof(model.action_size), "action_size");
  read_exact(input, &model.hidden_size, sizeof(model.hidden_size), "hidden_size");
  read_exact(input, &model.blocks, sizeof(model.blocks), "blocks");
  read_exact(input, &model.max_players, sizeof(model.max_players), "max_players");
  if (format_v2) {
    read_exact(input, &model.value_outputs, sizeof(model.value_outputs), "value_outputs");
  } else {
    model.value_outputs = 1;
  }
  if (model.feature_size != kNativeMpFeatureSize ||
      model.action_size != cczero::kBoardSize * cczero::kBoardSize ||
      model.hidden_size <= 0 || model.blocks < 0 || model.max_players != cczero::kMaxPlayers ||
      model.value_outputs <= 0 || model.value_outputs > cczero::kMaxPlayers) {
    throw std::runtime_error("unsupported native multiplayer model dimensions: " + path);
  }
  const size_t hidden = static_cast<size_t>(model.hidden_size);
  const size_t features = static_cast<size_t>(model.feature_size);
  const size_t blocks = static_cast<size_t>(model.blocks);
  const size_t actions = static_cast<size_t>(model.action_size);
  read_float_vector(input, model.input_w, hidden * features, "input_w");
  read_float_vector(input, model.input_b, hidden, "input_b");
  model.block_w1.resize(blocks * hidden * hidden);
  model.block_b1.resize(blocks * hidden);
  model.block_w2.resize(blocks * hidden * hidden);
  model.block_b2.resize(blocks * hidden);
  for (size_t block = 0; block < blocks; ++block) {
    read_float_values(input, model.block_w1.data() + block * hidden * hidden,
                      hidden * hidden, "block_w1");
    read_float_values(input, model.block_b1.data() + block * hidden, hidden, "block_b1");
    read_float_values(input, model.block_w2.data() + block * hidden * hidden,
                      hidden * hidden, "block_w2");
    read_float_values(input, model.block_b2.data() + block * hidden, hidden, "block_b2");
  }
  read_float_vector(input, model.policy_w, actions * hidden, "policy_w");
  read_float_vector(input, model.policy_b, actions, "policy_b");
  read_float_vector(input, model.value_w,
                    static_cast<size_t>(model.value_outputs) * hidden, "value_w");
  read_float_vector(input, model.value_b, static_cast<size_t>(model.value_outputs), "value_b");
  return model;
}

std::vector<float> encode_native_mp_features(const cczero::State& state,
                                             const cczero::Board& board,
                                             const cczero::RuleProfile& rules,
                                             int perspective_player) {
  std::vector<float> features;
  features.reserve(static_cast<size_t>(kNativeMpFeatureSize));
  for (int offset = 0; offset < cczero::kMaxPlayers; ++offset) {
    const int seat = offset < rules.player_count
                         ? (perspective_player + offset) % rules.player_count
                         : cczero::kInvalid;
    for (int id = 0; id < cczero::kBoardSize; ++id) {
      features.push_back(seat != cczero::kInvalid &&
                                 state.cells.at(static_cast<size_t>(id)) == seat
                             ? 1.0f
                             : 0.0f);
    }
  }
  for (int id = 0; id < cczero::kBoardSize; ++id) {
    features.push_back(state.cells.at(static_cast<size_t>(id)) == cczero::kEmpty ? 1.0f
                                                                                  : 0.0f);
  }
  features.push_back(rules.player_count == 3 ? 1.0f : 0.0f);
  features.push_back(rules.player_count == 4 ? 1.0f : 0.0f);
  features.push_back(rules.player_count == 6 ? 1.0f : 0.0f);
  for (int player = 0; player < cczero::kMaxPlayers; ++player) {
    features.push_back(player == perspective_player ? 1.0f : 0.0f);
  }
  features.push_back(std::clamp(static_cast<float>(state.ply) / 900.0f, 0.0f, 1.0f));
  const int phase = multiplayer_phase(state, board, rules, perspective_player, rules.max_plies);
  for (int index = 0; index < kNativeMpPhaseFeatures; ++index) {
    features.push_back(index == phase ? 1.0f : 0.0f);
  }
  if (features.size() != static_cast<size_t>(kNativeMpFeatureSize)) {
    throw std::runtime_error("internal native multiplayer feature size mismatch");
  }
  return features;
}

std::vector<float> native_mp_hidden(const NativeMultiplayerModel& model,
                                    const std::vector<float>& features) {
  const size_t hidden_size = static_cast<size_t>(model.hidden_size);
  const size_t feature_size = static_cast<size_t>(model.feature_size);
  std::vector<float> hidden(hidden_size, 0.0f);
  for (size_t row = 0; row < hidden_size; ++row) {
    float sum = model.input_b.at(row);
    const size_t offset = row * feature_size;
    for (size_t col = 0; col < feature_size; ++col) {
      sum += model.input_w.at(offset + col) * features.at(col);
    }
    hidden.at(row) = std::max(0.0f, sum);
  }
  std::vector<float> tmp(hidden_size, 0.0f);
  std::vector<float> residual(hidden_size, 0.0f);
  for (int block = 0; block < model.blocks; ++block) {
    const size_t block_offset = static_cast<size_t>(block) * hidden_size * hidden_size;
    const size_t bias_offset = static_cast<size_t>(block) * hidden_size;
    for (size_t row = 0; row < hidden_size; ++row) {
      float sum = model.block_b1.at(bias_offset + row);
      const size_t offset = block_offset + row * hidden_size;
      for (size_t col = 0; col < hidden_size; ++col) {
        sum += model.block_w1.at(offset + col) * hidden.at(col);
      }
      tmp.at(row) = std::max(0.0f, sum);
    }
    for (size_t row = 0; row < hidden_size; ++row) {
      float sum = model.block_b2.at(bias_offset + row);
      const size_t offset = block_offset + row * hidden_size;
      for (size_t col = 0; col < hidden_size; ++col) {
        sum += model.block_w2.at(offset + col) * tmp.at(col);
      }
      residual.at(row) = sum;
    }
    for (size_t row = 0; row < hidden_size; ++row) {
      hidden.at(row) = std::max(0.0f, hidden.at(row) + residual.at(row));
    }
  }
  return hidden;
}

double sigmoid(double value) {
  if (value >= 0.0) {
    const double z = std::exp(-value);
    return 1.0 / (1.0 + z);
  }
  const double z = std::exp(value);
  return z / (1.0 + z);
}

double native_mp_value(const NativeMultiplayerModel& model,
                       const std::vector<float>& hidden) {
  double value = model.value_b.at(0);
  for (size_t i = 0; i < hidden.size(); ++i) {
    value += static_cast<double>(model.value_w.at(i)) * hidden.at(i);
  }
  return std::clamp(sigmoid(value), 0.0, 1.0);
}

std::vector<double> native_mp_values(const NativeMultiplayerModel& model,
                                     const std::vector<float>& hidden) {
  std::vector<double> values(static_cast<size_t>(model.value_outputs), 0.0);
  const size_t hidden_size = hidden.size();
  for (int output = 0; output < model.value_outputs; ++output) {
    double value = model.value_b.at(static_cast<size_t>(output));
    const size_t offset = static_cast<size_t>(output) * hidden_size;
    for (size_t i = 0; i < hidden_size; ++i) {
      value += static_cast<double>(model.value_w.at(offset + i)) * hidden.at(i);
    }
    values.at(static_cast<size_t>(output)) = std::clamp(sigmoid(value), 0.0, 1.0);
  }
  return values;
}

double native_mp_policy_logit(const NativeMultiplayerModel& model,
                              const std::vector<float>& hidden, int action) {
  double value = model.policy_b.at(static_cast<size_t>(action));
  const size_t offset = static_cast<size_t>(action) * static_cast<size_t>(model.hidden_size);
  for (size_t i = 0; i < hidden.size(); ++i) {
    value += static_cast<double>(model.policy_w.at(offset + i)) * hidden.at(i);
  }
  return value;
}

double placement_score_for_group(const cczero::RuleProfile& rules, int first_index,
                                 int group_size) {
  double total = 0.0;
  for (int offset = 0; offset < group_size; ++offset) {
    total += rules.placement_points.at(static_cast<size_t>(first_index + offset));
  }
  return total / static_cast<double>(group_size);
}

MultiplayerOutcome score_multiplayer_outcome(const cczero::State& state,
                                             const cczero::Board& board,
                                             const cczero::RuleProfile& rules,
                                             const std::vector<int>& finish_round,
                                             const std::string& reason) {
  MultiplayerOutcome outcome;
  outcome.reason = reason;
  outcome.placements.assign(static_cast<size_t>(rules.player_count), 0);
  outcome.scores.assign(static_cast<size_t>(rules.player_count), 0.0);

  int placed = 0;
  std::map<int, std::vector<int>> finished_by_round;
  for (int player = 0; player < rules.player_count; ++player) {
    if (finish_round.at(static_cast<size_t>(player)) >= 0) {
      finished_by_round[finish_round.at(static_cast<size_t>(player))].push_back(player);
    }
  }
  for (auto& [round, players] : finished_by_round) {
    (void)round;
    std::sort(players.begin(), players.end());
    const int place = placed + 1;
    const double score = placement_score_for_group(rules, placed, static_cast<int>(players.size()));
    for (int player : players) {
      outcome.placements.at(static_cast<size_t>(player)) = place;
      outcome.scores.at(static_cast<size_t>(player)) = score;
    }
    placed += static_cast<int>(players.size());
  }

  if (reason == "no_legal_moves") {
    std::vector<int> remaining_players;
    for (int player = 0; player < rules.player_count; ++player) {
      if (outcome.placements.at(static_cast<size_t>(player)) == 0) {
        remaining_players.push_back(player);
      }
    }
    if (!remaining_players.empty()) {
      const int place = placed + 1;
      const double score =
          placement_score_for_group(rules, placed, static_cast<int>(remaining_players.size()));
      for (int player : remaining_players) {
        outcome.placements.at(static_cast<size_t>(player)) = place;
        outcome.scores.at(static_cast<size_t>(player)) = score;
      }
    }
    for (int player = 0; player < rules.player_count; ++player) {
      if (outcome.placements.at(static_cast<size_t>(player)) == 1) {
        outcome.winner_seats.push_back(player);
      }
    }
    return outcome;
  }

  struct ProgressRank {
    int player = 0;
    int goal_count = 0;
    int distance = 0;
    int home_count = 0;
  };
  std::vector<ProgressRank> remaining;
  for (int player = 0; player < rules.player_count; ++player) {
    if (outcome.placements.at(static_cast<size_t>(player)) != 0) {
      continue;
    }
    remaining.push_back(ProgressRank{
        player,
        multiplayer_pieces_in_goal(state, board, rules, player),
        multiplayer_total_goal_distance(state, board, rules, player),
        multiplayer_pieces_in_home(state, board, rules, player),
    });
  }
  std::sort(remaining.begin(), remaining.end(), [](const ProgressRank& a,
                                                   const ProgressRank& b) {
    if (a.goal_count != b.goal_count) {
      return a.goal_count > b.goal_count;
    }
    if (a.distance != b.distance) {
      return a.distance < b.distance;
    }
    if (a.home_count != b.home_count) {
      return a.home_count < b.home_count;
    }
    return a.player < b.player;
  });

  size_t index = 0;
  while (index < remaining.size()) {
    size_t end = index + 1;
    while (end < remaining.size() &&
           remaining.at(end).goal_count == remaining.at(index).goal_count &&
           remaining.at(end).distance == remaining.at(index).distance &&
           remaining.at(end).home_count == remaining.at(index).home_count) {
      ++end;
    }
    const int place = placed + 1;
    const int group_size = static_cast<int>(end - index);
    const double score = placement_score_for_group(rules, placed, group_size);
    for (size_t at = index; at < end; ++at) {
      const int player = remaining.at(at).player;
      outcome.placements.at(static_cast<size_t>(player)) = place;
      outcome.scores.at(static_cast<size_t>(player)) = score;
    }
    placed += group_size;
    index = end;
  }

  for (int player = 0; player < rules.player_count; ++player) {
    if (outcome.placements.at(static_cast<size_t>(player)) == 1) {
      outcome.winner_seats.push_back(player);
    }
  }
  return outcome;
}

std::vector<int> multiplayer_finish_round_snapshot(const cczero::State& state,
                                                   const cczero::Board& board,
                                                   const cczero::RuleProfile& rules) {
  std::vector<int> finish_round(static_cast<size_t>(rules.player_count), -1);
  for (int player = 0; player < rules.player_count; ++player) {
    if (multiplayer_pieces_in_goal(state, board, rules, player) >= cczero::kPiecesPerPlayer) {
      finish_round.at(static_cast<size_t>(player)) =
          std::max(0, state.ply / std::max(1, rules.player_count));
    }
  }
  return finish_round;
}

std::vector<double> multiplayer_no_legal_draw_value(const cczero::State& state,
                                                    const cczero::Board& board,
                                                    const cczero::RuleProfile& rules) {
  return score_multiplayer_outcome(state, board, rules,
                                   multiplayer_finish_round_snapshot(state, board, rules),
                                   "no_legal_moves")
      .scores;
}

struct MultiplayerMctsChild {
  cczero::Move move;
  double prior = 0.0;
  int visits = 0;
  int child_index = -1;
  std::vector<double> value_sum;
};

struct MultiplayerMctsNode {
  cczero::State state;
  bool expanded = false;
  int visits = 0;
  std::vector<double> value_sum;
  std::vector<MultiplayerMctsChild> children;
};

struct MultiplayerMctsResult {
  cczero::Move move;
  std::vector<cczero::Move> legal;
  std::vector<int> visits;
  std::vector<double> priors;
  int simulations = 0;
};

bool multiplayer_mcts_terminal(const cczero::State& state, const cczero::Board& board,
                               const cczero::RuleProfile& rules);

std::vector<double> multiplayer_leaf_value(const cczero::State& state,
                                           const cczero::Board& board,
                                           const cczero::RuleProfile& rules,
                                           const NativeMultiplayerModel* mp_model) {
  if (mp_model != nullptr && !multiplayer_mcts_terminal(state, board, rules)) {
    return native_mp_value_vector(*mp_model, state, board, rules);
  }
  std::vector<int> finish_round = multiplayer_finish_round_snapshot(state, board, rules);
  return score_multiplayer_outcome(state, board, rules, finish_round, "mcts_leaf").scores;
}

bool multiplayer_mcts_terminal(const cczero::State& state, const cczero::Board& board,
                               const cczero::RuleProfile& rules) {
  if (state.ply >= rules.max_plies) {
    return true;
  }
  bool any_finished = false;
  for (int player = 0; player < rules.player_count; ++player) {
    if (multiplayer_pieces_in_goal(state, board, rules, player) >= cczero::kPiecesPerPlayer) {
      any_finished = true;
      break;
    }
  }
  return any_finished && state.ply % std::max(1, rules.player_count) == 0;
}

void add_value(std::vector<double>& sum, const std::vector<double>& value) {
  if (sum.size() < value.size()) {
    sum.resize(value.size(), 0.0);
  }
  for (size_t i = 0; i < value.size(); ++i) {
    sum.at(i) += value.at(i);
  }
}

std::vector<double> run_multiplayer_mcts_simulation(
    std::vector<MultiplayerMctsNode>& nodes, int node_index, const cczero::Board& board,
    const cczero::RuleProfile& rules, MovegenBackend movegen, const PolicyModel* prior_model,
    const NativeMultiplayerModel* mp_model, MlpWorkspace* adapter_workspace,
    InferenceBackend adapter_backend, double cpuct) {
  MultiplayerMctsNode& node = nodes.at(static_cast<size_t>(node_index));
  if (multiplayer_mcts_terminal(node.state, board, rules)) {
    const std::vector<double> value = multiplayer_leaf_value(node.state, board, rules, mp_model);
    ++node.visits;
    add_value(node.value_sum, value);
    return value;
  }

  if (!node.expanded) {
    const std::vector<cczero::Move> legal =
        legal_moves_with_backend(node.state, board, rules, movegen);
    if (legal.empty()) {
      node.expanded = true;
      const std::vector<double> value =
          multiplayer_no_legal_draw_value(node.state, board, rules);
      ++node.visits;
      add_value(node.value_sum, value);
      return value;
    }
    std::vector<double> priors;
    if (mp_model != nullptr) {
      priors = native_mp_priors(*mp_model, node.state, board, rules, legal, 1.0);
    } else if (prior_model != nullptr) {
      priors = iter60_adapter_priors(*prior_model, node.state, board, rules, legal, 1.0,
                                     adapter_workspace, adapter_backend);
    } else {
      priors = uniform_multiplayer_priors(legal.size());
    }
    node.children.reserve(legal.size());
    for (size_t i = 0; i < legal.size(); ++i) {
      MultiplayerMctsChild child;
      child.move = legal.at(i);
      child.prior = i < priors.size() ? priors.at(i) : 0.0;
      child.value_sum.assign(static_cast<size_t>(rules.player_count), 0.0);
      node.children.push_back(std::move(child));
    }
    node.expanded = true;
    const std::vector<double> value = multiplayer_leaf_value(node.state, board, rules, mp_model);
    ++node.visits;
    add_value(node.value_sum, value);
    return value;
  }

  if (node.children.empty()) {
    const std::vector<double> value = multiplayer_no_legal_draw_value(node.state, board, rules);
    ++node.visits;
    add_value(node.value_sum, value);
    return value;
  }

  const int player = node.state.player_to_move;
  const double parent_sqrt = std::sqrt(static_cast<double>(std::max(1, node.visits)));
  int best_index = 0;
  double best_score = -std::numeric_limits<double>::infinity();
  for (size_t i = 0; i < node.children.size(); ++i) {
    const MultiplayerMctsChild& child = node.children.at(i);
    const double q = child.visits == 0
                         ? 0.5
                         : child.value_sum.at(static_cast<size_t>(player)) /
                               static_cast<double>(child.visits);
    const double u =
        cpuct * child.prior * parent_sqrt / (1.0 + static_cast<double>(child.visits));
    const double score = q + u;
    if (score > best_score) {
      best_score = score;
      best_index = static_cast<int>(i);
    }
  }

  int child_node_index = -1;
  {
    MultiplayerMctsChild& child = node.children.at(static_cast<size_t>(best_index));
    if (child.child_index < 0) {
      cczero::State child_state = node.state;
      if (!cczero::apply_move(child_state, child.move, rules)) {
        const std::vector<double> value = multiplayer_leaf_value(node.state, board, rules, mp_model);
        ++node.visits;
        add_value(node.value_sum, value);
        return value;
      }
      child.child_index = static_cast<int>(nodes.size());
      MultiplayerMctsNode child_node;
      child_node.state = child_state;
      child_node.value_sum.assign(static_cast<size_t>(rules.player_count), 0.0);
      nodes.push_back(std::move(child_node));
    }
    child_node_index = child.child_index;
  }

  const std::vector<double> value = run_multiplayer_mcts_simulation(
      nodes, child_node_index, board, rules, movegen, prior_model, mp_model, adapter_workspace,
      adapter_backend, cpuct);
  MultiplayerMctsNode& refreshed = nodes.at(static_cast<size_t>(node_index));
  MultiplayerMctsChild& child = refreshed.children.at(static_cast<size_t>(best_index));
  ++child.visits;
  add_value(child.value_sum, value);
  ++refreshed.visits;
  add_value(refreshed.value_sum, value);
  return value;
}

MultiplayerMctsResult run_multiplayer_vector_mcts(const cczero::State& state,
                                                  const cczero::Board& board,
                                                  const cczero::RuleProfile& rules,
                                                  MovegenBackend movegen,
                                                  const PolicyModel* prior_model,
                                                  const NativeMultiplayerModel* mp_model,
                                                  InferenceBackend adapter_backend,
                                                  int simulations, double cpuct) {
  std::vector<MultiplayerMctsNode> nodes;
  nodes.reserve(static_cast<size_t>(std::max(2, simulations + 1)));
  MultiplayerMctsNode root;
  root.state = state;
  root.value_sum.assign(static_cast<size_t>(rules.player_count), 0.0);
  nodes.push_back(std::move(root));
  MlpWorkspace adapter_workspace;
  for (int simulation = 0; simulation < simulations; ++simulation) {
    run_multiplayer_mcts_simulation(nodes, 0, board, rules, movegen, prior_model, mp_model,
                                    &adapter_workspace, adapter_backend, cpuct);
  }

  const MultiplayerMctsNode& root_node = nodes.front();
  MultiplayerMctsResult result;
  result.simulations = simulations;
  result.legal.reserve(root_node.children.size());
  result.visits.reserve(root_node.children.size());
  result.priors.reserve(root_node.children.size());
  int best_index = -1;
  for (size_t i = 0; i < root_node.children.size(); ++i) {
    const MultiplayerMctsChild& child = root_node.children.at(i);
    result.legal.push_back(child.move);
    result.visits.push_back(child.visits);
    result.priors.push_back(child.prior);
    if (best_index < 0 || child.visits > root_node.children.at(static_cast<size_t>(best_index)).visits ||
        (child.visits == root_node.children.at(static_cast<size_t>(best_index)).visits &&
         child.prior > root_node.children.at(static_cast<size_t>(best_index)).prior)) {
      best_index = static_cast<int>(i);
    }
  }
  if (best_index >= 0) {
    result.move = root_node.children.at(static_cast<size_t>(best_index)).move;
  }
  return result;
}

cczero::Move sample_multiplayer_mcts_move(const MultiplayerMctsResult& search,
                                          double temperature, std::mt19937_64& rng) {
  if (search.legal.empty()) {
    return cczero::Move{};
  }
  std::vector<double> weights;
  weights.reserve(search.legal.size());
  double total = 0.0;
  const double exponent = 1.0 / temperature;
  for (int visits : search.visits) {
    const double weight = visits > 0 ? std::pow(static_cast<double>(visits), exponent) : 0.0;
    weights.push_back(weight);
    total += weight;
  }
  if (weights.size() != search.legal.size() || total <= 0.0 || !std::isfinite(total)) {
    weights = search.priors;
  }
  if (weights.size() != search.legal.size()) {
    weights = uniform_multiplayer_priors(search.legal.size());
  }
  return search.legal.at(sample_multiplayer_policy_index(weights, rng));
}

void write_multiplayer_compact_record(std::ostream& out, const MultiplayerRecord& record,
                                      const cczero::RuleProfile& rules,
                                      const MultiplayerOutcome& outcome,
                                      const std::string& model_id) {
  const int player_count = rules.player_count;
  std::vector<int> seat_order;
  seat_order.reserve(static_cast<size_t>(player_count));
  for (int player = 0; player < player_count; ++player) {
    seat_order.push_back(player);
  }
  int visit_sum = 0;
  for (int visits : record.visits) {
    visit_sum += visits;
  }

  out << "{\"type\":\"selfplay_position\",\"schema\":\"cczero.selfplay.compact.mp.v1\""
      << ",\"game_id\":" << record.game_id
      << ",\"seed\":" << record.seed
      << ",\"ply\":" << record.ply
      << ",\"player_count\":" << player_count
      << ",\"player\":" << record.player
      << ",\"seat\":" << record.player
      << ",\"seat_order\":";
  write_int_array_json(out, seat_order);
  out << ",\"hash\":\"";
  write_hex_hash(out, record.hash);
  out << "\",\"cells\":\"" << record.cells
      << "\",\"model_id\":\"" << cczero::json_escape(model_id)
      << "\",\"rule_profile\":\"" << cczero::json_escape(rules.name)
      << "\",\"winner_seats\":";
  write_int_array_json(out, outcome.winner_seats);
  out << ",\"placements\":";
  write_int_array_json(out, outcome.placements);
  out << ",\"score_vector\":";
  write_double_array_json(out, outcome.scores);
  out << ",\"result_vector\":";
  write_double_array_json(out, outcome.scores);
  out << ",\"terminal_reason\":\"" << cczero::json_escape(outcome.reason)
      << "\",\"phase\":" << record.phase
      << ",\"requested_simulations\":" << record.requested_simulations
      << ",\"actual_simulations\":" << record.actual_simulations
      << ",\"chosen_action\":" << record.chosen_action
      << ",\"legal_count\":" << record.actions.size()
      << ",\"visit_sum\":" << visit_sum
      << ",\"actions\":";
  write_int_array_json(out, record.actions);
  out << ",\"visits\":";
  write_int_array_json(out, record.visits);
  out << ",\"priors\":";
  write_double_array_json(out, record.priors);
  out << "}\n";
}

std::optional<size_t> json_value_start(const std::string& line, const std::string& key) {
  const std::string needle = "\"" + key + "\"";
  const size_t start = line.find(needle);
  if (start == std::string::npos) {
    return std::nullopt;
  }
  size_t pos = start + needle.size();
  while (pos < line.size() && std::isspace(static_cast<unsigned char>(line.at(pos)))) {
    ++pos;
  }
  if (pos >= line.size() || line.at(pos) != ':') {
    return std::nullopt;
  }
  ++pos;
  while (pos < line.size() && std::isspace(static_cast<unsigned char>(line.at(pos)))) {
    ++pos;
  }
  return pos;
}

std::optional<std::string> json_string_field(const std::string& line, const std::string& key) {
  std::optional<size_t> value_start = json_value_start(line, key);
  if (!value_start.has_value() || value_start.value() >= line.size() ||
      line.at(value_start.value()) != '"') {
    return std::nullopt;
  }
  size_t pos = value_start.value() + 1;
  std::string out;
  while (pos < line.size()) {
    const char ch = line.at(pos++);
    if (ch == '"') {
      return out;
    }
    if (ch == '\\' && pos < line.size()) {
      out.push_back(line.at(pos++));
    } else {
      out.push_back(ch);
    }
  }
  return std::nullopt;
}

std::optional<int> json_int_field(const std::string& line, const std::string& key) {
  std::optional<size_t> value_start = json_value_start(line, key);
  if (!value_start.has_value()) {
    return std::nullopt;
  }
  size_t pos = value_start.value();
  size_t end = pos;
  if (end < line.size() && line.at(end) == '-') {
    ++end;
  }
  const size_t digit_start = end;
  while (end < line.size() && std::isdigit(static_cast<unsigned char>(line.at(end)))) {
    ++end;
  }
  if (end == digit_start) {
    return std::nullopt;
  }
  return std::stoi(line.substr(pos, end - pos));
}

std::optional<double> json_double_field(const std::string& line, const std::string& key) {
  std::optional<size_t> value_start = json_value_start(line, key);
  if (!value_start.has_value()) {
    return std::nullopt;
  }
  size_t pos = value_start.value();
  size_t end = pos;
  if (end < line.size() && (line.at(end) == '-' || line.at(end) == '+')) {
    ++end;
  }
  bool saw_digit = false;
  while (end < line.size() && std::isdigit(static_cast<unsigned char>(line.at(end)))) {
    saw_digit = true;
    ++end;
  }
  if (end < line.size() && line.at(end) == '.') {
    ++end;
    while (end < line.size() && std::isdigit(static_cast<unsigned char>(line.at(end)))) {
      saw_digit = true;
      ++end;
    }
  }
  if (!saw_digit) {
    return std::nullopt;
  }
  if (end < line.size() && (line.at(end) == 'e' || line.at(end) == 'E')) {
    size_t exp_end = end + 1;
    if (exp_end < line.size() && (line.at(exp_end) == '-' || line.at(exp_end) == '+')) {
      ++exp_end;
    }
    const size_t exp_digits = exp_end;
    while (exp_end < line.size() &&
           std::isdigit(static_cast<unsigned char>(line.at(exp_end)))) {
      ++exp_end;
    }
    if (exp_end > exp_digits) {
      end = exp_end;
    }
  }
  return std::stod(line.substr(pos, end - pos));
}

std::optional<std::vector<double>> json_double_array_field(const std::string& line,
                                                           const std::string& key) {
  std::optional<size_t> value_start = json_value_start(line, key);
  if (!value_start.has_value() || value_start.value() >= line.size() ||
      line.at(value_start.value()) != '[') {
    return std::nullopt;
  }
  size_t pos = value_start.value() + 1;
  std::vector<double> values;
  while (pos < line.size()) {
    while (pos < line.size() &&
           (std::isspace(static_cast<unsigned char>(line.at(pos))) || line.at(pos) == ',')) {
      ++pos;
    }
    if (pos >= line.size()) {
      return std::nullopt;
    }
    if (line.at(pos) == ']') {
      return values;
    }
    size_t end = pos;
    if (end < line.size() && (line.at(end) == '-' || line.at(end) == '+')) {
      ++end;
    }
    bool saw_digit = false;
    while (end < line.size() && std::isdigit(static_cast<unsigned char>(line.at(end)))) {
      saw_digit = true;
      ++end;
    }
    if (end < line.size() && line.at(end) == '.') {
      ++end;
      while (end < line.size() && std::isdigit(static_cast<unsigned char>(line.at(end)))) {
        saw_digit = true;
        ++end;
      }
    }
    if (!saw_digit) {
      return std::nullopt;
    }
    if (end < line.size() && (line.at(end) == 'e' || line.at(end) == 'E')) {
      size_t exp_end = end + 1;
      if (exp_end < line.size() && (line.at(exp_end) == '-' || line.at(exp_end) == '+')) {
        ++exp_end;
      }
      const size_t exp_digits = exp_end;
      while (exp_end < line.size() &&
             std::isdigit(static_cast<unsigned char>(line.at(exp_end)))) {
        ++exp_end;
      }
      if (exp_end > exp_digits) {
        end = exp_end;
      }
    }
    values.push_back(std::stod(line.substr(pos, end - pos)));
    pos = end;
  }
  return std::nullopt;
}

std::vector<double> json_first_double_array_field(const std::string& line,
                                                  const std::vector<std::string>& keys) {
  for (const std::string& key : keys) {
    std::optional<std::vector<double>> values = json_double_array_field(line, key);
    if (values.has_value()) {
      return std::move(values.value());
    }
  }
  return {};
}

std::optional<bool> json_bool_field(const std::string& line, const std::string& key) {
  std::optional<size_t> value_start = json_value_start(line, key);
  if (!value_start.has_value()) {
    return std::nullopt;
  }
  size_t pos = value_start.value();
  if (line.compare(pos, 4, "true") == 0) {
    return true;
  }
  if (line.compare(pos, 5, "false") == 0) {
    return false;
  }
  return std::nullopt;
}

std::optional<uint64_t> json_uint64_field(const std::string& line, const std::string& key) {
  std::optional<size_t> value_start = json_value_start(line, key);
  if (!value_start.has_value()) {
    return std::nullopt;
  }
  size_t pos = value_start.value();
  size_t end = pos;
  while (end < line.size() && std::isdigit(static_cast<unsigned char>(line.at(end)))) {
    ++end;
  }
  if (end == pos) {
    return std::nullopt;
  }
  return static_cast<uint64_t>(std::stoull(line.substr(pos, end - pos)));
}

double normalized_policy_entropy(const std::vector<double>& weights) {
  double total = 0.0;
  for (double weight : weights) {
    if (weight > 0.0 && std::isfinite(weight)) {
      total += weight;
    }
  }
  if (total <= 0.0 || weights.empty()) {
    return 0.0;
  }
  double entropy = 0.0;
  for (double weight : weights) {
    if (weight <= 0.0 || !std::isfinite(weight)) {
      continue;
    }
    const double p = weight / total;
    entropy -= p * std::log(p);
  }
  return entropy / std::log(static_cast<double>(std::max<size_t>(2, weights.size())));
}

double normalized_policy_surprise(const std::vector<double>& targets,
                                  const std::vector<double>& priors) {
  if (targets.empty() || targets.size() != priors.size()) {
    return 0.0;
  }
  double target_total = 0.0;
  double prior_total = 0.0;
  for (double target : targets) {
    if (target > 0.0 && std::isfinite(target)) {
      target_total += target;
    }
  }
  for (double prior : priors) {
    if (prior > 0.0 && std::isfinite(prior)) {
      prior_total += prior;
    }
  }
  if (target_total <= 0.0 || prior_total <= 0.0) {
    return 0.0;
  }
  constexpr double kEpsilon = 1.0e-12;
  double surprise = 0.0;
  for (size_t index = 0; index < targets.size(); ++index) {
    const double target = targets[index];
    if (target <= 0.0 || !std::isfinite(target)) {
      continue;
    }
    const double p = target / target_total;
    const double q = std::max(kEpsilon, priors[index] / prior_total);
    surprise += p * std::log(p / q);
  }
  return surprise / std::log(static_cast<double>(std::max<size_t>(2, targets.size())));
}

ReanalysisBudgetDecision choose_reanalysis_budget(const std::string& line, int phase,
                                                  int default_simulations,
                                                  const ReanalysisBudgetConfig& budget_config) {
  ReanalysisBudgetDecision decision;
  decision.simulations = default_simulations;
  if (!budget_config.enabled) {
    decision.reasons = {"default"};
    return decision;
  }
  std::vector<double> targets =
      json_first_double_array_field(line, {"policy_target", "p", "visits", "v"});
  std::vector<double> priors = json_first_double_array_field(line, {"priors", "prior"});
  decision.entropy = normalized_policy_entropy(targets);
  decision.surprise = normalized_policy_surprise(targets, priors);

  if (budget_config.routing_mode == "complexity") {
    const bool high = decision.entropy >= budget_config.high_entropy_threshold ||
                      decision.surprise >= budget_config.high_surprise_threshold;
    const bool low = decision.entropy <= budget_config.low_entropy_threshold &&
                     decision.surprise <= budget_config.low_surprise_threshold;
    if (high) {
      decision.simulations = budget_config.high_complexity_simulations;
      if (decision.entropy >= budget_config.high_entropy_threshold) {
        decision.reasons.push_back("high_entropy");
      }
      if (decision.surprise >= budget_config.high_surprise_threshold) {
        decision.reasons.push_back("high_surprise");
      }
    } else if (low) {
      decision.simulations = budget_config.low_complexity_simulations;
      decision.reasons = {"low_complexity"};
    } else {
      decision.reasons = {"medium_complexity"};
    }
  } else if (budget_config.routing_mode == "phase") {
    decision.reasons = {"default"};
    if (phase <= 0 && budget_config.opening_simulations > 0) {
      decision.simulations = budget_config.opening_simulations;
      decision.reasons = {"opening"};
    } else if (phase == 1 && budget_config.midgame_simulations > 0) {
      decision.simulations = budget_config.midgame_simulations;
      decision.reasons = {"midgame"};
    } else if (phase >= 2 && budget_config.conversion_simulations > 0) {
      decision.simulations = budget_config.conversion_simulations;
      decision.reasons = {"conversion"};
    }
    if (budget_config.high_entropy_simulations > 0 &&
        decision.entropy >= budget_config.high_entropy_threshold) {
      decision.simulations = std::max(decision.simulations,
                                      budget_config.high_entropy_simulations);
      decision.reasons.push_back("high_entropy");
    }
    if (budget_config.high_surprise_simulations > 0 &&
        decision.surprise >= budget_config.high_surprise_threshold) {
      decision.simulations = std::max(decision.simulations,
                                      budget_config.high_surprise_simulations);
      decision.reasons.push_back("high_surprise");
    }
  } else {
    throw std::runtime_error("unknown reanalysis routing mode: " + budget_config.routing_mode);
  }
  decision.simulations = std::max(1, decision.simulations);
  if (decision.reasons.empty()) {
    decision.reasons = {"default"};
  }
  return decision;
}

std::string budget_counts_string(const std::map<int, int>& counts) {
  std::ostringstream out;
  bool first = true;
  for (const auto& [budget, count] : counts) {
    if (!first) {
      out << ",";
    }
    first = false;
    out << budget << ":" << count;
  }
  return out.str();
}

std::string string_counts_string(const std::map<std::string, int>& counts) {
  std::ostringstream out;
  bool first = true;
  for (const auto& [key, count] : counts) {
    if (!first) {
      out << ",";
    }
    first = false;
    out << key << ":" << count;
  }
  return out.str();
}

std::string summary_token(std::string text) {
  for (char& ch : text) {
    const unsigned char value = static_cast<unsigned char>(ch);
    if (!std::isalnum(value) && ch != '_' && ch != '-') {
      ch = '_';
    }
  }
  return text.empty() ? "unknown" : text;
}

cczero::State state_from_compact(const std::string& cells, int player, int ply) {
  if (cells.size() != cczero::kBoardSize) {
    throw std::runtime_error("compact position has invalid cells length");
  }
  if (player < 0 || player >= cczero::kPlayers) {
    throw std::runtime_error("compact position player must be 0 or 1");
  }
  if (ply < 0) {
    throw std::runtime_error("compact position ply must be non-negative");
  }
  cczero::State state = cczero::State::empty();
  for (int id = 0; id < cczero::kBoardSize; ++id) {
    const char ch = cells.at(static_cast<size_t>(id));
    if (ch == '0') {
      state.cells.at(static_cast<size_t>(id)) = 0;
    } else if (ch == '1') {
      state.cells.at(static_cast<size_t>(id)) = 1;
    } else if (ch != '.' && ch != '_' && ch != '-') {
      throw std::runtime_error("compact position has invalid cell character");
    }
  }
  state.player_to_move = player;
  state.ply = ply;
  return state;
}

ReanalysisParsedRecord parse_reanalysis_record_line(
    const std::string& line, uint64_t seed, int default_simulations,
    const ReanalysisBudgetConfig& budget_config) {
  ReanalysisParsedRecord parsed_record;
  if (line.empty() || json_string_field(line, "type").value_or("") != "selfplay_position") {
    return parsed_record;
  }
  const std::string cells = json_string_field(line, "cells").value_or("");
  const int player = json_int_field(line, "player").value_or(0);
  const int ply = json_int_field(line, "ply").value_or(0);
  ReanalysisInput parsed;
  parsed.game_id = json_int_field(line, "game_id").value_or(0);
  parsed.record_seed = json_uint64_field(line, "seed").value_or(seed);
  parsed.player = player;
  parsed.ply = ply;
  parsed.result_value = json_int_field(line, "result").value_or(0);
  parsed.state = state_from_compact(cells, player, ply);
  const int phase = json_int_field(line, "phase").value_or(0);
  const ReanalysisBudgetDecision budget_decision =
      choose_reanalysis_budget(line, phase, default_simulations, budget_config);
  parsed.simulation_budget = budget_decision.simulations;

  const std::optional<double> score_margin = json_double_field(line, "score_margin");
  if (score_margin.has_value()) {
    parsed.has_score_margin = true;
    parsed.score_margin = std::clamp(score_margin.value(), -1.0, 1.0);
    parsed.finish_margin_moves = json_int_field(line, "finish_margin_moves").value_or(0);
    parsed.finish_margin_max_moves =
        json_int_field(line, "finish_margin_max_moves").value_or(kFinishMarginMaxMoves);
    parsed.finish_margin_capped = json_bool_field(line, "finish_margin_capped").value_or(false);
  }

  parsed_record.accepted = true;
  parsed_record.input = std::move(parsed);
  parsed_record.budget = budget_decision.simulations;
  parsed_record.budget_reasons = budget_decision.reasons;
  return parsed_record;
}

std::vector<SuitePosition> load_suite_positions(const std::string& path, int limit) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("failed to open positions: " + path);
  }
  std::vector<SuitePosition> positions;
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty()) {
      continue;
    }
    const std::optional<std::string> cells = json_string_field(line, "cells");
    if (!cells.has_value()) {
      continue;
    }
    SuitePosition position;
    position.id = json_string_field(line, "id").value_or("position_" + std::to_string(positions.size()));
    position.cells = cells.value();
    position.player = json_int_field(line, "player").value_or(0);
    position.ply = json_int_field(line, "ply").value_or(0);
    positions.push_back(std::move(position));
    if (limit > 0 && static_cast<int>(positions.size()) >= limit) {
      break;
    }
  }
  if (positions.empty()) {
    throw std::runtime_error("positions file contained no usable rows: " + path);
  }
  return positions;
}

cczero::TerminalStatus status_from_result(int result, int player) {
  if (result == 0) {
    return cczero::TerminalStatus{true, cczero::kInvalid, true, "reanalyzed_source_draw"};
  }
  return cczero::TerminalStatus{true, result > 0 ? player : 1 - player, false,
                                "reanalyzed_source_result"};
}

int run_match(int argc, char** argv) {
  cczero::BotKind p0 = cczero::BotKind::Random;
  cczero::BotKind p1 = cczero::BotKind::GreedyDistance;
  cczero::RuleProfile rules = cczero::RuleProfile::ccz_121_ab_lg_v1();
  uint64_t seed = 1;
  int max_plies = 500;
  std::string log_path = "-";
  std::string model_path;
  std::string p0_model_path;
  std::string p1_model_path;
  std::string initial_cells;
  int initial_player = cczero::kInvalid;
  int initial_ply = 0;
  MctsOverrides mcts_overrides;
  bool has_mcts_overrides = false;
  int opening_random_plies = 0;

  for (int i = 2; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--p0") {
      const std::string value = require_arg_value(i, argc, argv, arg);
      if (!cczero::parse_bot_kind(value, &p0)) {
        throw std::runtime_error("unknown --p0 bot: " + value);
      }
    } else if (arg == "--p1") {
      const std::string value = require_arg_value(i, argc, argv, arg);
      if (!cczero::parse_bot_kind(value, &p1)) {
        throw std::runtime_error("unknown --p1 bot: " + value);
      }
    } else if (arg == "--seed") {
      seed = static_cast<uint64_t>(std::stoull(require_arg_value(i, argc, argv, arg)));
    } else if (arg == "--max-plies") {
      max_plies = std::stoi(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--log") {
      log_path = require_arg_value(i, argc, argv, arg);
    } else if (arg == "--rules") {
      rules = parse_rule_profile(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--model") {
      model_path = require_arg_value(i, argc, argv, arg);
    } else if (arg == "--p0-model") {
      p0_model_path = require_arg_value(i, argc, argv, arg);
    } else if (arg == "--p1-model") {
      p1_model_path = require_arg_value(i, argc, argv, arg);
    } else if (arg == "--initial-cells") {
      initial_cells = require_arg_value(i, argc, argv, arg);
    } else if (arg == "--initial-player") {
      initial_player = std::stoi(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--initial-ply") {
      initial_ply = std::stoi(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--mcts-movegen") {
      mcts_overrides.movegen = parse_movegen_backend(require_arg_value(i, argc, argv, arg));
      has_mcts_overrides = true;
    } else if (arg == "--mcts-simulations") {
      mcts_overrides.simulations = std::stoi(require_arg_value(i, argc, argv, arg));
      has_mcts_overrides = true;
    } else if (arg == "--mcts-cpuct") {
      mcts_overrides.cpuct = std::stod(require_arg_value(i, argc, argv, arg));
      has_mcts_overrides = true;
    } else if (arg == "--mcts-temperature") {
      mcts_overrides.temperature = std::stod(require_arg_value(i, argc, argv, arg));
      has_mcts_overrides = true;
    } else if (arg == "--mcts-temperature-plies") {
      mcts_overrides.temperature_plies = std::stoi(require_arg_value(i, argc, argv, arg));
      has_mcts_overrides = true;
    } else if (arg == "--mcts-anti-draw-logit-scale") {
      mcts_overrides.anti_draw_logit_scale = std::stod(require_arg_value(i, argc, argv, arg));
      has_mcts_overrides = true;
    } else if (arg == "--mcts-progress-prior-scale") {
      mcts_overrides.progress_prior_scale = std::stod(require_arg_value(i, argc, argv, arg));
      has_mcts_overrides = true;
    } else if (arg == "--mcts-home-pressure-scale") {
      mcts_overrides.home_pressure_scale = std::stod(require_arg_value(i, argc, argv, arg));
      has_mcts_overrides = true;
    } else if (arg == "--mcts-inference-backend") {
      mcts_overrides.inference_backend =
          parse_inference_backend(require_arg_value(i, argc, argv, arg));
      has_mcts_overrides = true;
    } else if (arg == "--mcts-inference-batch-size") {
      mcts_overrides.inference_batch_size = std::stoi(require_arg_value(i, argc, argv, arg));
      has_mcts_overrides = true;
    } else if (arg == "--opening-random-plies") {
      opening_random_plies = std::stoi(require_arg_value(i, argc, argv, arg));
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }

  std::optional<cczero::State> start_state;
  if (!initial_cells.empty()) {
    if (initial_player == cczero::kInvalid) {
      throw std::runtime_error("--initial-cells requires --initial-player 0|1");
    }
    start_state = state_from_compact(initial_cells, initial_player, initial_ply);
    if (opening_random_plies > 0) {
      throw std::runtime_error("--opening-random-plies cannot be combined with --initial-cells");
    }
  }

  if (p0_model_path.empty()) {
    p0_model_path = model_path;
  }
  if (p1_model_path.empty()) {
    p1_model_path = model_path;
  }
  std::optional<PolicyModel> p0_policy_model;
  std::optional<PolicyModel> p1_policy_model;
  const PolicyModel* p0_model = nullptr;
  const PolicyModel* p1_model = nullptr;
  if (!p0_model_path.empty()) {
    p0_policy_model = load_policy_model(p0_model_path);
    p0_model = &*p0_policy_model;
  }
  if (!p1_model_path.empty()) {
    if (p1_model_path == p0_model_path && p0_policy_model) {
      p1_model = &*p0_policy_model;
    } else {
      p1_policy_model = load_policy_model(p1_model_path);
      p1_model = &*p1_policy_model;
    }
  }
  std::ofstream file;
  std::ostream* out = open_output_stream(log_path, file);
  const MatchSummary summary =
      play_game(0, p0, p1, rules, seed, max_plies, out, nullptr, p0_model, p1_model,
                has_mcts_overrides ? &mcts_overrides : nullptr, opening_random_plies,
                start_state ? &*start_state : nullptr);

  std::cerr << "game complete: plies=" << summary.plies << " result=";
  if (summary.status.draw) {
    std::cerr << "draw";
  } else if (summary.status.winner != cczero::kInvalid) {
    std::cerr << "p" << summary.status.winner;
  } else {
    std::cerr << "unknown";
  }
  std::cerr << " reason=" << summary.status.reason << "\n";
  return 0;
}

int run_match_suite(int argc, char** argv) {
  cczero::BotKind p0 = cczero::BotKind::Random;
  cczero::BotKind p1 = cczero::BotKind::GreedyDistance;
  cczero::RuleProfile rules = cczero::RuleProfile::ccz_121_ab_lg_v1();
  uint64_t seed = 1;
  int max_plies = 500;
  int limit = 0;
  int workers = 1;
  bool no_swap = false;
  std::string log_path = "-";
  std::string positions_path;
  std::string model_path;
  std::string p0_model_path;
  std::string p1_model_path;
  MctsOverrides mcts_overrides;
  bool has_mcts_overrides = false;

  for (int i = 2; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--p0") {
      const std::string value = require_arg_value(i, argc, argv, arg);
      if (!cczero::parse_bot_kind(value, &p0)) {
        throw std::runtime_error("unknown --p0 bot: " + value);
      }
    } else if (arg == "--p1") {
      const std::string value = require_arg_value(i, argc, argv, arg);
      if (!cczero::parse_bot_kind(value, &p1)) {
        throw std::runtime_error("unknown --p1 bot: " + value);
      }
    } else if (arg == "--seed") {
      seed = static_cast<uint64_t>(std::stoull(require_arg_value(i, argc, argv, arg)));
    } else if (arg == "--max-plies") {
      max_plies = std::stoi(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--limit") {
      limit = std::stoi(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--workers") {
      workers = std::stoi(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--no-swap") {
      no_swap = true;
    } else if (arg == "--log") {
      log_path = require_arg_value(i, argc, argv, arg);
    } else if (arg == "--positions") {
      positions_path = require_arg_value(i, argc, argv, arg);
    } else if (arg == "--rules") {
      rules = parse_rule_profile(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--model") {
      model_path = require_arg_value(i, argc, argv, arg);
    } else if (arg == "--p0-model") {
      p0_model_path = require_arg_value(i, argc, argv, arg);
    } else if (arg == "--p1-model") {
      p1_model_path = require_arg_value(i, argc, argv, arg);
    } else if (arg == "--mcts-movegen") {
      mcts_overrides.movegen = parse_movegen_backend(require_arg_value(i, argc, argv, arg));
      has_mcts_overrides = true;
    } else if (arg == "--mcts-simulations") {
      mcts_overrides.simulations = std::stoi(require_arg_value(i, argc, argv, arg));
      has_mcts_overrides = true;
    } else if (arg == "--mcts-cpuct") {
      mcts_overrides.cpuct = std::stod(require_arg_value(i, argc, argv, arg));
      has_mcts_overrides = true;
    } else if (arg == "--mcts-anti-draw-logit-scale") {
      mcts_overrides.anti_draw_logit_scale = std::stod(require_arg_value(i, argc, argv, arg));
      has_mcts_overrides = true;
    } else if (arg == "--mcts-progress-prior-scale") {
      mcts_overrides.progress_prior_scale = std::stod(require_arg_value(i, argc, argv, arg));
      has_mcts_overrides = true;
    } else if (arg == "--mcts-home-pressure-scale") {
      mcts_overrides.home_pressure_scale = std::stod(require_arg_value(i, argc, argv, arg));
      has_mcts_overrides = true;
    } else if (arg == "--mcts-inference-backend") {
      mcts_overrides.inference_backend =
          parse_inference_backend(require_arg_value(i, argc, argv, arg));
      has_mcts_overrides = true;
    } else if (arg == "--mcts-inference-batch-size") {
      mcts_overrides.inference_batch_size = std::stoi(require_arg_value(i, argc, argv, arg));
      has_mcts_overrides = true;
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }

  if (positions_path.empty()) {
    throw std::runtime_error("match-suite requires --positions PATH");
  }
  if (workers <= 0) {
    throw std::runtime_error("--workers must be positive");
  }
  if (p0_model_path.empty()) {
    p0_model_path = model_path;
  }
  if (p1_model_path.empty()) {
    p1_model_path = model_path;
  }

  std::optional<PolicyModel> p0_policy_model;
  std::optional<PolicyModel> p1_policy_model;
  const PolicyModel* p0_model = nullptr;
  const PolicyModel* p1_model = nullptr;
  if (!p0_model_path.empty()) {
    p0_policy_model = load_policy_model(p0_model_path);
    p0_model = &*p0_policy_model;
  }
  if (!p1_model_path.empty()) {
    if (p1_model_path == p0_model_path && p0_policy_model) {
      p1_model = &*p0_policy_model;
    } else {
      p1_policy_model = load_policy_model(p1_model_path);
      p1_model = &*p1_policy_model;
    }
  }

  const std::vector<SuitePosition> positions = load_suite_positions(positions_path, limit);
  std::vector<SuiteTask> tasks;
  tasks.reserve(positions.size() * (no_swap ? 1 : 2));
  int game_id = 0;
  for (size_t index = 0; index < positions.size(); ++index) {
    const SuitePosition& position = positions.at(index);
    SuiteTask task;
    task.game_id = game_id;
    task.position_index = static_cast<int>(index);
    task.swapped = false;
    task.seed = seed + static_cast<uint64_t>(game_id);
    task.state = state_from_compact(position.cells, position.player, position.ply);
    tasks.push_back(std::move(task));
    ++game_id;
    if (!no_swap) {
      SuiteTask swapped;
      swapped.game_id = game_id;
      swapped.position_index = static_cast<int>(index);
      swapped.swapped = true;
      swapped.seed = seed + static_cast<uint64_t>(game_id);
      swapped.state = state_from_compact(position.cells, position.player, position.ply);
      tasks.push_back(std::move(swapped));
      ++game_id;
    }
  }

  std::vector<SuiteOutput> outputs(tasks.size());
  const int active_workers = std::min(workers, std::max(1, static_cast<int>(tasks.size())));
  for (size_t start = 0; start < tasks.size(); start += static_cast<size_t>(active_workers)) {
    const size_t end = std::min(tasks.size(), start + static_cast<size_t>(active_workers));
    std::vector<std::future<SuiteOutput>> futures;
    futures.reserve(end - start);
    for (size_t index = start; index < end; ++index) {
      futures.push_back(std::async(std::launch::async, [&, index]() {
        const SuiteTask& task = tasks.at(index);
        const cczero::BotKind game_p0 = task.swapped ? p1 : p0;
        const cczero::BotKind game_p1 = task.swapped ? p0 : p1;
        const PolicyModel* game_p0_model = task.swapped ? p1_model : p0_model;
        const PolicyModel* game_p1_model = task.swapped ? p0_model : p1_model;
        std::ostringstream log;
        SuiteOutput output;
        output.game_id = task.game_id;
        output.summary =
            play_game(task.game_id, game_p0, game_p1, rules, task.seed, max_plies, &log,
                      nullptr, game_p0_model, game_p1_model,
                      has_mcts_overrides ? &mcts_overrides : nullptr, 0, &task.state);
        output.log = log.str();
        return output;
      }));
    }
    for (size_t local = 0; local < futures.size(); ++local) {
      SuiteOutput output = futures.at(local).get();
      outputs.at(static_cast<size_t>(output.game_id)) = std::move(output);
    }
    std::cerr << "match-suite progress: " << std::min(end, tasks.size()) << "/"
              << tasks.size() << "\n";
  }

  std::ofstream file;
  std::ostream* out = open_output_stream(log_path, file);
  for (const SuiteOutput& output : outputs) {
    *out << output.log;
  }
  std::cerr << "match-suite complete: games=" << tasks.size()
            << " positions=" << positions.size()
            << " workers=" << active_workers << "\n";
  return 0;
}

int run_perft(int argc, char** argv) {
  int depth = 2;
  std::string fixture = "initial";
  cczero::RuleProfile rules = cczero::RuleProfile::ccz_121_ab_lg_v1();
  for (int i = 2; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--depth") {
      depth = std::stoi(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--fixture") {
      fixture = require_arg_value(i, argc, argv, arg);
    } else if (arg == "--rules") {
      rules = parse_rule_profile(require_arg_value(i, argc, argv, arg));
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }

  const cczero::Board& board = cczero::Board::standard();
  cczero::State ref_state =
      fixture == "initial" ? cczero::State::initial(board, rules) : fixture_state(fixture, board);
  cczero::State fast_state = ref_state;
  const uint64_t ref = cczero::perft(ref_state, board, rules, depth, false);
  const uint64_t fast = cczero::perft(fast_state, board, rules, depth, true);

  std::cout << "{\"fixture\":\"" << cczero::json_escape(fixture)
            << "\",\"rule_profile\":\"" << cczero::json_escape(rules.name)
            << "\",\"depth\":" << depth
            << ",\"reference_nodes\":" << ref << ",\"fast_nodes\":" << fast
            << ",\"match\":" << (ref == fast ? "true" : "false") << "}\n";
  return ref == fast ? 0 : 2;
}

int run_validate_movegen(int argc, char** argv) {
  int positions = 200;
  int plies = 40;
  uint64_t seed = 1;
  cczero::RuleProfile rules = cczero::RuleProfile::ccz_121_ab_lg_v1();
  for (int i = 2; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--positions") {
      positions = std::stoi(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--plies") {
      plies = std::stoi(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--seed") {
      seed = static_cast<uint64_t>(std::stoull(require_arg_value(i, argc, argv, arg)));
    } else if (arg == "--rules") {
      rules = parse_rule_profile(require_arg_value(i, argc, argv, arg));
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }

  const cczero::Board& board = cczero::Board::standard();
  rules.max_plies = std::max(rules.max_plies, plies + 10);
  cczero::State state = cczero::State::initial(board, rules);
  std::mt19937_64 rng(seed);

  int checked = 0;
  while (checked < positions) {
    const std::vector<cczero::Move> ref = cczero::legal_moves_reference(state, board, rules);
    const std::vector<cczero::Move> fast = cczero::legal_moves_fast(state, board, rules);
    const std::vector<cczero::Move> bitboard = cczero::legal_moves_bitboard(state, board, rules);
    if (endpoint_signature(ref) != endpoint_signature(fast)) {
      throw std::runtime_error("fast/reference endpoint mismatch at ply " + std::to_string(state.ply));
    }
    if (endpoint_signature(ref) != endpoint_signature(bitboard)) {
      throw std::runtime_error("bitboard/reference endpoint mismatch at ply " +
                               std::to_string(state.ply));
    }
    for (const cczero::Move& move : fast) {
      std::string error;
      if (!cczero::validate_move_witness(state, move, board, rules, &error)) {
        throw std::runtime_error("fast move witness invalid at ply " + std::to_string(state.ply) +
                                 ": " + error);
      }
    }
    for (const cczero::Move& move : bitboard) {
      std::string error;
      if (!cczero::validate_move_witness(state, move, board, rules, &error)) {
        throw std::runtime_error("bitboard move witness invalid at ply " +
                                 std::to_string(state.ply) + ": " + error);
      }
    }
    ++checked;

    if (state.ply >= plies || ref.empty()) {
      state = cczero::State::initial(board, rules);
      continue;
    }

    std::uniform_int_distribution<size_t> dist(0, ref.size() - 1);
    cczero::apply_move(state, ref.at(dist(rng)), rules);
  }

  std::cout << "{\"checked_positions\":" << checked << ",\"seed\":" << seed
            << ",\"rule_profile\":\"" << cczero::json_escape(rules.name) << "\""
            << ",\"status\":\"ok\"}\n";
  return 0;
}

int run_benchmark_movegen(int argc, char** argv) {
  int positions = 500;
  int plies = 80;
  uint64_t seed = 1;
  cczero::RuleProfile rules = cczero::RuleProfile::ccz_121_strict_lg_v1();
  std::vector<MovegenBackend> backends = {MovegenBackend::Reference, MovegenBackend::Fast,
                                          MovegenBackend::Bitboard};
  for (int i = 2; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--positions") {
      positions = std::stoi(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--plies") {
      plies = std::stoi(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--seed") {
      seed = static_cast<uint64_t>(std::stoull(require_arg_value(i, argc, argv, arg)));
    } else if (arg == "--rules") {
      rules = parse_rule_profile(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--backends") {
      backends = parse_movegen_backend_list(require_arg_value(i, argc, argv, arg));
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }
  if (positions <= 0) {
    throw std::runtime_error("--positions must be positive");
  }

  const cczero::Board& board = cczero::Board::standard();
  rules.max_plies = std::max(rules.max_plies, plies + 10);
  std::mt19937_64 rng(seed);
  std::vector<cczero::State> states;
  states.reserve(static_cast<size_t>(positions));
  cczero::State state = cczero::State::initial(board, rules);
  while (static_cast<int>(states.size()) < positions) {
    states.push_back(state);
    const std::vector<cczero::Move> moves = cczero::legal_moves_reference(state, board, rules);
    if (state.ply >= plies || moves.empty()) {
      state = cczero::State::initial(board, rules);
      continue;
    }
    std::uniform_int_distribution<size_t> dist(0, moves.size() - 1);
    cczero::apply_move(state, moves.at(dist(rng)), rules);
  }

  std::vector<std::vector<std::pair<int, int>>> reference_signatures;
  reference_signatures.reserve(states.size());
  for (const cczero::State& sample : states) {
    reference_signatures.push_back(
        endpoint_signature(cczero::legal_moves_reference(sample, board, rules)));
  }

  std::cout << "{\"type\":\"movegen_benchmark\",\"positions\":" << positions
            << ",\"plies\":" << plies << ",\"seed\":" << seed
            << ",\"rule_profile\":\"" << cczero::json_escape(rules.name)
            << "\",\"rows\":[";
  for (size_t backend_index = 0; backend_index < backends.size(); ++backend_index) {
    const MovegenBackend backend = backends.at(backend_index);
    const auto start = std::chrono::steady_clock::now();
    uint64_t move_count = 0;
    bool ok = true;
    for (size_t i = 0; i < states.size(); ++i) {
      const std::vector<cczero::Move> moves =
          legal_moves_with_backend(states.at(i), board, rules, backend);
      move_count += moves.size();
      if (endpoint_signature(moves) != reference_signatures.at(i)) {
        ok = false;
      }
    }
    const auto end = std::chrono::steady_clock::now();
    const double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    const double seconds = std::max(1.0e-9, elapsed_ms / 1000.0);
    if (backend_index != 0) {
      std::cout << ",";
    }
    std::cout << "{\"backend\":\"" << movegen_backend_name(backend) << "\""
              << ",\"ok\":" << (ok ? "true" : "false")
              << ",\"moves\":" << move_count
              << ",\"elapsed_ms\":" << std::fixed << std::setprecision(3) << elapsed_ms
              << ",\"positions_per_sec\":" << std::setprecision(1)
              << (static_cast<double>(positions) / seconds)
              << ",\"moves_per_sec\":" << (static_cast<double>(move_count) / seconds)
              << std::defaultfloat << "}";
  }
  std::cout << "]}\n";
  return 0;
}

int run_tournament(int argc, char** argv) {
  std::vector<cczero::BotKind> bots = cczero::all_bot_kinds();
  int games_per_pair = 2;
  int max_plies = 300;
  uint64_t seed = 1;
  std::string out_path;
  cczero::RuleProfile rules = cczero::RuleProfile::ccz_121_ab_lg_v1();
  std::string model_path;

  for (int i = 2; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--bots") {
      bots = parse_bot_list(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--games") {
      games_per_pair = std::stoi(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--seed") {
      seed = static_cast<uint64_t>(std::stoull(require_arg_value(i, argc, argv, arg)));
    } else if (arg == "--max-plies") {
      max_plies = std::stoi(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--out") {
      out_path = require_arg_value(i, argc, argv, arg);
    } else if (arg == "--rules") {
      rules = parse_rule_profile(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--model") {
      model_path = require_arg_value(i, argc, argv, arg);
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }

  std::optional<PolicyModel> policy_model;
  if (!model_path.empty()) {
    policy_model = load_policy_model(model_path);
  }

  std::ofstream file;
  std::ostream* out = nullptr;
  if (!out_path.empty()) {
    out = open_output_stream(out_path, file);
  }

  std::vector<ScoreRow> rows(bots.size());
  int game_id = 0;
  for (size_t i = 0; i < bots.size(); ++i) {
    for (size_t j = i + 1; j < bots.size(); ++j) {
      for (int g = 0; g < games_per_pair; ++g) {
        for (int swap = 0; swap < 2; ++swap) {
          const size_t p0_index = swap == 0 ? i : j;
          const size_t p1_index = swap == 0 ? j : i;
          const uint64_t game_seed = seed + static_cast<uint64_t>(game_id) * 7919ULL;
          const MatchSummary summary =
              play_game(game_id, bots.at(p0_index), bots.at(p1_index), rules, game_seed, max_plies,
                        nullptr, nullptr, policy_model ? &*policy_model : nullptr,
                        policy_model ? &*policy_model : nullptr);

          rows.at(p0_index).games += 1;
          rows.at(p1_index).games += 1;
          double p0_points = 0.5;
          double p1_points = 0.5;
          if (!summary.status.draw && summary.status.winner != cczero::kInvalid) {
            p0_points = summary.status.winner == 0 ? 1.0 : 0.0;
            p1_points = 1.0 - p0_points;
          }
          rows.at(p0_index).points += p0_points;
          rows.at(p1_index).points += p1_points;
          rows.at(p0_index).wins += p0_points == 1.0 ? 1 : 0;
          rows.at(p0_index).draws += p0_points == 0.5 ? 1 : 0;
          rows.at(p0_index).losses += p0_points == 0.0 ? 1 : 0;
          rows.at(p1_index).wins += p1_points == 1.0 ? 1 : 0;
          rows.at(p1_index).draws += p1_points == 0.5 ? 1 : 0;
          rows.at(p1_index).losses += p1_points == 0.0 ? 1 : 0;

          if (out != nullptr) {
            *out << "{\"type\":\"tournament_game\",\"game_id\":" << game_id
                 << ",\"seed\":" << game_seed << ",\"p0\":\""
                 << cczero::bot_name(bots.at(p0_index)) << "\",\"p1\":\""
                 << cczero::bot_name(bots.at(p1_index)) << "\",\"rule_profile\":\""
                 << cczero::json_escape(rules.name) << "\",\"plies\":"
                 << summary.plies << ",\"draw\":"
                 << (summary.status.draw ? "true" : "false") << ",\"winner\":";
            if (summary.status.winner == cczero::kInvalid) {
              *out << "null";
            } else {
              *out << summary.status.winner;
            }
            *out << ",\"reason\":\"" << cczero::json_escape(summary.status.reason)
                 << "\"}\n";
          }
          ++game_id;
        }
      }
    }
  }

  std::cout << "bot,games,wins,draws,losses,score_rate,field_elo\n";
  for (size_t i = 0; i < bots.size(); ++i) {
    const ScoreRow& row = rows.at(i);
    const double raw = row.games == 0 ? 0.5 : row.points / static_cast<double>(row.games);
    const double score = std::clamp(raw, 0.01, 0.99);
    const double elo = 400.0 * std::log10(score / (1.0 - score));
    std::cout << cczero::bot_name(bots.at(i)) << "," << row.games << "," << row.wins
              << "," << row.draws << "," << row.losses << "," << std::fixed
              << std::setprecision(3) << raw << "," << std::setprecision(1) << elo << "\n";
  }
  return 0;
}

int run_dataset(int argc, char** argv) {
  cczero::BotKind generator = cczero::BotKind::BeamSearch;
  cczero::BotKind opponent = cczero::BotKind::Pvs;
  int games = 10;
  int max_plies = 300;
  uint64_t seed = 1;
  std::string out_path;
  cczero::RuleProfile rules = cczero::RuleProfile::ccz_121_ab_lg_v1();
  std::string model_path;

  for (int i = 2; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--bot") {
      const std::string value = require_arg_value(i, argc, argv, arg);
      if (!cczero::parse_bot_kind(value, &generator)) {
        throw std::runtime_error("unknown --bot value: " + value);
      }
    } else if (arg == "--opponent") {
      const std::string value = require_arg_value(i, argc, argv, arg);
      if (!cczero::parse_bot_kind(value, &opponent)) {
        throw std::runtime_error("unknown --opponent value: " + value);
      }
    } else if (arg == "--games") {
      games = std::stoi(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--seed") {
      seed = static_cast<uint64_t>(std::stoull(require_arg_value(i, argc, argv, arg)));
    } else if (arg == "--max-plies") {
      max_plies = std::stoi(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--out") {
      out_path = require_arg_value(i, argc, argv, arg);
    } else if (arg == "--rules") {
      rules = parse_rule_profile(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--model") {
      model_path = require_arg_value(i, argc, argv, arg);
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }

  if (out_path.empty()) {
    throw std::runtime_error("dataset requires --out PATH");
  }

  std::ofstream file;
  std::ostream* out = open_output_stream(out_path, file);
  std::optional<PolicyModel> policy_model;
  if (!model_path.empty()) {
    policy_model = load_policy_model(model_path);
  }
  int records_written = 0;
  for (int game_id = 0; game_id < games; ++game_id) {
    std::vector<DatasetRecord> records;
    const bool swap = game_id % 2 == 1;
    const cczero::BotKind p0 = swap ? opponent : generator;
    const cczero::BotKind p1 = swap ? generator : opponent;
    const MatchSummary summary =
        play_game(game_id, p0, p1, rules, seed + static_cast<uint64_t>(game_id) * 104729ULL,
                  max_plies, nullptr, &records, policy_model ? &*policy_model : nullptr,
                  policy_model ? &*policy_model : nullptr);
    for (const DatasetRecord& record : records) {
      write_dataset_record(*out, record, summary.status, generator, opponent, rules);
      ++records_written;
    }
  }

  std::cerr << "dataset complete: games=" << games << " records=" << records_written
            << " out=" << out_path << "\n";
  return 0;
}

int run_multiplayer_dataset(int argc, char** argv) {
  int games = 1;
  int max_plies = -1;
  uint64_t seed = 1;
  std::string out_path;
  std::string model_id = "random-mp-v0";
  bool model_id_set = false;
  std::string model_path;
  std::string mp_model_path;
  MultiplayerDataPolicy policy = MultiplayerDataPolicy::Random;
  double temperature = 1.0;
  int simulations = 64;
  double cpuct = 1.5;
  MovegenBackend movegen = MovegenBackend::Bitboard;
  InferenceBackend inference_backend = InferenceBackend::Auto;
  cczero::RuleProfile rules = cczero::RuleProfile::ccz_121_mp3_v1();

  for (int i = 2; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--games") {
      games = std::stoi(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--seed") {
      seed = static_cast<uint64_t>(std::stoull(require_arg_value(i, argc, argv, arg)));
    } else if (arg == "--max-plies") {
      max_plies = std::stoi(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--out") {
      out_path = require_arg_value(i, argc, argv, arg);
    } else if (arg == "--rules") {
      rules = parse_rule_profile(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--model-id") {
      model_id = require_arg_value(i, argc, argv, arg);
      model_id_set = true;
    } else if (arg == "--model") {
      model_path = require_arg_value(i, argc, argv, arg);
    } else if (arg == "--mp-model" || arg == "--native-model") {
      mp_model_path = require_arg_value(i, argc, argv, arg);
    } else if (arg == "--policy") {
      policy = parse_multiplayer_data_policy(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--temperature") {
      temperature = std::stod(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--simulations") {
      simulations = std::stoi(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--cpuct") {
      cpuct = std::stod(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--movegen") {
      movegen = parse_movegen_backend(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--inference-backend") {
      inference_backend = parse_inference_backend(require_arg_value(i, argc, argv, arg));
    } else {
      throw std::runtime_error("unknown multiplayer-dataset argument: " + arg);
    }
  }

  if (!rules.is_multiplayer() || (rules.player_count != 3 && rules.player_count != 4 &&
                                  rules.player_count != 6)) {
    throw std::runtime_error("multiplayer-dataset requires --rules mp3, mp4, or mp6");
  }
  if (out_path.empty()) {
    throw std::runtime_error("multiplayer-dataset requires --out PATH");
  }
  if (games <= 0) {
    throw std::runtime_error("--games must be positive");
  }
  if (policy == MultiplayerDataPolicy::Iter60Adapter && model_path.empty()) {
    throw std::runtime_error("multiplayer-dataset --policy iter60-adapter requires --model PATH");
  }
  if (policy == MultiplayerDataPolicy::VectorMcts && model_path.empty() &&
      mp_model_path.empty()) {
    throw std::runtime_error(
        "multiplayer-dataset --policy vector-mcts requires --model PATH or --mp-model PATH");
  }
  if (temperature <= 0.0 || !std::isfinite(temperature)) {
    throw std::runtime_error("--temperature must be finite and positive");
  }
  if (simulations <= 0) {
    throw std::runtime_error("--simulations must be positive");
  }
  if (cpuct <= 0.0 || !std::isfinite(cpuct)) {
    throw std::runtime_error("--cpuct must be finite and positive");
  }
  if (max_plies <= 0) {
    max_plies = rules.max_plies;
  }
  rules.max_plies = max_plies;
  if (policy != MultiplayerDataPolicy::Random && !model_id_set) {
    const std::filesystem::path model_fs_path(
        policy == MultiplayerDataPolicy::VectorMcts && !mp_model_path.empty() ? mp_model_path
                                                                               : model_path);
    std::string model_stem = model_fs_path.stem().string();
    if (model_stem == "model" && !model_fs_path.parent_path().empty()) {
      model_stem = model_fs_path.parent_path().filename().string();
    }
    model_id = model_stem +
               (policy == MultiplayerDataPolicy::VectorMcts
                    ? (mp_model_path.empty() ? "-vector_mcts" : "-mp_model_vector_mcts")
                    : "-own_vs_rest_adapter");
  }

  if (out_path != "-") {
    const std::filesystem::path parent = std::filesystem::path(out_path).parent_path();
    if (!parent.empty()) {
      std::filesystem::create_directories(parent);
    }
  }
  std::ofstream file;
  std::ostream* out = open_output_stream(out_path, file);
  const cczero::Board& board = cczero::Board::standard();
  std::optional<PolicyModel> adapter_model;
  if ((policy == MultiplayerDataPolicy::Iter60Adapter ||
       policy == MultiplayerDataPolicy::VectorMcts) &&
      !model_path.empty()) {
    adapter_model = load_policy_model(model_path);
  }
  std::optional<NativeMultiplayerModel> native_mp_model;
  if (!mp_model_path.empty()) {
    native_mp_model = load_native_multiplayer_model(mp_model_path);
  }
  MlpWorkspace adapter_workspace;

  int records_written = 0;
  int total_plies = 0;
  std::map<std::string, int> reasons;
  for (int game_id = 0; game_id < games; ++game_id) {
    const uint64_t game_seed = seed + static_cast<uint64_t>(game_id) * 104729ULL;
    std::mt19937_64 rng(game_seed);
    cczero::State state = cczero::State::initial(board, rules);
    std::vector<MultiplayerRecord> records;
    std::vector<int> finish_round(static_cast<size_t>(rules.player_count), -1);
    std::string reason = "max_plies";
    int first_finish_round = -1;

    while (state.ply < max_plies) {
      const int player = state.player_to_move;
      if (finish_round.at(static_cast<size_t>(player)) >= 0) {
        advance_multiplayer_turn(state, rules);
        if (first_finish_round >= 0 && state.ply % rules.player_count == 0 &&
            (state.ply - 1) / rules.player_count >= first_finish_round) {
          reason = "equal_turn_goal";
          break;
        }
        continue;
      }

      std::vector<cczero::Move> legal =
          legal_moves_with_backend(state, board, rules, movegen);
      if (legal.empty()) {
        reason = "no_legal_moves";
        break;
      }
      std::vector<double> priors;
      std::vector<int> visits;
      cczero::Move move;
      int actual_simulations = 0;
      if (policy == MultiplayerDataPolicy::VectorMcts) {
        const MultiplayerMctsResult search = run_multiplayer_vector_mcts(
            state, board, rules, movegen, adapter_model ? &*adapter_model : nullptr,
            native_mp_model ? &*native_mp_model : nullptr, inference_backend, simulations,
            cpuct);
        legal = search.legal;
        priors = search.priors;
        visits = search.visits;
        move = sample_multiplayer_mcts_move(search, temperature, rng);
        actual_simulations = search.simulations;
      } else {
        priors = policy == MultiplayerDataPolicy::Iter60Adapter
                     ? iter60_adapter_priors(*adapter_model, state, board, rules, legal,
                                             temperature, &adapter_workspace,
                                             inference_backend)
                     : uniform_multiplayer_priors(legal.size());
        move = legal.at(sample_multiplayer_policy_index(priors, rng));
        visits = one_hot_visits(legal, move);
      }
      if (!move.is_valid()) {
        reason = "no_search_move";
        break;
      }
      std::string error;
      if (!cczero::validate_move_witness(state, move, board, rules, &error)) {
        throw std::runtime_error("multiplayer data policy produced invalid move: " + error);
      }

      const cczero::State before = state;
      records.push_back(make_multiplayer_record(
          game_id, game_seed, before, move, legal, visits, priors, board, rules, max_plies,
          policy == MultiplayerDataPolicy::VectorMcts ? simulations : 0, actual_simulations));
      if (!cczero::apply_move(state, move, rules)) {
        throw std::runtime_error("failed to apply multiplayer data-policy move");
      }
      if (multiplayer_pieces_in_goal(state, board, rules, player) >= cczero::kPiecesPerPlayer &&
          finish_round.at(static_cast<size_t>(player)) < 0) {
        finish_round.at(static_cast<size_t>(player)) =
            (state.ply - 1) / std::max(1, rules.player_count);
        if (first_finish_round < 0) {
          first_finish_round = finish_round.at(static_cast<size_t>(player));
        }
      }
      if (first_finish_round >= 0 && state.ply % rules.player_count == 0 &&
          (state.ply - 1) / rules.player_count >= first_finish_round) {
        reason = "equal_turn_goal";
        break;
      }
    }

    const MultiplayerOutcome outcome =
        score_multiplayer_outcome(state, board, rules, finish_round, reason);
    for (const MultiplayerRecord& record : records) {
      write_multiplayer_compact_record(*out, record, rules, outcome, model_id);
      ++records_written;
    }
    total_plies += state.ply;
    ++reasons[reason];
  }

  std::cerr << "multiplayer-dataset complete: games=" << games
            << " records=" << records_written
            << " avg_plies=" << (games == 0 ? 0.0 : static_cast<double>(total_plies) / games)
            << " rules=" << rules.name
            << " movegen=" << movegen_backend_name(movegen)
            << " out=" << out_path
            << " reasons={";
  bool first = true;
  for (const auto& [reason, count] : reasons) {
    if (!first) {
      std::cerr << ",";
    }
    first = false;
    std::cerr << reason << ":" << count;
  }
  std::cerr << "}\n";
  return 0;
}

struct MultiplayerEvalGame {
  int game_id = 0;
  uint64_t seed = 0;
  int candidate_seat = 0;
  int plies = 0;
  std::string reason;
  std::vector<int> placements;
  std::vector<double> scores;
  std::vector<int> winner_seats;
};

MultiplayerEvalGame play_multiplayer_eval_game(
    int game_id, uint64_t game_seed, int candidate_seat, const cczero::Board& board,
    const cczero::RuleProfile& rules, MovegenBackend movegen, const PolicyModel& adapter_model,
    const NativeMultiplayerModel& candidate_model, InferenceBackend inference_backend,
    int simulations, double cpuct, double temperature) {
  std::mt19937_64 rng(game_seed);
  cczero::State state = cczero::State::initial(board, rules);
  std::vector<int> finish_round(static_cast<size_t>(rules.player_count), -1);
  std::string reason = "max_plies";
  int first_finish_round = -1;

  while (state.ply < rules.max_plies) {
    const int player = state.player_to_move;
    if (finish_round.at(static_cast<size_t>(player)) >= 0) {
      advance_multiplayer_turn(state, rules);
      if (first_finish_round >= 0 && state.ply % rules.player_count == 0 &&
          (state.ply - 1) / rules.player_count >= first_finish_round) {
        reason = "equal_turn_goal";
        break;
      }
      continue;
    }

    const std::vector<cczero::Move> legal =
        legal_moves_with_backend(state, board, rules, movegen);
    if (legal.empty()) {
      reason = "no_legal_moves";
      break;
    }

    const bool candidate_turn = player == candidate_seat;
    const MultiplayerMctsResult search = run_multiplayer_vector_mcts(
        state, board, rules, movegen, candidate_turn ? nullptr : &adapter_model,
        candidate_turn ? &candidate_model : nullptr, inference_backend, simulations, cpuct);
    const cczero::Move move = sample_multiplayer_mcts_move(search, temperature, rng);
    if (!move.is_valid()) {
      reason = "no_search_move";
      break;
    }
    std::string error;
    if (!cczero::validate_move_witness(state, move, board, rules, &error)) {
      throw std::runtime_error("multiplayer eval produced invalid move: " + error);
    }
    if (!cczero::apply_move(state, move, rules)) {
      throw std::runtime_error("failed to apply multiplayer eval move");
    }
    if (multiplayer_pieces_in_goal(state, board, rules, player) >= cczero::kPiecesPerPlayer &&
        finish_round.at(static_cast<size_t>(player)) < 0) {
      finish_round.at(static_cast<size_t>(player)) =
          (state.ply - 1) / std::max(1, rules.player_count);
      if (first_finish_round < 0) {
        first_finish_round = finish_round.at(static_cast<size_t>(player));
      }
    }
    if (first_finish_round >= 0 && state.ply % rules.player_count == 0 &&
        (state.ply - 1) / rules.player_count >= first_finish_round) {
      reason = "equal_turn_goal";
      break;
    }
  }

  const MultiplayerOutcome outcome =
      score_multiplayer_outcome(state, board, rules, finish_round, reason);
  MultiplayerEvalGame game;
  game.game_id = game_id;
  game.seed = game_seed;
  game.candidate_seat = candidate_seat;
  game.plies = state.ply;
  game.reason = outcome.reason;
  game.placements = outcome.placements;
  game.scores = outcome.scores;
  game.winner_seats = outcome.winner_seats;
  return game;
}

int run_multiplayer_eval(int argc, char** argv) {
  int games = 4;
  int max_plies = -1;
  int candidate_seat = -1;
  uint64_t seed = 1;
  std::string out_path;
  std::string candidate_model_path;
  std::string adapter_model_path;
  int simulations = 32;
  double cpuct = 1.5;
  double temperature = 1.15;
  MovegenBackend movegen = MovegenBackend::Bitboard;
  InferenceBackend inference_backend = InferenceBackend::Auto;
  cczero::RuleProfile rules = cczero::RuleProfile::ccz_121_mp3_v1();

  for (int i = 2; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--games") {
      games = std::stoi(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--seed") {
      seed = static_cast<uint64_t>(std::stoull(require_arg_value(i, argc, argv, arg)));
    } else if (arg == "--max-plies") {
      max_plies = std::stoi(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--candidate-seat") {
      candidate_seat = std::stoi(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--out") {
      out_path = require_arg_value(i, argc, argv, arg);
    } else if (arg == "--rules") {
      rules = parse_rule_profile(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--candidate-mp-model" || arg == "--mp-model") {
      candidate_model_path = require_arg_value(i, argc, argv, arg);
    } else if (arg == "--adapter-model" || arg == "--model") {
      adapter_model_path = require_arg_value(i, argc, argv, arg);
    } else if (arg == "--temperature") {
      temperature = std::stod(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--simulations") {
      simulations = std::stoi(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--cpuct") {
      cpuct = std::stod(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--movegen") {
      movegen = parse_movegen_backend(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--inference-backend") {
      inference_backend = parse_inference_backend(require_arg_value(i, argc, argv, arg));
    } else {
      throw std::runtime_error("unknown multiplayer-eval argument: " + arg);
    }
  }

  if (!rules.is_multiplayer() || (rules.player_count != 3 && rules.player_count != 4 &&
                                  rules.player_count != 6)) {
    throw std::runtime_error("multiplayer-eval requires --rules mp3, mp4, or mp6");
  }
  if (candidate_model_path.empty() || adapter_model_path.empty()) {
    throw std::runtime_error(
        "multiplayer-eval requires --candidate-mp-model PATH and --adapter-model PATH");
  }
  if (out_path.empty()) {
    throw std::runtime_error("multiplayer-eval requires --out PATH");
  }
  if (games <= 0) {
    throw std::runtime_error("--games must be positive");
  }
  if (candidate_seat >= rules.player_count) {
    throw std::runtime_error("--candidate-seat is outside the rule profile player count");
  }
  if (temperature <= 0.0 || !std::isfinite(temperature)) {
    throw std::runtime_error("--temperature must be finite and positive");
  }
  if (simulations <= 0) {
    throw std::runtime_error("--simulations must be positive");
  }
  if (cpuct <= 0.0 || !std::isfinite(cpuct)) {
    throw std::runtime_error("--cpuct must be finite and positive");
  }
  if (max_plies <= 0) {
    max_plies = rules.max_plies;
  }
  rules.max_plies = max_plies;

  if (out_path != "-") {
    const std::filesystem::path parent = std::filesystem::path(out_path).parent_path();
    if (!parent.empty()) {
      std::filesystem::create_directories(parent);
    }
  }
  std::ofstream file;
  std::ostream* out = open_output_stream(out_path, file);

  const cczero::Board& board = cczero::Board::standard();
  const PolicyModel adapter_model = load_policy_model(adapter_model_path);
  const NativeMultiplayerModel candidate_model = load_native_multiplayer_model(candidate_model_path);
  const int seat_begin = candidate_seat >= 0 ? candidate_seat : 0;
  const int seat_end = candidate_seat >= 0 ? candidate_seat + 1 : rules.player_count;

  std::vector<MultiplayerEvalGame> played;
  std::map<std::string, int> reasons;
  std::vector<int> candidate_firsts(static_cast<size_t>(rules.player_count), 0);
  std::vector<double> candidate_score_sum(static_cast<size_t>(rules.player_count), 0.0);
  std::vector<double> candidate_place_sum(static_cast<size_t>(rules.player_count), 0.0);
  int global_game_id = 0;
  for (int seat = seat_begin; seat < seat_end; ++seat) {
    for (int game_index = 0; game_index < games; ++game_index) {
      const uint64_t game_seed =
          seed + static_cast<uint64_t>(seat) * 1000003ULL +
          static_cast<uint64_t>(game_index) * 104729ULL;
      MultiplayerEvalGame game = play_multiplayer_eval_game(
          global_game_id, game_seed, seat, board, rules, movegen, adapter_model, candidate_model,
          inference_backend, simulations, cpuct, temperature);
      ++global_game_id;
      ++reasons[game.reason];
      if (seat < static_cast<int>(game.placements.size())) {
        candidate_place_sum.at(static_cast<size_t>(seat)) +=
            game.placements.at(static_cast<size_t>(seat));
        if (game.placements.at(static_cast<size_t>(seat)) == 1) {
          ++candidate_firsts.at(static_cast<size_t>(seat));
        }
      }
      if (seat < static_cast<int>(game.scores.size())) {
        candidate_score_sum.at(static_cast<size_t>(seat)) +=
            game.scores.at(static_cast<size_t>(seat));
      }
      played.push_back(std::move(game));
    }
  }

  *out << "{\"format\":\"cczero.multiplayer_eval.v1\"";
  *out << ",\"rules\":\"" << cczero::json_escape(rules.name) << "\"";
  *out << ",\"candidate_mp_model\":\"" << cczero::json_escape(candidate_model_path) << "\"";
  *out << ",\"adapter_model\":\"" << cczero::json_escape(adapter_model_path) << "\"";
  *out << ",\"games_per_seat\":" << games;
  *out << ",\"total_games\":" << played.size();
  *out << ",\"simulations\":" << simulations;
  *out << ",\"temperature\":" << temperature;
  *out << ",\"max_plies\":" << rules.max_plies;
  *out << ",\"terminal_reasons\":{";
  size_t reason_index = 0;
  for (const auto& [reason, count] : reasons) {
    if (reason_index++ != 0) {
      *out << ",";
    }
    *out << "\"" << cczero::json_escape(reason) << "\":" << count;
  }
  *out << "},\"candidate_by_seat\":[";
  for (int seat = seat_begin; seat < seat_end; ++seat) {
    if (seat != seat_begin) {
      *out << ",";
    }
    *out << "{\"seat\":" << seat << ",\"games\":" << games
         << ",\"firsts\":" << candidate_firsts.at(static_cast<size_t>(seat))
         << ",\"avg_placement\":"
         << candidate_place_sum.at(static_cast<size_t>(seat)) / static_cast<double>(games)
         << ",\"avg_score\":"
         << candidate_score_sum.at(static_cast<size_t>(seat)) / static_cast<double>(games) << "}";
  }
  *out << "],\"games\":[";
  for (size_t i = 0; i < played.size(); ++i) {
    const MultiplayerEvalGame& game = played.at(i);
    if (i != 0) {
      *out << ",";
    }
    *out << "{\"game_id\":" << game.game_id << ",\"seed\":" << game.seed
         << ",\"candidate_seat\":" << game.candidate_seat << ",\"plies\":" << game.plies
         << ",\"terminal_reason\":\"" << cczero::json_escape(game.reason)
         << "\",\"placements\":";
    write_int_array_json(*out, game.placements);
    *out << ",\"score_vector\":";
    write_double_array_json(*out, game.scores);
    *out << ",\"winner_seats\":";
    write_int_array_json(*out, game.winner_seats);
    *out << "}";
  }
  *out << "]}\n";

  std::cerr << "multiplayer-eval complete: games=" << played.size()
            << " rules=" << rules.name << " reasons={";
  size_t index = 0;
  for (const auto& [reason, count] : reasons) {
    if (index++ != 0) {
      std::cerr << ",";
    }
    std::cerr << reason << ":" << count;
  }
  std::cerr << "} out=" << out_path << "\n";
  return 0;
}

SelfplayGameResult run_selfplay_game(int game_id, int local_game_id, uint64_t seed,
                                     int max_plies, const std::string& log_dir,
                                     const PolicyModel& policy_model,
                                     const cczero::Board& board, cczero::RuleProfile rules,
                                     const MctsConfig& config, double opening_temperature,
                                     int temperature_plies, int opening_random_plies) {
  const uint64_t game_seed = seed + static_cast<uint64_t>(local_game_id) * 104729ULL;
  std::mt19937_64 rng(game_seed);
  cczero::State state = cczero::State::initial(board);
  std::unordered_map<uint64_t, int> repetition_counts;
  repetition_counts[state.hash()] = 1;
  for (int opening_ply = 0; opening_ply < opening_random_plies; ++opening_ply) {
    const cczero::TerminalStatus opening_status =
        cczero::terminal_status(state, board, rules, &repetition_counts);
    if (opening_status.terminal) {
      break;
    }
    const std::vector<cczero::Move> opening_moves =
        legal_moves_with_backend(state, board, rules, config.movegen);
    if (opening_moves.empty()) {
      break;
    }
    std::uniform_int_distribution<size_t> dist(0, opening_moves.size() - 1);
    if (!cczero::apply_move(state, opening_moves.at(dist(rng)))) {
      throw std::runtime_error("failed to apply random selfplay opening move");
    }
    ++repetition_counts[state.hash()];
  }

  std::ofstream log_file;
  std::ostream* log = nullptr;
  if (!log_dir.empty()) {
    std::ostringstream name;
    name << "selfplay_" << std::setw(4) << std::setfill('0') << game_id << ".jsonl";
    log_file.open(std::filesystem::path(log_dir) / name.str());
    if (!log_file) {
      throw std::runtime_error("failed to open selfplay log");
    }
    log = &log_file;
    cczero::write_jsonl_game_start(*log, game_seed, cczero::BotKind::Mcts,
                                   cczero::BotKind::Mcts, rules, max_plies, &state);
  }

  SelfplayGameResult result;
  MctsSearchContext search_context;
  cczero::TerminalStatus status =
      cczero::terminal_status(state, board, rules, &repetition_counts);
  while (!status.terminal) {
    MctsConfig move_config = config;
    move_config.temperature = state.ply < temperature_plies ? opening_temperature : 0.0;
    const cczero::State before = state;
    MctsResult search = run_mcts_search(state, board, rules, policy_model, repetition_counts,
                                        move_config, rng, &search_context);
    if (search.stats.reused_tree) {
      ++result.reuse_hits;
    }
    result.nodes += static_cast<uint64_t>(std::max(0, search.stats.nodes));
    result.evals += static_cast<uint64_t>(std::max(0, search.stats.evals));
    result.simulations += static_cast<uint64_t>(std::max(0, search.stats.simulations));
    result.root_legal_moves +=
        static_cast<uint64_t>(std::max(0, search.stats.root_legal_moves));
    result.transposition_hits +=
        static_cast<uint64_t>(std::max(0, search.stats.transposition_hits));
    result.adaptive_stops += search.stats.adaptive_stopped ? 1ULL : 0ULL;
    result.inference_batches +=
        static_cast<uint64_t>(std::max(0, search.stats.inference_batches));
    result.search_ms += search.stats.elapsed_ms;
    result.movegen_ms += search.stats.movegen_ms;
    result.eval_ms += search.stats.eval_ms;
    result.policy_ms += search.stats.policy_ms;
    if (!search.move.is_valid()) {
      status = cczero::TerminalStatus{true, 1 - state.player_to_move, false, "no_legal_moves"};
      break;
    }
    std::string error;
    if (!cczero::validate_move_witness(state, search.move, board, rules, &error)) {
      throw std::runtime_error("selfplay produced invalid move: " + error);
    }

    result.records.push_back(make_selfplay_record(game_id, game_seed, before, search.move,
                                                  search.root_moves, search.stats, board));
    if (!cczero::apply_move(state, search.move)) {
      throw std::runtime_error("failed to apply selfplay move");
    }
    ++repetition_counts[state.hash()];
    if (log != nullptr) {
      cczero::write_jsonl_move(*log, state, search.move, board, before.player_to_move);
    }
    status = cczero::terminal_status(state, board, rules, &repetition_counts);
  }

  if (log != nullptr) {
    cczero::write_jsonl_game_end(*log, status, state.ply);
  }
  const FinishMarginInfo finish_margin =
      finish_margin_from_terminal_state(state, status, board, rules, config.movegen);
  annotate_score_margin(result.records, status, finish_margin);
  result.status = status;
  result.plies = state.ply;
  return result;
}

int run_selfplay(int argc, char** argv) {
  int games = 2;
  int workers = 1;
  int game_id_offset = 0;
  int max_plies = 240;
  uint64_t seed = 1;
  std::string out_path;
  std::string log_dir;
  std::string model_path;
  std::string model_id = "model";
  StorageFormat storage_format = StorageFormat::Rich;
  std::optional<bool> materialize_root_moves;
  cczero::RuleProfile rules = cczero::RuleProfile::ccz_121_strict_lg_v1();
  MctsConfig config;
  config.simulations = 24;
  config.cpuct = 1.4;
  config.add_root_noise = true;
  config.root_dirichlet_alpha = 0.3;
  config.root_noise_fraction = 0.25;
  config.movegen = MovegenBackend::Bitboard;
  config.reuse_tree = true;
  config.inference_batch_size = 64;
  double opening_temperature = 1.0;
  int temperature_plies = 30;
  int opening_random_plies = 0;

  for (int i = 2; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--games") {
      games = std::stoi(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--workers") {
      workers = std::stoi(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--game-id-offset") {
      game_id_offset = std::stoi(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--seed") {
      seed = static_cast<uint64_t>(std::stoull(require_arg_value(i, argc, argv, arg)));
    } else if (arg == "--max-plies") {
      max_plies = std::stoi(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--out") {
      out_path = require_arg_value(i, argc, argv, arg);
    } else if (arg == "--log-dir") {
      log_dir = require_arg_value(i, argc, argv, arg);
    } else if (arg == "--rules") {
      rules = parse_rule_profile(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--model") {
      model_path = require_arg_value(i, argc, argv, arg);
    } else if (arg == "--model-id") {
      model_id = require_arg_value(i, argc, argv, arg);
    } else if (arg == "--storage-format") {
      storage_format = parse_storage_format(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--simulations") {
      config.simulations = std::stoi(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--cpuct") {
      config.cpuct = std::stod(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--temperature-plies") {
      temperature_plies = std::stoi(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--temperature") {
      opening_temperature = std::stod(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--opening-random-plies") {
      opening_random_plies = std::stoi(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--root-dirichlet-alpha") {
      config.root_dirichlet_alpha = std::stod(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--root-noise-fraction") {
      config.root_noise_fraction = std::stod(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--draw-leaf-value") {
      config.draw_leaf_value = std::stod(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--anti-draw-logit-scale") {
      config.anti_draw_logit_scale = std::stod(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--progress-prior-scale") {
      config.progress_prior_scale = std::stod(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--home-pressure-scale") {
      config.home_pressure_scale = std::stod(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--no-transpositions") {
      config.transpositions = false;
    } else if (arg == "--no-reuse-tree") {
      config.reuse_tree = false;
    } else if (arg == "--profile-mcts") {
      config.profile_mcts = true;
    } else if (arg == "--materialize-root-moves") {
      materialize_root_moves = true;
    } else if (arg == "--no-materialize-root-moves") {
      materialize_root_moves = false;
    } else if (arg == "--adaptive-simulations") {
      config.adaptive_simulations = true;
    } else if (arg == "--min-simulations") {
      config.min_simulations = std::stoi(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--adaptive-check-interval") {
      config.adaptive_check_interval = std::stoi(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--adaptive-confidence") {
      config.adaptive_confidence = std::stod(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--movegen") {
      config.movegen = parse_movegen_backend(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--inference-backend") {
      config.inference_backend = parse_inference_backend(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--inference-batch-size") {
      config.inference_batch_size = std::stoi(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--no-root-noise") {
      config.add_root_noise = false;
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }

  if (model_path.empty()) {
    throw std::runtime_error("selfplay requires --model PATH");
  }
  if (out_path.empty()) {
    throw std::runtime_error("selfplay requires --out PATH");
  }
  if (games <= 0) {
    throw std::runtime_error("--games must be positive");
  }
  if (workers <= 0) {
    throw std::runtime_error("--workers must be positive");
  }
  workers = std::min(workers, games);
  if (model_id == "model") {
    model_id = std::filesystem::path(model_path).stem().string();
  }
  config.materialize_root_moves =
      materialize_root_moves.value_or(storage_format == StorageFormat::Rich);

  std::optional<PolicyModel> policy_model = load_policy_model(model_path);
  std::ofstream out_file;
  std::ostream* out = open_output_stream(out_path, out_file);
  if (!log_dir.empty()) {
    std::filesystem::create_directories(log_dir);
  }

  const cczero::Board& board = cczero::Board::standard();
  rules.max_plies = max_plies;
  int records_written = 0;
  int total_plies = 0;
  uint64_t total_nodes = 0;
  uint64_t total_evals = 0;
  uint64_t total_simulations = 0;
  uint64_t total_root_legal_moves = 0;
  uint64_t total_transposition_hits = 0;
  uint64_t total_adaptive_stops = 0;
  uint64_t total_reuse_hits = 0;
  uint64_t total_inference_batches = 0;
  double total_search_ms = 0.0;
  double total_movegen_ms = 0.0;
  double total_eval_ms = 0.0;
  double total_policy_ms = 0.0;
  std::map<std::string, int> reasons;

  std::vector<SelfplayGameResult> game_results(static_cast<size_t>(games));
  if (workers == 1) {
    for (int local_game_id = 0; local_game_id < games; ++local_game_id) {
      game_results[static_cast<size_t>(local_game_id)] = run_selfplay_game(
          game_id_offset + local_game_id, local_game_id, seed, max_plies, log_dir, *policy_model,
          board, rules, config, opening_temperature, temperature_plies, opening_random_plies);
    }
  } else {
    std::vector<std::future<void>> futures;
    futures.reserve(static_cast<size_t>(workers));
    for (int worker = 0; worker < workers; ++worker) {
      futures.push_back(std::async(std::launch::async, [&, worker]() {
        for (int local_game_id = worker; local_game_id < games; local_game_id += workers) {
          game_results[static_cast<size_t>(local_game_id)] = run_selfplay_game(
              game_id_offset + local_game_id, local_game_id, seed, max_plies, log_dir,
              *policy_model, board, rules, config, opening_temperature, temperature_plies,
              opening_random_plies);
        }
      }));
    }
    std::exception_ptr first_error;
    for (std::future<void>& future : futures) {
      try {
        future.get();
      } catch (...) {
        if (!first_error) {
          first_error = std::current_exception();
        }
      }
    }
    if (first_error) {
      std::rethrow_exception(first_error);
    }
  }

  for (const SelfplayGameResult& game : game_results) {
    for (const SelfplayRecord& record : game.records) {
      MctsConfig record_config = config;
      record_config.temperature = record.ply < temperature_plies ? opening_temperature : 0.0;
      write_selfplay_record(*out, record, game.status, rules, model_id, record_config, max_plies,
                            storage_format);
      ++records_written;
    }
    total_plies += game.plies;
    total_nodes += game.nodes;
    total_evals += game.evals;
    total_simulations += game.simulations;
    total_root_legal_moves += game.root_legal_moves;
    total_transposition_hits += game.transposition_hits;
    total_adaptive_stops += game.adaptive_stops;
    total_reuse_hits += game.reuse_hits;
    total_inference_batches += game.inference_batches;
    total_search_ms += game.search_ms;
    total_movegen_ms += game.movegen_ms;
    total_eval_ms += game.eval_ms;
    total_policy_ms += game.policy_ms;
    ++reasons[game.status.reason];
  }

  std::cerr << "selfplay complete: games=" << games << " workers=" << workers
            << " records=" << records_written
            << " avg_plies=" << (games == 0 ? 0.0 : static_cast<double>(total_plies) / games)
            << " movegen=" << movegen_backend_name(config.movegen)
            << " inference_backend=" << inference_backend_name(config.inference_backend)
            << " inference_batch_size=" << config.inference_batch_size
            << " storage_format=" << (storage_format == StorageFormat::Compact ? "compact" : "rich")
            << " materialize_root_moves="
            << (config.materialize_root_moves ? "true" : "false")
            << " profile_mcts=" << (config.profile_mcts ? "true" : "false")
            << " nodes_per_sec="
            << (total_search_ms <= 0.0 ? 0.0 : static_cast<double>(total_nodes) / (total_search_ms / 1000.0))
            << " evals_per_sec="
            << (total_search_ms <= 0.0 ? 0.0 : static_cast<double>(total_evals) / (total_search_ms / 1000.0))
            << " avg_root_legal="
            << (records_written == 0 ? 0.0 : static_cast<double>(total_root_legal_moves) / records_written)
            << " simulations=" << total_simulations
            << " transposition_hits=" << total_transposition_hits
            << " reuse_hits=" << total_reuse_hits
            << " inference_batches=" << total_inference_batches
            << " adaptive_stops=" << total_adaptive_stops
            << " movegen_ms=" << total_movegen_ms
            << " eval_ms=" << total_eval_ms
            << " policy_ms=" << total_policy_ms
            << " out=" << out_path << " reasons=";
  bool first = true;
  for (const auto& [reason, count] : reasons) {
    if (!first) {
      std::cerr << ",";
    }
    first = false;
    std::cerr << reason << ":" << count;
  }
  std::cerr << "\n";
  return 0;
}

ReanalysisOutput run_reanalysis_record(const ReanalysisInput& input, const PolicyModel& model,
                                       const cczero::Board& board,
                                       const cczero::RuleProfile& rules,
                                       const MctsConfig& config,
                                       const std::string& model_id, int max_plies,
                                       uint64_t seed, MctsSearchContext& search_context,
                                       StorageFormat storage_format) {
  MctsConfig record_config = config;
  if (input.simulation_budget > 0) {
    record_config.simulations = input.simulation_budget;
  }
  std::unordered_map<uint64_t, int> repetitions;
  repetitions[input.state.hash()] = 1;
  std::mt19937_64 rng(seed + input.record_seed + static_cast<uint64_t>(input.ply) * 7919ULL);
  MctsResult search =
      run_mcts_search(input.state, board, rules, model, repetitions, record_config, rng,
                      &search_context);

  ReanalysisOutput output;
  output.stats = search.stats;
  if (!search.move.is_valid()) {
    const cczero::TerminalStatus status =
        cczero::terminal_status(input.state, board, rules, nullptr);
    if (status.terminal) {
      output.skip_reason = "terminal_" + summary_token(status.reason);
      return output;
    }
    if (legal_moves_with_backend(input.state, board, rules, config.movegen).empty()) {
      output.skip_reason = "no_legal_moves";
      return output;
    }
    std::ostringstream error;
    error << "reanalyze search produced no valid move for non-terminal position"
          << " game_id=" << input.game_id
          << " ply=" << input.ply
          << " player=" << input.player
          << " hash=" << input.state.hash();
    throw std::runtime_error(error.str());
  }
  SelfplayRecord record = make_selfplay_record(input.game_id, input.record_seed, input.state,
                                               search.move, search.root_moves, search.stats,
                                               board);
  copy_score_margin(record, input);
  std::ostringstream line;
  write_selfplay_record(line, record, status_from_result(input.result_value, input.player), rules,
                        model_id, record_config, max_plies, storage_format);
  output.written = true;
  output.json = line.str();
  return output;
}

ReanalysisBatchResult run_reanalysis_batch(
    const std::vector<ReanalysisInput>& inputs, const PolicyModel& model,
    const cczero::Board& board, const cczero::RuleProfile& rules, const MctsConfig& config,
    const std::string& model_id, int max_plies, uint64_t seed, StorageFormat storage_format,
    int workers, std::atomic<size_t>& completed_records) {
  ReanalysisBatchResult result;
  result.outputs.resize(inputs.size());
  if (inputs.empty()) {
    return result;
  }
  const int active_workers =
      std::min(std::max(1, workers), static_cast<int>(inputs.size()));
  if (active_workers == 1) {
    MctsSearchContext search_context;
    for (size_t index = 0; index < inputs.size(); ++index) {
      result.outputs[index] = run_reanalysis_record(inputs[index], model, board, rules, config,
                                                    model_id, max_plies, seed, search_context,
                                                    storage_format);
      completed_records.fetch_add(1, std::memory_order_relaxed);
    }
    return result;
  }

  std::atomic<size_t> next_index{0};
  std::vector<std::future<void>> futures;
  futures.reserve(static_cast<size_t>(active_workers));
  for (int worker = 0; worker < active_workers; ++worker) {
    futures.push_back(std::async(std::launch::async, [&]() {
      MctsSearchContext search_context;
      while (true) {
        const size_t index = next_index.fetch_add(1, std::memory_order_relaxed);
        if (index >= inputs.size()) {
          break;
        }
        result.outputs[index] = run_reanalysis_record(inputs[index], model, board, rules, config,
                                                      model_id, max_plies, seed, search_context,
                                                      storage_format);
        completed_records.fetch_add(1, std::memory_order_relaxed);
      }
    }));
  }
  for (std::future<void>& future : futures) {
    try {
      future.get();
    } catch (...) {
      if (!result.error) {
        result.error = std::current_exception();
      }
    }
  }
  return result;
}

std::string normalized_path_string(const std::string& path) {
  if (path.empty()) {
    return "";
  }
  return std::filesystem::absolute(std::filesystem::path(path)).lexically_normal().string();
}

std::vector<GateOpponentSpec> parse_gate_opponents(const std::string& text,
                                                   const std::string& champion_model_path) {
  std::vector<GateOpponentSpec> opponents;
  auto trim = [](std::string value) {
    const size_t first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
      return std::string();
    }
    const size_t last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
  };
  std::stringstream input(text);
  std::string item;
  while (std::getline(input, item, ',')) {
    item = trim(item);
    if (item.empty()) {
      continue;
    }
    std::string bot_text = item;
    std::string model_path;
    const size_t colon = item.find(':');
    if (colon != std::string::npos) {
      bot_text = trim(item.substr(0, colon));
      model_path = trim(item.substr(colon + 1));
    }
    cczero::BotKind bot = cczero::BotKind::Random;
    if (!cczero::parse_bot_kind(bot_text, &bot)) {
      throw std::runtime_error("unknown gate opponent: " + bot_text);
    }
    if (model_path.empty() && is_policy_bot(bot)) {
      model_path = champion_model_path;
    }
    GateOpponentSpec spec;
    spec.bot = bot;
    spec.label = bot_text;
    spec.model_path = normalized_path_string(model_path);
    opponents.push_back(std::move(spec));
  }
  if (opponents.empty()) {
    throw std::runtime_error("promotion gate opponent list is empty");
  }
  return opponents;
}

double gate_elo_from_score(double score) {
  const double clipped = std::clamp(score, 0.01, 0.99);
  return 400.0 * std::log10(clipped / (1.0 - clipped));
}

std::pair<double, double> gate_wilson(double points, int games) {
  if (games <= 0) {
    return {0.0, 0.0};
  }
  const double p = points / static_cast<double>(games);
  constexpr double z = 1.96;
  const double denom = 1.0 + z * z / static_cast<double>(games);
  const double center = (p + z * z / (2.0 * games)) / denom;
  const double margin =
      z * std::sqrt((p * (1.0 - p) + z * z / (4.0 * games)) / games) / denom;
  return {std::max(0.0, center - margin), std::min(1.0, center + margin)};
}

bool gate_row_can_reach_threshold(double points, int played, int total, double threshold) {
  const int remaining = std::max(0, total - played);
  return (points + remaining) / static_cast<double>(std::max(1, total)) >= threshold;
}

int gate_initial_position_pairs(int games_per_color, double fraction) {
  if (games_per_color <= 0 || fraction <= 0.0) {
    return 0;
  }
  const int requested =
      static_cast<int>(std::ceil(static_cast<double>(games_per_color) * fraction));
  return std::min(games_per_color, std::max(1, requested));
}

bool gate_pair_starts_from_initial(int pair_index, int games_per_color, int initial_pairs) {
  if (initial_pairs <= 0) {
    return false;
  }
  return pair_index < std::min(games_per_color, initial_pairs);
}

GateGameResult run_gate_game(const GateTask& task, const GateOpponentSpec& opponent,
                             const PolicyModel& candidate_model,
                             const PolicyModel* opponent_model,
                             const cczero::RuleProfile& rules, int max_plies,
                             const MctsOverrides& mcts_overrides,
                             const std::filesystem::path& out_dir) {
  const cczero::BotKind p0 = task.candidate_as_p0 ? cczero::BotKind::Mcts : opponent.bot;
  const cczero::BotKind p1 = task.candidate_as_p0 ? opponent.bot : cczero::BotKind::Mcts;
  std::ostringstream log_name;
  log_name << "gate_" << std::setw(4) << std::setfill('0') << task.game_id << "_"
           << cczero::bot_name(p0) << "_vs_" << cczero::bot_name(p1) << ".jsonl";
  const std::filesystem::path log_path = out_dir / log_name.str();
  std::ofstream log_file(log_path);
  if (!log_file) {
    throw std::runtime_error("failed to open promotion gate log: " + log_path.string());
  }

  const PolicyModel* p0_model = task.candidate_as_p0 ? &candidate_model : opponent_model;
  const PolicyModel* p1_model = task.candidate_as_p0 ? opponent_model : &candidate_model;
  const MatchSummary summary =
      play_game(task.game_id, p0, p1, rules, task.seed, max_plies, &log_file, nullptr,
                p0_model, p1_model, &mcts_overrides, task.opening_random_plies);

  GateGameResult result;
  result.game_id = task.game_id;
  result.opponent_index = task.opponent_index;
  result.candidate_as_p0 = task.candidate_as_p0;
  result.status = summary.status;
  result.plies = summary.plies;
  result.opening_random_plies = task.opening_random_plies;
  result.log_path = log_path.string();
  if (summary.status.draw) {
    result.points = 0.5;
    result.draw = 1;
    return result;
  }
  const bool candidate_won =
      (task.candidate_as_p0 && summary.status.winner == 0) ||
      (!task.candidate_as_p0 && summary.status.winner == 1);
  result.points = candidate_won ? 1.0 : 0.0;
  result.win = candidate_won ? 1 : 0;
  result.loss = candidate_won ? 0 : 1;
  return result;
}

void write_gate_report_json(std::ostream& out, const std::string& candidate_model_path,
                            const std::string& champion_model_path,
                            const cczero::RuleProfile& rules, double threshold,
                            double champion_threshold, bool require_champion_threshold,
                            const std::optional<int>& simulations, int workers,
                            bool early_stop, bool early_stopped, bool promote,
                            int opening_random_plies, double initial_position_fraction,
                            double total_points, int total_games,
                            const std::vector<GateRow>& rows, const MctsOverrides& overrides) {
  const auto [low, high] = gate_wilson(total_points, total_games);
  const double total_score =
      total_games == 0 ? 0.0 : total_points / static_cast<double>(total_games);
  std::optional<double> champion_score;
  for (const GateRow& row : rows) {
    if (row.opponent.bot == cczero::BotKind::Mcts &&
        row.opponent.model_path == champion_model_path) {
      champion_score = row.games == 0 ? 0.0 : row.points / static_cast<double>(row.games);
      break;
    }
  }

  out << "{";
  out << "\"candidate_model\":\"" << cczero::json_escape(candidate_model_path) << "\"";
  out << ",\"champion_model\":\"" << cczero::json_escape(champion_model_path) << "\"";
  out << ",\"rules\":\"" << cczero::json_escape(rules.name) << "\"";
  out << ",\"threshold\":" << threshold;
  out << ",\"champion_threshold\":" << champion_threshold;
  out << ",\"require_champion_threshold\":"
      << (require_champion_threshold ? "true" : "false");
  out << ",\"champion_score\":";
  if (champion_score) {
    out << *champion_score;
  } else {
    out << "null";
  }
  out << ",\"movegen\":\""
      << movegen_backend_name(overrides.movegen.value_or(MovegenBackend::Bitboard)) << "\"";
  out << ",\"inference_backend\":\""
      << inference_backend_name(overrides.inference_backend.value_or(InferenceBackend::Auto))
      << "\"";
  out << ",\"inference_batch_size\":"
      << overrides.inference_batch_size.value_or(1);
  out << ",\"cpuct\":";
  if (overrides.cpuct) {
    out << *overrides.cpuct;
  } else {
    out << "null";
  }
  out << ",\"simulations\":";
  if (simulations) {
    out << *simulations;
  } else {
    out << "null";
  }
  out << ",\"opening_random_plies\":" << opening_random_plies;
  out << ",\"initial_position_fraction\":" << initial_position_fraction;
  out << ",\"workers\":" << workers;
  out << ",\"worker_mode\":\"native\"";
  out << ",\"early_stop\":" << (early_stop ? "true" : "false");
  out << ",\"early_stopped\":" << (early_stopped ? "true" : "false");
  out << ",\"promote\":" << (promote ? "true" : "false");
  out << ",\"aggregate\":{\"games\":" << total_games << ",\"score\":" << total_score
      << ",\"elo_delta\":" << gate_elo_from_score(total_score)
      << ",\"score_ci95\":[" << low << "," << high << "]}";
  out << ",\"rows\":[";
  for (size_t row_index = 0; row_index < rows.size(); ++row_index) {
    const GateRow& row = rows[row_index];
    if (row_index != 0) {
      out << ",";
    }
    const double score = row.games == 0 ? 0.0 : row.points / static_cast<double>(row.games);
    const auto [row_low, row_high] = gate_wilson(row.points, row.games);
    out << "{\"opponent\":\"" << cczero::json_escape(row.opponent.label) << "\"";
    out << ",\"opponent_model\":";
    if (row.opponent.model_path.empty()) {
      out << "null";
    } else {
      out << "\"" << cczero::json_escape(row.opponent.model_path) << "\"";
    }
    out << ",\"games\":" << row.games << ",\"wins\":" << row.wins
        << ",\"draws\":" << row.draws << ",\"losses\":" << row.losses
        << ",\"score\":" << score << ",\"elo_delta\":" << gate_elo_from_score(score)
        << ",\"score_ci95\":[" << row_low << "," << row_high << "]";
    out << ",\"reasons\":{";
    bool first_reason = true;
    for (const auto& [reason, count] : row.reasons) {
      if (!first_reason) {
        out << ",";
      }
      first_reason = false;
      out << "\"" << cczero::json_escape(reason) << "\":" << count;
    }
    out << "},\"logs\":[";
    for (size_t log_index = 0; log_index < row.logs.size(); ++log_index) {
      if (log_index != 0) {
        out << ",";
      }
      out << "\"" << cczero::json_escape(row.logs[log_index]) << "\"";
    }
    out << "],\"scheduled_games\":" << row.scheduled_games
        << ",\"initial_position_games\":" << row.initial_position_games
        << ",\"seeded_opening_games\":" << row.seeded_opening_games
        << ",\"scheduled_initial_position_games\":" << row.scheduled_initial_position_games
        << ",\"scheduled_seeded_opening_games\":" << row.scheduled_seeded_opening_games
        << ",\"early_stopped\":" << (row.games < row.scheduled_games ? "true" : "false")
        << "}";
  }
  out << "]}\n";
}

int run_reanalyze(int argc, char** argv) {
  std::string model_path;
  std::string data_path;
  std::string out_path;
  std::string model_id = "reanalyzed";
  StorageFormat storage_format = StorageFormat::Rich;
  std::optional<bool> materialize_root_moves;
  cczero::RuleProfile rules = cczero::RuleProfile::ccz_121_strict_lg_v1();
  MctsConfig config;
  config.simulations = 24;
  config.cpuct = 1.4;
  config.add_root_noise = false;
  config.temperature = 0.0;
  config.movegen = MovegenBackend::Bitboard;
  config.inference_batch_size = 64;
  int max_plies = 240;
  int workers = 1;
  uint64_t seed = 70000;
  int progress_interval_seconds = 60;
  int streaming_window_records = kDefaultReanalysisStreamingWindowRecords;
  ReanalysisBudgetConfig budget_config;

  for (int i = 2; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--model") {
      model_path = require_arg_value(i, argc, argv, arg);
    } else if (arg == "--data") {
      data_path = require_arg_value(i, argc, argv, arg);
    } else if (arg == "--out") {
      out_path = require_arg_value(i, argc, argv, arg);
    } else if (arg == "--model-id") {
      model_id = require_arg_value(i, argc, argv, arg);
    } else if (arg == "--storage-format") {
      storage_format = parse_storage_format(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--rules") {
      rules = parse_rule_profile(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--simulations") {
      config.simulations = std::stoi(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--cpuct") {
      config.cpuct = std::stod(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--max-plies") {
      max_plies = std::stoi(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--workers") {
      workers = std::stoi(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--seed") {
      seed = static_cast<uint64_t>(std::stoull(require_arg_value(i, argc, argv, arg)));
    } else if (arg == "--movegen") {
      config.movegen = parse_movegen_backend(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--inference-backend") {
      config.inference_backend = parse_inference_backend(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--inference-batch-size") {
      config.inference_batch_size = std::stoi(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--draw-leaf-value") {
      config.draw_leaf_value = std::stod(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--anti-draw-logit-scale") {
      config.anti_draw_logit_scale = std::stod(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--progress-prior-scale") {
      config.progress_prior_scale = std::stod(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--home-pressure-scale") {
      config.home_pressure_scale = std::stod(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--no-transpositions") {
      config.transpositions = false;
    } else if (arg == "--reuse-tree") {
      config.reuse_tree = true;
    } else if (arg == "--no-reuse-tree") {
      config.reuse_tree = false;
    } else if (arg == "--profile-mcts") {
      config.profile_mcts = true;
    } else if (arg == "--materialize-root-moves") {
      materialize_root_moves = true;
    } else if (arg == "--no-materialize-root-moves") {
      materialize_root_moves = false;
    } else if (arg == "--adaptive-simulations") {
      config.adaptive_simulations = true;
    } else if (arg == "--min-simulations") {
      config.min_simulations = std::stoi(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--adaptive-check-interval") {
      config.adaptive_check_interval = std::stoi(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--adaptive-confidence") {
      config.adaptive_confidence = std::stod(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--budgeted-reanalysis") {
      budget_config.enabled = true;
    } else if (arg == "--routing-mode" || arg == "--budget-routing-mode") {
      budget_config.enabled = true;
      budget_config.routing_mode = require_arg_value(i, argc, argv, arg);
    } else if (arg == "--low-complexity-simulations") {
      budget_config.enabled = true;
      budget_config.low_complexity_simulations =
          std::stoi(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--high-complexity-simulations") {
      budget_config.enabled = true;
      budget_config.high_complexity_simulations =
          std::stoi(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--low-entropy-threshold") {
      budget_config.enabled = true;
      budget_config.low_entropy_threshold = std::stod(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--low-surprise-threshold") {
      budget_config.enabled = true;
      budget_config.low_surprise_threshold = std::stod(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--opening-simulations") {
      budget_config.enabled = true;
      budget_config.opening_simulations = std::stoi(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--midgame-simulations") {
      budget_config.enabled = true;
      budget_config.midgame_simulations = std::stoi(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--conversion-simulations") {
      budget_config.enabled = true;
      budget_config.conversion_simulations = std::stoi(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--high-entropy-threshold") {
      budget_config.enabled = true;
      budget_config.high_entropy_threshold = std::stod(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--high-entropy-simulations") {
      budget_config.enabled = true;
      budget_config.high_entropy_simulations = std::stoi(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--high-surprise-threshold") {
      budget_config.enabled = true;
      budget_config.high_surprise_threshold = std::stod(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--high-surprise-simulations") {
      budget_config.enabled = true;
      budget_config.high_surprise_simulations = std::stoi(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--progress-interval-seconds") {
      progress_interval_seconds = std::stoi(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--streaming-window-records") {
      streaming_window_records = std::stoi(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--no-progress") {
      progress_interval_seconds = 0;
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }
  if (model_path.empty() || data_path.empty() || out_path.empty()) {
    throw std::runtime_error("reanalyze requires --model PATH --data IN --out OUT");
  }
  if (model_id == "reanalyzed") {
    model_id = std::filesystem::path(model_path).stem().string() + "_reanalyzed";
  }
  if (workers <= 0) {
    throw std::runtime_error("--workers must be positive");
  }
  if (streaming_window_records <= 0) {
    throw std::runtime_error("--streaming-window-records must be positive");
  }
  if (budget_config.routing_mode != "complexity" && budget_config.routing_mode != "phase") {
    throw std::runtime_error("--routing-mode must be complexity or phase");
  }
  config.materialize_root_moves =
      materialize_root_moves.value_or(storage_format == StorageFormat::Rich);

  const PolicyModel model = load_policy_model(model_path);
  std::ifstream input_file;
  std::istream* input = &std::cin;
  if (data_path != "-") {
    input_file.open(data_path);
    input = &input_file;
  }
  if (!*input) {
    throw std::runtime_error("failed to open reanalysis input: " + data_path);
  }
  std::ofstream out_file;
  std::ostream* out = open_output_stream(out_path, out_file);
  const cczero::Board& board = cczero::Board::standard();
  rules.max_plies = max_plies;

  size_t read_records = 0;
  size_t written_records = 0;
  size_t skipped_records = 0;
  SearchTotals totals;
  std::map<int, int> budget_counts;
  std::map<std::string, int> reason_counts;
  std::map<std::string, int> skip_counts;
  std::atomic<size_t> completed_records{0};
  std::atomic<size_t> parsed_records{0};
  std::atomic<bool> input_done{false};
  std::atomic<bool> progress_done{false};
  std::mutex progress_mutex;
  std::condition_variable progress_cv;
  const auto progress_start = std::chrono::steady_clock::now();

  auto print_progress = [&](bool final) {
    const size_t completed = completed_records.load(std::memory_order_relaxed);
    const size_t total = parsed_records.load(std::memory_order_relaxed);
    const bool finished_reading = input_done.load(std::memory_order_relaxed);
    const auto now = std::chrono::steady_clock::now();
    const double elapsed_seconds =
        std::chrono::duration<double>(now - progress_start).count();
    const double records_per_sec =
        elapsed_seconds <= 0.0 ? 0.0 : static_cast<double>(completed) / elapsed_seconds;
    const double percent =
        total == 0
            ? (finished_reading ? 100.0 : 0.0)
            : std::min(100.0, 100.0 * static_cast<double>(completed) /
                                  static_cast<double>(total));
    const double eta_seconds =
        !finished_reading || records_per_sec <= 0.0 || completed >= total
            ? 0.0
            : static_cast<double>(total - completed) / records_per_sec;
    std::ostringstream progress;
    progress << (final ? "reanalyze progress final: " : "reanalyze progress: ")
             << "completed=" << completed
             << " total=" << total
             << " input_done=" << (finished_reading ? "true" : "false")
             << " pct=" << std::fixed << std::setprecision(2) << percent
             << " elapsed_sec=" << std::setprecision(1) << elapsed_seconds
             << " records_per_sec=" << std::setprecision(3) << records_per_sec
             << " eta_sec=" << std::setprecision(1) << eta_seconds;
    std::cerr << progress.str() << "\n";
  };

  std::thread progress_thread;
  if (progress_interval_seconds > 0) {
    std::cerr << "reanalyze start: workers=" << workers
              << " simulations=" << config.simulations
              << " budgeted_reanalysis=" << (budget_config.enabled ? "true" : "false")
              << " reuse_tree=" << (config.reuse_tree ? "true" : "false")
              << " profile_mcts=" << (config.profile_mcts ? "true" : "false")
              << " streaming_window_records=" << streaming_window_records
              << " scheduling=dynamic"
              << " progress_interval_seconds=" << progress_interval_seconds
              << "\n";
    progress_thread = std::thread([&]() {
      std::unique_lock<std::mutex> lock(progress_mutex);
      while (!progress_done.load(std::memory_order_relaxed)) {
        if (progress_cv.wait_for(lock, std::chrono::seconds(progress_interval_seconds),
                                 [&]() { return progress_done.load(std::memory_order_relaxed); })) {
          break;
        }
        print_progress(false);
      }
    });
  }

  std::exception_ptr reanalysis_error;
  std::vector<ReanalysisInput> window;
  window.reserve(static_cast<size_t>(streaming_window_records));
  auto write_batch_outputs = [&](const std::vector<ReanalysisOutput>& outputs) {
    for (const ReanalysisOutput& output : outputs) {
      add_search_stats(totals, output.stats);
      if (!output.written) {
        if (!output.skip_reason.empty()) {
          ++skipped_records;
          skip_counts[output.skip_reason] += 1;
        }
        continue;
      }
      *out << output.json;
      ++written_records;
    }
  };
  auto flush_window = [&]() {
    if (window.empty()) {
      return;
    }
    ReanalysisBatchResult batch =
        run_reanalysis_batch(window, model, board, rules, config, model_id, max_plies, seed,
                             storage_format, workers, completed_records);
    if (batch.error) {
      std::rethrow_exception(batch.error);
    }
    write_batch_outputs(batch.outputs);
    window.clear();
  };

  try {
    std::string line;
    while (std::getline(*input, line)) {
      ReanalysisParsedRecord parsed =
          parse_reanalysis_record_line(line, seed, config.simulations, budget_config);
      if (!parsed.accepted) {
        continue;
      }
      ++read_records;
      parsed_records.store(read_records, std::memory_order_relaxed);
      budget_counts[parsed.budget] += 1;
      for (const std::string& reason : parsed.budget_reasons) {
        reason_counts[std::to_string(parsed.budget) + ":" + reason] += 1;
      }
      window.push_back(std::move(parsed.input));
      if (window.size() >= static_cast<size_t>(streaming_window_records)) {
        flush_window();
      }
    }
    input_done.store(true, std::memory_order_relaxed);
    flush_window();
  } catch (...) {
    input_done.store(true, std::memory_order_relaxed);
    reanalysis_error = std::current_exception();
  }
  if (progress_thread.joinable()) {
    progress_done.store(true, std::memory_order_relaxed);
    progress_cv.notify_all();
    progress_thread.join();
    print_progress(true);
  }
  if (reanalysis_error) {
    std::rethrow_exception(reanalysis_error);
  }

  std::cerr << "reanalyze complete: read=" << read_records << " written=" << written_records
            << " skipped=" << skipped_records
            << " workers=" << workers
            << " scheduling=dynamic"
            << " streaming_window_records=" << streaming_window_records
            << " movegen=" << movegen_backend_name(config.movegen)
            << " inference_backend=" << inference_backend_name(config.inference_backend)
            << " inference_batch_size=" << config.inference_batch_size
            << " storage_format=" << (storage_format == StorageFormat::Compact ? "compact" : "rich")
            << " materialize_root_moves="
            << (config.materialize_root_moves ? "true" : "false")
            << " reuse_tree=" << (config.reuse_tree ? "true" : "false")
            << " profile_mcts=" << (config.profile_mcts ? "true" : "false")
            << " budgeted_reanalysis=" << (budget_config.enabled ? "true" : "false")
            << " budget_counts=" << budget_counts_string(budget_counts)
            << " budget_reasons=" << string_counts_string(reason_counts)
            << " skip_reasons=" << string_counts_string(skip_counts)
            << " nodes_per_sec="
            << (totals.search_ms <= 0.0 ? 0.0 : static_cast<double>(totals.nodes) / (totals.search_ms / 1000.0))
            << " evals_per_sec="
            << (totals.search_ms <= 0.0 ? 0.0 : static_cast<double>(totals.evals) / (totals.search_ms / 1000.0))
            << " simulations=" << totals.simulations
            << " transposition_hits=" << totals.transposition_hits
            << " reuse_hits=" << totals.reuse_hits
            << " inference_batches=" << totals.inference_batches
            << " adaptive_stops=" << totals.adaptive_stops
            << " movegen_ms=" << totals.movegen_ms
            << " eval_ms=" << totals.eval_ms
            << " policy_ms=" << totals.policy_ms
            << " out=" << out_path << "\n";
  return 0;
}

int run_promotion_gate(int argc, char** argv) {
  std::string candidate_model_path;
  std::string champion_model_path;
  std::string opponents_text = "mcts,converter,traffic-greedy";
  std::string out_dir = "experiments/promotion_gate";
  cczero::RuleProfile rules = cczero::RuleProfile::ccz_121_strict_lg_v1();
  int games = 2;
  int workers = 1;
  int max_plies = 240;
  int opening_random_plies = 0;
  double initial_position_fraction = 0.0;
  uint64_t seed = 12000;
  double threshold = 0.55;
  std::optional<double> champion_threshold_override;
  bool require_champion_threshold = true;
  bool early_stop = true;
  bool paired_seeds = true;
  MctsOverrides mcts_overrides;
  mcts_overrides.movegen = MovegenBackend::Bitboard;
  mcts_overrides.inference_backend = InferenceBackend::Auto;
  mcts_overrides.inference_batch_size = 64;
  std::optional<int> simulations;

  for (int i = 2; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--candidate-model") {
      candidate_model_path = require_arg_value(i, argc, argv, arg);
    } else if (arg == "--champion-model") {
      champion_model_path = require_arg_value(i, argc, argv, arg);
    } else if (arg == "--opponents") {
      opponents_text = require_arg_value(i, argc, argv, arg);
    } else if (arg == "--out-dir") {
      out_dir = require_arg_value(i, argc, argv, arg);
    } else if (arg == "--rules") {
      rules = parse_rule_profile(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--games") {
      games = std::stoi(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--workers") {
      workers = std::stoi(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--seed") {
      seed = static_cast<uint64_t>(std::stoull(require_arg_value(i, argc, argv, arg)));
    } else if (arg == "--max-plies") {
      max_plies = std::stoi(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--threshold") {
      threshold = std::stod(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--champion-threshold") {
      champion_threshold_override = std::stod(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--require-champion-threshold") {
      require_champion_threshold = true;
    } else if (arg == "--no-require-champion-threshold") {
      require_champion_threshold = false;
    } else if (arg == "--early-stop") {
      early_stop = true;
    } else if (arg == "--no-early-stop") {
      early_stop = false;
    } else if (arg == "--paired-seeds") {
      paired_seeds = true;
    } else if (arg == "--no-paired-seeds") {
      paired_seeds = false;
    } else if (arg == "--opening-random-plies") {
      opening_random_plies = std::stoi(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--initial-position-fraction") {
      initial_position_fraction = std::stod(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--movegen") {
      mcts_overrides.movegen = parse_movegen_backend(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--simulations") {
      simulations = std::stoi(require_arg_value(i, argc, argv, arg));
      mcts_overrides.simulations = *simulations;
    } else if (arg == "--mcts-cpuct") {
      mcts_overrides.cpuct = std::stod(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--inference-backend") {
      mcts_overrides.inference_backend =
          parse_inference_backend(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--inference-batch-size") {
      mcts_overrides.inference_batch_size = std::stoi(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--mcts-anti-draw-logit-scale") {
      mcts_overrides.anti_draw_logit_scale = std::stod(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--mcts-progress-prior-scale") {
      mcts_overrides.progress_prior_scale = std::stod(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--mcts-home-pressure-scale") {
      mcts_overrides.home_pressure_scale = std::stod(require_arg_value(i, argc, argv, arg));
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }

  if (candidate_model_path.empty() || champion_model_path.empty()) {
    throw std::runtime_error(
        "promotion-gate requires --candidate-model PATH --champion-model PATH");
  }
  if (games <= 0) {
    throw std::runtime_error("--games must be positive");
  }
  if (workers <= 0) {
    throw std::runtime_error("--workers must be positive");
  }
  if (initial_position_fraction < 0.0 || initial_position_fraction > 1.0) {
    throw std::runtime_error("--initial-position-fraction must be in [0, 1]");
  }
  const double champion_threshold =
      champion_threshold_override ? *champion_threshold_override : threshold;
  candidate_model_path = normalized_path_string(candidate_model_path);
  champion_model_path = normalized_path_string(champion_model_path);
  std::filesystem::create_directories(out_dir);

  const PolicyModel candidate_model = load_policy_model(candidate_model_path);
  std::map<std::string, PolicyModel> model_cache;
  model_cache.emplace(champion_model_path, load_policy_model(champion_model_path));
  std::vector<GateOpponentSpec> opponents =
      parse_gate_opponents(opponents_text, champion_model_path);
  for (const GateOpponentSpec& opponent : opponents) {
    if (!opponent.model_path.empty() && opponent.model_path != candidate_model_path &&
        model_cache.find(opponent.model_path) == model_cache.end()) {
      model_cache.emplace(opponent.model_path, load_policy_model(opponent.model_path));
    }
  }

  rules.max_plies = max_plies;
  workers = std::max(1, workers);
  std::vector<GateRow> rows;
  rows.reserve(opponents.size());
  int game_id = 0;
  bool early_stopped = false;
  for (size_t opponent_index = 0; opponent_index < opponents.size(); ++opponent_index) {
    const GateOpponentSpec& opponent = opponents[opponent_index];
    const PolicyModel* opponent_model = nullptr;
    if (!opponent.model_path.empty()) {
      if (opponent.model_path == candidate_model_path) {
        opponent_model = &candidate_model;
      } else {
        const auto found = model_cache.find(opponent.model_path);
        if (found == model_cache.end()) {
          throw std::runtime_error("missing cached opponent model: " + opponent.model_path);
        }
        opponent_model = &found->second;
      }
    }
    GateRow row;
    row.opponent = opponent;
    std::vector<GateTask> tasks;
    tasks.reserve(static_cast<size_t>(games) * 2);
    const int initial_pairs =
        opening_random_plies > 0 ? gate_initial_position_pairs(games, initial_position_fraction)
                                 : games;
    for (int game = 0; game < games; ++game) {
      const uint64_t pair_seed = seed + static_cast<uint64_t>(game) * 7919ULL;
      const int pair_opening_random_plies =
          gate_pair_starts_from_initial(game, games, initial_pairs) ? 0 : opening_random_plies;
      for (bool candidate_as_p0 : {true, false}) {
        GateTask task;
        task.game_id = game_id;
        task.opponent_index = opponent_index;
        task.candidate_as_p0 = candidate_as_p0;
        task.seed = paired_seeds ? pair_seed : seed + static_cast<uint64_t>(game_id) * 7919ULL;
        task.opening_random_plies = pair_opening_random_plies;
        tasks.push_back(task);
        if (task.opening_random_plies == 0) {
          row.scheduled_initial_position_games += 1;
        } else {
          row.scheduled_seeded_opening_games += 1;
        }
        ++game_id;
      }
    }
    row.scheduled_games = static_cast<int>(tasks.size());
    const bool champion_row =
        opponent.bot == cczero::BotKind::Mcts && opponent.model_path == champion_model_path;
    const int active_workers = std::min(workers, std::max(1, static_cast<int>(tasks.size())));
    for (size_t start = 0; start < tasks.size(); start += static_cast<size_t>(active_workers)) {
      const size_t end = std::min(tasks.size(), start + static_cast<size_t>(active_workers));
      std::vector<std::future<GateGameResult>> futures;
      futures.reserve(end - start);
      for (size_t index = start; index < end; ++index) {
        futures.push_back(std::async(std::launch::async, [&, index]() {
          return run_gate_game(tasks[index], opponent, candidate_model, opponent_model, rules,
                               max_plies, mcts_overrides, out_dir);
        }));
      }
      for (std::future<GateGameResult>& future : futures) {
        const GateGameResult result = future.get();
        row.points += result.points;
        if (result.opening_random_plies == 0) {
          row.initial_position_games += 1;
        } else {
          row.seeded_opening_games += 1;
        }
        row.wins += result.win;
        row.draws += result.draw;
        row.losses += result.loss;
        row.games += 1;
        row.logs.push_back(result.log_path);
        row.reasons[result.status.reason] += 1;
      }
      if (early_stop && champion_row &&
          !gate_row_can_reach_threshold(row.points, row.games, row.scheduled_games,
                                        champion_threshold)) {
        early_stopped = true;
        break;
      }
    }
    rows.push_back(std::move(row));
    if (early_stopped) {
      break;
    }
  }

  double total_points = 0.0;
  int total_games = 0;
  std::optional<double> champion_score;
  for (const GateRow& row : rows) {
    total_points += row.points;
    total_games += row.games;
    if (row.opponent.bot == cczero::BotKind::Mcts &&
        row.opponent.model_path == champion_model_path) {
      champion_score = row.games == 0 ? 0.0 : row.points / static_cast<double>(row.games);
    }
  }
  const double total_score =
      total_games == 0 ? 0.0 : total_points / static_cast<double>(total_games);
  const bool champion_ok = !require_champion_threshold || !champion_score ||
                           *champion_score >= champion_threshold;
  const bool promote = total_score >= threshold && champion_ok;

  const std::filesystem::path ratings_path = std::filesystem::path(out_dir) / "ratings.json";
  std::ofstream ratings(ratings_path);
  if (!ratings) {
    throw std::runtime_error("failed to open ratings output: " + ratings_path.string());
  }
  write_gate_report_json(ratings, candidate_model_path, champion_model_path, rules, threshold,
                         champion_threshold, require_champion_threshold, simulations, workers,
                         early_stop, early_stopped, promote, opening_random_plies,
                         initial_position_fraction, total_points, total_games, rows, mcts_overrides);

  std::cout << "opponent,games,wins,draws,losses,score,elo_delta,ci95_low,ci95_high\n";
  for (const GateRow& row : rows) {
    const double score = row.games == 0 ? 0.0 : row.points / static_cast<double>(row.games);
    const auto [low, high] = gate_wilson(row.points, row.games);
    std::cout << row.opponent.label << "," << row.games << "," << row.wins << ","
              << row.draws << "," << row.losses << "," << std::fixed << std::setprecision(3)
              << score << "," << std::setprecision(1) << gate_elo_from_score(score) << ","
              << std::setprecision(3) << low << "," << high << "\n";
  }
  std::cout << "aggregate," << total_games << ",score=" << std::fixed << std::setprecision(3)
            << total_score << ",elo_delta=" << std::setprecision(1)
            << gate_elo_from_score(total_score) << ",champion_score=";
  if (champion_score) {
    std::cout << std::setprecision(3) << *champion_score;
  } else {
    std::cout << "n/a";
  }
  std::cout << ",promote=" << (promote ? "true" : "false") << "\n";
  std::cout << "wrote " << ratings_path.string() << "\n";
  return 0;
}

int run_model_info(int argc, char** argv) {
  std::string model_path;
  for (int i = 2; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--model") {
      model_path = require_arg_value(i, argc, argv, arg);
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }
  if (model_path.empty()) {
    throw std::runtime_error("model-info requires --model PATH");
  }
  const PolicyModel model = load_policy_model(model_path);
  std::cout << "{\"path\":\"" << cczero::json_escape(model_path)
            << "\",\"kind\":\"" << model_kind_name(model.kind)
            << "\",\"policy_head\":\"" << policy_head_kind_name(model.policy_head)
            << "\",\"feature_size\":" << model.feature_size
            << ",\"action_size\":" << model.action_size
            << ",\"hidden_size\":" << model.hidden_size
            << ",\"blocks\":" << model.blocks
            << ",\"move_embed\":" << model.move_embed_size
            << ",\"move_hidden\":" << model.move_hidden_size
            << ",\"move_feature_size\":" << model.move_feature_size
            << ",\"parameters\":" << policy_model_parameter_count(model)
            << ",\"storage_bytes\":" << policy_model_storage_bytes(model)
            << ",\"accelerate_compiled\":" << (accelerate_compiled() ? "true" : "false")
            << ",\"auto_inference_backend\":\""
            << inference_backend_name(resolve_inference_backend(InferenceBackend::Auto)) << "\""
            << ",\"release_build\":"
#if defined(NDEBUG) || defined(CCZERO_RELEASE_BUILD)
            << "true"
#else
            << "false"
#endif
            << ",\"native_build\":"
#if defined(CCZERO_NATIVE_BUILD)
            << "true"
#else
            << "false"
#endif
            << "}\n";
  return 0;
}

int run_position_info(int argc, char** argv) {
  std::string cells;
  std::string rules_name = "strict";
  int player = cczero::kInvalid;
  int ply = 0;
  MovegenBackend movegen = MovegenBackend::Bitboard;

  for (int i = 2; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--cells") {
      cells = require_arg_value(i, argc, argv, arg);
    } else if (arg == "--player") {
      player = std::stoi(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--ply") {
      ply = std::stoi(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--rules") {
      rules_name = require_arg_value(i, argc, argv, arg);
    } else if (arg == "--movegen") {
      movegen = parse_movegen_backend(require_arg_value(i, argc, argv, arg));
    } else {
      throw std::runtime_error("unknown position-info argument: " + arg);
    }
  }

  if (cells.empty()) {
    throw std::runtime_error("position-info requires --cells COMPACT121");
  }
  if (player == cczero::kInvalid) {
    throw std::runtime_error("position-info requires --player 0|1");
  }

  const cczero::Board& board = cczero::Board::standard();
  const cczero::RuleProfile rules = parse_rule_profile(rules_name);
  const cczero::State state = state_from_compact(cells, player, ply);
  const cczero::TerminalStatus status = cczero::terminal_status(state, board, rules, nullptr);
  const std::vector<cczero::Move> moves = legal_moves_with_backend(state, board, rules, movegen);

  std::cout << "{\"type\":\"position_info\",\"rule_profile\":\""
            << cczero::json_escape(rules.name) << "\",\"cells\":\""
            << cczero::state_to_compact_string(state) << "\",\"player\":"
            << state.player_to_move << ",\"ply\":" << state.ply << ",\"hash\":\"";
  write_hex_hash(std::cout, state.hash());
  std::cout << "\",\"terminal\":{\"terminal\":" << (status.terminal ? "true" : "false")
            << ",\"draw\":" << (status.draw ? "true" : "false") << ",\"winner\":";
  if (status.winner == cczero::kInvalid) {
    std::cout << "null";
  } else {
    std::cout << status.winner;
  }
  std::cout << ",\"reason\":\"" << cczero::json_escape(status.reason) << "\"}";
  std::cout << ",\"features\":{";
  for (int p = 0; p < cczero::kPlayers; ++p) {
    if (p != 0) {
      std::cout << ",";
    }
    std::cout << "\"p" << p << "\":{\"pieces\":" << state.count_pieces(p)
              << ",\"goal_count\":" << cczero::pieces_in_goal(state, board, p)
              << ",\"home_count\":";
    int home_count = 0;
    for (int id : board.home_cell_ids(p)) {
      if (state.cells.at(static_cast<size_t>(id)) == p) {
        ++home_count;
      }
    }
    std::cout << home_count
              << ",\"goal_blockers\":" << cczero::goal_blocker_count(state, board, p)
              << ",\"goal_distance\":" << cczero::total_goal_distance(state, board, p)
              << "}";
  }
  std::cout << "},\"legal_count\":" << moves.size() << ",\"legal\":[";
  for (size_t i = 0; i < moves.size(); ++i) {
    if (i != 0) {
      std::cout << ",";
    }
    write_move_object(std::cout, moves.at(i));
  }
  std::cout << "]}\n";
  return 0;
}

cczero::State state_from_compact_cells(const std::string& cells, int player, int ply) {
  if (cells.size() != cczero::kBoardSize) {
    throw std::runtime_error("--cells must contain exactly 121 characters");
  }
  if (player < 0 || player >= cczero::kPlayers) {
    throw std::runtime_error("--player must be 0 or 1");
  }
  cczero::State state = cczero::State::empty();
  for (size_t i = 0; i < cells.size(); ++i) {
    const char ch = cells[i];
    if (ch == '.' || ch == '_' || ch == '-') {
      state.cells[i] = cczero::kEmpty;
    } else if (ch == '0' || ch == '1') {
      state.cells[i] = static_cast<int8_t>(ch - '0');
    } else {
      throw std::runtime_error("--cells may only contain '.', '0', and '1'");
    }
  }
  state.player_to_move = player;
  state.ply = std::max(0, ply);
  return state;
}

int run_best_move(int argc, char** argv) {
  std::string model_path;
  std::string bot_text;
  std::string cells;
  std::string rules_name = "strict";
  int player = cczero::kInvalid;
  int ply = 0;
  uint64_t seed = 1;
  MctsOverrides overrides;

  for (int i = 2; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--model") {
      model_path = require_arg_value(i, argc, argv, arg);
    } else if (arg == "--bot") {
      bot_text = require_arg_value(i, argc, argv, arg);
    } else if (arg == "--cells") {
      cells = require_arg_value(i, argc, argv, arg);
    } else if (arg == "--player") {
      player = std::stoi(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--ply") {
      ply = std::stoi(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--seed") {
      seed = std::stoull(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--rules") {
      rules_name = require_arg_value(i, argc, argv, arg);
    } else if (arg == "--simulations" || arg == "--mcts-simulations") {
      overrides.simulations = std::stoi(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--cpuct" || arg == "--mcts-cpuct") {
      overrides.cpuct = std::stod(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--movegen" || arg == "--mcts-movegen") {
      overrides.movegen = parse_movegen_backend(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--inference-backend" || arg == "--mcts-inference-backend") {
      overrides.inference_backend =
          parse_inference_backend(require_arg_value(i, argc, argv, arg));
    } else if (arg == "--inference-batch-size" || arg == "--mcts-inference-batch-size") {
      overrides.inference_batch_size = std::stoi(require_arg_value(i, argc, argv, arg));
    } else {
      throw std::runtime_error("unknown best-move argument: " + arg);
    }
  }

  cczero::BotKind bot = cczero::BotKind::Mcts;
  if (!bot_text.empty() && !cczero::parse_bot_kind(bot_text, &bot)) {
    throw std::runtime_error("unknown best-move --bot: " + bot_text);
  }
  if (model_path.empty() && bot_text.empty()) {
    throw std::runtime_error("best-move requires --model PATH or --bot BOT");
  }
  if (model_path.empty() && is_policy_bot(bot)) {
    throw std::runtime_error("best-move --bot " + cczero::bot_name(bot) +
                             " requires --model PATH");
  }
  if (cells.empty()) {
    throw std::runtime_error("best-move requires --cells COMPACT121");
  }
  if (player == cczero::kInvalid) {
    throw std::runtime_error("best-move requires --player 0|1");
  }

  const cczero::Board& board = cczero::Board::standard();
  const cczero::RuleProfile rules = parse_rule_profile(rules_name);
  const cczero::State state = state_from_compact_cells(cells, player, ply);
  std::unordered_map<uint64_t, int> repetition_counts;
  repetition_counts[state.hash()] = 1;
  std::mt19937_64 rng(seed);

  if (!is_policy_bot(bot)) {
    const cczero::Move move =
        cczero::choose_move_avoiding_repetition(bot, state, board, rules, rng, repetition_counts);
    std::cout << "{\"move\":{\"player\":" << player << ",\"from\":" << move.from
              << ",\"to\":" << move.to << ",\"path\":[";
    for (size_t i = 0; i < move.path.size(); ++i) {
      if (i) std::cout << ",";
      std::cout << move.path[i];
    }
    std::cout << "],\"move\":\"" << cczero::json_escape(cczero::move_to_string(move, board))
              << "\"},\"bot\":\"" << cczero::json_escape(cczero::bot_name(bot))
              << "\",\"seed\":" << seed << "}\n";
    return 0;
  }

  const PolicyModel model = load_policy_model(model_path);
  MctsConfig config = cczero::default_eval_mcts_config(state, model);
  if (overrides.simulations) config.simulations = *overrides.simulations;
  if (overrides.cpuct) config.cpuct = *overrides.cpuct;
  if (overrides.movegen) config.movegen = *overrides.movegen;
  if (overrides.inference_backend) config.inference_backend = *overrides.inference_backend;
  if (overrides.inference_batch_size) config.inference_batch_size = *overrides.inference_batch_size;

  MctsResult result =
      run_mcts_search(state, board, rules, model, repetition_counts, config, rng);

  std::cout << "{\"move\":{\"player\":" << player << ",\"from\":" << result.move.from
            << ",\"to\":" << result.move.to << ",\"path\":[";
  for (size_t i = 0; i < result.move.path.size(); ++i) {
    if (i) std::cout << ",";
    std::cout << result.move.path[i];
  }
  std::cout << "],\"move\":\"" << cczero::json_escape(cczero::move_to_string(result.move, board))
            << "\"},\"simulations\":" << config.simulations << ",\"movegen\":\""
            << movegen_backend_name(config.movegen) << "\",\"inference_backend\":\""
            << inference_backend_name(resolve_inference_backend(config.inference_backend))
            << "\",\"stats\":{\"simulations\":" << result.stats.simulations
            << ",\"nodes\":" << result.stats.nodes << ",\"evals\":" << result.stats.evals
            << ",\"root_legal_moves\":" << result.stats.root_legal_moves
            << ",\"inference_batches\":" << result.stats.inference_batches
            << ",\"elapsed_ms\":" << result.stats.elapsed_ms << "}}\n";
  return 0;
}

}  // namespace

int cczero_cli_main(int argc, char** argv) {
  try {
    if (argc < 2 || std::string(argv[1]) == "help" || std::string(argv[1]) == "--help" ||
        std::string(argv[1]) == "-h") {
      print_help(std::cout);
      return 0;
    }
    const std::string command = argv[1];
    if (command == "match") {
      return run_match(argc, argv);
    }
    if (command == "match-suite") {
      return run_match_suite(argc, argv);
    }
    if (command == "perft") {
      return run_perft(argc, argv);
    }
    if (command == "validate-movegen") {
      return run_validate_movegen(argc, argv);
    }
    if (command == "benchmark-movegen") {
      return run_benchmark_movegen(argc, argv);
    }
    if (command == "tournament") {
      return run_tournament(argc, argv);
    }
    if (command == "dataset") {
      return run_dataset(argc, argv);
    }
    if (command == "multiplayer-dataset") {
      return run_multiplayer_dataset(argc, argv);
    }
    if (command == "multiplayer-eval") {
      return run_multiplayer_eval(argc, argv);
    }
    if (command == "selfplay") {
      return run_selfplay(argc, argv);
    }
    if (command == "reanalyze") {
      return run_reanalyze(argc, argv);
    }
    if (command == "promotion-gate") {
      return run_promotion_gate(argc, argv);
    }
    if (command == "position-info") {
      return run_position_info(argc, argv);
    }
    if (command == "best-move") {
      return run_best_move(argc, argv);
    }
    if (command == "model-info") {
      return run_model_info(argc, argv);
    }
    throw std::runtime_error("unknown command: " + command);
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << "\n\n";
    print_help(std::cerr);
    return 1;
  }
}
