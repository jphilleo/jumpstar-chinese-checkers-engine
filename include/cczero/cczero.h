#pragma once

#include <array>
#include <bitset>
#include <cstdint>
#include <iosfwd>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace cczero {

constexpr int kBoardSize = 121;
constexpr int kPlayers = 2;
constexpr int kMaxPlayers = 6;
constexpr int kStarArms = 6;
constexpr int kPiecesPerPlayer = 10;
constexpr int kInvalid = -1;
constexpr int kEmpty = -1;

struct Coord {
  int q = 0;
  int r = 0;

  int s() const { return -q - r; }
};

int hex_distance(Coord a, Coord b);
std::string coord_to_string(Coord coord);

struct Move {
  int from = kInvalid;
  int to = kInvalid;
  std::vector<int> path;

  bool is_valid() const { return from != kInvalid && to != kInvalid; }
};

struct MoveEndpoint {
  int from = kInvalid;
  int to = kInvalid;
  int path_length = 0;

  bool is_valid() const { return from != kInvalid && to != kInvalid; }
};

struct RuleProfile {
  std::string name;
  bool side_triangles_playable = true;
  bool goal_locking = true;
  bool anti_block_terminal = true;
  // When false (v1 behavior): anti-block terminal requires the winning player
  // to have completely vacated their home triangle (home_count == 0). This
  // creates a stall-draw loophole: an opponent that strands a single piece in
  // its own home (= the winning player's goal-blocker territory) can keep the
  // game alive until max_ply.
  // When true (v2 behavior): anti-block fires as soon as the goal is fully
  // occupied with at least one blocker, regardless of where the winning
  // player's other pieces sit. This closes the loophole and matches the
  // natural reading of the rule.
  bool tighten_anti_block = false;
  bool repetition_draw = true;
  int repetition_count = 3;
  int max_plies = 500;
  int player_count = kPlayers;
  std::array<int, kMaxPlayers> home_arms{};
  std::array<double, kMaxPlayers> placement_points{};

  bool is_multiplayer() const { return player_count > kPlayers; }
  bool valid_player(int player) const { return player >= 0 && player < player_count; }
  int home_arm(int player) const;
  int goal_arm(int player) const;

  // v1 profiles are preserved verbatim so old logs and benchmarks remain
  // reproducible. v2 profiles enable the tighter anti-block rule and are the
  // canonical profiles for new work and the CCERL-2P10-v2 benchmark.
  static RuleProfile ccz_121_ab_lg_v1();
  static RuleProfile ccz_121_strict_lg_v1();
  static RuleProfile ccz_121_ab_lg_v2();
  static RuleProfile ccz_121_strict_lg_v2();
  static RuleProfile ccz_121_mp3_v1();
  static RuleProfile ccz_121_mp4_v1();
  static RuleProfile ccz_121_mp6_v1();
};

class Board {
 public:
  static const Board& standard();

  int size() const { return static_cast<int>(coords_.size()); }
  const Coord& coord(int id) const { return coords_.at(static_cast<size_t>(id)); }
  int id_at(int q, int r) const;
  bool contains(int q, int r) const { return id_at(q, r) != kInvalid; }

  const std::array<int, 6>& neighbors(int id) const {
    return neighbors_.at(static_cast<size_t>(id));
  }
  int jump_mid(int id, int dir) const {
    return jump_mid_.at(static_cast<size_t>(id)).at(static_cast<size_t>(dir));
  }
  int jump_landing(int id, int dir) const {
    return jump_landing_.at(static_cast<size_t>(id)).at(static_cast<size_t>(dir));
  }

  bool is_home(int player, int id) const;
  bool is_goal(int player, int id) const;
  bool is_home(const RuleProfile& rules, int player, int id) const;
  bool is_goal(const RuleProfile& rules, int player, int id) const;
  bool is_side_triangle(int id) const { return side_triangles_.test(static_cast<size_t>(id)); }
  int goal_distance(int player, int id) const {
    return goal_distance_[static_cast<size_t>(player)][static_cast<size_t>(id)];
  }
  int goal_distance(const RuleProfile& rules, int player, int id) const;

  const std::bitset<kBoardSize>& home_mask(int player) const;
  const std::bitset<kBoardSize>& goal_mask(int player) const;
  const std::bitset<kBoardSize>& home_mask(const RuleProfile& rules, int player) const;
  const std::bitset<kBoardSize>& goal_mask(const RuleProfile& rules, int player) const;
  const std::bitset<kBoardSize>& arm_mask(int arm) const;
  const std::bitset<kBoardSize>& side_triangle_mask() const { return side_triangles_; }
  const std::array<int, kPiecesPerPlayer>& home_cell_ids(int player) const;
  const std::array<int, kPiecesPerPlayer>& goal_cell_ids(int player) const;
  const std::array<int, kPiecesPerPlayer>& home_cell_ids(const RuleProfile& rules,
                                                         int player) const;
  const std::array<int, kPiecesPerPlayer>& goal_cell_ids(const RuleProfile& rules,
                                                         int player) const;
  const std::array<int, kPiecesPerPlayer>& arm_cell_ids(int arm) const;

  std::vector<int> home_cells(int player) const;
  std::vector<int> goal_cells(int player) const;
  std::vector<int> home_cells(const RuleProfile& rules, int player) const;
  std::vector<int> goal_cells(const RuleProfile& rules, int player) const;

 private:
  Board();

  std::vector<Coord> coords_;
  std::array<int, 17 * 17> id_by_offset_{};
  std::vector<std::array<int, 6>> neighbors_;
  std::vector<std::array<int, 6>> jump_mid_;
  std::vector<std::array<int, 6>> jump_landing_;
  std::array<std::bitset<kBoardSize>, kPlayers> home_{};
  std::array<std::bitset<kBoardSize>, kPlayers> goal_{};
  std::array<std::array<uint8_t, kBoardSize>, kPlayers> home_lookup_{};
  std::array<std::array<uint8_t, kBoardSize>, kPlayers> goal_lookup_{};
  std::array<std::array<int, kPiecesPerPlayer>, kPlayers> home_cells_{};
  std::array<std::array<int, kPiecesPerPlayer>, kPlayers> goal_cells_{};
  std::array<std::array<int, kBoardSize>, kPlayers> goal_distance_{};
  std::array<std::bitset<kBoardSize>, kStarArms> arms_{};
  std::array<std::array<int, kPiecesPerPlayer>, kStarArms> arm_cells_{};
  std::array<std::array<int, kBoardSize>, kStarArms> arm_distance_{};
  std::bitset<kBoardSize> side_triangles_;
};

struct State {
  std::array<int8_t, kBoardSize> cells{};
  int player_to_move = 0;
  int ply = 0;

  static State initial(const Board& board = Board::standard());
  static State initial(const Board& board, const RuleProfile& rules);
  static State empty();

  int count_pieces(int player) const;
  bool is_empty(int id) const { return cells.at(static_cast<size_t>(id)) == kEmpty; }
  uint64_t hash() const;
};

struct Undo {
  Move move;
  int previous_player = 0;
  int previous_ply = 0;
};

struct TerminalStatus {
  bool terminal = false;
  int winner = kInvalid;
  bool draw = false;
  std::string reason;
};

std::vector<Move> legal_moves(const State& state, const Board& board,
                              const RuleProfile& rules);
std::vector<Move> legal_moves_reference(const State& state, const Board& board,
                                        const RuleProfile& rules);
std::vector<Move> legal_moves_fast(const State& state, const Board& board,
                                   const RuleProfile& rules);
std::vector<Move> legal_moves_bitboard(const State& state, const Board& board,
                                       const RuleProfile& rules);
std::vector<MoveEndpoint> legal_move_endpoints_bitboard(const State& state,
                                                        const Board& board,
                                                        const RuleProfile& rules);
bool validate_move_witness(const State& before, const Move& move, const Board& board,
                           const RuleProfile& rules, std::string* error);
bool apply_move(State& state, const Move& move, Undo* undo = nullptr);
bool apply_move(State& state, const Move& move, const RuleProfile& rules,
                Undo* undo = nullptr);
void undo_move(State& state, const Undo& undo);
uint64_t perft(State& state, const Board& board, const RuleProfile& rules, int depth,
               bool use_fast = false);

TerminalStatus terminal_status(
    const State& state, const Board& board, const RuleProfile& rules,
    const std::unordered_map<uint64_t, int>* repetition_counts = nullptr);

int pieces_in_goal(const State& state, const Board& board, int player);
int goal_distance_for_cell(const Board& board, int player, int id);
int total_goal_distance(const State& state, const Board& board, int player);
int goal_blocker_count(const State& state, const Board& board, int player);
int evaluate_state(const State& state, const Board& board, const RuleProfile& rules,
                   int perspective_player);
std::string state_to_compact_string(const State& state);

std::string move_to_string(const Move& move, const Board& board);
std::string path_to_string(const Move& move, const Board& board);

enum class BotKind {
  Random,
  GreedyDistance,
  TrafficGreedy,
  HandEval,
  BeamSearch,
  Pvs,
  Converter,
  TtPvs,
  Policy,
  PolicyBeam,
  PuctLite,
  Mcts,
};

std::vector<BotKind> all_bot_kinds();
std::string bot_name(BotKind kind);
bool parse_bot_kind(const std::string& text, BotKind* kind);
Move choose_move(BotKind kind, const State& state, const Board& board,
                 const RuleProfile& rules, std::mt19937_64& rng);
Move choose_move_avoiding_repetition(
    BotKind kind, const State& state, const Board& board, const RuleProfile& rules,
    std::mt19937_64& rng, const std::unordered_map<uint64_t, int>& repetition_counts);

std::string json_escape(const std::string& text);
void write_jsonl_game_start(std::ostream& out, uint64_t seed, BotKind p0, BotKind p1,
                            const RuleProfile& rules, int max_plies,
                            const State* initial_state = nullptr);
void write_jsonl_move(std::ostream& out, const State& after, const Move& move,
                      const Board& board, int player);
void write_jsonl_game_end(std::ostream& out, const TerminalStatus& status, int plies);

}  // namespace cczero
