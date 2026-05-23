#include "cczero/cczero.h"

#include <algorithm>
#include <cassert>
#include <bit>
#include <cmath>
#include <iomanip>
#include <initializer_list>
#include <limits>
#include <map>
#include <optional>
#include <ostream>
#include <set>
#include <sstream>
#include <stdexcept>

namespace cczero {
namespace {

constexpr std::array<Coord, 6> kDirs = {
    Coord{1, 0},   Coord{1, -1},  Coord{0, -1},
    Coord{-1, 0},  Coord{-1, 1},  Coord{0, 1},
};

int offset_index(int q, int r) {
  if (q < -8 || q > 8 || r < -8 || r > 8) {
    return kInvalid;
  }
  return (r + 8) * 17 + (q + 8);
}

uint64_t splitmix64(uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

bool occupied_during_hop(const State& state, int start, int current, int id) {
  if (id == start && current != start) {
    return false;
  }
  if (id == current) {
    return true;
  }
  return state.cells.at(static_cast<size_t>(id)) != kEmpty;
}

bool legal_landing_for_goal_lock(const Board& board, const RuleProfile& rules, int player,
                                 bool locked_in_goal, int landing) {
  if (!rules.side_triangles_playable && board.is_side_triangle(landing)) {
    return false;
  }
  if (!rules.goal_locking) {
    return true;
  }
  return !locked_in_goal || board.is_goal(rules, player, landing);
}

void dfs_hops(const State& state, const Board& board, const RuleProfile& rules, int player,
              int start, int current, bool locked_in_goal, std::vector<int>& path,
              std::array<bool, kBoardSize>& visited, std::array<bool, kBoardSize>& emitted,
              std::vector<Move>& moves) {
  for (int dir = 0; dir < 6; ++dir) {
    const int mid = board.jump_mid(current, dir);
    const int landing = board.jump_landing(current, dir);
    if (mid == kInvalid || landing == kInvalid) {
      continue;
    }
    if (!occupied_during_hop(state, start, current, mid)) {
      continue;
    }
    if (!state.is_empty(landing) || visited.at(static_cast<size_t>(landing))) {
      continue;
    }
    if (!legal_landing_for_goal_lock(board, rules, player, locked_in_goal, landing)) {
      continue;
    }

    const bool next_locked = locked_in_goal || board.is_goal(rules, player, landing);
    path.push_back(landing);
    visited.at(static_cast<size_t>(landing)) = true;

    if (!emitted.at(static_cast<size_t>(landing))) {
      emitted.at(static_cast<size_t>(landing)) = true;
      moves.push_back(Move{start, landing, path});
    }

    dfs_hops(state, board, rules, player, start, landing, next_locked, path, visited,
             emitted, moves);

    visited.at(static_cast<size_t>(landing)) = false;
    path.pop_back();
  }
}

int best_goal_distance_for_cell(const Board& board, int player, int id) {
  return board.goal_distance(player, id);
}

int mobility_after_landing(const State& state, const Board& board, int landing) {
  int mobility = 0;
  for (int neighbor : board.neighbors(landing)) {
    if (neighbor != kInvalid && state.is_empty(neighbor)) {
      ++mobility;
    }
  }
  return mobility;
}

int congestion_around(const State& state, const Board& board, int landing) {
  int congestion = 0;
  for (int neighbor : board.neighbors(landing)) {
    if (neighbor != kInvalid && !state.is_empty(neighbor)) {
      ++congestion;
    }
  }
  return congestion;
}

std::string uint64_to_hex(uint64_t value) {
  std::ostringstream out;
  out << "0x" << std::hex << std::setw(16) << std::setfill('0') << value;
  return out.str();
}

bool move_less(const Move& a, const Move& b) {
  if (a.from != b.from) {
    return a.from < b.from;
  }
  if (a.to != b.to) {
    return a.to < b.to;
  }
  if (a.path.size() != b.path.size()) {
    return a.path.size() < b.path.size();
  }
  return a.path < b.path;
}

int star_arm_for_coord(Coord coord) {
  if (coord.r < -4) {
    return 0;
  }
  if (coord.q > 4) {
    return 1;
  }
  if (coord.s() < -4) {
    return 2;
  }
  if (coord.r > 4) {
    return 3;
  }
  if (coord.q < -4) {
    return 4;
  }
  if (coord.s() > 4) {
    return 5;
  }
  return kInvalid;
}

RuleProfile make_rule_profile(const std::string& name, int player_count,
                              std::initializer_list<int> home_arms,
                              std::initializer_list<double> placement_points,
                              int max_plies = 500) {
  RuleProfile rules;
  rules.name = name;
  rules.side_triangles_playable = true;
  rules.goal_locking = true;
  rules.anti_block_terminal = true;
  rules.repetition_draw = true;
  rules.repetition_count = 3;
  rules.max_plies = max_plies;
  rules.player_count = player_count;
  rules.home_arms.fill(kInvalid);
  rules.placement_points.fill(0.0);

  if (player_count < 2 || player_count > kMaxPlayers) {
    throw std::runtime_error("rule profile player count out of range");
  }
  if (static_cast<int>(home_arms.size()) != player_count) {
    throw std::runtime_error("rule profile home-arm count mismatch");
  }
  if (static_cast<int>(placement_points.size()) != player_count) {
    throw std::runtime_error("rule profile placement count mismatch");
  }

  int index = 0;
  for (int arm : home_arms) {
    if (arm < 0 || arm >= kStarArms) {
      throw std::runtime_error("rule profile home arm out of range");
    }
    rules.home_arms[static_cast<size_t>(index++)] = arm;
  }

  index = 0;
  for (double points : placement_points) {
    rules.placement_points[static_cast<size_t>(index++)] = points;
  }
  return rules;
}

int next_player_after(int player, const RuleProfile& rules) {
  if (!rules.valid_player(player)) {
    return kInvalid;
  }
  return (player + 1) % rules.player_count;
}

struct HoleBits {
  uint64_t lo = 0;
  uint64_t hi = 0;

  bool empty() const { return lo == 0 && hi == 0; }

  bool test(int id) const {
    return id < 64 ? ((lo >> id) & 1ULL) != 0 : ((hi >> (id - 64)) & 1ULL) != 0;
  }

  void set(int id) {
    if (id < 64) {
      lo |= 1ULL << id;
    } else {
      hi |= 1ULL << (id - 64);
    }
  }

  void reset(int id) {
    if (id < 64) {
      lo &= ~(1ULL << id);
    } else {
      hi &= ~(1ULL << (id - 64));
    }
  }

  int pop_lsb() {
    if (lo != 0) {
      const int bit = std::countr_zero(lo);
      lo &= lo - 1;
      return bit;
    }
    const int bit = std::countr_zero(hi);
    hi &= hi - 1;
    return bit + 64;
  }
};

struct PositionBits {
  HoleBits player;
  HoleBits occupied;
};

PositionBits position_bits(const State& state, int player) {
  PositionBits bits;
  for (int id = 0; id < kBoardSize; ++id) {
    const int occupant = state.cells[static_cast<size_t>(id)];
    if (occupant == kEmpty) {
      continue;
    }
    bits.occupied.set(id);
    if (occupant == player) {
      bits.player.set(id);
    }
  }
  return bits;
}

void emit_buffered_move(std::vector<Move>& moves, int from, int to,
                        const std::array<int, kBoardSize>& path, int path_len) {
  Move move;
  move.from = from;
  move.to = to;
  move.path.assign(path.data(), path.data() + path_len);
  moves.push_back(std::move(move));
}

void emit_endpoint(std::vector<MoveEndpoint>& moves, int from, int to, int path_len) {
  moves.push_back(MoveEndpoint{from, to, path_len});
}

void dfs_hops_bitboard(const Board& board, const RuleProfile& rules, int player,
                       const HoleBits& occupied, int start, int current,
                       bool locked_in_goal, std::array<int, kBoardSize>& path,
                       int path_len, HoleBits& visited, HoleBits& emitted,
                       std::vector<Move>& moves) {
  for (int dir = 0; dir < 6; ++dir) {
    const int mid = board.jump_mid(current, dir);
    const int landing = board.jump_landing(current, dir);
    if (mid == kInvalid || landing == kInvalid) {
      continue;
    }
    const bool midpoint_occupied =
        (mid == start && current != start) ? false : (mid == current || occupied.test(mid));
    if (!midpoint_occupied || occupied.test(landing) || visited.test(landing)) {
      continue;
    }
    if (!legal_landing_for_goal_lock(board, rules, player, locked_in_goal, landing)) {
      continue;
    }

    const bool next_locked = locked_in_goal || board.is_goal(rules, player, landing);
    path[static_cast<size_t>(path_len)] = landing;
    visited.set(landing);

    if (!emitted.test(landing)) {
      emitted.set(landing);
      emit_buffered_move(moves, start, landing, path, path_len + 1);
    }

    dfs_hops_bitboard(board, rules, player, occupied, start, landing, next_locked, path,
                      path_len + 1, visited, emitted, moves);
    visited.reset(landing);
  }
}

void dfs_hop_endpoints_bitboard(const Board& board, const RuleProfile& rules, int player,
                                const HoleBits& occupied, int start, int current,
                                bool locked_in_goal, int path_len, HoleBits& visited,
                                HoleBits& emitted, std::vector<MoveEndpoint>& moves) {
  for (int dir = 0; dir < 6; ++dir) {
    const int mid = board.jump_mid(current, dir);
    const int landing = board.jump_landing(current, dir);
    if (mid == kInvalid || landing == kInvalid) {
      continue;
    }
    const bool midpoint_occupied =
        (mid == start && current != start) ? false : (mid == current || occupied.test(mid));
    if (!midpoint_occupied || occupied.test(landing) || visited.test(landing)) {
      continue;
    }
    if (!legal_landing_for_goal_lock(board, rules, player, locked_in_goal, landing)) {
      continue;
    }

    const bool next_locked = locked_in_goal || board.is_goal(rules, player, landing);
    visited.set(landing);
    if (!emitted.test(landing)) {
      emitted.set(landing);
      emit_endpoint(moves, start, landing, path_len + 1);
    }
    dfs_hop_endpoints_bitboard(board, rules, player, occupied, start, landing, next_locked,
                               path_len + 1, visited, emitted, moves);
    visited.reset(landing);
  }
}

}  // namespace

int hex_distance(Coord a, Coord b) {
  return (std::abs(a.q - b.q) + std::abs(a.r - b.r) + std::abs(a.s() - b.s())) / 2;
}

std::string coord_to_string(Coord coord) {
  std::ostringstream out;
  out << "(" << coord.q << "," << coord.r << ")";
  return out.str();
}

RuleProfile RuleProfile::ccz_121_ab_lg_v1() {
  return make_rule_profile("CCZ-121-AB-LG-v1", 2, {0, 3}, {1.0, 0.0});
}

RuleProfile RuleProfile::ccz_121_strict_lg_v1() {
  return make_rule_profile("CCZ-121-Strict-LG-v1", 2, {0, 3}, {1.0, 0.0});
}

// v2 profiles enable the tighter anti-block rule that closes the stall-draw
// loophole present in v1. See include/cczero/cczero.h for details. These are
// the canonical profiles for new self-play, evaluation, and the CCERL-2P10-v2
// benchmark. v1 builders above remain available for replaying historical logs.
RuleProfile RuleProfile::ccz_121_ab_lg_v2() {
  RuleProfile profile = make_rule_profile("CCZ-121-AB-LG-v2", 2, {0, 3}, {1.0, 0.0});
  profile.tighten_anti_block = true;
  return profile;
}

RuleProfile RuleProfile::ccz_121_strict_lg_v2() {
  RuleProfile profile = make_rule_profile("CCZ-121-Strict-LG-v2", 2, {0, 3}, {1.0, 0.0});
  profile.tighten_anti_block = true;
  return profile;
}

RuleProfile RuleProfile::ccz_121_mp3_v1() {
  return make_rule_profile("CCZ-121-MP3-v1", 3, {0, 2, 4}, {1.0, 0.35, 0.0}, 600);
}

RuleProfile RuleProfile::ccz_121_mp4_v1() {
  return make_rule_profile("CCZ-121-MP4-v1", 4, {0, 1, 3, 4},
                           {1.0, 0.45, 0.20, 0.0}, 700);
}

RuleProfile RuleProfile::ccz_121_mp6_v1() {
  return make_rule_profile("CCZ-121-MP6-v1", 6, {0, 1, 2, 3, 4, 5},
                           {1.0, 0.60, 0.40, 0.25, 0.10, 0.0}, 900);
}

int RuleProfile::home_arm(int player) const {
  if (!valid_player(player)) {
    throw std::out_of_range("player out of range for rule profile");
  }
  return home_arms.at(static_cast<size_t>(player));
}

int RuleProfile::goal_arm(int player) const {
  return (home_arm(player) + 3) % kStarArms;
}

const Board& Board::standard() {
  static const Board board;
  return board;
}

Board::Board() {
  id_by_offset_.fill(kInvalid);

  std::set<std::pair<int, int>> coords;

  for (int q = -4; q <= 4; ++q) {
    for (int r = -4; r <= 4; ++r) {
      const int s = -q - r;
      if (std::max({std::abs(q), std::abs(r), std::abs(s)}) <= 4) {
        coords.insert({q, r});
      }
    }
  }

  for (int d = 1; d <= 4; ++d) {
    for (int r = -4; r <= -d; ++r) {
      coords.insert({4 + d, r});
    }
    for (int r = d; r <= 4; ++r) {
      coords.insert({-4 - d, r});
    }
    for (int q = -4; q <= -d; ++q) {
      coords.insert({q, 4 + d});
    }
    for (int q = d; q <= 4; ++q) {
      coords.insert({q, -4 - d});
    }
    for (int q = -4; q <= -d; ++q) {
      coords.insert({q, -4 - d - q});
    }
    for (int q = d; q <= 4; ++q) {
      coords.insert({q, 4 + d - q});
    }
  }

  coords_.reserve(coords.size());
  for (const auto& [q, r] : coords) {
    coords_.push_back(Coord{q, r});
  }
  std::sort(coords_.begin(), coords_.end(), [](Coord a, Coord b) {
    if (a.r != b.r) {
      return a.r < b.r;
    }
    return a.q < b.q;
  });

  if (coords_.size() != kBoardSize) {
    throw std::runtime_error("standard Chinese Checkers board must contain 121 holes");
  }

  for (int id = 0; id < kBoardSize; ++id) {
    const Coord c = coords_.at(static_cast<size_t>(id));
    id_by_offset_.at(static_cast<size_t>(offset_index(c.q, c.r))) = id;
  }

  neighbors_.resize(kBoardSize);
  jump_mid_.resize(kBoardSize);
  jump_landing_.resize(kBoardSize);
  for (int id = 0; id < kBoardSize; ++id) {
    neighbors_.at(static_cast<size_t>(id)).fill(kInvalid);
    jump_mid_.at(static_cast<size_t>(id)).fill(kInvalid);
    jump_landing_.at(static_cast<size_t>(id)).fill(kInvalid);
    const Coord c = coords_.at(static_cast<size_t>(id));
    for (int dir = 0; dir < 6; ++dir) {
      const Coord d = kDirs.at(static_cast<size_t>(dir));
      neighbors_.at(static_cast<size_t>(id)).at(static_cast<size_t>(dir)) =
          id_at(c.q + d.q, c.r + d.r);
      jump_mid_.at(static_cast<size_t>(id)).at(static_cast<size_t>(dir)) =
          id_at(c.q + d.q, c.r + d.r);
      jump_landing_.at(static_cast<size_t>(id)).at(static_cast<size_t>(dir)) =
          id_at(c.q + 2 * d.q, c.r + 2 * d.r);
    }
  }

  for (int id = 0; id < kBoardSize; ++id) {
    const Coord c = coord(id);
    const bool top = c.r < -4;
    const bool bottom = c.r > 4;
    const bool side = c.q > 4 || c.q < -4 || c.s() > 4 || c.s() < -4;
    if (top) {
      home_[0].set(static_cast<size_t>(id));
      goal_[1].set(static_cast<size_t>(id));
    }
    if (bottom) {
      home_[1].set(static_cast<size_t>(id));
      goal_[0].set(static_cast<size_t>(id));
    }
    if (side) {
      side_triangles_.set(static_cast<size_t>(id));
    }
    const int arm = star_arm_for_coord(c);
    if (arm != kInvalid) {
      arms_[static_cast<size_t>(arm)].set(static_cast<size_t>(id));
    }
  }

  for (int arm = 0; arm < kStarArms; ++arm) {
    int arm_count = 0;
    for (int id = 0; id < kBoardSize; ++id) {
      if (arms_[static_cast<size_t>(arm)].test(static_cast<size_t>(id))) {
        if (arm_count >= kPiecesPerPlayer) {
          throw std::runtime_error("star arm contains too many holes");
        }
        arm_cells_[static_cast<size_t>(arm)].at(static_cast<size_t>(arm_count++)) = id;
      }
    }
    if (arm_count != kPiecesPerPlayer) {
      throw std::runtime_error("star arms must each contain 10 holes");
    }
  }

  for (int player = 0; player < kPlayers; ++player) {
    int home_count = 0;
    int goal_count = 0;
    for (int id = 0; id < kBoardSize; ++id) {
      if (home_[static_cast<size_t>(player)].test(static_cast<size_t>(id))) {
        if (home_count >= kPiecesPerPlayer) {
          throw std::runtime_error("home triangle contains too many holes");
        }
        home_cells_[static_cast<size_t>(player)].at(static_cast<size_t>(home_count++)) = id;
      }
      if (goal_[static_cast<size_t>(player)].test(static_cast<size_t>(id))) {
        if (goal_count >= kPiecesPerPlayer) {
          throw std::runtime_error("goal triangle contains too many holes");
        }
        goal_cells_[static_cast<size_t>(player)].at(static_cast<size_t>(goal_count++)) = id;
      }
      home_lookup_[static_cast<size_t>(player)].at(static_cast<size_t>(id)) =
          home_[static_cast<size_t>(player)].test(static_cast<size_t>(id)) ? 1 : 0;
      goal_lookup_[static_cast<size_t>(player)].at(static_cast<size_t>(id)) =
          goal_[static_cast<size_t>(player)].test(static_cast<size_t>(id)) ? 1 : 0;
    }
    if (home_count != kPiecesPerPlayer || goal_count != kPiecesPerPlayer) {
      throw std::runtime_error("home and goal triangles must each contain 10 holes");
    }
  }

  for (int player = 0; player < kPlayers; ++player) {
    for (int id = 0; id < kBoardSize; ++id) {
      int best = std::numeric_limits<int>::max();
      for (int goal : goal_cells_[static_cast<size_t>(player)]) {
        best = std::min(best, hex_distance(coord(id), coord(goal)));
      }
      goal_distance_[static_cast<size_t>(player)].at(static_cast<size_t>(id)) = best;
    }
  }

  for (int arm = 0; arm < kStarArms; ++arm) {
    for (int id = 0; id < kBoardSize; ++id) {
      int best = std::numeric_limits<int>::max();
      for (int goal : arm_cells_[static_cast<size_t>(arm)]) {
        best = std::min(best, hex_distance(coord(id), coord(goal)));
      }
      arm_distance_[static_cast<size_t>(arm)].at(static_cast<size_t>(id)) = best;
    }
  }
}

int Board::id_at(int q, int r) const {
  const int index = offset_index(q, r);
  if (index == kInvalid) {
    return kInvalid;
  }
  return id_by_offset_.at(static_cast<size_t>(index));
}

bool Board::is_home(int player, int id) const {
  return home_lookup_[static_cast<size_t>(player)][static_cast<size_t>(id)] != 0;
}

bool Board::is_goal(int player, int id) const {
  return goal_lookup_[static_cast<size_t>(player)][static_cast<size_t>(id)] != 0;
}

bool Board::is_home(const RuleProfile& rules, int player, int id) const {
  return arm_mask(rules.home_arm(player)).test(static_cast<size_t>(id));
}

bool Board::is_goal(const RuleProfile& rules, int player, int id) const {
  return arm_mask(rules.goal_arm(player)).test(static_cast<size_t>(id));
}

int Board::goal_distance(const RuleProfile& rules, int player, int id) const {
  return arm_distance_.at(static_cast<size_t>(rules.goal_arm(player))).at(static_cast<size_t>(id));
}

const std::bitset<kBoardSize>& Board::home_mask(int player) const {
  return home_.at(static_cast<size_t>(player));
}

const std::bitset<kBoardSize>& Board::goal_mask(int player) const {
  return goal_.at(static_cast<size_t>(player));
}

const std::bitset<kBoardSize>& Board::home_mask(const RuleProfile& rules, int player) const {
  return arm_mask(rules.home_arm(player));
}

const std::bitset<kBoardSize>& Board::goal_mask(const RuleProfile& rules, int player) const {
  return arm_mask(rules.goal_arm(player));
}

const std::bitset<kBoardSize>& Board::arm_mask(int arm) const {
  return arms_.at(static_cast<size_t>(arm));
}

const std::array<int, kPiecesPerPlayer>& Board::home_cell_ids(int player) const {
  return home_cells_[static_cast<size_t>(player)];
}

const std::array<int, kPiecesPerPlayer>& Board::goal_cell_ids(int player) const {
  return goal_cells_[static_cast<size_t>(player)];
}

const std::array<int, kPiecesPerPlayer>& Board::home_cell_ids(const RuleProfile& rules,
                                                              int player) const {
  return arm_cell_ids(rules.home_arm(player));
}

const std::array<int, kPiecesPerPlayer>& Board::goal_cell_ids(const RuleProfile& rules,
                                                              int player) const {
  return arm_cell_ids(rules.goal_arm(player));
}

const std::array<int, kPiecesPerPlayer>& Board::arm_cell_ids(int arm) const {
  return arm_cells_.at(static_cast<size_t>(arm));
}

std::vector<int> Board::home_cells(int player) const {
  const auto& ids = home_cell_ids(player);
  std::vector<int> cells(ids.begin(), ids.end());
  return cells;
}

std::vector<int> Board::goal_cells(int player) const {
  const auto& ids = goal_cell_ids(player);
  std::vector<int> cells(ids.begin(), ids.end());
  return cells;
}

std::vector<int> Board::home_cells(const RuleProfile& rules, int player) const {
  const auto& ids = home_cell_ids(rules, player);
  std::vector<int> cells(ids.begin(), ids.end());
  return cells;
}

std::vector<int> Board::goal_cells(const RuleProfile& rules, int player) const {
  const auto& ids = goal_cell_ids(rules, player);
  std::vector<int> cells(ids.begin(), ids.end());
  return cells;
}

State State::empty() {
  State state;
  state.cells.fill(kEmpty);
  state.player_to_move = 0;
  state.ply = 0;
  return state;
}

State State::initial(const Board& board) {
  State state = State::empty();
  for (int id : board.home_cell_ids(0)) {
    state.cells.at(static_cast<size_t>(id)) = 0;
  }
  for (int id : board.home_cell_ids(1)) {
    state.cells.at(static_cast<size_t>(id)) = 1;
  }
  return state;
}

State State::initial(const Board& board, const RuleProfile& rules) {
  State state = State::empty();
  for (int player = 0; player < rules.player_count; ++player) {
    for (int id : board.home_cell_ids(rules, player)) {
      state.cells.at(static_cast<size_t>(id)) = static_cast<int8_t>(player);
    }
  }
  state.player_to_move = 0;
  return state;
}

int State::count_pieces(int player) const {
  return static_cast<int>(std::count(cells.begin(), cells.end(), static_cast<int8_t>(player)));
}

uint64_t State::hash() const {
  uint64_t h = 0x6c6f6e67686f7073ULL;
  for (int id = 0; id < kBoardSize; ++id) {
    const int occupant = cells.at(static_cast<size_t>(id));
    if (occupant != kEmpty) {
      h ^= splitmix64((static_cast<uint64_t>(id) + 1) * 0x100000001b3ULL ^
                      (static_cast<uint64_t>(occupant) + 1) * 0x9e3779b97f4a7c15ULL);
    }
  }
  h ^= splitmix64(static_cast<uint64_t>(player_to_move) + 0xd1b54a32d192ed03ULL);
  return h;
}

std::vector<Move> legal_moves_reference(const State& state, const Board& board,
                                        const RuleProfile& rules) {
  std::vector<Move> moves;
  const int player = state.player_to_move;

  for (int from = 0; from < kBoardSize; ++from) {
    if (state.cells.at(static_cast<size_t>(from)) != player) {
      continue;
    }

    const bool starts_locked = board.is_goal(rules, player, from);

    for (int to : board.neighbors(from)) {
      if (to == kInvalid || !state.is_empty(to)) {
        continue;
      }
      if (!legal_landing_for_goal_lock(board, rules, player, starts_locked, to)) {
        continue;
      }
      moves.push_back(Move{from, to, {from, to}});
    }

    std::array<bool, kBoardSize> visited{};
    std::array<bool, kBoardSize> emitted{};
    visited.fill(false);
    emitted.fill(false);
    visited.at(static_cast<size_t>(from)) = true;
    std::vector<int> path{from};
    dfs_hops(state, board, rules, player, from, from, starts_locked, path, visited, emitted,
             moves);
  }

  std::sort(moves.begin(), moves.end(), move_less);
  return moves;
}

std::vector<Move> legal_moves_fast(const State& state, const Board& board,
                                   const RuleProfile& rules) {
  struct HopFrame {
    int current = kInvalid;
    bool locked = false;
    std::vector<int> path;
    std::array<bool, kBoardSize> visited{};
  };

  std::vector<Move> moves;
  const int player = state.player_to_move;
  std::array<bool, kBoardSize> occupied{};
  occupied.fill(false);
  for (int id = 0; id < kBoardSize; ++id) {
    occupied.at(static_cast<size_t>(id)) =
        state.cells.at(static_cast<size_t>(id)) != kEmpty;
  }

  for (int from = 0; from < kBoardSize; ++from) {
    if (state.cells.at(static_cast<size_t>(from)) != player) {
      continue;
    }

    const bool starts_locked = board.is_goal(rules, player, from);

    for (int to : board.neighbors(from)) {
      if (to == kInvalid || occupied.at(static_cast<size_t>(to))) {
        continue;
      }
      if (!legal_landing_for_goal_lock(board, rules, player, starts_locked, to)) {
        continue;
      }
      moves.push_back(Move{from, to, {from, to}});
    }

    std::array<bool, kBoardSize> emitted{};
    emitted.fill(false);

    HopFrame root;
    root.current = from;
    root.locked = starts_locked;
    root.path = {from};
    root.visited.fill(false);
    root.visited.at(static_cast<size_t>(from)) = true;

    std::vector<HopFrame> stack;
    stack.push_back(root);
    while (!stack.empty()) {
      HopFrame frame = stack.back();
      stack.pop_back();

      for (int dir = 5; dir >= 0; --dir) {
        const int mid = board.jump_mid(frame.current, dir);
        const int landing = board.jump_landing(frame.current, dir);
        if (mid == kInvalid || landing == kInvalid) {
          continue;
        }

        const bool midpoint_occupied =
            (mid == from && frame.current != from)
                ? false
                : (mid == frame.current || occupied.at(static_cast<size_t>(mid)));
        if (!midpoint_occupied) {
          continue;
        }
        if (occupied.at(static_cast<size_t>(landing)) ||
            frame.visited.at(static_cast<size_t>(landing))) {
          continue;
        }
        if (!legal_landing_for_goal_lock(board, rules, player, frame.locked, landing)) {
          continue;
        }

        HopFrame next = frame;
        next.current = landing;
        next.locked = frame.locked || board.is_goal(rules, player, landing);
        next.path.push_back(landing);
        next.visited.at(static_cast<size_t>(landing)) = true;

        if (!emitted.at(static_cast<size_t>(landing))) {
          emitted.at(static_cast<size_t>(landing)) = true;
          moves.push_back(Move{from, landing, next.path});
        }
        stack.push_back(std::move(next));
      }
    }
  }

  std::sort(moves.begin(), moves.end(), move_less);
  return moves;
}

std::vector<Move> legal_moves_bitboard(const State& state, const Board& board,
                                       const RuleProfile& rules) {
  std::vector<Move> moves;
  const int player = state.player_to_move;
  const PositionBits bits = position_bits(state, player);
  HoleBits pieces = bits.player;
  moves.reserve(64);

  while (!pieces.empty()) {
    const int from = pieces.pop_lsb();
    const bool starts_locked = board.is_goal(rules, player, from);
    for (int to : board.neighbors(from)) {
      if (to == kInvalid || bits.occupied.test(to)) {
        continue;
      }
      if (!legal_landing_for_goal_lock(board, rules, player, starts_locked, to)) {
        continue;
      }
      moves.push_back(Move{from, to, {from, to}});
    }

    HoleBits emitted;
    HoleBits visited;
    visited.set(from);
    std::array<int, kBoardSize> path;
    path[0] = from;
    dfs_hops_bitboard(board, rules, player, bits.occupied, from, from, starts_locked, path, 1,
                      visited, emitted, moves);
  }

  std::sort(moves.begin(), moves.end(), move_less);
  return moves;
}

std::vector<MoveEndpoint> legal_move_endpoints_bitboard(const State& state,
                                                        const Board& board,
                                                        const RuleProfile& rules) {
  std::vector<MoveEndpoint> moves;
  const int player = state.player_to_move;
  const PositionBits bits = position_bits(state, player);
  HoleBits pieces = bits.player;
  moves.reserve(64);

  while (!pieces.empty()) {
    const int from = pieces.pop_lsb();
    const bool starts_locked = board.is_goal(rules, player, from);
    for (int to : board.neighbors(from)) {
      if (to == kInvalid || bits.occupied.test(to)) {
        continue;
      }
      if (!legal_landing_for_goal_lock(board, rules, player, starts_locked, to)) {
        continue;
      }
      emit_endpoint(moves, from, to, 2);
    }

    HoleBits emitted;
    HoleBits visited;
    visited.set(from);
    dfs_hop_endpoints_bitboard(board, rules, player, bits.occupied, from, from, starts_locked,
                               1, visited, emitted, moves);
  }

  std::sort(moves.begin(), moves.end(), [](const MoveEndpoint& a, const MoveEndpoint& b) {
    if (a.from != b.from) {
      return a.from < b.from;
    }
    if (a.to != b.to) {
      return a.to < b.to;
    }
    return a.path_length < b.path_length;
  });
  return moves;
}

std::vector<Move> legal_moves(const State& state, const Board& board,
                              const RuleProfile& rules) {
  return legal_moves_reference(state, board, rules);
}

bool validate_move_witness(const State& before, const Move& move, const Board& board,
                           const RuleProfile& rules, std::string* error) {
  auto fail = [&](const std::string& message) {
    if (error != nullptr) {
      *error = message;
    }
    return false;
  };

  if (!move.is_valid()) {
    return fail("move has invalid endpoints");
  }
  if (move.from < 0 || move.from >= kBoardSize || move.to < 0 || move.to >= kBoardSize) {
    return fail("move endpoint out of range");
  }
  if (move.path.size() < 2 || move.path.front() != move.from || move.path.back() != move.to) {
    return fail("move witness path does not match endpoints");
  }

  const int player = before.player_to_move;
  if (before.cells.at(static_cast<size_t>(move.from)) != player) {
    return fail("move source does not contain the player to move");
  }
  if (!before.is_empty(move.to)) {
    return fail("move destination is occupied");
  }

  std::array<bool, kBoardSize> visited{};
  visited.fill(false);
  visited.at(static_cast<size_t>(move.from)) = true;

  bool locked = board.is_goal(rules, player, move.from);
  bool saw_step = false;
  bool saw_hop = false;

  for (size_t i = 1; i < move.path.size(); ++i) {
    const int prev = move.path.at(i - 1);
    const int next = move.path.at(i);
    if (next < 0 || next >= kBoardSize) {
      return fail("move path contains out-of-range hole id");
    }
    if (!before.is_empty(next)) {
      return fail("move path lands on an occupied hole");
    }
    if (visited.at(static_cast<size_t>(next))) {
      return fail("move path revisits a landing hole");
    }
    if (!legal_landing_for_goal_lock(board, rules, player, locked, next)) {
      return fail("move violates goal-locking");
    }

    const int distance = hex_distance(board.coord(prev), board.coord(next));
    if (distance == 1) {
      saw_step = true;
      if (move.path.size() != 2) {
        return fail("adjacent steps cannot be chained");
      }
    } else if (distance == 2) {
      saw_hop = true;
      std::optional<int> midpoint;
      for (int dir = 0; dir < 6; ++dir) {
        if (board.jump_landing(prev, dir) == next) {
          midpoint = board.jump_mid(prev, dir);
          break;
        }
      }
      if (!midpoint.has_value() || midpoint.value() == kInvalid) {
        return fail("hop segment has no valid midpoint");
      }
      if (!occupied_during_hop(before, move.from, prev, midpoint.value())) {
        return fail("hop segment does not jump over an occupied midpoint");
      }
    } else {
      return fail("path segment is neither a step nor a hop");
    }

    locked = locked || board.is_goal(rules, player, next);
    visited.at(static_cast<size_t>(next)) = true;
  }

  if (saw_step && saw_hop) {
    return fail("move mixes step and hop segments");
  }

  return true;
}

bool apply_move(State& state, const Move& move, Undo* undo) {
  if (undo != nullptr) {
    *undo = Undo{move, state.player_to_move, state.ply};
  }
  const int player = state.player_to_move;
  if (move.from < 0 || move.from >= kBoardSize || move.to < 0 || move.to >= kBoardSize) {
    return false;
  }
  if (state.cells.at(static_cast<size_t>(move.from)) != player ||
      !state.is_empty(move.to)) {
    return false;
  }
  state.cells.at(static_cast<size_t>(move.from)) = kEmpty;
  state.cells.at(static_cast<size_t>(move.to)) = static_cast<int8_t>(player);
  state.player_to_move = (state.player_to_move + 1) % kPlayers;
  ++state.ply;
  return true;
}

bool apply_move(State& state, const Move& move, const RuleProfile& rules, Undo* undo) {
  if (undo != nullptr) {
    *undo = Undo{move, state.player_to_move, state.ply};
  }
  const int player = state.player_to_move;
  if (!rules.valid_player(player)) {
    return false;
  }
  if (move.from < 0 || move.from >= kBoardSize || move.to < 0 || move.to >= kBoardSize) {
    return false;
  }
  if (state.cells.at(static_cast<size_t>(move.from)) != player ||
      !state.is_empty(move.to)) {
    return false;
  }
  state.cells.at(static_cast<size_t>(move.from)) = kEmpty;
  state.cells.at(static_cast<size_t>(move.to)) = static_cast<int8_t>(player);
  state.player_to_move = next_player_after(state.player_to_move, rules);
  ++state.ply;
  return true;
}

void undo_move(State& state, const Undo& undo) {
  const int player = undo.previous_player;
  state.cells.at(static_cast<size_t>(undo.move.to)) = kEmpty;
  state.cells.at(static_cast<size_t>(undo.move.from)) = static_cast<int8_t>(player);
  state.player_to_move = undo.previous_player;
  state.ply = undo.previous_ply;
}

uint64_t perft(State& state, const Board& board, const RuleProfile& rules, int depth,
               bool use_fast) {
  if (depth < 0) {
    throw std::invalid_argument("perft depth must be non-negative");
  }
  if (depth == 0) {
    return 1;
  }

  const std::vector<Move> moves =
      use_fast ? legal_moves_fast(state, board, rules) : legal_moves_reference(state, board, rules);
  uint64_t nodes = 0;
  for (const Move& move : moves) {
    Undo undo;
    if (!apply_move(state, move, rules, &undo)) {
      throw std::runtime_error("perft failed to apply legal move");
    }
    nodes += perft(state, board, rules, depth - 1, use_fast);
    undo_move(state, undo);
  }
  return nodes;
}

int count_player_goal_pieces(const State& state, const Board& board, int player) {
  int count = 0;
  for (int id : board.goal_cell_ids(player)) {
    if (state.cells[static_cast<size_t>(id)] == player) {
      ++count;
    }
  }
  return count;
}

int count_player_goal_pieces(const State& state, const Board& board, const RuleProfile& rules,
                             int player) {
  int count = 0;
  for (int id : board.goal_cell_ids(rules, player)) {
    if (state.cells[static_cast<size_t>(id)] == player) {
      ++count;
    }
  }
  return count;
}

bool has_immediate_goal_finish(const State& state, const Board& board,
                               const RuleProfile& rules, int player,
                               int current_goal_count) {
  if (current_goal_count != kPiecesPerPlayer - 1) {
    return false;
  }
  State probe = state;
  probe.player_to_move = player;
  for (const Move& move : legal_moves_fast(probe, board, rules)) {
    State next = probe;
    if (!apply_move(next, move, rules)) {
      continue;
    }
    if (count_player_goal_pieces(next, board, rules, player) == kPiecesPerPlayer) {
      return true;
    }
  }
  return false;
}

TerminalStatus terminal_status(
    const State& state, const Board& board, const RuleProfile& rules,
    const std::unordered_map<uint64_t, int>* repetition_counts) {
  std::array<int, kMaxPlayers> goal_count{};
  std::array<int, kMaxPlayers> goal_blockers{};
  std::array<int, kMaxPlayers> home_count{};
  for (int player = 0; player < rules.player_count; ++player) {
    for (int id : board.goal_cell_ids(rules, player)) {
      const int occupant = state.cells[static_cast<size_t>(id)];
      if (occupant == player) {
        ++goal_count[static_cast<size_t>(player)];
      } else if (occupant != kEmpty && rules.valid_player(occupant)) {
        ++goal_blockers[static_cast<size_t>(player)];
      }
    }
    for (int id : board.home_cell_ids(rules, player)) {
      if (state.cells[static_cast<size_t>(id)] == player) {
        ++home_count[static_cast<size_t>(player)];
      }
    }
  }

  if (rules.player_count == kPlayers) {
    const bool p0_finished = goal_count[0] == kPiecesPerPlayer;
    const bool p1_finished = goal_count[1] == kPiecesPerPlayer;
    if (p0_finished && p1_finished) {
      return TerminalStatus{true, kInvalid, true, "equal_turn_goal_draw"};
    }
    if (p0_finished) {
      if (state.player_to_move == 1 &&
          has_immediate_goal_finish(state, board, rules, 1, goal_count[1])) {
        return TerminalStatus{true, kInvalid, true, "equal_turn_goal_draw"};
      }
      return TerminalStatus{true, 0, false, "all_pieces_in_goal"};
    }
    if (p1_finished) {
      return TerminalStatus{true, 1, false, "all_pieces_in_goal"};
    }
  } else {
    std::vector<int> finished;
    for (int player = 0; player < rules.player_count; ++player) {
      if (goal_count[static_cast<size_t>(player)] == kPiecesPerPlayer) {
        finished.push_back(player);
      }
    }
    if (!finished.empty() && state.player_to_move == 0) {
      return TerminalStatus{true, finished.front(), finished.size() > 1,
                            finished.size() > 1 ? "multiplayer_equal_turn_tie"
                                                : "multiplayer_round_goal"};
    }
  }

  if (rules.anti_block_terminal) {
    // Anti-block terminal: if your goal triangle is fully occupied (your own
    // pieces plus opponent blockers fill every cell) and you have at least one
    // of your own pieces in there, you win.
    //
    // The v1 version of this rule additionally required home_count[player] == 0
    // — the winning player had to have completely vacated their home triangle.
    // That gate created an exploitable stall: an opponent could strand a single
    // piece in their own home (which is also the winning player's goal-blocker
    // territory) and keep the position alive until max_ply. The v2 rule
    // (rules.tighten_anti_block == true) drops the home_count gate so the win
    // fires as soon as the goal is fully occupied.
    for (int player = 0; player < rules.player_count; ++player) {
      const int own = goal_count[static_cast<size_t>(player)];
      const int blockers = goal_blockers[static_cast<size_t>(player)];
      if (own > 0 && blockers > 0 && own + blockers == kPiecesPerPlayer &&
          (rules.tighten_anti_block ||
           home_count[static_cast<size_t>(player)] == 0)) {
        return TerminalStatus{true, player, false, "anti_block_goal_full"};
      }
    }
  }

  if (rules.repetition_draw && repetition_counts != nullptr) {
    const auto found = repetition_counts->find(state.hash());
    if (found != repetition_counts->end() && found->second >= rules.repetition_count) {
      return TerminalStatus{true, kInvalid, true, "repetition"};
    }
  }

  if (rules.max_plies > 0 && state.ply >= rules.max_plies) {
    return TerminalStatus{true, kInvalid, true, "max_ply"};
  }

  return TerminalStatus{};
}

int pieces_in_goal(const State& state, const Board& board, int player) {
  int count = 0;
  for (int id : board.goal_cell_ids(player)) {
    if (state.cells.at(static_cast<size_t>(id)) == player) {
      ++count;
    }
  }
  return count;
}

int goal_distance_for_cell(const Board& board, int player, int id) {
  return best_goal_distance_for_cell(board, player, id);
}

int total_goal_distance(const State& state, const Board& board, int player) {
  int total = 0;
  for (int id = 0; id < kBoardSize; ++id) {
    if (state.cells.at(static_cast<size_t>(id)) == player) {
      total += best_goal_distance_for_cell(board, player, id);
    }
  }
  return total;
}

int goal_blocker_count(const State& state, const Board& board, int player) {
  int count = 0;
  const int opponent = 1 - player;
  for (int id : board.goal_cell_ids(player)) {
    if (state.cells.at(static_cast<size_t>(id)) == opponent) {
      ++count;
    }
  }
  return count;
}

int evaluate_state(const State& state, const Board& board, const RuleProfile& rules,
                   int perspective_player) {
  const TerminalStatus status = terminal_status(state, board, rules, nullptr);
  if (status.terminal) {
    if (status.draw || status.winner == kInvalid) {
      return 0;
    }
    return status.winner == perspective_player ? 1000000 : -1000000;
  }

  const int opponent = 1 - perspective_player;
  const int own_distance = total_goal_distance(state, board, perspective_player);
  const int opp_distance = total_goal_distance(state, board, opponent);
  const int own_goal = pieces_in_goal(state, board, perspective_player);
  const int opp_goal = pieces_in_goal(state, board, opponent);

  int own_home = 0;
  int opp_home = 0;
  int own_rear_lag = 0;
  int opp_rear_lag = 0;

  for (int id = 0; id < kBoardSize; ++id) {
    const int occupant = state.cells.at(static_cast<size_t>(id));
    if (occupant == perspective_player) {
      if (board.is_home(perspective_player, id)) {
        ++own_home;
      }
      own_rear_lag =
          std::max(own_rear_lag, best_goal_distance_for_cell(board, perspective_player, id));
    } else if (occupant == opponent) {
      if (board.is_home(opponent, id)) {
        ++opp_home;
      }
      opp_rear_lag = std::max(opp_rear_lag, best_goal_distance_for_cell(board, opponent, id));
    }
  }

  int score = 0;
  score += 18 * (opp_distance - own_distance);
  const int goal_slots_left = kPiecesPerPlayer - own_goal;
  score += 120 * (own_goal - opp_goal);
  score += 16 * (opp_home - own_home);
  score += 7 * (opp_rear_lag - own_rear_lag);
  score += 45 * (goal_blocker_count(state, board, opponent) -
                 goal_blocker_count(state, board, perspective_player));
  if (goal_slots_left <= 3) {
    score += 45 * (opp_distance - own_distance);
    score += 180 * own_goal;
    score -= 80 * own_home;
  }
  return score;
}

std::string state_to_compact_string(const State& state) {
  std::string out;
  out.reserve(kBoardSize);
  for (int id = 0; id < kBoardSize; ++id) {
    const int occupant = state.cells.at(static_cast<size_t>(id));
    if (occupant >= 0 && occupant <= 9) {
      out.push_back(static_cast<char>('0' + occupant));
    } else {
      out.push_back('.');
    }
  }
  return out;
}

std::string move_to_string(const Move& move, const Board& board) {
  if (!move.is_valid()) {
    return "invalid";
  }
  std::ostringstream out;
  out << coord_to_string(board.coord(move.from)) << "->" << coord_to_string(board.coord(move.to));
  return out.str();
}

std::string path_to_string(const Move& move, const Board& board) {
  std::ostringstream out;
  for (size_t i = 0; i < move.path.size(); ++i) {
    if (i != 0) {
      out << " ";
    }
    out << coord_to_string(board.coord(move.path.at(i)));
  }
  return out.str();
}

std::string bot_name(BotKind kind) {
  switch (kind) {
    case BotKind::Random:
      return "random";
    case BotKind::GreedyDistance:
      return "greedy";
    case BotKind::TrafficGreedy:
      return "traffic-greedy";
    case BotKind::HandEval:
      return "hand-eval";
    case BotKind::BeamSearch:
      return "beam";
    case BotKind::Pvs:
      return "pvs";
    case BotKind::Converter:
      return "converter";
    case BotKind::TtPvs:
      return "tt-pvs";
    case BotKind::Policy:
      return "policy";
    case BotKind::PolicyBeam:
      return "policy-beam";
    case BotKind::PuctLite:
      return "puct-lite";
    case BotKind::Mcts:
      return "mcts";
  }
  return "unknown";
}

std::vector<BotKind> all_bot_kinds() {
  return {
      BotKind::Random,
      BotKind::GreedyDistance,
      BotKind::TrafficGreedy,
      BotKind::HandEval,
      BotKind::BeamSearch,
      BotKind::Pvs,
      BotKind::Converter,
      BotKind::TtPvs,
  };
}

bool parse_bot_kind(const std::string& text, BotKind* kind) {
  if (text == "random" || text == "legal-random") {
    *kind = BotKind::Random;
    return true;
  }
  if (text == "greedy" || text == "greedy-distance") {
    *kind = BotKind::GreedyDistance;
    return true;
  }
  if (text == "traffic" || text == "traffic-greedy") {
    *kind = BotKind::TrafficGreedy;
    return true;
  }
  if (text == "hand" || text == "hand-eval" || text == "handeval") {
    *kind = BotKind::HandEval;
    return true;
  }
  if (text == "beam" || text == "beam-search") {
    *kind = BotKind::BeamSearch;
    return true;
  }
  if (text == "pvs" || text == "alpha-beta" || text == "alphabeta") {
    *kind = BotKind::Pvs;
    return true;
  }
  if (text == "converter" || text == "conversion" || text == "goal-converter") {
    *kind = BotKind::Converter;
    return true;
  }
  if (text == "tt-pvs" || text == "ttpvs" || text == "tt" || text == "classical") {
    *kind = BotKind::TtPvs;
    return true;
  }
  if (text == "policy") {
    *kind = BotKind::Policy;
    return true;
  }
  if (text == "policy-beam" || text == "policybeam") {
    *kind = BotKind::PolicyBeam;
    return true;
  }
  if (text == "puct-lite" || text == "puct" || text == "puctlite") {
    *kind = BotKind::PuctLite;
    return true;
  }
  if (text == "mcts" || text == "puct-mcts") {
    *kind = BotKind::Mcts;
    return true;
  }
  return false;
}

namespace {

int searched_score(State& node, const Board& board, const RuleProfile& rules, int root_player,
                   int depth, int alpha, int beta, size_t beam_width) {
  const TerminalStatus status = terminal_status(node, board, rules, nullptr);
  if (depth == 0 || status.terminal) {
    return evaluate_state(node, board, rules, root_player);
  }

  std::vector<Move> moves = legal_moves(node, board, rules);
  if (moves.empty()) {
    return node.player_to_move == root_player ? -900000 : 900000;
  }

  std::vector<std::pair<int, Move>> ordered;
  ordered.reserve(moves.size());
  for (const Move& move : moves) {
    State next = node;
    apply_move(next, move);
    ordered.push_back({evaluate_state(next, board, rules, root_player), move});
  }
  const bool maximize = node.player_to_move == root_player;
  std::sort(ordered.begin(), ordered.end(), [&](const auto& a, const auto& b) {
    if (a.first != b.first) {
      return maximize ? a.first > b.first : a.first < b.first;
    }
    return move_less(a.second, b.second);
  });
  if (beam_width > 0 && ordered.size() > beam_width) {
    ordered.resize(beam_width);
  }

  if (maximize) {
    int best = std::numeric_limits<int>::min();
    for (const auto& [_, move] : ordered) {
      Undo undo;
      apply_move(node, move, &undo);
      best = std::max(best, searched_score(node, board, rules, root_player, depth - 1, alpha, beta,
                                           beam_width));
      undo_move(node, undo);
      alpha = std::max(alpha, best);
      if (alpha >= beta) {
        break;
      }
    }
    return best;
  }

  int best = std::numeric_limits<int>::max();
  for (const auto& [_, move] : ordered) {
    Undo undo;
    apply_move(node, move, &undo);
    best = std::min(best, searched_score(node, board, rules, root_player, depth - 1, alpha, beta,
                                         beam_width));
    undo_move(node, undo);
    beta = std::min(beta, best);
    if (alpha >= beta) {
      break;
    }
  }
  return best;
}

struct TTEntry {
  int depth = -1;
  int score = 0;
};

uint64_t tt_key(const State& state, int root_player) {
  return state.hash() ^ (static_cast<uint64_t>(root_player + 1) * 0x517cc1b727220a95ULL);
}

int searched_score_tt(State& node, const Board& board, const RuleProfile& rules, int root_player,
                      int depth, int alpha, int beta, size_t beam_width,
                      std::unordered_map<uint64_t, TTEntry>& table) {
  const TerminalStatus status = terminal_status(node, board, rules, nullptr);
  if (depth == 0 || status.terminal) {
    return evaluate_state(node, board, rules, root_player);
  }

  const uint64_t key = tt_key(node, root_player);
  const auto cached = table.find(key);
  if (cached != table.end() && cached->second.depth >= depth) {
    return cached->second.score;
  }

  std::vector<Move> moves = legal_moves(node, board, rules);
  if (moves.empty()) {
    return node.player_to_move == root_player ? -900000 : 900000;
  }

  std::vector<std::pair<int, Move>> ordered;
  ordered.reserve(moves.size());
  for (const Move& move : moves) {
    State next = node;
    apply_move(next, move);
    ordered.push_back({evaluate_state(next, board, rules, root_player), move});
  }
  const bool maximize = node.player_to_move == root_player;
  std::sort(ordered.begin(), ordered.end(), [&](const auto& a, const auto& b) {
    if (a.first != b.first) {
      return maximize ? a.first > b.first : a.first < b.first;
    }
    return move_less(a.second, b.second);
  });
  if (beam_width > 0 && ordered.size() > beam_width) {
    ordered.resize(beam_width);
  }

  int best = maximize ? std::numeric_limits<int>::min() : std::numeric_limits<int>::max();
  for (const auto& [_, move] : ordered) {
    Undo undo;
    apply_move(node, move, &undo);
    const int score = searched_score_tt(node, board, rules, root_player, depth - 1, alpha, beta,
                                        beam_width, table);
    undo_move(node, undo);
    if (maximize) {
      best = std::max(best, score);
      alpha = std::max(alpha, best);
    } else {
      best = std::min(best, score);
      beta = std::min(beta, best);
    }
    if (alpha >= beta) {
      break;
    }
  }

  table[key] = TTEntry{depth, best};
  return best;
}

int one_ply_move_score(const State& state, const Board& board, const RuleProfile& rules,
                       const Move& move, BotKind kind) {
  const int player = state.player_to_move;
  State next = state;
  apply_move(next, move);

  if (kind == BotKind::HandEval) {
    return evaluate_state(next, board, rules, player);
  }

  if (kind == BotKind::Converter) {
    int own_home = 0;
    int opp_home = 0;
    const int opponent = 1 - player;
    for (int id = 0; id < kBoardSize; ++id) {
      if (next.cells.at(static_cast<size_t>(id)) == player && board.is_home(player, id)) {
        ++own_home;
      }
      if (next.cells.at(static_cast<size_t>(id)) == opponent && board.is_home(opponent, id)) {
        ++opp_home;
      }
    }

    const int before_dist = best_goal_distance_for_cell(board, player, move.from);
    const int after_dist = best_goal_distance_for_cell(board, player, move.to);
    int score = evaluate_state(next, board, rules, player);
    score += 40 * (before_dist - after_dist);
    score += 260 * pieces_in_goal(next, board, player);
    score -= 180 * own_home;
    score += 45 * opp_home;
    if (board.is_home(player, move.from) && !board.is_home(player, move.to)) {
      score += 240;
    }
    if (!board.is_goal(player, move.from) && board.is_goal(player, move.to)) {
      score += 320;
    }
    if (board.is_goal(player, move.from) && board.is_goal(player, move.to)) {
      score += 40;
    }
    score += static_cast<int>(move.path.size()) * 4;
    return score;
  }

  int score = -10 * total_goal_distance(next, board, player) +
              25 * pieces_in_goal(next, board, player);

  const int before_dist = best_goal_distance_for_cell(board, player, move.from);
  const int after_dist = best_goal_distance_for_cell(board, player, move.to);
  score += 6 * (before_dist - after_dist);

  if (kind == BotKind::TrafficGreedy) {
    score += 2 * mobility_after_landing(next, board, move.to);
    score -= 3 * congestion_around(next, board, move.to);
    if (board.is_home(player, move.from) && !board.is_home(player, move.to)) {
      score += 8;
    }
    if (board.is_goal(player, move.to)) {
      score += 12;
    }
  }

  return score;
}

}  // namespace

Move choose_move_impl(BotKind kind, const State& state, const Board& board,
                      const RuleProfile& rules, std::mt19937_64& rng,
                      const std::unordered_map<uint64_t, int>* repetition_counts) {
  const std::vector<Move> moves = legal_moves(state, board, rules);
  if (moves.empty()) {
    return Move{};
  }

  if (kind == BotKind::Random) {
    std::vector<Move> candidates;
    candidates.reserve(moves.size());
    for (const Move& move : moves) {
      State next = state;
      apply_move(next, move);
      const bool repeats =
          repetition_counts != nullptr && repetition_counts->find(next.hash()) != repetition_counts->end();
      if (!repeats) {
        candidates.push_back(move);
      }
    }
    const std::vector<Move>& pool = candidates.empty() ? moves : candidates;
    std::uniform_int_distribution<size_t> dist(0, pool.size() - 1);
    return pool.at(dist(rng));
  }

  int best_score = std::numeric_limits<int>::min();
  std::vector<Move> tied;
  std::unordered_map<uint64_t, TTEntry> table;

  for (const Move& move : moves) {
    int score = 0;
    if (kind == BotKind::TtPvs) {
      State next = state;
      apply_move(next, move);
      score = searched_score_tt(next, board, rules, state.player_to_move, 2,
                                std::numeric_limits<int>::min() / 2,
                                std::numeric_limits<int>::max() / 2, 24, table);
    } else if (kind == BotKind::BeamSearch || kind == BotKind::Pvs) {
      State next = state;
      apply_move(next, move);
      const int depth = 2;
      const size_t beam_width = kind == BotKind::BeamSearch ? 8 : 16;
      score = searched_score(next, board, rules, state.player_to_move, depth - 1,
                             std::numeric_limits<int>::min() / 2,
                             std::numeric_limits<int>::max() / 2, beam_width);
    } else {
      score = one_ply_move_score(state, board, rules, move, kind);
    }

    if (repetition_counts != nullptr) {
      State next = state;
      apply_move(next, move);
      const auto found = repetition_counts->find(next.hash());
      if (found != repetition_counts->end()) {
        score -= 20000 * found->second;
        if (rules.repetition_draw && found->second + 1 >= rules.repetition_count) {
          score -= 500000;
        }
      }
    }

    if (score > best_score) {
      best_score = score;
      tied.clear();
      tied.push_back(move);
    } else if (score == best_score) {
      tied.push_back(move);
    }
  }

  std::uniform_int_distribution<size_t> dist(0, tied.size() - 1);
  return tied.at(dist(rng));
}

Move choose_move(BotKind kind, const State& state, const Board& board,
                 const RuleProfile& rules, std::mt19937_64& rng) {
  return choose_move_impl(kind, state, board, rules, rng, nullptr);
}

Move choose_move_avoiding_repetition(
    BotKind kind, const State& state, const Board& board, const RuleProfile& rules,
    std::mt19937_64& rng, const std::unordered_map<uint64_t, int>& repetition_counts) {
  return choose_move_impl(kind, state, board, rules, rng, &repetition_counts);
}

std::string json_escape(const std::string& text) {
  std::ostringstream out;
  for (const char ch : text) {
    switch (ch) {
      case '\\':
        out << "\\\\";
        break;
      case '"':
        out << "\\\"";
        break;
      case '\n':
        out << "\\n";
        break;
      case '\r':
        out << "\\r";
        break;
      case '\t':
        out << "\\t";
        break;
      default:
        out << ch;
        break;
    }
  }
  return out.str();
}

void write_jsonl_game_start(std::ostream& out, uint64_t seed, BotKind p0, BotKind p1,
                            const RuleProfile& rules, int max_plies,
                            const State* initial_state) {
  out << "{\"type\":\"game_start\",\"seed\":" << seed << ",\"rule_profile\":\""
      << json_escape(rules.name) << "\",\"p0\":\"" << bot_name(p0) << "\",\"p1\":\""
      << bot_name(p1) << "\",\"max_plies\":" << max_plies;
  if (initial_state != nullptr) {
    out << ",\"initial_cells\":\"" << state_to_compact_string(*initial_state) << "\""
        << ",\"initial_player\":" << initial_state->player_to_move
        << ",\"initial_ply\":" << initial_state->ply;
  }
  out << "}\n";
}

void write_jsonl_move(std::ostream& out, const State& after, const Move& move,
                      const Board& board, int player) {
  out << "{\"type\":\"move\",\"ply\":" << (after.ply - 1) << ",\"player\":" << player
      << ",\"move\":\"" << json_escape(move_to_string(move, board)) << "\",\"from\":"
      << move.from << ",\"to\":" << move.to << ",\"path\":[";
  for (size_t i = 0; i < move.path.size(); ++i) {
    if (i != 0) {
      out << ",";
    }
    out << move.path.at(i);
  }
  out << "],\"hash\":\"" << uint64_to_hex(after.hash()) << "\"}\n";
}

void write_jsonl_game_end(std::ostream& out, const TerminalStatus& status, int plies) {
  out << "{\"type\":\"game_end\",\"plies\":" << plies << ",\"draw\":"
      << (status.draw ? "true" : "false") << ",\"winner\":";
  if (status.winner == kInvalid) {
    out << "null";
  } else {
    out << status.winner;
  }
  out << ",\"reason\":\"" << json_escape(status.reason) << "\"}\n";
}

}  // namespace cczero
