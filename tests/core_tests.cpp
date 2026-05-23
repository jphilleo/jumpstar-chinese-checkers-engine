#include "cczero/cczero.h"
#include "cczero/cli_utils.h"
#include "cczero/model.h"
#include "cczero/mcts.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <filesystem>
#include <functional>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void require_eq(int actual, int expected, const std::string& message) {
  if (actual != expected) {
    std::ostringstream out;
    out << message << ": expected " << expected << ", got " << actual;
    throw std::runtime_error(out.str());
  }
}

void require_close(double actual, double expected, double tolerance, const std::string& message) {
  if (std::abs(actual - expected) > tolerance) {
    std::ostringstream out;
    out << message << ": expected " << expected << " +/- " << tolerance << ", got " << actual;
    throw std::runtime_error(out.str());
  }
}

void test_board_geometry() {
  const cczero::Board& board = cczero::Board::standard();
  require_eq(board.size(), cczero::kBoardSize, "standard board size");
  require_eq(static_cast<int>(board.home_mask(0).count()), 10, "p0 home size");
  require_eq(static_cast<int>(board.home_mask(1).count()), 10, "p1 home size");
  require_eq(static_cast<int>(board.goal_mask(0).count()), 10, "p0 goal size");
  require_eq(static_cast<int>(board.goal_mask(1).count()), 10, "p1 goal size");
  require_eq(static_cast<int>(board.side_triangle_mask().count()), 40, "side triangle size");
  require(board.id_at(0, 0) != cczero::kInvalid, "center cell exists");
  require(board.id_at(4, -8) != cczero::kInvalid, "top tip exists");
  require(board.id_at(-4, 8) != cczero::kInvalid, "bottom tip exists");
}

void test_board_cached_tables() {
  const cczero::Board& board = cczero::Board::standard();
  for (int player = 0; player < cczero::kPlayers; ++player) {
    std::bitset<cczero::kBoardSize> seen_home;
    for (int id : board.home_cell_ids(player)) {
      require(id >= 0 && id < cczero::kBoardSize, "cached home id is valid");
      require(board.is_home(player, id), "cached home id matches lookup");
      require(board.home_mask(player).test(static_cast<size_t>(id)),
              "cached home id matches mask");
      require(!seen_home.test(static_cast<size_t>(id)), "cached home ids are unique");
      seen_home.set(static_cast<size_t>(id));
    }
    require_eq(static_cast<int>(seen_home.count()), cczero::kPiecesPerPlayer,
               "cached home id count");

    std::bitset<cczero::kBoardSize> seen_goal;
    for (int id : board.goal_cell_ids(player)) {
      require(id >= 0 && id < cczero::kBoardSize, "cached goal id is valid");
      require(board.is_goal(player, id), "cached goal id matches lookup");
      require(board.goal_mask(player).test(static_cast<size_t>(id)),
              "cached goal id matches mask");
      require(!seen_goal.test(static_cast<size_t>(id)), "cached goal ids are unique");
      seen_goal.set(static_cast<size_t>(id));
    }
    require_eq(static_cast<int>(seen_goal.count()), cczero::kPiecesPerPlayer,
               "cached goal id count");

    for (int id = 0; id < cczero::kBoardSize; ++id) {
      require_eq(board.goal_distance(player, id),
                 cczero::goal_distance_for_cell(board, player, id),
                 "cached goal distance matches helper");
    }
  }
}

void test_neighbor_symmetry() {
  const cczero::Board& board = cczero::Board::standard();
  for (int id = 0; id < cczero::kBoardSize; ++id) {
    for (int dir = 0; dir < 6; ++dir) {
      const int neighbor = board.neighbors(id).at(static_cast<size_t>(dir));
      if (neighbor == cczero::kInvalid) {
        continue;
      }
      const int opposite = (dir + 3) % 6;
      require(board.neighbors(neighbor).at(static_cast<size_t>(opposite)) == id,
              "neighbor relation is symmetric");
    }
  }
}

void test_jump_geometry() {
  const cczero::Board& board = cczero::Board::standard();
  int jump_count = 0;
  for (int id = 0; id < cczero::kBoardSize; ++id) {
    for (int dir = 0; dir < 6; ++dir) {
      const int mid = board.jump_mid(id, dir);
      const int landing = board.jump_landing(id, dir);
      if (landing == cczero::kInvalid) {
        continue;
      }
      require(mid != cczero::kInvalid, "valid jump landing has valid midpoint");
      require(cczero::hex_distance(board.coord(id), board.coord(mid)) == 1,
              "jump midpoint is adjacent to source");
      require(cczero::hex_distance(board.coord(id), board.coord(landing)) == 2,
              "jump landing is distance two from source");
      ++jump_count;
    }
  }
  require(jump_count > 0, "board has at least one geometric jump");
}

void test_opening_moves_and_witnesses() {
  const cczero::Board& board = cczero::Board::standard();
  const cczero::RuleProfile rules = cczero::RuleProfile::ccz_121_ab_lg_v1();
  const cczero::State state = cczero::State::initial(board);
  const std::vector<cczero::Move> moves = cczero::legal_moves(state, board, rules);
  require(!moves.empty(), "initial position has legal moves");
  for (const cczero::Move& move : moves) {
    std::string error;
    require(cczero::validate_move_witness(state, move, board, rules, &error),
            "opening move witness validates: " + error);
  }
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

std::vector<std::pair<int, int>> endpoint_signature(
    const std::vector<cczero::MoveEndpoint>& moves) {
  std::vector<std::pair<int, int>> signature;
  signature.reserve(moves.size());
  for (const cczero::MoveEndpoint& move : moves) {
    signature.push_back({move.from, move.to});
  }
  std::sort(signature.begin(), signature.end());
  signature.erase(std::unique(signature.begin(), signature.end()), signature.end());
  return signature;
}

void require_fast_matches_reference(const cczero::State& state, const cczero::Board& board,
                                    const cczero::RuleProfile& rules,
                                    const std::string& label) {
  const std::vector<cczero::Move> reference =
      cczero::legal_moves_reference(state, board, rules);
  const std::vector<cczero::Move> fast = cczero::legal_moves_fast(state, board, rules);
  const std::vector<cczero::Move> bitboard = cczero::legal_moves_bitboard(state, board, rules);
  const std::vector<cczero::MoveEndpoint> bitboard_endpoints =
      cczero::legal_move_endpoints_bitboard(state, board, rules);
  require(endpoint_signature(reference) == endpoint_signature(fast),
          "fast/reference endpoints match for " + label);
  require(endpoint_signature(reference) == endpoint_signature(bitboard),
          "bitboard/reference endpoints match for " + label);
  require(endpoint_signature(reference) == endpoint_signature(bitboard_endpoints),
          "bitboard endpoint/reference endpoints match for " + label);
  for (const cczero::Move& move : fast) {
    std::string error;
    require(cczero::validate_move_witness(state, move, board, rules, &error),
            "fast witness validates for " + label + ": " + error);
  }
  for (const cczero::Move& move : bitboard) {
    std::string error;
    require(cczero::validate_move_witness(state, move, board, rules, &error),
            "bitboard witness validates for " + label + ": " + error);
  }
}

void test_fast_matches_reference() {
  const cczero::Board& board = cczero::Board::standard();
  const cczero::RuleProfile rules = cczero::RuleProfile::ccz_121_ab_lg_v1();

  require_fast_matches_reference(cczero::State::initial(board), board, rules, "initial");

  cczero::State hop = cczero::State::empty();
  hop.cells.at(static_cast<size_t>(board.id_at(0, 0))) = 0;
  hop.cells.at(static_cast<size_t>(board.id_at(1, 0))) = 1;
  hop.cells.at(static_cast<size_t>(board.id_at(3, 0))) = 1;
  require_fast_matches_reference(hop, board, rules, "hop-chain");

  cczero::State state = cczero::State::initial(board);
  std::mt19937_64 rng(2026);
  for (int i = 0; i < 60; ++i) {
    require_fast_matches_reference(state, board, rules, "random-ply-" + std::to_string(i));
    const std::vector<cczero::Move> moves = cczero::legal_moves_reference(state, board, rules);
    require(!moves.empty(), "random fast/reference walk has legal moves");
    std::uniform_int_distribution<size_t> dist(0, moves.size() - 1);
    require(cczero::apply_move(state, moves.at(dist(rng))), "random fast/reference walk applies");
  }
}

void test_frozen_perft_counts() {
  const cczero::Board& board = cczero::Board::standard();
  const cczero::RuleProfile rules = cczero::RuleProfile::ccz_121_ab_lg_v1();

  cczero::State initial_ref = cczero::State::initial(board);
  cczero::State initial_fast = initial_ref;
  cczero::State initial_bitboard = initial_ref;
  require(cczero::perft(initial_ref, board, rules, 1, false) == 14,
          "initial reference perft depth 1");
  require(cczero::perft(initial_fast, board, rules, 1, true) == 14,
          "initial fast perft depth 1");
  require(cczero::legal_moves_bitboard(initial_bitboard, board, rules).size() == 14,
          "initial bitboard move count depth 1");

  initial_ref = cczero::State::initial(board);
  initial_fast = initial_ref;
  require(cczero::perft(initial_ref, board, rules, 2, false) == 196,
          "initial reference perft depth 2");
  require(cczero::perft(initial_fast, board, rules, 2, true) == 196,
          "initial fast perft depth 2");

  cczero::State hop = cczero::State::empty();
  hop.cells.at(static_cast<size_t>(board.id_at(0, 0))) = 0;
  hop.cells.at(static_cast<size_t>(board.id_at(1, 0))) = 1;
  hop.cells.at(static_cast<size_t>(board.id_at(3, 0))) = 1;
  cczero::State hop_fast = hop;
  require(cczero::perft(hop, board, rules, 1, false) == 7,
          "hop-chain reference perft depth 1");
  require(cczero::perft(hop_fast, board, rules, 1, true) == 7,
          "hop-chain fast perft depth 1");

  cczero::State hop_depth_2 = cczero::State::empty();
  hop_depth_2.cells.at(static_cast<size_t>(board.id_at(0, 0))) = 0;
  hop_depth_2.cells.at(static_cast<size_t>(board.id_at(1, 0))) = 1;
  hop_depth_2.cells.at(static_cast<size_t>(board.id_at(3, 0))) = 1;
  cczero::State hop_depth_2_fast = hop_depth_2;
  require(cczero::perft(hop_depth_2, board, rules, 2, false) == 81,
          "hop-chain reference perft depth 2");
  require(cczero::perft(hop_depth_2_fast, board, rules, 2, true) == 81,
          "hop-chain fast perft depth 2");

  cczero::State goal_lock = cczero::State::empty();
  goal_lock.cells.at(static_cast<size_t>(board.id_at(-1, 5))) = 0;
  goal_lock.cells.at(static_cast<size_t>(board.id_at(-2, 5))) = 1;
  cczero::State goal_lock_fast = goal_lock;
  require(cczero::perft(goal_lock, board, rules, 1, false) == 2,
          "goal-lock reference perft depth 1");
  require(cczero::perft(goal_lock_fast, board, rules, 1, true) == 2,
          "goal-lock fast perft depth 1");

  goal_lock = cczero::State::empty();
  goal_lock.cells.at(static_cast<size_t>(board.id_at(-1, 5))) = 0;
  goal_lock.cells.at(static_cast<size_t>(board.id_at(-2, 5))) = 1;
  goal_lock_fast = goal_lock;
  require(cczero::perft(goal_lock, board, rules, 2, false) == 11,
          "goal-lock reference perft depth 2");
  require(cczero::perft(goal_lock_fast, board, rules, 2, true) == 11,
          "goal-lock fast perft depth 2");
}

void test_hop_chain_witness() {
  const cczero::Board& board = cczero::Board::standard();
  const cczero::RuleProfile rules = cczero::RuleProfile::ccz_121_ab_lg_v1();
  cczero::State state = cczero::State::empty();
  const int start = board.id_at(0, 0);
  const int blocker_a = board.id_at(1, 0);
  const int landing_a = board.id_at(2, 0);
  const int blocker_b = board.id_at(3, 0);
  const int landing_b = board.id_at(4, 0);
  require(start != cczero::kInvalid && blocker_a != cczero::kInvalid &&
              landing_a != cczero::kInvalid && blocker_b != cczero::kInvalid &&
              landing_b != cczero::kInvalid,
          "hop-chain fixture cells exist");

  state.cells.at(static_cast<size_t>(start)) = 0;
  state.cells.at(static_cast<size_t>(blocker_a)) = 1;
  state.cells.at(static_cast<size_t>(blocker_b)) = 1;

  const std::vector<cczero::Move> moves = cczero::legal_moves(state, board, rules);
  auto found = std::find_if(moves.begin(), moves.end(), [&](const cczero::Move& move) {
    return move.from == start && move.to == landing_b && move.path.size() == 3 &&
           move.path.at(1) == landing_a;
  });
  require(found != moves.end(), "multi-hop chain to second landing is generated");
  std::string error;
  require(cczero::validate_move_witness(state, *found, board, rules, &error),
          "multi-hop witness validates: " + error);
}

void test_goal_locking() {
  const cczero::Board& board = cczero::Board::standard();
  const cczero::RuleProfile rules = cczero::RuleProfile::ccz_121_ab_lg_v1();
  cczero::State state = cczero::State::empty();
  const int in_goal = board.id_at(-1, 5);
  const int outside_goal = board.id_at(0, 4);
  const int inside_goal = board.id_at(-2, 5);
  require(in_goal != cczero::kInvalid && outside_goal != cczero::kInvalid &&
              inside_goal != cczero::kInvalid,
          "goal-lock fixture cells exist");
  require(board.is_goal(0, in_goal), "fixture source is in p0 goal");
  require(!board.is_goal(0, outside_goal), "fixture outside cell is outside p0 goal");
  require(board.is_goal(0, inside_goal), "fixture inside cell is inside p0 goal");

  state.cells.at(static_cast<size_t>(in_goal)) = 0;
  const std::vector<cczero::Move> moves = cczero::legal_moves(state, board, rules);
  const bool can_leave = std::any_of(moves.begin(), moves.end(), [&](const cczero::Move& move) {
    return move.from == in_goal && move.to == outside_goal;
  });
  const bool can_move_inside =
      std::any_of(moves.begin(), moves.end(), [&](const cczero::Move& move) {
        return move.from == in_goal && move.to == inside_goal;
      });
  require(!can_leave, "goal-locked piece cannot step out of own goal");
  require(can_move_inside, "goal-locked piece can move within own goal");
}

void test_side_triangle_toggle() {
  const cczero::Board& board = cczero::Board::standard();
  cczero::RuleProfile rules = cczero::RuleProfile::ccz_121_ab_lg_v1();
  rules.side_triangles_playable = false;
  cczero::State state = cczero::State::empty();

  int source = cczero::kInvalid;
  int side_landing = cczero::kInvalid;
  for (int id = 0; id < cczero::kBoardSize && source == cczero::kInvalid; ++id) {
    if (board.is_side_triangle(id)) {
      continue;
    }
    for (int neighbor : board.neighbors(id)) {
      if (neighbor != cczero::kInvalid && board.is_side_triangle(neighbor)) {
        source = id;
        side_landing = neighbor;
        break;
      }
    }
  }

  require(source != cczero::kInvalid && side_landing != cczero::kInvalid,
          "side-triangle toggle fixture exists");
  state.cells.at(static_cast<size_t>(source)) = 0;
  const std::vector<cczero::Move> moves = cczero::legal_moves(state, board, rules);
  const bool can_land_in_side =
      std::any_of(moves.begin(), moves.end(), [&](const cczero::Move& move) {
        return move.to == side_landing;
      });
  require(!can_land_in_side, "disabled side triangles cannot be used as landings");
}

void test_apply_undo_round_trip() {
  const cczero::Board& board = cczero::Board::standard();
  const cczero::RuleProfile rules = cczero::RuleProfile::ccz_121_ab_lg_v1();
  cczero::State state = cczero::State::initial(board);
  const cczero::State before = state;
  const std::vector<cczero::Move> moves = cczero::legal_moves(state, board, rules);
  require(!moves.empty(), "round-trip fixture has a move");
  cczero::Undo undo;
  require(cczero::apply_move(state, moves.front(), &undo), "apply move succeeds");
  cczero::undo_move(state, undo);
  require(state.cells == before.cells, "undo restores cells");
  require_eq(state.player_to_move, before.player_to_move, "undo restores player");
  require_eq(state.ply, before.ply, "undo restores ply");
  require(state.hash() == before.hash(), "undo restores hash");
}

void test_apply_undo_random_walk() {
  const cczero::Board& board = cczero::Board::standard();
  const cczero::RuleProfile rules = cczero::RuleProfile::ccz_121_ab_lg_v1();
  cczero::State state = cczero::State::initial(board);
  std::mt19937_64 rng(424242);

  for (int ply = 0; ply < 100; ++ply) {
    const std::vector<cczero::Move> moves =
        cczero::legal_moves_reference(state, board, rules);
    require(!moves.empty(), "random undo walk has legal moves");
    std::uniform_int_distribution<size_t> dist(0, moves.size() - 1);
    const cczero::Move move = moves.at(dist(rng));
    std::string error;
    require(cczero::validate_move_witness(state, move, board, rules, &error),
            "random undo walk move validates: " + error);

    const cczero::State before = state;
    const uint64_t before_hash = state.hash();
    cczero::Undo undo;
    require(cczero::apply_move(state, move, &undo), "random undo walk apply succeeds");
    cczero::undo_move(state, undo);
    require(state.cells == before.cells, "random undo walk restores cells");
    require_eq(state.player_to_move, before.player_to_move,
               "random undo walk restores player");
    require_eq(state.ply, before.ply, "random undo walk restores ply");
    require(state.hash() == before_hash, "random undo walk restores hash");
    require(cczero::apply_move(state, move), "random undo walk advances");
  }
}

void test_terminal_scaffolding() {
  const cczero::Board& board = cczero::Board::standard();
  const cczero::RuleProfile rules = cczero::RuleProfile::ccz_121_ab_lg_v1();
  const cczero::State initial = cczero::State::initial(board);
  const cczero::TerminalStatus initial_status =
      cczero::terminal_status(initial, board, rules, nullptr);
  require(!initial_status.terminal, "initial position is not an anti-block terminal");

  cczero::State won = cczero::State::empty();
  for (int id : board.goal_cells(0)) {
    won.cells.at(static_cast<size_t>(id)) = 0;
  }
  const cczero::TerminalStatus win_status = cczero::terminal_status(won, board, rules, nullptr);
  require(win_status.terminal && win_status.winner == 0, "all p0 pieces in goal wins");

  cczero::State equal_turn = cczero::State::empty();
  bool found_equal_turn_fixture = false;
  int equal_turn_from = cczero::kInvalid;
  int equal_turn_to = cczero::kInvalid;
  const std::vector<int> p1_goal_cells = board.goal_cells(1);
  for (size_t missing_index = 0;
       missing_index < p1_goal_cells.size() && !found_equal_turn_fixture; ++missing_index) {
    cczero::State candidate = cczero::State::empty();
    for (int id : board.goal_cells(0)) {
      candidate.cells.at(static_cast<size_t>(id)) = 0;
    }
    for (size_t i = 0; i < p1_goal_cells.size(); ++i) {
      if (i != missing_index) {
        candidate.cells.at(static_cast<size_t>(p1_goal_cells.at(i))) = 1;
      }
    }
    candidate.player_to_move = 1;
    candidate.ply = 81;
    const int missing_goal = p1_goal_cells.at(missing_index);
    for (int source = 0; source < board.size() && !found_equal_turn_fixture; ++source) {
      if (!candidate.is_empty(source) || board.is_goal(1, source)) {
        continue;
      }
      cczero::State with_source = candidate;
      with_source.cells.at(static_cast<size_t>(source)) = 1;
      for (const cczero::Move& move : cczero::legal_moves(with_source, board, rules)) {
        if (move.from == source && move.to == missing_goal) {
          equal_turn = with_source;
          equal_turn_from = source;
          equal_turn_to = missing_goal;
          found_equal_turn_fixture = true;
          break;
        }
      }
    }
  }
  require(found_equal_turn_fixture, "equal-turn draw fixture has immediate p1 finish");
  const cczero::TerminalStatus equal_turn_status =
      cczero::terminal_status(equal_turn, board, rules, nullptr);
  require(equal_turn_status.terminal && equal_turn_status.draw &&
              equal_turn_status.winner == cczero::kInvalid &&
              equal_turn_status.reason == "equal_turn_goal_draw",
          "p1 immediate reply after p0 goal fill is scored as a draw");

  cczero::State completed_reply = equal_turn;
  cczero::Move finish;
  finish.from = equal_turn_from;
  finish.to = equal_turn_to;
  finish.path = {equal_turn_from, equal_turn_to};
  require(cczero::apply_move(completed_reply, finish), "equal-turn finishing move applies");
  const cczero::TerminalStatus completed_reply_status =
      cczero::terminal_status(completed_reply, board, rules, nullptr);
  require(completed_reply_status.terminal && completed_reply_status.draw &&
              completed_reply_status.reason == "equal_turn_goal_draw",
          "both players full in goal remains a draw");

  cczero::State blocked = cczero::State::empty();
  const std::vector<int> p0_goal = board.goal_cells(0);
  for (size_t i = 0; i + 1 < p0_goal.size(); ++i) {
    blocked.cells.at(static_cast<size_t>(p0_goal.at(i))) = 0;
  }
  blocked.cells.at(static_cast<size_t>(p0_goal.back())) = 1;
  const cczero::TerminalStatus blocked_status =
      cczero::terminal_status(blocked, board, rules, nullptr);
  require(blocked_status.terminal && blocked_status.winner == 0 &&
              blocked_status.reason == "anti_block_goal_full",
          "effective blocked goal awards the player who filled all unblocked goal holes");

  cczero::State red_blocked = cczero::State::empty();
  const std::vector<int> p1_goal = board.goal_cells(1);
  for (size_t i = 0; i + 1 < p1_goal.size(); ++i) {
    red_blocked.cells.at(static_cast<size_t>(p1_goal.at(i))) = 1;
  }
  red_blocked.cells.at(static_cast<size_t>(p1_goal.back())) = 0;
  const cczero::TerminalStatus red_blocked_status =
      cczero::terminal_status(red_blocked, board, rules, nullptr);
  require(red_blocked_status.terminal && red_blocked_status.winner == 1 &&
              red_blocked_status.reason == "anti_block_goal_full",
          "p1/red gets the same effective blocked-goal win as p0");

  // v1 anti-block requires home_count[player] == 0; placing a piece back in
  // the winning player's home should suppress the terminal.
  cczero::State blocked_with_home = blocked;
  blocked_with_home.cells.at(static_cast<size_t>(board.home_cells(0).front())) = 0;
  const cczero::TerminalStatus blocked_with_home_status =
      cczero::terminal_status(blocked_with_home, board, rules, nullptr);
  require(!blocked_with_home_status.terminal,
          "v1 anti-block requires the player to have vacated their own home");

  // v2 anti-block (rules.tighten_anti_block == true) closes the stall-draw
  // loophole: the terminal fires even if the winning player still has pieces
  // in their home triangle.
  const cczero::RuleProfile rules_v2 = cczero::RuleProfile::ccz_121_strict_lg_v2();
  require(rules_v2.tighten_anti_block,
          "v2 strict profile has tightened anti-block enabled");
  const cczero::TerminalStatus v2_blocked_with_home_status =
      cczero::terminal_status(blocked_with_home, board, rules_v2, nullptr);
  require(v2_blocked_with_home_status.terminal &&
              v2_blocked_with_home_status.winner == 0 &&
              v2_blocked_with_home_status.reason == "anti_block_goal_full",
          "v2 anti-block fires even if the winning player still has pieces in their home");

  cczero::State not_full = cczero::State::empty();
  for (size_t i = 0; i + 2 < p1_goal.size(); ++i) {
    not_full.cells.at(static_cast<size_t>(p1_goal.at(i))) = 1;
  }
  not_full.cells.at(static_cast<size_t>(p1_goal.back())) = 0;
  const cczero::TerminalStatus not_full_status =
      cczero::terminal_status(not_full, board, rules, nullptr);
  require(!not_full_status.terminal,
          "anti-block terminal requires every goal hole to be owned or blocked");

  // v2 still requires every goal hole to be owned or blocked. The change is
  // only in the home_count gate, not in the fill condition.
  const cczero::TerminalStatus v2_not_full_status =
      cczero::terminal_status(not_full, board, rules_v2, nullptr);
  require(!v2_not_full_status.terminal,
          "v2 anti-block still requires every goal hole to be owned or blocked");
}

void test_strict_rule_profile() {
  const cczero::Board& board = cczero::Board::standard();
  const cczero::RuleProfile strict = cczero::RuleProfile::ccz_121_strict_lg_v1();
  require(strict.name == "CCZ-121-Strict-LG-v1", "strict profile has stable name");
  require(strict.anti_block_terminal, "strict profile enables effective blocked-goal wins");

  cczero::State blocked = cczero::State::empty();
  const std::vector<int> p0_goal = board.goal_cells(0);
  for (size_t i = 0; i + 1 < p0_goal.size(); ++i) {
    blocked.cells.at(static_cast<size_t>(p0_goal.at(i))) = 0;
  }
  blocked.cells.at(static_cast<size_t>(p0_goal.back())) = 1;
  const cczero::TerminalStatus blocked_status =
      cczero::terminal_status(blocked, board, strict, nullptr);
  require(blocked_status.terminal && blocked_status.winner == 0 &&
              blocked_status.reason == "anti_block_goal_full",
          "strict profile awards effective blocked-goal wins");

  cczero::State won = cczero::State::empty();
  for (int id : board.goal_cells(0)) {
    won.cells.at(static_cast<size_t>(id)) = 0;
  }
  const cczero::TerminalStatus win_status =
      cczero::terminal_status(won, board, strict, nullptr);
  require(win_status.terminal && win_status.winner == 0 &&
              win_status.reason == "all_pieces_in_goal",
          "strict profile still awards goal-fill wins");
}

void test_multiplayer_rule_profiles() {
  const cczero::Board& board = cczero::Board::standard();
  for (int arm = 0; arm < cczero::kStarArms; ++arm) {
    require_eq(static_cast<int>(board.arm_mask(arm).count()), cczero::kPiecesPerPlayer,
               "star arm has 10 holes");
    std::bitset<cczero::kBoardSize> seen;
    for (int id : board.arm_cell_ids(arm)) {
      require(id >= 0 && id < cczero::kBoardSize, "arm cell id is valid");
      require(board.arm_mask(arm).test(static_cast<size_t>(id)), "arm cell id matches mask");
      require(!seen.test(static_cast<size_t>(id)), "arm cell ids are unique");
      seen.set(static_cast<size_t>(id));
    }
  }

  const std::vector<cczero::RuleProfile> profiles = {
      cczero::RuleProfile::ccz_121_mp3_v1(),
      cczero::RuleProfile::ccz_121_mp4_v1(),
      cczero::RuleProfile::ccz_121_mp6_v1(),
  };
  const std::vector<int> expected_counts = {3, 4, 6};
  const std::vector<std::vector<int>> expected_home_arms = {
      {0, 2, 4},
      {0, 1, 3, 4},
      {0, 1, 2, 3, 4, 5},
  };

  for (size_t profile_index = 0; profile_index < profiles.size(); ++profile_index) {
    const cczero::RuleProfile& rules = profiles.at(profile_index);
    require_eq(rules.player_count, expected_counts.at(profile_index),
               "multiplayer profile player count");
    for (int player = 0; player < rules.player_count; ++player) {
      require_eq(rules.home_arm(player), expected_home_arms.at(profile_index).at(player),
                 "multiplayer home arm mapping");
      require_eq(rules.goal_arm(player), (rules.home_arm(player) + 3) % cczero::kStarArms,
                 "multiplayer goal arm mapping");
      require(rules.placement_points.at(static_cast<size_t>(player)) >= 0.0,
              "multiplayer placement points are non-negative");
    }

    cczero::State state = cczero::State::initial(board, rules);
    require_eq(state.player_to_move, 0, "multiplayer initial player");
    require(!cczero::terminal_status(state, board, rules, nullptr).terminal,
            "multiplayer initial state is not terminal");
    for (int player = 0; player < rules.player_count; ++player) {
      require_eq(state.count_pieces(player), cczero::kPiecesPerPlayer,
                 "multiplayer initial piece count");
      for (int id : board.home_cell_ids(rules, player)) {
        require_eq(state.cells.at(static_cast<size_t>(id)), player,
                   "multiplayer home cells contain the player");
      }
    }
    for (int player = rules.player_count; player < cczero::kMaxPlayers; ++player) {
      require_eq(state.count_pieces(player), 0, "inactive multiplayer player has no pieces");
    }

    require_fast_matches_reference(state, board, rules, rules.name + " initial");
    const std::vector<cczero::Move> moves = cczero::legal_moves_reference(state, board, rules);
    require(!moves.empty(), "multiplayer initial state has legal moves");
    std::string error;
    require(cczero::validate_move_witness(state, moves.front(), board, rules, &error),
            "multiplayer initial witness validates: " + error);
    cczero::State moved = state;
    cczero::Undo undo;
    require(cczero::apply_move(moved, moves.front(), rules, &undo),
            "multiplayer apply move succeeds");
    require_eq(moved.player_to_move, 1 % rules.player_count,
               "multiplayer apply advances to next seat");
    cczero::undo_move(moved, undo);
    require(moved.hash() == state.hash(), "multiplayer undo restores hash");

    cczero::State finished = cczero::State::empty();
    for (int id : board.goal_cell_ids(rules, 0)) {
      finished.cells.at(static_cast<size_t>(id)) = 0;
    }
    finished.player_to_move = 1 % rules.player_count;
    require(!cczero::terminal_status(finished, board, rules, nullptr).terminal,
            "multiplayer finisher waits until equal-turn round ends");
    finished.player_to_move = 0;
    const cczero::TerminalStatus status = cczero::terminal_status(finished, board, rules, nullptr);
    require(status.terminal && status.winner == 0 && status.reason == "multiplayer_round_goal",
            "multiplayer terminal fires at equal-turn round boundary");
  }
}

void test_random_playout_smoke() {
  const cczero::Board& board = cczero::Board::standard();
  cczero::RuleProfile rules = cczero::RuleProfile::ccz_121_ab_lg_v1();
  rules.max_plies = 120;
  cczero::State state = cczero::State::initial(board);
  std::mt19937_64 rng(12345);
  std::unordered_map<uint64_t, int> repetitions;
  repetitions[state.hash()] = 1;

  for (;;) {
    const cczero::TerminalStatus status =
        cczero::terminal_status(state, board, rules, &repetitions);
    if (status.terminal) {
      require(state.ply <= rules.max_plies, "random playout respects max plies");
      break;
    }
    const cczero::Move move =
        cczero::choose_move(cczero::BotKind::Random, state, board, rules, rng);
    require(move.is_valid(), "random playout has legal move before terminal");
    std::string error;
    require(cczero::validate_move_witness(state, move, board, rules, &error),
            "random playout witness validates: " + error);
    require(cczero::apply_move(state, move), "random playout apply succeeds");
    require_eq(state.count_pieces(0), 10, "random playout p0 piece count");
    require_eq(state.count_pieces(1), 10, "random playout p1 piece count");
    ++repetitions[state.hash()];
  }
}

void test_baseline_bots_return_legal_moves() {
  const cczero::Board& board = cczero::Board::standard();
  const cczero::RuleProfile rules = cczero::RuleProfile::ccz_121_ab_lg_v1();
  const cczero::State state = cczero::State::initial(board);
  std::mt19937_64 rng(77);
  for (cczero::BotKind bot : cczero::all_bot_kinds()) {
    const cczero::Move move = cczero::choose_move(bot, state, board, rules, rng);
    require(move.is_valid(), "bot returns a move: " + cczero::bot_name(bot));
    std::string error;
    require(cczero::validate_move_witness(state, move, board, rules, &error),
            "bot move validates for " + cczero::bot_name(bot) + ": " + error);
  }
}

void write_zero_floats(std::ofstream& output, size_t count) {
  if (count == 0) {
    return;
  }
  std::vector<float> zeros(count, 0.0f);
  output.write(reinterpret_cast<const char*>(zeros.data()),
               static_cast<std::streamsize>(zeros.size() * sizeof(float)));
}

void write_tiny_ccpv_fixture(const std::string& path, int feature_size = 243) {
  std::ofstream output(path, std::ios::binary);
  require(static_cast<bool>(output), "tiny ccpv fixture opens for writing");

  std::array<char, 16> magic{};
  const std::string magic_text = "CCZPVMLPv1";
  std::copy(magic_text.begin(), magic_text.end(), magic.begin());
  output.write(magic.data(), static_cast<std::streamsize>(magic.size()));

  const int action_size = cczero::kBoardSize * cczero::kBoardSize;
  const int hidden_size = 4;
  const int blocks = 0;
  output.write(reinterpret_cast<const char*>(&feature_size), sizeof(feature_size));
  output.write(reinterpret_cast<const char*>(&action_size), sizeof(action_size));
  output.write(reinterpret_cast<const char*>(&hidden_size), sizeof(hidden_size));
  output.write(reinterpret_cast<const char*>(&blocks), sizeof(blocks));

  write_zero_floats(output, static_cast<size_t>(hidden_size) * feature_size);
  write_zero_floats(output, hidden_size);
  write_zero_floats(output, 0);
  write_zero_floats(output, 0);
  write_zero_floats(output, 0);
  write_zero_floats(output, 0);
  write_zero_floats(output, static_cast<size_t>(action_size) * hidden_size);
  write_zero_floats(output, action_size);
  write_zero_floats(output, hidden_size);
  const float value_b = 0.0f;
  output.write(reinterpret_cast<const char*>(&value_b), sizeof(value_b));
  require(static_cast<bool>(output), "tiny ccpv fixture writes completely");
}

void write_tiny_move_ccpv_fixture(const std::string& path, int feature_size = 243) {
  std::ofstream output(path, std::ios::binary);
  require(static_cast<bool>(output), "tiny move ccpv fixture opens for writing");

  std::array<char, 16> magic{};
  const std::string magic_text = "CCZPVMLPv2";
  std::copy(magic_text.begin(), magic_text.end(), magic.begin());
  output.write(magic.data(), static_cast<std::streamsize>(magic.size()));

  const int action_size = cczero::kBoardSize * cczero::kBoardSize;
  const int hidden_size = 4;
  const int blocks = 0;
  const int policy_head = 1;
  const int move_embed = 2;
  const int move_hidden = 3;
  const int move_feature_size = 20;
  output.write(reinterpret_cast<const char*>(&feature_size), sizeof(feature_size));
  output.write(reinterpret_cast<const char*>(&action_size), sizeof(action_size));
  output.write(reinterpret_cast<const char*>(&hidden_size), sizeof(hidden_size));
  output.write(reinterpret_cast<const char*>(&blocks), sizeof(blocks));
  output.write(reinterpret_cast<const char*>(&policy_head), sizeof(policy_head));
  output.write(reinterpret_cast<const char*>(&move_embed), sizeof(move_embed));
  output.write(reinterpret_cast<const char*>(&move_hidden), sizeof(move_hidden));
  output.write(reinterpret_cast<const char*>(&move_feature_size), sizeof(move_feature_size));

  write_zero_floats(output, static_cast<size_t>(hidden_size) * feature_size);
  write_zero_floats(output, hidden_size);
  write_zero_floats(output, 0);
  write_zero_floats(output, 0);
  write_zero_floats(output, 0);
  write_zero_floats(output, 0);
  write_zero_floats(output, static_cast<size_t>(cczero::kBoardSize) * move_embed);
  write_zero_floats(output, static_cast<size_t>(cczero::kBoardSize) * move_embed);
  write_zero_floats(output,
                    static_cast<size_t>(move_hidden) *
                        (2ULL * static_cast<size_t>(move_embed) + move_feature_size));
  write_zero_floats(output, static_cast<size_t>(move_hidden) * hidden_size);
  write_zero_floats(output, move_hidden);
  write_zero_floats(output, move_hidden);
  const float move_out_b = 0.0f;
  output.write(reinterpret_cast<const char*>(&move_out_b), sizeof(move_out_b));
  write_zero_floats(output, hidden_size);
  const float value_b = 0.0f;
  output.write(reinterpret_cast<const char*>(&value_b), sizeof(value_b));
  require(static_cast<bool>(output), "tiny move ccpv fixture writes completely");
}

void write_tiny_bilinear_ccpv_fixture(const std::string& path, int feature_size = 243) {
  std::ofstream output(path, std::ios::binary);
  require(static_cast<bool>(output), "tiny bilinear ccpv fixture opens for writing");

  std::array<char, 16> magic{};
  const std::string magic_text = "CCZPVMLPv2";
  std::copy(magic_text.begin(), magic_text.end(), magic.begin());
  output.write(magic.data(), static_cast<std::streamsize>(magic.size()));

  const int action_size = cczero::kBoardSize * cczero::kBoardSize;
  const int hidden_size = 4;
  const int blocks = 0;
  const int policy_head = 2;
  const int move_embed = 2;
  const int move_hidden = 3;
  const int move_feature_size = 20;
  output.write(reinterpret_cast<const char*>(&feature_size), sizeof(feature_size));
  output.write(reinterpret_cast<const char*>(&action_size), sizeof(action_size));
  output.write(reinterpret_cast<const char*>(&hidden_size), sizeof(hidden_size));
  output.write(reinterpret_cast<const char*>(&blocks), sizeof(blocks));
  output.write(reinterpret_cast<const char*>(&policy_head), sizeof(policy_head));
  output.write(reinterpret_cast<const char*>(&move_embed), sizeof(move_embed));
  output.write(reinterpret_cast<const char*>(&move_hidden), sizeof(move_hidden));
  output.write(reinterpret_cast<const char*>(&move_feature_size), sizeof(move_feature_size));

  write_zero_floats(output, static_cast<size_t>(hidden_size) * feature_size);
  write_zero_floats(output, hidden_size);
  write_zero_floats(output, 0);
  write_zero_floats(output, 0);
  write_zero_floats(output, 0);
  write_zero_floats(output, 0);
  write_zero_floats(output, static_cast<size_t>(cczero::kBoardSize) * move_embed);
  write_zero_floats(output, static_cast<size_t>(cczero::kBoardSize) * move_embed);
  write_zero_floats(output,
                    static_cast<size_t>(move_hidden) *
                        (2ULL * static_cast<size_t>(move_embed) + move_feature_size));
  write_zero_floats(output, static_cast<size_t>(move_hidden) * hidden_size);
  write_zero_floats(output, move_hidden);
  write_zero_floats(output, 2ULL * static_cast<size_t>(move_embed) + move_feature_size);
  const float move_bias_b = 0.0f;
  output.write(reinterpret_cast<const char*>(&move_bias_b), sizeof(move_bias_b));
  write_zero_floats(output, hidden_size);
  const float value_b = 0.0f;
  output.write(reinterpret_cast<const char*>(&value_b), sizeof(value_b));
  require(static_cast<bool>(output), "tiny bilinear ccpv fixture writes completely");
}

cczero::PolicyModel load_tiny_policy_value_model(const std::string& name,
                                                 int feature_size = 243) {
  const std::filesystem::path path = std::filesystem::temp_directory_path() / name;
  write_tiny_ccpv_fixture(path.string(), feature_size);
  return cczero::load_policy_model(path.string());
}

cczero::PolicyModel load_tiny_move_policy_value_model(const std::string& name,
                                                      int feature_size = 243) {
  const std::filesystem::path path = std::filesystem::temp_directory_path() / name;
  write_tiny_move_ccpv_fixture(path.string(), feature_size);
  return cczero::load_policy_model(path.string());
}

cczero::PolicyModel load_tiny_bilinear_policy_value_model(const std::string& name,
                                                          int feature_size = 243) {
  const std::filesystem::path path = std::filesystem::temp_directory_path() / name;
  write_tiny_bilinear_ccpv_fixture(path.string(), feature_size);
  return cczero::load_policy_model(path.string());
}

const cczero::MctsRootMove* find_root_move(const std::vector<cczero::MctsRootMove>& moves,
                                           int from, int to) {
  const auto found = std::find_if(moves.begin(), moves.end(), [&](const cczero::MctsRootMove& move) {
    return move.move.from == from && move.move.to == to;
  });
  return found == moves.end() ? nullptr : &*found;
}

void test_policy_value_model_runtime_fixture() {
  const cczero::PolicyModel model =
      load_tiny_policy_value_model("cczero_core_tiny_model.ccpv");
  require(model.kind == cczero::ModelKind::PolicyValueMlp, "tiny model loads as policy/value MLP");
  require(model.policy_head == cczero::PolicyHeadKind::Dense, "tiny model uses dense policy head");
  require_eq(model.feature_size, 243, "tiny model feature size");
  require_eq(model.action_size, cczero::kBoardSize * cczero::kBoardSize,
             "tiny model action size");
  require_eq(model.hidden_size, 4, "tiny model hidden size");
  require_eq(model.blocks, 0, "tiny model block count");
  cczero::validate_policy_model(model, "tiny_policy_value");
  const size_t expected_parameters =
      static_cast<size_t>(model.hidden_size) * static_cast<size_t>(model.feature_size) +
      static_cast<size_t>(model.hidden_size) +
      2ULL * static_cast<size_t>(model.blocks) *
          static_cast<size_t>(model.hidden_size) * static_cast<size_t>(model.hidden_size) +
      2ULL * static_cast<size_t>(model.blocks) * static_cast<size_t>(model.hidden_size) +
      static_cast<size_t>(model.action_size) * static_cast<size_t>(model.hidden_size) +
      static_cast<size_t>(model.action_size) + static_cast<size_t>(model.hidden_size) + 1ULL;
  require(cczero::policy_model_parameter_count(model) == expected_parameters,
          "policy/value model parameter count matches sections");
  require(cczero::policy_model_storage_bytes(model) >= expected_parameters * sizeof(float),
          "policy/value model storage includes parameters");

  const cczero::Board& board = cczero::Board::standard();
  const cczero::RuleProfile rules = cczero::RuleProfile::ccz_121_ab_lg_v1();
  const cczero::State state = cczero::State::initial(board);
  const std::vector<float> features =
      cczero::encode_policy_features(state, state.player_to_move);
  require_eq(static_cast<int>(features.size()), 243, "encoded feature count");
  require(features.at(242) == 1.0f, "encoded side-to-move feature");
  const std::vector<float> geometry_features =
      cczero::encode_policy_features(state, state.player_to_move, 1465);
  require_eq(static_cast<int>(geometry_features.size()), 1465, "geometry feature count");
  require(geometry_features.at(242) == 1.0f, "geometry side-to-move feature");
  const std::vector<float> geometry_v2_features =
      cczero::encode_policy_features(state, state.player_to_move, 2562);
  require_eq(static_cast<int>(geometry_v2_features.size()), 2562, "geometry v2 feature count");
  require(geometry_v2_features.at(242) == 1.0f, "geometry v2 side-to-move feature");
  const std::vector<float> geometry_v3_features =
      cczero::encode_policy_features(state, state.player_to_move, 3064);
  require_eq(static_cast<int>(geometry_v3_features.size()), 3064, "geometry v3 feature count");
  require(geometry_v3_features.at(242) == 1.0f, "geometry v3 side-to-move feature");

  const std::vector<float> reference_hidden = cczero::mlp_hidden(model, features);
  cczero::MlpWorkspace workspace;
  cczero::mlp_hidden_optimized(model, state, state.player_to_move, workspace,
                               cczero::InferenceBackend::Portable);
  require(workspace.hidden.size() >= reference_hidden.size(), "optimized hidden buffer exists");
  for (size_t i = 0; i < reference_hidden.size(); ++i) {
    require(std::abs(workspace.hidden.at(i) - reference_hidden.at(i)) <= 1.0e-6f,
            "optimized hidden matches scalar hidden");
  }
  const float value = cczero::mlp_value_from_hidden_ptr(model, workspace.hidden.data());
  require(std::abs(value) <= 1.0e-6f, "tiny model value is zero");

  const std::vector<cczero::Move> moves =
      cczero::legal_moves_bitboard(state, board, rules);
  require(!moves.empty(), "tiny model fixture has a legal policy action");
  require(std::abs(cczero::policy_score(model, state, state.player_to_move, moves.front())) <=
              1.0e-6f,
          "tiny model policy score is zero");

  const cczero::PolicyModel move_model =
      load_tiny_move_policy_value_model("cczero_core_tiny_move_model.ccpv");
  require(move_model.kind == cczero::ModelKind::PolicyValueMlp,
          "tiny move model loads as policy/value MLP");
  require(move_model.policy_head == cczero::PolicyHeadKind::MoveMlp,
          "tiny move model uses move policy head");
  require_eq(move_model.move_embed_size, 2, "tiny move model embed size");
  require_eq(move_model.move_hidden_size, 3, "tiny move model hidden size");
  require_eq(move_model.move_feature_size, 20, "tiny move model feature size");
  cczero::validate_policy_model(move_model, "tiny_move_policy_value");
  const size_t expected_move_parameters =
      static_cast<size_t>(move_model.hidden_size) *
          static_cast<size_t>(move_model.feature_size) +
      static_cast<size_t>(move_model.hidden_size) +
      static_cast<size_t>(cczero::kBoardSize) *
          static_cast<size_t>(move_model.move_embed_size) * 2ULL +
      static_cast<size_t>(move_model.move_hidden_size) *
          (2ULL * static_cast<size_t>(move_model.move_embed_size) +
           static_cast<size_t>(move_model.move_feature_size)) +
      static_cast<size_t>(move_model.move_hidden_size) *
          static_cast<size_t>(move_model.hidden_size) +
      static_cast<size_t>(move_model.move_hidden_size) +
      static_cast<size_t>(move_model.move_hidden_size) + 1ULL +
      static_cast<size_t>(move_model.hidden_size) + 1ULL;
  require(cczero::policy_model_parameter_count(move_model) == expected_move_parameters,
          "move policy/value model parameter count matches sections");
  require(std::abs(cczero::policy_score(move_model, state, state.player_to_move, moves.front())) <=
              1.0e-6f,
          "tiny move model policy score is zero");

  const cczero::PolicyModel bilinear_model =
      load_tiny_bilinear_policy_value_model("cczero_core_tiny_bilinear_model.ccpv");
  require(bilinear_model.kind == cczero::ModelKind::PolicyValueMlp,
          "tiny bilinear model loads as policy/value MLP");
  require(bilinear_model.policy_head == cczero::PolicyHeadKind::MoveBilinear,
          "tiny bilinear model uses bilinear policy head");
  cczero::validate_policy_model(bilinear_model, "tiny_bilinear_policy_value");
  const size_t expected_bilinear_parameters =
      static_cast<size_t>(bilinear_model.hidden_size) *
          static_cast<size_t>(bilinear_model.feature_size) +
      static_cast<size_t>(bilinear_model.hidden_size) +
      static_cast<size_t>(cczero::kBoardSize) *
          static_cast<size_t>(bilinear_model.move_embed_size) * 2ULL +
      static_cast<size_t>(bilinear_model.move_hidden_size) *
          (2ULL * static_cast<size_t>(bilinear_model.move_embed_size) +
           static_cast<size_t>(bilinear_model.move_feature_size)) +
      static_cast<size_t>(bilinear_model.move_hidden_size) *
          static_cast<size_t>(bilinear_model.hidden_size) +
      static_cast<size_t>(bilinear_model.move_hidden_size) +
      (2ULL * static_cast<size_t>(bilinear_model.move_embed_size) +
       static_cast<size_t>(bilinear_model.move_feature_size)) +
      1ULL + static_cast<size_t>(bilinear_model.hidden_size) + 1ULL;
  require(cczero::policy_model_parameter_count(bilinear_model) ==
              expected_bilinear_parameters,
          "bilinear policy/value model parameter count matches sections");
  require(std::abs(cczero::policy_score(bilinear_model, state, state.player_to_move,
                                        moves.front())) <= 1.0e-6f,
          "tiny bilinear model policy score is zero");

  const std::vector<const cczero::State*> batch_states{&state, &state};
  cczero::MlpWorkspace batch_workspace;
  cczero::mlp_hidden_batch_optimized(model, batch_states, batch_workspace,
                                     cczero::InferenceBackend::Portable);
  require(batch_workspace.values.size() >= batch_states.size(), "batch values buffer exists");
  require(std::abs(batch_workspace.values.at(0)) <= 1.0e-6f &&
              std::abs(batch_workspace.values.at(1)) <= 1.0e-6f,
          "tiny model batched values are zero");

  const cczero::PolicyModel geometry_model =
      load_tiny_policy_value_model("cczero_core_tiny_geometry_model.ccpv", 1465);
  require_eq(geometry_model.feature_size, 1465, "tiny geometry model feature size");
  const std::vector<float> geometry_reference_hidden =
      cczero::mlp_hidden(geometry_model, geometry_features);
  cczero::MlpWorkspace geometry_workspace;
  cczero::mlp_hidden_optimized(geometry_model, state, state.player_to_move, geometry_workspace,
                               cczero::InferenceBackend::Portable);
  for (size_t i = 0; i < geometry_reference_hidden.size(); ++i) {
    require(std::abs(geometry_workspace.hidden.at(i) - geometry_reference_hidden.at(i)) <=
                1.0e-6f,
            "geometry optimized hidden matches scalar hidden");
  }
  const cczero::PolicyModel geometry_v2_model =
      load_tiny_policy_value_model("cczero_core_tiny_geometry_v2_model.ccpv", 2562);
  require_eq(geometry_v2_model.feature_size, 2562, "tiny geometry v2 model feature size");
  const std::vector<float> geometry_v2_reference_hidden =
      cczero::mlp_hidden(geometry_v2_model, geometry_v2_features);
  cczero::MlpWorkspace geometry_v2_workspace;
  cczero::mlp_hidden_optimized(geometry_v2_model, state, state.player_to_move,
                               geometry_v2_workspace, cczero::InferenceBackend::Portable);
  for (size_t i = 0; i < geometry_v2_reference_hidden.size(); ++i) {
    require(std::abs(geometry_v2_workspace.hidden.at(i) - geometry_v2_reference_hidden.at(i)) <=
                1.0e-6f,
            "geometry v2 optimized hidden matches scalar hidden");
  }
  const cczero::PolicyModel geometry_v3_model =
      load_tiny_policy_value_model("cczero_core_tiny_geometry_v3_model.ccpv", 3064);
  require_eq(geometry_v3_model.feature_size, 3064, "tiny geometry v3 model feature size");
  const std::vector<float> geometry_v3_reference_hidden =
      cczero::mlp_hidden(geometry_v3_model, geometry_v3_features);
  cczero::MlpWorkspace geometry_v3_workspace;
  cczero::mlp_hidden_optimized(geometry_v3_model, state, state.player_to_move,
                               geometry_v3_workspace, cczero::InferenceBackend::Portable);
  for (size_t i = 0; i < geometry_v3_reference_hidden.size(); ++i) {
    require(std::abs(geometry_v3_workspace.hidden.at(i) - geometry_v3_reference_hidden.at(i)) <=
                1.0e-6f,
            "geometry v3 optimized hidden matches scalar hidden");
  }
}

void test_mcts_runtime_fixture() {
  const cczero::PolicyModel model =
      load_tiny_policy_value_model("cczero_core_tiny_mcts_model.ccpv");
  const cczero::PolicyModel move_model =
      load_tiny_move_policy_value_model("cczero_core_tiny_move_mcts_model.ccpv");

  const cczero::Board& board = cczero::Board::standard();
  cczero::RuleProfile rules = cczero::RuleProfile::ccz_121_strict_lg_v1();
  rules.max_plies = 1;
  const cczero::State state = cczero::State::initial(board);
  std::unordered_map<uint64_t, int> repetitions;
  repetitions[state.hash()] = 1;
  cczero::MctsConfig config;
  config.simulations = 6;
  config.add_root_noise = false;
  config.temperature = 0.0;
  config.movegen = cczero::MovegenBackend::Bitboard;
  config.inference_backend = cczero::InferenceBackend::Portable;
  config.inference_batch_size = 1;
  std::mt19937_64 rng(20260516);

  const cczero::MctsResult result =
      cczero::run_mcts_search(state, board, rules, model, repetitions, config, rng);
  require(result.move.is_valid(), "tiny MCTS returns a valid move");
  std::string error;
  require(cczero::validate_move_witness(state, result.move, board, rules, &error),
          "tiny MCTS move witness validates: " + error);
  require_eq(result.move.from, 3, "tiny MCTS golden move source");
  require_eq(result.move.to, 14, "tiny MCTS golden move destination");

  std::mt19937_64 move_rng(20260516);
  const cczero::MctsResult move_result =
      cczero::run_mcts_search(state, board, rules, move_model, repetitions, config, move_rng);
  require(move_result.move.is_valid(), "tiny move-head MCTS returns a valid move");
  require_eq(move_result.move.from, 3, "tiny move-head MCTS golden move source");
  require_eq(move_result.move.to, 14, "tiny move-head MCTS golden move destination");
  require_eq(result.stats.simulations, 6, "tiny MCTS simulation count");
  require(!result.root_moves.empty(), "tiny MCTS exposes root visit targets");
  require_eq(result.stats.root_legal_moves, static_cast<int>(result.root_moves.size()),
             "tiny MCTS root legal count matches target count");
  require_eq(static_cast<int>(result.root_moves.size()), 14, "tiny MCTS golden root move count");

  const std::vector<std::pair<int, int>> expected_endpoints = {
      {3, 14}, {3, 16}, {4, 15}, {4, 17}, {5, 16}, {5, 18}, {6, 14},
      {6, 15}, {7, 15}, {7, 16}, {8, 16}, {8, 17}, {9, 17}, {9, 18},
  };
  for (size_t i = 0; i < expected_endpoints.size(); ++i) {
    require_eq(result.root_moves.at(i).move.from, expected_endpoints.at(i).first,
               "tiny MCTS golden root source");
    require_eq(result.root_moves.at(i).move.to, expected_endpoints.at(i).second,
               "tiny MCTS golden root destination");
    require_eq(result.root_moves.at(i).visits, i < 6 ? 1 : 0,
               "tiny MCTS golden root visit count");
  }
  for (size_t i = 0; i < 6; ++i) {
    require_close(result.root_moves.at(i).prior, 1.0 / 14.0, 1.0e-5,
                  "tiny MCTS zero-shaping uniform prior");
  }

  cczero::MctsConfig endpoint_only = config;
  endpoint_only.materialize_root_moves = false;
  std::mt19937_64 endpoint_rng(20260516);
  const cczero::MctsResult endpoint_result =
      cczero::run_mcts_search(state, board, rules, model, repetitions, endpoint_only,
                              endpoint_rng);
  require_eq(endpoint_result.move.from, result.move.from,
             "tiny MCTS endpoint-only preserves move source");
  require_eq(endpoint_result.move.to, result.move.to,
             "tiny MCTS endpoint-only preserves move destination");
  require(cczero::validate_move_witness(state, endpoint_result.move, board, rules, &error),
          "tiny MCTS endpoint-only chosen witness validates: " + error);
  require_eq(static_cast<int>(endpoint_result.root_moves.size()),
             static_cast<int>(result.root_moves.size()),
             "tiny MCTS endpoint-only preserves root size");
  for (size_t i = 0; i < result.root_moves.size(); ++i) {
    require_eq(endpoint_result.root_moves.at(i).move.from, result.root_moves.at(i).move.from,
               "tiny MCTS endpoint-only root source");
    require_eq(endpoint_result.root_moves.at(i).move.to, result.root_moves.at(i).move.to,
               "tiny MCTS endpoint-only root destination");
    require_eq(endpoint_result.root_moves.at(i).visits, result.root_moves.at(i).visits,
               "tiny MCTS endpoint-only visits");
    require(endpoint_result.root_moves.at(i).move.path.empty(),
            "tiny MCTS endpoint-only root path is not materialized");
  }

  cczero::MctsConfig no_transpositions = config;
  no_transpositions.transpositions = false;
  std::mt19937_64 no_transposition_rng(20260516);
  const cczero::MctsResult without_transpositions =
      cczero::run_mcts_search(state, board, rules, model, repetitions, no_transpositions,
                              no_transposition_rng);
  require_eq(without_transpositions.move.from, result.move.from,
             "tiny MCTS transposition toggle preserves move source");
  require_eq(without_transpositions.move.to, result.move.to,
             "tiny MCTS transposition toggle preserves move destination");
  require_eq(static_cast<int>(without_transpositions.root_moves.size()),
             static_cast<int>(result.root_moves.size()),
             "tiny MCTS transposition toggle preserves root size");
  for (size_t i = 0; i < result.root_moves.size(); ++i) {
    require_eq(without_transpositions.root_moves.at(i).visits, result.root_moves.at(i).visits,
               "tiny MCTS transposition toggle preserves visits");
  }
}

void test_mcts_reuse_and_adaptive_runtime() {
  const cczero::PolicyModel model =
      load_tiny_policy_value_model("cczero_core_tiny_mcts_reuse_model.ccpv");
  const cczero::Board& board = cczero::Board::standard();
  const cczero::RuleProfile rules = cczero::RuleProfile::ccz_121_strict_lg_v1();
  cczero::State state = cczero::State::initial(board);
  std::unordered_map<uint64_t, int> repetitions;
  repetitions[state.hash()] = 1;

  cczero::MctsConfig config;
  config.simulations = 6;
  config.add_root_noise = false;
  config.temperature = 0.0;
  config.movegen = cczero::MovegenBackend::Bitboard;
  config.inference_backend = cczero::InferenceBackend::Portable;
  config.inference_batch_size = 1;
  config.reuse_tree = true;
  std::mt19937_64 rng(8080);

  cczero::MctsResult first =
      cczero::run_mcts_search(state, board, rules, model, repetitions, config, rng);
  require(!first.stats.reused_tree, "first MCTS search does not reuse a tree");
  require(!first.tree.empty(), "reuse-enabled search returns a reusable tree");
  require(first.selected_child_node >= 0, "reuse-enabled search returns selected child node");
  require(cczero::apply_move(state, first.move), "reuse test applies first move");
  ++repetitions[state.hash()];

  std::vector<cczero::MctsNode> reusable_tree = std::move(first.tree);
  int reusable_root = first.selected_child_node;
  cczero::MctsResult second = cczero::run_mcts_search(
      state, board, rules, model, repetitions, config, rng, &reusable_tree, &reusable_root);
  require(second.stats.reused_tree, "second MCTS search reuses the selected subtree");
  require(second.move.is_valid(), "reused MCTS search returns a valid move");

  cczero::State context_state = cczero::State::initial(board);
  std::unordered_map<uint64_t, int> context_repetitions;
  context_repetitions[context_state.hash()] = 1;
  cczero::MctsSearchContext context;
  std::mt19937_64 context_rng(8181);
  const cczero::MctsResult context_first =
      cczero::run_mcts_search(context_state, board, rules, model, context_repetitions, config,
                              context_rng, &context);
  require(!context_first.stats.reused_tree, "context first search starts fresh");
  require(cczero::apply_move(context_state, context_first.move), "context reuse applies move");
  ++context_repetitions[context_state.hash()];
  const cczero::MctsResult context_second =
      cczero::run_mcts_search(context_state, board, rules, model, context_repetitions, config,
                              context_rng, &context);
  require(context_second.stats.reused_tree, "context second search reuses selected subtree");
  context.clear();
  const cczero::MctsResult context_after_clear =
      cczero::run_mcts_search(context_state, board, rules, model, context_repetitions, config,
                              context_rng, &context);
  require(!context_after_clear.stats.reused_tree, "cleared context starts fresh");

  cczero::MctsConfig adaptive = config;
  adaptive.reuse_tree = false;
  adaptive.simulations = 16;
  adaptive.min_simulations = 4;
  adaptive.adaptive_simulations = true;
  adaptive.adaptive_check_interval = 1;
  adaptive.adaptive_confidence = 0.20;
  std::mt19937_64 adaptive_rng(9090);
  const cczero::MctsResult adaptive_result =
      cczero::run_mcts_search(cczero::State::initial(board), board, rules, model,
                              {{cczero::State::initial(board).hash(), 1}}, adaptive, adaptive_rng);
  require(adaptive_result.stats.adaptive_stopped, "adaptive MCTS stops early");
  require(adaptive_result.stats.simulations < adaptive.simulations,
          "adaptive MCTS uses fewer than configured simulations");
}

void test_mcts_repetition_penalty_runtime() {
  const cczero::PolicyModel model =
      load_tiny_policy_value_model("cczero_core_tiny_mcts_repetition_model.ccpv");
  const cczero::Board& board = cczero::Board::standard();
  const cczero::RuleProfile rules = cczero::RuleProfile::ccz_121_strict_lg_v1();
  const cczero::State state = cczero::State::initial(board);
  const std::vector<cczero::Move> legal = cczero::legal_moves_bitboard(state, board, rules);
  const auto repeated_move = std::find_if(legal.begin(), legal.end(), [](const cczero::Move& move) {
    return move.from == 3 && move.to == 14;
  });
  require(repeated_move != legal.end(), "repetition penalty fixture move exists");

  cczero::State repeated_child = state;
  require(cczero::apply_move(repeated_child, *repeated_move), "repetition penalty child applies");
  std::unordered_map<uint64_t, int> repetitions;
  repetitions[state.hash()] = 1;
  repetitions[repeated_child.hash()] = rules.repetition_count - 1;

  cczero::MctsConfig config;
  config.simulations = 6;
  config.add_root_noise = false;
  config.temperature = 0.0;
  config.movegen = cczero::MovegenBackend::Bitboard;
  config.inference_backend = cczero::InferenceBackend::Portable;
  config.inference_batch_size = 1;
  std::mt19937_64 rng(10010);
  const cczero::MctsResult result =
      cczero::run_mcts_search(state, board, rules, model, repetitions, config, rng);

  const cczero::MctsRootMove* penalized = find_root_move(result.root_moves, 3, 14);
  const cczero::MctsRootMove* unpenalized = find_root_move(result.root_moves, 3, 16);
  require(penalized != nullptr && unpenalized != nullptr,
          "repetition penalty root moves are present");
  require(penalized->prior < 0.01, "repetition penalty strongly reduces repeated child prior");
  require(unpenalized->prior > 0.05, "repetition penalty leaves sibling prior available");
  require(!(result.move.from == 3 && result.move.to == 14),
          "repetition penalty avoids choosing the repeated child");
}

void test_policy_bot_names_parse() {
  cczero::BotKind bot = cczero::BotKind::Random;
  require(cczero::parse_bot_kind("policy", &bot), "policy bot parses");
  require(bot == cczero::BotKind::Policy, "policy bot kind");
  require(cczero::parse_bot_kind("policy-beam", &bot), "policy-beam bot parses");
  require(bot == cczero::BotKind::PolicyBeam, "policy-beam bot kind");
  require(cczero::parse_bot_kind("puct-lite", &bot), "puct-lite bot parses");
  require(bot == cczero::BotKind::PuctLite, "puct-lite bot kind");
  require(cczero::bot_name(cczero::BotKind::PuctLite) == "puct-lite",
          "puct-lite bot name is stable");
  require(cczero::parse_bot_kind("mcts", &bot), "mcts bot parses");
  require(bot == cczero::BotKind::Mcts, "mcts bot kind");
  require(cczero::bot_name(cczero::BotKind::Mcts) == "mcts", "mcts bot name is stable");
}

void test_cli_parser_helpers() {
  // "strict" now defaults to the v2 profile (anti-block loophole closed).
  // The explicit "-v1" suffix routes to the legacy profile for replaying
  // historical games.
  const cczero::RuleProfile strict = cczero::parse_rule_profile("strict");
  require(strict.name == "CCZ-121-Strict-LG-v2", "strict rule helper parses to v2");
  require(strict.tighten_anti_block, "default strict profile is v2 (tightened anti-block)");
  const cczero::RuleProfile strict_v1 = cczero::parse_rule_profile("strict-v1");
  require(strict_v1.name == "CCZ-121-Strict-LG-v1", "explicit strict-v1 routes to legacy profile");
  require(!strict_v1.tighten_anti_block, "strict-v1 preserves the v1 anti-block gate");
  const std::vector<cczero::BotKind> bots = cczero::parse_bot_list("random,mcts");
  require_eq(static_cast<int>(bots.size()), 2, "bot helper parses two entries");
  require(bots.at(0) == cczero::BotKind::Random && bots.at(1) == cczero::BotKind::Mcts,
          "bot helper preserves order");
  const std::vector<cczero::MovegenBackend> backends =
      cczero::parse_movegen_backend_list("reference, bitboard");
  require_eq(static_cast<int>(backends.size()), 2, "movegen helper parses two entries");
  require(backends.at(0) == cczero::MovegenBackend::Reference &&
              backends.at(1) == cczero::MovegenBackend::Bitboard,
          "movegen helper preserves order");
}

}  // namespace

int main() {
  const std::vector<std::pair<std::string, std::function<void()>>> tests = {
      {"board geometry", test_board_geometry},
      {"board cached tables", test_board_cached_tables},
      {"neighbor symmetry", test_neighbor_symmetry},
      {"jump geometry", test_jump_geometry},
      {"opening moves and witnesses", test_opening_moves_and_witnesses},
      {"fast matches reference", test_fast_matches_reference},
      {"frozen perft counts", test_frozen_perft_counts},
      {"hop-chain witness", test_hop_chain_witness},
      {"goal locking", test_goal_locking},
      {"side-triangle toggle", test_side_triangle_toggle},
      {"apply/undo round trip", test_apply_undo_round_trip},
      {"apply/undo random walk", test_apply_undo_random_walk},
      {"terminal scaffolding", test_terminal_scaffolding},
      {"strict rule profile", test_strict_rule_profile},
      {"multiplayer rule profiles", test_multiplayer_rule_profiles},
      {"random playout smoke", test_random_playout_smoke},
      {"baseline bots return legal moves", test_baseline_bots_return_legal_moves},
      {"policy/value model runtime fixture", test_policy_value_model_runtime_fixture},
      {"mcts runtime fixture", test_mcts_runtime_fixture},
      {"mcts reuse and adaptive runtime", test_mcts_reuse_and_adaptive_runtime},
      {"mcts repetition penalty runtime", test_mcts_repetition_penalty_runtime},
      {"policy bot names parse", test_policy_bot_names_parse},
      {"cli parser helpers", test_cli_parser_helpers},
  };

  int failed = 0;
  for (const auto& [name, test] : tests) {
    try {
      test();
      std::cout << "[PASS] " << name << "\n";
    } catch (const std::exception& error) {
      ++failed;
      std::cerr << "[FAIL] " << name << ": " << error.what() << "\n";
    }
  }

  if (failed != 0) {
    std::cerr << failed << " test(s) failed\n";
    return 1;
  }
  std::cout << tests.size() << " test(s) passed\n";
  return 0;
}
