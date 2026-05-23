#pragma once

#include "cczero/cczero.h"
#include "cczero/mcts.h"

#include <fstream>
#include <iosfwd>
#include <string>
#include <vector>

namespace cczero {

std::string require_arg_value(int& index, int argc, char** argv, const std::string& flag);
std::ostream* open_output_stream(const std::string& path, std::ofstream& file);
std::vector<BotKind> parse_bot_list(const std::string& text);
RuleProfile parse_rule_profile(const std::string& text);
std::vector<MovegenBackend> parse_movegen_backend_list(const std::string& text);

}  // namespace cczero
