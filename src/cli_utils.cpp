#include "cczero/cli_utils.h"

#include <algorithm>
#include <cctype>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace cczero {

std::string require_arg_value(int& index, int argc, char** argv, const std::string& flag) {
  if (index + 1 >= argc) {
    throw std::runtime_error("missing value for " + flag);
  }
  ++index;
  return argv[index];
}

std::ostream* open_output_stream(const std::string& path, std::ofstream& file) {
  if (path == "-") {
    return &std::cout;
  }
  file.open(path);
  if (!file) {
    throw std::runtime_error("failed to open output path: " + path);
  }
  return &file;
}

std::vector<BotKind> parse_bot_list(const std::string& text) {
  if (text.empty() || text == "all") {
    return all_bot_kinds();
  }

  std::vector<BotKind> bots;
  std::stringstream input(text);
  std::string item;
  while (std::getline(input, item, ',')) {
    item.erase(std::remove_if(item.begin(), item.end(), [](unsigned char ch) {
                 return std::isspace(ch) != 0;
               }),
               item.end());
    if (item.empty()) {
      continue;
    }
    BotKind bot = BotKind::Random;
    if (!parse_bot_kind(item, &bot)) {
      throw std::runtime_error("unknown bot in list: " + item);
    }
    bots.push_back(bot);
  }
  if (bots.empty()) {
    throw std::runtime_error("bot list is empty");
  }
  return bots;
}

RuleProfile parse_rule_profile(const std::string& text) {
  // v2 profiles close the anti-block stall loophole present in v1. Pass an
  // explicit "-v1" suffix to opt back into the legacy rule for replaying
  // pre-2026-05-21 logs. Default "ab" / "strict" routes to v2 going forward;
  // accept the long-form v1 names for explicit reproducibility.
  if (text == "ab-v2" || text == "anti-block-v2" || text == "CCZ-121-AB-LG-v2") {
    return RuleProfile::ccz_121_ab_lg_v2();
  }
  if (text == "strict-v2" || text == "strict-lg-v2" || text == "CCZ-121-Strict-LG-v2") {
    return RuleProfile::ccz_121_strict_lg_v2();
  }
  if (text == "ab" || text == "anti-block") {
    return RuleProfile::ccz_121_ab_lg_v2();
  }
  if (text == "strict" || text == "strict-lg") {
    return RuleProfile::ccz_121_strict_lg_v2();
  }
  if (text == "ab-v1" || text == "anti-block-v1" || text == "CCZ-121-AB-LG-v1") {
    return RuleProfile::ccz_121_ab_lg_v1();
  }
  if (text == "strict-v1" || text == "strict-lg-v1" || text == "CCZ-121-Strict-LG-v1") {
    return RuleProfile::ccz_121_strict_lg_v1();
  }
  if (text == "mp3" || text == "multiplayer3" || text == "CCZ-121-MP3-v1") {
    return RuleProfile::ccz_121_mp3_v1();
  }
  if (text == "mp4" || text == "multiplayer4" || text == "CCZ-121-MP4-v1") {
    return RuleProfile::ccz_121_mp4_v1();
  }
  if (text == "mp6" || text == "multiplayer6" || text == "CCZ-121-MP6-v1") {
    return RuleProfile::ccz_121_mp6_v1();
  }
  throw std::runtime_error("unknown rule profile: " + text);
}

std::vector<MovegenBackend> parse_movegen_backend_list(const std::string& text) {
  if (text.empty() || text == "all") {
    return {MovegenBackend::Reference, MovegenBackend::Fast, MovegenBackend::Bitboard};
  }

  std::vector<MovegenBackend> backends;
  std::stringstream input(text);
  std::string item;
  while (std::getline(input, item, ',')) {
    item.erase(std::remove_if(item.begin(), item.end(), [](unsigned char ch) {
                 return std::isspace(ch) != 0;
               }),
               item.end());
    if (item.empty()) {
      continue;
    }
    backends.push_back(parse_movegen_backend(item));
  }
  if (backends.empty()) {
    throw std::runtime_error("movegen backend list is empty");
  }
  return backends;
}

}  // namespace cczero
