#include "cczero/model.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>

#ifdef CCZERO_USE_ACCELERATE
#include <Accelerate/Accelerate.h>
#endif

namespace cczero {

void encode_move_features(int player, int from, int to, float* features);

namespace {

constexpr int kBaseFeatureSize = 243;
constexpr int kGeometryFeatureSize = kBaseFeatureSize + kBoardSize * 10 + 12;
constexpr int kGeometryV2FeatureSize = kGeometryFeatureSize + kBoardSize * 9 + 8;
constexpr int kGeometryV3FeatureSize = kGeometryV2FeatureSize + kBoardSize * 4 + 18;
constexpr int kMoveFeatureSize = 20;

size_t vector_bytes(const std::vector<float>& values) {
  return values.size() * sizeof(float);
}

void require_size(const std::vector<float>& values, size_t expected, const std::string& name,
                  const std::string& label) {
  if (values.size() != expected) {
    throw std::runtime_error("invalid policy model section size for '" + name + "'" + label);
  }
}

void require_finite(const std::vector<float>& values, const std::string& name,
                    const std::string& label) {
  for (float value : values) {
    if (!std::isfinite(value)) {
      throw std::runtime_error("non-finite policy model value in '" + name + "'" + label);
    }
  }
}

bool is_supported_feature_size(int feature_size) {
  return feature_size == kBaseFeatureSize || feature_size == kGeometryFeatureSize ||
         feature_size == kGeometryV2FeatureSize || feature_size == kGeometryV3FeatureSize;
}

void add_feature_column(const PolicyModel& model, int feature, float scale, float* hidden) {
  if (scale == 0.0f) {
    return;
  }
  const size_t hidden_size = static_cast<size_t>(model.hidden_size);
  const float* column =
      model.input_w_feature_major.data() + static_cast<size_t>(feature) * hidden_size;
  for (size_t row = 0; row < hidden_size; ++row) {
    hidden[row] += scale * column[row];
  }
}

void build_geometry_static_hidden(PolicyModel& model) {
  if (model.kind != ModelKind::PolicyValueMlp || model.feature_size == kBaseFeatureSize) {
    return;
  }
  const Board& board = Board::standard();
  const size_t hidden_size = static_cast<size_t>(model.hidden_size);
  for (int player = 0; player < kPlayers; ++player) {
    const int opponent = 1 - player;
    std::vector<float>& static_hidden = model.geometry_static_hidden[static_cast<size_t>(player)];
    static_hidden = model.input_b;
    for (int id = 0; id < kBoardSize; ++id) {
      add_feature_column(model, kBaseFeatureSize + id,
                         static_cast<float>(board.coord(id).q) / 8.0f, static_hidden.data());
      add_feature_column(model, kBaseFeatureSize + kBoardSize + id,
                         static_cast<float>(board.coord(id).r) / 8.0f, static_hidden.data());
      add_feature_column(model, kBaseFeatureSize + kBoardSize * 2 + id,
                         static_cast<float>(board.coord(id).s()) / 8.0f, static_hidden.data());
      add_feature_column(model, kBaseFeatureSize + kBoardSize * 3 + id,
                         static_cast<float>(board.goal_distance(player, id)) / 16.0f,
                         static_hidden.data());
      add_feature_column(model, kBaseFeatureSize + kBoardSize * 4 + id,
                         static_cast<float>(board.goal_distance(opponent, id)) / 16.0f,
                         static_hidden.data());
      add_feature_column(model, kBaseFeatureSize + kBoardSize * 5 + id,
                         board.is_home(player, id) ? 1.0f : 0.0f, static_hidden.data());
      add_feature_column(model, kBaseFeatureSize + kBoardSize * 6 + id,
                         board.is_goal(player, id) ? 1.0f : 0.0f, static_hidden.data());
      add_feature_column(model, kBaseFeatureSize + kBoardSize * 7 + id,
                         board.is_home(opponent, id) ? 1.0f : 0.0f, static_hidden.data());
      add_feature_column(model, kBaseFeatureSize + kBoardSize * 8 + id,
                         board.is_goal(opponent, id) ? 1.0f : 0.0f, static_hidden.data());
      add_feature_column(model, kBaseFeatureSize + kBoardSize * 9 + id,
                         board.is_side_triangle(id) ? 1.0f : 0.0f, static_hidden.data());
    }
    if (model.feature_size == kGeometryV2FeatureSize ||
        model.feature_size == kGeometryV3FeatureSize) {
      const int center_offset = kGeometryFeatureSize;
      const int forward_offset = center_offset + kBoardSize;
      for (int id = 0; id < kBoardSize; ++id) {
        const Coord origin{0, 0};
        add_feature_column(model, center_offset + id,
                           static_cast<float>(hex_distance(board.coord(id), origin)) / 8.0f,
                           static_hidden.data());
        const float forward =
            player == 0 ? static_cast<float>(board.coord(id).r + 8) / 16.0f
                        : static_cast<float>(8 - board.coord(id).r) / 16.0f;
        add_feature_column(model, forward_offset + id, forward, static_hidden.data());
      }
    }
    if (static_hidden.size() != hidden_size) {
      throw std::runtime_error("internal geometry static hidden size mismatch");
    }
  }
}

void build_move_action_hidden(PolicyModel& model) {
  if (model.kind != ModelKind::PolicyValueMlp ||
      (model.policy_head != PolicyHeadKind::MoveMlp &&
       model.policy_head != PolicyHeadKind::MoveBilinear)) {
    return;
  }
  const size_t action_size = static_cast<size_t>(model.action_size);
  const size_t move_embed = static_cast<size_t>(model.move_embed_size);
  const size_t move_hidden = static_cast<size_t>(model.move_hidden_size);
  const size_t move_feature = static_cast<size_t>(model.move_feature_size);
  const size_t move_input = 2 * move_embed + move_feature;
  model.move_action_hidden.assign(2ULL * action_size * move_hidden, 0.0f);
  if (model.policy_head == PolicyHeadKind::MoveBilinear) {
    model.move_action_bias.assign(2ULL * action_size, 0.0f);
  }
  std::array<float, kMoveFeatureSize> features{};
  std::vector<float> input_values(move_input, 0.0f);
  for (int player = 0; player < kPlayers; ++player) {
    for (int action = 0; action < model.action_size; ++action) {
      const int from = action / kBoardSize;
      const int to = action % kBoardSize;
      encode_move_features(player, from, to, features.data());
      const float* from_embed =
          model.move_from_embed.data() + static_cast<size_t>(from) * move_embed;
      const float* to_embed =
          model.move_to_embed.data() + static_cast<size_t>(to) * move_embed;
      float* cached = model.move_action_hidden.data() +
                      (static_cast<size_t>(player) * action_size +
                       static_cast<size_t>(action)) *
                          move_hidden;
      size_t input_offset = 0;
      for (size_t col = 0; col < move_embed; ++col) {
        input_values[input_offset + col] = from_embed[col];
      }
      input_offset += move_embed;
      for (size_t col = 0; col < move_embed; ++col) {
        input_values[input_offset + col] = to_embed[col];
      }
      input_offset += move_embed;
      for (size_t col = 0; col < move_feature; ++col) {
        input_values[input_offset + col] = features[col];
      }
      for (size_t row = 0; row < move_hidden; ++row) {
        const float* move_weights = model.move_w.data() + row * move_input;
        float activation =
            model.policy_head == PolicyHeadKind::MoveBilinear ? model.move_b[row] : 0.0f;
        for (size_t col = 0; col < move_input; ++col) {
          activation += move_weights[col] * input_values[col];
        }
        cached[row] = activation;
      }
      if (model.policy_head == PolicyHeadKind::MoveBilinear) {
        float bias = model.move_bias_b;
        for (size_t col = 0; col < move_input; ++col) {
          bias += model.move_bias_w[col] * input_values[col];
        }
        model.move_action_bias[static_cast<size_t>(player) * action_size +
                               static_cast<size_t>(action)] = bias;
      }
    }
  }
}

}  // namespace

void MlpWorkspace::ensure(size_t batch, size_t hidden_size) {
  const size_t count = batch * hidden_size;
  if (hidden.size() < count) {
    hidden.resize(count);
  }
  if (tmp.size() < count) {
    tmp.resize(count);
  }
  if (residual.size() < count) {
    residual.resize(count);
  }
  if (values.size() < batch) {
    values.resize(batch);
  }
}

void MlpWorkspace::ensure_policy_state(size_t move_hidden_size) {
  if (policy_state.size() < move_hidden_size) {
    policy_state.resize(move_hidden_size);
  }
}

PolicyModel load_policy_model(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("failed to open policy model: " + path);
  }
  char magic[16] = {};
  input.read(magic, sizeof(magic));
  if (!input) {
    throw std::runtime_error("invalid policy model magic: " + path);
  }

  PolicyModel model;
  if (std::memcmp(magic, "CCZPOLICYv1", 11) == 0) {
    model.kind = ModelKind::LinearPolicy;
    input.read(reinterpret_cast<char*>(&model.feature_size), sizeof(model.feature_size));
    input.read(reinterpret_cast<char*>(&model.action_size), sizeof(model.action_size));
    if (!input || model.feature_size != kBaseFeatureSize || model.action_size != 121 * 121) {
      throw std::runtime_error("unsupported policy model dimensions: " + path);
    }
    model.w.resize(static_cast<size_t>(model.feature_size) * model.action_size);
    model.b.resize(static_cast<size_t>(model.action_size));
    input.read(reinterpret_cast<char*>(model.w.data()),
               static_cast<std::streamsize>(model.w.size() * sizeof(float)));
    input.read(reinterpret_cast<char*>(model.b.data()),
               static_cast<std::streamsize>(model.b.size() * sizeof(float)));
    if (!input) {
      throw std::runtime_error("truncated policy model: " + path);
    }
    validate_policy_model(model, path);
    return model;
  }

  if (std::memcmp(magic, "CCZPVMLPv1", 10) != 0) {
    if (std::memcmp(magic, "CCZPVMLPv2", 10) != 0) {
      throw std::runtime_error("invalid policy model magic: " + path);
    }
  }
  model.kind = ModelKind::PolicyValueMlp;
  const bool is_v2 = std::memcmp(magic, "CCZPVMLPv2", 10) == 0;
  input.read(reinterpret_cast<char*>(&model.feature_size), sizeof(model.feature_size));
  input.read(reinterpret_cast<char*>(&model.action_size), sizeof(model.action_size));
  input.read(reinterpret_cast<char*>(&model.hidden_size), sizeof(model.hidden_size));
  input.read(reinterpret_cast<char*>(&model.blocks), sizeof(model.blocks));
  if (is_v2) {
    int policy_head = 0;
    input.read(reinterpret_cast<char*>(&policy_head), sizeof(policy_head));
    input.read(reinterpret_cast<char*>(&model.move_embed_size), sizeof(model.move_embed_size));
    input.read(reinterpret_cast<char*>(&model.move_hidden_size), sizeof(model.move_hidden_size));
    input.read(reinterpret_cast<char*>(&model.move_feature_size), sizeof(model.move_feature_size));
    if (policy_head == 1) {
      model.policy_head = PolicyHeadKind::MoveMlp;
    } else if (policy_head == 2) {
      model.policy_head = PolicyHeadKind::MoveBilinear;
    } else {
      throw std::runtime_error("unsupported policy/value model policy head: " + path);
    }
  } else {
    model.policy_head = PolicyHeadKind::Dense;
  }
  if (!input || !is_supported_feature_size(model.feature_size) ||
      model.action_size != 121 * 121 ||
      model.hidden_size <= 0 || model.blocks < 0 ||
      (model.policy_head != PolicyHeadKind::Dense &&
       (model.move_embed_size <= 0 || model.move_hidden_size <= 0 ||
        model.move_feature_size != kMoveFeatureSize))) {
    throw std::runtime_error("unsupported policy/value model dimensions: " + path);
  }

  auto read_floats = [&](std::vector<float>& values, size_t count, const std::string& name) {
    values.resize(count);
    input.read(reinterpret_cast<char*>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(float)));
    if (!input) {
      throw std::runtime_error("truncated policy/value model section '" + name + "': " + path);
    }
  };

  const size_t feature_size = static_cast<size_t>(model.feature_size);
  const size_t hidden_size = static_cast<size_t>(model.hidden_size);
  const size_t blocks = static_cast<size_t>(model.blocks);
  const size_t action_size = static_cast<size_t>(model.action_size);
  read_floats(model.input_w, hidden_size * feature_size, "input_w");
  read_floats(model.input_b, hidden_size, "input_b");
  read_floats(model.block_w1, blocks * hidden_size * hidden_size, "block_w1");
  read_floats(model.block_b1, blocks * hidden_size, "block_b1");
  read_floats(model.block_w2, blocks * hidden_size * hidden_size, "block_w2");
  read_floats(model.block_b2, blocks * hidden_size, "block_b2");
  if (model.policy_head == PolicyHeadKind::Dense) {
    read_floats(model.policy_w, action_size * hidden_size, "policy_w");
    read_floats(model.policy_b, action_size, "policy_b");
  } else {
    const size_t move_embed = static_cast<size_t>(model.move_embed_size);
    const size_t move_hidden = static_cast<size_t>(model.move_hidden_size);
    const size_t move_feature = static_cast<size_t>(model.move_feature_size);
    read_floats(model.move_from_embed, kBoardSize * move_embed, "move_from_embed");
    read_floats(model.move_to_embed, kBoardSize * move_embed, "move_to_embed");
    read_floats(model.move_w, move_hidden * (2 * move_embed + move_feature), "move_w");
    read_floats(model.move_state_w, move_hidden * hidden_size, "move_state_w");
    read_floats(model.move_b, move_hidden, "move_b");
    if (model.policy_head == PolicyHeadKind::MoveMlp) {
      read_floats(model.move_out_w, move_hidden, "move_out_w");
      input.read(reinterpret_cast<char*>(&model.move_out_b), sizeof(model.move_out_b));
      if (!input) {
        throw std::runtime_error("truncated policy/value model section 'move_out_b': " + path);
      }
    } else {
      read_floats(model.move_bias_w, 2 * move_embed + move_feature, "move_bias_w");
      input.read(reinterpret_cast<char*>(&model.move_bias_b), sizeof(model.move_bias_b));
      if (!input) {
        throw std::runtime_error("truncated policy/value model section 'move_bias_b': " + path);
      }
    }
  }
  read_floats(model.value_w, hidden_size, "value_w");
  input.read(reinterpret_cast<char*>(&model.value_b), sizeof(model.value_b));
  if (!input) {
    throw std::runtime_error("truncated policy/value model section 'value_b': " + path);
  }
  model.input_w_feature_major.resize(feature_size * hidden_size);
  for (size_t row = 0; row < hidden_size; ++row) {
    for (size_t col = 0; col < feature_size; ++col) {
      model.input_w_feature_major.at(col * hidden_size + row) =
          model.input_w.at(row * feature_size + col);
    }
  }
  build_geometry_static_hidden(model);
  build_move_action_hidden(model);
  validate_policy_model(model, path);
  return model;
}

size_t policy_model_parameter_count(const PolicyModel& model) {
  return model.w.size() + model.b.size() + model.input_w.size() + model.input_b.size() +
         model.block_w1.size() + model.block_b1.size() + model.block_w2.size() +
         model.block_b2.size() + model.policy_w.size() + model.policy_b.size() +
         model.move_from_embed.size() + model.move_to_embed.size() + model.move_w.size() +
         model.move_state_w.size() + model.move_b.size() + model.move_out_w.size() +
         (model.policy_head == PolicyHeadKind::MoveMlp ? 1ULL : 0ULL) +
         model.move_bias_w.size() +
         (model.policy_head == PolicyHeadKind::MoveBilinear ? 1ULL : 0ULL) +
         model.value_w.size() + (model.kind == ModelKind::PolicyValueMlp ? 1ULL : 0ULL);
}

size_t policy_model_storage_bytes(const PolicyModel& model) {
  const size_t scalar_bytes =
      sizeof(model.value_b) +
      (model.policy_head == PolicyHeadKind::MoveMlp ? sizeof(model.move_out_b) : 0ULL) +
      (model.policy_head == PolicyHeadKind::MoveBilinear ? sizeof(model.move_bias_b) : 0ULL);
  return vector_bytes(model.w) + vector_bytes(model.b) + vector_bytes(model.input_w) +
         vector_bytes(model.input_b) + vector_bytes(model.block_w1) +
         vector_bytes(model.block_b1) + vector_bytes(model.block_w2) +
         vector_bytes(model.block_b2) + vector_bytes(model.policy_w) +
         vector_bytes(model.policy_b) + vector_bytes(model.move_from_embed) +
         vector_bytes(model.move_to_embed) + vector_bytes(model.move_w) +
         vector_bytes(model.move_state_w) + vector_bytes(model.move_b) +
         vector_bytes(model.move_out_w) + vector_bytes(model.move_action_hidden) +
         vector_bytes(model.move_bias_w) + vector_bytes(model.move_action_bias) +
         vector_bytes(model.value_w) +
         vector_bytes(model.input_w_feature_major) +
         vector_bytes(model.geometry_static_hidden[0]) +
         vector_bytes(model.geometry_static_hidden[1]) + scalar_bytes;
}

void validate_policy_model(const PolicyModel& model, const std::string& label) {
  const std::string suffix = label.empty() ? "" : ": " + label;
  if (!is_supported_feature_size(model.feature_size) || model.action_size != kBoardSize * kBoardSize) {
    throw std::runtime_error("unsupported policy model dimensions" + suffix);
  }
  if (model.kind == ModelKind::LinearPolicy) {
    require_size(model.w, static_cast<size_t>(model.feature_size) * model.action_size, "w",
                 suffix);
    require_size(model.b, static_cast<size_t>(model.action_size), "b", suffix);
    require_finite(model.w, "w", suffix);
    require_finite(model.b, "b", suffix);
    return;
  }

  if (model.kind != ModelKind::PolicyValueMlp) {
    throw std::runtime_error("unknown policy model kind" + suffix);
  }
  if (model.hidden_size <= 0 || model.blocks < 0) {
    throw std::runtime_error("unsupported policy/value model dimensions" + suffix);
  }
  const size_t feature_size = static_cast<size_t>(model.feature_size);
  const size_t hidden_size = static_cast<size_t>(model.hidden_size);
  const size_t blocks = static_cast<size_t>(model.blocks);
  const size_t action_size = static_cast<size_t>(model.action_size);
  require_size(model.input_w, hidden_size * feature_size, "input_w", suffix);
  require_size(model.input_b, hidden_size, "input_b", suffix);
  require_size(model.block_w1, blocks * hidden_size * hidden_size, "block_w1", suffix);
  require_size(model.block_b1, blocks * hidden_size, "block_b1", suffix);
  require_size(model.block_w2, blocks * hidden_size * hidden_size, "block_w2", suffix);
  require_size(model.block_b2, blocks * hidden_size, "block_b2", suffix);
  if (model.policy_head == PolicyHeadKind::Dense) {
    require_size(model.policy_w, action_size * hidden_size, "policy_w", suffix);
    require_size(model.policy_b, action_size, "policy_b", suffix);
  } else if (model.policy_head == PolicyHeadKind::MoveMlp ||
             model.policy_head == PolicyHeadKind::MoveBilinear) {
    if (model.move_embed_size <= 0 || model.move_hidden_size <= 0 ||
        model.move_feature_size != kMoveFeatureSize) {
      throw std::runtime_error("unsupported move policy head dimensions" + suffix);
    }
    const size_t move_embed = static_cast<size_t>(model.move_embed_size);
    const size_t move_hidden = static_cast<size_t>(model.move_hidden_size);
    const size_t move_feature = static_cast<size_t>(model.move_feature_size);
    require_size(model.move_from_embed, kBoardSize * move_embed, "move_from_embed", suffix);
    require_size(model.move_to_embed, kBoardSize * move_embed, "move_to_embed", suffix);
    require_size(model.move_w, move_hidden * (2 * move_embed + move_feature), "move_w", suffix);
    require_size(model.move_state_w, move_hidden * hidden_size, "move_state_w", suffix);
    require_size(model.move_b, move_hidden, "move_b", suffix);
    if (model.policy_head == PolicyHeadKind::MoveMlp) {
      require_size(model.move_out_w, move_hidden, "move_out_w", suffix);
    } else {
      require_size(model.move_bias_w, 2 * move_embed + move_feature, "move_bias_w", suffix);
      require_size(model.move_action_bias, 2ULL * action_size, "move_action_bias", suffix);
    }
    require_size(model.move_action_hidden, 2ULL * action_size * move_hidden,
                 "move_action_hidden", suffix);
  } else {
    throw std::runtime_error("unknown policy head kind" + suffix);
  }
  require_size(model.value_w, hidden_size, "value_w", suffix);
  require_size(model.input_w_feature_major, feature_size * hidden_size,
               "input_w_feature_major", suffix);
  if (model.feature_size != kBaseFeatureSize) {
    require_size(model.geometry_static_hidden[0], hidden_size, "geometry_static_hidden[0]",
                 suffix);
    require_size(model.geometry_static_hidden[1], hidden_size, "geometry_static_hidden[1]",
                 suffix);
  }
  require_finite(model.input_w, "input_w", suffix);
  require_finite(model.input_b, "input_b", suffix);
  require_finite(model.block_w1, "block_w1", suffix);
  require_finite(model.block_b1, "block_b1", suffix);
  require_finite(model.block_w2, "block_w2", suffix);
  require_finite(model.block_b2, "block_b2", suffix);
  require_finite(model.policy_w, "policy_w", suffix);
  require_finite(model.policy_b, "policy_b", suffix);
  require_finite(model.move_from_embed, "move_from_embed", suffix);
  require_finite(model.move_to_embed, "move_to_embed", suffix);
  require_finite(model.move_w, "move_w", suffix);
  require_finite(model.move_state_w, "move_state_w", suffix);
  require_finite(model.move_b, "move_b", suffix);
  require_finite(model.move_out_w, "move_out_w", suffix);
  require_finite(model.move_bias_w, "move_bias_w", suffix);
  require_finite(model.move_action_hidden, "move_action_hidden", suffix);
  require_finite(model.move_action_bias, "move_action_bias", suffix);
  require_finite(model.value_w, "value_w", suffix);
  if (model.feature_size != kBaseFeatureSize) {
    require_finite(model.geometry_static_hidden[0], "geometry_static_hidden[0]", suffix);
    require_finite(model.geometry_static_hidden[1], "geometry_static_hidden[1]", suffix);
  }
  if (!std::isfinite(model.move_out_b)) {
    throw std::runtime_error("non-finite policy model value in 'move_out_b'" + suffix);
  }
  if (!std::isfinite(model.move_bias_b)) {
    throw std::runtime_error("non-finite policy model value in 'move_bias_b'" + suffix);
  }
  if (!std::isfinite(model.value_b)) {
    throw std::runtime_error("non-finite policy model value in 'value_b'" + suffix);
  }
}

std::vector<float> encode_policy_features(const State& state, int player, int feature_size) {
  if (!is_supported_feature_size(feature_size)) {
    throw std::runtime_error("unsupported policy feature size");
  }
  const Board& board = Board::standard();
  const int opponent = 1 - player;
  std::vector<float> features(kBaseFeatureSize, 0.0f);
  for (int id = 0; id < kBoardSize; ++id) {
    const int occupant = state.cells.at(static_cast<size_t>(id));
    if (occupant == player) {
      features.at(static_cast<size_t>(id)) = 1.0f;
    } else if (occupant == 1 - player) {
      features.at(static_cast<size_t>(kBoardSize + id)) = 1.0f;
    }
  }
  features.at(242) = player == 0 ? 1.0f : -1.0f;
  if (feature_size == kBaseFeatureSize) {
    return features;
  }

  features.reserve(static_cast<size_t>(kGeometryFeatureSize));
  for (int id = 0; id < kBoardSize; ++id) {
    features.push_back(static_cast<float>(board.coord(id).q) / 8.0f);
  }
  for (int id = 0; id < kBoardSize; ++id) {
    features.push_back(static_cast<float>(board.coord(id).r) / 8.0f);
  }
  for (int id = 0; id < kBoardSize; ++id) {
    features.push_back(static_cast<float>(board.coord(id).s()) / 8.0f);
  }
  for (int id = 0; id < kBoardSize; ++id) {
    features.push_back(static_cast<float>(board.goal_distance(player, id)) / 16.0f);
  }
  for (int id = 0; id < kBoardSize; ++id) {
    features.push_back(static_cast<float>(board.goal_distance(opponent, id)) / 16.0f);
  }
  for (int id = 0; id < kBoardSize; ++id) {
    features.push_back(board.is_home(player, id) ? 1.0f : 0.0f);
  }
  for (int id = 0; id < kBoardSize; ++id) {
    features.push_back(board.is_goal(player, id) ? 1.0f : 0.0f);
  }
  for (int id = 0; id < kBoardSize; ++id) {
    features.push_back(board.is_home(opponent, id) ? 1.0f : 0.0f);
  }
  for (int id = 0; id < kBoardSize; ++id) {
    features.push_back(board.is_goal(opponent, id) ? 1.0f : 0.0f);
  }
  for (int id = 0; id < kBoardSize; ++id) {
    features.push_back(board.is_side_triangle(id) ? 1.0f : 0.0f);
  }

  float own_goal = 0.0f;
  float opp_goal = 0.0f;
  float own_home = 0.0f;
  float opp_home = 0.0f;
  float own_distance = 0.0f;
  float opp_distance = 0.0f;
  float own_blockers = 0.0f;
  float opp_blockers = 0.0f;
  for (int id = 0; id < kBoardSize; ++id) {
    const int occupant = state.cells.at(static_cast<size_t>(id));
    if (occupant == player) {
      own_distance += static_cast<float>(board.goal_distance(player, id));
      if (board.is_goal(player, id)) {
        own_goal += 1.0f;
      }
      if (board.is_home(player, id)) {
        own_home += 1.0f;
      }
      if (board.is_goal(opponent, id)) {
        opp_blockers += 1.0f;
      }
    } else if (occupant == opponent) {
      opp_distance += static_cast<float>(board.goal_distance(opponent, id));
      if (board.is_goal(opponent, id)) {
        opp_goal += 1.0f;
      }
      if (board.is_home(opponent, id)) {
        opp_home += 1.0f;
      }
      if (board.is_goal(player, id)) {
        own_blockers += 1.0f;
      }
    }
  }
  const float ply = static_cast<float>(std::min(state.ply, 240));
  const float phase = state.ply < 40 ? 0.0f : (state.ply < 90 ? 0.5f : 1.0f);
  features.push_back(own_goal / 10.0f);
  features.push_back(opp_goal / 10.0f);
  features.push_back(own_home / 10.0f);
  features.push_back(opp_home / 10.0f);
  features.push_back(own_distance / 110.0f);
  features.push_back(opp_distance / 110.0f);
  features.push_back((opp_distance - own_distance) / 110.0f);
  features.push_back(own_blockers / 10.0f);
  features.push_back(opp_blockers / 10.0f);
  features.push_back(ply / 240.0f);
  features.push_back(phase);
  features.push_back(player == 0 ? 1.0f : -1.0f);
  if (feature_size == kGeometryV2FeatureSize || feature_size == kGeometryV3FeatureSize) {
    for (int id = 0; id < kBoardSize; ++id) {
      features.push_back(static_cast<float>(hex_distance(board.coord(id), Coord{0, 0})) / 8.0f);
    }
    for (int id = 0; id < kBoardSize; ++id) {
      const float forward =
          player == 0 ? static_cast<float>(board.coord(id).r + 8) / 16.0f
                      : static_cast<float>(8 - board.coord(id).r) / 16.0f;
      features.push_back(forward);
    }

    std::array<float, kBoardSize> own_neighbor{};
    std::array<float, kBoardSize> opp_neighbor{};
    std::array<float, kBoardSize> empty_neighbor{};
    std::array<float, kBoardSize> own_step_origin{};
    std::array<float, kBoardSize> opp_step_origin{};
    std::array<float, kBoardSize> own_jump_origin{};
    std::array<float, kBoardSize> opp_jump_origin{};
    int own_step_total = 0;
    int opp_step_total = 0;
    int own_jump_total = 0;
    int opp_jump_total = 0;
    for (int id = 0; id < kBoardSize; ++id) {
      int own_n = 0;
      int opp_n = 0;
      int empty_n = 0;
      int own_steps = 0;
      int opp_steps = 0;
      int own_jumps = 0;
      int opp_jumps = 0;
      for (int dir = 0; dir < 6; ++dir) {
        const int neighbor = board.neighbors(id)[static_cast<size_t>(dir)];
        if (neighbor != kInvalid) {
          const int occupant = state.cells[static_cast<size_t>(neighbor)];
          if (occupant == player) {
            ++own_n;
          } else if (occupant == opponent) {
            ++opp_n;
          } else {
            ++empty_n;
            if (state.cells[static_cast<size_t>(id)] == player) {
              ++own_steps;
            } else if (state.cells[static_cast<size_t>(id)] == opponent) {
              ++opp_steps;
            }
          }
        }
        const int mid = board.jump_mid(id, dir);
        const int landing = board.jump_landing(id, dir);
        if (mid != kInvalid && landing != kInvalid && state.cells[static_cast<size_t>(mid)] != kEmpty &&
            state.cells[static_cast<size_t>(landing)] == kEmpty) {
          if (state.cells[static_cast<size_t>(id)] == player) {
            ++own_jumps;
          } else if (state.cells[static_cast<size_t>(id)] == opponent) {
            ++opp_jumps;
          }
        }
      }
      own_neighbor[static_cast<size_t>(id)] = static_cast<float>(own_n) / 6.0f;
      opp_neighbor[static_cast<size_t>(id)] = static_cast<float>(opp_n) / 6.0f;
      empty_neighbor[static_cast<size_t>(id)] = static_cast<float>(empty_n) / 6.0f;
      own_step_origin[static_cast<size_t>(id)] = static_cast<float>(own_steps) / 6.0f;
      opp_step_origin[static_cast<size_t>(id)] = static_cast<float>(opp_steps) / 6.0f;
      own_jump_origin[static_cast<size_t>(id)] = static_cast<float>(own_jumps) / 6.0f;
      opp_jump_origin[static_cast<size_t>(id)] = static_cast<float>(opp_jumps) / 6.0f;
      own_step_total += own_steps;
      opp_step_total += opp_steps;
      own_jump_total += own_jumps;
      opp_jump_total += opp_jumps;
    }
    for (float value : own_neighbor) features.push_back(value);
    for (float value : opp_neighbor) features.push_back(value);
    for (float value : empty_neighbor) features.push_back(value);
    for (float value : own_step_origin) features.push_back(value);
    for (float value : opp_step_origin) features.push_back(value);
    for (float value : own_jump_origin) features.push_back(value);
    for (float value : opp_jump_origin) features.push_back(value);
    int own_movable = 0;
    int opp_movable = 0;
    for (int id = 0; id < kBoardSize; ++id) {
      if (state.cells[static_cast<size_t>(id)] == player &&
          (own_step_origin[static_cast<size_t>(id)] > 0.0f ||
           own_jump_origin[static_cast<size_t>(id)] > 0.0f)) {
        ++own_movable;
      } else if (state.cells[static_cast<size_t>(id)] == opponent &&
                 (opp_step_origin[static_cast<size_t>(id)] > 0.0f ||
                  opp_jump_origin[static_cast<size_t>(id)] > 0.0f)) {
        ++opp_movable;
      }
    }
    const int own_mobility = own_step_total + 2 * own_jump_total;
    const int opp_mobility = opp_step_total + 2 * opp_jump_total;
    features.push_back(static_cast<float>(own_movable) / 10.0f);
    features.push_back(static_cast<float>(opp_movable) / 10.0f);
    features.push_back(static_cast<float>(std::min(own_step_total, 60)) / 60.0f);
    features.push_back(static_cast<float>(std::min(opp_step_total, 60)) / 60.0f);
    features.push_back(static_cast<float>(std::min(own_jump_total, 60)) / 60.0f);
    features.push_back(static_cast<float>(std::min(opp_jump_total, 60)) / 60.0f);
    features.push_back(0.0f);
    features.push_back(std::clamp(static_cast<float>(own_mobility - opp_mobility) / 120.0f,
                                  -1.0f, 1.0f));
    if (feature_size == kGeometryV3FeatureSize) {
      std::array<float, kBoardSize> own_goal_pressure{};
      std::array<float, kBoardSize> opp_goal_pressure{};
      std::array<float, kBoardSize> own_back_rank{};
      std::array<float, kBoardSize> opp_back_rank{};
      int own_goal_empty = 0;
      int opp_goal_empty = 0;
      int own_home_stragglers = 0;
      int opp_home_stragglers = 0;
      int own_isolated = 0;
      int opp_isolated = 0;
      int own_backward = 0;
      int opp_backward = 0;
      float own_forward_sum = 0.0f;
      float opp_forward_sum = 0.0f;
      for (int id = 0; id < kBoardSize; ++id) {
        const int occupant = state.cells[static_cast<size_t>(id)];
        const bool own_piece = occupant == player;
        const bool opp_piece = occupant == opponent;
        const float forward =
            player == 0 ? static_cast<float>(board.coord(id).r + 8) / 16.0f
                        : static_cast<float>(8 - board.coord(id).r) / 16.0f;
        own_goal_pressure[static_cast<size_t>(id)] =
            board.is_goal(player, id) ? (own_piece ? 1.0f : (opp_piece ? -1.0f : 0.0f)) : 0.0f;
        opp_goal_pressure[static_cast<size_t>(id)] =
            board.is_goal(opponent, id) ? (opp_piece ? 1.0f : (own_piece ? -1.0f : 0.0f)) : 0.0f;
        own_back_rank[static_cast<size_t>(id)] =
            own_piece ? static_cast<float>(board.goal_distance(player, id)) / 16.0f : 0.0f;
        opp_back_rank[static_cast<size_t>(id)] =
            opp_piece ? static_cast<float>(board.goal_distance(opponent, id)) / 16.0f : 0.0f;
        own_forward_sum += own_piece ? forward : 0.0f;
        opp_forward_sum += opp_piece ? (1.0f - forward) : 0.0f;
        if (board.is_goal(player, id) && occupant == kEmpty) {
          ++own_goal_empty;
        }
        if (board.is_goal(opponent, id) && occupant == kEmpty) {
          ++opp_goal_empty;
        }
        if (board.is_home(player, id) && own_piece) {
          ++own_home_stragglers;
        }
        if (board.is_home(opponent, id) && opp_piece) {
          ++opp_home_stragglers;
        }
        if (own_piece && own_neighbor[static_cast<size_t>(id)] <= 0.0f &&
            own_step_origin[static_cast<size_t>(id)] <= 0.0f &&
            own_jump_origin[static_cast<size_t>(id)] <= 0.0f) {
          ++own_isolated;
        }
        if (opp_piece && opp_neighbor[static_cast<size_t>(id)] <= 0.0f &&
            opp_step_origin[static_cast<size_t>(id)] <= 0.0f &&
            opp_jump_origin[static_cast<size_t>(id)] <= 0.0f) {
          ++opp_isolated;
        }
        if (own_back_rank[static_cast<size_t>(id)] >= 0.625f) {
          ++own_backward;
        }
        if (opp_back_rank[static_cast<size_t>(id)] >= 0.625f) {
          ++opp_backward;
        }
      }
      for (float value : own_goal_pressure) features.push_back(value);
      for (float value : opp_goal_pressure) features.push_back(value);
      for (float value : own_back_rank) features.push_back(value);
      for (float value : opp_back_rank) features.push_back(value);
      const float own_fill = own_goal / std::max(1.0f, own_goal + static_cast<float>(own_goal_empty));
      const float opp_fill = opp_goal / std::max(1.0f, opp_goal + static_cast<float>(opp_goal_empty));
      const float v3_scalars[] = {
          static_cast<float>(own_goal_empty) / 10.0f,
          static_cast<float>(opp_goal_empty) / 10.0f,
          own_fill,
          opp_fill,
          own_fill - opp_fill,
          static_cast<float>(own_home_stragglers) / 10.0f,
          static_cast<float>(opp_home_stragglers) / 10.0f,
          static_cast<float>(opp_home_stragglers - own_home_stragglers) / 10.0f,
          static_cast<float>(own_isolated) / 10.0f,
          static_cast<float>(opp_isolated) / 10.0f,
          static_cast<float>(opp_isolated - own_isolated) / 10.0f,
          static_cast<float>(own_backward) / 10.0f,
          static_cast<float>(opp_backward) / 10.0f,
          static_cast<float>(opp_backward - own_backward) / 10.0f,
          own_forward_sum / 10.0f,
          opp_forward_sum / 10.0f,
          (own_forward_sum - opp_forward_sum) / 10.0f,
          phase * ((own_fill - opp_fill) +
                   static_cast<float>(opp_home_stragglers - own_home_stragglers) / 10.0f),
      };
      for (float value : v3_scalars) {
        features.push_back(value);
      }
    }
  }
  if (features.size() != static_cast<size_t>(feature_size)) {
    throw std::runtime_error("internal policy feature encoder size mismatch");
  }
  return features;
}

std::vector<float> mlp_hidden(const PolicyModel& model, const std::vector<float>& features) {
  const size_t feature_size = static_cast<size_t>(model.feature_size);
  const size_t hidden_size = static_cast<size_t>(model.hidden_size);
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
    const size_t matrix_offset = static_cast<size_t>(block) * hidden_size * hidden_size;
    const size_t bias_offset = static_cast<size_t>(block) * hidden_size;
    for (size_t row = 0; row < hidden_size; ++row) {
      float sum = model.block_b1.at(bias_offset + row);
      const size_t offset = matrix_offset + row * hidden_size;
      for (size_t col = 0; col < hidden_size; ++col) {
        sum += model.block_w1.at(offset + col) * hidden.at(col);
      }
      tmp.at(row) = std::max(0.0f, sum);
    }
    for (size_t row = 0; row < hidden_size; ++row) {
      float sum = model.block_b2.at(bias_offset + row);
      const size_t offset = matrix_offset + row * hidden_size;
      for (size_t col = 0; col < hidden_size; ++col) {
        sum += model.block_w2.at(offset + col) * tmp.at(col);
      }
      residual.at(row) = std::max(0.0f, hidden.at(row) + sum);
    }
    hidden.swap(residual);
  }
  return hidden;
}

void encode_move_features(int player, int from, int to, float* features) {
  const Board& board = Board::standard();
  const int opponent = 1 - player;
  const Coord& from_coord = board.coord(from);
  const Coord& to_coord = board.coord(to);
  const int from_s = from_coord.s();
  const int to_s = to_coord.s();
  features[0] = static_cast<float>(from_coord.q) / 8.0f;
  features[1] = static_cast<float>(from_coord.r) / 8.0f;
  features[2] = static_cast<float>(from_s) / 8.0f;
  features[3] = static_cast<float>(to_coord.q) / 8.0f;
  features[4] = static_cast<float>(to_coord.r) / 8.0f;
  features[5] = static_cast<float>(to_s) / 8.0f;
  features[6] = static_cast<float>(to_coord.q - from_coord.q) / 16.0f;
  features[7] = static_cast<float>(to_coord.r - from_coord.r) / 16.0f;
  features[8] = static_cast<float>(to_s - from_s) / 16.0f;
  features[9] = static_cast<float>(board.goal_distance(player, from)) / 16.0f;
  features[10] = static_cast<float>(board.goal_distance(player, to)) / 16.0f;
  features[11] = static_cast<float>(board.goal_distance(player, from) -
                                     board.goal_distance(player, to)) /
                 16.0f;
  features[12] = static_cast<float>(board.goal_distance(opponent, from)) / 16.0f;
  features[13] = static_cast<float>(board.goal_distance(opponent, to)) / 16.0f;
  features[14] = board.is_home(player, from) ? 1.0f : 0.0f;
  features[15] = board.is_home(player, to) ? 1.0f : 0.0f;
  features[16] = board.is_goal(player, from) ? 1.0f : 0.0f;
  features[17] = board.is_goal(player, to) ? 1.0f : 0.0f;
  features[18] = board.is_side_triangle(from) ? 1.0f : 0.0f;
  features[19] = board.is_side_triangle(to) ? 1.0f : 0.0f;
}

float mlp_dense_policy_logit_action_ptr(const PolicyModel& model, const float* hidden,
                                        int action) {
  float score = model.policy_b.at(static_cast<size_t>(action));
  const size_t hidden_size = static_cast<size_t>(model.hidden_size);
  const size_t offset = static_cast<size_t>(action) * hidden_size;
  for (size_t i = 0; i < hidden_size; ++i) {
    score += model.policy_w.at(offset + i) * hidden[i];
  }
  return score;
}

void mlp_policy_state_projection(const PolicyModel& model, const float* hidden, float* output) {
  if (model.policy_head != PolicyHeadKind::MoveMlp &&
      model.policy_head != PolicyHeadKind::MoveBilinear) {
    return;
  }
  const size_t hidden_size = static_cast<size_t>(model.hidden_size);
  const size_t move_hidden = static_cast<size_t>(model.move_hidden_size);
  for (size_t row = 0; row < move_hidden; ++row) {
    float activation = model.policy_head == PolicyHeadKind::MoveMlp ? model.move_b[row] : 0.0f;
    const float* state_weights = model.move_state_w.data() + row * hidden_size;
    for (size_t col = 0; col < hidden_size; ++col) {
      activation += state_weights[col] * hidden[col];
    }
    output[row] = activation;
  }
}

float mlp_move_policy_logit_action_projected_ptr(const PolicyModel& model,
                                                 const float* state_projection, int player,
                                                 int from, int to) {
  const size_t move_hidden = static_cast<size_t>(model.move_hidden_size);
  const int action = from * kBoardSize + to;
  if (!model.move_action_hidden.empty()) {
    const float* move_hidden_base =
        model.move_action_hidden.data() +
        (static_cast<size_t>(player) * static_cast<size_t>(model.action_size) +
         static_cast<size_t>(action)) *
            move_hidden;
    float score = model.move_out_b;
    for (size_t row = 0; row < move_hidden; ++row) {
      score +=
          model.move_out_w[row] * std::max(0.0f, state_projection[row] + move_hidden_base[row]);
    }
    return score;
  }
  std::array<float, kMoveFeatureSize> features{};
  encode_move_features(player, from, to, features.data());
  const size_t move_embed = static_cast<size_t>(model.move_embed_size);
  const size_t move_feature = static_cast<size_t>(model.move_feature_size);
  const size_t move_input = 2 * move_embed + move_feature;
  const float* from_embed =
      model.move_from_embed.data() + static_cast<size_t>(from) * move_embed;
  const float* to_embed = model.move_to_embed.data() + static_cast<size_t>(to) * move_embed;
  float score = model.move_out_b;
  for (size_t row = 0; row < move_hidden; ++row) {
    float activation = state_projection[row];
    const float* move_weights = model.move_w.data() + row * move_input;
    size_t offset = 0;
    for (size_t col = 0; col < move_embed; ++col) {
      activation += move_weights[offset + col] * from_embed[col];
    }
    offset += move_embed;
    for (size_t col = 0; col < move_embed; ++col) {
      activation += move_weights[offset + col] * to_embed[col];
    }
    offset += move_embed;
    for (size_t col = 0; col < move_feature; ++col) {
      activation += move_weights[offset + col] * features[col];
    }
    score += model.move_out_w[row] * std::max(0.0f, activation);
  }
  return score;
}

float mlp_bilinear_policy_logit_action_projected_ptr(const PolicyModel& model,
                                                     const float* state_projection,
                                                     int player, int from, int to) {
  const size_t move_hidden = static_cast<size_t>(model.move_hidden_size);
  const int action = from * kBoardSize + to;
  const float scale = 1.0f / std::sqrt(static_cast<float>(model.move_hidden_size));
  if (!model.move_action_hidden.empty() && !model.move_action_bias.empty()) {
    const size_t action_offset =
        static_cast<size_t>(player) * static_cast<size_t>(model.action_size) +
        static_cast<size_t>(action);
    const float* move_projection =
        model.move_action_hidden.data() + action_offset * move_hidden;
    float score = model.move_action_bias[action_offset];
    float interaction = 0.0f;
    for (size_t row = 0; row < move_hidden; ++row) {
      interaction += state_projection[row] * move_projection[row];
    }
    return score + interaction * scale;
  }

  std::array<float, kMoveFeatureSize> features{};
  encode_move_features(player, from, to, features.data());
  const size_t move_embed = static_cast<size_t>(model.move_embed_size);
  const size_t move_feature = static_cast<size_t>(model.move_feature_size);
  const size_t move_input = 2 * move_embed + move_feature;
  const float* from_embed =
      model.move_from_embed.data() + static_cast<size_t>(from) * move_embed;
  const float* to_embed = model.move_to_embed.data() + static_cast<size_t>(to) * move_embed;
  std::vector<float> input_values(move_input, 0.0f);
  size_t offset = 0;
  for (size_t col = 0; col < move_embed; ++col) {
    input_values[offset + col] = from_embed[col];
  }
  offset += move_embed;
  for (size_t col = 0; col < move_embed; ++col) {
    input_values[offset + col] = to_embed[col];
  }
  offset += move_embed;
  for (size_t col = 0; col < move_feature; ++col) {
    input_values[offset + col] = features[col];
  }

  float score = model.move_bias_b;
  for (size_t col = 0; col < move_input; ++col) {
    score += model.move_bias_w[col] * input_values[col];
  }
  float interaction = 0.0f;
  for (size_t row = 0; row < move_hidden; ++row) {
    const float* move_weights = model.move_w.data() + row * move_input;
    float move_projection = model.move_b[row];
    for (size_t col = 0; col < move_input; ++col) {
      move_projection += move_weights[col] * input_values[col];
    }
    interaction += state_projection[row] * move_projection;
  }
  return score + interaction * scale;
}

float mlp_move_policy_logit_action_ptr(const PolicyModel& model, const float* hidden,
                                       int player, int from, int to) {
  std::vector<float> state_projection(static_cast<size_t>(model.move_hidden_size));
  mlp_policy_state_projection(model, hidden, state_projection.data());
  if (model.policy_head == PolicyHeadKind::MoveBilinear) {
    return mlp_bilinear_policy_logit_action_projected_ptr(model, state_projection.data(), player,
                                                          from, to);
  }
  return mlp_move_policy_logit_action_projected_ptr(model, state_projection.data(), player, from,
                                                    to);
}

float mlp_policy_logit_action_projected_ptr(const PolicyModel& model, const float* hidden,
                                            const float* state_projection, int action,
                                            int player) {
  if (model.policy_head == PolicyHeadKind::MoveMlp) {
    return mlp_move_policy_logit_action_projected_ptr(model, state_projection, player,
                                                     action / kBoardSize, action % kBoardSize);
  }
  if (model.policy_head == PolicyHeadKind::MoveBilinear) {
    return mlp_bilinear_policy_logit_action_projected_ptr(
        model, state_projection, player, action / kBoardSize, action % kBoardSize);
  }
  return mlp_dense_policy_logit_action_ptr(model, hidden, action);
}

float mlp_policy_logit_action(const PolicyModel& model, const std::vector<float>& hidden,
                              int from, int to) {
  if (model.policy_head == PolicyHeadKind::MoveMlp ||
      model.policy_head == PolicyHeadKind::MoveBilinear) {
    throw std::runtime_error("compact policy scoring requires player-aware scoring");
  }
  const int action = from * kBoardSize + to;
  return mlp_dense_policy_logit_action_ptr(model, hidden.data(), action);
}

bool accelerate_compiled() {
#ifdef CCZERO_USE_ACCELERATE
  return true;
#else
  return false;
#endif
}

InferenceBackend resolve_inference_backend(InferenceBackend backend) {
  if (backend == InferenceBackend::Auto) {
    return accelerate_compiled() ? InferenceBackend::Accelerate : InferenceBackend::Portable;
  }
  if (backend == InferenceBackend::Accelerate && !accelerate_compiled()) {
    throw std::runtime_error(
        "accelerate inference backend requested but this binary was built without Accelerate");
  }
  return backend;
}

namespace {

void relu_in_place(float* values, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    values[i] = std::max(0.0f, values[i]);
  }
}

void mlp_sparse_input_hidden(const PolicyModel& model, const State& state, int player,
                             float* hidden) {
  const size_t hidden_size = static_cast<size_t>(model.hidden_size);
  std::copy(model.input_b.begin(), model.input_b.end(), hidden);
  const float* feature_major = model.input_w_feature_major.data();
  for (int id = 0; id < kBoardSize; ++id) {
    const int occupant = state.cells[static_cast<size_t>(id)];
    int feature = -1;
    if (occupant == player) {
      feature = id;
    } else if (occupant == 1 - player) {
      feature = kBoardSize + id;
    }
    if (feature < 0) {
      continue;
    }
    const float* column = feature_major + static_cast<size_t>(feature) * hidden_size;
    for (size_t row = 0; row < hidden_size; ++row) {
      hidden[row] += column[row];
    }
  }
  const float side = player == 0 ? 1.0f : -1.0f;
  const float* side_column = feature_major + static_cast<size_t>(242) * hidden_size;
  for (size_t row = 0; row < hidden_size; ++row) {
    hidden[row] += side * side_column[row];
  }
  relu_in_place(hidden, hidden_size);
}

void mlp_geometry_input_hidden(const PolicyModel& model, const State& state, int player,
                               float* hidden) {
  const size_t hidden_size = static_cast<size_t>(model.hidden_size);
  const Board& board = Board::standard();
  const int opponent = 1 - player;
  const std::vector<float>& static_hidden =
      model.geometry_static_hidden[static_cast<size_t>(player)];
  std::copy(static_hidden.begin(), static_hidden.end(), hidden);
  for (int id = 0; id < kBoardSize; ++id) {
    const int occupant = state.cells[static_cast<size_t>(id)];
    if (occupant == player) {
      add_feature_column(model, id, 1.0f, hidden);
    } else if (occupant == opponent) {
      add_feature_column(model, kBoardSize + id, 1.0f, hidden);
    }
  }
  add_feature_column(model, 242, player == 0 ? 1.0f : -1.0f, hidden);

  float own_goal = 0.0f;
  float opp_goal = 0.0f;
  float own_home = 0.0f;
  float opp_home = 0.0f;
  float own_distance = 0.0f;
  float opp_distance = 0.0f;
  float own_blockers = 0.0f;
  float opp_blockers = 0.0f;
  for (int id = 0; id < kBoardSize; ++id) {
    const int occupant = state.cells[static_cast<size_t>(id)];
    if (occupant == player) {
      own_distance += static_cast<float>(board.goal_distance(player, id));
      own_goal += board.is_goal(player, id) ? 1.0f : 0.0f;
      own_home += board.is_home(player, id) ? 1.0f : 0.0f;
      opp_blockers += board.is_goal(opponent, id) ? 1.0f : 0.0f;
    } else if (occupant == opponent) {
      opp_distance += static_cast<float>(board.goal_distance(opponent, id));
      opp_goal += board.is_goal(opponent, id) ? 1.0f : 0.0f;
      opp_home += board.is_home(opponent, id) ? 1.0f : 0.0f;
      own_blockers += board.is_goal(player, id) ? 1.0f : 0.0f;
    }
  }
  const float ply = static_cast<float>(std::min(state.ply, 240));
  const float phase = state.ply < 40 ? 0.0f : (state.ply < 90 ? 0.5f : 1.0f);
  const float v1_scalars[] = {
      own_goal / 10.0f,
      opp_goal / 10.0f,
      own_home / 10.0f,
      opp_home / 10.0f,
      own_distance / 110.0f,
      opp_distance / 110.0f,
      (opp_distance - own_distance) / 110.0f,
      own_blockers / 10.0f,
      opp_blockers / 10.0f,
      ply / 240.0f,
      phase,
      player == 0 ? 1.0f : -1.0f,
  };
  for (int i = 0; i < 12; ++i) {
    add_feature_column(model, kBaseFeatureSize + kBoardSize * 10 + i, v1_scalars[i], hidden);
  }

  if (model.feature_size == kGeometryV2FeatureSize ||
      model.feature_size == kGeometryV3FeatureSize) {
    constexpr int kV2DynamicOffset = kGeometryFeatureSize + kBoardSize * 2;
    std::array<float, kBoardSize> own_neighbor{};
    std::array<float, kBoardSize> opp_neighbor{};
    std::array<float, kBoardSize> empty_neighbor{};
    std::array<float, kBoardSize> own_step_origin{};
    std::array<float, kBoardSize> opp_step_origin{};
    std::array<float, kBoardSize> own_jump_origin{};
    std::array<float, kBoardSize> opp_jump_origin{};
    int own_step_total = 0;
    int opp_step_total = 0;
    int own_jump_total = 0;
    int opp_jump_total = 0;
    for (int id = 0; id < kBoardSize; ++id) {
      int own_n = 0;
      int opp_n = 0;
      int empty_n = 0;
      int own_steps = 0;
      int opp_steps = 0;
      int own_jumps = 0;
      int opp_jumps = 0;
      for (int dir = 0; dir < 6; ++dir) {
        const int neighbor = board.neighbors(id)[static_cast<size_t>(dir)];
        if (neighbor != kInvalid) {
          const int occupant = state.cells[static_cast<size_t>(neighbor)];
          if (occupant == player) {
            ++own_n;
          } else if (occupant == opponent) {
            ++opp_n;
          } else {
            ++empty_n;
            if (state.cells[static_cast<size_t>(id)] == player) {
              ++own_steps;
            } else if (state.cells[static_cast<size_t>(id)] == opponent) {
              ++opp_steps;
            }
          }
        }
        const int mid = board.jump_mid(id, dir);
        const int landing = board.jump_landing(id, dir);
        if (mid != kInvalid && landing != kInvalid &&
            state.cells[static_cast<size_t>(mid)] != kEmpty &&
            state.cells[static_cast<size_t>(landing)] == kEmpty) {
          if (state.cells[static_cast<size_t>(id)] == player) {
            ++own_jumps;
          } else if (state.cells[static_cast<size_t>(id)] == opponent) {
            ++opp_jumps;
          }
        }
      }
      own_neighbor[static_cast<size_t>(id)] = static_cast<float>(own_n) / 6.0f;
      opp_neighbor[static_cast<size_t>(id)] = static_cast<float>(opp_n) / 6.0f;
      empty_neighbor[static_cast<size_t>(id)] = static_cast<float>(empty_n) / 6.0f;
      own_step_origin[static_cast<size_t>(id)] = static_cast<float>(own_steps) / 6.0f;
      opp_step_origin[static_cast<size_t>(id)] = static_cast<float>(opp_steps) / 6.0f;
      own_jump_origin[static_cast<size_t>(id)] = static_cast<float>(own_jumps) / 6.0f;
      opp_jump_origin[static_cast<size_t>(id)] = static_cast<float>(opp_jumps) / 6.0f;
      own_step_total += own_steps;
      opp_step_total += opp_steps;
      own_jump_total += own_jumps;
      opp_jump_total += opp_jumps;
    }
    auto add_plane = [&](int plane, const std::array<float, kBoardSize>& values) {
      for (int id = 0; id < kBoardSize; ++id) {
        add_feature_column(model, kV2DynamicOffset + plane * kBoardSize + id,
                           values[static_cast<size_t>(id)], hidden);
      }
    };
    add_plane(0, own_neighbor);
    add_plane(1, opp_neighbor);
    add_plane(2, empty_neighbor);
    add_plane(3, own_step_origin);
    add_plane(4, opp_step_origin);
    add_plane(5, own_jump_origin);
    add_plane(6, opp_jump_origin);
    int own_movable = 0;
    int opp_movable = 0;
    for (int id = 0; id < kBoardSize; ++id) {
      if (state.cells[static_cast<size_t>(id)] == player &&
          (own_step_origin[static_cast<size_t>(id)] > 0.0f ||
           own_jump_origin[static_cast<size_t>(id)] > 0.0f)) {
        ++own_movable;
      } else if (state.cells[static_cast<size_t>(id)] == opponent &&
                 (opp_step_origin[static_cast<size_t>(id)] > 0.0f ||
                  opp_jump_origin[static_cast<size_t>(id)] > 0.0f)) {
        ++opp_movable;
      }
    }
    const int own_mobility = own_step_total + 2 * own_jump_total;
    const int opp_mobility = opp_step_total + 2 * opp_jump_total;
    const float v2_scalars[] = {
        static_cast<float>(own_movable) / 10.0f,
        static_cast<float>(opp_movable) / 10.0f,
        static_cast<float>(std::min(own_step_total, 60)) / 60.0f,
        static_cast<float>(std::min(opp_step_total, 60)) / 60.0f,
        static_cast<float>(std::min(own_jump_total, 60)) / 60.0f,
        static_cast<float>(std::min(opp_jump_total, 60)) / 60.0f,
        0.0f,
        std::clamp(static_cast<float>(own_mobility - opp_mobility) / 120.0f, -1.0f, 1.0f),
    };
    constexpr int kV2ScalarOffset = kV2DynamicOffset + kBoardSize * 7;
    for (int i = 0; i < 8; ++i) {
      add_feature_column(model, kV2ScalarOffset + i, v2_scalars[i], hidden);
    }
    if (model.feature_size == kGeometryV3FeatureSize) {
      constexpr int kV3PlaneOffset = kGeometryV2FeatureSize;
      constexpr int kV3ScalarOffset = kV3PlaneOffset + kBoardSize * 4;
      std::array<float, kBoardSize> own_goal_pressure{};
      std::array<float, kBoardSize> opp_goal_pressure{};
      std::array<float, kBoardSize> own_back_rank{};
      std::array<float, kBoardSize> opp_back_rank{};
      int own_goal_empty = 0;
      int opp_goal_empty = 0;
      int own_home_stragglers = 0;
      int opp_home_stragglers = 0;
      int own_isolated = 0;
      int opp_isolated = 0;
      int own_backward = 0;
      int opp_backward = 0;
      float own_forward_sum = 0.0f;
      float opp_forward_sum = 0.0f;
      for (int id = 0; id < kBoardSize; ++id) {
        const int occupant = state.cells[static_cast<size_t>(id)];
        const bool own_piece = occupant == player;
        const bool opp_piece = occupant == opponent;
        const float forward =
            player == 0 ? static_cast<float>(board.coord(id).r + 8) / 16.0f
                        : static_cast<float>(8 - board.coord(id).r) / 16.0f;
        own_goal_pressure[static_cast<size_t>(id)] =
            board.is_goal(player, id) ? (own_piece ? 1.0f : (opp_piece ? -1.0f : 0.0f)) : 0.0f;
        opp_goal_pressure[static_cast<size_t>(id)] =
            board.is_goal(opponent, id) ? (opp_piece ? 1.0f : (own_piece ? -1.0f : 0.0f)) : 0.0f;
        own_back_rank[static_cast<size_t>(id)] =
            own_piece ? static_cast<float>(board.goal_distance(player, id)) / 16.0f : 0.0f;
        opp_back_rank[static_cast<size_t>(id)] =
            opp_piece ? static_cast<float>(board.goal_distance(opponent, id)) / 16.0f : 0.0f;
        own_forward_sum += own_piece ? forward : 0.0f;
        opp_forward_sum += opp_piece ? (1.0f - forward) : 0.0f;
        if (board.is_goal(player, id) && occupant == kEmpty) {
          ++own_goal_empty;
        }
        if (board.is_goal(opponent, id) && occupant == kEmpty) {
          ++opp_goal_empty;
        }
        if (board.is_home(player, id) && own_piece) {
          ++own_home_stragglers;
        }
        if (board.is_home(opponent, id) && opp_piece) {
          ++opp_home_stragglers;
        }
        if (own_piece && own_neighbor[static_cast<size_t>(id)] <= 0.0f &&
            own_step_origin[static_cast<size_t>(id)] <= 0.0f &&
            own_jump_origin[static_cast<size_t>(id)] <= 0.0f) {
          ++own_isolated;
        }
        if (opp_piece && opp_neighbor[static_cast<size_t>(id)] <= 0.0f &&
            opp_step_origin[static_cast<size_t>(id)] <= 0.0f &&
            opp_jump_origin[static_cast<size_t>(id)] <= 0.0f) {
          ++opp_isolated;
        }
        if (own_back_rank[static_cast<size_t>(id)] >= 0.625f) {
          ++own_backward;
        }
        if (opp_back_rank[static_cast<size_t>(id)] >= 0.625f) {
          ++opp_backward;
        }
      }
      auto add_v3_plane = [&](int plane, const std::array<float, kBoardSize>& values) {
        for (int id = 0; id < kBoardSize; ++id) {
          add_feature_column(model, kV3PlaneOffset + plane * kBoardSize + id,
                             values[static_cast<size_t>(id)], hidden);
        }
      };
      add_v3_plane(0, own_goal_pressure);
      add_v3_plane(1, opp_goal_pressure);
      add_v3_plane(2, own_back_rank);
      add_v3_plane(3, opp_back_rank);
      const float own_fill = own_goal / std::max(1.0f, own_goal + static_cast<float>(own_goal_empty));
      const float opp_fill = opp_goal / std::max(1.0f, opp_goal + static_cast<float>(opp_goal_empty));
      const float v3_scalars[] = {
          static_cast<float>(own_goal_empty) / 10.0f,
          static_cast<float>(opp_goal_empty) / 10.0f,
          own_fill,
          opp_fill,
          own_fill - opp_fill,
          static_cast<float>(own_home_stragglers) / 10.0f,
          static_cast<float>(opp_home_stragglers) / 10.0f,
          static_cast<float>(opp_home_stragglers - own_home_stragglers) / 10.0f,
          static_cast<float>(own_isolated) / 10.0f,
          static_cast<float>(opp_isolated) / 10.0f,
          static_cast<float>(opp_isolated - own_isolated) / 10.0f,
          static_cast<float>(own_backward) / 10.0f,
          static_cast<float>(opp_backward) / 10.0f,
          static_cast<float>(opp_backward - own_backward) / 10.0f,
          own_forward_sum / 10.0f,
          opp_forward_sum / 10.0f,
          (own_forward_sum - opp_forward_sum) / 10.0f,
          phase * ((own_fill - opp_fill) +
                   static_cast<float>(opp_home_stragglers - own_home_stragglers) / 10.0f),
      };
      for (int i = 0; i < 18; ++i) {
        add_feature_column(model, kV3ScalarOffset + i, v3_scalars[i], hidden);
      }
    }
  }
  relu_in_place(hidden, hidden_size);
}

void dense_relu_portable(const float* weights, const float* bias, const float* input,
                         float* output, size_t hidden_size) {
  for (size_t row = 0; row < hidden_size; ++row) {
    const float* w = weights + row * hidden_size;
    float sum = bias[row];
    for (size_t col = 0; col < hidden_size; ++col) {
      sum += w[col] * input[col];
    }
    output[row] = std::max(0.0f, sum);
  }
}

void dense_residual_relu_portable(const float* weights, const float* bias, const float* input,
                                  const float* residual, float* output, size_t hidden_size) {
  for (size_t row = 0; row < hidden_size; ++row) {
    const float* w = weights + row * hidden_size;
    float sum = bias[row];
    for (size_t col = 0; col < hidden_size; ++col) {
      sum += w[col] * input[col];
    }
    output[row] = std::max(0.0f, residual[row] + sum);
  }
}

void dense_relu_accelerate(const float* weights, const float* bias, const float* input,
                           float* output, size_t hidden_size, InferenceBackend backend) {
  if (backend == InferenceBackend::Accelerate) {
#ifdef CCZERO_USE_ACCELERATE
    std::copy(bias, bias + hidden_size, output);
    cblas_sgemv(CblasRowMajor, CblasNoTrans, static_cast<int>(hidden_size),
                static_cast<int>(hidden_size), 1.0f, weights, static_cast<int>(hidden_size),
                input, 1, 1.0f, output, 1);
    relu_in_place(output, hidden_size);
    return;
#else
    (void)weights;
    (void)bias;
    (void)input;
    (void)output;
    (void)hidden_size;
#endif
  }
  dense_relu_portable(weights, bias, input, output, hidden_size);
}

void dense_residual_relu_accelerate(const float* weights, const float* bias, const float* input,
                                    const float* residual, float* output, size_t hidden_size,
                                    InferenceBackend backend) {
  if (backend == InferenceBackend::Accelerate) {
#ifdef CCZERO_USE_ACCELERATE
    std::copy(bias, bias + hidden_size, output);
    cblas_sgemv(CblasRowMajor, CblasNoTrans, static_cast<int>(hidden_size),
                static_cast<int>(hidden_size), 1.0f, weights, static_cast<int>(hidden_size),
                input, 1, 1.0f, output, 1);
    for (size_t row = 0; row < hidden_size; ++row) {
      output[row] = std::max(0.0f, residual[row] + output[row]);
    }
    return;
#else
    (void)weights;
    (void)bias;
    (void)input;
    (void)residual;
    (void)output;
    (void)hidden_size;
#endif
  }
  dense_residual_relu_portable(weights, bias, input, residual, output, hidden_size);
}

}  // namespace

void mlp_hidden_optimized(const PolicyModel& model, const State& state, int player,
                          MlpWorkspace& workspace, InferenceBackend backend) {
  const size_t hidden_size = static_cast<size_t>(model.hidden_size);
  workspace.ensure(1, hidden_size);
  float* hidden = workspace.hidden.data();
  float* tmp = workspace.tmp.data();
  float* residual = workspace.residual.data();
  if (model.feature_size == kBaseFeatureSize) {
    mlp_sparse_input_hidden(model, state, player, hidden);
  } else {
    mlp_geometry_input_hidden(model, state, player, hidden);
  }
  backend = resolve_inference_backend(backend);
  for (int block = 0; block < model.blocks; ++block) {
    const size_t matrix_offset = static_cast<size_t>(block) * hidden_size * hidden_size;
    const size_t bias_offset = static_cast<size_t>(block) * hidden_size;
    dense_relu_accelerate(model.block_w1.data() + matrix_offset,
                          model.block_b1.data() + bias_offset, hidden, tmp, hidden_size, backend);
    dense_residual_relu_accelerate(model.block_w2.data() + matrix_offset,
                                   model.block_b2.data() + bias_offset, tmp, hidden, residual,
                                   hidden_size, backend);
    std::swap(hidden, residual);
    if (hidden != workspace.hidden.data()) {
      std::copy(hidden, hidden + hidden_size, workspace.hidden.data());
      hidden = workspace.hidden.data();
      residual = workspace.residual.data();
    }
  }
}

namespace {

#ifdef CCZERO_USE_ACCELERATE
void fill_batch_bias(float* values, const float* bias, size_t rows, size_t cols) {
  for (size_t row = 0; row < rows; ++row) {
    std::copy(bias, bias + cols, values + row * cols);
  }
}
#endif

}  // namespace

float mlp_value_from_hidden_ptr(const PolicyModel& model, const float* hidden) {
  const size_t hidden_size = static_cast<size_t>(model.hidden_size);
  float score = model.value_b;
  for (size_t i = 0; i < hidden_size; ++i) {
    score += model.value_w[i] * hidden[i];
  }
  return std::tanh(score);
}

void mlp_hidden_batch_optimized(const PolicyModel& model, const std::vector<const State*>& states,
                                MlpWorkspace& workspace, InferenceBackend backend) {
  const size_t batch = states.size();
  const size_t hidden_size = static_cast<size_t>(model.hidden_size);
  workspace.ensure(batch, hidden_size);
  for (size_t row = 0; row < batch; ++row) {
    const State& state = *states.at(row);
    if (model.feature_size == kBaseFeatureSize) {
      mlp_sparse_input_hidden(model, state, state.player_to_move,
                              workspace.hidden.data() + row * hidden_size);
    } else {
      mlp_geometry_input_hidden(model, state, state.player_to_move,
                                workspace.hidden.data() + row * hidden_size);
    }
  }

  backend = resolve_inference_backend(backend);
  if (backend == InferenceBackend::Accelerate && batch > 1) {
#ifdef CCZERO_USE_ACCELERATE
    for (int block = 0; block < model.blocks; ++block) {
      const size_t matrix_offset = static_cast<size_t>(block) * hidden_size * hidden_size;
      const size_t bias_offset = static_cast<size_t>(block) * hidden_size;
      fill_batch_bias(workspace.tmp.data(), model.block_b1.data() + bias_offset, batch,
                      hidden_size);
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, static_cast<int>(batch),
                  static_cast<int>(hidden_size), static_cast<int>(hidden_size), 1.0f,
                  workspace.hidden.data(), static_cast<int>(hidden_size),
                  model.block_w1.data() + matrix_offset, static_cast<int>(hidden_size), 1.0f,
                  workspace.tmp.data(), static_cast<int>(hidden_size));
      relu_in_place(workspace.tmp.data(), batch * hidden_size);

      fill_batch_bias(workspace.residual.data(), model.block_b2.data() + bias_offset, batch,
                      hidden_size);
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, static_cast<int>(batch),
                  static_cast<int>(hidden_size), static_cast<int>(hidden_size), 1.0f,
                  workspace.tmp.data(), static_cast<int>(hidden_size),
                  model.block_w2.data() + matrix_offset, static_cast<int>(hidden_size), 1.0f,
                  workspace.residual.data(), static_cast<int>(hidden_size));
      for (size_t i = 0; i < batch * hidden_size; ++i) {
        workspace.hidden[i] = std::max(0.0f, workspace.hidden[i] + workspace.residual[i]);
      }
    }
    cblas_sgemv(CblasRowMajor, CblasNoTrans, static_cast<int>(batch),
                static_cast<int>(hidden_size), 1.0f, workspace.hidden.data(),
                static_cast<int>(hidden_size), model.value_w.data(), 1, 0.0f,
                workspace.values.data(), 1);
    for (size_t row = 0; row < batch; ++row) {
      workspace.values[row] = std::tanh(model.value_b + workspace.values[row]);
    }
    return;
#endif
  }

  for (size_t row = 0; row < batch; ++row) {
    float* hidden = workspace.hidden.data() + row * hidden_size;
    float* tmp = workspace.tmp.data() + row * hidden_size;
    float* residual = workspace.residual.data() + row * hidden_size;
    for (int block = 0; block < model.blocks; ++block) {
      const size_t matrix_offset = static_cast<size_t>(block) * hidden_size * hidden_size;
      const size_t bias_offset = static_cast<size_t>(block) * hidden_size;
      dense_relu_accelerate(model.block_w1.data() + matrix_offset,
                            model.block_b1.data() + bias_offset, hidden, tmp, hidden_size,
                            backend);
      dense_residual_relu_accelerate(model.block_w2.data() + matrix_offset,
                                     model.block_b2.data() + bias_offset, tmp, hidden, residual,
                                     hidden_size, backend);
      std::copy(residual, residual + hidden_size, hidden);
    }
    workspace.values[row] = mlp_value_from_hidden_ptr(model, hidden);
  }
}

float mlp_policy_logit_action_ptr(const PolicyModel& model, const float* hidden, int action,
                                  int player) {
  if (model.policy_head == PolicyHeadKind::MoveMlp ||
      model.policy_head == PolicyHeadKind::MoveBilinear) {
    return mlp_move_policy_logit_action_ptr(model, hidden, player, action / kBoardSize,
                                           action % kBoardSize);
  }
  float score = model.policy_b[static_cast<size_t>(action)];
  const size_t hidden_size = static_cast<size_t>(model.hidden_size);
  const float* weights = model.policy_w.data() + static_cast<size_t>(action) * hidden_size;
  size_t i = 0;
  float sum0 = 0.0f;
  float sum1 = 0.0f;
  float sum2 = 0.0f;
  float sum3 = 0.0f;
  for (; i + 3 < hidden_size; i += 4) {
    sum0 += weights[i] * hidden[i];
    sum1 += weights[i + 1] * hidden[i + 1];
    sum2 += weights[i + 2] * hidden[i + 2];
    sum3 += weights[i + 3] * hidden[i + 3];
  }
  score += sum0 + sum1 + sum2 + sum3;
  for (; i < hidden_size; ++i) {
    score += weights[i] * hidden[i];
  }
  return score;
}

float policy_score_action(const PolicyModel& model, const State& state, int player,
                          int from, int to) {
  if (model.kind == ModelKind::PolicyValueMlp) {
    const std::vector<float> features = encode_policy_features(state, player, model.feature_size);
    const std::vector<float> hidden = mlp_hidden(model, features);
    if (model.policy_head == PolicyHeadKind::MoveMlp ||
        model.policy_head == PolicyHeadKind::MoveBilinear) {
      return mlp_move_policy_logit_action_ptr(model, hidden.data(), player, from, to);
    }
    return mlp_policy_logit_action(model, hidden, from, to);
  }

  const int action = from * kBoardSize + to;
  float score = model.b.at(static_cast<size_t>(action));
  for (int id = 0; id < kBoardSize; ++id) {
    const int occupant = state.cells.at(static_cast<size_t>(id));
    if (occupant == player) {
      score += model.w.at(static_cast<size_t>(id) * model.action_size + action);
    } else if (occupant == 1 - player) {
      score += model.w.at(static_cast<size_t>(kBoardSize + id) * model.action_size + action);
    }
  }
  score += model.w.at(static_cast<size_t>(242) * model.action_size + action) *
           (player == 0 ? 1.0f : -1.0f);
  return score;
}

float policy_score(const PolicyModel& model, const State& state, int player,
                   const Move& move) {
  return policy_score_action(model, state, player, move.from, move.to);
}

std::string inference_backend_name(InferenceBackend backend) {
  switch (backend) {
    case InferenceBackend::Auto:
      return "auto";
    case InferenceBackend::Portable:
      return "portable";
    case InferenceBackend::Accelerate:
      return "accelerate";
  }
  return "unknown";
}

std::string model_kind_name(ModelKind kind) {
  switch (kind) {
    case ModelKind::LinearPolicy:
      return "linear_policy";
    case ModelKind::PolicyValueMlp:
      return "policy_value_mlp";
  }
  return "unknown";
}

std::string policy_head_kind_name(PolicyHeadKind kind) {
  switch (kind) {
    case PolicyHeadKind::Dense:
      return "dense";
    case PolicyHeadKind::MoveMlp:
      return "move_mlp";
    case PolicyHeadKind::MoveBilinear:
      return "move_bilinear";
  }
  return "unknown";
}

InferenceBackend parse_inference_backend(const std::string& text) {
  if (text == "auto") {
    return InferenceBackend::Auto;
  }
  if (text == "portable" || text == "scalar") {
    return InferenceBackend::Portable;
  }
  if (text == "accelerate" || text == "apple" || text == "vecLib") {
    return InferenceBackend::Accelerate;
  }
  throw std::runtime_error("unknown inference backend: " + text);
}

}  // namespace cczero
