#pragma once

#include "cczero/cczero.h"

#include <cstddef>
#include <string>
#include <vector>

namespace cczero {

enum class ModelKind {
  LinearPolicy,
  PolicyValueMlp,
};

enum class PolicyHeadKind {
  Dense,
  MoveMlp,
  MoveBilinear,
};

enum class InferenceBackend {
  Auto,
  Portable,
  Accelerate,
};

struct PolicyModel {
  ModelKind kind = ModelKind::LinearPolicy;
  PolicyHeadKind policy_head = PolicyHeadKind::Dense;
  int feature_size = 0;
  int action_size = 0;
  int hidden_size = 0;
  int blocks = 0;
  int move_embed_size = 0;
  int move_hidden_size = 0;
  int move_feature_size = 0;
  std::vector<float> w;
  std::vector<float> b;
  std::vector<float> input_w;
  std::vector<float> input_b;
  std::vector<float> block_w1;
  std::vector<float> block_b1;
  std::vector<float> block_w2;
  std::vector<float> block_b2;
  std::vector<float> policy_w;
  std::vector<float> policy_b;
  std::vector<float> move_from_embed;
  std::vector<float> move_to_embed;
  std::vector<float> move_w;
  std::vector<float> move_state_w;
  std::vector<float> move_b;
  std::vector<float> move_out_w;
  float move_out_b = 0.0f;
  std::vector<float> move_bias_w;
  float move_bias_b = 0.0f;
  std::vector<float> move_action_hidden;
  std::vector<float> move_action_bias;
  std::vector<float> value_w;
  float value_b = 0.0f;
  std::vector<float> input_w_feature_major;
  std::array<std::vector<float>, kPlayers> geometry_static_hidden;
};

struct MlpWorkspace {
  std::vector<float> hidden;
  std::vector<float> tmp;
  std::vector<float> residual;
  std::vector<float> values;
  std::vector<float> policy_state;

  void ensure(size_t batch, size_t hidden_size);
  void ensure_policy_state(size_t move_hidden_size);
};

PolicyModel load_policy_model(const std::string& path);
size_t policy_model_parameter_count(const PolicyModel& model);
size_t policy_model_storage_bytes(const PolicyModel& model);
void validate_policy_model(const PolicyModel& model, const std::string& label = "");

std::vector<float> encode_policy_features(const State& state, int player,
                                          int feature_size = 243);
std::vector<float> mlp_hidden(const PolicyModel& model, const std::vector<float>& features);
float mlp_policy_logit_action(const PolicyModel& model, const std::vector<float>& hidden,
                              int from, int to);

bool accelerate_compiled();
InferenceBackend resolve_inference_backend(InferenceBackend backend);
std::string inference_backend_name(InferenceBackend backend);
std::string model_kind_name(ModelKind kind);
std::string policy_head_kind_name(PolicyHeadKind kind);
InferenceBackend parse_inference_backend(const std::string& text);

void mlp_hidden_optimized(const PolicyModel& model, const State& state, int player,
                          MlpWorkspace& workspace, InferenceBackend backend);
void mlp_hidden_batch_optimized(const PolicyModel& model, const std::vector<const State*>& states,
                                MlpWorkspace& workspace, InferenceBackend backend);
float mlp_value_from_hidden_ptr(const PolicyModel& model, const float* hidden);
float mlp_policy_logit_action_ptr(const PolicyModel& model, const float* hidden, int action,
                                  int player);
void mlp_policy_state_projection(const PolicyModel& model, const float* hidden, float* output);
float mlp_policy_logit_action_projected_ptr(const PolicyModel& model, const float* hidden,
                                            const float* state_projection, int action,
                                            int player);

float policy_score_action(const PolicyModel& model, const State& state, int player,
                          int from, int to);
float policy_score(const PolicyModel& model, const State& state, int player,
                   const Move& move);

}  // namespace cczero
