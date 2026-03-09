#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "tts_types.h"

#include <string>
#include <map>
#include <vector>
#include <memory>

namespace qwen3 {

// Extended config for the LLM part of the Talker
struct Qwen3TalkerLLMConfig {
    int32_t vocab_size = 151936;     // text vocab size
    int32_t codec_vocab_size = 3072; // codec vocab size
    int32_t hidden_size = 1024;
    int32_t text_hidden_size = 2048; // text embedding dim (before projection)
    int32_t n_layers = 28;
    int32_t n_heads = 16;
    int32_t n_kv_heads = 8;
    int32_t intermediate_size = 3072;
    int32_t head_dim = 128;
    float rms_norm_eps = 1e-6f;
    float rope_theta = 1000000.0f;

    // MRoPE sections: temporal, height, width (rotation pair counts; ×2 for actual dims)
    // e.g. [24, 20, 20] → dims [48, 40, 40], sum = 128 = head_dim
    std::vector<int32_t> mrope_section = {24, 20, 20};
};

struct Qwen3TalkerLayer {
    struct ggml_tensor * attn_norm = nullptr;
    
    struct ggml_tensor * attn_q = nullptr;
    struct ggml_tensor * attn_k = nullptr;
    struct ggml_tensor * attn_v = nullptr;
    struct ggml_tensor * attn_output = nullptr;
    
    // Qwen2/3 specific norms if present
    struct ggml_tensor * attn_q_norm = nullptr;
    struct ggml_tensor * attn_k_norm = nullptr;
    
    struct ggml_tensor * ffn_norm = nullptr;
    
    struct ggml_tensor * ffn_gate = nullptr;
    struct ggml_tensor * ffn_up = nullptr;
    struct ggml_tensor * ffn_down = nullptr;
};

// Talker model weights
struct Qwen3TalkerModel {
    Qwen3TalkerLLMConfig config;

    // Text embedding: text tokens → 2048-d
    struct ggml_tensor * text_embd = nullptr;

    // Text projection: 2048 → hidden_size (1024)
    struct ggml_tensor * text_proj_fc1_w = nullptr;
    struct ggml_tensor * text_proj_fc1_b = nullptr;
    struct ggml_tensor * text_proj_fc2_w = nullptr;
    struct ggml_tensor * text_proj_fc2_b = nullptr;

    // Codec embedding: codec tokens → hidden_size (1024)
    struct ggml_tensor * codec_embd = nullptr;

    // Transformer layers
    std::vector<Qwen3TalkerLayer> layers;

    // Final RMSNorm
    struct ggml_tensor * output_norm = nullptr;

    // LM head (codec_head)
    struct ggml_tensor * output = nullptr;

    // GGML context for tensor metadata
    struct ggml_context * ctx = nullptr;

    // Backend buffer for weights
    ggml_backend_buffer_t buffer = nullptr;

    // Tensor name to tensor mapping
    std::map<std::string, struct ggml_tensor *> tensors;
};

// KV cache for autoregressive generation
struct Qwen3KVCache {
    std::vector<struct ggml_tensor *> k_cache;  // Per-layer K cache
    std::vector<struct ggml_tensor *> v_cache;  // Per-layer V cache
    
    struct ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    
    int32_t n_ctx = 0;      // Maximum context length
    int32_t n_used = 0;     // Current number of cached tokens
    int32_t head_dim = 0;
    int32_t n_kv_heads = 0;
    int32_t n_layers = 0;
};

// Runtime state
struct Qwen3TalkerState {
    ggml_backend_t backend = nullptr;
    ggml_backend_sched_t sched = nullptr;
    
    std::vector<uint8_t> compute_meta;
    
    Qwen3KVCache cache;
};

// Code predictor config (5-layer transformer, standard 1D RoPE)
struct Qwen3CodePredictorConfig {
    int32_t vocab_size = 2048;       // codec vocab (smaller than talker's 3072)
    int32_t hidden_size = 1024;
    int32_t n_layers = 5;
    int32_t n_heads = 16;
    int32_t n_kv_heads = 8;
    int32_t intermediate_size = 3072;
    int32_t head_dim = 128;
    float rms_norm_eps = 1e-6f;
    float rope_theta = 10000.0f;     // standard RoPE (NOT 1M like talker)
    int32_t num_code_groups = 16;    // total codebooks (code_0 from talker + 15 predicted)
};

// Code predictor model weights
struct Qwen3CodePredictorModel {
    Qwen3CodePredictorConfig config;

    // Per-codebook embeddings: codec_embd[i] embeds codebook i+1
    std::vector<struct ggml_tensor *> codec_embd;  // [num_code_groups - 1]

    // Per-codebook LM heads: lm_head[i] predicts codebook i+1
    std::vector<struct ggml_tensor *> lm_head;     // [num_code_groups - 1]

    // Transformer layers (same structure as talker layers)
    std::vector<Qwen3TalkerLayer> layers;

    // Final RMSNorm
    struct ggml_tensor * output_norm = nullptr;
};

// Main Talker LLM class (includes code predictor)
class Qwen3TalkerLLM {
public:
    Qwen3TalkerLLM();
    ~Qwen3TalkerLLM();

    // Load model from GGUF file
    bool load_model(const std::string & model_path);

    // Initialize KV cache for given context length
    bool init_kv_cache(int32_t n_ctx);

    // Clear KV cache (for new sequence)
    void clear_kv_cache();

    // Forward pass: compute logits for input tokens
    // tokens: input token IDs [n_tokens]
    // pos_ids: MRoPE position IDs [n_tokens * 4] (temporal, height, width, extra)
    // n_past: number of tokens already in KV cache
    // is_codec: if true, use codec_embd; if false, use text_embd + text_proj
    // output: logits [n_tokens, codec_vocab_size] (flattened)
    // hidden_out: if non-null, filled with hidden states [n_tokens, hidden_size] (pre-lm_head)
    bool forward(const int32_t * tokens, const int32_t * pos_ids, int32_t n_tokens, int32_t n_past,
                 bool is_codec, std::vector<float> & output,
                 std::vector<float> * hidden_out = nullptr);

    // Forward pass with pre-computed embeddings
    // embeds: [n_tokens, hidden_size] pre-computed input embeddings
    // pos_ids: MRoPE position IDs [n_tokens * 4] (temporal, height, width, extra)
    bool forward_embeds(const float * embeds, const int32_t * pos_ids, int32_t n_tokens, int32_t n_past,
                        std::vector<float> & output,
                        std::vector<float> * hidden_out = nullptr);

    // Predict codes 1-15 given talker hidden state and code_0
    // talker_hidden: [hidden_size] from last talker forward pass
    // code_0: the code sampled from talker's logits
    // out_codes: filled with [num_code_groups - 1] predicted codes
    bool predict_codes(const float * talker_hidden, int32_t code_0,
                       std::vector<int32_t> & out_codes);

    // Compute text embedding: text_proj(text_embd(token_id)) → float[hidden_size]
    void compute_text_embedding(int32_t token_id, float * out) const;

    // Compute codec embedding: codec_embd(token_id) → float[hidden_size]
    void compute_codec_embedding(int32_t token_id, float * out) const;

    // Compute code predictor codebook embedding: code_pred.codec_embd[codebook_idx](token_id) → float[hidden_size]
    void compute_code_pred_embedding(int32_t codebook_idx, int32_t token_id, float * out) const;

    const Qwen3TalkerLLMConfig & get_config() const { return model_.config; }
    const Qwen3CodePredictorConfig & get_code_pred_config() const { return code_pred_.config; }
    const std::string & get_error() const { return error_msg_; }

private:
    // Build computation graph for forward pass
    struct ggml_cgraph * build_graph(const int32_t * tokens, const int32_t * pos_ids, int32_t n_tokens, int32_t n_past, bool is_codec);

    // Build computation graph with pre-computed embeddings
    struct ggml_cgraph * build_graph_embeds(int32_t n_tokens, int32_t n_past);

    // Internal: shared graph builder (embed_mode: 0=text, 1=codec, 2=external)
    struct ggml_cgraph * build_graph_internal(int32_t n_tokens, int32_t n_past, int embed_mode);

    // Build code predictor graph (standard 1D RoPE, 5 layers)
    // Takes pre-computed embeddings, outputs hidden states + logits
    // codebook_idx: which LM head to apply (-1 = no LM head, 0-14 = apply lm_head[idx])
    struct ggml_cgraph * build_code_pred_graph(int32_t n_tokens, int32_t n_past, int32_t codebook_idx = -1);

    // Initialize code predictor KV cache
    bool init_code_pred_cache(int32_t n_ctx);

    // Clear code predictor KV cache
    void clear_code_pred_cache();

    // Parse hyperparameters from GGUF
    bool parse_config(struct gguf_context * ctx);

    // Create tensor structures
    bool create_tensors(struct gguf_context * ctx, struct ggml_context * meta_ctx);

    // Load tensor data from file
    bool load_tensor_data(const std::string & path, struct gguf_context * ctx);

    Qwen3TalkerModel model_;
    Qwen3CodePredictorModel code_pred_;
    Qwen3TalkerState state_;
    Qwen3KVCache code_pred_cache_;
    std::string error_msg_;

    // Cleanup helpers
    void free_model();
    void free_kv_cache();
    void free_code_pred_cache();
};

} // namespace qwen3
