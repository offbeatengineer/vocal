#include "tts_model.h"
#include "tts_common.h"
#include "ggml.h"
#include "gguf.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstring>
#include <cstdio>
#include <iostream>

#define QWEN3_TALKER_MAX_NODES 8192

namespace qwen3 {

Qwen3TalkerLLM::Qwen3TalkerLLM() = default;

Qwen3TalkerLLM::~Qwen3TalkerLLM() {
    free_code_pred_cache();
    free_kv_cache();
    free_model();
}

void Qwen3TalkerLLM::free_model() {
    if (state_.sched) {
        ggml_backend_sched_free(state_.sched);
        state_.sched = nullptr;
    }
    if (state_.backend) {
        ggml_backend_free(state_.backend);
        state_.backend = nullptr;
    }
    if (model_.buffer) {
        ggml_backend_buffer_free(model_.buffer);
        model_.buffer = nullptr;
    }
    if (model_.ctx) {
        ggml_free(model_.ctx);
        model_.ctx = nullptr;
    }
    model_.tensors.clear();
    model_.layers.clear();
}

void Qwen3TalkerLLM::free_kv_cache() {
    if (state_.cache.buffer) {
        ggml_backend_buffer_free(state_.cache.buffer);
        state_.cache.buffer = nullptr;
    }
    if (state_.cache.ctx) {
        ggml_free(state_.cache.ctx);
        state_.cache.ctx = nullptr;
    }
    state_.cache.k_cache.clear();
    state_.cache.v_cache.clear();
    state_.cache.n_ctx = 0;
    state_.cache.n_used = 0;
}

bool Qwen3TalkerLLM::load_model(const std::string & model_path) {
    struct ggml_context * meta_ctx = nullptr;
    struct gguf_init_params params = {
        /*.no_alloc =*/ true,
        /*.ctx      =*/ &meta_ctx,
    };
    
    struct gguf_context * ctx = gguf_init_from_file(model_path.c_str(), params);
    if (!ctx) {
        error_msg_ = "Failed to open GGUF file: " + model_path;
        return false;
    }
    
    if (!parse_config(ctx)) {
        gguf_free(ctx);
        if (meta_ctx) ggml_free(meta_ctx);
        return false;
    }
    
    if (!create_tensors(ctx, meta_ctx)) {
        gguf_free(ctx);
        if (meta_ctx) ggml_free(meta_ctx);
        return false;
    }
    
    if (!load_tensor_data(model_path, ctx)) {
        free_model();
        gguf_free(ctx);
        if (meta_ctx) ggml_free(meta_ctx);
        return false;
    }
    
    gguf_free(ctx);
    if (meta_ctx) ggml_free(meta_ctx);
    
    // Initialize backends
    std::vector<ggml_backend_t> backends;

#ifdef GGML_USE_CUDA
    ggml_backend_t cuda_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    if (cuda_backend) {
        backends.push_back(cuda_backend);
    } else {
        std::cerr << "Warning: CUDA backend failed to init." << std::endl;
    }
#endif

    // Always add CPU backend last
    ggml_backend_t cpu_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!cpu_backend) {
        error_msg_ = "Failed to initialize CPU backend";
        return false;
    }
    backends.push_back(cpu_backend);
    
    // Store primary backend for simple access
    state_.backend = backends[0]; 
    
    // Create scheduler
    state_.sched = ggml_backend_sched_new(backends.data(), nullptr, backends.size(), QWEN3_TALKER_MAX_NODES, false, true);
    if (!state_.sched) {
        error_msg_ = "Failed to create backend scheduler";
        return false;
    }
    
    // Reserve space for compute meta
    state_.compute_meta.resize(ggml_tensor_overhead() * QWEN3_TALKER_MAX_NODES + ggml_graph_overhead());
    
    return true;
}

bool Qwen3TalkerLLM::parse_config(struct gguf_context * ctx) {
    auto get_u32 = [&](const char * key, int32_t default_val) -> int32_t {
        int64_t idx = gguf_find_key(ctx, key);
        if (idx < 0) return default_val;
        return (int32_t)gguf_get_val_u32(ctx, idx);
    };

    auto get_f32 = [&](const char * key, float default_val) -> float {
        int64_t idx = gguf_find_key(ctx, key);
        if (idx < 0) return default_val;
        return gguf_get_val_f32(ctx, idx);
    };

    auto & cfg = model_.config;
    // Keys from the GGUF use "qwen3-tts." prefix
    cfg.vocab_size = get_u32("qwen3-tts.text_vocab_size", get_u32("qwen3-tts.vocab_size", 151936));
    cfg.codec_vocab_size = get_u32("qwen3-tts.vocab_size", 3072);
    cfg.hidden_size = get_u32("qwen3-tts.embedding_length", 1024);
    cfg.text_hidden_size = get_u32("qwen3-tts.text_hidden_size", 2048);
    cfg.n_layers = get_u32("qwen3-tts.block_count", 28);
    cfg.n_heads = get_u32("qwen3-tts.attention.head_count", 16);
    cfg.n_kv_heads = get_u32("qwen3-tts.attention.head_count_kv", 8);
    cfg.intermediate_size = get_u32("qwen3-tts.feed_forward_length", 3072);
    cfg.head_dim = get_u32("qwen3-tts.attention.key_length", 128);
    cfg.rms_norm_eps = get_f32("qwen3-tts.attention.layer_norm_rms_epsilon", 1e-6f);
    cfg.rope_theta = get_f32("qwen3-tts.rope.freq_base", 1000000.0f);

    // Read MRoPE sections from GGUF (values are rotation pair counts, not raw dimensions)
    // e.g. [24, 20, 20] means actual dims [48, 40, 40] (×2 for cos/sin pairs)
    {
        int64_t idx = gguf_find_key(ctx, "qwen3-tts.rope.mrope_section");
        if (idx >= 0) {
            cfg.mrope_section.clear();
            size_t n = gguf_get_arr_n(ctx, idx);
            const int32_t * data = (const int32_t *)gguf_get_arr_data(ctx, idx);
            for (size_t i = 0; i < n; i++) {
                cfg.mrope_section.push_back(data[i]);
            }
        }
    }

    // Code predictor config
    auto & cp = code_pred_.config;
    cp.n_layers = get_u32("qwen3-tts.code_predictor.layer_count", 5);
    cp.vocab_size = get_u32("qwen3-tts.code_predictor.vocab_size", 2048);
    cp.num_code_groups = get_u32("qwen3-tts.num_code_groups", 16);
    // Code predictor may have its own hidden_size (1.7B: 1024 vs talker's 2048)
    cp.hidden_size = get_u32("qwen3-tts.code_predictor.hidden_size", cfg.hidden_size);
    cp.intermediate_size = get_u32("qwen3-tts.code_predictor.intermediate_size", cfg.intermediate_size);
    cp.n_heads = cfg.n_heads;
    cp.n_kv_heads = cfg.n_kv_heads;
    cp.head_dim = cfg.head_dim;
    cp.rms_norm_eps = cfg.rms_norm_eps;
    // Code predictor uses standard RoPE with theta=10000 (NOT MRoPE with 1M)
    cp.rope_theta = 10000.0f;
    // Input dimension = talker hidden_size (embeddings are in this space)
    cp.input_dim = cfg.hidden_size;
    cp.has_mtp_proj = (cp.input_dim != cp.hidden_size);

    return true;
}

bool Qwen3TalkerLLM::create_tensors(struct gguf_context * ctx, struct ggml_context * meta_ctx) {
    const int64_t n_tensors = gguf_get_n_tensors(ctx);
    const auto & cfg = model_.config;
    const auto & cp_cfg = code_pred_.config;

    const size_t ctx_size = n_tensors * ggml_tensor_overhead();
    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    model_.ctx = ggml_init(params);
    if (!model_.ctx) {
        error_msg_ = "Failed to create GGML context";
        return false;
    }

    model_.layers.resize(cfg.n_layers);

    // Initialize code predictor storage
    int n_codebooks = cp_cfg.num_code_groups - 1;  // 15 for 16 code groups
    code_pred_.layers.resize(cp_cfg.n_layers);
    code_pred_.codec_embd.resize(n_codebooks, nullptr);
    code_pred_.lm_head.resize(n_codebooks, nullptr);

    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(ctx, i);

        // Only load talker.* and code_pred.* tensors
        bool is_talker = (strncmp(name, "talker.", 7) == 0);
        bool is_code_pred = (strncmp(name, "code_pred.", 10) == 0);
        if (!is_talker && !is_code_pred) continue;

        struct ggml_tensor * tensor = ggml_get_tensor(model_.ctx, name);

        if (!tensor) {
            struct ggml_tensor * t_meta = ggml_get_tensor(meta_ctx, name);
            if (!t_meta) continue;

            tensor = ggml_new_tensor(model_.ctx, t_meta->type, ggml_n_dims(t_meta), t_meta->ne);
            ggml_set_name(tensor, name);
        }

        model_.tensors[name] = tensor;

        if (is_talker) {
            // Talker tensors
            if (strcmp(name, "talker.text_embd.weight") == 0) {
                model_.text_embd = tensor;
            } else if (strcmp(name, "talker.text_proj.fc1.weight") == 0) {
                model_.text_proj_fc1_w = tensor;
            } else if (strcmp(name, "talker.text_proj.fc1.bias") == 0) {
                model_.text_proj_fc1_b = tensor;
            } else if (strcmp(name, "talker.text_proj.fc2.weight") == 0) {
                model_.text_proj_fc2_w = tensor;
            } else if (strcmp(name, "talker.text_proj.fc2.bias") == 0) {
                model_.text_proj_fc2_b = tensor;
            } else if (strcmp(name, "talker.codec_embd.weight") == 0) {
                model_.codec_embd = tensor;
            } else if (strcmp(name, "talker.output_norm.weight") == 0) {
                model_.output_norm = tensor;
            } else if (strcmp(name, "talker.codec_head.weight") == 0) {
                model_.output = tensor;
            } else if (strstr(name, "talker.blk.")) {
                int layer_idx = -1;
                if (sscanf(name, "talker.blk.%d.", &layer_idx) == 1 &&
                    layer_idx >= 0 && layer_idx < cfg.n_layers) {
                    auto & layer = model_.layers[layer_idx];

                    if (strstr(name, "attn_q_norm.weight")) layer.attn_q_norm = tensor;
                    else if (strstr(name, "attn_k_norm.weight")) layer.attn_k_norm = tensor;
                    else if (strstr(name, "attn_norm.weight")) layer.attn_norm = tensor;
                    else if (strstr(name, "attn_output.weight")) layer.attn_output = tensor;
                    else if (strstr(name, "attn_q.weight")) layer.attn_q = tensor;
                    else if (strstr(name, "attn_k.weight")) layer.attn_k = tensor;
                    else if (strstr(name, "attn_v.weight")) layer.attn_v = tensor;
                    else if (strstr(name, "ffn_norm.weight")) layer.ffn_norm = tensor;
                    else if (strstr(name, "ffn_gate.weight")) layer.ffn_gate = tensor;
                    else if (strstr(name, "ffn_up.weight")) layer.ffn_up = tensor;
                    else if (strstr(name, "ffn_down.weight")) layer.ffn_down = tensor;
                }
            }
        } else {
            // Code predictor tensors
            if (strcmp(name, "code_pred.output_norm.weight") == 0) {
                code_pred_.output_norm = tensor;
            } else if (strcmp(name, "code_pred.mtp_proj.weight") == 0) {
                code_pred_.mtp_proj_w = tensor;
            } else if (strcmp(name, "code_pred.mtp_proj.bias") == 0) {
                code_pred_.mtp_proj_b = tensor;
            } else if (strstr(name, "code_pred.codec_embd.")) {
                int idx = -1;
                if (sscanf(name, "code_pred.codec_embd.%d.weight", &idx) == 1 &&
                    idx >= 0 && idx < n_codebooks) {
                    code_pred_.codec_embd[idx] = tensor;
                }
            } else if (strstr(name, "code_pred.lm_head.")) {
                int idx = -1;
                if (sscanf(name, "code_pred.lm_head.%d.weight", &idx) == 1 &&
                    idx >= 0 && idx < n_codebooks) {
                    code_pred_.lm_head[idx] = tensor;
                }
            } else if (strstr(name, "code_pred.blk.")) {
                int layer_idx = -1;
                if (sscanf(name, "code_pred.blk.%d.", &layer_idx) == 1 &&
                    layer_idx >= 0 && layer_idx < cp_cfg.n_layers) {
                    auto & layer = code_pred_.layers[layer_idx];

                    if (strstr(name, "attn_q_norm.weight")) layer.attn_q_norm = tensor;
                    else if (strstr(name, "attn_k_norm.weight")) layer.attn_k_norm = tensor;
                    else if (strstr(name, "attn_norm.weight")) layer.attn_norm = tensor;
                    else if (strstr(name, "attn_output.weight")) layer.attn_output = tensor;
                    else if (strstr(name, "attn_q.weight")) layer.attn_q = tensor;
                    else if (strstr(name, "attn_k.weight")) layer.attn_k = tensor;
                    else if (strstr(name, "attn_v.weight")) layer.attn_v = tensor;
                    else if (strstr(name, "ffn_norm.weight")) layer.ffn_norm = tensor;
                    else if (strstr(name, "ffn_gate.weight")) layer.ffn_gate = tensor;
                    else if (strstr(name, "ffn_up.weight")) layer.ffn_up = tensor;
                    else if (strstr(name, "ffn_down.weight")) layer.ffn_down = tensor;
                }
            }
        }
    }

    return true;
}

bool Qwen3TalkerLLM::load_tensor_data(const std::string & path, struct gguf_context * ctx) {
    // Need a temporary backend usage for allocation
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    // Actually we should use the state_.backend logic but that's not init yet.
    // Let's alloc on CPU for simplicity of loading, but if we want GPU, we should init it earlier.
    // NOTE: In load_model we init backend AFTER this. 
    // To support GPU offloading properly, we should init backend FIRST.
    // But let's assume CPU loading for now to be safe, or just move backend init up.
    // Actually, ggml_backend_alloc_ctx_tensors allocates memory on the backend.
    // If we want CUDA, we must provide CUDA backend here.
    
    // Re-doing backend init logic locally for loader
#ifdef GGML_USE_CUDA
    ggml_backend_t load_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    if (!load_backend) load_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
#else
    ggml_backend_t load_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
#endif

    if (!load_backend) return false;

    model_.buffer = ggml_backend_alloc_ctx_tensors(model_.ctx, load_backend);
    if (!model_.buffer) {
        ggml_backend_free(load_backend);
        return false;
    }
    
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        ggml_backend_free(load_backend);
        return false;
    }
    
    const size_t data_offset = gguf_get_data_offset(ctx);
    const int64_t n_tensors = gguf_get_n_tensors(ctx);
    std::vector<uint8_t> read_buf;
    
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(ctx, i);
        struct ggml_tensor * tensor = ggml_get_tensor(model_.ctx, name);
        if (!tensor) continue;
        
        size_t offset = gguf_get_tensor_offset(ctx, i);
        size_t nbytes = ggml_nbytes(tensor);
        
        // Direct read if backend supports it (CPU), else read to buffer
        if (ggml_backend_is_cpu(load_backend)) {
            fseek(f, data_offset + offset, SEEK_SET);
            fread(tensor->data, 1, nbytes, f);
        } else {
            // For GPU, read to host buf then copy
            if (read_buf.size() < nbytes) read_buf.resize(nbytes);
            fseek(f, data_offset + offset, SEEK_SET);
            fread(read_buf.data(), 1, nbytes, f);
            ggml_backend_tensor_set(tensor, read_buf.data(), 0, nbytes);
        }
    }
    
    fclose(f);
    ggml_backend_free(load_backend);
    
    return true;
}

bool Qwen3TalkerLLM::init_kv_cache(int32_t n_ctx) {
    const auto & cfg = model_.config;
    
    free_kv_cache();
    
    state_.cache.n_ctx = n_ctx;
    state_.cache.n_used = 0;
    state_.cache.head_dim = cfg.head_dim;
    state_.cache.n_kv_heads = cfg.n_kv_heads;
    state_.cache.n_layers = cfg.n_layers;
    
    const size_t n_tensors = cfg.n_layers * 2;
    const size_t ctx_size = n_tensors * ggml_tensor_overhead();
    
    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    
    state_.cache.ctx = ggml_init(params);
    
    state_.cache.k_cache.resize(cfg.n_layers);
    state_.cache.v_cache.resize(cfg.n_layers);
    
    for (int il = 0; il < cfg.n_layers; ++il) {
        state_.cache.k_cache[il] = ggml_new_tensor_3d(
            state_.cache.ctx, GGML_TYPE_F16, // Use F16 for KV cache efficiency
            state_.cache.head_dim, state_.cache.n_kv_heads, n_ctx);
        ggml_format_name(state_.cache.k_cache[il], "k_cache_%d", il);
        
        state_.cache.v_cache[il] = ggml_new_tensor_3d(
            state_.cache.ctx, GGML_TYPE_F16,
            state_.cache.head_dim, state_.cache.n_kv_heads, n_ctx);
        ggml_format_name(state_.cache.v_cache[il], "v_cache_%d", il);
    }
    
    state_.cache.buffer = ggml_backend_alloc_ctx_tensors(state_.cache.ctx, state_.backend);
    
    return true;
}

void Qwen3TalkerLLM::clear_kv_cache() {
    state_.cache.n_used = 0;
}

// embed_mode: 0=text, 1=codec, 2=external (pre-computed embeddings)
struct ggml_cgraph * Qwen3TalkerLLM::build_graph_internal(int32_t n_tokens, int32_t n_past, int embed_mode) {
    const auto & cfg = model_.config;
    struct ggml_init_params params = {
        /*.mem_size   =*/ state_.compute_meta.size(),
        /*.mem_buffer =*/ state_.compute_meta.data(),
        /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, QWEN3_TALKER_MAX_NODES, false);

    // Position IDs for MRoPE: [n_tokens * 4] (temporal, height, width, extra)
    struct ggml_tensor * inp_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens * 4);
    ggml_set_name(inp_pos, "inp_pos");
    ggml_set_input(inp_pos);

    // Embedding
    struct ggml_tensor * cur;
    if (embed_mode == 2) {
        // External pre-computed embeddings
        cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, cfg.hidden_size, n_tokens);
        ggml_set_name(cur, "inp_embeds");
        ggml_set_input(cur);
    } else {
        struct ggml_tensor * inp_tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
        ggml_set_name(inp_tokens, "inp_tokens");
        ggml_set_input(inp_tokens);

        if (embed_mode == 1) {
            cur = ggml_get_rows(ctx0, model_.codec_embd, inp_tokens);
        } else {
            cur = ggml_get_rows(ctx0, model_.text_embd, inp_tokens);
            cur = ggml_mul_mat(ctx0, model_.text_proj_fc1_w, cur);
            cur = ggml_add(ctx0, cur, model_.text_proj_fc1_b);
            cur = ggml_silu(ctx0, cur);
            cur = ggml_mul_mat(ctx0, model_.text_proj_fc2_w, cur);
            cur = ggml_add(ctx0, cur, model_.text_proj_fc2_b);
        }
    }
    
    const float eps = cfg.rms_norm_eps;
    const int n_head = cfg.n_heads;
    const int n_kv_head = cfg.n_kv_heads;
    const int head_dim = cfg.head_dim;
    const float rope_theta = cfg.rope_theta;
    
    const auto & mrope_section = cfg.mrope_section;
    
    for (int il = 0; il < cfg.n_layers; ++il) {
        // std::cout << "Building layer " << il << std::endl;
        const auto & layer = model_.layers[il];
        
        struct ggml_tensor * residual = cur;
        
        // Norm
        if (!layer.attn_norm) {
             std::cerr << "Error: attn_norm missing for layer " << il << std::endl;
             return nullptr;
        }
        cur = ggml_rms_norm(ctx0, cur, eps);
        cur = ggml_mul(ctx0, cur, layer.attn_norm);
        
        // QKV
        struct ggml_tensor * Qcur = ggml_mul_mat(ctx0, layer.attn_q, cur);
        struct ggml_tensor * Kcur = ggml_mul_mat(ctx0, layer.attn_k, cur);
        struct ggml_tensor * Vcur = ggml_mul_mat(ctx0, layer.attn_v, cur);

        // Reshape [N, head_dim * n_head] -> [head_dim, n_head, N]
        Qcur = ggml_cont(ctx0, ggml_reshape_3d(ctx0, Qcur, head_dim, n_head, n_tokens));
        Kcur = ggml_cont(ctx0, ggml_reshape_3d(ctx0, Kcur, head_dim, n_kv_head, n_tokens));
        Vcur = ggml_cont(ctx0, ggml_reshape_3d(ctx0, Vcur, head_dim, n_kv_head, n_tokens));

        // QK norm (Qwen3 feature)
        if (layer.attn_q_norm) {
            Qcur = ggml_rms_norm(ctx0, Qcur, eps);
            Qcur = ggml_mul(ctx0, Qcur, layer.attn_q_norm);
        }
        if (layer.attn_k_norm) {
            Kcur = ggml_rms_norm(ctx0, Kcur, eps);
            Kcur = ggml_mul(ctx0, Kcur, layer.attn_k_norm);
        }
        
        // Multimodal RoPE using ggml_rope_multi (GGML_ROPE_TYPE_MROPE)
        // sections = [temporal, height, width, extra] (rotation pair counts, NOT doubled)
        // Position tensor is [n_tokens * 4] with layout: [temporal, height, width, extra]
        int sections[4] = {mrope_section[0], mrope_section[1], mrope_section[2], 0};

        Qcur = ggml_rope_multi(ctx0, Qcur, inp_pos, nullptr,
                               head_dim, sections, GGML_ROPE_TYPE_MROPE, 0,
                               rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        Kcur = ggml_rope_multi(ctx0, Kcur, inp_pos, nullptr,
                               head_dim, sections, GGML_ROPE_TYPE_MROPE, 0,
                               rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        
        // KV Cache Update
        struct ggml_tensor * k_cache = state_.cache.k_cache[il];
        struct ggml_tensor * v_cache = state_.cache.v_cache[il];
        
        // view into cache for current batch
        struct ggml_tensor * k_cache_view = ggml_view_3d(ctx0, k_cache, 
            head_dim, n_kv_head, n_tokens, 
            k_cache->nb[1], k_cache->nb[2], 
            n_past * k_cache->nb[2]);
            
        struct ggml_tensor * v_cache_view = ggml_view_3d(ctx0, v_cache, 
            head_dim, n_kv_head, n_tokens, 
            v_cache->nb[1], v_cache->nb[2], 
            n_past * v_cache->nb[2]);
            
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, Kcur, k_cache_view));
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, Vcur, v_cache_view));
        
        // Attention
        // View full context
        int n_ctx_curr = n_past + n_tokens;
        struct ggml_tensor * K = ggml_view_3d(ctx0, k_cache, head_dim, n_kv_head, n_ctx_curr, k_cache->nb[1], k_cache->nb[2], 0);
        struct ggml_tensor * V = ggml_view_3d(ctx0, v_cache, head_dim, n_kv_head, n_ctx_curr, v_cache->nb[1], v_cache->nb[2], 0);
        
        // Permute Q: [head_dim, n_head, N] -> [head_dim, N, n_head]
        // Actually typical ggml attention:
        // Q: [head_dim, N, n_head]
        // K: [head_dim, n_ctx, n_kv_head]
        struct ggml_tensor * Q = ggml_permute(ctx0, Qcur, 0, 2, 1, 3);
        K = ggml_permute(ctx0, K, 0, 2, 1, 3);
        V = ggml_permute(ctx0, V, 0, 2, 1, 3);
        
        // K * Q
        struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q); // [n_ctx, N, n_head]
        
        // Scale
        KQ = ggml_scale(ctx0, KQ, 1.0f / sqrtf(float(head_dim)));
        
        // Mask (Causal)
        KQ = ggml_diag_mask_inf(ctx0, KQ, n_past);
        
        // Softmax
        KQ = ggml_soft_max(ctx0, KQ);
        
        // V * KQ
        // V needs to be contiguous/transposed properly? 
        // ggml_mul_mat V: [head_dim, n_ctx, n_head] * KQ: [n_ctx, N, n_head] -> [head_dim, N, n_head]
        V = ggml_cont(ctx0, ggml_transpose(ctx0, V));
        struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V, KQ);
        
        // Permute back: [head_dim, N, n_head] -> [head_dim, n_head, N] -> flatten to [hidden, N]
        KQV = ggml_permute(ctx0, KQV, 0, 2, 1, 3);
        cur = ggml_cont_2d(ctx0, KQV, head_dim * n_head, n_tokens);
        
        // Output proj
        cur = ggml_mul_mat(ctx0, layer.attn_output, cur);
        
        // Residual
        cur = ggml_add(ctx0, cur, residual);
        
        // FFN
        residual = cur;
        cur = ggml_rms_norm(ctx0, cur, eps);
        cur = ggml_mul(ctx0, cur, layer.ffn_norm);
        
        struct ggml_tensor * gate = ggml_mul_mat(ctx0, layer.ffn_gate, cur);
        struct ggml_tensor * up = ggml_mul_mat(ctx0, layer.ffn_up, cur);
        
        gate = ggml_silu(ctx0, gate);
        cur = ggml_mul(ctx0, gate, up);
        // Cast ffn_down to F32 to avoid Metal F16 overflow when gateup > 65504
        cur = ggml_mul_mat(ctx0, ggml_cast(ctx0, layer.ffn_down, GGML_TYPE_F32), cur);

        cur = ggml_add(ctx0, cur, residual);
    }
    
    // Output Norm
    cur = ggml_rms_norm(ctx0, cur, eps);
    cur = ggml_mul(ctx0, cur, model_.output_norm);

    // Mark hidden states as output (for code predictor)
    ggml_set_name(cur, "hidden");
    ggml_set_output(cur);

    // LM Head
    cur = ggml_mul_mat(ctx0, model_.output, cur);
    ggml_set_name(cur, "logits");
    ggml_set_output(cur);
    
    ggml_build_forward_expand(gf, cur);
    
    ggml_free(ctx0);
    return gf;
}

struct ggml_cgraph * Qwen3TalkerLLM::build_graph(const int32_t * tokens, const int32_t * pos_ids, int32_t n_tokens, int32_t n_past, bool is_codec) {
    (void)tokens; (void)pos_ids;
    return build_graph_internal(n_tokens, n_past, is_codec ? 1 : 0);
}

struct ggml_cgraph * Qwen3TalkerLLM::build_graph_embeds(int32_t n_tokens, int32_t n_past) {
    return build_graph_internal(n_tokens, n_past, 2);
}

bool Qwen3TalkerLLM::forward(const int32_t * tokens, const int32_t * pos_ids, int32_t n_tokens, int32_t n_past,
                             bool is_codec, std::vector<float> & output,
                             std::vector<float> * hidden_out) {
    if (!model_.ctx) return false;
    if (state_.cache.n_ctx == 0) init_kv_cache(4096);

    struct ggml_cgraph * gf = build_graph(tokens, pos_ids, n_tokens, n_past, is_codec);

    if (!ggml_backend_sched_alloc_graph(state_.sched, gf)) {
        std::cerr << "Failed to allocate graph" << std::endl;
        return false;
    }

    struct ggml_tensor * inp_tokens = ggml_graph_get_tensor(gf, "inp_tokens");
    ggml_backend_tensor_set(inp_tokens, tokens, 0, n_tokens * sizeof(int32_t));

    struct ggml_tensor * inp_pos = ggml_graph_get_tensor(gf, "inp_pos");
    ggml_backend_tensor_set(inp_pos, pos_ids, 0, 4 * n_tokens * sizeof(int32_t));

    if (ggml_backend_sched_graph_compute(state_.sched, gf) != GGML_STATUS_SUCCESS) {
        std::cerr << "Graph compute failed" << std::endl;
        return false;
    }

    struct ggml_tensor * logits = ggml_graph_get_tensor(gf, "logits");
    output.resize(n_tokens * model_.config.codec_vocab_size);
    ggml_backend_tensor_get(logits, output.data(), 0, output.size() * sizeof(float));

    if (hidden_out) {
        struct ggml_tensor * hidden = ggml_graph_get_tensor(gf, "hidden");
        if (hidden) {
            hidden_out->resize(n_tokens * model_.config.hidden_size);
            ggml_backend_tensor_get(hidden, hidden_out->data(), 0, hidden_out->size() * sizeof(float));
        }
    }

    state_.cache.n_used = n_past + n_tokens;
    ggml_backend_sched_reset(state_.sched);

    return true;
}

#if 0 // Old duplicated build_graph_embeds - replaced by build_graph_internal
struct ggml_cgraph * Qwen3TalkerLLM_OLD_build_graph_embeds(int32_t n_tokens, int32_t n_past) {
    const auto & cfg = model_.config;
    struct ggml_init_params params = {
        /*.mem_size   =*/ state_.compute_meta.size(),
        /*.mem_buffer =*/ state_.compute_meta.data(),
        /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, QWEN3_TALKER_MAX_NODES, false);

    // Pre-computed embeddings input: [hidden_size, n_tokens] (GGML is column-major)
    struct ggml_tensor * cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, cfg.hidden_size, n_tokens);
    ggml_set_name(cur, "inp_embeds");
    ggml_set_input(cur);

    // Position IDs: [n_tokens, 3]
    struct ggml_tensor * inp_pos = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_tokens, 3);
    ggml_set_name(inp_pos, "inp_pos");
    ggml_set_input(inp_pos);

    // Same transformer body as build_graph (from here on)
    const float eps = cfg.rms_norm_eps;
    const int n_head = cfg.n_heads;
    const int n_kv_head = cfg.n_kv_heads;
    const int head_dim = cfg.head_dim;
    const float rope_theta = cfg.rope_theta;
    const auto & mrope_section = cfg.mrope_section;

    for (int il = 0; il < cfg.n_layers; ++il) {
        const auto & layer = model_.layers[il];

        struct ggml_tensor * residual = cur;

        // Attention Norm
        cur = ggml_rms_norm(ctx0, cur, eps);
        cur = ggml_mul(ctx0, cur, layer.attn_norm);

        // Q, K, V
        struct ggml_tensor * Qcur = ggml_mul_mat(ctx0, layer.attn_q, cur);
        struct ggml_tensor * Kcur = ggml_mul_mat(ctx0, layer.attn_k, cur);
        struct ggml_tensor * Vcur = ggml_mul_mat(ctx0, layer.attn_v, cur);

        // Reshape to [head_dim, n_head/n_kv_head, n_tokens]
        Qcur = ggml_cont(ctx0, ggml_reshape_3d(ctx0, Qcur, head_dim, n_head, n_tokens));
        Kcur = ggml_cont(ctx0, ggml_reshape_3d(ctx0, Kcur, head_dim, n_kv_head, n_tokens));
        Vcur = ggml_cont(ctx0, ggml_reshape_3d(ctx0, Vcur, head_dim, n_kv_head, n_tokens));

        // QK Norm
        if (layer.attn_q_norm && layer.attn_k_norm) {
            Qcur = ggml_rms_norm(ctx0, Qcur, eps);
            Qcur = ggml_mul(ctx0, Qcur, layer.attn_q_norm);

            Kcur = ggml_rms_norm(ctx0, Kcur, eps);
            Kcur = ggml_mul(ctx0, Kcur, layer.attn_k_norm);
        }

        // Multimodal RoPE
        int sec0 = mrope_section[0], sec1 = mrope_section[1], sec2 = mrope_section[2];
        struct ggml_tensor * pos_temporal = ggml_view_1d(ctx0, inp_pos, n_tokens, 0);
        struct ggml_tensor * pos_height  = ggml_view_1d(ctx0, inp_pos, n_tokens, n_tokens * sizeof(int32_t));
        struct ggml_tensor * pos_width   = ggml_view_1d(ctx0, inp_pos, n_tokens, 2 * n_tokens * sizeof(int32_t));

        struct ggml_tensor * Q_parts[3], * K_parts[3];
        int offset = 0;
        struct ggml_tensor * positions[3] = {pos_temporal, pos_height, pos_width};
        int sections[3] = {sec0, sec1, sec2};
        for (int s = 0; s < 3; s++) {
            int sec_dim = sections[s] * 2;
            struct ggml_tensor * Qs = ggml_view_3d(ctx0, Qcur, sec_dim, n_head, n_tokens,
                                                     Qcur->nb[1], Qcur->nb[2], offset * sizeof(float));
            struct ggml_tensor * Ks = ggml_view_3d(ctx0, Kcur, sec_dim, n_kv_head, n_tokens,
                                                     Kcur->nb[1], Kcur->nb[2], offset * sizeof(float));
            Qs = ggml_cont(ctx0, Qs);
            Ks = ggml_cont(ctx0, Ks);
            Qs = ggml_rope_ext(ctx0, Qs, positions[s], nullptr, sec_dim, 0, 0, n_tokens, rope_theta, 1.0f, 1.0f, 0.0f, 1.0f);
            Ks = ggml_rope_ext(ctx0, Ks, positions[s], nullptr, sec_dim, 0, 0, n_tokens, rope_theta, 1.0f, 1.0f, 0.0f, 1.0f);
            Q_parts[s] = Qs;
            K_parts[s] = Ks;
            offset += sec_dim;
        }

        Qcur = ggml_concat(ctx0, Q_parts[0], Q_parts[1], 0);
        Qcur = ggml_concat(ctx0, Qcur, Q_parts[2], 0);
        Kcur = ggml_concat(ctx0, K_parts[0], K_parts[1], 0);
        Kcur = ggml_concat(ctx0, Kcur, K_parts[2], 0);

        // KV Cache
        {
            struct ggml_tensor * k_cache = state_.cache.k_cache[il];
            struct ggml_tensor * v_cache = state_.cache.v_cache[il];
            struct ggml_tensor * Kview = ggml_view_3d(ctx0, k_cache,
                head_dim, n_kv_head, n_tokens,
                k_cache->nb[1], k_cache->nb[2],
                n_past * head_dim * ggml_type_size(k_cache->type));
            ggml_build_forward_expand(gf, ggml_cpy(ctx0, ggml_cont(ctx0, Kcur), Kview));
            struct ggml_tensor * Vview = ggml_view_3d(ctx0, v_cache,
                head_dim, n_kv_head, n_tokens,
                v_cache->nb[1], v_cache->nb[2],
                n_past * head_dim * ggml_type_size(v_cache->type));
            ggml_build_forward_expand(gf, ggml_cpy(ctx0, ggml_cont(ctx0, Vcur), Vview));

            int n_ctx = n_past + n_tokens;
            struct ggml_tensor * K = ggml_view_3d(ctx0, k_cache,
                head_dim, n_kv_head, n_ctx,
                k_cache->nb[1], k_cache->nb[2], 0);
            struct ggml_tensor * V = ggml_view_3d(ctx0, v_cache,
                head_dim, n_kv_head, n_ctx,
                v_cache->nb[1], v_cache->nb[2], 0);

            // GQA repeat
            if (n_kv_head < n_head) {
                int rep = n_head / n_kv_head;
                K = ggml_cont(ctx0, K);
                V = ggml_cont(ctx0, V);
                K = ggml_reshape_4d(ctx0, K, head_dim, 1, n_kv_head, n_ctx);
                K = ggml_repeat(ctx0, K, ggml_new_tensor_4d(ctx0, K->type, head_dim, rep, n_kv_head, n_ctx));
                K = ggml_reshape_3d(ctx0, K, head_dim, n_head, n_ctx);
                V = ggml_reshape_4d(ctx0, V, head_dim, 1, n_kv_head, n_ctx);
                V = ggml_repeat(ctx0, V, ggml_new_tensor_4d(ctx0, V->type, head_dim, rep, n_kv_head, n_ctx));
                V = ggml_reshape_3d(ctx0, V, head_dim, n_head, n_ctx);
            }

            struct ggml_tensor * Q = ggml_permute(ctx0, Qcur, 0, 2, 1, 3);
            K = ggml_permute(ctx0, K, 0, 2, 1, 3);
            V = ggml_permute(ctx0, V, 0, 2, 1, 3);

            struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);
            KQ = ggml_scale(ctx0, KQ, 1.0f / sqrtf(float(head_dim)));
            KQ = ggml_diag_mask_inf(ctx0, KQ, n_past);
            KQ = ggml_soft_max(ctx0, KQ);

            V = ggml_cont(ctx0, ggml_transpose(ctx0, V));
            struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V, KQ);

            KQV = ggml_permute(ctx0, KQV, 0, 2, 1, 3);
            cur = ggml_cont_2d(ctx0, KQV, head_dim * n_head, n_tokens);
        }

        cur = ggml_mul_mat(ctx0, layer.attn_output, cur);
        cur = ggml_add(ctx0, cur, residual);

        // FFN
        residual = cur;
        cur = ggml_rms_norm(ctx0, cur, eps);
        cur = ggml_mul(ctx0, cur, layer.ffn_norm);

        struct ggml_tensor * gate = ggml_mul_mat(ctx0, layer.ffn_gate, cur);
        struct ggml_tensor * up = ggml_mul_mat(ctx0, layer.ffn_up, cur);

        gate = ggml_silu(ctx0, gate);
        cur = ggml_mul(ctx0, gate, up);
        // Cast ffn_down to F32 to avoid Metal F16 overflow when gateup > 65504
        cur = ggml_mul_mat(ctx0, ggml_cast(ctx0, layer.ffn_down, GGML_TYPE_F32), cur);

        cur = ggml_add(ctx0, cur, residual);
    }

    // Output Norm
    cur = ggml_rms_norm(ctx0, cur, eps);
    cur = ggml_mul(ctx0, cur, model_.output_norm);

    ggml_set_name(cur, "hidden");
    ggml_set_output(cur);

    // LM Head
    cur = ggml_mul_mat(ctx0, model_.output, cur);
    ggml_set_name(cur, "logits");
    ggml_set_output(cur);

    ggml_build_forward_expand(gf, cur);

    ggml_free(ctx0);
    return gf;
}
#endif

bool Qwen3TalkerLLM::forward_embeds(const float * embeds, const int32_t * pos_ids, int32_t n_tokens, int32_t n_past,
                                     std::vector<float> & output,
                                     std::vector<float> * hidden_out) {
    if (!model_.ctx) return false;
    if (state_.cache.n_ctx == 0) init_kv_cache(4096);

    struct ggml_cgraph * gf = build_graph_embeds(n_tokens, n_past);

    if (!ggml_backend_sched_alloc_graph(state_.sched, gf)) {
        std::cerr << "Failed to allocate graph (embeds)" << std::endl;
        return false;
    }

    // Set inputs
    struct ggml_tensor * inp_embeds = ggml_graph_get_tensor(gf, "inp_embeds");
    ggml_backend_tensor_set(inp_embeds, embeds, 0, n_tokens * model_.config.hidden_size * sizeof(float));

    struct ggml_tensor * inp_pos = ggml_graph_get_tensor(gf, "inp_pos");
    ggml_backend_tensor_set(inp_pos, pos_ids, 0, 4 * n_tokens * sizeof(int32_t));

    // Compute
    if (ggml_backend_sched_graph_compute(state_.sched, gf) != GGML_STATUS_SUCCESS) {
        std::cerr << "Graph compute failed (embeds)" << std::endl;
        return false;
    }

    // Output logits
    struct ggml_tensor * logits = ggml_graph_get_tensor(gf, "logits");
    output.resize(n_tokens * model_.config.codec_vocab_size);
    ggml_backend_tensor_get(logits, output.data(), 0, output.size() * sizeof(float));

    if (hidden_out) {
        struct ggml_tensor * hidden = ggml_graph_get_tensor(gf, "hidden");
        if (hidden) {
            hidden_out->resize(n_tokens * model_.config.hidden_size);
            ggml_backend_tensor_get(hidden, hidden_out->data(), 0, hidden_out->size() * sizeof(float));
        }
    }

    state_.cache.n_used = n_past + n_tokens;
    ggml_backend_sched_reset(state_.sched);

    return true;
}

// ============================================================
// Code Predictor
// ============================================================

bool Qwen3TalkerLLM::init_code_pred_cache(int32_t n_ctx) {
    const auto & cp = code_pred_.config;

    free_code_pred_cache();

    code_pred_cache_.n_ctx = n_ctx;
    code_pred_cache_.n_used = 0;
    code_pred_cache_.head_dim = cp.head_dim;
    code_pred_cache_.n_kv_heads = cp.n_kv_heads;
    code_pred_cache_.n_layers = cp.n_layers;

    const size_t n_tensors = cp.n_layers * 2;
    const size_t ctx_size = n_tensors * ggml_tensor_overhead();

    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    code_pred_cache_.ctx = ggml_init(params);

    code_pred_cache_.k_cache.resize(cp.n_layers);
    code_pred_cache_.v_cache.resize(cp.n_layers);

    for (int il = 0; il < cp.n_layers; ++il) {
        code_pred_cache_.k_cache[il] = ggml_new_tensor_3d(
            code_pred_cache_.ctx, GGML_TYPE_F16,
            cp.head_dim, cp.n_kv_heads, n_ctx);
        ggml_format_name(code_pred_cache_.k_cache[il], "cp_k_cache_%d", il);

        code_pred_cache_.v_cache[il] = ggml_new_tensor_3d(
            code_pred_cache_.ctx, GGML_TYPE_F16,
            cp.head_dim, cp.n_kv_heads, n_ctx);
        ggml_format_name(code_pred_cache_.v_cache[il], "cp_v_cache_%d", il);
    }

    code_pred_cache_.buffer = ggml_backend_alloc_ctx_tensors(code_pred_cache_.ctx, state_.backend);

    return true;
}

void Qwen3TalkerLLM::clear_code_pred_cache() {
    code_pred_cache_.n_used = 0;
}

void Qwen3TalkerLLM::free_code_pred_cache() {
    if (code_pred_cache_.buffer) {
        ggml_backend_buffer_free(code_pred_cache_.buffer);
        code_pred_cache_.buffer = nullptr;
    }
    if (code_pred_cache_.ctx) {
        ggml_free(code_pred_cache_.ctx);
        code_pred_cache_.ctx = nullptr;
    }
    code_pred_cache_.k_cache.clear();
    code_pred_cache_.v_cache.clear();
    code_pred_cache_.n_ctx = 0;
    code_pred_cache_.n_used = 0;
}

// Build code predictor graph: 5-layer Qwen3 transformer with standard 1D RoPE
// Input: pre-computed embeddings [hidden_size, n_tokens]
// Output: hidden states + optionally logits from lm_head[codebook_idx]
struct ggml_cgraph * Qwen3TalkerLLM::build_code_pred_graph(int32_t n_tokens, int32_t n_past, int32_t codebook_idx) {
    const auto & cp = code_pred_.config;
    struct ggml_init_params params = {
        /*.mem_size   =*/ state_.compute_meta.size(),
        /*.mem_buffer =*/ state_.compute_meta.data(),
        /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, QWEN3_TALKER_MAX_NODES, false);

    // Position IDs: standard 1D (NOT MRoPE)
    struct ggml_tensor * inp_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(inp_pos, "cp_pos");
    ggml_set_input(inp_pos);

    // Pre-computed embeddings input (input_dim may differ from hidden_size in 1.7B)
    struct ggml_tensor * cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, cp.input_dim, n_tokens);
    ggml_set_name(cur, "cp_embeds");
    ggml_set_input(cur);

    // Input projection: input_dim → hidden_size (1.7B only)
    if (cp.has_mtp_proj && code_pred_.mtp_proj_w) {
        cur = ggml_mul_mat(ctx0, code_pred_.mtp_proj_w, cur);
        if (code_pred_.mtp_proj_b) {
            cur = ggml_add(ctx0, cur, code_pred_.mtp_proj_b);
        }
    }

    const float eps = cp.rms_norm_eps;
    const int n_head = cp.n_heads;
    const int n_kv_head = cp.n_kv_heads;
    const int head_dim = cp.head_dim;
    const float rope_theta = cp.rope_theta;

    for (int il = 0; il < cp.n_layers; ++il) {
        const auto & layer = code_pred_.layers[il];

        struct ggml_tensor * residual = cur;

        // Attention norm
        cur = ggml_rms_norm(ctx0, cur, eps);
        cur = ggml_mul(ctx0, cur, layer.attn_norm);

        // QKV
        struct ggml_tensor * Qcur = ggml_mul_mat(ctx0, layer.attn_q, cur);
        struct ggml_tensor * Kcur = ggml_mul_mat(ctx0, layer.attn_k, cur);
        struct ggml_tensor * Vcur = ggml_mul_mat(ctx0, layer.attn_v, cur);

        Qcur = ggml_cont(ctx0, ggml_reshape_3d(ctx0, Qcur, head_dim, n_head, n_tokens));
        Kcur = ggml_cont(ctx0, ggml_reshape_3d(ctx0, Kcur, head_dim, n_kv_head, n_tokens));
        Vcur = ggml_cont(ctx0, ggml_reshape_3d(ctx0, Vcur, head_dim, n_kv_head, n_tokens));

        // QK norm
        if (layer.attn_q_norm) {
            Qcur = ggml_rms_norm(ctx0, Qcur, eps);
            Qcur = ggml_mul(ctx0, Qcur, layer.attn_q_norm);
        }
        if (layer.attn_k_norm) {
            Kcur = ggml_rms_norm(ctx0, Kcur, eps);
            Kcur = ggml_mul(ctx0, Kcur, layer.attn_k_norm);
        }

        // Standard 1D RoPE with NeoX-style rotation, rope_theta=10000
        Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr,
                             head_dim, GGML_ROPE_TYPE_NEOX, 32768, rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr,
                             head_dim, GGML_ROPE_TYPE_NEOX, 32768, rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        // KV Cache
        struct ggml_tensor * k_cache = code_pred_cache_.k_cache[il];
        struct ggml_tensor * v_cache = code_pred_cache_.v_cache[il];

        struct ggml_tensor * k_cache_view = ggml_view_3d(ctx0, k_cache,
            head_dim, n_kv_head, n_tokens,
            k_cache->nb[1], k_cache->nb[2],
            n_past * k_cache->nb[2]);
        struct ggml_tensor * v_cache_view = ggml_view_3d(ctx0, v_cache,
            head_dim, n_kv_head, n_tokens,
            v_cache->nb[1], v_cache->nb[2],
            n_past * v_cache->nb[2]);

        ggml_build_forward_expand(gf, ggml_cpy(ctx0, Kcur, k_cache_view));
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, Vcur, v_cache_view));

        // Attention
        int n_ctx_curr = n_past + n_tokens;
        struct ggml_tensor * K = ggml_view_3d(ctx0, k_cache, head_dim, n_kv_head, n_ctx_curr,
                                              k_cache->nb[1], k_cache->nb[2], 0);
        struct ggml_tensor * V = ggml_view_3d(ctx0, v_cache, head_dim, n_kv_head, n_ctx_curr,
                                              v_cache->nb[1], v_cache->nb[2], 0);

        struct ggml_tensor * Q = ggml_permute(ctx0, Qcur, 0, 2, 1, 3);
        K = ggml_permute(ctx0, K, 0, 2, 1, 3);
        V = ggml_permute(ctx0, V, 0, 2, 1, 3);

        struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);
        KQ = ggml_scale(ctx0, KQ, 1.0f / sqrtf(float(head_dim)));
        KQ = ggml_diag_mask_inf(ctx0, KQ, n_past);
        KQ = ggml_soft_max(ctx0, KQ);

        V = ggml_cont(ctx0, ggml_transpose(ctx0, V));
        struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V, KQ);

        KQV = ggml_permute(ctx0, KQV, 0, 2, 1, 3);
        cur = ggml_cont_2d(ctx0, KQV, head_dim * n_head, n_tokens);

        // Output proj
        cur = ggml_mul_mat(ctx0, layer.attn_output, cur);
        cur = ggml_add(ctx0, cur, residual);

        // FFN
        residual = cur;
        cur = ggml_rms_norm(ctx0, cur, eps);
        cur = ggml_mul(ctx0, cur, layer.ffn_norm);

        struct ggml_tensor * gate = ggml_mul_mat(ctx0, layer.ffn_gate, cur);
        struct ggml_tensor * up = ggml_mul_mat(ctx0, layer.ffn_up, cur);

        gate = ggml_silu(ctx0, gate);
        cur = ggml_mul(ctx0, gate, up);

        // Cast ffn_down to F32 to avoid Metal F16 overflow when gateup > 65504
        cur = ggml_mul_mat(ctx0, ggml_cast(ctx0, layer.ffn_down, GGML_TYPE_F32), cur);

        cur = ggml_add(ctx0, cur, residual);
    }

    // Output norm
    cur = ggml_rms_norm(ctx0, cur, eps);
    cur = ggml_mul(ctx0, cur, code_pred_.output_norm);

    ggml_set_name(cur, "cp_hidden");
    ggml_set_output(cur);

    // Apply LM head on GPU if codebook_idx specified
    if (codebook_idx >= 0 && codebook_idx < (int32_t)code_pred_.lm_head.size()) {
        struct ggml_tensor * logits = ggml_mul_mat(ctx0, code_pred_.lm_head[codebook_idx], cur);
        ggml_set_name(logits, "cp_logits");
        ggml_set_output(logits);
        ggml_build_forward_expand(gf, logits);
    } else {
        ggml_build_forward_expand(gf, cur);
    }

    ggml_free(ctx0);
    return gf;
}

bool Qwen3TalkerLLM::predict_codes(const float * talker_hidden, int32_t code_0,
                                     std::vector<int32_t> & out_codes) {
    const auto & cp = code_pred_.config;
    const int D = cp.input_dim;     // embedding dimension (= talker hidden_size)
    const int H = cp.hidden_size;   // transformer internal dimension
    const int V = cp.vocab_size;    // 2048
    const int n_codebooks = cp.num_code_groups - 1;  // 15

    out_codes.resize(n_codebooks);

    // Initialize code predictor KV cache if needed
    if (code_pred_cache_.n_ctx == 0) {
        init_code_pred_cache(64);  // max 2 + 15 = 17 tokens per step
    }
    clear_code_pred_cache();

    // Step 1: Prefill with [talker_hidden_state, talker_codec_embd(code_0)]
    // Include lm_head[0] in the graph to compute logits on GPU
    std::vector<float> prefill_embeds(2 * D);
    memcpy(prefill_embeds.data(), talker_hidden, D * sizeof(float));
    compute_codec_embedding(code_0, prefill_embeds.data() + D);

    int32_t prefill_pos[2] = {0, 1};

    struct ggml_cgraph * gf = build_code_pred_graph(2, 0, 0);  // codebook_idx=0
    if (!ggml_backend_sched_alloc_graph(state_.sched, gf)) {
        std::cerr << "Code predictor: failed to allocate prefill graph" << std::endl;
        return false;
    }

    struct ggml_tensor * cp_embeds = ggml_graph_get_tensor(gf, "cp_embeds");
    ggml_backend_tensor_set(cp_embeds, prefill_embeds.data(), 0, 2 * D * sizeof(float));

    struct ggml_tensor * cp_pos = ggml_graph_get_tensor(gf, "cp_pos");
    ggml_backend_tensor_set(cp_pos, prefill_pos, 0, 2 * sizeof(int32_t));

    if (ggml_backend_sched_graph_compute(state_.sched, gf) != GGML_STATUS_SUCCESS) {
        std::cerr << "Code predictor: prefill compute failed" << std::endl;
        return false;
    }

    // Read logits from GPU (only last token's logits: V floats = 8KB)
    struct ggml_tensor * cp_logits = ggml_graph_get_tensor(gf, "cp_logits");
    std::vector<float> logits_buf(V);
    // Last token's logits at offset V (token 1 of 2)
    ggml_backend_tensor_get(cp_logits, logits_buf.data(), V * sizeof(float), V * sizeof(float));

    // Greedy argmax for code predictor (no need for fancy sampling here)
    auto argmax = [](const float * data, int n) -> int32_t {
        int32_t best = 0;
        float best_val = data[0];
        for (int i = 1; i < n; i++) {
            if (data[i] > best_val) { best_val = data[i]; best = i; }
        }
        return best;
    };

    out_codes[0] = argmax(logits_buf.data(), V);

    code_pred_cache_.n_used = 2;
    ggml_backend_sched_reset(state_.sched);

    // Step 2: Autoregressive generation for codebooks 2-15
    int32_t n_past = 2;
    for (int i = 1; i < n_codebooks; i++) {
        // Embed previous code with code_pred.codec_embd[i-1]
        std::vector<float> step_embed(D);
        compute_code_pred_embedding(i - 1, out_codes[i - 1], step_embed.data());

        int32_t step_pos = n_past;

        // Build graph with lm_head[i] included
        gf = build_code_pred_graph(1, n_past, i);
        if (!ggml_backend_sched_alloc_graph(state_.sched, gf)) {
            std::cerr << "Code predictor: failed to allocate step graph" << std::endl;
            return false;
        }

        cp_embeds = ggml_graph_get_tensor(gf, "cp_embeds");
        ggml_backend_tensor_set(cp_embeds, step_embed.data(), 0, D * sizeof(float));

        cp_pos = ggml_graph_get_tensor(gf, "cp_pos");
        ggml_backend_tensor_set(cp_pos, &step_pos, 0, sizeof(int32_t));

        if (ggml_backend_sched_graph_compute(state_.sched, gf) != GGML_STATUS_SUCCESS) {
            std::cerr << "Code predictor: step compute failed at codebook " << i << std::endl;
            return false;
        }

        // Read logits from GPU (V floats = 8KB — trivial transfer)
        cp_logits = ggml_graph_get_tensor(gf, "cp_logits");
        ggml_backend_tensor_get(cp_logits, logits_buf.data(), 0, V * sizeof(float));

        out_codes[i] = argmax(logits_buf.data(), V);

        n_past++;
        ggml_backend_sched_reset(state_.sched);
    }

    code_pred_cache_.n_used = n_past;

    return true;
}

void Qwen3TalkerLLM::compute_code_pred_embedding(int32_t codebook_idx, int32_t token_id, float * out) const {
    const int D = code_pred_.config.input_dim;  // embedding dim (= talker hidden_size)

    std::vector<ggml_fp16_t> row_f16(D);
    ggml_backend_tensor_get(code_pred_.codec_embd[codebook_idx], row_f16.data(),
                            (size_t)token_id * D * sizeof(ggml_fp16_t),
                            D * sizeof(ggml_fp16_t));
    ggml_fp16_to_fp32_row(row_f16.data(), out, D);
}

void Qwen3TalkerLLM::compute_text_embedding(int32_t token_id, float * out) const {
    const int text_dim = model_.config.text_hidden_size;  // 2048
    const int hidden_dim = model_.config.hidden_size;     // 1024

    // Read text_embd row (F16 → F32)
    std::vector<ggml_fp16_t> row_f16(text_dim);
    ggml_backend_tensor_get(model_.text_embd, row_f16.data(),
                            (size_t)token_id * text_dim * sizeof(ggml_fp16_t),
                            text_dim * sizeof(ggml_fp16_t));
    std::vector<float> x(text_dim);
    ggml_fp16_to_fp32_row(row_f16.data(), x.data(), text_dim);

    // fc1: y = GELU(x @ W1^T + b1)
    // W1: [text_dim, text_dim] stored as F16, b1: [text_dim] stored as F32
    std::vector<ggml_fp16_t> w1_f16(text_dim * text_dim);
    ggml_backend_tensor_get(model_.text_proj_fc1_w, w1_f16.data(), 0,
                            text_dim * text_dim * sizeof(ggml_fp16_t));
    std::vector<float> b1(text_dim);
    ggml_backend_tensor_get(model_.text_proj_fc1_b, b1.data(), 0,
                            text_dim * sizeof(float));

    std::vector<float> y(text_dim);
    for (int j = 0; j < text_dim; j++) {
        float sum = b1[j];
        for (int i = 0; i < text_dim; i++) {
            sum += ggml_fp16_to_fp32(w1_f16[j * text_dim + i]) * x[i];
        }
        // SiLU (Swish): x * sigmoid(x)
        y[j] = sum / (1.0f + expf(-sum));
    }

    // fc2: out = y @ W2^T + b2
    // W2: [text_dim, hidden_dim] stored as F16, b2: [hidden_dim] stored as F32
    std::vector<ggml_fp16_t> w2_f16(text_dim * hidden_dim);
    ggml_backend_tensor_get(model_.text_proj_fc2_w, w2_f16.data(), 0,
                            text_dim * hidden_dim * sizeof(ggml_fp16_t));
    std::vector<float> b2(hidden_dim);
    ggml_backend_tensor_get(model_.text_proj_fc2_b, b2.data(), 0,
                            hidden_dim * sizeof(float));

    for (int j = 0; j < hidden_dim; j++) {
        float sum = b2[j];
        for (int i = 0; i < text_dim; i++) {
            sum += ggml_fp16_to_fp32(w2_f16[j * text_dim + i]) * y[i];
        }
        out[j] = sum;
    }
}

void Qwen3TalkerLLM::compute_codec_embedding(int32_t token_id, float * out) const {
    const int hidden_dim = model_.config.hidden_size;  // 1024

    std::vector<ggml_fp16_t> row_f16(hidden_dim);
    ggml_backend_tensor_get(model_.codec_embd, row_f16.data(),
                            (size_t)token_id * hidden_dim * sizeof(ggml_fp16_t),
                            hidden_dim * sizeof(ggml_fp16_t));
    ggml_fp16_to_fp32_row(row_f16.data(), out, hidden_dim);
}

} // namespace qwen3
