#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <string>
#include <map>
#include <vector>

namespace vocal_tts {

class GGUFLoader {
public:
    GGUFLoader();
    ~GGUFLoader();

    bool open(const std::string & path);
    void close();

    const std::string & get_error() const { return error_msg_; }

    int64_t get_n_tensors() const;
    const char * get_tensor_name(int64_t idx) const;
    enum ggml_type get_tensor_type(int64_t idx) const;
    size_t get_tensor_offset(int64_t idx) const;
    size_t get_tensor_size(int64_t idx) const;

    int32_t get_u32(const char * key, int32_t default_val = 0) const;
    float get_f32(const char * key, float default_val = 0.0f) const;

    size_t get_data_offset() const;

    struct gguf_context * get_ctx() const { return ctx_; }
    struct ggml_context * get_meta_ctx() const { return meta_ctx_; }

protected:
    struct gguf_context * ctx_ = nullptr;
    struct ggml_context * meta_ctx_ = nullptr;
    std::string error_msg_;
    std::string file_path_;
};

bool load_tensor_data_from_file(
    const std::string & path,
    struct gguf_context * ctx,
    struct ggml_context * model_ctx,
    const std::map<std::string, struct ggml_tensor *> & tensors,
    ggml_backend_buffer_t & buffer,
    std::string & error_msg,
    enum ggml_backend_dev_type preferred_backend_type = GGML_BACKEND_DEVICE_TYPE_CPU
);

ggml_backend_t init_preferred_backend(const char * component_name, std::string * error_msg);
void release_preferred_backend(ggml_backend_t backend);

void free_ggml_resources(struct ggml_context * ctx, ggml_backend_buffer_t buffer);

} // namespace vocal_tts
