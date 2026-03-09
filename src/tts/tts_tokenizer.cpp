#include "tts_tokenizer.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <climits>
#include <cctype>

namespace vocal_tts {

TTSTokenizer::TTSTokenizer() = default;
TTSTokenizer::~TTSTokenizer() = default;

// --- Simple JSON parsing helpers (just enough for tokenizer.json) ---

static std::string read_file(const std::string & path) {
    std::ifstream f(path);
    if (!f) return "";
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// Skip whitespace
static size_t skip_ws(const std::string & s, size_t pos) {
    while (pos < s.size() && (s[pos] == ' ' || s[pos] == '\n' || s[pos] == '\r' || s[pos] == '\t'))
        pos++;
    return pos;
}

// Parse a JSON string (handling escapes)
static bool parse_json_string(const std::string & s, size_t & pos, std::string & out) {
    pos = skip_ws(s, pos);
    if (pos >= s.size() || s[pos] != '"') return false;
    pos++; // skip opening "
    out.clear();
    while (pos < s.size() && s[pos] != '"') {
        if (s[pos] == '\\' && pos + 1 < s.size()) {
            pos++;
            switch (s[pos]) {
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case 'u': {
                    // Parse \uXXXX
                    if (pos + 4 < s.size()) {
                        char hex[5] = {s[pos+1], s[pos+2], s[pos+3], s[pos+4], 0};
                        uint32_t cp = (uint32_t)strtoul(hex, nullptr, 16);
                        pos += 4;
                        // UTF-8 encode
                        if (cp < 0x80) {
                            out += (char)cp;
                        } else if (cp < 0x800) {
                            out += (char)(0xC0 | (cp >> 6));
                            out += (char)(0x80 | (cp & 0x3F));
                        } else {
                            out += (char)(0xE0 | (cp >> 12));
                            out += (char)(0x80 | ((cp >> 6) & 0x3F));
                            out += (char)(0x80 | (cp & 0x3F));
                        }
                    }
                    break;
                }
                default: out += s[pos]; break;
            }
        } else {
            out += s[pos];
        }
        pos++;
    }
    if (pos < s.size()) pos++; // skip closing "
    return true;
}

// Parse a JSON integer
static bool parse_json_int(const std::string & s, size_t & pos, int32_t & out) {
    pos = skip_ws(s, pos);
    size_t start = pos;
    if (pos < s.size() && s[pos] == '-') pos++;
    while (pos < s.size() && s[pos] >= '0' && s[pos] <= '9') pos++;
    if (pos == start) return false;
    out = (int32_t)atol(s.substr(start, pos - start).c_str());
    return true;
}

// Skip a JSON value (string, number, object, array, bool, null)
static void skip_json_value(const std::string & s, size_t & pos) {
    pos = skip_ws(s, pos);
    if (pos >= s.size()) return;

    if (s[pos] == '"') {
        std::string dummy;
        parse_json_string(s, pos, dummy);
    } else if (s[pos] == '{') {
        pos++;
        int depth = 1;
        while (pos < s.size() && depth > 0) {
            if (s[pos] == '{') depth++;
            else if (s[pos] == '}') depth--;
            else if (s[pos] == '"') { std::string d; parse_json_string(s, pos, d); continue; }
            pos++;
        }
    } else if (s[pos] == '[') {
        pos++;
        int depth = 1;
        while (pos < s.size() && depth > 0) {
            if (s[pos] == '[') depth++;
            else if (s[pos] == ']') depth--;
            else if (s[pos] == '"') { std::string d; parse_json_string(s, pos, d); continue; }
            pos++;
        }
    } else {
        // number, bool, null
        while (pos < s.size() && s[pos] != ',' && s[pos] != '}' && s[pos] != ']')
            pos++;
    }
}

bool TTSTokenizer::parse_json(const std::string & json) {
    // Find "model" object
    size_t model_pos = json.find("\"model\"");
    if (model_pos == std::string::npos) {
        error_ = "tokenizer.json: 'model' key not found";
        return false;
    }

    // Find "vocab" within model
    size_t vocab_pos = json.find("\"vocab\"", model_pos);
    if (vocab_pos == std::string::npos) {
        error_ = "tokenizer.json: 'vocab' key not found";
        return false;
    }

    // Parse vocab: { "token": id, ... }
    size_t pos = vocab_pos + 7; // skip "vocab"
    pos = skip_ws(json, pos);
    if (pos < json.size() && json[pos] == ':') pos++;
    pos = skip_ws(json, pos);

    if (pos >= json.size() || json[pos] != '{') {
        error_ = "tokenizer.json: expected '{' for vocab";
        return false;
    }
    pos++; // skip {

    int vocab_count = 0;
    while (pos < json.size()) {
        pos = skip_ws(json, pos);
        if (json[pos] == '}') { pos++; break; }
        if (json[pos] == ',') { pos++; continue; }

        std::string token_str;
        if (!parse_json_string(json, pos, token_str)) break;

        pos = skip_ws(json, pos);
        if (json[pos] == ':') pos++;

        int32_t token_id;
        if (!parse_json_int(json, pos, token_id)) break;

        std::vector<uint8_t> token_bytes(token_str.begin(), token_str.end());
        vocab_[token_bytes] = token_id;
        id_to_token_[token_id] = token_bytes;
        vocab_count++;
    }

    fprintf(stderr, "  Loaded %d vocab entries\n", vocab_count);

    // Find "merges" within model
    size_t merges_pos = json.find("\"merges\"", model_pos);
    if (merges_pos == std::string::npos) {
        // No merges — might be a character-level tokenizer
        return true;
    }

    pos = merges_pos + 8; // skip "merges"
    pos = skip_ws(json, pos);
    if (pos < json.size() && json[pos] == ':') pos++;
    pos = skip_ws(json, pos);

    if (pos >= json.size() || json[pos] != '[') {
        error_ = "tokenizer.json: expected '[' for merges";
        return false;
    }
    pos++; // skip [

    int merge_count = 0;
    while (pos < json.size()) {
        pos = skip_ws(json, pos);
        if (json[pos] == ']') { pos++; break; }
        if (json[pos] == ',') { pos++; continue; }

        std::string first, second;

        if (json[pos] == '[') {
            // Array format: ["token1", "token2"]
            pos++; // skip [
            pos = skip_ws(json, pos);
            if (!parse_json_string(json, pos, first)) break;
            pos = skip_ws(json, pos);
            if (json[pos] == ',') pos++;
            pos = skip_ws(json, pos);
            if (!parse_json_string(json, pos, second)) break;
            pos = skip_ws(json, pos);
            if (json[pos] == ']') pos++;
        } else if (json[pos] == '"') {
            // String format: "token1 token2"
            std::string merge_str;
            if (!parse_json_string(json, pos, merge_str)) break;
            size_t space = merge_str.find(' ');
            if (space == std::string::npos) continue;
            first = merge_str.substr(0, space);
            second = merge_str.substr(space + 1);
        } else {
            break;
        }

        std::vector<uint8_t> first_bytes(first.begin(), first.end());
        std::vector<uint8_t> second_bytes(second.begin(), second.end());

        merges_.push_back({first_bytes, second_bytes});
        merge_ranks_[{first_bytes, second_bytes}] = merge_count;
        merge_count++;
    }

    fprintf(stderr, "  Loaded %d merge rules\n", merge_count);
    return true;
}

bool TTSTokenizer::load(const std::string & path) {
    fprintf(stderr, "Loading tokenizer from %s...\n", path.c_str());

    std::string json = read_file(path);
    if (json.empty()) {
        error_ = "Failed to read tokenizer file: " + path;
        return false;
    }

    if (!parse_json(json)) {
        return false;
    }

    loaded_ = true;
    fprintf(stderr, "Tokenizer loaded (%zu vocab, %zu merges)\n",
            vocab_.size(), merges_.size());
    return true;
}

// GPT-2 style pre-tokenization: split on whitespace boundaries and punctuation
std::vector<std::string> TTSTokenizer::pre_tokenize(const std::string & text) const {
    std::vector<std::string> chunks;
    std::string current;

    for (size_t i = 0; i < text.size(); ) {
        uint8_t c = (uint8_t)text[i];

        // Handle contractions: 's, 't, 're, 've, 'm, 'll, 'd
        if (c == '\'' && i + 1 < text.size()) {
            char next = text[i + 1];
            if (next == 's' || next == 't' || next == 'm' || next == 'd') {
                if (!current.empty()) { chunks.push_back(current); current.clear(); }
                current += text[i]; current += text[i + 1];
                chunks.push_back(current); current.clear();
                i += 2;
                continue;
            }
            if (next == 'r' && i + 2 < text.size() && text[i + 2] == 'e') {
                if (!current.empty()) { chunks.push_back(current); current.clear(); }
                current += text[i]; current += text[i + 1]; current += text[i + 2];
                chunks.push_back(current); current.clear();
                i += 3;
                continue;
            }
            if (next == 'v' && i + 2 < text.size() && text[i + 2] == 'e') {
                if (!current.empty()) { chunks.push_back(current); current.clear(); }
                current += text[i]; current += text[i + 1]; current += text[i + 2];
                chunks.push_back(current); current.clear();
                i += 3;
                continue;
            }
            if (next == 'l' && i + 2 < text.size() && text[i + 2] == 'l') {
                if (!current.empty()) { chunks.push_back(current); current.clear(); }
                current += text[i]; current += text[i + 1]; current += text[i + 2];
                chunks.push_back(current); current.clear();
                i += 3;
                continue;
            }
        }

        if (c == ' ') {
            // Space starts a new word — include space as prefix (GPT-2 style Ġ)
            if (!current.empty()) { chunks.push_back(current); current.clear(); }
            current += ' ';
            i++;
            // Collect the word after the space
            while (i < text.size() && (uint8_t)text[i] > ' ' &&
                   std::isalnum((uint8_t)text[i])) {
                current += text[i];
                i++;
            }
            if (!current.empty()) { chunks.push_back(current); current.clear(); }
        } else if (std::isalnum(c) || c >= 0x80) {
            // Alphanumeric or UTF-8 continuation
            current += text[i];
            i++;
        } else {
            // Punctuation: emit current, then emit punctuation as own chunk
            if (!current.empty()) { chunks.push_back(current); current.clear(); }
            current += text[i];
            chunks.push_back(current);
            current.clear();
            i++;
        }
    }

    if (!current.empty()) {
        chunks.push_back(current);
    }

    return chunks;
}

std::vector<int32_t> TTSTokenizer::bpe_encode(const std::string & chunk) const {
    if (chunk.empty()) return {};

    // Initialize: each byte is a separate token
    std::vector<std::vector<uint8_t>> tokens;
    for (uint8_t c : chunk) {
        tokens.push_back({c});
    }

    // Check if the whole chunk is a single vocab entry
    std::vector<uint8_t> whole(chunk.begin(), chunk.end());
    auto it = vocab_.find(whole);
    if (it != vocab_.end()) {
        return {it->second};
    }

    // BPE merge loop
    while (tokens.size() > 1) {
        // Find the pair with lowest merge rank
        int best_rank = INT_MAX;
        int best_pos = -1;

        for (size_t i = 0; i + 1 < tokens.size(); i++) {
            auto key = std::make_pair(tokens[i], tokens[i + 1]);
            auto mit = merge_ranks_.find(key);
            if (mit != merge_ranks_.end() && mit->second < best_rank) {
                best_rank = mit->second;
                best_pos = (int)i;
            }
        }

        if (best_pos < 0) break; // No more merges

        // Merge at best_pos
        std::vector<uint8_t> merged = tokens[best_pos];
        merged.insert(merged.end(), tokens[best_pos + 1].begin(), tokens[best_pos + 1].end());
        tokens[best_pos] = merged;
        tokens.erase(tokens.begin() + best_pos + 1);
    }

    // Convert to token IDs
    std::vector<int32_t> ids;
    for (const auto & tok : tokens) {
        auto vit = vocab_.find(tok);
        if (vit != vocab_.end()) {
            ids.push_back(vit->second);
        } else {
            // Unknown token — encode individual bytes
            for (uint8_t b : tok) {
                std::vector<uint8_t> single = {b};
                auto sit = vocab_.find(single);
                if (sit != vocab_.end()) {
                    ids.push_back(sit->second);
                }
                // If even single byte not in vocab, skip (shouldn't happen with byte-level BPE)
            }
        }
    }

    return ids;
}

std::vector<int32_t> TTSTokenizer::encode(const std::string & text) const {
    auto chunks = pre_tokenize(text);

    std::vector<int32_t> all_ids;
    for (const auto & chunk : chunks) {
        auto ids = bpe_encode(chunk);
        all_ids.insert(all_ids.end(), ids.begin(), ids.end());
    }

    return all_ids;
}

std::vector<int32_t> TTSTokenizer::build_tts_prompt(const std::string & text) const {
    // TTS prompt format:
    // [tts_bos] <|im_start|> assistant \n {text} <|im_end|> \n
    //
    // The model generates audio codes after this prompt until [tts_eos]

    std::vector<int32_t> prompt;

    // TTS begin
    prompt.push_back(special_.tts_bos);

    // <|im_start|>
    prompt.push_back(special_.im_start);

    // "assistant"
    prompt.push_back(special_.assistant);

    // \n
    prompt.push_back(special_.newline);

    // Encoded text
    auto text_ids = encode(text);
    prompt.insert(prompt.end(), text_ids.begin(), text_ids.end());

    // <|im_end|>
    prompt.push_back(special_.im_end);

    // \n
    prompt.push_back(special_.newline);

    return prompt;
}

} // namespace vocal_tts
