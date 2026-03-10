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

// GPT-2 byte-to-unicode mapping (used by HuggingFace ByteLevel pre-tokenizer).
// Vocab entries in tokenizer.json use Unicode representations of bytes, not raw bytes.
// We need to reverse this mapping: decode vocab strings back to raw byte sequences.
//
// bytes_to_unicode(): printable bytes map to themselves, others map to U+0100..U+0143
static void build_unicode_to_byte(std::map<uint32_t, uint8_t> & u2b) {
    // Bytes that map to themselves (printable ranges)
    // 33..126, 161..172, 174..255
    std::vector<int> bs;
    for (int i = 33; i <= 126; i++) bs.push_back(i);   // ! through ~
    for (int i = 161; i <= 172; i++) bs.push_back(i);   // ¡ through ¬
    for (int i = 174; i <= 255; i++) bs.push_back(i);   // ® through ÿ

    // These bytes map to themselves
    for (int b : bs) {
        u2b[(uint32_t)b] = (uint8_t)b;
    }

    // Remaining bytes (0..32, 127..160, 173) map to U+0100..U+0143
    std::vector<bool> in_bs(256, false);
    for (int b : bs) in_bs[b] = true;

    int n = 0;
    for (int b = 0; b < 256; b++) {
        if (!in_bs[b]) {
            u2b[(uint32_t)(256 + n)] = (uint8_t)b;
            n++;
        }
    }
}

// Decode a byte-level encoded string (from tokenizer.json) to raw bytes.
// Each Unicode codepoint in the string maps to one raw byte.
static std::vector<uint8_t> decode_byte_level(const std::string & s,
                                               const std::map<uint32_t, uint8_t> & u2b) {
    std::vector<uint8_t> out;
    for (size_t i = 0; i < s.size(); ) {
        uint32_t cp;
        uint8_t c = (uint8_t)s[i];
        int len;
        if (c < 0x80) {
            cp = c; len = 1;
        } else if ((c & 0xE0) == 0xC0) {
            cp = c & 0x1F; len = 2;
        } else if ((c & 0xF0) == 0xE0) {
            cp = c & 0x0F; len = 3;
        } else {
            cp = c & 0x07; len = 4;
        }
        for (int j = 1; j < len && i + j < s.size(); j++) {
            cp = (cp << 6) | ((uint8_t)s[i + j] & 0x3F);
        }
        i += len;

        auto it = u2b.find(cp);
        if (it != u2b.end()) {
            out.push_back(it->second);
        }
        // If not found in mapping, skip (shouldn't happen with valid tokenizer)
    }
    return out;
}

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
[[maybe_unused]]
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
    // Build the unicode-to-byte mapping for decoding byte-level tokenizer entries
    std::map<uint32_t, uint8_t> u2b;
    build_unicode_to_byte(u2b);

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

        // Decode byte-level representation to raw bytes
        std::vector<uint8_t> token_bytes = decode_byte_level(token_str, u2b);
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

        // Decode byte-level representations to raw bytes
        std::vector<uint8_t> first_bytes = decode_byte_level(first, u2b);
        std::vector<uint8_t> second_bytes = decode_byte_level(second, u2b);

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

// Decode one UTF-8 codepoint from string, return codepoint and advance pos
static uint32_t decode_utf8(const std::string & s, size_t & pos) {
    uint8_t c = (uint8_t)s[pos];
    uint32_t cp;
    int len;
    if (c < 0x80)       { cp = c;           len = 1; }
    else if ((c & 0xE0) == 0xC0) { cp = c & 0x1F; len = 2; }
    else if ((c & 0xF0) == 0xE0) { cp = c & 0x0F; len = 3; }
    else                { cp = c & 0x07; len = 4; }
    for (int j = 1; j < len && pos + j < s.size(); j++) {
        cp = (cp << 6) | ((uint8_t)s[pos + j] & 0x3F);
    }
    pos += len;
    return cp;
}

// Check if a Unicode codepoint is a "letter" (approximation of \p{L})
static bool is_unicode_letter(uint32_t cp) {
    if (cp < 0x80) return std::isalpha((int)cp);
    // Latin Extended
    if (cp >= 0x00C0 && cp <= 0x024F) return true;
    // Cyrillic
    if (cp >= 0x0400 && cp <= 0x04FF) return true;
    // Arabic
    if (cp >= 0x0600 && cp <= 0x06FF) return true;
    // CJK Unified Ideographs
    if (cp >= 0x4E00 && cp <= 0x9FFF) return true;
    // CJK Extension A
    if (cp >= 0x3400 && cp <= 0x4DBF) return true;
    // CJK Compatibility Ideographs
    if (cp >= 0xF900 && cp <= 0xFAFF) return true;
    // Hiragana
    if (cp >= 0x3040 && cp <= 0x309F) return true;
    // Katakana
    if (cp >= 0x30A0 && cp <= 0x30FF) return true;
    // Hangul
    if (cp >= 0xAC00 && cp <= 0xD7AF) return true;
    // CJK Extension B+
    if (cp >= 0x20000 && cp <= 0x2A6DF) return true;
    // Thai
    if (cp >= 0x0E00 && cp <= 0x0E7F) return true;
    // Devanagari
    if (cp >= 0x0900 && cp <= 0x097F) return true;
    return false;
}

// Check if a Unicode codepoint is a digit (\p{N})
static bool is_unicode_digit(uint32_t cp) {
    if (cp >= '0' && cp <= '9') return true;
    // Fullwidth digits
    if (cp >= 0xFF10 && cp <= 0xFF19) return true;
    return false;
}


// GPT-4 style pre-tokenization:
// Pattern: (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
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

        // Decode the current UTF-8 codepoint (peek, don't advance yet)
        size_t peek = i;
        uint32_t cp = decode_utf8(text, peek);
        int clen = (int)(peek - i);

        if (c == ' ' || c == '\t') {
            // Whitespace: flush current, start new chunk with space prefix
            if (!current.empty()) { chunks.push_back(current); current.clear(); }
            current += text[i];
            i++;
            // Collect following letters (ASCII or Unicode) as part of this chunk
            while (i < text.size()) {
                size_t p2 = i;
                uint32_t cp2 = decode_utf8(text, p2);
                if (is_unicode_letter(cp2)) {
                    current.append(text, i, p2 - i);
                    i = p2;
                } else {
                    break;
                }
            }
            if (!current.empty()) { chunks.push_back(current); current.clear(); }
        } else if (c == '\n' || c == '\r') {
            // Newline
            if (!current.empty()) { chunks.push_back(current); current.clear(); }
            current += text[i]; i++;
            chunks.push_back(current); current.clear();
        } else if (is_unicode_letter(cp)) {
            // Letter: accumulate consecutive letters
            current.append(text, i, clen);
            i += clen;
        } else if (is_unicode_digit(cp)) {
            // Digit: each digit is its own chunk
            if (!current.empty()) { chunks.push_back(current); current.clear(); }
            current.append(text, i, clen);
            chunks.push_back(current); current.clear();
            i += clen;
        } else {
            // Punctuation/symbol: may start a new chunk with following letters
            // Pattern: [^\r\n\p{L}\p{N}]?\p{L}+
            if (!current.empty()) { chunks.push_back(current); current.clear(); }
            current.append(text, i, clen);
            i += clen;
            // Check if letters follow — if so, include them in this chunk
            bool has_letters = false;
            while (i < text.size()) {
                size_t p2 = i;
                uint32_t cp2 = decode_utf8(text, p2);
                if (is_unicode_letter(cp2)) {
                    current.append(text, i, p2 - i);
                    i = p2;
                    has_letters = true;
                } else {
                    break;
                }
            }
            if (!has_letters) {
                // No letters follow — collect non-letter, non-digit, non-space chars
                while (i < text.size()) {
                    size_t p2 = i;
                    uint32_t cp2 = decode_utf8(text, p2);
                    if (!is_unicode_letter(cp2) && !is_unicode_digit(cp2) &&
                        cp2 != ' ' && cp2 != '\n' && cp2 != '\r' && cp2 != '\t') {
                        current.append(text, i, p2 - i);
                        i = p2;
                    } else {
                        break;
                    }
                }
            }
            chunks.push_back(current); current.clear();
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
