#include "minilm.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>

#include "minilm_internal.hpp"
#include "rt.hpp"       // load_safetensors
#include "rt_math.hpp"  // the Accelerate/register-blocked GEMM

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
// The portable build reuses rt's register-blocked GEMM via cblas-compatible
// shims below; declaring just what we need avoids a BLAS dependency.
#endif

namespace minilm {
namespace {

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// UTF-8 + BERT basic tokenization
// ---------------------------------------------------------------------------

// Decode one UTF-8 codepoint at s[i]; advances i. Invalid bytes decode as
// U+FFFD and advance by one, like Python's errors="replace".
uint32_t decode_utf8(const std::string& s, size_t& i) {
  const unsigned char c = s[i];
  if (c < 0x80) { i += 1; return c; }
  auto cont = [&](size_t k) {
    return i + k < s.size() && (s[i + k] & 0xC0) == 0x80;
  };
  if ((c & 0xE0) == 0xC0 && cont(1)) {
    uint32_t cp = ((c & 0x1F) << 6) | (s[i + 1] & 0x3F);
    i += 2;
    return cp;
  }
  if ((c & 0xF0) == 0xE0 && cont(1) && cont(2)) {
    uint32_t cp = ((c & 0x0F) << 12) | ((s[i + 1] & 0x3F) << 6) |
                  (s[i + 2] & 0x3F);
    i += 3;
    return cp;
  }
  if ((c & 0xF8) == 0xF0 && cont(1) && cont(2) && cont(3)) {
    uint32_t cp = ((c & 0x07) << 18) | ((s[i + 1] & 0x3F) << 12) |
                  ((s[i + 2] & 0x3F) << 6) | (s[i + 3] & 0x3F);
    i += 4;
    return cp;
  }
  i += 1;
  return 0xFFFD;
}

void append_utf8(std::string& out, uint32_t cp) {
  if (cp < 0x80) {
    out.push_back((char)cp);
  } else if (cp < 0x800) {
    out.push_back((char)(0xC0 | (cp >> 6)));
    out.push_back((char)(0x80 | (cp & 0x3F)));
  } else if (cp < 0x10000) {
    out.push_back((char)(0xE0 | (cp >> 12)));
    out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back((char)(0x80 | (cp & 0x3F)));
  } else {
    out.push_back((char)(0xF0 | (cp >> 18)));
    out.push_back((char)(0x80 | ((cp >> 12) & 0x3F)));
    out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back((char)(0x80 | (cp & 0x3F)));
  }
}

bool is_whitespace(uint32_t cp) {
  return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' || cp == 0xA0 ||
         (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029 ||
         cp == 0x202F || cp == 0x205F || cp == 0x3000 || cp == 0x1680;
}

bool is_control(uint32_t cp) {
  if (cp == '\t' || cp == '\n' || cp == '\r') return false;  // whitespace
  return cp < 0x20 || cp == 0x7F || (cp >= 0x80 && cp <= 0x9F) ||
         cp == 0x200B || cp == 0x200C || cp == 0x200D ||     // Cf zero-widths
         cp == 0xFEFF || (cp >= 0x202A && cp <= 0x202E);
}

// BERT treats every non-alnum ASCII char as punctuation, plus Unicode P*.
// The blocks below cover what real cell text contains.
bool is_punct(uint32_t cp) {
  if ((cp >= '!' && cp <= '/') || (cp >= ':' && cp <= '@') ||
      (cp >= '[' && cp <= '`') || (cp >= '{' && cp <= '~'))
    return true;
  return (cp >= 0x2010 && cp <= 0x2027) ||   // dashes, quotes, bullets
         (cp >= 0x2030 && cp <= 0x205E) ||   // permille .. punct
         (cp >= 0x00A1 && cp <= 0x00BF && cp != 0x00AA && cp != 0x00B2 &&
          cp != 0x00B3 && cp != 0x00B5 && cp != 0x00B9 && cp != 0x00BA &&
          cp != 0x00BC && cp != 0x00BD && cp != 0x00BE) ||  // Latin-1 punct
         (cp >= 0x3001 && cp <= 0x303F) ||   // CJK punctuation
         (cp >= 0xFF01 && cp <= 0xFF0F) || (cp >= 0xFF1A && cp <= 0xFF20) ||
         (cp >= 0xFF3B && cp <= 0xFF40) || (cp >= 0xFF5B && cp <= 0xFF65);
}

bool is_cjk(uint32_t cp) {
  return (cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0x3400 && cp <= 0x4DBF) ||
         (cp >= 0x20000 && cp <= 0x2A6DF) || (cp >= 0x2A700 && cp <= 0x2B73F) ||
         (cp >= 0x2B740 && cp <= 0x2B81F) || (cp >= 0x2B820 && cp <= 0x2CEAF) ||
         (cp >= 0xF900 && cp <= 0xFAFF) || (cp >= 0x2F800 && cp <= 0x2FA1F);
}

bool is_combining_mark(uint32_t cp) {
  return (cp >= 0x0300 && cp <= 0x036F) || (cp >= 0x1AB0 && cp <= 0x1AFF) ||
         (cp >= 0x1DC0 && cp <= 0x1DFF) || (cp >= 0x20D0 && cp <= 0x20FF) ||
         (cp >= 0xFE20 && cp <= 0xFE2F);
}

// Lowercase + NFD-and-drop-marks for the Latin ranges real data carries.
// Beyond these ranges the codepoint passes through unchanged; an unmatched
// word WordPiece-tokenizes to [UNK] either way.
uint32_t lower_strip(uint32_t cp) {
  if (cp >= 'A' && cp <= 'Z') return cp + 32;
  if (cp >= 0x0410 && cp <= 0x042F) return cp + 0x20;   // Cyrillic А-Я
  if (cp >= 0x0400 && cp <= 0x040F) return cp + 0x50;   // Ѐ-Џ
  if (cp >= 0x0391 && cp <= 0x03A9 && cp != 0x03A2) return cp + 0x20;  // Greek
  if (cp >= 0x00C0 && cp <= 0x00DE && cp != 0x00D7) cp += 0x20;  // À-Þ -> à-þ
  // Latin-1 accented lowercase -> base letter (NFD drop of the mark).
  static const struct { uint32_t lo, hi; char base; } latin1[] = {
      {0xE0, 0xE5, 'a'}, {0xE7, 0xE7, 'c'}, {0xE8, 0xEB, 'e'},
      {0xEC, 0xEF, 'i'}, {0xF1, 0xF1, 'n'}, {0xF2, 0xF6, 'o'},
      {0xF9, 0xFC, 'u'}, {0xFD, 0xFD, 'y'}, {0xFF, 0xFF, 'y'}};
  for (const auto& r : latin1)
    if (cp >= r.lo && cp <= r.hi) return (uint32_t)r.base;
  // Latin Extended-A: pairs of (upper, lower) with a base letter; decompose
  // by the block's regular structure. 0x100-0x17F alternates U/l per letter.
  if (cp >= 0x0100 && cp <= 0x017F) {
    static const char* base =
        // 0100-010F aaaa cccc cccc dddd  (Ā ā Ă ă Ą ą Ć ć Ĉ ĉ Ċ ċ Č č Ď ď)
        "aaaaaaccccccccdd"
        // 0110-011F dddd eeee eeee gggg  (Đ đ Ē ē Ĕ ĕ Ė ė Ę ę Ě ě Ĝ ĝ Ğ ğ)
        "ddeeeeeeeeeegggg"
        // 0120-012F gggg hhhh iiii iiii
        "gggghhhhiiiiiiii"
        // 0130-013F iijj jjkk kll llll l  (İ ı Ĳ ĳ Ĵ ĵ Ķ ķ ĸ Ĺ ĺ Ļ ļ Ľ ľ Ŀ)
        "iiiijjkkklllllll"
        // 0140-014F l lll nnnn nnnn noo   (ŀ Ł ł Ń ń Ņ ņ Ň ň ŉ Ŋ ŋ Ō ō Ŏ ŏ)
        "lllnnnnnnnnnoooo"
        // 0150-015F oooo eeer rrrr rsss   (Ő ő Œ œ Ŕ ŕ Ŗ ŗ Ř ř Ś ś Ŝ ŝ Ş ş)
        "ooooeerrrrrrssss"
        // 0160-016F ssst tttt uuuu uuuu
        "ssttttttuuuuuuuu"
        // 0170-017F uuuu wwyy yzzz zzzs
        "uuuuwwyyyzzzzzzs";
    return (uint32_t)base[cp - 0x0100];
  }
  return cp;
}

std::vector<std::string> basic_tokenize(const std::string& text) {
  // clean + CJK spacing + lowercase/strip, then whitespace/punct split.
  std::string cleaned;
  cleaned.reserve(text.size());
  size_t i = 0;
  while (i < text.size()) {
    uint32_t cp = decode_utf8(text, i);
    if (cp == 0 || cp == 0xFFFD || is_control(cp)) continue;
    if (is_whitespace(cp)) {
      cleaned.push_back(' ');
      continue;
    }
    if (is_cjk(cp)) {
      cleaned.push_back(' ');
      append_utf8(cleaned, cp);
      cleaned.push_back(' ');
      continue;
    }
    cp = lower_strip(cp);
    if (is_combining_mark(cp)) continue;      // NFD-dropped marks
    append_utf8(cleaned, cp);
  }

  std::vector<std::string> tokens;
  std::string cur;
  i = 0;
  while (i < cleaned.size()) {
    uint32_t cp = decode_utf8(cleaned, i);
    if (cp == ' ') {
      if (!cur.empty()) tokens.push_back(std::move(cur));
      cur.clear();
      continue;
    }
    if (is_punct(cp)) {
      if (!cur.empty()) tokens.push_back(std::move(cur));
      cur.clear();
      std::string p;
      append_utf8(p, cp);
      tokens.push_back(std::move(p));
      continue;
    }
    append_utf8(cur, cp);
  }
  if (!cur.empty()) tokens.push_back(std::move(cur));
  return tokens;
}

// ---------------------------------------------------------------------------
// model weights
// ---------------------------------------------------------------------------

using LayerW = detail::LayerW;

void gemm_xwt(const float* x, const float* w, const float* b, float* y,
              int rows, int in, int out) {
  // y[rows,out] = x[rows,in] @ w[out,in]^T + b — rt's Accelerate/
  // register-blocked kernel; the previous portable fallback here was a naive
  // scalar loop that made long-text encodes ~100ms/text on Linux.
  rt::math::gemm_nt(x, w, y, rows, out, in, in, in, out);
  if (b)
    for (int r = 0; r < rows; ++r)
      for (int o = 0; o < out; ++o) y[(size_t)r * out + o] += b[o];
}

void layer_norm(float* x, int rows, int dim, const float* w, const float* b) {
  for (int r = 0; r < rows; ++r) {
    float* row = x + (size_t)r * dim;
    float mean = 0.f;
    for (int i = 0; i < dim; ++i) mean += row[i];
    mean /= dim;
    float var = 0.f;
    for (int i = 0; i < dim; ++i) {
      float d = row[i] - mean;
      var += d * d;
    }
    var /= dim;
    const float inv = 1.0f / std::sqrt(var + 1e-12f);
    for (int i = 0; i < dim; ++i)
      row[i] = (row[i] - mean) * inv * w[i] + b[i];
  }
}

inline float gelu_erf(float x) {   // transformers' "gelu": exact erf form
  return 0.5f * x * (1.0f + std::erf(x * 0.70710678118654752440f));
}

}  // namespace

// ---------------------------------------------------------------------------

std::string resolve_snapshot() {
  std::string home;
  if (const char* hf = std::getenv("HF_HOME")) {
    home = std::string(hf) + "/hub";
  } else if (const char* h = std::getenv("HOME")) {
    home = std::string(h) + "/.cache/huggingface/hub";
  } else {
    return "";
  }
  const fs::path snaps = fs::path(home) /
      "models--sentence-transformers--all-MiniLM-L12-v2" / "snapshots";
  std::error_code ec;
  fs::path best;
  fs::file_time_type best_time{};
  for (const auto& entry : fs::directory_iterator(snaps, ec)) {
    if (!entry.is_directory(ec)) continue;
    if (!fs::exists(entry.path() / "model.safetensors", ec)) continue;
    const auto t = fs::last_write_time(entry.path(), ec);
    if (best.empty() || t > best_time) {
      best = entry.path();
      best_time = t;
    }
  }
  return best.string();
}

struct MiniLM::Impl {
  std::unordered_map<std::string, rt::Tensor> store;
  std::unordered_map<std::string, int32_t> vocab;
  const float *word_emb, *pos_emb, *type_emb, *eln_w, *eln_b;
  LayerW layers[12];
  int64_t vocab_rows = 0;
  int32_t cls_id = 101, sep_id = 102, unk_id = 100;

  const float* need(const std::string& name) {
    auto it = store.find(name);
    if (it == store.end())
      throw std::runtime_error("MiniLM checkpoint is missing tensor " + name);
    if (it->second.data.empty())
      throw std::runtime_error("MiniLM tensor " + name +
                               " is quantized; fp32 required");
    return it->second.data.data();
  }
};

MiniLM::MiniLM() : impl_(new Impl) {}
MiniLM::~MiniLM() = default;

std::unique_ptr<MiniLM> MiniLM::load(const std::string& snapshot_dir,
                                     std::string* err) {
  std::string dir = snapshot_dir.empty() ? resolve_snapshot() : snapshot_dir;
  if (dir.empty()) {
    if (err)
      *err = "no MiniLM snapshot found (set RELATIVEDB_MINILM_DIR, or "
             "download sentence-transformers/all-MiniLM-L12-v2 into the HF "
             "cache)";
    return nullptr;
  }
  try {
    std::unique_ptr<MiniLM> m(new MiniLM());
    Impl& im = *m->impl_;
    im.store = rt::load_safetensors(dir + "/model.safetensors");

    // vocab.txt: one wordpiece per line, id = line number.
    std::ifstream vf(dir + "/vocab.txt");
    if (!vf) throw std::runtime_error(dir + "/vocab.txt not found");
    std::string line;
    int32_t id = 0;
    while (std::getline(vf, line)) {
      if (!line.empty() && line.back() == '\r') line.pop_back();
      im.vocab.emplace(line, id++);
    }
    auto special = [&](const char* tok) {
      auto it = im.vocab.find(tok);
      if (it == im.vocab.end())
        throw std::runtime_error(std::string("vocab.txt is missing ") + tok);
      return it->second;
    };
    im.cls_id = special("[CLS]");
    im.sep_id = special("[SEP]");
    im.unk_id = special("[UNK]");

    // BertModel tensor names; some exports carry a "bert." prefix.
    auto pick = [&](const std::string& name) -> std::string {
      if (im.store.count(name)) return name;
      if (im.store.count("bert." + name)) return "bert." + name;
      return name;   // let need() report the miss
    };
    im.word_emb = im.need(pick("embeddings.word_embeddings.weight"));
    im.vocab_rows =
        im.store.at(pick("embeddings.word_embeddings.weight")).shape[0];
    im.pos_emb = im.need(pick("embeddings.position_embeddings.weight"));
    im.type_emb = im.need(pick("embeddings.token_type_embeddings.weight"));
    im.eln_w = im.need(pick("embeddings.LayerNorm.weight"));
    im.eln_b = im.need(pick("embeddings.LayerNorm.bias"));
    for (int l = 0; l < 12; ++l) {
      const std::string p = "encoder.layer." + std::to_string(l) + ".";
      LayerW& w = im.layers[l];
      w.q_w = im.need(pick(p + "attention.self.query.weight"));
      w.q_b = im.need(pick(p + "attention.self.query.bias"));
      w.k_w = im.need(pick(p + "attention.self.key.weight"));
      w.k_b = im.need(pick(p + "attention.self.key.bias"));
      w.v_w = im.need(pick(p + "attention.self.value.weight"));
      w.v_b = im.need(pick(p + "attention.self.value.bias"));
      w.ao_w = im.need(pick(p + "attention.output.dense.weight"));
      w.ao_b = im.need(pick(p + "attention.output.dense.bias"));
      w.aln_w = im.need(pick(p + "attention.output.LayerNorm.weight"));
      w.aln_b = im.need(pick(p + "attention.output.LayerNorm.bias"));
      w.i_w = im.need(pick(p + "intermediate.dense.weight"));
      w.i_b = im.need(pick(p + "intermediate.dense.bias"));
      w.o_w = im.need(pick(p + "output.dense.weight"));
      w.o_b = im.need(pick(p + "output.dense.bias"));
      w.oln_w = im.need(pick(p + "output.LayerNorm.weight"));
      w.oln_b = im.need(pick(p + "output.LayerNorm.bias"));
    }
    return m;
  } catch (const std::exception& e) {
    if (err) *err = e.what();
    return nullptr;
  }
}

std::vector<int32_t> MiniLM::tokenize(const std::string& text) const {
  const Impl& im = *impl_;
  std::vector<int32_t> ids{im.cls_id};
  for (const std::string& word : basic_tokenize(text)) {
    if ((int)ids.size() >= kMaxTokens - 1) break;
    // WordPiece: greedy longest-match; >100 chars or no match -> [UNK].
    if (word.size() > 100) {
      ids.push_back(im.unk_id);
      continue;
    }
    std::vector<int32_t> pieces;
    size_t start = 0;
    bool bad = false;
    while (start < word.size()) {
      size_t end = word.size();
      int32_t cur = -1;
      while (start < end) {
        std::string sub = (start ? "##" : "") + word.substr(start, end - start);
        auto it = im.vocab.find(sub);
        if (it != im.vocab.end()) {
          cur = it->second;
          break;
        }
        // shrink by whole codepoints, not bytes, or a multibyte char splits
        do { --end; } while (end > start && (word[end] & 0xC0) == 0x80);
      }
      if (cur < 0) {
        bad = true;
        break;
      }
      pieces.push_back(cur);
      start = end;
    }
    if (bad) {
      ids.push_back(im.unk_id);
    } else {
      for (int32_t p : pieces) {
        if ((int)ids.size() >= kMaxTokens - 1) break;
        ids.push_back(p);
      }
    }
  }
  if ((int)ids.size() > kMaxTokens - 1) ids.resize(kMaxTokens - 1);
  ids.push_back(im.sep_id);
  return ids;
}

bool MiniLM::encode(const std::vector<std::string>& texts, bool normalize,
                    float* out, std::string* err) const {
  const Impl& im = *impl_;
  constexpr int D = kDim, H = 12, HD = D / 12, FF = 1536;
  // Tokenize up front (cheap, striped) so the device encoder can bucket
  // texts by token count.
  std::vector<std::vector<int32_t>> all_ids(texts.size());
  {
    std::atomic<size_t> tnext{0};
    const size_t nt = std::min<size_t>(
        std::max(1u, std::thread::hardware_concurrency()), texts.size());
    auto tok = [&]() {
      for (;;) {
        const size_t t = tnext.fetch_add(1);
        if (t >= texts.size()) break;
        all_ids[t] = tokenize(texts[t]);
      }
    };
    std::vector<std::thread> tp;
    for (size_t i = 1; i < nt; ++i) tp.emplace_back(tok);
    tok();
    for (std::thread& th : tp) th.join();
  }
#ifdef RT_CUDA
  // The 12-block BERT is GPU work: a CUDA device encodes thousands of long
  // texts in seconds where a (often cgroup-capped) CPU takes minutes.
  // RT_MINILM_CPU=1 forces the CPU path (parity bisection knob).
  if (std::getenv("RT_MINILM_CPU") == nullptr && detail::cuda_available()) {
    detail::HostWeights hw{im.word_emb, im.pos_emb,   im.type_emb,
                           im.eln_w,    im.eln_b,     im.layers,
                           im.vocab_rows};
    std::string gpu_err;
    if (detail::cuda_encode(hw, all_ids, out, &gpu_err)) {
      (void)normalize;   // pipeline output is always unit-norm (see below)
      return true;
    }
    std::fprintf(stderr,
                 "[minilm] cuda encode failed (%s); falling back to CPU\n",
                 gpu_err.c_str());
  }
#endif
  // Texts are independent (per-call state is all local), so stripe them
  // across hardware threads: encoding is the wall clock of any task that
  // meets fresh cell text, and one core cannot hide a 12-block BERT.
  std::atomic<size_t> next{0};
  std::atomic<bool> failed{false};
  std::mutex err_mu;
  std::string first_err;
  auto worker = [&]() {
  try {
    for (;;) {
      const size_t t = next.fetch_add(1);
      if (t >= texts.size() || failed.load(std::memory_order_relaxed)) break;
      const std::vector<int32_t>& ids = all_ids[t];
      const int S = (int)ids.size();

      std::vector<float> x((size_t)S * D);
      for (int s = 0; s < S; ++s)
        for (int i = 0; i < D; ++i)
          x[(size_t)s * D + i] = im.word_emb[(size_t)ids[s] * D + i] +
                                 im.pos_emb[(size_t)s * D + i] +
                                 im.type_emb[i];   // token_type 0
      layer_norm(x.data(), S, D, im.eln_w, im.eln_b);

      std::vector<float> q((size_t)S * D), k((size_t)S * D), v((size_t)S * D),
          attn((size_t)S * D), scores((size_t)S * S), tmp((size_t)S * FF),
          y((size_t)S * D);
      for (int l = 0; l < 12; ++l) {
        const LayerW& w = im.layers[l];
        gemm_xwt(x.data(), w.q_w, w.q_b, q.data(), S, D, D);
        gemm_xwt(x.data(), w.k_w, w.k_b, k.data(), S, D, D);
        gemm_xwt(x.data(), w.v_w, w.v_b, v.data(), S, D, D);
        // heads: scores = softmax(Q K^T / sqrt(HD)) V, no padding (S is the
        // real length; each text runs alone, so there is no mask).
        const float scale = 1.0f / std::sqrt((float)HD);
        for (int h = 0; h < H; ++h) {
          const int off = h * HD;
          for (int a = 0; a < S; ++a) {
            float* srow = scores.data() + (size_t)a * S;
            const float* qa = q.data() + (size_t)a * D + off;
            float mx = -1e30f;
            for (int b = 0; b < S; ++b) {
              const float* kb = k.data() + (size_t)b * D + off;
              float acc = 0.f;
              for (int i = 0; i < HD; ++i) acc += qa[i] * kb[i];
              srow[b] = acc * scale;
              mx = std::max(mx, srow[b]);
            }
            float sum = 0.f;
            for (int b = 0; b < S; ++b) {
              srow[b] = std::exp(srow[b] - mx);
              sum += srow[b];
            }
            const float inv = 1.0f / sum;
            float* orow = attn.data() + (size_t)a * D + off;
            for (int i = 0; i < HD; ++i) orow[i] = 0.f;
            for (int b = 0; b < S; ++b) {
              const float p = srow[b] * inv;
              const float* vb = v.data() + (size_t)b * D + off;
              for (int i = 0; i < HD; ++i) orow[i] += p * vb[i];
            }
          }
        }
        gemm_xwt(attn.data(), w.ao_w, w.ao_b, y.data(), S, D, D);
        for (size_t i = 0; i < x.size(); ++i) y[i] += x[i];   // residual
        layer_norm(y.data(), S, D, w.aln_w, w.aln_b);

        gemm_xwt(y.data(), w.i_w, w.i_b, tmp.data(), S, D, FF);
        for (size_t i = 0; i < (size_t)S * FF; ++i) tmp[i] = gelu_erf(tmp[i]);
        gemm_xwt(tmp.data(), w.o_w, w.o_b, x.data(), S, FF, D);
        for (size_t i = 0; i < x.size(); ++i) x[i] += y[i];   // residual
        layer_norm(x.data(), S, D, w.oln_w, w.oln_b);
      }

      // mean pooling over the real tokens (mask is all ones per text)
      float* o = out + t * D;
      for (int i = 0; i < D; ++i) o[i] = 0.f;
      for (int s = 0; s < S; ++s)
        for (int i = 0; i < D; ++i) o[i] += x[(size_t)s * D + i];
      for (int i = 0; i < D; ++i) o[i] /= (float)S;
      // The all-MiniLM-L12-v2 pipeline ends in a Normalize module
      // (modules.json), so its output is unit-norm regardless of the
      // caller's normalize_embeddings flag — RT-J was trained on those
      // vectors. `normalize` is therefore honored trivially; both paths
      // produce the pipeline's output. Clamped norm, never 0/0.
      (void)normalize;
      float n = 0.f;
      for (int i = 0; i < D; ++i) n += o[i] * o[i];
      n = std::max(std::sqrt(n), 1e-12f);
      for (int i = 0; i < D; ++i) o[i] /= n;
    }
  } catch (const std::exception& e) {
    failed.store(true);
    std::lock_guard<std::mutex> lock(err_mu);
    if (first_err.empty()) first_err = e.what();
  }
  };
  const size_t n_threads = std::min<size_t>(
      std::max(1u, std::thread::hardware_concurrency()), texts.size());
  std::vector<std::thread> pool;
  pool.reserve(n_threads > 0 ? n_threads - 1 : 0);
  for (size_t i = 1; i < n_threads; ++i) pool.emplace_back(worker);
  worker();
  for (std::thread& th : pool) th.join();
  if (failed.load()) {
    if (err) *err = first_err;
    return false;
  }
  return true;
}

}  // namespace minilm
