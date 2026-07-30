/* test_minilm.cpp — native MiniLM conformance against sentence-transformers.
 *
 * Two tiers in one binary:
 *   - always: C ABI smoke checks that need no model (bad-arg handling).
 *   - goldens: when a snapshot AND cpp/testdata/minilm_golden.json exist
 *     (regenerate with cpp/tools/dump_minilm_golden.py), every golden text
 *     must tokenize to the exact HF ids and embed within tight fp32
 *     tolerance of the reference. Self-skips (exit 0) when either is
 *     missing, like rt_train_test's Metal sections.
 */
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "json.hpp"
#include "minilm.hpp"
#include "minilm_c.h"

namespace {

int checks = 0, fails = 0;

void ok(bool cond, const std::string& what) {
  ++checks;
  if (!cond) {
    ++fails;
    std::printf("FAIL: %s\n", what.c_str());
  }
}

std::string read_file(const std::string& path) {
  std::ifstream f(path);
  if (!f) return "";
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

}  // namespace

int main(int argc, char** argv) {
  // -- C ABI without a model ------------------------------------------------
  {
    char err[256] = {0};
    float out[minilm::kDim];
    int rc = minilm_encode(nullptr, nullptr, 1, 0, out, err, sizeof(err));
    ok(rc != 0, "encode on a null handle is an error");
  }

  const std::string golden_path =
      argc > 1 ? argv[1] : "../testdata/minilm_golden.json";
  const std::string snapshot = minilm::resolve_snapshot();
  const std::string golden_json = read_file(golden_path);
  if (snapshot.empty() || golden_json.empty()) {
    std::printf("PASS: %d checks (SKIP goldens: %s)\n", checks,
                snapshot.empty() ? "no MiniLM snapshot in the HF cache"
                                 : "no testdata/minilm_golden.json");
    return fails ? 1 : 0;
  }

  std::string err;
  auto model = minilm::MiniLM::load(snapshot, &err);
  ok(model != nullptr, "load snapshot: " + err);
  if (!model) return 1;

  relql::JsonValue goldens = relql::json_parse(golden_json);
  int token_mismatches = 0;
  double max_abs = 0.0, min_cos = 1.0;
  std::string worst_tok, worst_emb;
  for (const relql::JsonValue& g : goldens.arr) {
    const std::string& text = g.find("text")->str;
    // tokenizer parity: exact id sequence
    const relql::JsonValue& ids = *g.find("ids");
    std::vector<int32_t> got = model->tokenize(text);
    bool same = got.size() == ids.arr.size();
    for (size_t i = 0; same && i < got.size(); ++i)
      same = got[i] == (int32_t)ids.arr[i].num;
    if (!same) {
      ++token_mismatches;
      if (worst_tok.empty()) worst_tok = text;
    }

    // embedding parity, raw and normalized
    for (const char* which : {"raw", "norm"}) {
      const relql::JsonValue& want = *g.find(which);
      float out[minilm::kDim];
      std::string e;
      ok(model->encode({text}, std::string(which) == "norm", out, &e),
         "encode '" + text.substr(0, 30) + "': " + e);
      double dot = 0, na = 0, nb = 0;
      for (int i = 0; i < minilm::kDim; ++i) {
        const double w = want.arr[i].num;
        const double d = std::abs(out[i] - w);
        if (d > max_abs) {
          max_abs = d;
          worst_emb = text;
        }
        dot += out[i] * w;
        na += out[i] * out[i];
        nb += w * w;
      }
      const double cos = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-12);
      if (cos < min_cos) min_cos = cos;
    }
  }
  ok(token_mismatches == 0,
     "tokenizer ids match HF (" + std::to_string(token_mismatches) +
         " mismatches; first: '" + worst_tok.substr(0, 60) + "')");
  ok(max_abs < 2e-4, "max |delta| " + std::to_string(max_abs) +
                         " < 2e-4 (worst: '" + worst_emb.substr(0, 40) + "')");
  ok(min_cos > 0.9999, "min cosine " + std::to_string(min_cos) + " > 0.9999");
  std::printf("goldens: %zu texts, tokenizer mismatches %d, max|d| %.2e, "
              "min cos %.6f\n",
              goldens.arr.size(), token_mismatches, max_abs, min_cos);

  // C ABI round trip on the loaded snapshot.
  {
    char cerr[512] = {0};
    minilm_t* h = minilm_load(snapshot.c_str(), cerr, sizeof(cerr));
    ok(h != nullptr, std::string("minilm_load: ") + cerr);
    if (h) {
      const char* texts[2] = {"hello", "qty of orders"};
      std::vector<float> out(2 * minilm::kDim);
      int rc = minilm_encode(h, texts, 2, 1, out.data(), cerr, sizeof(cerr));
      ok(rc == 0, std::string("minilm_encode: ") + cerr);
      double n = 0;
      for (int i = 0; i < minilm::kDim; ++i) n += out[i] * out[i];
      ok(std::abs(n - 1.0) < 1e-4, "normalized output has unit norm");
      ok(minilm_dim(h) == minilm::kDim, "minilm_dim");
      minilm_free(h);
    }
  }

  std::printf("%s: %d checks, %d failures\n", fails ? "FAIL" : "PASS", checks,
              fails);
  return fails ? 1 : 0;
}
