// test_pql.cpp — self-contained conformance test for the C++ RelQL parser.
//
// 1. Parses every line of examples.pql (default ../python/tests/data/examples.pql
//    relative to the build dir; overridable via argv[1]).
// 2. Asserts a fixed set of malformed queries are all rejected.
// Exits nonzero if any assertion fails.

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "pql.hpp"

namespace {

std::string trim(const std::string& s) {
  size_t a = 0, b = s.size();
  while (a < b && (unsigned char)s[a] <= ' ') a++;
  while (b > a && (unsigned char)s[b - 1] <= ' ') b--;
  return s.substr(a, b - a);
}

}  // namespace

int main(int argc, char** argv) {
  std::string path = "../python/tests/data/examples.pql";
  if (argc > 1) path = argv[1];

  std::ifstream in(path);
  if (!in) {
    std::fprintf(stderr, "ERROR: cannot open %s\n", path.c_str());
    return 2;
  }

  std::vector<std::string> lines;
  std::string line;
  while (std::getline(in, line)) {
    std::string t = trim(line);
    if (!t.empty()) lines.push_back(t);
  }

  int parsed = 0;
  bool ok = true;
  bool printedFirst = false;
  for (const std::string& q : lines) {
    try {
      std::string json = pql::parse_to_json(q);
      if (!printedFirst) {
        std::printf("line 1 JSON:\n%s\n\n", json.c_str());
        printedFirst = true;
      }
      parsed++;
    } catch (const std::exception& e) {
      ok = false;
      std::fprintf(stderr, "FAIL (should parse): %s\n  -> %s\n", q.c_str(),
                   e.what());
    }
  }
  std::printf("PASS: %d/%d parsed\n", parsed, (int)lines.size());
  if (parsed != 54) {
    std::fprintf(stderr, "ERROR: expected 54 example queries, got %d\n",
                 (int)lines.size());
    ok = false;
  }

  const std::vector<std::string> bad = {
      "PREDICT FOR EACH e.id",
      "SUM(t.x) OVER (30 DAYS FOLLOWING) FOR EACH e.id",  // missing PREDICT
      "PREDICT SUM(t.x) OVER (30 DAYS FOLLOWING)",        // missing FOR
      "PREDICT SUM(t.x) OVER (30 DAYS FOLLOWING) FOR EACH e",  // no .column
      "PREDICT SUM(t.x, 0, 30) FOR EACH e.id",  // positional form removed
      "PREDICT SUM(t.x) OVER (30 DAYS) FOR EACH e.id",  // no PRECEDING/FOLLOWING
      "PREDICT SUM(t.x) OVER (30 FOLLOWING) FOR EACH e.id",  // missing unit
      "PREDICT SUM(t.x) OVER (0 DAYS FOLLOWING) FOR EACH e.id",  // zero duration
      "PREDICT SUM(t.x) OVER (RANGE BETWEEN 30 DAYS FOLLOWING) FOR EACH e.id",
      "PREDICT SUM(t.x) OVER (30 DAYS FOLLOWING HORIZONS 0) FOR EACH e.id",
      "PREDICT SUM(t.x) OVER (UNBOUNDED PRECEDING HORIZONS 3) FOR EACH e.id",
      "PREDICT SUM(t.x) OVER (RANGE BETWEEN 24 HOURS PRECEDING AND 3 MONTHS "
      "FOLLOWING) FOR EACH e.id",                      // mixed unit domains
      "PREDICT SUM(t.x) OVER undeclared_win FOR EACH e.id",  // undeclared window
      "PREDICT BOGUS(t.x) OVER (30 DAYS FOLLOWING) FOR EACH e.id",  // bad func
      "PREDICT SUM(t.x) OVER (30 DAYS FOLLOWING) FOR EACH e.id WHERE",
      "PREDICT LIST_DISTINCT(t.a) OVER (30 DAYS FOLLOWING) RANK TOP -1 FOR EACH "
      "e.id",
      "PREDICT SUM(t.x) OVER (30 DAYS FOLLOWING) FOR EACH e.id EXTRA JUNK",
  };
  int rejected = 0;
  for (const std::string& q : bad) {
    bool threw = false;
    try {
      pql::parse_to_json(q);
    } catch (const std::exception&) {
      threw = true;
    }
    if (threw) {
      rejected++;
    } else {
      ok = false;
      std::fprintf(stderr, "FAIL (should reject): %s\n", q.c_str());
    }
  }
  std::printf("PASS: rejected %d/%d\n", rejected, (int)bad.size());
  if (rejected != (int)bad.size()) ok = false;

  return ok ? 0 : 1;
}
