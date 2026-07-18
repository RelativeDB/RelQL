#include "pql.hpp"

#include <cctype>
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace pql {
namespace {

// ---------------------------------------------------------------------------
// Lexer
// ---------------------------------------------------------------------------
struct Token {
  std::string kind;   // keyword name | IDENT | INT | FLOAT | STRING | DATE |
                      // op string | EOF
  std::string text;   // original / processed text (STRING body, DATE text,
                      // raw numeric text)
  long long ival = 0; // for INT
  double dval = 0.0;  // for FLOAT
  size_t pos = 0;
};

const std::unordered_set<std::string>& keywords() {
  static const std::unordered_set<std::string> k = {
      // structural / clauses
      "PREDICT", "FOR", "EACH", "WHERE", "ASSUMING", "CLASSIFY", "RANK", "TOP",
      "AS", "OF", "RETURN", "ABLATE", "TABLE", "EXPLAIN", "PLAN", "CONTEXT",
      "ANALYZE", "ABLATION", "FORMAT", "TEXT", "JSON", "WINDOW", "OVER",
      // frames
      "RANGE", "BETWEEN", "PRECEDING", "FOLLOWING", "UNBOUNDED", "NOW",
      "HORIZONS", "STEP",
      // aggregations
      "SUM", "AVG", "MIN", "MAX", "COUNT", "COUNT_DISTINCT", "LIST_DISTINCT",
      "FIRST", "LAST", "EXISTS",
      // value functions / literals
      "CASE", "WHEN", "THEN", "ELSE", "END", "COALESCE", "NULLIF", "ABS", "LOG",
      "EXP", "LEAST", "GREATEST", "TRUE", "FALSE",
      // RETURN outputs
      "EXPECTED", "VALUE", "PROBABILITY", "CLASS", "DISTRIBUTION", "QUANTILES",
      "INTERVAL", "MULTILABEL", "MULTICLASS",
      // boolean / predicate operators
      "AND", "OR", "NOT", "IN", "IS", "NULL", "LIKE", "CONTAINS", "STARTS",
      "ENDS", "WITH",
      // duration units (singular + plural)
      "SECOND", "SECONDS", "MINUTE", "MINUTES", "HOUR", "HOURS", "DAY", "DAYS",
      "WEEK", "WEEKS", "MONTH", "MONTHS", "YEAR", "YEARS"};
  return k;
}

// Soft keywords may also appear as table/column identifiers. Everything except
// the truly structural/boolean words is soft, preserving backward compatibility
// with schemas whose names collide with keywords (e.g. a column named "value").
const std::unordered_set<std::string>& softKeywords() {
  static const std::unordered_set<std::string> s = [] {
    std::unordered_set<std::string> out = keywords();
    for (const char* w : {"PREDICT", "FOR", "WHERE", "ASSUMING", "AND", "OR",
                          "NOT", "NULL"})
      out.erase(w);
    return out;
  }();
  return s;
}

const std::unordered_set<std::string>& aggFuncNames() {
  static const std::unordered_set<std::string> s = {
      "SUM", "AVG", "MIN", "MAX", "COUNT", "COUNT_DISTINCT", "LIST_DISTINCT",
      "FIRST", "LAST"};
  return s;
}

const std::unordered_set<std::string>& valueFuncNames() {
  static const std::unordered_set<std::string> s = {
      "COALESCE", "NULLIF", "ABS", "LOG", "EXP", "LEAST", "GREATEST"};
  return s;
}

// unit keyword -> canonical TimeUnit
const std::unordered_map<std::string, TimeUnit>& unitNames() {
  static const std::unordered_map<std::string, TimeUnit> m = {
      {"SECOND", TimeUnit::SECONDS}, {"SECONDS", TimeUnit::SECONDS},
      {"MINUTE", TimeUnit::MINUTES}, {"MINUTES", TimeUnit::MINUTES},
      {"HOUR", TimeUnit::HOURS},     {"HOURS", TimeUnit::HOURS},
      {"DAY", TimeUnit::DAYS},       {"DAYS", TimeUnit::DAYS},
      {"WEEK", TimeUnit::WEEKS},     {"WEEKS", TimeUnit::WEEKS},
      {"MONTH", TimeUnit::MONTHS},   {"MONTHS", TimeUnit::MONTHS},
      {"YEAR", TimeUnit::YEARS},     {"YEARS", TimeUnit::YEARS}};
  return m;
}

[[noreturn]] void syntaxError(const std::string& message, size_t pos,
                              const std::string& text) {
  std::string loc = " at position " + std::to_string(pos);
  std::string snippet;
  if (!text.empty() && pos <= text.size()) {
    size_t lo = pos >= 10 ? pos - 10 : 0;
    size_t hiLen = 15;
    snippet = ": ..." + text.substr(lo, pos - lo) + ">>>" +
              text.substr(pos, hiLen);
  }
  throw PqlError("RelQL syntax error" + loc + ": " + message + snippet);
}

inline bool isDigit(char c) { return c >= '0' && c <= '9'; }
inline bool isIdentStart(char c) {
  return c == '_' || (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}
inline bool isIdentPart(char c) { return isIdentStart(c) || isDigit(c); }

std::string upper(const std::string& s) {
  std::string r = s;
  for (char& c : r) c = (char)std::toupper((unsigned char)c);
  return r;
}

// Try to match a DATE at pos: \d{4}-\d{2}-\d{2}( \d{2}:\d{2}:\d{2})?
bool matchDate(const std::string& t, size_t pos, size_t& len) {
  size_t n = t.size();
  if (pos + 10 > n) return false;
  auto d = [&](size_t i) { return isDigit(t[i]); };
  if (!(d(pos) && d(pos + 1) && d(pos + 2) && d(pos + 3) && t[pos + 4] == '-' &&
        d(pos + 5) && d(pos + 6) && t[pos + 7] == '-' && d(pos + 8) &&
        d(pos + 9)))
    return false;
  len = 10;
  // optional " HH:MM:SS"
  if (pos + 19 <= n && t[pos + 10] == ' ' && d(pos + 11) && d(pos + 12) &&
      t[pos + 13] == ':' && d(pos + 14) && d(pos + 15) && t[pos + 16] == ':' &&
      d(pos + 17) && d(pos + 18)) {
    len = 19;
  }
  return true;
}

std::vector<Token> lex(const std::string& text) {
  std::vector<Token> tokens;
  size_t pos = 0;
  size_t n = text.size();
  while (pos < n) {
    char c = text[pos];
    // ---- whitespace / comments ----
    if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
      pos++;
      continue;
    }
    if (c == '-' && pos + 1 < n && text[pos + 1] == '-') {  // line comment
      pos += 2;
      while (pos < n && text[pos] != '\r' && text[pos] != '\n') pos++;
      continue;
    }
    if (c == '/' && pos + 1 < n && text[pos + 1] == '*') {  // block comment
      size_t end = text.find("*/", pos + 2);
      if (end == std::string::npos)
        syntaxError("unterminated block comment", pos, text);
      pos = end + 2;
      continue;
    }
    // ---- DATE (before INT/FLOAT) ----
    {
      size_t len;
      if (isDigit(c) && matchDate(text, pos, len)) {
        Token tok;
        tok.kind = "DATE";
        tok.text = text.substr(pos, len);
        tok.pos = pos;
        tokens.push_back(std::move(tok));
        pos += len;
        continue;
      }
    }
    // ---- FLOAT / INT ----
    if (isDigit(c)) {
      size_t start = pos;
      while (pos < n && isDigit(text[pos])) pos++;
      bool isFloat = false;
      if (pos < n && text[pos] == '.' && pos + 1 < n && isDigit(text[pos + 1])) {
        isFloat = true;
        pos++;
        while (pos < n && isDigit(text[pos])) pos++;
      }
      std::string num = text.substr(start, pos - start);
      Token tok;
      tok.pos = start;
      tok.text = num;
      if (isFloat) {
        tok.kind = "FLOAT";
        tok.dval = std::stod(num);
      } else {
        tok.kind = "INT";
        tok.ival = std::stoll(num);
      }
      tokens.push_back(std::move(tok));
      continue;
    }
    // ---- STRING ----
    if (c == '\'' || c == '"') {
      char q = c;
      size_t start = pos;
      size_t i = pos + 1;
      bool closed = false;
      while (i < n) {
        char ch = text[i];
        if (ch == '\\' && i + 1 < n) {
          i += 2;
          continue;
        }
        if (ch == q) {
          if (i + 1 < n && text[i + 1] == q) {  // doubled quote escape
            i += 2;
            continue;
          }
          closed = true;
          break;
        }
        i++;
      }
      if (!closed) syntaxError("unterminated string literal", start, text);
      std::string inner = text.substr(start + 1, i - (start + 1));
      std::string body;
      body.reserve(inner.size());
      for (size_t k = 0; k < inner.size();) {
        if (k + 1 < inner.size() && inner[k] == q && inner[k + 1] == q) {
          body.push_back(q);
          k += 2;
        } else {
          body.push_back(inner[k]);
          k++;
        }
      }
      std::string out;
      out.reserve(body.size());
      for (size_t k = 0; k < body.size();) {
        if (body[k] == '\\' && k + 1 < body.size()) {
          out.push_back(body[k + 1]);
          k += 2;
        } else {
          out.push_back(body[k]);
          k++;
        }
      }
      Token tok;
      tok.kind = "STRING";
      tok.text = out;
      tok.pos = start;
      tokens.push_back(std::move(tok));
      pos = i + 1;
      continue;
    }
    // ---- IDENT / keyword ----
    if (isIdentStart(c)) {
      size_t start = pos;
      while (pos < n && isIdentPart(text[pos])) pos++;
      std::string word = text.substr(start, pos - start);
      std::string up = upper(word);
      Token tok;
      tok.pos = start;
      tok.text = word;
      if (keywords().count(up))
        tok.kind = up;
      else
        tok.kind = "IDENT";
      tokens.push_back(std::move(tok));
      continue;
    }
    // ---- operators ----
    {
      Token tok;
      tok.pos = pos;
      if (pos + 1 < n) {
        std::string two = text.substr(pos, 2);
        if (two == ">=" || two == "<=" || two == "!=" || two == "==") {
          tok.kind = two;
          tok.text = two;
          tokens.push_back(std::move(tok));
          pos += 2;
          continue;
        }
      }
      static const std::string singles = "><=(),.*+-/:%";
      if (singles.find(c) != std::string::npos) {
        tok.kind = std::string(1, c);
        tok.text = tok.kind;
        tokens.push_back(std::move(tok));
        pos++;
        continue;
      }
    }
    syntaxError(std::string("unexpected character '") + c + "'", pos, text);
  }
  Token eof;
  eof.kind = "EOF";
  eof.pos = n;
  tokens.push_back(std::move(eof));
  return tokens;
}

// ---------------------------------------------------------------------------
// Parser helpers
// ---------------------------------------------------------------------------
AggFunc aggFuncFromName(const std::string& k) {
  static const std::unordered_map<std::string, AggFunc> m = {
      {"SUM", AggFunc::SUM}, {"AVG", AggFunc::AVG}, {"MIN", AggFunc::MIN},
      {"MAX", AggFunc::MAX}, {"COUNT", AggFunc::COUNT},
      {"COUNT_DISTINCT", AggFunc::COUNT_DISTINCT},
      {"LIST_DISTINCT", AggFunc::LIST_DISTINCT}, {"FIRST", AggFunc::FIRST},
      {"LAST", AggFunc::LAST}, {"EXISTS", AggFunc::EXISTS}};
  return m.at(k);
}

Operator comparisonSymbol(const std::string& k) {
  if (k == ">") return Operator::GT;
  if (k == "<") return Operator::LT;
  if (k == "=") return Operator::EQ;
  if (k == "==") return Operator::EQ;
  if (k == "!=") return Operator::NEQ;
  if (k == ">=") return Operator::GE;
  if (k == "<=") return Operator::LE;
  return Operator::EQ;  // unreachable
}

bool isComparisonSymbol(const std::string& k) {
  return k == ">" || k == "<" || k == "=" || k == "==" || k == "!=" ||
         k == ">=" || k == "<=";
}

// fixed-duration units convert exactly to seconds; calendar units convert
// exactly to months (1 year = 12 months). A frame may not mix the two domains.
bool isCalendarUnit(TimeUnit u) {
  return u == TimeUnit::MONTHS || u == TimeUnit::YEARS;
}
double unitSeconds(TimeUnit u) {
  switch (u) {
    case TimeUnit::SECONDS: return 1.0;
    case TimeUnit::MINUTES: return 60.0;
    case TimeUnit::HOURS: return 3600.0;
    case TimeUnit::DAYS: return 86400.0;
    case TimeUnit::WEEKS: return 604800.0;
    default: return 0.0;  // calendar
  }
}
double unitMonths(TimeUnit u) {
  return u == TimeUnit::YEARS ? 12.0 : 1.0;  // MONTHS
}

// A single frame endpoint: an offset with an optional unit (NOW / UNBOUNDED
// carry no unit).
struct BoundVal {
  double off = 0.0;   // signed offset in `unit` (or 0 / +/-inf)
  bool finite = true;
  bool has_unit = false;
  TimeUnit unit = TimeUnit::DAYS;
};

class Parser {
 public:
  explicit Parser(const std::string& text) : text_(text), tokens_(lex(text)) {}

  ParsedQuery parseQuery() {
    ParsedQuery q;
    parseExplainPrefix(q);
    expect("PREDICT", "'PREDICT'");
    q.target = parseExpr();
    if (accept("CLASSIFY")) {
      q.rank = RankKind::CLASSIFY;
    } else if (peek().kind == "RANK" && peek(1).kind == "TOP") {
      next();
      next();
      Token k = expect("INT", "an integer after RANK TOP");
      q.rank = RankKind::RANK;
      q.has_top_k = true;
      q.top_k = k.ival;
    }
    expect("FOR", "'FOR'");
    expect("EACH", "'EACH' after 'FOR'");
    parseColumnRefInto(q.entity_table, q.entity_column);
    parseTrailingClauses(q);
    expect("EOF", "end of query");
    resolveWindowRefs(q);
    deriveForecasts(q);
    return q;
  }

 private:
  const std::string& text_;
  std::vector<Token> tokens_;
  size_t i_ = 0;
  std::vector<Expr*> window_ref_sites_;  // aggs with an unresolved OVER <name>

  const Token& peek(size_t offset = 0) const {
    size_t j = i_ + offset;
    if (j >= tokens_.size()) j = tokens_.size() - 1;
    return tokens_[j];
  }
  Token next() {
    Token t = tokens_[i_];
    if (t.kind != "EOF") i_++;
    return t;
  }
  bool accept(const std::string& kind) {
    if (peek().kind == kind) {
      next();
      return true;
    }
    return false;
  }
  Token expect(const std::string& kind, const std::string& what) {
    const Token& t = peek();
    if (t.kind != kind)
      syntaxError("expected " + (what.empty() ? kind : what) + ", found " +
                      t.kind,
                  t.pos, text_);
    return next();
  }

  // ---- prefix / trailing clauses ----------------------------------------
  void parseExplainPrefix(ParsedQuery& q) {
    if (peek().kind != "EXPLAIN") return;
    next();
    q.explain.present = true;
    q.explain.mode = ExplainMode::PLAN;
    if (accept("PLAN"))
      q.explain.mode = ExplainMode::PLAN;
    else if (accept("CONTEXT"))
      q.explain.mode = ExplainMode::CONTEXT;
    else if (accept("ANALYZE"))
      q.explain.mode = ExplainMode::ANALYZE;
    else if (accept("ABLATION"))
      q.explain.mode = ExplainMode::ABLATION;
    if (accept("FORMAT")) {
      if (accept("TEXT"))
        q.explain.format = ExplainFormat::TEXT;
      else if (accept("JSON"))
        q.explain.format = ExplainFormat::JSON;
      else
        syntaxError("expected TEXT or JSON after FORMAT", peek().pos, text_);
    }
  }

  void parseTrailingClauses(ParsedQuery& q) {
    bool have_where = false, have_assuming = false, have_asof = false,
         have_return = false;
    for (;;) {
      const std::string k = peek().kind;
      if (k == "WHERE") {
        if (have_where) syntaxError("duplicate WHERE clause", peek().pos, text_);
        next();
        q.where = parseExpr();
        have_where = true;
      } else if (k == "ASSUMING") {
        if (have_assuming)
          syntaxError("duplicate ASSUMING clause", peek().pos, text_);
        next();
        q.assuming = parseExpr();
        have_assuming = true;
      } else if (k == "AS") {
        if (have_asof) syntaxError("duplicate AS OF clause", peek().pos, text_);
        next();
        expect("OF", "'OF' after AS");
        parseAsOf(q);
        have_asof = true;
      } else if (k == "ABLATE") {
        next();
        expect("TABLE", "'TABLE' after ABLATE");
        Ablation ab;
        ab.name = parseName("a table name after ABLATE TABLE");
        q.ablations.push_back(std::move(ab));
      } else if (k == "RETURN") {
        if (have_return)
          syntaxError("duplicate RETURN clause", peek().pos, text_);
        next();
        parseReturn(q);
        have_return = true;
      } else if (k == "WINDOW") {
        next();
        parseWindowDecl(q);
      } else {
        break;
      }
    }
  }

  void parseAsOf(ParsedQuery& q) {
    q.as_of.present = true;
    if (accept(":")) {
      q.as_of.kind = AnchorKind::PARAM;
      q.as_of.value = parseName("a bind parameter name after ':'");
    } else if (peek().kind == "DATE") {
      q.as_of.kind = AnchorKind::DATE;
      q.as_of.value = next().text;
    } else if (accept("NOW")) {
      q.as_of.kind = AnchorKind::NOW;
    } else {
      syntaxError("expected :param, a DATE, or NOW after AS OF", peek().pos,
                  text_);
    }
  }

  void parseReturn(ParsedQuery& q) {
    q.ret.present = true;
    Token t = next();
    const std::string& k = t.kind;
    if (k == "EXPECTED") {
      expect("VALUE", "'VALUE' after EXPECTED");
      q.ret.kind = ReturnKind::EXPECTED_VALUE;
    } else if (k == "PROBABILITY") {
      q.ret.kind = ReturnKind::PROBABILITY;
    } else if (k == "CLASS") {
      q.ret.kind = ReturnKind::CLASS;
    } else if (k == "DISTRIBUTION") {
      q.ret.kind = ReturnKind::DISTRIBUTION;
    } else if (k == "MULTILABEL") {
      q.ret.kind = ReturnKind::MULTILABEL;
    } else if (k == "MULTICLASS") {
      q.ret.kind = ReturnKind::MULTICLASS;
    } else if (k == "QUANTILES") {
      q.ret.kind = ReturnKind::QUANTILES;
      expect("(", "'(' after QUANTILES");
      q.ret.quantiles.push_back(parseNumber());
      while (accept(",")) q.ret.quantiles.push_back(parseNumber());
      expect(")", "')' to close QUANTILES");
    } else if (k == "INTERVAL") {
      q.ret.kind = ReturnKind::INTERVAL;
      Token iv = expect("INT", "an integer percent after INTERVAL");
      q.ret.has_interval = true;
      q.ret.interval = iv.ival;
      accept("%");  // optional percent sign
    } else {
      syntaxError("expected a RETURN output type, found " + k, t.pos, text_);
    }
  }

  double parseNumber() {
    Token t = next();
    if (t.kind == "INT") return (double)t.ival;
    if (t.kind == "FLOAT") return t.dval;
    syntaxError("expected a number, found " + t.kind, t.pos, text_);
  }

  void parseWindowDecl(ParsedQuery& q) {
    std::string name = parseName("a window name after WINDOW");
    if (q.windows.count(name))
      syntaxError("window '" + name + "' declared more than once", peek().pos,
                  text_);
    expect("AS", "'AS' in WINDOW declaration");
    expect("(", "'(' to open a window spec");
    Window w = parseWindowSpec();
    expect(")", "')' to close a window spec");
    q.windows[name] = w;
  }

  // ---- boolean / value expression ---------------------------------------
  // expr precedence: OR > AND > NOT > predicate; predicate = value [cmp rhs].
  ExprPtr parseExpr() {
    ExprPtr left = parseAnd();
    while (accept("OR")) {
      auto e = std::make_shared<Expr>();
      e->kind = ExprKind::Logic;
      e->bop = BoolOp::OR;
      e->rleft = left;
      e->rright = parseAnd();
      left = e;
    }
    return left;
  }

  ExprPtr parseAnd() {
    ExprPtr left = parseNot();
    while (accept("AND")) {
      auto e = std::make_shared<Expr>();
      e->kind = ExprKind::Logic;
      e->bop = BoolOp::AND;
      e->rleft = left;
      e->rright = parseNot();
      left = e;
    }
    return left;
  }

  ExprPtr parseNot() {
    if (accept("NOT")) {
      auto e = std::make_shared<Expr>();
      e->kind = ExprKind::Not;
      e->inner = parseNot();
      return e;
    }
    return parsePredicate();
  }

  ExprPtr makeCond(ExprPtr value, Operator op) {
    auto e = std::make_shared<Expr>();
    e->kind = ExprKind::Cond;
    e->left = value;
    e->op = op;
    return e;
  }

  ExprPtr parsePredicate() {
    ExprPtr value = parseAddExpr();
    const Token& t = peek();
    if (isComparisonSymbol(t.kind)) {
      Operator op = comparisonSymbol(t.kind);
      next();
      auto e = makeCond(value, op);
      ExprPtr rhs = parseAddExpr();
      if (rhs->kind == ExprKind::Lit) {
        e->right = rhs->lit;
        e->has_right = true;
      } else {
        e->right_expr = rhs;
        e->has_right = true;
      }
      return e;
    }
    if (t.kind == "STARTS") {
      next();
      expect("WITH", "'WITH' after STARTS");
      auto e = makeCond(value, Operator::STARTS_WITH);
      e->right = parseLiteral();
      e->has_right = true;
      return e;
    }
    if (t.kind == "ENDS") {
      next();
      expect("WITH", "'WITH' after ENDS");
      auto e = makeCond(value, Operator::ENDS_WITH);
      e->right = parseLiteral();
      e->has_right = true;
      return e;
    }
    if (t.kind == "CONTAINS") {
      next();
      auto e = makeCond(value, Operator::CONTAINS);
      e->right = parseLiteral();
      e->has_right = true;
      return e;
    }
    if (t.kind == "LIKE") {
      next();
      auto e = makeCond(value, Operator::LIKE);
      e->right = parseLiteral();
      e->has_right = true;
      return e;
    }
    if (t.kind == "NOT" && (peek(1).kind == "CONTAINS" ||
                            peek(1).kind == "LIKE" || peek(1).kind == "IN")) {
      next();
      std::string opTok = next().kind;
      if (opTok == "CONTAINS") {
        auto e = makeCond(value, Operator::NOT_CONTAINS);
        e->right = parseLiteral();
        e->has_right = true;
        return e;
      }
      if (opTok == "LIKE") {
        auto e = makeCond(value, Operator::NOT_LIKE);
        e->right = parseLiteral();
        e->has_right = true;
        return e;
      }
      auto e = makeCond(value, Operator::NOT_IN);
      e->right = listLit(parseListLiteral());
      e->has_right = true;
      return e;
    }
    if (t.kind == "IN") {
      next();
      auto e = makeCond(value, Operator::IN);
      e->right = listLit(parseListLiteral());
      e->has_right = true;
      return e;
    }
    if (t.kind == "IS") {
      if (peek(1).kind == "IN") {
        next();
        next();
        auto e = makeCond(value, Operator::IN);
        e->right = listLit(parseListLiteral());
        e->has_right = true;
        return e;
      }
      next();
      bool negated = accept("NOT");
      expect("NULL", "'NULL'");
      return makeCond(value, negated ? Operator::IS_NOT_NULL
                                     : Operator::IS_NULL);
    }
    return value;  // bare value predicate (regression / value target)
  }

  // arithmetic: + - (lowest), * / (higher), unary -, then primary value.
  ExprPtr parseAddExpr() {
    ExprPtr left = parseMulExpr();
    while (peek().kind == "+" || peek().kind == "-") {
      char op = next().kind[0];
      auto e = std::make_shared<Expr>();
      e->kind = ExprKind::Arith;
      e->arith_op = op;
      e->a_left = left;
      e->a_right = parseMulExpr();
      left = e;
    }
    return left;
  }

  ExprPtr parseMulExpr() {
    ExprPtr left = parseUnary();
    while (peek().kind == "*" || peek().kind == "/") {
      char op = next().kind[0];
      auto e = std::make_shared<Expr>();
      e->kind = ExprKind::Arith;
      e->arith_op = op;
      e->a_left = left;
      e->a_right = parseUnary();
      left = e;
    }
    return left;
  }

  ExprPtr parseUnary() {
    if (peek().kind == "-") {
      next();
      ExprPtr inner = parseUnary();
      if (inner->kind == ExprKind::Lit) {  // fold -literal
        if (inner->lit.kind == LitKind::Int) {
          inner->lit.ival = -inner->lit.ival;
          return inner;
        }
        if (inner->lit.kind == LitKind::Float) {
          inner->lit.dval = -inner->lit.dval;
          inner->lit.sval = "-" + inner->lit.sval;
          return inner;
        }
      }
      auto zero = std::make_shared<Expr>();
      zero->kind = ExprKind::Lit;
      zero->lit.kind = LitKind::Int;
      zero->lit.ival = 0;
      auto e = std::make_shared<Expr>();
      e->kind = ExprKind::Arith;
      e->arith_op = '-';
      e->a_left = zero;
      e->a_right = inner;
      return e;
    }
    if (peek().kind == "+") next();  // unary plus, no-op
    return parsePrimaryValue();
  }

  ExprPtr litExpr(Lit l) {
    auto e = std::make_shared<Expr>();
    e->kind = ExprKind::Lit;
    e->lit = std::move(l);
    return e;
  }

  ExprPtr parsePrimaryValue() {
    const Token& t = peek();
    if (t.kind == "(") {
      next();
      ExprPtr inner = parseExpr();
      expect(")", "')'");
      return inner;
    }
    if (t.kind == "CASE") return parseCase();
    if (valueFuncNames().count(t.kind) && peek(1).kind == "(")
      return parseFunc();
    if (t.kind == "EXISTS" && peek(1).kind == "(") return parseAggregation();
    if (aggFuncNames().count(t.kind) && peek(1).kind == "(")
      return parseAggregation();
    if (t.kind == "TRUE") {
      next();
      Lit l;
      l.kind = LitKind::Bool;
      l.bval = true;
      return litExpr(l);
    }
    if (t.kind == "FALSE") {
      next();
      Lit l;
      l.kind = LitKind::Bool;
      l.bval = false;
      return litExpr(l);
    }
    if (t.kind == "NULL") {
      next();
      Lit l;
      l.kind = LitKind::Null;
      return litExpr(l);
    }
    if (t.kind == "STRING" || t.kind == "DATE" || t.kind == "INT" ||
        t.kind == "FLOAT") {
      return litExpr(parseLiteral());
    }
    return parseColumnRef();
  }

  ExprPtr parseCase() {
    expect("CASE", "'CASE'");
    auto e = std::make_shared<Expr>();
    e->kind = ExprKind::Case;
    if (peek().kind != "WHEN")
      syntaxError("expected WHEN after CASE", peek().pos, text_);
    while (accept("WHEN")) {
      ExprPtr cond = parseExpr();
      expect("THEN", "'THEN' in CASE");
      ExprPtr then = parseAddExpr();
      e->when_conds.push_back(cond);
      e->when_thens.push_back(then);
    }
    if (accept("ELSE")) e->case_else = parseAddExpr();
    expect("END", "'END' to close CASE");
    return e;
  }

  ExprPtr parseFunc() {
    auto e = std::make_shared<Expr>();
    e->kind = ExprKind::Func;
    e->func_name = next().kind;  // canonical uppercase keyword
    expect("(", "'(' after " + e->func_name);
    e->args.push_back(parseAddExpr());
    while (accept(",")) e->args.push_back(parseAddExpr());
    expect(")", "')' to close " + e->func_name);
    size_t nargs = e->args.size();
    if ((e->func_name == "ABS" || e->func_name == "LOG" ||
         e->func_name == "EXP") &&
        nargs != 1)
      syntaxError(e->func_name + " takes exactly 1 argument", peek().pos, text_);
    if (e->func_name == "NULLIF" && nargs != 2)
      syntaxError("NULLIF takes exactly 2 arguments", peek().pos, text_);
    return e;
  }

  ExprPtr parseAggregation() {
    auto e = std::make_shared<Expr>();
    e->kind = ExprKind::Agg;
    e->func = aggFuncFromName(next().kind);
    expect("(", "'('");
    parseColumnRefInto(e->table, e->column);
    if (accept("WHERE")) e->filter = parseExpr();
    expect(")", "')' to close aggregation");
    if (accept("OVER")) {
      if (peek().kind == "(") {
        next();
        e->window = parseWindowSpec();
        e->has_window = true;
        expect(")", "')' to close window spec");
      } else {
        e->window_ref = parseName("a window name after OVER");
        window_ref_sites_.push_back(e.get());
      }
    }
    return e;
  }

  // ---- window frames -----------------------------------------------------
  // parse a duration `<positive-int> <unit>` -> (value, unit)
  void parseDuration(double& value, TimeUnit& unit) {
    Token num = peek();
    if (num.kind != "INT")
      syntaxError("expected a positive number in a duration, found " +
                      num.kind,
                  num.pos, text_);
    if (num.ival <= 0)
      syntaxError("durations must be positive", num.pos, text_);
    next();
    Token u = next();
    auto it = unitNames().find(u.kind);
    if (it == unitNames().end())
      syntaxError("expected a duration unit (e.g. DAYS), found " + u.kind,
                  u.pos, text_);
    value = (double)num.ival;
    unit = it->second;
  }

  BoundVal parseBound() {
    BoundVal b;
    if (accept("NOW")) {
      b.off = 0.0;
      b.finite = true;
      b.has_unit = false;
      return b;
    }
    if (accept("UNBOUNDED")) {
      if (accept("PRECEDING")) {
        b.finite = false;
        b.off = -std::numeric_limits<double>::infinity();
        return b;
      }
      if (accept("FOLLOWING")) {
        b.finite = false;
        b.off = std::numeric_limits<double>::infinity();
        return b;
      }
      syntaxError("expected PRECEDING or FOLLOWING after UNBOUNDED", peek().pos,
                  text_);
    }
    double v;
    TimeUnit u;
    parseDuration(v, u);
    b.has_unit = true;
    b.unit = u;
    b.finite = true;
    if (accept("PRECEDING")) {
      b.off = -v;
      return b;
    }
    if (accept("FOLLOWING")) {
      b.off = v;
      return b;
    }
    syntaxError("expected PRECEDING or FOLLOWING after a duration", peek().pos,
                text_);
  }

  // window_spec := frame [HORIZONS int [STEP duration]]
  Window parseWindowSpec() {
    BoundVal lo, hi;
    if (accept("RANGE")) {
      expect("BETWEEN", "'BETWEEN' after RANGE");
      lo = parseBound();
      expect("AND", "'AND' between window bounds");
      hi = parseBound();
    } else if (peek().kind == "UNBOUNDED") {
      // shorthand: UNBOUNDED PRECEDING => (-inf, NOW]
      lo = parseBound();
      hi.off = 0.0;
      hi.finite = true;
      hi.has_unit = false;
    } else {
      // shorthand: <dur> PRECEDING => (-dur, NOW]; <dur> FOLLOWING => (NOW, +dur]
      BoundVal b = parseBound();
      if (b.off < 0) {
        lo = b;
        hi.off = 0.0;
        hi.finite = true;
        hi.has_unit = false;
      } else {
        lo.off = 0.0;
        lo.finite = true;
        lo.has_unit = false;
        hi = b;
      }
    }

    // normalize lo/hi to a single unit
    TimeUnit unit;
    double lower, upper;
    normalizeFrame(lo, hi, unit, lower, upper);

    Window w;
    w.start = lower;
    w.end = upper;
    w.unit = unit;
    w.horizons = 1;
    w.has_step = false;

    // validate ordering (extended reals: lower must be strictly below upper)
    if (!(lower < upper))
      syntaxError("invalid frame: lower bound must be strictly less than upper",
                  peek().pos, text_);

    if (accept("HORIZONS")) {
      Token h = expect("INT", "a positive integer after HORIZONS");
      if (h.ival < 1)
        syntaxError("HORIZONS must be a positive integer", h.pos, text_);
      w.horizons = h.ival;
      if (accept("STEP")) {
        double sv;
        TimeUnit su;
        parseStepDuration(sv, su);
        // normalize step to the frame unit (same domain required)
        w.step = convertToUnit(sv, su, unit, h.pos);
        w.has_step = true;
      }
      if (w.horizons > 1) {
        if (std::isinf(lower) || std::isinf(upper))
          syntaxError("a multi-horizon frame must have finite bounds", h.pos,
                      text_);
        if (!w.has_step) {
          w.step = upper - lower;  // default stride = frame width
        }
      }
    }
    return w;
  }

  void parseStepDuration(double& value, TimeUnit& unit) {
    parseDuration(value, unit);
  }

  // convert an offset expressed in `from` to `to` (must share a domain)
  double convertToUnit(double v, TimeUnit from, TimeUnit to, size_t pos) {
    if (isCalendarUnit(from) != isCalendarUnit(to))
      syntaxError("cannot mix fixed and calendar duration units in one frame",
                  pos, text_);
    if (from == to) return v;
    if (isCalendarUnit(from))
      return v * (unitMonths(from) / unitMonths(to));
    return v * (unitSeconds(from) / unitSeconds(to));
  }

  // choose a common unit for the two bounds and express offsets in it
  void normalizeFrame(const BoundVal& lo, const BoundVal& hi, TimeUnit& unit,
                      double& lower, double& upper) {
    bool loU = lo.finite && lo.has_unit;
    bool hiU = hi.finite && hi.has_unit;
    if (!loU && !hiU) {
      unit = TimeUnit::DAYS;  // only NOW / UNBOUNDED bounds; unit irrelevant
      lower = lo.off;
      upper = hi.off;
      return;
    }
    // determine domain and detect mixing
    bool anyCal = (loU && isCalendarUnit(lo.unit)) ||
                  (hiU && isCalendarUnit(hi.unit));
    bool anyFixed = (loU && !isCalendarUnit(lo.unit)) ||
                    (hiU && !isCalendarUnit(hi.unit));
    if (anyCal && anyFixed)
      syntaxError("cannot mix fixed and calendar duration units in one frame",
                  peek().pos, text_);
    // pick target unit: smallest fixed present, or MONTHS for calendar
    if (anyCal) {
      unit = TimeUnit::MONTHS;
    } else {
      unit = TimeUnit::WEEKS;  // start large, shrink to smallest present
      auto shrink = [&](TimeUnit u) {
        if (unitSeconds(u) < unitSeconds(unit)) unit = u;
      };
      if (loU) shrink(lo.unit);
      if (hiU) shrink(hi.unit);
    }
    lower = loU ? convertToUnit(lo.off, lo.unit, unit, peek().pos) : lo.off;
    upper = hiU ? convertToUnit(hi.off, hi.unit, unit, peek().pos) : hi.off;
  }

  void resolveWindowRefs(ParsedQuery& q) {
    for (Expr* e : window_ref_sites_) {
      auto it = q.windows.find(e->window_ref);
      if (it == q.windows.end())
        syntaxError("undeclared window '" + e->window_ref + "'", 0, text_);
      e->window = it->second;
      e->has_window = true;
    }
  }

  // ---- shared helpers ----------------------------------------------------
  ExprPtr parseColumnRef() {
    auto e = std::make_shared<Expr>();
    e->kind = ExprKind::Col;
    parseColumnRefInto(e->table, e->column);
    return e;
  }

  void parseColumnRefInto(std::string& table, std::string& column) {
    table = parseName("a table name");
    expect(".", "'.' in table.column reference");
    if (accept("*")) {
      column = "*";
      return;
    }
    column = parseName("a column name");
  }

  std::string parseName(const std::string& what) {
    const Token& t = peek();
    if (t.kind == "IDENT" || softKeywords().count(t.kind)) {
      Token tok = next();
      return tok.text;
    }
    syntaxError("expected " + what + ", found " + t.kind, t.pos, text_);
  }

  std::vector<Lit> parseListLiteral() {
    expect("(", "'(' to open a literal list");
    std::vector<Lit> items;
    items.push_back(parseLiteral());
    while (accept(",")) items.push_back(parseLiteral());
    expect(")", "')' to close a literal list");
    return items;
  }

  static Lit listLit(std::vector<Lit> items) {
    Lit l;
    l.kind = LitKind::List;
    l.items = std::move(items);
    return l;
  }

  Lit parseLiteral() {
    Token t = next();
    Lit l;
    if (t.kind == "STRING") {
      l.kind = LitKind::Str;
      l.sval = t.text;
      return l;
    }
    if (t.kind == "DATE") {
      l.kind = LitKind::Date;
      l.sval = t.text;
      return l;
    }
    if (t.kind == "NULL") {
      l.kind = LitKind::Null;
      return l;
    }
    if (t.kind == "TRUE") {
      l.kind = LitKind::Bool;
      l.bval = true;
      return l;
    }
    if (t.kind == "FALSE") {
      l.kind = LitKind::Bool;
      l.bval = false;
      return l;
    }
    if (t.kind == "+" || t.kind == "-") {
      bool neg = t.kind == "-";
      Token nnum = next();
      if (nnum.kind == "INT") {
        l.kind = LitKind::Int;
        l.ival = neg ? -nnum.ival : nnum.ival;
        return l;
      }
      if (nnum.kind == "FLOAT") {
        l.kind = LitKind::Float;
        l.dval = neg ? -nnum.dval : nnum.dval;
        l.sval = neg ? ("-" + nnum.text) : nnum.text;
        return l;
      }
      syntaxError("expected a number after '" + t.kind + "'", nnum.pos, text_);
    }
    if (t.kind == "INT") {
      l.kind = LitKind::Int;
      l.ival = t.ival;
      return l;
    }
    if (t.kind == "FLOAT") {
      l.kind = LitKind::Float;
      l.dval = t.dval;
      l.sval = t.text;
      return l;
    }
    syntaxError("expected a literal, found " + t.kind, t.pos, text_);
  }

  // ---- post-parse: forecasting derived from target window horizons -------
  static void collectAggs(const Expr* e, std::vector<const Expr*>& out) {
    if (!e) return;
    switch (e->kind) {
      case ExprKind::Agg: out.push_back(e); break;
      case ExprKind::Cond:
        collectAggs(e->left.get(), out);
        collectAggs(e->right_expr.get(), out);
        break;
      case ExprKind::Logic:
        collectAggs(e->rleft.get(), out);
        collectAggs(e->rright.get(), out);
        break;
      case ExprKind::Not: collectAggs(e->inner.get(), out); break;
      case ExprKind::Arith:
        collectAggs(e->a_left.get(), out);
        collectAggs(e->a_right.get(), out);
        break;
      case ExprKind::Func:
        for (const auto& a : e->args) collectAggs(a.get(), out);
        break;
      case ExprKind::Case:
        for (const auto& c : e->when_conds) collectAggs(c.get(), out);
        for (const auto& th : e->when_thens) collectAggs(th.get(), out);
        collectAggs(e->case_else.get(), out);
        break;
      default: break;
    }
  }

  void deriveForecasts(ParsedQuery& q) {
    std::vector<const Expr*> aggs;
    collectAggs(q.target.get(), aggs);
    for (const Expr* a : aggs) {
      if (a->has_window && a->window.horizons > 1) {
        q.has_num_forecasts = true;
        q.num_forecasts = a->window.horizons;
        break;
      }
    }
  }
};

// ---------------------------------------------------------------------------
// JSON emitter
// ---------------------------------------------------------------------------
void emitJsonString(std::string& out, const std::string& s) {
  out.push_back('"');
  for (unsigned char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\b': out += "\\b"; break;
      case '\f': out += "\\f"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (c < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          out += buf;
        } else {
          out.push_back((char)c);
        }
    }
  }
  out.push_back('"');
}

const char* aggFuncName(AggFunc f) {
  switch (f) {
    case AggFunc::SUM: return "SUM";
    case AggFunc::AVG: return "AVG";
    case AggFunc::MIN: return "MIN";
    case AggFunc::MAX: return "MAX";
    case AggFunc::COUNT: return "COUNT";
    case AggFunc::COUNT_DISTINCT: return "COUNT_DISTINCT";
    case AggFunc::LIST_DISTINCT: return "LIST_DISTINCT";
    case AggFunc::FIRST: return "FIRST";
    case AggFunc::LAST: return "LAST";
    case AggFunc::EXISTS: return "EXISTS";
  }
  return "";
}

const char* operatorName(Operator o) {
  switch (o) {
    case Operator::GT: return "GT";
    case Operator::LT: return "LT";
    case Operator::EQ: return "EQ";
    case Operator::NEQ: return "NEQ";
    case Operator::GE: return "GE";
    case Operator::LE: return "LE";
    case Operator::STARTS_WITH: return "STARTS_WITH";
    case Operator::ENDS_WITH: return "ENDS_WITH";
    case Operator::CONTAINS: return "CONTAINS";
    case Operator::NOT_CONTAINS: return "NOT_CONTAINS";
    case Operator::LIKE: return "LIKE";
    case Operator::NOT_LIKE: return "NOT_LIKE";
    case Operator::IN: return "IN";
    case Operator::NOT_IN: return "NOT_IN";
    case Operator::IS_NULL: return "IS_NULL";
    case Operator::IS_NOT_NULL: return "IS_NOT_NULL";
  }
  return "";
}

const char* timeUnitName(TimeUnit u) {
  switch (u) {
    case TimeUnit::SECONDS: return "seconds";
    case TimeUnit::MINUTES: return "minutes";
    case TimeUnit::HOURS: return "hours";
    case TimeUnit::DAYS: return "days";
    case TimeUnit::WEEKS: return "weeks";
    case TimeUnit::MONTHS: return "months";
    case TimeUnit::YEARS: return "years";
  }
  return "";
}

const char* taskTypeName(TaskType t) {
  switch (t) {
    case TaskType::REGRESSION: return "regression";
    case TaskType::BINARY_CLASSIFICATION: return "binary_classification";
    case TaskType::MULTICLASS_CLASSIFICATION:
      return "multiclass_classification";
    case TaskType::MULTILABEL_RANKING: return "multilabel_ranking";
    case TaskType::FORECASTING: return "forecasting";
  }
  return "";
}

const char* returnKindName(ReturnKind k) {
  switch (k) {
    case ReturnKind::EXPECTED_VALUE: return "EXPECTED_VALUE";
    case ReturnKind::PROBABILITY: return "PROBABILITY";
    case ReturnKind::CLASS: return "CLASS";
    case ReturnKind::DISTRIBUTION: return "DISTRIBUTION";
    case ReturnKind::QUANTILES: return "QUANTILES";
    case ReturnKind::INTERVAL: return "INTERVAL";
    case ReturnKind::MULTILABEL: return "MULTILABEL";
    case ReturnKind::MULTICLASS: return "MULTICLASS";
  }
  return "";
}

void emitBound(std::string& out, double v) {
  if (std::isinf(v)) {
    out += v > 0 ? "\"inf\"" : "\"-inf\"";
  } else {
    out += std::to_string((long long)v);
  }
}

void emitLit(std::string& out, const Lit& l) {
  switch (l.kind) {
    case LitKind::Int:
      out += std::to_string(l.ival);
      break;
    case LitKind::Float:
      out += l.sval;  // raw numeric text (valid JSON number)
      break;
    case LitKind::Str:
      emitJsonString(out, l.sval);
      break;
    case LitKind::Bool:
      out += l.bval ? "true" : "false";
      break;
    case LitKind::Null:
      out += "null";
      break;
    case LitKind::Date:
      out += "{\"date\":";
      emitJsonString(out, l.sval);
      out += "}";
      break;
    case LitKind::List: {
      out += "[";
      for (size_t k = 0; k < l.items.size(); k++) {
        if (k) out += ",";
        emitLit(out, l.items[k]);
      }
      out += "]";
      break;
    }
  }
}

void emitWindow(std::string& out, const Window& w) {
  out += "{\"start\":";
  emitBound(out, w.start);
  out += ",\"end\":";
  emitBound(out, w.end);
  out += ",\"unit\":\"";
  out += timeUnitName(w.unit);
  out += "\",\"horizons\":";
  out += std::to_string(w.horizons);
  out += ",\"step\":";
  if (w.has_step)
    out += std::to_string((long long)w.step);
  else
    out += "null";
  out += "}";
}

void emitExpr(std::string& out, const Expr& e) {
  switch (e.kind) {
    case ExprKind::Agg:
      out += "{\"kind\":\"agg\",\"func\":\"";
      out += aggFuncName(e.func);
      out += "\",\"column\":{\"table\":";
      emitJsonString(out, e.table);
      out += ",\"column\":";
      emitJsonString(out, e.column);
      out += "},\"filter\":";
      if (e.filter)
        emitExpr(out, *e.filter);
      else
        out += "null";
      out += ",\"window\":";
      if (e.has_window)
        emitWindow(out, e.window);
      else
        out += "null";
      out += "}";
      break;
    case ExprKind::Col:
      out += "{\"kind\":\"col\",\"table\":";
      emitJsonString(out, e.table);
      out += ",\"column\":";
      emitJsonString(out, e.column);
      out += "}";
      break;
    case ExprKind::Cond:
      out += "{\"kind\":\"cond\",\"left\":";
      emitExpr(out, *e.left);
      out += ",\"op\":\"";
      out += operatorName(e.op);
      out += "\",\"right\":";
      if (e.right_expr)
        out += "null";
      else if (e.has_right)
        emitLit(out, e.right);
      else
        out += "null";
      out += ",\"right_expr\":";
      if (e.right_expr)
        emitExpr(out, *e.right_expr);
      else
        out += "null";
      out += "}";
      break;
    case ExprKind::Logic:
      out += "{\"kind\":\"logic\",\"op\":\"";
      out += (e.bop == BoolOp::AND ? "AND" : "OR");
      out += "\",\"left\":";
      emitExpr(out, *e.rleft);
      out += ",\"right\":";
      emitExpr(out, *e.rright);
      out += "}";
      break;
    case ExprKind::Not:
      out += "{\"kind\":\"not\",\"expr\":";
      emitExpr(out, *e.inner);
      out += "}";
      break;
    case ExprKind::Arith:
      out += "{\"kind\":\"arith\",\"op\":\"";
      out.push_back(e.arith_op);
      out += "\",\"left\":";
      emitExpr(out, *e.a_left);
      out += ",\"right\":";
      emitExpr(out, *e.a_right);
      out += "}";
      break;
    case ExprKind::Func:
      out += "{\"kind\":\"func\",\"name\":";
      emitJsonString(out, e.func_name);
      out += ",\"args\":[";
      for (size_t k = 0; k < e.args.size(); k++) {
        if (k) out += ",";
        emitExpr(out, *e.args[k]);
      }
      out += "]}";
      break;
    case ExprKind::Case:
      out += "{\"kind\":\"case\",\"whens\":[";
      for (size_t k = 0; k < e.when_conds.size(); k++) {
        if (k) out += ",";
        out += "{\"cond\":";
        emitExpr(out, *e.when_conds[k]);
        out += ",\"then\":";
        emitExpr(out, *e.when_thens[k]);
        out += "}";
      }
      out += "],\"else\":";
      if (e.case_else)
        emitExpr(out, *e.case_else);
      else
        out += "null";
      out += "}";
      break;
    case ExprKind::Lit:
      out += "{\"kind\":\"lit\",\"value\":";
      emitLit(out, e.lit);
      out += "}";
      break;
  }
}

const char* explainModeName(ExplainMode m) {
  switch (m) {
    case ExplainMode::PLAN: return "PLAN";
    case ExplainMode::CONTEXT: return "CONTEXT";
    case ExplainMode::ANALYZE: return "ANALYZE";
    case ExplainMode::ABLATION: return "ABLATION";
    case ExplainMode::NONE: return "PLAN";
  }
  return "PLAN";
}

}  // namespace

ParsedQuery parse(const std::string& query) {
  bool blank = true;
  for (char c : query)
    if (!std::isspace((unsigned char)c)) {
      blank = false;
      break;
    }
  if (blank) throw PqlError("RelQL syntax error: empty query");
  Parser p(query);
  return p.parseQuery();
}

TaskType task_type(const ParsedQuery& q) {
  if (q.has_num_forecasts) return TaskType::FORECASTING;
  if (q.rank == RankKind::RANK) return TaskType::MULTILABEL_RANKING;
  if (q.rank == RankKind::CLASSIFY) return TaskType::MULTICLASS_CLASSIFICATION;
  const Expr& t = *q.target;
  if (t.kind == ExprKind::Cond || t.kind == ExprKind::Logic ||
      t.kind == ExprKind::Not)
    return TaskType::BINARY_CLASSIFICATION;
  if (t.kind == ExprKind::Agg) {
    if (t.func == AggFunc::EXISTS) return TaskType::BINARY_CLASSIFICATION;
    if (t.func == AggFunc::LIST_DISTINCT) return TaskType::MULTILABEL_RANKING;
    if (t.func == AggFunc::FIRST || t.func == AggFunc::LAST)
      return TaskType::MULTICLASS_CLASSIFICATION;
    return TaskType::REGRESSION;
  }
  if (t.kind == ExprKind::Arith || t.kind == ExprKind::Func ||
      t.kind == ExprKind::Case)
    return TaskType::REGRESSION;
  if (t.kind == ExprKind::Lit) {
    if (t.lit.kind == LitKind::Bool) return TaskType::BINARY_CLASSIFICATION;
    return TaskType::REGRESSION;
  }
  // bare static column: schema-less default.
  return TaskType::MULTICLASS_CLASSIFICATION;
}

std::string to_json(const ParsedQuery& q) {
  std::string out;
  out += "{\"explain\":";
  if (q.explain.present) {
    out += "{\"mode\":\"";
    out += explainModeName(q.explain.mode);
    out += "\",\"format\":\"";
    out += (q.explain.format == ExplainFormat::JSON ? "JSON" : "TEXT");
    out += "\"}";
  } else {
    out += "null";
  }
  out += ",\"target\":";
  emitExpr(out, *q.target);
  out += ",\"entity_key\":{\"table\":";
  emitJsonString(out, q.entity_table);
  out += ",\"column\":";
  emitJsonString(out, q.entity_column);
  out += "},\"where\":";
  if (q.where)
    emitExpr(out, *q.where);
  else
    out += "null";
  out += ",\"assuming\":";
  if (q.assuming)
    emitExpr(out, *q.assuming);
  else
    out += "null";
  out += ",\"as_of\":";
  if (q.as_of.present) {
    out += "{\"kind\":\"";
    out += (q.as_of.kind == AnchorKind::PARAM
                ? "param"
                : (q.as_of.kind == AnchorKind::DATE ? "date" : "now"));
    out += "\",\"value\":";
    if (q.as_of.kind == AnchorKind::NOW)
      out += "null";
    else
      emitJsonString(out, q.as_of.value);
    out += "}";
  } else {
    out += "null";
  }
  out += ",\"ablations\":[";
  for (size_t k = 0; k < q.ablations.size(); k++) {
    if (k) out += ",";
    out += "{\"kind\":\"table\",\"name\":";
    emitJsonString(out, q.ablations[k].name);
    out += "}";
  }
  out += "],\"ret\":";
  if (q.ret.present) {
    out += "{\"kind\":\"";
    out += returnKindName(q.ret.kind);
    out += "\",\"quantiles\":[";
    for (size_t k = 0; k < q.ret.quantiles.size(); k++) {
      if (k) out += ",";
      char buf[32];
      std::snprintf(buf, sizeof(buf), "%g", q.ret.quantiles[k]);
      out += buf;
    }
    out += "],\"interval\":";
    out += q.ret.has_interval ? std::to_string(q.ret.interval) : "null";
    out += "}";
  } else {
    out += "null";
  }
  out += ",\"windows\":{";
  {
    size_t k = 0;
    for (const auto& kv : q.windows) {
      if (k++) out += ",";
      emitJsonString(out, kv.first);
      out += ":";
      emitWindow(out, kv.second);
    }
  }
  out += "},\"rank\":";
  if (q.rank == RankKind::CLASSIFY)
    out += "\"CLASSIFY\"";
  else if (q.rank == RankKind::RANK)
    out += "\"RANK\"";
  else
    out += "null";
  out += ",\"top_k\":";
  out += q.has_top_k ? std::to_string(q.top_k) : "null";
  out += ",\"num_forecasts\":";
  out += q.has_num_forecasts ? std::to_string(q.num_forecasts) : "null";
  out += ",\"task_type\":\"";
  out += taskTypeName(task_type(q));
  out += "\"}";
  return out;
}

std::string parse_to_json(const std::string& query) {
  return to_json(parse(query));
}

}  // namespace pql
