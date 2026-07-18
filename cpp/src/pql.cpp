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
      "PREDICT", "FORECAST", "TIMEFRAMES", "FOR", "EACH", "WHERE", "ASSUMING",
      "CLASSIFY", "RANK", "TOP",
      "SUM", "AVG", "MIN", "MAX", "COUNT", "COUNT_DISTINCT", "LIST_DISTINCT",
      "FIRST", "LAST",
      "AND", "OR", "NOT", "IN", "IS", "NULL", "LIKE", "CONTAINS", "STARTS",
      "ENDS", "WITH",
      "SECONDS", "MINUTES", "HOURS", "DAYS", "WEEKS", "MONTHS", "INF"};
  return k;
}

// _SOFT_KEYWORDS = _KEYWORDS - structural/boolean words.
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

const std::unordered_set<std::string>& timeUnitNames() {
  static const std::unordered_set<std::string> s = {
      "SECONDS", "MINUTES", "HOURS", "DAYS", "WEEKS", "MONTHS"};
  return s;
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
  throw PqlError("PQL syntax error" + loc + ": " + message + snippet);
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
      // 1) doubled-quote -> single quote
      std::string dq(2, q);
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
      // 2) remove backslash escapes: \x -> x
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
      static const std::string singles = "><=(),.*+-";
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
// Parser
// ---------------------------------------------------------------------------
AggFunc aggFuncFromName(const std::string& k) {
  static const std::unordered_map<std::string, AggFunc> m = {
      {"SUM", AggFunc::SUM}, {"AVG", AggFunc::AVG}, {"MIN", AggFunc::MIN},
      {"MAX", AggFunc::MAX}, {"COUNT", AggFunc::COUNT},
      {"COUNT_DISTINCT", AggFunc::COUNT_DISTINCT},
      {"LIST_DISTINCT", AggFunc::LIST_DISTINCT}, {"FIRST", AggFunc::FIRST},
      {"LAST", AggFunc::LAST}};
  return m.at(k);
}

TimeUnit timeUnitFromName(const std::string& k) {
  static const std::unordered_map<std::string, TimeUnit> m = {
      {"SECONDS", TimeUnit::SECONDS}, {"MINUTES", TimeUnit::MINUTES},
      {"HOURS", TimeUnit::HOURS},     {"DAYS", TimeUnit::DAYS},
      {"WEEKS", TimeUnit::WEEKS},     {"MONTHS", TimeUnit::MONTHS}};
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

class Parser {
 public:
  explicit Parser(const std::string& text) : text_(text), tokens_(lex(text)) {}

  ParsedQuery parseQuery() {
    expect("PREDICT", "'PREDICT'");
    ParsedQuery q;
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
    if (peek().kind == "FORECAST" && peek(1).kind == "INT") {
      next();
      Token nf = next();
      q.has_num_forecasts = true;
      q.num_forecasts = nf.ival;
      expect("TIMEFRAMES", "'TIMEFRAMES'");
    }
    expect("FOR", "'FOR'");
    // EACH is a soft keyword: only consume as the EACH marker when it is not
    // itself the table name of the entity columnRef (`EACH.x`).
    if (peek().kind == "EACH" && peek(1).kind != ".") next();
    parseColumnRefInto(q.entity_table, q.entity_column);
    if (accept("=")) {
      q.entity_ids.push_back(parseLiteral());
    } else if (peek().kind == "IN" && peek(1).kind == "(") {
      next();
      q.entity_ids = parseListLiteral();
    }
    if (accept("WHERE")) q.where = parseExpr();
    if (accept("ASSUMING")) q.assuming = parseExpr();
    expect("EOF", "end of query");
    return q;
  }

 private:
  const std::string& text_;
  std::vector<Token> tokens_;
  size_t i_ = 0;

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

  // expr precedence: parens > NOT > AND > OR
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
    return parsePrimary();
  }

  ExprPtr parsePrimary() {
    if (accept("(")) {
      ExprPtr inner = parseExpr();
      expect(")", "')'");
      return inner;
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
    ExprPtr value = parseValueExpr();
    const Token& t = peek();
    if (isComparisonSymbol(t.kind)) {
      Operator op = comparisonSymbol(t.kind);
      next();
      auto e = makeCond(value, op);
      e->right = parseLiteral();
      e->has_right = true;
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
    return value;  // bare value predicate (regression target)
  }

  ExprPtr parseValueExpr() {
    const Token& t = peek();
    if (aggFuncNames().count(t.kind) && peek(1).kind == "(")
      return parseAggregation();
    return parseColumnRef();
  }

  ExprPtr parseAggregation() {
    auto e = std::make_shared<Expr>();
    e->kind = ExprKind::Agg;
    e->func = aggFuncFromName(next().kind);
    expect("(", "'('");
    parseColumnRefInto(e->table, e->column);
    if (accept("WHERE")) e->filter = parseExpr();
    if (accept(",")) {
      e->has_window = true;
      double start = parseBound();
      expect(",", "',' between window bounds");
      double end = parseBound();
      TimeUnit unit = TimeUnit::DAYS;
      if (accept(",")) {
        Token ut = next();
        if (!timeUnitNames().count(ut.kind))
          syntaxError("expected a time unit, found " + ut.kind, ut.pos, text_);
        unit = timeUnitFromName(ut.kind);
      }
      e->window.start = start;
      e->window.end = end;
      e->window.unit = unit;
    }
    expect(")", "')' to close aggregation");
    return e;
  }

  double parseBound() {
    double sign = 1.0;
    if (accept("+"))
      sign = 1.0;
    else if (accept("-"))
      sign = -1.0;
    Token t = next();
    if (t.kind == "INT") return sign * (double)t.ival;
    if (t.kind == "INF") return sign * std::numeric_limits<double>::infinity();
    syntaxError("expected a window bound, found " + t.kind, t.pos, text_);
  }

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
  out += "\"}";
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
      if (e.has_right)
        emitLit(out, e.right);
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
  }
}

}  // namespace

ParsedQuery parse(const std::string& query) {
  // Mirror python: empty/whitespace-only query is a syntax error.
  bool blank = true;
  for (char c : query)
    if (!std::isspace((unsigned char)c)) {
      blank = false;
      break;
    }
  if (blank) throw PqlError("PQL syntax error: empty query");
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
    if (t.func == AggFunc::LIST_DISTINCT) return TaskType::MULTILABEL_RANKING;
    // FIRST/LAST: schema-less default is multiclass; else regression.
    if (t.func == AggFunc::FIRST || t.func == AggFunc::LAST)
      return TaskType::MULTICLASS_CLASSIFICATION;
    return TaskType::REGRESSION;
  }
  // bare static column: schema-less default.
  return TaskType::MULTICLASS_CLASSIFICATION;
}

std::string to_json(const ParsedQuery& q) {
  std::string out;
  out += "{\"target\":";
  emitExpr(out, *q.target);
  out += ",\"entity_key\":{\"table\":";
  emitJsonString(out, q.entity_table);
  out += ",\"column\":";
  emitJsonString(out, q.entity_column);
  out += "},\"entity_ids\":[";
  for (size_t k = 0; k < q.entity_ids.size(); k++) {
    if (k) out += ",";
    emitLit(out, q.entity_ids[k]);
  }
  out += "],\"where\":";
  if (q.where)
    emitExpr(out, *q.where);
  else
    out += "null";
  out += ",\"assuming\":";
  if (q.assuming)
    emitExpr(out, *q.assuming);
  else
    out += "null";
  out += ",\"rank\":";
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
