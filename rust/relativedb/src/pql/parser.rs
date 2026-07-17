//! Hand-written recursive-descent PQL parser, faithful to `grammar/Pql.g4`.
//!
//! Chosen over an ANTLR runtime to avoid a codegen/runtime dependency; the
//! grammar is small and stable. The full 44-query corpus in
//! `grammar/examples.pql` is the conformance suite (see tests). This mirrors
//! the Python `relativedb.pql.parser` line for line.

use std::fmt;

use chrono::{DateTime, NaiveDate, NaiveDateTime, Utc};

use super::ast::{
    AggFunc, Aggregation, BoolOp, ColumnRef, CondRhs, Condition, Literal, LogicalOp, Operator,
    ParsedQuery, RankKind, TargetExpr, TimeUnit, Window,
};
use crate::schema::Schema;

/// A positioned PQL syntax error.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct SyntaxError {
    pub message: String,
    pub pos: isize,
    pub rendered: String,
}

impl SyntaxError {
    fn new(message: impl Into<String>, pos: isize, text: &[char]) -> SyntaxError {
        let message = message.into();
        let loc = if pos >= 0 {
            format!(" at position {}", pos)
        } else {
            String::new()
        };
        let mut snippet = String::new();
        if !text.is_empty() && pos >= 0 && (pos as usize) <= text.len() {
            let p = pos as usize;
            let lo = p.saturating_sub(10);
            let before: String = text[lo..p].iter().collect();
            let after: String = text[p..(p + 15).min(text.len())].iter().collect();
            snippet = format!(": ...{}>>>{}", before, after);
        }
        let rendered = format!("PQL syntax error{}: {}{}", loc, message, snippet);
        SyntaxError { message, pos, rendered }
    }
}

impl fmt::Display for SyntaxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.rendered)
    }
}
impl std::error::Error for SyntaxError {}

/// A schema-binding validation error.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ValidationError(pub String);

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PQL validation error: {}", self.0)
    }
}
impl std::error::Error for ValidationError {}

// ---------------------------------------------------------------------------
// Lexer
// ---------------------------------------------------------------------------

const KEYWORDS: &[&str] = &[
    "PREDICT", "FORECAST", "TIMEFRAMES", "FOR", "EACH", "WHERE", "ASSUMING", "CLASSIFY", "RANK",
    "TOP", "SUM", "AVG", "MIN", "MAX", "COUNT", "COUNT_DISTINCT", "LIST_DISTINCT", "FIRST", "LAST",
    "AND", "OR", "NOT", "IN", "IS", "NULL", "LIKE", "CONTAINS", "STARTS", "ENDS", "WITH", "SECONDS",
    "MINUTES", "HOURS", "DAYS", "WEEKS", "MONTHS", "INF",
];

// Structural clause words + boolean/null words excluded from softKeyword.
const NON_SOFT: &[&str] = &["PREDICT", "FOR", "WHERE", "ASSUMING", "AND", "OR", "NOT", "NULL"];
const AGG_FUNCS: &[&str] = &[
    "SUM", "AVG", "MIN", "MAX", "COUNT", "COUNT_DISTINCT", "LIST_DISTINCT", "FIRST", "LAST",
];
const TIME_UNITS: &[&str] = &["SECONDS", "MINUTES", "HOURS", "DAYS", "WEEKS", "MONTHS"];

fn is_keyword(s: &str) -> bool {
    KEYWORDS.contains(&s)
}
fn is_soft_keyword(s: &str) -> bool {
    is_keyword(s) && !NON_SOFT.contains(&s)
}
fn is_agg_func(s: &str) -> bool {
    AGG_FUNCS.contains(&s)
}
fn is_time_unit(s: &str) -> bool {
    TIME_UNITS.contains(&s)
}

#[derive(Clone, Debug)]
enum TokVal {
    None,
    Int(i64),
    Float(f64),
    Str(String),
}

#[derive(Clone, Debug)]
struct Token {
    kind: String, // keyword name | IDENT | INT | FLOAT | STRING | DATE | op | EOF
    val: TokVal,
    pos: usize,
}

fn is_ident_start(c: char) -> bool {
    c.is_ascii_alphabetic() || c == '_'
}
fn is_ident_part(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

fn matches_date(chars: &[char], i: usize) -> bool {
    // YYYY-MM-DD : 4 digits '-' 2 digits '-' 2 digits
    if i + 10 > chars.len() {
        return false;
    }
    let d = |k: usize| chars[i + k].is_ascii_digit();
    d(0) && d(1) && d(2) && d(3)
        && chars[i + 4] == '-'
        && d(5) && d(6)
        && chars[i + 7] == '-'
        && d(8) && d(9)
}

fn matches_time_suffix(chars: &[char], i: usize) -> bool {
    // " HH:MM:SS" starting at i (a literal space)
    if i + 9 > chars.len() {
        return false;
    }
    let d = |k: usize| chars[i + k].is_ascii_digit();
    chars[i] == ' '
        && d(1) && d(2)
        && chars[i + 3] == ':'
        && d(4) && d(5)
        && chars[i + 6] == ':'
        && d(7) && d(8)
}

fn lex(text: &[char]) -> Result<Vec<Token>, SyntaxError> {
    let n = text.len();
    let mut tokens: Vec<Token> = Vec::new();
    let mut i = 0usize;
    while i < n {
        let c = text[i];
        if c.is_whitespace() {
            i += 1;
            continue;
        }
        // line comment `-- ... EOL`
        if c == '-' && i + 1 < n && text[i + 1] == '-' {
            i += 2;
            while i < n && text[i] != '\n' && text[i] != '\r' {
                i += 1;
            }
            continue;
        }
        // block comment `/* ... */`
        if c == '/' && i + 1 < n && text[i + 1] == '*' {
            i += 2;
            while i + 1 < n && !(text[i] == '*' && text[i + 1] == '/') {
                i += 1;
            }
            i = (i + 2).min(n);
            continue;
        }
        let start = i;
        if c.is_ascii_digit() && matches_date(text, i) {
            let mut end = i + 10;
            if matches_time_suffix(text, end) {
                end += 9;
            }
            let s: String = text[i..end].iter().collect();
            tokens.push(Token { kind: "DATE".into(), val: TokVal::Str(s), pos: start });
            i = end;
        } else if c.is_ascii_digit() {
            let mut j = i;
            while j < n && text[j].is_ascii_digit() {
                j += 1;
            }
            if j < n && text[j] == '.' && j + 1 < n && text[j + 1].is_ascii_digit() {
                j += 1;
                while j < n && text[j].is_ascii_digit() {
                    j += 1;
                }
                let s: String = text[i..j].iter().collect();
                let f: f64 = s.parse().unwrap();
                tokens.push(Token { kind: "FLOAT".into(), val: TokVal::Float(f), pos: start });
            } else {
                let s: String = text[i..j].iter().collect();
                let v: i64 = s.parse().map_err(|_| {
                    SyntaxError::new(format!("integer literal out of range: {}", s), start as isize, text)
                })?;
                tokens.push(Token { kind: "INT".into(), val: TokVal::Int(v), pos: start });
            }
            i = j;
        } else if c == '\'' || c == '"' {
            let q = c;
            let mut j = i + 1;
            loop {
                if j >= n {
                    return Err(SyntaxError::new("unterminated string literal", start as isize, text));
                }
                let cj = text[j];
                if cj == '\\' {
                    j += 2;
                    continue;
                }
                if cj == q {
                    if j + 1 < n && text[j + 1] == q {
                        j += 2; // doubled-quote escape
                        continue;
                    }
                    break;
                }
                j += 1;
            }
            // decode inner [i+1, j)
            let mut s = String::new();
            let mut k = i + 1;
            while k < j {
                let ck = text[k];
                if ck == '\\' && k + 1 < j {
                    s.push(text[k + 1]);
                    k += 2;
                } else if ck == q && k + 1 < j && text[k + 1] == q {
                    s.push(q);
                    k += 2;
                } else {
                    s.push(ck);
                    k += 1;
                }
            }
            tokens.push(Token { kind: "STRING".into(), val: TokVal::Str(s), pos: start });
            i = j + 1;
        } else if is_ident_start(c) {
            let mut j = i;
            while j < n && is_ident_part(text[j]) {
                j += 1;
            }
            let raw: String = text[i..j].iter().collect();
            let upper = raw.to_ascii_uppercase();
            if is_keyword(&upper) {
                tokens.push(Token { kind: upper, val: TokVal::Str(raw), pos: start });
            } else {
                tokens.push(Token { kind: "IDENT".into(), val: TokVal::Str(raw), pos: start });
            }
            i = j;
        } else {
            // operators / punctuation
            let two: Option<&str> = if i + 1 < n {
                match (c, text[i + 1]) {
                    ('>', '=') => Some(">="),
                    ('<', '=') => Some("<="),
                    ('!', '=') => Some("!="),
                    ('=', '=') => Some("=="),
                    _ => None,
                }
            } else {
                None
            };
            if let Some(op) = two {
                tokens.push(Token { kind: op.into(), val: TokVal::None, pos: start });
                i += 2;
            } else if "><=(),.*+-".contains(c) {
                tokens.push(Token { kind: c.to_string(), val: TokVal::None, pos: start });
                i += 1;
            } else {
                return Err(SyntaxError::new(
                    format!("unexpected character {:?}", c),
                    start as isize,
                    text,
                ));
            }
        }
    }
    tokens.push(Token { kind: "EOF".into(), val: TokVal::None, pos: n });
    Ok(tokens)
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

fn comparison_symbol(kind: &str) -> Option<Operator> {
    Some(match kind {
        ">" => Operator::Gt,
        "<" => Operator::Lt,
        "=" => Operator::Eq,
        "==" => Operator::Eq,
        "!=" => Operator::Neq,
        ">=" => Operator::Ge,
        "<=" => Operator::Le,
        _ => return None,
    })
}

struct Parser {
    text: Vec<char>,
    tokens: Vec<Token>,
    i: usize,
}

impl Parser {
    fn new(text: &str) -> Result<Parser, SyntaxError> {
        let chars: Vec<char> = text.chars().collect();
        let tokens = lex(&chars)?;
        Ok(Parser { text: chars, tokens, i: 0 })
    }

    fn peek(&self, offset: usize) -> &Token {
        let j = (self.i + offset).min(self.tokens.len() - 1);
        &self.tokens[j]
    }
    fn kind(&self, offset: usize) -> String {
        self.peek(offset).kind.clone()
    }

    fn advance(&mut self) -> Token {
        let t = self.tokens[self.i].clone();
        if t.kind != "EOF" {
            self.i += 1;
        }
        t
    }

    fn accept(&mut self, kind: &str) -> Option<Token> {
        if self.peek(0).kind == kind {
            Some(self.advance())
        } else {
            None
        }
    }

    fn expect(&mut self, kind: &str, what: &str) -> Result<Token, SyntaxError> {
        let t = self.peek(0);
        if t.kind != kind {
            let what = if what.is_empty() { kind } else { what };
            return Err(SyntaxError::new(
                format!("expected {}, found {}", what, t.kind),
                t.pos as isize,
                &self.text,
            ));
        }
        Ok(self.advance())
    }

    // -- grammar ------------------------------------------------------------

    fn parse_query(&mut self) -> Result<ParsedQuery, SyntaxError> {
        self.expect("PREDICT", "'PREDICT'")?;
        let target = self.parse_expr()?;
        let mut rank: Option<RankKind> = None;
        let mut top_k: Option<i64> = None;
        if self.accept("CLASSIFY").is_some() {
            rank = Some(RankKind::Classify);
        } else if self.kind(0) == "RANK" && self.kind(1) == "TOP" {
            self.advance();
            self.advance();
            let t = self.expect("INT", "an integer after RANK TOP")?;
            top_k = Some(match t.val {
                TokVal::Int(v) => v,
                _ => unreachable!(),
            });
            rank = Some(RankKind::Rank);
        }
        let mut num_forecasts: Option<i64> = None;
        if self.kind(0) == "FORECAST" && self.kind(1) == "INT" {
            self.advance();
            let t = self.advance();
            num_forecasts = Some(match t.val {
                TokVal::Int(v) => v,
                _ => unreachable!(),
            });
            self.expect("TIMEFRAMES", "'TIMEFRAMES'")?;
        }
        self.expect("FOR", "'FOR'")?;
        // EACH is a soft keyword: only consume it as the EACH marker when it is
        // not itself the table name of the entity columnRef (`EACH.x`).
        if self.kind(0) == "EACH" && self.kind(1) != "." {
            self.advance();
        }
        let entity_key = self.parse_column_ref()?;
        let mut entity_ids: Vec<Literal> = Vec::new();
        if self.accept("=").is_some() {
            entity_ids.push(self.parse_literal()?);
        } else if self.kind(0) == "IN" && self.kind(1) == "(" {
            self.advance();
            entity_ids = self.parse_list_literal()?;
        }
        let where_ = if self.accept("WHERE").is_some() {
            Some(self.parse_expr()?)
        } else {
            None
        };
        let assuming = if self.accept("ASSUMING").is_some() {
            Some(self.parse_expr()?)
        } else {
            None
        };
        self.expect("EOF", "end of query")?;
        Ok(ParsedQuery {
            target,
            entity_key,
            entity_ids,
            where_,
            assuming,
            rank,
            top_k,
            num_forecasts,
            text: self.text.iter().collect(),
        })
    }

    // expr precedence: parens > NOT > AND > OR
    fn parse_expr(&mut self) -> Result<TargetExpr, SyntaxError> {
        let mut left = self.parse_and()?;
        while self.accept("OR").is_some() {
            let right = self.parse_and()?;
            left = TargetExpr::LogicalOp(LogicalOp {
                left: Box::new(left),
                op: BoolOp::Or,
                right: Box::new(right),
            });
        }
        Ok(left)
    }

    fn parse_and(&mut self) -> Result<TargetExpr, SyntaxError> {
        let mut left = self.parse_not()?;
        while self.accept("AND").is_some() {
            let right = self.parse_not()?;
            left = TargetExpr::LogicalOp(LogicalOp {
                left: Box::new(left),
                op: BoolOp::And,
                right: Box::new(right),
            });
        }
        Ok(left)
    }

    fn parse_not(&mut self) -> Result<TargetExpr, SyntaxError> {
        if self.accept("NOT").is_some() {
            let inner = self.parse_not()?;
            return Ok(TargetExpr::Not(Box::new(inner)));
        }
        self.parse_primary()
    }

    fn parse_primary(&mut self) -> Result<TargetExpr, SyntaxError> {
        if self.accept("(").is_some() {
            let inner = self.parse_expr()?;
            self.expect(")", "')'")?;
            return Ok(inner);
        }
        self.parse_predicate()
    }

    fn parse_predicate(&mut self) -> Result<TargetExpr, SyntaxError> {
        let value = self.parse_value_expr()?;
        let k = self.kind(0);
        // symbol comparisons
        if let Some(op) = comparison_symbol(&k) {
            self.advance();
            let rhs = self.parse_literal()?;
            return Ok(cond(value, op, CondRhs::One(rhs)));
        }
        match k.as_str() {
            "STARTS" => {
                self.advance();
                self.expect("WITH", "'WITH' after STARTS")?;
                let rhs = self.parse_literal()?;
                Ok(cond(value, Operator::StartsWith, CondRhs::One(rhs)))
            }
            "ENDS" => {
                self.advance();
                self.expect("WITH", "'WITH' after ENDS")?;
                let rhs = self.parse_literal()?;
                Ok(cond(value, Operator::EndsWith, CondRhs::One(rhs)))
            }
            "CONTAINS" => {
                self.advance();
                let rhs = self.parse_literal()?;
                Ok(cond(value, Operator::Contains, CondRhs::One(rhs)))
            }
            "LIKE" => {
                self.advance();
                let rhs = self.parse_literal()?;
                Ok(cond(value, Operator::Like, CondRhs::One(rhs)))
            }
            "NOT" if matches!(self.kind(1).as_str(), "CONTAINS" | "LIKE" | "IN") => {
                self.advance();
                let op_tok = self.advance().kind;
                match op_tok.as_str() {
                    "CONTAINS" => {
                        let rhs = self.parse_literal()?;
                        Ok(cond(value, Operator::NotContains, CondRhs::One(rhs)))
                    }
                    "LIKE" => {
                        let rhs = self.parse_literal()?;
                        Ok(cond(value, Operator::NotLike, CondRhs::One(rhs)))
                    }
                    _ => {
                        let rhs = self.parse_list_literal()?;
                        Ok(cond(value, Operator::NotIn, CondRhs::List(rhs)))
                    }
                }
            }
            "IN" => {
                self.advance();
                let rhs = self.parse_list_literal()?;
                Ok(cond(value, Operator::In, CondRhs::List(rhs)))
            }
            "IS" => {
                if self.kind(1) == "IN" {
                    self.advance();
                    self.advance();
                    let rhs = self.parse_list_literal()?;
                    return Ok(cond(value, Operator::In, CondRhs::List(rhs)));
                }
                self.advance();
                let negated = self.accept("NOT").is_some();
                self.expect("NULL", "'NULL'")?;
                let op = if negated { Operator::IsNotNull } else { Operator::IsNull };
                Ok(cond(value, op, CondRhs::Empty))
            }
            _ => Ok(value), // bare value predicate (regression target)
        }
    }

    fn parse_value_expr(&mut self) -> Result<TargetExpr, SyntaxError> {
        let k = self.kind(0);
        if is_agg_func(&k) && self.kind(1) == "(" {
            return Ok(TargetExpr::Aggregation(self.parse_aggregation()?));
        }
        Ok(TargetExpr::ColumnRef(self.parse_column_ref()?))
    }

    fn parse_aggregation(&mut self) -> Result<Aggregation, SyntaxError> {
        let func = AggFunc::from_keyword(&self.advance().kind).unwrap();
        self.expect("(", "'('")?;
        let column = self.parse_column_ref()?;
        let filter = if self.accept("WHERE").is_some() {
            Some(Box::new(self.parse_expr()?))
        } else {
            None
        };
        let mut window: Option<Window> = None;
        if self.accept(",").is_some() {
            let start = self.parse_bound()?;
            self.expect(",", "',' between window bounds")?;
            let end = self.parse_bound()?;
            let mut unit = TimeUnit::Days;
            if self.accept(",").is_some() {
                let ut = self.advance();
                if !is_time_unit(&ut.kind) {
                    return Err(SyntaxError::new(
                        format!("expected a time unit, found {}", ut.kind),
                        ut.pos as isize,
                        &self.text,
                    ));
                }
                unit = TimeUnit::from_keyword(&ut.kind).unwrap();
            }
            window = Some(Window { start, end, unit });
        }
        self.expect(")", "')' to close aggregation")?;
        Ok(Aggregation { func, column, filter, window })
    }

    fn parse_bound(&mut self) -> Result<f64, SyntaxError> {
        let mut sign = 1.0f64;
        if self.accept("+").is_some() {
            sign = 1.0;
        } else if self.accept("-").is_some() {
            sign = -1.0;
        }
        let t = self.advance();
        match (t.kind.as_str(), &t.val) {
            ("INT", TokVal::Int(v)) => Ok(sign * (*v as f64)),
            ("INF", _) => Ok(sign * f64::INFINITY),
            _ => Err(SyntaxError::new(
                format!("expected a window bound, found {}", t.kind),
                t.pos as isize,
                &self.text,
            )),
        }
    }

    fn parse_column_ref(&mut self) -> Result<ColumnRef, SyntaxError> {
        let table = self.parse_name("a table name")?;
        self.expect(".", "'.' in table.column reference")?;
        if self.accept("*").is_some() {
            return Ok(ColumnRef::new(table, "*"));
        }
        let column = self.parse_name("a column name")?;
        Ok(ColumnRef::new(table, column))
    }

    fn parse_name(&mut self, what: &str) -> Result<String, SyntaxError> {
        let t = self.peek(0);
        if t.kind == "IDENT" || is_soft_keyword(&t.kind) {
            let tok = self.advance();
            match tok.val {
                TokVal::Str(s) => Ok(s),
                _ => Ok(tok.kind),
            }
        } else {
            Err(SyntaxError::new(
                format!("expected {}, found {}", what, t.kind),
                t.pos as isize,
                &self.text,
            ))
        }
    }

    fn parse_list_literal(&mut self) -> Result<Vec<Literal>, SyntaxError> {
        self.expect("(", "'(' to open a literal list")?;
        let mut items = vec![self.parse_literal()?];
        while self.accept(",").is_some() {
            items.push(self.parse_literal()?);
        }
        self.expect(")", "')' to close a literal list")?;
        Ok(items)
    }

    fn parse_literal(&mut self) -> Result<Literal, SyntaxError> {
        let t = self.advance();
        match t.kind.as_str() {
            "STRING" => match t.val {
                TokVal::Str(s) => Ok(Literal::Str(s)),
                _ => unreachable!(),
            },
            "DATE" => {
                let raw = match &t.val {
                    TokVal::Str(s) => s.clone(),
                    _ => unreachable!(),
                };
                let dt = parse_date_literal(&raw).map_err(|_| {
                    SyntaxError::new(format!("invalid date literal {:?}", raw), t.pos as isize, &self.text)
                })?;
                Ok(Literal::Date(dt))
            }
            "NULL" => Ok(Literal::Null),
            "+" | "-" => {
                let sign = if t.kind == "-" { -1.0 } else { 1.0 };
                let nt = self.advance();
                match (nt.kind.as_str(), &nt.val) {
                    ("INT", TokVal::Int(v)) => Ok(Literal::Num(sign * (*v as f64))),
                    ("FLOAT", TokVal::Float(v)) => Ok(Literal::Num(sign * *v)),
                    _ => Err(SyntaxError::new(
                        format!("expected a number after {:?}", t.kind),
                        nt.pos as isize,
                        &self.text,
                    )),
                }
            }
            "INT" => match t.val {
                TokVal::Int(v) => Ok(Literal::Num(v as f64)),
                _ => unreachable!(),
            },
            "FLOAT" => match t.val {
                TokVal::Float(v) => Ok(Literal::Num(v)),
                _ => unreachable!(),
            },
            other => Err(SyntaxError::new(
                format!("expected a literal, found {}", other),
                t.pos as isize,
                &self.text,
            )),
        }
    }
}

fn cond(left: TargetExpr, op: Operator, right: CondRhs) -> TargetExpr {
    TargetExpr::Condition(Condition { left: Box::new(left), op, right })
}

fn parse_date_literal(s: &str) -> Result<DateTime<Utc>, ()> {
    if s.contains(' ') {
        NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S")
            .map(|nd| nd.and_utc())
            .map_err(|_| ())
    } else {
        NaiveDate::parse_from_str(s, "%Y-%m-%d")
            .map(|d| d.and_hms_opt(0, 0, 0).unwrap().and_utc())
            .map_err(|_| ())
    }
}

/// Parse only — no schema needed.
pub fn parse(query: &str) -> Result<ParsedQuery, SyntaxError> {
    if query.trim().is_empty() {
        return Err(SyntaxError { message: "empty query".into(), pos: -1, rendered: "PQL syntax error: empty query".into() });
    }
    Parser::new(query)?.parse_query()
}

// ---------------------------------------------------------------------------
// Schema-bound validation
// ---------------------------------------------------------------------------

/// A parsed query bound to a schema, with its inferred task type.
#[derive(Clone, PartialEq, Debug)]
pub struct ValidatedQuery {
    pub query: ParsedQuery,
    pub task_type: super::ast::TaskType,
}

fn check_column(
    col: &ColumnRef,
    schema: &Schema,
    allow_star: bool,
    allow_pk: bool,
    allow_fk: bool,
) -> Result<(), ValidationError> {
    let table = schema
        .table(&col.table)
        .ok_or_else(|| ValidationError(format!("unknown table {:?}", col.table)))?;
    if col.column == "*" {
        if !allow_star {
            return Err(ValidationError(format!("{}: '*' not allowed here", col)));
        }
        return Ok(());
    }
    if table.column(&col.column).is_none() {
        if allow_pk && table.primary_key.as_deref() == Some(col.column.as_str()) {
            return Ok(());
        }
        if allow_fk
            && schema
                .links_from(&col.table)
                .iter()
                .any(|l| l.fk_column == col.column)
        {
            return Ok(());
        }
        return Err(ValidationError(format!(
            "unknown column {:?} on table {:?}",
            col.column, col.table
        )));
    }
    Ok(())
}

fn walk_columns(expr: &TargetExpr, schema: &Schema) -> Result<(), ValidationError> {
    match expr {
        TargetExpr::Aggregation(a) => {
            // FK columns are legal aggregation targets for set/count aggregations
            // (the recommendation pattern: LIST_DISTINCT / COUNT over a foreign
            // key); only FIRST/LAST exclude them per the docs.
            let fk_ok = matches!(
                a.func,
                AggFunc::ListDistinct | AggFunc::Count | AggFunc::CountDistinct
            );
            check_column(&a.column, schema, true, false, fk_ok)?;
            if a.window.is_some() {
                if let Some(t) = schema.table(&a.column.table) {
                    if t.time_column.is_none() {
                        return Err(ValidationError(format!(
                            "windowed aggregation over {:?}, which has no time_column",
                            a.column.table
                        )));
                    }
                }
            }
            if let Some(f) = &a.filter {
                walk_columns(f, schema)?;
            }
            Ok(())
        }
        TargetExpr::ColumnRef(c) => check_column(c, schema, false, false, false),
        TargetExpr::Condition(c) => walk_columns(&c.left, schema),
        TargetExpr::LogicalOp(l) => {
            walk_columns(&l.left, schema)?;
            walk_columns(&l.right, schema)
        }
        TargetExpr::Not(e) => walk_columns(e, schema),
    }
}

/// Parse + bind against a schema: tables/columns exist, the entity key is a
/// primary key, target windows are future-facing (start >= 0).
pub fn validate(query: &ParsedQuery, schema: &Schema) -> Result<ValidatedQuery, ValidationError> {
    let ek = &query.entity_key;
    let table = schema
        .table(&ek.table)
        .ok_or_else(|| ValidationError(format!("unknown entity table {:?}", ek.table)))?;
    if table.primary_key.as_deref() != Some(ek.column.as_str()) {
        return Err(ValidationError(format!(
            "FOR EACH {}: {:?} is not the primary key of {:?} (expected {:?})",
            ek, ek.column, ek.table, table.primary_key
        )));
    }
    walk_columns(&query.target, schema)?;
    for agg in query.target_aggregations() {
        if let Some(w) = &agg.window {
            if w.start < 0.0 {
                return Err(ValidationError(format!(
                    "target window ({}, {}] must be future-facing (start >= 0)",
                    w.start, w.end
                )));
            }
        }
    }
    if let Some(w) = &query.where_ {
        walk_columns(w, schema)?;
    }
    if let Some(a) = &query.assuming {
        walk_columns(a, schema)?;
    }
    let task_type = query.task_type(Some(schema));
    Ok(ValidatedQuery { query: query.clone(), task_type })
}

/// Convenience: parse a string then validate.
pub fn parse_and_validate(query: &str, schema: &Schema) -> Result<ValidatedQuery, crate::Error> {
    let pq = parse(query)?;
    Ok(validate(&pq, schema)?)
}
