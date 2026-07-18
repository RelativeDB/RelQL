//! The PQL parser — the single source of truth is the shared C++ parser
//! (`pql_parse` in `librt_c`); this module calls it and deserializes its JSON
//! AST into the crate's [`ParsedQuery`](super::ast::ParsedQuery). The same
//! parser backs the Python and Java bindings, so the grammar lives in exactly
//! one place.
//!
//! `librt_c` is a hard runtime dependency (it is already required to run the
//! RT-J model). When it cannot be loaded, [`parse_native`] returns a clear
//! [`SyntaxError`] naming the searched paths — there is no hand-written
//! fallback.
//!
//! No `serde`/`serde_json` dependency: the JSON schema `pql_parse` emits is
//! small and fixed (see the schema doc in `cpp/src/pql.*`), so a tiny
//! hand-written value parser suffices.

use std::ffi::{c_char, CString};
use std::fmt;
use std::path::Path;
use std::sync::OnceLock;

use chrono::{DateTime, NaiveDate, NaiveDateTime, Utc};
use libloading::Library;

use super::ast::{
    AggFunc, Aggregation, BoolOp, ColumnRef, CondRhs, Condition, Literal, LogicalOp, Operator,
    ParsedQuery, RankKind, TargetExpr, TimeUnit, Window,
};

const OUT: usize = 1 << 16; // 64 KiB JSON buffer — beyond any real query's AST
const ERR: usize = 1024;

/// A PQL parse error — a syntax error reported by the shared C++ parser, or a
/// hard "library unavailable" error when `librt_c` cannot be loaded.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct SyntaxError {
    pub message: String,
    pub pos: isize,
    pub rendered: String,
}

impl fmt::Display for SyntaxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.rendered)
    }
}
impl std::error::Error for SyntaxError {}

// ---------------------------------------------------------------------------
// library binding
// ---------------------------------------------------------------------------

type PqlParseFn =
    unsafe extern "C" fn(*const c_char, *mut c_char, usize, *mut c_char, usize) -> i32;

struct PqlLib {
    _lib: Library,
    parse: PqlParseFn,
}

// The bound symbol is a plain reentrant C function (see pql_c.h).
unsafe impl Send for PqlLib {}
unsafe impl Sync for PqlLib {}

fn lib() -> Option<&'static PqlLib> {
    static LIB: OnceLock<Option<PqlLib>> = OnceLock::new();
    LIB.get_or_init(load).as_ref()
}

fn load() -> Option<PqlLib> {
    for cand in crate::native::candidate_lib_paths() {
        if cand.is_empty() || !Path::new(&cand).exists() {
            continue;
        }
        unsafe {
            let l = match Library::new(&cand) {
                Ok(l) => l,
                Err(_) => continue,
            };
            let parse: PqlParseFn = match l.get::<PqlParseFn>(b"pql_parse\0") {
                Ok(s) => *s,
                Err(_) => continue,
            };
            return Some(PqlLib { _lib: l, parse });
        }
    }
    None
}

/// Whether the shared C++ parser (`librt_c`) is loadable.
pub fn native_available() -> bool {
    lib().is_some()
}

fn syn(msg: impl Into<String>) -> SyntaxError {
    let message = msg.into();
    let rendered = format!("PQL syntax error: {}", message);
    SyntaxError { message, pos: -1, rendered }
}

fn cstr(buf: &[u8]) -> String {
    let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    String::from_utf8_lossy(&buf[..end]).into_owned()
}

/// Parse `query` with the shared C++ parser and deserialize its JSON AST into a
/// [`ParsedQuery`]. Errors with a clear message when `librt_c` is unavailable
/// (it is a hard runtime dependency — there is no hand-written fallback).
pub fn parse_native(query: &str) -> Result<ParsedQuery, SyntaxError> {
    let lib = lib().ok_or_else(|| {
        syn(format!(
            "librt_c could not be loaded, so PQL cannot be parsed (it is a hard runtime \
             dependency). Build cpp/ with cmake, or set RELATIVEDB_RT_LIB to the built library. \
             Searched: {}",
            crate::native::candidate_lib_paths().join(", ")
        ))
    })?;
    if query.trim().is_empty() {
        return Err(syn("empty query"));
    }
    let cq = CString::new(query).map_err(|_| syn("query contains an interior NUL byte"))?;
    let mut out = vec![0u8; OUT];
    let mut err = vec![0u8; ERR];
    let rc = unsafe {
        (lib.parse)(
            cq.as_ptr(),
            out.as_mut_ptr() as *mut c_char,
            OUT,
            err.as_mut_ptr() as *mut c_char,
            ERR,
        )
    };
    if rc != 0 {
        let m = cstr(&err);
        return Err(syn(if m.is_empty() { "parse failed".to_string() } else { m }));
    }
    let json = cstr(&out);
    let value = Json::parse(&json).map_err(|e| syn(format!("malformed AST JSON from pql_parse: {}", e)))?;
    query_from_json(&value, query)
}

// ---------------------------------------------------------------------------
// tiny JSON value parser (no external deps)
// ---------------------------------------------------------------------------

#[derive(Debug)]
enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    fn parse(s: &str) -> Result<Json, String> {
        let chars: Vec<char> = s.chars().collect();
        let mut p = JsonParser { chars, i: 0 };
        p.skip_ws();
        let v = p.value()?;
        p.skip_ws();
        if p.i != p.chars.len() {
            return Err(format!("trailing characters at {}", p.i));
        }
        Ok(v)
    }

    fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(fields) => fields.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }
}

struct JsonParser {
    chars: Vec<char>,
    i: usize,
}

impl JsonParser {
    fn skip_ws(&mut self) {
        while self.i < self.chars.len() && self.chars[self.i].is_whitespace() {
            self.i += 1;
        }
    }

    fn value(&mut self) -> Result<Json, String> {
        self.skip_ws();
        match self.chars.get(self.i) {
            None => Err("unexpected end of JSON".into()),
            Some('{') => self.object(),
            Some('[') => self.array(),
            Some('"') => Ok(Json::Str(self.string()?)),
            Some('t') => self.literal("true", Json::Bool(true)),
            Some('f') => self.literal("false", Json::Bool(false)),
            Some('n') => self.literal("null", Json::Null),
            Some(c) if *c == '-' || c.is_ascii_digit() => self.number(),
            Some(c) => Err(format!("unexpected character {:?}", c)),
        }
    }

    fn literal(&mut self, word: &str, val: Json) -> Result<Json, String> {
        for w in word.chars() {
            if self.chars.get(self.i) != Some(&w) {
                return Err(format!("expected {:?}", word));
            }
            self.i += 1;
        }
        Ok(val)
    }

    fn number(&mut self) -> Result<Json, String> {
        let start = self.i;
        while let Some(c) = self.chars.get(self.i) {
            if c.is_ascii_digit() || matches!(c, '-' | '+' | '.' | 'e' | 'E') {
                self.i += 1;
            } else {
                break;
            }
        }
        let s: String = self.chars[start..self.i].iter().collect();
        s.parse::<f64>().map(Json::Num).map_err(|_| format!("bad number {:?}", s))
    }

    fn string(&mut self) -> Result<String, String> {
        // assumes current char is '"'
        self.i += 1;
        let mut out = String::new();
        loop {
            let c = *self.chars.get(self.i).ok_or("unterminated string")?;
            self.i += 1;
            match c {
                '"' => return Ok(out),
                '\\' => {
                    let e = *self.chars.get(self.i).ok_or("unterminated escape")?;
                    self.i += 1;
                    match e {
                        '"' => out.push('"'),
                        '\\' => out.push('\\'),
                        '/' => out.push('/'),
                        'b' => out.push('\u{0008}'),
                        'f' => out.push('\u{000C}'),
                        'n' => out.push('\n'),
                        'r' => out.push('\r'),
                        't' => out.push('\t'),
                        'u' => {
                            let mut code: u32 = 0;
                            for _ in 0..4 {
                                let h = *self.chars.get(self.i).ok_or("bad \\u escape")?;
                                self.i += 1;
                                code = code * 16
                                    + h.to_digit(16).ok_or("bad hex in \\u escape")?;
                            }
                            out.push(char::from_u32(code).unwrap_or('\u{FFFD}'));
                        }
                        other => return Err(format!("bad escape \\{}", other)),
                    }
                }
                other => out.push(other),
            }
        }
    }

    fn array(&mut self) -> Result<Json, String> {
        self.i += 1; // '['
        let mut items = Vec::new();
        self.skip_ws();
        if self.chars.get(self.i) == Some(&']') {
            self.i += 1;
            return Ok(Json::Arr(items));
        }
        loop {
            items.push(self.value()?);
            self.skip_ws();
            match self.chars.get(self.i) {
                Some(',') => {
                    self.i += 1;
                    self.skip_ws();
                }
                Some(']') => {
                    self.i += 1;
                    return Ok(Json::Arr(items));
                }
                _ => return Err("expected ',' or ']'".into()),
            }
        }
    }

    fn object(&mut self) -> Result<Json, String> {
        self.i += 1; // '{'
        let mut fields = Vec::new();
        self.skip_ws();
        if self.chars.get(self.i) == Some(&'}') {
            self.i += 1;
            return Ok(Json::Obj(fields));
        }
        loop {
            self.skip_ws();
            if self.chars.get(self.i) != Some(&'"') {
                return Err("expected object key".into());
            }
            let key = self.string()?;
            self.skip_ws();
            if self.chars.get(self.i) != Some(&':') {
                return Err("expected ':'".into());
            }
            self.i += 1;
            let val = self.value()?;
            fields.push((key, val));
            self.skip_ws();
            match self.chars.get(self.i) {
                Some(',') => {
                    self.i += 1;
                }
                Some('}') => {
                    self.i += 1;
                    return Ok(Json::Obj(fields));
                }
                _ => return Err("expected ',' or '}'".into()),
            }
        }
    }
}

// ---------------------------------------------------------------------------
// JSON -> AST (must reproduce the hand-written parser's values exactly)
// ---------------------------------------------------------------------------

fn bound(j: &Json) -> Result<f64, SyntaxError> {
    match j {
        Json::Str(s) if s == "inf" => Ok(f64::INFINITY),
        Json::Str(s) if s == "-inf" => Ok(f64::NEG_INFINITY),
        Json::Num(n) => Ok(*n),
        _ => Err(syn("invalid window bound in AST JSON")),
    }
}

fn num_i64(j: &Json) -> Option<i64> {
    match j {
        Json::Num(n) => Some(*n as i64),
        _ => None,
    }
}

fn parse_date(s: &str) -> Result<DateTime<Utc>, SyntaxError> {
    let dt = if s.contains(' ') {
        NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S").map(|nd| nd.and_utc())
    } else {
        NaiveDate::parse_from_str(s, "%Y-%m-%d")
            .map(|d| d.and_hms_opt(0, 0, 0).unwrap().and_utc())
    };
    dt.map_err(|_| syn(format!("invalid date literal {:?}", s)))
}

fn lit(j: &Json) -> Result<Literal, SyntaxError> {
    match j {
        Json::Null => Ok(Literal::Null),
        Json::Str(s) => Ok(Literal::Str(s.clone())),
        Json::Num(n) => Ok(Literal::Num(*n)),
        // The PQL grammar has no boolean literal; the Rust AST has no Bool
        // variant. Booleans should never appear, but map defensively so the
        // parser never panics on unexpected JSON.
        Json::Bool(b) => Ok(Literal::Num(if *b { 1.0 } else { 0.0 })),
        Json::Obj(_) => {
            let d = j.get("date").and_then(Json::as_str).ok_or_else(|| syn("bad literal object"))?;
            Ok(Literal::Date(parse_date(d)?))
        }
        Json::Arr(_) => Err(syn("unexpected list where a scalar literal was expected")),
    }
}

fn operator(name: &str) -> Result<Operator, SyntaxError> {
    Ok(match name {
        "GT" => Operator::Gt,
        "LT" => Operator::Lt,
        "EQ" => Operator::Eq,
        "NEQ" => Operator::Neq,
        "GE" => Operator::Ge,
        "LE" => Operator::Le,
        "STARTS_WITH" => Operator::StartsWith,
        "ENDS_WITH" => Operator::EndsWith,
        "CONTAINS" => Operator::Contains,
        "NOT_CONTAINS" => Operator::NotContains,
        "LIKE" => Operator::Like,
        "NOT_LIKE" => Operator::NotLike,
        "IN" => Operator::In,
        "NOT_IN" => Operator::NotIn,
        "IS_NULL" => Operator::IsNull,
        "IS_NOT_NULL" => Operator::IsNotNull,
        other => return Err(syn(format!("unknown operator {:?}", other))),
    })
}

fn column_ref(j: &Json) -> Result<ColumnRef, SyntaxError> {
    let table = j.get("table").and_then(Json::as_str).ok_or_else(|| syn("column ref missing table"))?;
    let column = j.get("column").and_then(Json::as_str).ok_or_else(|| syn("column ref missing column"))?;
    Ok(ColumnRef::new(table, column))
}

fn opt_expr(j: Option<&Json>) -> Result<Option<TargetExpr>, SyntaxError> {
    match j {
        None | Some(Json::Null) => Ok(None),
        Some(o) => Ok(Some(expr(o)?)),
    }
}

fn expr(o: &Json) -> Result<TargetExpr, SyntaxError> {
    let kind = o.get("kind").and_then(Json::as_str).ok_or_else(|| syn("expr missing kind"))?;
    match kind {
        "col" => Ok(TargetExpr::ColumnRef(column_ref(o)?)),
        "agg" => {
            let func_name = o.get("func").and_then(Json::as_str).ok_or_else(|| syn("agg missing func"))?;
            let func = AggFunc::from_keyword(func_name)
                .ok_or_else(|| syn(format!("unknown aggregation {:?}", func_name)))?;
            let column = column_ref(o.get("column").ok_or_else(|| syn("agg missing column"))?)?;
            let filter = opt_expr(o.get("filter"))?.map(Box::new);
            let window = match o.get("window") {
                None | Some(Json::Null) => None,
                Some(w) => {
                    let start = bound(w.get("start").ok_or_else(|| syn("window missing start"))?)?;
                    let end = bound(w.get("end").ok_or_else(|| syn("window missing end"))?)?;
                    let unit_name = w.get("unit").and_then(Json::as_str).ok_or_else(|| syn("window missing unit"))?;
                    let unit = TimeUnit::from_keyword(&unit_name.to_ascii_uppercase())
                        .ok_or_else(|| syn(format!("unknown time unit {:?}", unit_name)))?;
                    Some(Window { start, end, unit })
                }
            };
            Ok(TargetExpr::Aggregation(Aggregation { func, column, filter, window }))
        }
        "cond" => {
            let left = Box::new(expr(o.get("left").ok_or_else(|| syn("cond missing left"))?)?);
            let op_name = o.get("op").and_then(Json::as_str).ok_or_else(|| syn("cond missing op"))?;
            let op = operator(op_name)?;
            let right_json = o.get("right");
            let right = match op {
                Operator::IsNull | Operator::IsNotNull => CondRhs::Empty,
                Operator::In | Operator::NotIn => {
                    let arr = match right_json {
                        Some(Json::Arr(items)) => items,
                        _ => return Err(syn("IN/NOT IN expects a list literal")),
                    };
                    let mut lits = Vec::with_capacity(arr.len());
                    for it in arr {
                        lits.push(lit(it)?);
                    }
                    CondRhs::List(lits)
                }
                _ => CondRhs::One(lit(right_json.ok_or_else(|| syn("cond missing right"))?)?),
            };
            Ok(TargetExpr::Condition(Condition { left, op, right }))
        }
        "logic" => {
            let op = match o.get("op").and_then(Json::as_str) {
                Some("AND") => BoolOp::And,
                Some("OR") => BoolOp::Or,
                other => return Err(syn(format!("unknown logic op {:?}", other))),
            };
            let left = Box::new(expr(o.get("left").ok_or_else(|| syn("logic missing left"))?)?);
            let right = Box::new(expr(o.get("right").ok_or_else(|| syn("logic missing right"))?)?);
            Ok(TargetExpr::LogicalOp(LogicalOp { left, op, right }))
        }
        "not" => Ok(TargetExpr::Not(Box::new(expr(
            o.get("expr").ok_or_else(|| syn("not missing expr"))?,
        )?))),
        other => Err(syn(format!("unknown expr kind {:?}", other))),
    }
}

fn query_from_json(o: &Json, text: &str) -> Result<ParsedQuery, SyntaxError> {
    let target = expr(o.get("target").ok_or_else(|| syn("query missing target"))?)?;
    let entity_key = column_ref(o.get("entity_key").ok_or_else(|| syn("query missing entity_key"))?)?;
    let entity_ids = match o.get("entity_ids") {
        Some(Json::Arr(items)) => {
            let mut v = Vec::with_capacity(items.len());
            for it in items {
                v.push(lit(it)?);
            }
            v
        }
        _ => Vec::new(),
    };
    let where_ = opt_expr(o.get("where"))?;
    let assuming = opt_expr(o.get("assuming"))?;
    let rank = match o.get("rank") {
        Some(Json::Str(s)) if s == "CLASSIFY" => Some(RankKind::Classify),
        Some(Json::Str(s)) if s == "RANK" => Some(RankKind::Rank),
        _ => None,
    };
    let top_k = o.get("top_k").and_then(num_i64);
    let num_forecasts = o.get("num_forecasts").and_then(num_i64);
    Ok(ParsedQuery {
        target,
        entity_key,
        entity_ids,
        where_,
        assuming,
        rank,
        top_k,
        num_forecasts,
        text: text.to_string(),
    })
}
