//! Expression evaluation over an assembled context.
//!
//! Used for WHERE-clause entity filtering and by the built-in history baseline
//! backend. Aggregation windows are `(anchor + start, anchor + end]` (start
//! excluded, end included), matching the grammar's window semantics. Mirrors
//! `relativedb.evaluate` (Python).

use std::cmp::Ordering;
use std::collections::HashMap;

use chrono::{DateTime, Utc};

use crate::pql::ast::{
    AggFunc, Aggregation, CondRhs, Literal, Operator, TargetExpr, Window,
};
use crate::retrieve::{Row, Value};

/// A value produced by evaluating a RelQL value-expression.
#[derive(Clone, PartialEq, Debug)]
pub enum EvalValue {
    Null,
    Num(f64),
    Bool(bool),
    Text(String),
    Date(DateTime<Utc>),
    List(Vec<Value>),
}

impl EvalValue {
    pub fn as_number(&self) -> Option<f64> {
        match self {
            EvalValue::Num(n) => Some(*n),
            EvalValue::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
            _ => None,
        }
    }
    pub fn truthy(&self) -> bool {
        match self {
            EvalValue::Null => false,
            EvalValue::Num(n) => *n != 0.0,
            EvalValue::Bool(b) => *b,
            EvalValue::Text(s) => !s.is_empty(),
            EvalValue::Date(_) => true,
            EvalValue::List(l) => !l.is_empty(),
        }
    }
}

fn value_to_eval(v: &Value) -> EvalValue {
    match v {
        Value::Number(n) => EvalValue::Num(*n),
        Value::Boolean(b) => EvalValue::Bool(*b),
        Value::Text(s) => EvalValue::Text(s.clone()),
        Value::Datetime(d) => EvalValue::Date(*d),
    }
}

fn window_bounds(w: &Window, anchor: DateTime<Utc>) -> (Option<DateTime<Utc>>, Option<DateTime<Utc>>) {
    let lo = w.start_offset().map(|d| anchor + d);
    let hi = w.end_offset().map(|d| anchor + d);
    (lo, hi)
}

fn rows_in_window<'a>(
    rows: &'a [Row],
    window: Option<&Window>,
    anchor: Option<DateTime<Utc>>,
) -> Vec<&'a Row> {
    let mut picked: Vec<&Row> = match (window, anchor) {
        (Some(w), Some(a)) => {
            let (lo, hi) = window_bounds(w, a);
            rows.iter()
                .filter(|r| {
                    let ts = match r.timestamp {
                        Some(t) => t,
                        None => return false,
                    };
                    if let Some(lo) = lo {
                        if !(ts > lo) {
                            return false;
                        }
                    }
                    if let Some(hi) = hi {
                        if !(ts <= hi) {
                            return false;
                        }
                    }
                    true
                })
                .collect()
        }
        _ => rows.iter().collect(),
    };
    // static rows (no ts) first, then ascending by timestamp
    picked.sort_by(|a, b| {
        let ka = a.timestamp.is_some();
        let kb = b.timestamp.is_some();
        ka.cmp(&kb).then_with(|| match (a.timestamp, b.timestamp) {
            (Some(x), Some(y)) => x.cmp(&y),
            _ => Ordering::Equal,
        })
    });
    picked
}

/// Row-level inline filter, e.g. `COUNT(t.* WHERE t.amount > 100, ...)`.
pub fn eval_row_predicate(expr: &TargetExpr, row: &Row) -> Result<bool, String> {
    match expr {
        TargetExpr::LogicalOp(l) => {
            let left = eval_row_predicate(&l.left, row)?;
            match l.op {
                crate::pql::ast::BoolOp::And => {
                    Ok(left && eval_row_predicate(&l.right, row)?)
                }
                crate::pql::ast::BoolOp::Or => {
                    Ok(left || eval_row_predicate(&l.right, row)?)
                }
            }
        }
        TargetExpr::Not(e) => Ok(!eval_row_predicate(e, row)?),
        TargetExpr::Condition(c) => {
            let col = match c.left.as_ref() {
                TargetExpr::ColumnRef(cr) => cr,
                _ => return Err("inline aggregation filters must compare columns".into()),
            };
            let left = if col.table == row.table {
                row.get_cell(&col.column).map(value_to_eval).unwrap_or(EvalValue::Null)
            } else {
                EvalValue::Null
            };
            Ok(compare(c.op, &left, &c.right))
        }
        _ => Err(format!("unsupported row predicate: {:?}", expr)),
    }
}

fn row_filter_ok(expr: &TargetExpr, row: &Row) -> bool {
    // unevaluable sub-filter: keep the row (best effort), like Python
    eval_row_predicate(expr, row).unwrap_or(true)
}

fn agg_rows<'a>(
    agg: &Aggregation,
    rows_by_table: &'a HashMap<String, Vec<Row>>,
    anchor: Option<DateTime<Utc>>,
) -> Vec<&'a Row> {
    let rows: &[Row] = match rows_by_table.get(&agg.column.table) {
        Some(r) => r.as_slice(),
        None => &[],
    };
    let mut picked = rows_in_window(rows, agg.window.as_ref(), anchor);
    if let Some(filter) = &agg.filter {
        picked.retain(|r| row_filter_ok(filter, r));
    }
    picked
}

/// Evaluate a valueExpr (aggregation or static column) over the context.
pub fn eval_value(
    expr: &TargetExpr,
    rows_by_table: &HashMap<String, Vec<Row>>,
    entity_cells: &[(String, Value)],
    anchor: Option<DateTime<Utc>>,
) -> EvalValue {
    let agg = match expr {
        TargetExpr::ColumnRef(c) => {
            return entity_cells
                .iter()
                .find(|(k, _)| k == &c.column)
                .map(|(_, v)| value_to_eval(v))
                .unwrap_or(EvalValue::Null);
        }
        TargetExpr::Aggregation(a) => a,
        TargetExpr::Lit(l) => return lit_to_eval(l),
        // Boolean sub-expressions in value position (e.g. a CASE WHEN cond).
        TargetExpr::Condition(_) | TargetExpr::LogicalOp(_) | TargetExpr::Not(_) => {
            return EvalValue::Bool(eval_bool(expr, rows_by_table, entity_cells, anchor));
        }
        TargetExpr::Arith(a) => {
            let l = eval_value(&a.left, rows_by_table, entity_cells, anchor);
            let r = eval_value(&a.right, rows_by_table, entity_cells, anchor);
            return eval_arith(a.op, &l, &r);
        }
        TargetExpr::Func(f) => {
            return eval_func(f, rows_by_table, entity_cells, anchor);
        }
        TargetExpr::Case(c) => {
            for (cond, then) in &c.whens {
                if eval_bool(cond, rows_by_table, entity_cells, anchor) {
                    return eval_value(then, rows_by_table, entity_cells, anchor);
                }
            }
            return match &c.else_ {
                Some(e) => eval_value(e, rows_by_table, entity_cells, anchor),
                None => EvalValue::Null,
            };
        }
    };

    let rows = agg_rows(agg, rows_by_table, anchor);
    let col = &agg.column.column;
    if agg.func == AggFunc::Exists {
        return EvalValue::Bool(!rows.is_empty());
    }
    if agg.func == AggFunc::Count {
        if col == "*" {
            return EvalValue::Num(rows.len() as f64);
        }
        let c = rows.iter().filter(|r| r.get_cell(col).is_some()).count();
        return EvalValue::Num(c as f64);
    }
    // collected non-null cell values (for real columns)
    let values: Vec<Value> = if col == "*" {
        Vec::new()
    } else {
        rows.iter().filter_map(|r| r.get_cell(col).cloned()).collect()
    };
    match agg.func {
        AggFunc::CountDistinct => {
            let mut seen: Vec<Value> = Vec::new();
            for v in &values {
                if !seen.iter().any(|s| s == v) {
                    seen.push(v.clone());
                }
            }
            EvalValue::Num(seen.len() as f64)
        }
        AggFunc::ListDistinct => {
            let mut seen: Vec<Value> = Vec::new();
            for v in &values {
                if !seen.iter().any(|s| s == v) {
                    seen.push(v.clone());
                }
            }
            EvalValue::List(seen)
        }
        AggFunc::First => values.first().map(value_to_eval).unwrap_or(EvalValue::Null),
        AggFunc::Last => values.last().map(value_to_eval).unwrap_or(EvalValue::Null),
        AggFunc::Sum | AggFunc::Avg | AggFunc::Min | AggFunc::Max => {
            let nums: Vec<f64> = values.iter().filter_map(|v| v.as_number()).collect();
            if agg.func == AggFunc::Sum {
                return EvalValue::Num(nums.iter().sum());
            }
            if nums.is_empty() {
                return EvalValue::Null;
            }
            match agg.func {
                AggFunc::Avg => EvalValue::Num(nums.iter().sum::<f64>() / nums.len() as f64),
                AggFunc::Min => EvalValue::Num(nums.iter().cloned().fold(f64::INFINITY, f64::min)),
                AggFunc::Max => {
                    EvalValue::Num(nums.iter().cloned().fold(f64::NEG_INFINITY, f64::max))
                }
                _ => unreachable!(),
            }
        }
        _ => EvalValue::Null,
    }
}

// -- value expressions ------------------------------------------------------

fn eval_arith(op: char, l: &EvalValue, r: &EvalValue) -> EvalValue {
    // SQL-style NULL propagation.
    let (a, b) = match (l.as_number(), r.as_number()) {
        (Some(a), Some(b)) => (a, b),
        _ => return EvalValue::Null,
    };
    match op {
        '+' => EvalValue::Num(a + b),
        '-' => EvalValue::Num(a - b),
        '*' => EvalValue::Num(a * b),
        // division by zero → NULL (SQL semantics).
        '/' => {
            if b == 0.0 {
                EvalValue::Null
            } else {
                EvalValue::Num(a / b)
            }
        }
        _ => EvalValue::Null,
    }
}

fn eval_func(
    f: &crate::pql::ast::Func,
    rows_by_table: &HashMap<String, Vec<Row>>,
    entity_cells: &[(String, Value)],
    anchor: Option<DateTime<Utc>>,
) -> EvalValue {
    let args: Vec<EvalValue> = f
        .args
        .iter()
        .map(|a| eval_value(a, rows_by_table, entity_cells, anchor))
        .collect();
    match f.name.as_str() {
        "COALESCE" => args
            .into_iter()
            .find(|v| !matches!(v, EvalValue::Null))
            .unwrap_or(EvalValue::Null),
        "NULLIF" => match (args.first(), args.get(1)) {
            (Some(a), Some(b)) if a == b => EvalValue::Null,
            (Some(a), _) => a.clone(),
            _ => EvalValue::Null,
        },
        "ABS" => args.first().and_then(EvalValue::as_number).map(|n| EvalValue::Num(n.abs())).unwrap_or(EvalValue::Null),
        "LOG" => args.first().and_then(EvalValue::as_number).map(|n| EvalValue::Num(n.ln())).unwrap_or(EvalValue::Null),
        "EXP" => args.first().and_then(EvalValue::as_number).map(|n| EvalValue::Num(n.exp())).unwrap_or(EvalValue::Null),
        "LEAST" | "GREATEST" => {
            let nums: Vec<f64> = args.iter().filter_map(EvalValue::as_number).collect();
            if nums.len() != args.len() || nums.is_empty() {
                return EvalValue::Null;
            }
            let v = if f.name == "LEAST" {
                nums.iter().cloned().fold(f64::INFINITY, f64::min)
            } else {
                nums.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
            };
            EvalValue::Num(v)
        }
        _ => EvalValue::Null,
    }
}

// -- comparison -------------------------------------------------------------

fn like_match(text: &str, pattern: &str) -> bool {
    // `%` -> any run, `_` -> any single char; case-insensitive; anchored.
    let t: Vec<char> = text.to_lowercase().chars().collect();
    let p: Vec<char> = pattern.to_lowercase().chars().collect();
    fn go(t: &[char], ti: usize, p: &[char], pi: usize) -> bool {
        if pi == p.len() {
            return ti == t.len();
        }
        match p[pi] {
            '%' => {
                for k in ti..=t.len() {
                    if go(t, k, p, pi + 1) {
                        return true;
                    }
                }
                false
            }
            '_' => ti < t.len() && go(t, ti + 1, p, pi + 1),
            c => ti < t.len() && t[ti] == c && go(t, ti + 1, p, pi + 1),
        }
    }
    go(&t, 0, &p, 0)
}

fn lit_to_string(l: &Literal) -> String {
    match l {
        Literal::Str(s) => s.clone(),
        Literal::Num(n) => {
            if n.fract() == 0.0 {
                format!("{}", *n as i64)
            } else {
                format!("{}", n)
            }
        }
        Literal::Bool(b) => if *b { "True".into() } else { "False".into() },
        Literal::Date(d) => d.to_rfc3339(),
        Literal::Null => "None".into(),
    }
}

fn lit_to_eval(l: &Literal) -> EvalValue {
    match l {
        Literal::Str(s) => EvalValue::Text(s.clone()),
        Literal::Num(n) => EvalValue::Num(*n),
        Literal::Bool(b) => EvalValue::Bool(*b),
        Literal::Date(d) => EvalValue::Date(*d),
        Literal::Null => EvalValue::Null,
    }
}

fn eval_to_string(v: &EvalValue) -> String {
    match v {
        EvalValue::Text(s) => s.clone(),
        EvalValue::Num(n) => {
            if n.fract() == 0.0 {
                format!("{}", *n as i64)
            } else {
                format!("{}", n)
            }
        }
        EvalValue::Bool(b) => if *b { "True".into() } else { "False".into() },
        EvalValue::Date(d) => d.to_rfc3339(),
        EvalValue::Null => "None".into(),
        EvalValue::List(_) => "[...]".into(),
    }
}

fn eq_lit(left: &EvalValue, right: &Literal) -> bool {
    match (left, right) {
        (EvalValue::Num(a), Literal::Num(b)) => a == b,
        (EvalValue::Bool(a), Literal::Num(b)) => (if *a { 1.0 } else { 0.0 }) == *b,
        (EvalValue::Bool(a), Literal::Bool(b)) => a == b,
        (EvalValue::Num(a), Literal::Bool(b)) => *a == (if *b { 1.0 } else { 0.0 }),
        (EvalValue::Text(a), Literal::Str(b)) => a == b,
        (EvalValue::Date(a), Literal::Date(b)) => a == b,
        (EvalValue::Null, Literal::Null) => true,
        _ => false,
    }
}

fn ord_lit(left: &EvalValue, right: &Literal) -> Option<Ordering> {
    match (left, right) {
        (EvalValue::Num(a), Literal::Num(b)) => a.partial_cmp(b),
        (EvalValue::Bool(a), Literal::Num(b)) => (if *a { 1.0 } else { 0.0 }).partial_cmp(b),
        (EvalValue::Bool(a), Literal::Bool(b)) => Some(a.cmp(b)),
        (EvalValue::Text(a), Literal::Str(b)) => Some(a.cmp(b)),
        (EvalValue::Date(a), Literal::Date(b)) => Some(a.cmp(b)),
        _ => None,
    }
}

fn eq_val(a: &EvalValue, b: &EvalValue) -> bool {
    match (a.as_number(), b.as_number()) {
        (Some(x), Some(y)) => x == y,
        _ => match (a, b) {
            (EvalValue::Text(x), EvalValue::Text(y)) => x == y,
            (EvalValue::Date(x), EvalValue::Date(y)) => x == y,
            (EvalValue::Null, EvalValue::Null) => true,
            _ => false,
        },
    }
}

fn ord_val(a: &EvalValue, b: &EvalValue) -> Option<Ordering> {
    match (a.as_number(), b.as_number()) {
        (Some(x), Some(y)) => x.partial_cmp(&y),
        _ => match (a, b) {
            (EvalValue::Text(x), EvalValue::Text(y)) => Some(x.cmp(y)),
            (EvalValue::Date(x), EvalValue::Date(y)) => Some(x.cmp(y)),
            _ => None,
        },
    }
}

/// Compare two evaluated values (expression-to-expression RHS).
fn compare_values(op: Operator, left: &EvalValue, right: &EvalValue) -> bool {
    if matches!(left, EvalValue::Null) || matches!(right, EvalValue::Null) {
        return false;
    }
    match op {
        Operator::Eq => eq_val(left, right),
        Operator::Neq => !eq_val(left, right),
        Operator::Gt => ord_val(left, right) == Some(Ordering::Greater),
        Operator::Lt => ord_val(left, right) == Some(Ordering::Less),
        Operator::Ge => matches!(ord_val(left, right), Some(Ordering::Greater | Ordering::Equal)),
        Operator::Le => matches!(ord_val(left, right), Some(Ordering::Less | Ordering::Equal)),
        Operator::StartsWith => eval_to_string(left).starts_with(&eval_to_string(right)),
        Operator::EndsWith => eval_to_string(left).ends_with(&eval_to_string(right)),
        Operator::Contains => eval_to_string(left).contains(&eval_to_string(right)),
        Operator::NotContains => !eval_to_string(left).contains(&eval_to_string(right)),
        Operator::Like => like_match(&eval_to_string(left), &eval_to_string(right)),
        Operator::NotLike => !like_match(&eval_to_string(left), &eval_to_string(right)),
        _ => false,
    }
}

fn compare(op: Operator, left: &EvalValue, right: &CondRhs) -> bool {
    match op {
        Operator::IsNull => return matches!(left, EvalValue::Null),
        Operator::IsNotNull => return !matches!(left, EvalValue::Null),
        Operator::In => {
            if matches!(left, EvalValue::Null) {
                return false;
            }
            if let CondRhs::List(items) = right {
                return items.iter().any(|l| eq_lit(left, l));
            }
            return false;
        }
        Operator::NotIn => {
            if matches!(left, EvalValue::Null) {
                return false;
            }
            if let CondRhs::List(items) = right {
                return !items.iter().any(|l| eq_lit(left, l));
            }
            return false;
        }
        _ => {}
    }
    if matches!(left, EvalValue::Null) {
        return false;
    }
    let rhs = match right {
        CondRhs::One(l) => l,
        _ => return false,
    };
    match op {
        Operator::StartsWith
        | Operator::EndsWith
        | Operator::Contains
        | Operator::NotContains
        | Operator::Like
        | Operator::NotLike => {
            let s = eval_to_string(left);
            let pat = lit_to_string(rhs);
            match op {
                Operator::StartsWith => s.starts_with(&pat),
                Operator::EndsWith => s.ends_with(&pat),
                Operator::Contains => s.contains(&pat),
                Operator::NotContains => !s.contains(&pat),
                Operator::Like => like_match(&s, &pat),
                Operator::NotLike => !like_match(&s, &pat),
                _ => unreachable!(),
            }
        }
        Operator::Eq => eq_lit(left, rhs),
        Operator::Neq => !eq_lit(left, rhs),
        Operator::Gt => ord_lit(left, rhs) == Some(Ordering::Greater),
        Operator::Lt => ord_lit(left, rhs) == Some(Ordering::Less),
        Operator::Ge => matches!(ord_lit(left, rhs), Some(Ordering::Greater | Ordering::Equal)),
        Operator::Le => matches!(ord_lit(left, rhs), Some(Ordering::Less | Ordering::Equal)),
        _ => false,
    }
}

/// Evaluate a boolean expression over the context.
pub fn eval_bool(
    expr: &TargetExpr,
    rows_by_table: &HashMap<String, Vec<Row>>,
    entity_cells: &[(String, Value)],
    anchor: Option<DateTime<Utc>>,
) -> bool {
    match expr {
        TargetExpr::LogicalOp(l) => {
            let left = eval_bool(&l.left, rows_by_table, entity_cells, anchor);
            match l.op {
                crate::pql::ast::BoolOp::And => {
                    left && eval_bool(&l.right, rows_by_table, entity_cells, anchor)
                }
                crate::pql::ast::BoolOp::Or => {
                    left || eval_bool(&l.right, rows_by_table, entity_cells, anchor)
                }
            }
        }
        TargetExpr::Not(e) => !eval_bool(e, rows_by_table, entity_cells, anchor),
        TargetExpr::Condition(c) => {
            let left = eval_value(&c.left, rows_by_table, entity_cells, anchor);
            if let CondRhs::Expr(re) = &c.right {
                let right = eval_value(re, rows_by_table, entity_cells, anchor);
                return compare_values(c.op, &left, &right);
            }
            compare(c.op, &left, &c.right)
        }
        other => eval_value(other, rows_by_table, entity_cells, anchor).truthy(),
    }
}
