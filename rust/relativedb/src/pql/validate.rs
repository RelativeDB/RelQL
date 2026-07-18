//! Schema-bound validation + task-type binding for a [`ParsedQuery`].
//!
//! Parsing itself is single-sourced on the shared C++ parser (see
//! [`super::native`]); this module owns only the Rust-side concerns the C ABI
//! does not cover: binding a parsed query to a [`Schema`] (tables/columns
//! exist, the entity key is a primary key, target windows are future-facing)
//! and inferring its task type.

use std::fmt;

use super::ast::{AggFunc, ColumnRef, CondRhs, ParsedQuery, TargetExpr};
use crate::schema::Schema;

/// A schema-binding validation error.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ValidationError(pub String);

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RelQL validation error: {}", self.0)
    }
}
impl std::error::Error for ValidationError {}

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
        TargetExpr::Condition(c) => {
            walk_columns(&c.left, schema)?;
            if let CondRhs::Expr(e) = &c.right {
                walk_columns(e, schema)?;
            }
            Ok(())
        }
        TargetExpr::LogicalOp(l) => {
            walk_columns(&l.left, schema)?;
            walk_columns(&l.right, schema)
        }
        TargetExpr::Not(e) => walk_columns(e, schema),
        TargetExpr::Arith(a) => {
            walk_columns(&a.left, schema)?;
            walk_columns(&a.right, schema)
        }
        TargetExpr::Func(f) => {
            for a in &f.args {
                walk_columns(a, schema)?;
            }
            Ok(())
        }
        TargetExpr::Case(c) => {
            for (cond, then) in &c.whens {
                walk_columns(cond, schema)?;
                walk_columns(then, schema)?;
            }
            if let Some(e) = &c.else_ {
                walk_columns(e, schema)?;
            }
            Ok(())
        }
        TargetExpr::Lit(_) => Ok(()),
    }
}

/// Error if any aggregation reachable from `expr` carries a multi-horizon
/// (`HORIZONS > 1`) window — permitted only on the PREDICT target.
fn reject_multi_horizon(expr: &TargetExpr, clause: &str) -> Result<(), ValidationError> {
    for agg in expr.aggregations() {
        if let Some(w) = &agg.window {
            if w.horizons > 1 {
                return Err(ValidationError(format!(
                    "HORIZONS > 1 is only allowed on the PREDICT target, not in {}",
                    clause
                )));
            }
        }
    }
    Ok(())
}

/// Bind a parsed query against a schema: tables/columns exist, the entity key is
/// a primary key, target windows are future-facing (start >= 0).
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
        reject_multi_horizon(w, "WHERE")?;
    }
    if let Some(a) = &query.assuming {
        walk_columns(a, schema)?;
        reject_multi_horizon(a, "ASSUMING")?;
    }
    let task_type = query.task_type(Some(schema));
    if let Some(ret) = &query.ret {
        validate_return(ret, task_type)?;
    }
    Ok(ValidatedQuery { query: query.clone(), task_type })
}

/// Enforce the RETURN/task compatibility matrix (RETURN execution contract §1).
fn validate_return(
    ret: &super::ast::ReturnSpec,
    task: super::ast::TaskType,
) -> Result<(), ValidationError> {
    use super::ast::TaskType::*;
    let allowed: &[super::ast::TaskType] = match ret.kind.as_str() {
        "EXPECTED_VALUE" => &[Regression, Forecasting, BinaryClassification],
        "PROBABILITY" => &[BinaryClassification],
        "CLASS" => &[BinaryClassification, MulticlassClassification],
        "DISTRIBUTION" => &[BinaryClassification, MulticlassClassification],
        "QUANTILES" => &[Regression, Forecasting],
        "INTERVAL" => &[Regression, Forecasting],
        "MULTILABEL" => &[MultilabelRanking],
        "MULTICLASS" => &[MulticlassClassification],
        other => {
            return Err(ValidationError(format!(
                "unknown RETURN kind {:?}",
                other
            )))
        }
    };
    if !allowed.contains(&task) {
        return Err(ValidationError(format!(
            "RETURN {} is not compatible with the inferred task type {:?}",
            ret.kind, task
        )));
    }
    if ret.kind == "QUANTILES" {
        for &q in &ret.quantiles {
            if !(q > 0.0 && q < 1.0) {
                return Err(ValidationError(format!(
                    "RETURN QUANTILES: each quantile must be in (0, 1), got {}",
                    q
                )));
            }
        }
    }
    if ret.kind == "INTERVAL" {
        let pct = ret.interval.unwrap_or(0);
        if !(pct > 0 && pct < 100) {
            return Err(ValidationError(format!(
                "RETURN INTERVAL: percent must be in (0, 100), got {}",
                pct
            )));
        }
    }
    Ok(())
}

/// Convenience: parse a string (via the shared native parser) then validate.
pub fn parse_and_validate(query: &str, schema: &Schema) -> Result<ValidatedQuery, crate::Error> {
    let pq = super::parse(query)?;
    Ok(validate(&pq, schema)?)
}
