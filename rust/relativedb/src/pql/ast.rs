//! Typed RelQL AST + task-type inference.
//!
//! Mirrors the `com.relativedb.query` records (Java) / `relativedb.pql.ast`
//! (Python).

use std::collections::HashMap;

use chrono::{DateTime, Duration, Utc};

use crate::schema::{Schema, ValueType};

/// The platform aggregations.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum AggFunc {
    Sum,
    Avg,
    Min,
    Max,
    Count,
    CountDistinct,
    ListDistinct,
    First,
    Last,
    /// Boolean-existence aggregation (`EXISTS(t.*)`).
    Exists,
}

impl AggFunc {
    /// Uppercase keyword name (as it appears in RelQL / the grammar).
    pub fn keyword(&self) -> &'static str {
        match self {
            AggFunc::Sum => "SUM",
            AggFunc::Avg => "AVG",
            AggFunc::Min => "MIN",
            AggFunc::Max => "MAX",
            AggFunc::Count => "COUNT",
            AggFunc::CountDistinct => "COUNT_DISTINCT",
            AggFunc::ListDistinct => "LIST_DISTINCT",
            AggFunc::First => "FIRST",
            AggFunc::Last => "LAST",
            AggFunc::Exists => "EXISTS",
        }
    }

    pub fn from_keyword(kw: &str) -> Option<AggFunc> {
        Some(match kw {
            "SUM" => AggFunc::Sum,
            "AVG" => AggFunc::Avg,
            "MIN" => AggFunc::Min,
            "MAX" => AggFunc::Max,
            "COUNT" => AggFunc::Count,
            "COUNT_DISTINCT" => AggFunc::CountDistinct,
            "LIST_DISTINCT" => AggFunc::ListDistinct,
            "FIRST" => AggFunc::First,
            "LAST" => AggFunc::Last,
            "EXISTS" => AggFunc::Exists,
            _ => return None,
        })
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum TimeUnit {
    Seconds,
    Minutes,
    Hours,
    Days,
    Weeks,
    Months,
    Years,
}

impl TimeUnit {
    pub fn from_keyword(kw: &str) -> Option<TimeUnit> {
        Some(match kw {
            "SECONDS" => TimeUnit::Seconds,
            "MINUTES" => TimeUnit::Minutes,
            "HOURS" => TimeUnit::Hours,
            "DAYS" => TimeUnit::Days,
            "WEEKS" => TimeUnit::Weeks,
            "MONTHS" => TimeUnit::Months,
            "YEARS" => TimeUnit::Years,
            _ => return None,
        })
    }

    fn unit_seconds(&self) -> f64 {
        match self {
            TimeUnit::Seconds => 1.0,
            TimeUnit::Minutes => 60.0,
            TimeUnit::Hours => 3600.0,
            TimeUnit::Days => 86_400.0,
            TimeUnit::Weeks => 604_800.0,
            // MONTHS: calendar months are irregular; 30-day approximation,
            // matching the engine's window arithmetic.
            TimeUnit::Months => 2_592_000.0,
            // YEARS: defensive only — the C++ parser normalizes calendar
            // frames to MONTHS and never emits "years". 365-day approximation.
            TimeUnit::Years => 31_536_000.0,
        }
    }

    pub fn delta(&self, n: f64) -> Duration {
        Duration::milliseconds((n * self.unit_seconds() * 1000.0).round() as i64)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Operator {
    Gt,
    Lt,
    Eq,
    Neq,
    Ge,
    Le,
    StartsWith,
    EndsWith,
    Contains,
    NotContains,
    Like,
    NotLike,
    In,
    NotIn,
    IsNull,
    IsNotNull,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum BoolOp {
    And,
    Or,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum RankKind {
    Classify,
    Rank,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum TaskType {
    Regression,
    BinaryClassification,
    MulticlassClassification,
    MultilabelRanking,
    Forecasting,
}

impl TaskType {
    pub fn is_classification(&self) -> bool {
        matches!(
            self,
            TaskType::BinaryClassification
                | TaskType::MulticlassClassification
                | TaskType::MultilabelRanking
        )
    }
}

/// `table.column` — column may be `"*"`.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct ColumnRef {
    pub table: String,
    pub column: String,
}

impl ColumnRef {
    pub fn new(table: impl Into<String>, column: impl Into<String>) -> ColumnRef {
        ColumnRef { table: table.into(), column: column.into() }
    }
}

impl std::fmt::Display for ColumnRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}", self.table, self.column)
    }
}

/// A RelQL literal (string / number / boolean / date / null).
#[derive(Clone, PartialEq, Debug)]
pub enum Literal {
    Str(String),
    Num(f64),
    Bool(bool),
    Date(DateTime<Utc>),
    Null,
}

/// Aggregation window `(start, end]` in `unit`.
///
/// `start` is EXCLUDED, `end` is INCLUDED; `±inf` for unbounded.
///
/// `horizons` is the number of shifted frame copies (default 1; >1 = multi-
/// horizon forecasting). `step` is the distance between horizon starts in
/// `unit` (default = the frame width `end - start`).
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Window {
    pub start: f64,
    pub end: f64,
    pub unit: TimeUnit,
    pub horizons: i64,
    pub step: Option<f64>,
}

impl Window {
    /// Offset from anchor for the (excluded) start, or `None` if `-inf`.
    pub fn start_offset(&self) -> Option<Duration> {
        if self.start.is_infinite() {
            None
        } else {
            Some(self.unit.delta(self.start))
        }
    }

    /// Offset from anchor for the (included) end, or `None` if `+inf`.
    pub fn end_offset(&self) -> Option<Duration> {
        if self.end.is_infinite() {
            None
        } else {
            Some(self.unit.delta(self.end))
        }
    }

    pub fn span(&self) -> Option<Duration> {
        if self.start.is_infinite() || self.end.is_infinite() {
            None
        } else {
            Some(self.unit.delta(self.end - self.start))
        }
    }
}

/// A temporal aggregation over a fact column.
#[derive(Clone, PartialEq, Debug)]
pub struct Aggregation {
    pub func: AggFunc,
    pub column: ColumnRef,
    /// Inline `WHERE` inside the aggregation.
    pub filter: Option<Box<TargetExpr>>,
    /// `None` = static (windowless) agg.
    pub window: Option<Window>,
}

/// A comparison / membership / null-test condition, or a bare value predicate.
#[derive(Clone, PartialEq, Debug)]
pub struct Condition {
    pub left: Box<TargetExpr>,
    pub op: Operator,
    pub right: CondRhs,
}

/// The right-hand side of a [`Condition`].
#[derive(Clone, PartialEq, Debug)]
pub enum CondRhs {
    /// `IS NULL` / `IS NOT NULL` — no operand.
    Empty,
    One(Literal),
    List(Vec<Literal>),
    /// Column-to-column / expression RHS comparison (`right_expr` in the AST).
    Expr(Box<TargetExpr>),
}

#[derive(Clone, PartialEq, Debug)]
pub struct LogicalOp {
    pub left: Box<TargetExpr>,
    pub op: BoolOp,
    pub right: Box<TargetExpr>,
}

/// Arithmetic combination of two value expressions (`+ - * /`).
#[derive(Clone, PartialEq, Debug)]
pub struct Arith {
    pub op: char,
    pub left: Box<TargetExpr>,
    pub right: Box<TargetExpr>,
}

/// A scalar function call
/// (`COALESCE|NULLIF|ABS|LOG|EXP|LEAST|GREATEST`).
#[derive(Clone, PartialEq, Debug)]
pub struct Func {
    pub name: String,
    pub args: Vec<TargetExpr>,
}

/// `CASE WHEN cond THEN then ... [ELSE else] END`.
#[derive(Clone, PartialEq, Debug)]
pub struct Case {
    pub whens: Vec<(TargetExpr, TargetExpr)>,
    pub else_: Option<Box<TargetExpr>>,
}

/// The typed target/where/assuming expression tree.
#[derive(Clone, PartialEq, Debug)]
pub enum TargetExpr {
    Aggregation(Aggregation),
    ColumnRef(ColumnRef),
    Condition(Condition),
    LogicalOp(LogicalOp),
    Not(Box<TargetExpr>),
    Arith(Arith),
    Func(Func),
    Case(Case),
    Lit(Literal),
}

impl TargetExpr {
    /// Every aggregation reachable via boolean/comparison structure.
    pub fn aggregations(&self) -> Vec<&Aggregation> {
        let mut out = Vec::new();
        self.collect_aggregations(&mut out);
        out
    }

    fn collect_aggregations<'a>(&'a self, out: &mut Vec<&'a Aggregation>) {
        match self {
            TargetExpr::Aggregation(a) => out.push(a),
            TargetExpr::Condition(c) => {
                c.left.collect_aggregations(out);
                if let CondRhs::Expr(e) = &c.right {
                    e.collect_aggregations(out);
                }
            }
            TargetExpr::LogicalOp(l) => {
                l.left.collect_aggregations(out);
                l.right.collect_aggregations(out);
            }
            TargetExpr::Not(e) => e.collect_aggregations(out),
            TargetExpr::Arith(a) => {
                a.left.collect_aggregations(out);
                a.right.collect_aggregations(out);
            }
            TargetExpr::Func(f) => {
                for a in &f.args {
                    a.collect_aggregations(out);
                }
            }
            TargetExpr::Case(c) => {
                for (cond, then) in &c.whens {
                    cond.collect_aggregations(out);
                    then.collect_aggregations(out);
                }
                if let Some(e) = &c.else_ {
                    e.collect_aggregations(out);
                }
            }
            TargetExpr::ColumnRef(_) | TargetExpr::Lit(_) => {}
        }
    }
}

/// `EXPLAIN [mode] [FORMAT fmt]` prefix.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Explain {
    /// `PLAN | CONTEXT | ANALYZE | ABLATION`.
    pub mode: String,
    /// `TEXT | JSON`.
    pub format: String,
}

/// `AS OF <anchor>` — binds NOW.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct AsOf {
    /// `param | date | now`.
    pub kind: String,
    /// The parameter name (no colon) or the date string; `None` for `now`.
    pub value: Option<String>,
}

/// `ABLATE TABLE <name>`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Ablation {
    /// Always `"table"` for now.
    pub kind: String,
    pub name: String,
}

/// `RETURN <output>` — explicit output intent.
#[derive(Clone, PartialEq, Debug)]
pub struct ReturnSpec {
    /// `EXPECTED_VALUE|PROBABILITY|CLASS|DISTRIBUTION|QUANTILES|INTERVAL|MULTILABEL|MULTICLASS`.
    pub kind: String,
    pub quantiles: Vec<f64>,
    pub interval: Option<i64>,
}

/// The parse result — no schema needed. `validate` binds it to one.
#[derive(Clone, PartialEq, Debug)]
pub struct ParsedQuery {
    pub target: TargetExpr,
    pub entity_key: ColumnRef,
    pub where_: Option<TargetExpr>,
    pub assuming: Option<TargetExpr>,
    pub rank: Option<RankKind>,
    pub top_k: Option<i64>,
    pub num_forecasts: Option<i64>,
    /// `EXPLAIN` prefix (represented, not executed).
    pub explain: Option<Explain>,
    /// `AS OF` anchor binding (represented, not executed).
    pub as_of: Option<AsOf>,
    /// `ABLATE TABLE` clauses (represented, not executed).
    pub ablations: Vec<Ablation>,
    /// `RETURN` output intent (represented, not executed).
    pub ret: Option<ReturnSpec>,
    /// Declared `WINDOW name AS (...)` templates (normalized).
    pub windows: HashMap<String, Window>,
    pub text: String,
}

impl ParsedQuery {
    pub fn target_aggregations(&self) -> Vec<&Aggregation> {
        self.target.aggregations()
    }

    /// Infer the task type (design §4: execution semantics, step 1).
    pub fn task_type(&self, schema: Option<&Schema>) -> TaskType {
        if self.num_forecasts.is_some() {
            return TaskType::Forecasting;
        }
        match self.rank {
            Some(RankKind::Rank) => return TaskType::MultilabelRanking,
            Some(RankKind::Classify) => return TaskType::MulticlassClassification,
            None => {}
        }
        match &self.target {
            TargetExpr::Condition(_) | TargetExpr::LogicalOp(_) | TargetExpr::Not(_) => {
                TaskType::BinaryClassification
            }
            TargetExpr::Aggregation(a) => match a.func {
                AggFunc::Exists => TaskType::BinaryClassification,
                AggFunc::ListDistinct => TaskType::MultilabelRanking,
                AggFunc::First | AggFunc::Last => Self::static_or_categorical(
                    &a.column,
                    schema,
                    TaskType::MulticlassClassification,
                ),
                _ => TaskType::Regression,
            },
            TargetExpr::ColumnRef(c) => {
                Self::static_or_categorical(c, schema, TaskType::MulticlassClassification)
            }
            // Boolean literal target is a (degenerate) binary target; other
            // literals and arithmetic/function/CASE value expressions are
            // numeric → regression.
            TargetExpr::Lit(Literal::Bool(_)) => TaskType::BinaryClassification,
            TargetExpr::Lit(_)
            | TargetExpr::Arith(_)
            | TargetExpr::Func(_)
            | TargetExpr::Case(_) => TaskType::Regression,
        }
    }

    fn static_or_categorical(col: &ColumnRef, schema: Option<&Schema>, default: TaskType) -> TaskType {
        let schema = match schema {
            Some(s) => s,
            None => return default,
        };
        let cdef = schema.table(&col.table).and_then(|t| t.column(&col.column));
        match cdef {
            None => default,
            Some(c) => match c.value_type {
                ValueType::Number => TaskType::Regression,
                ValueType::Boolean => TaskType::BinaryClassification,
                ValueType::Datetime => TaskType::Regression,
                ValueType::Text => TaskType::MulticlassClassification,
            },
        }
    }
}
