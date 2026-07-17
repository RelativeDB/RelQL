//! PQL: parsing, typed AST, schema-bound validation, task-type inference.

pub mod ast;
pub mod parser;

pub use ast::{
    AggFunc, Aggregation, BoolOp, ColumnRef, CondRhs, Condition, Literal, LogicalOp, Operator,
    ParsedQuery, RankKind, TargetExpr, TaskType, TimeUnit, Window,
};
pub use parser::{parse, parse_and_validate, validate, SyntaxError, ValidatedQuery, ValidationError};
