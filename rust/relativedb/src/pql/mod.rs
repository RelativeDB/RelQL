//! RelQL: parsing (via the shared C++ parser), typed AST, schema-bound
//! validation, task-type inference.

pub mod ast;
pub mod native;
pub mod validate;

pub use ast::{
    Ablation, AggFunc, Aggregation, Arith, AsOf, BoolOp, Case, ColumnRef, CondRhs, Condition,
    Explain, Func, Literal, LogicalOp, Operator, ParsedQuery, RankKind, ReturnSpec, TargetExpr,
    TaskType, TimeUnit, Window,
};
pub use native::SyntaxError;
pub use validate::{parse_and_validate, validate, ValidatedQuery, ValidationError};

/// Parse a RelQL query string into a [`ParsedQuery`] (no schema needed).
///
/// The parser is single-sourced on the shared C++ implementation (`pql_parse`
/// in `librt_c`) — the same parser used by the Python and Java bindings. There
/// is no hand-written fallback: `librt_c` is a hard runtime dependency, and if
/// it cannot be loaded this returns a clear [`SyntaxError`] naming the searched
/// paths.
pub fn parse(query: &str) -> Result<ParsedQuery, SyntaxError> {
    native::parse_native(query)
}
