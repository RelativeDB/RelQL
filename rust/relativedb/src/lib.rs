//! # relativedb — predictive queries (RelQL) over your own data
//!
//! The Rust peer of the Java (`com.relativedb.*`) and Python (`relativedb`)
//! libraries. GraphQL-style execution: the engine owns the query language,
//! planning, context assembly, and model routing — all data access goes through
//! user-defined **retrievers**. No bundled database connectors.
//!
//! Modules:
//! * [`schema`] — [`Schema`], [`TableDef`], [`ColumnDef`], [`LinkDef`],
//!   [`ValueType`]; builder + validation (links resolve, link targets need PKs,
//!   PK/FK columns may not be feature columns — the F17 invariant).
//! * [`retrieve`] — the SPI traits ([`EntityRetriever`], [`LinkRetriever`],
//!   [`CohortRetriever`], [`TableScanner`], [`StatsProvider`]), [`Row`],
//!   [`TemporalBound`], [`RetrieverWiring`]. **Synchronous** SPI (see the
//!   module docs for the async-vs-sync rationale).
//! * [`pql`] — RelQL parsing single-sourced on the shared C++ parser (`pql_parse`
//!   in `librt_c`), a typed AST, schema-bound validation, and task-type
//!   inference.
//! * [`engine`] — [`Engine`], [`ExecutionInput`], [`ContextPolicy`],
//!   [`SamplerMode`], [`PredictionResult`]; the real hop-loop context assembly
//!   and the CSC in-memory sampler ([`csc`]).
//! * [`model`] — [`ModelConfig`] with default RT-J URIs + task-type routing.
//! * [`native`] — the shared native backend binding the C++ RT-J engine via its
//!   C ABI (`librt_c`), plus a precomputed-embeddings [`TextEncoder`].

pub mod csc;
pub mod csc_native;
pub mod engine;
pub mod evaluate;
pub mod model;
pub mod native;
pub mod pql;
pub mod retrieve;
pub mod schema;

use std::fmt;

pub use engine::{
    ContextPolicy, Engine, EntityContext, EntityPrediction, ExecutionError, ExecutionInput,
    HistoryBaselineBackend, ModelBackend, PredictionResult, SamplerMode,
};
pub use model::{
    EmbeddingMismatchError, ModelConfig, DEFAULT_CLASSIFICATION_MODEL_URI,
    DEFAULT_EMBEDDING_MODEL, DEFAULT_REGRESSION_MODEL_URI,
};
pub use pql::{
    parse, validate, AggFunc, Aggregation, BoolOp, ColumnRef, CondRhs, Condition, Literal,
    LogicalOp, Operator, ParsedQuery, RankKind, SyntaxError, TargetExpr, TaskType, TimeUnit,
    ValidatedQuery, ValidationError, Window,
};
pub use retrieve::{
    CohortRetriever, ColumnStats, DatetimeStats, EntityId, EntityRetriever, LinkRetriever, Row,
    RetrieverWiring, StatsProvider, TableScanner, TemporalBound, Value, WiringError,
};
pub use schema::{ColumnDef, LinkDef, Schema, SchemaError, TableDef, ValueType};

/// The crate's unified error type, aggregating the errors owned by the engine
/// (parse / validate / wiring / execution / native).
#[derive(Debug)]
pub enum Error {
    Syntax(SyntaxError),
    Validation(ValidationError),
    Schema(SchemaError),
    Wiring(WiringError),
    Execution(ExecutionError),
    Rt(native::RtError),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Syntax(e) => write!(f, "{}", e),
            Error::Validation(e) => write!(f, "{}", e),
            Error::Schema(e) => write!(f, "{}", e),
            Error::Wiring(e) => write!(f, "{}", e),
            Error::Execution(e) => write!(f, "{}", e),
            Error::Rt(e) => write!(f, "{}", e),
        }
    }
}
impl std::error::Error for Error {}

impl From<SyntaxError> for Error {
    fn from(e: SyntaxError) -> Self {
        Error::Syntax(e)
    }
}
impl From<ValidationError> for Error {
    fn from(e: ValidationError) -> Self {
        Error::Validation(e)
    }
}
impl From<SchemaError> for Error {
    fn from(e: SchemaError) -> Self {
        Error::Schema(e)
    }
}
impl From<WiringError> for Error {
    fn from(e: WiringError) -> Self {
        Error::Wiring(e)
    }
}
impl From<ExecutionError> for Error {
    fn from(e: ExecutionError) -> Self {
        Error::Execution(e)
    }
}
impl From<native::RtError> for Error {
    fn from(e: native::RtError) -> Self {
        Error::Rt(e)
    }
}

/// The crate result type.
pub type Result<T> = std::result::Result<T, Error>;
