use thiserror::Error;

#[derive(Debug, Error)]
pub enum FilterError {
    #[error("Column not found: {0}")]
    ColumnNotFound(String),

    #[error("Expected boolean value")]
    ExpectedBoolean,

    #[error("Unsupported expression type")]
    UnsupportedExpression,

    #[error("Unsupported operator: {0}")]
    UnsupportedOperator(String),

    #[error("Unsupported SQL value")]
    UnsupportedValue,

    #[error("Type mismatch: {0} vs {1}")]
    TypeMismatch(String, String),

    #[error("Invalid LIKE pattern: {0}")]
    InvalidLikePattern(String),

    #[error("Cannot extract value from expression: {0}")]
    ExtractionError(String),

    #[error("Parse error: {0}")]
    ParseError(String),
}
pub type Result<T> = std::result::Result<T, FilterError>;
