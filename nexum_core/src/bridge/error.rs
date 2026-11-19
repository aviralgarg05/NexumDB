use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCode {
    BridgeNotInitialized,
    BridgeInterpreterUnavailable,
    BridgePythonError,
    BridgeDirectoryError,
    BridgeInvalidPath,
}
#[derive(Debug, Error)]
pub enum BridgeError {
    #[error("Python bridge not initialized")]
    NotInitialized,

    #[error("Python interpreter not available")]
    InterpreterUnavailable,

    #[error("Python operation failed: {0}")]
    PythonError(String),

    #[error("Failed to get current directory")]
    DirectoryError,

    #[error("Invalid path")]
    InvalidPath,
}

pub type Result<T> = std::result::Result<T, BridgeError>;

impl BridgeError {
    pub fn code(&self) -> ErrorCode {
        match self {
            BridgeError::NotInitialized => ErrorCode::BridgeNotInitialized,
            BridgeError::InterpreterUnavailable => ErrorCode::BridgeInterpreterUnavailable,
            BridgeError::PythonError(_) => ErrorCode::BridgePythonError,
            BridgeError::DirectoryError => ErrorCode::BridgeDirectoryError,
            BridgeError::InvalidPath => ErrorCode::BridgeInvalidPath,
        }
    }
}
