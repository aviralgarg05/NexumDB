use thiserror::Error;

/// Stable error codes for programmatic handling
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageErrorCode {
    OpenFailed,
    ReadFailed,
    WriteFailed,
    KeyNotFound,
    SerializationFailed,
}
#[derive(Error, Debug)]
pub enum StorageError {
    #[error("failed to open database")]
    Open {
        #[source]
        source: sled::Error,
    },

    #[error("failed to write to database")]
    Write {
        #[source]
        source: sled::Error,
    },

    #[error("failed to read from database")]
    Read {
        #[source]
        source: sled::Error,
    },

    #[error("semantic cache write failed")]
    CacheWrite {
        #[source]
        source: anyhow::Error,
    },

    #[error("semantic cache read failed")]
    CacheRead {
        #[source]
        source: anyhow::Error,
    },

    #[error("Key not found: {key}")]
    KeyNotFound { key: String },

    #[error("Serialization error")]
    Serialization {
        #[source]
        source: serde_json::Error,
    },
    // new variant for application-level logical write errors
    #[error("logical write error: {message}")]
    LogicalWrite { message: String },
}

impl StorageError {
    /// Machine-readable error codes
    pub fn code(&self) -> StorageErrorCode {
        match self {
            StorageError::Open { .. } => StorageErrorCode::OpenFailed,
            StorageError::Write { .. } => StorageErrorCode::WriteFailed,
            StorageError::Read { .. } => StorageErrorCode::ReadFailed,
            StorageError::KeyNotFound { .. } => StorageErrorCode::KeyNotFound,
            StorageError::Serialization { .. } => StorageErrorCode::SerializationFailed,
            StorageError::CacheWrite { .. } => StorageErrorCode::WriteFailed,
            StorageError::CacheRead { .. } => StorageErrorCode::ReadFailed,
            StorageError::LogicalWrite { .. } => StorageErrorCode::WriteFailed,
        }
    }
}
// //implement From conversions for error propagation
// impl From<sled::Error> for StorageError {
//     fn from(err: sled::Error) -> Self {
//         StorageError::Write{source: err}
//         }
//     }

impl From<serde_json::Error> for StorageError {
    fn from(err: serde_json::Error) -> Self {
        StorageError::Serialization { source: err }
    }
}
