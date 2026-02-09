use std::fmt;
use thiserror::Error;

/// Error codes for programmatic handling of storage errors.
/// Format: NXM-STOR-xxx where xxx is a three-digit code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCode {
    /// Database open failure
    NxmStor101,
    /// Write operation failure
    NxmStor102,
    /// Read operation failure
    NxmStor103,
    /// Key/table not found
    NxmStor104,
    /// Serialization failure
    NxmStor105,
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ErrorCode::NxmStor101 => write!(f, "NXM-STOR-101"),
            ErrorCode::NxmStor102 => write!(f, "NXM-STOR-102"),
            ErrorCode::NxmStor103 => write!(f, "NXM-STOR-103"),
            ErrorCode::NxmStor104 => write!(f, "NXM-STOR-104"),
            ErrorCode::NxmStor105 => write!(f, "NXM-STOR-105"),
        }
    }
}

/// Storage layer errors with enhanced context and suggestions.
#[derive(Error, Debug)]
pub enum StorageError {
    #[error("[{code}] Failed to open database\n  Reason: {reason}\n  Suggestion: {suggestion}")]
    OpenError {
        code: ErrorCode,
        reason: String,
        suggestion: String,
    },

    #[error(
        "[{code}] Failed to write to database\n  Reason: {reason}\n  Suggestion: {suggestion}"
    )]
    WriteError {
        code: ErrorCode,
        reason: String,
        suggestion: String,
    },

    #[error(
        "[{code}] Failed to read from database\n  Reason: {reason}\n  Suggestion: {suggestion}"
    )]
    ReadError {
        code: ErrorCode,
        reason: String,
        suggestion: String,
    },

    #[error("[{code}] Table or key not found: '{key}'\n  Context: {context}\n  Suggestion: {suggestion}")]
    KeyNotFound {
        code: ErrorCode,
        key: String,
        context: String,
        suggestion: String,
    },

    #[error("[{code}] Data serialization failed\n  Reason: {reason}\n  Data preview: {data_preview}\n  Suggestion: {suggestion}")]
    SerializationError {
        code: ErrorCode,
        reason: String,
        data_preview: String,
        suggestion: String,
    },
}

impl StorageError {
    /// Get the error code for programmatic handling.
    ///
    /// # Returns
    ///
    /// The `ErrorCode` associated with this error.
    pub fn code(&self) -> ErrorCode {
        match self {
            StorageError::OpenError { code, .. } => *code,
            StorageError::WriteError { code, .. } => *code,
            StorageError::ReadError { code, .. } => *code,
            StorageError::KeyNotFound { code, .. } => *code,
            StorageError::SerializationError { code, .. } => *code,
        }
    }

    /// Create a user-friendly OpenError with context-aware suggestions.
    ///
    /// # Arguments
    ///
    /// * `reason` - The underlying error message
    ///
    /// # Returns
    ///
    /// A `StorageError::OpenError` with helpful suggestions based on the error.
    ///
    /// # Example
    ///
    /// ```
    /// use nexum_core::storage::StorageError;
    /// let err = StorageError::OpenError("Permission denied".to_string());
    /// ```
    #[allow(non_snake_case)]
    pub fn OpenError(reason: String) -> Self {
        Self::open_error(reason)
    }

    /// Create a user-friendly WriteError with context-aware suggestions.
    ///
    /// # Arguments
    ///
    /// * `reason` - The underlying error message
    ///
    /// # Returns
    ///
    /// A `StorageError::WriteError` with helpful suggestions based on the error.
    ///
    /// # Example
    ///
    /// ```
    /// use nexum_core::storage::StorageError;
    /// let err = StorageError::WriteError("Disk full".to_string());
    /// ```
    #[allow(non_snake_case)]
    pub fn WriteError(reason: String) -> Self {
        Self::write_error(reason)
    }

    /// Create a user-friendly ReadError with context-aware suggestions.
    ///
    /// # Arguments
    ///
    /// * `reason` - The underlying error message
    ///
    /// # Returns
    ///
    /// A `StorageError::ReadError` with helpful suggestions based on the error.
    ///
    /// # Example
    ///
    /// ```
    /// use nexum_core::storage::StorageError;
    /// let err = StorageError::ReadError("Table not found".to_string());
    /// ```
    #[allow(non_snake_case)]
    pub fn ReadError(reason: String) -> Self {
        Self::read_error(reason)
    }

    /// Create a user-friendly KeyNotFound error.
    ///
    /// # Arguments
    ///
    /// * `key` - The key that was not found
    ///
    /// # Returns
    ///
    /// A `StorageError::KeyNotFound` with suggestions.
    ///
    /// # Example
    ///
    /// ```
    /// use nexum_core::storage::StorageError;
    /// let err = StorageError::KeyNotFound("users".to_string());
    /// ```
    #[allow(non_snake_case)]
    pub fn KeyNotFound(key: String) -> Self {
        Self::key_not_found(key, "Key lookup failed", vec![])
    }

    /// Create a user-friendly SerializationError.
    ///
    /// # Arguments
    ///
    /// * `reason` - The underlying error message
    ///
    /// # Returns
    ///
    /// A `StorageError::SerializationError` with helpful suggestions.
    ///
    /// # Example
    ///
    /// ```
    /// use nexum_core::storage::StorageError;
    /// let err = StorageError::SerializationError("Invalid JSON".to_string());
    /// ```
    #[allow(non_snake_case)]
    pub fn SerializationError(reason: String) -> Self {
        Self::serialization_error(reason, "")
    }

    /// Internal helper to create OpenError with context-aware suggestions.
    fn open_error(reason: impl Into<String>) -> Self {
        let reason = reason.into();
        let suggestion = if reason.contains("Permission denied") {
            "Check file permissions or run with appropriate privileges".to_string()
        } else if reason.contains("No space left") {
            "Free up disk space or change the database path".to_string()
        } else if reason.contains("already in use") || reason.contains("locked") {
            "Close other instances of NexumDB or check for stale lock files".to_string()
        } else {
            "Verify the database path exists and is accessible".to_string()
        };

        StorageError::OpenError {
            code: ErrorCode::NxmStor101,
            reason,
            suggestion,
        }
    }

    /// Internal helper to create WriteError with context-aware suggestions.
    fn write_error(reason: impl Into<String>) -> Self {
        let reason = reason.into();
        let suggestion = if reason.contains("read-only") {
            "The database is in read-only mode. Check permissions or mount options".to_string()
        } else if reason.contains("No space left") {
            "Free up disk space to continue writing data".to_string()
        } else if reason.contains("corrupted") {
            "The database may be corrupted. Consider restoring from backup".to_string()
        } else {
            "Ensure the database has write permissions and sufficient disk space".to_string()
        };

        StorageError::WriteError {
            code: ErrorCode::NxmStor102,
            reason,
            suggestion,
        }
    }

    /// Internal helper to create ReadError with context-aware suggestions.
    fn read_error(reason: impl Into<String>) -> Self {
        let reason = reason.into();
        let suggestion = if reason.contains("corrupted") {
            "The data may be corrupted. Try rebuilding the database or restoring from backup"
                .to_string()
        } else if reason.contains("not found") {
            "The database file may have been moved or deleted".to_string()
        } else {
            "Verify the database integrity and try reopening the connection".to_string()
        };

        StorageError::ReadError {
            code: ErrorCode::NxmStor103,
            reason,
            suggestion,
        }
    }

    /// Create a KeyNotFound error with "Did you mean?" suggestions.
    ///
    /// # Arguments
    ///
    /// * `key` - The key or table name that was not found
    /// * `context` - Additional context about where the error occurred
    /// * `similar_keys` - List of similar key names for suggestions
    ///
    /// # Returns
    ///
    /// A `StorageError::KeyNotFound` with fuzzy-matched suggestions.
    pub fn key_not_found(
        key: impl Into<String>,
        context: impl Into<String>,
        similar_keys: Vec<String>,
    ) -> Self {
        let key = key.into();
        let context = context.into();

        let suggestion = if !similar_keys.is_empty() {
            if similar_keys.len() == 1 {
                format!(
                    "Did you mean '{}'?\nUse 'SHOW TABLES' to see all available tables",
                    similar_keys[0]
                )
            } else {
                format!(
                    "Did you mean one of these? {}\nUse 'SHOW TABLES' to see all available tables",
                    similar_keys.join(", ")
                )
            }
        } else {
            "Use 'SHOW TABLES' to see all available tables or create it with 'CREATE TABLE'"
                .to_string()
        };

        StorageError::KeyNotFound {
            code: ErrorCode::NxmStor104,
            key,
            context,
            suggestion,
        }
    }

    /// Internal helper to create SerializationError.
    fn serialization_error(reason: impl Into<String>, data_preview: impl Into<String>) -> Self {
        let reason = reason.into();
        let data_preview = data_preview.into();

        let suggestion = if reason.contains("type") {
            "Check that your data types match the table schema".to_string()
        } else if reason.contains("EOF") || reason.contains("unexpected end") {
            "The data may be incomplete or corrupted".to_string()
        } else if reason.contains("invalid") {
            "Ensure your data is valid JSON and matches the expected schema".to_string()
        } else {
            "Verify your data format matches the table schema and retry the operation".to_string()
        };

        StorageError::SerializationError {
            code: ErrorCode::NxmStor105,
            reason,
            data_preview: truncate_preview(&data_preview, 100),
            suggestion,
        }
    }

    /// Add additional context to an existing error.
    ///
    /// # Arguments
    ///
    /// * `context` - Additional contextual information
    ///
    /// # Returns
    ///
    /// The error with updated context.
    pub fn with_context(self, context: impl Into<String>) -> Self {
        let ctx = context.into();
        match self {
            StorageError::KeyNotFound {
                code,
                key,
                context: old_ctx,
                suggestion,
            } => StorageError::KeyNotFound {
                code,
                key,
                context: format!("{} | {}", old_ctx, ctx),
                suggestion,
            },
            _ => self,
        }
    }
}

/// Helper function to truncate data previews for error messages.
///
/// # Arguments
///
/// * `s` - The string to truncate
/// * `max_len` - Maximum length before truncation
///
/// # Returns
///
/// Truncated string with ellipsis if needed.
fn truncate_preview(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}... (truncated)", &s[..max_len])
    }
}

/// Find similar keys using Levenshtein distance for "Did you mean?" suggestions.
///
/// # Arguments
///
/// * `target` - The key that was not found
/// * `candidates` - List of available keys to compare against
/// * `max_distance` - Maximum edit distance to consider a match
///
/// # Returns
///
/// List of up to 3 similar keys.
#[allow(dead_code)]
pub fn find_similar_keys(target: &str, candidates: &[String], max_distance: usize) -> Vec<String> {
    let mut results: Vec<(String, usize)> = candidates
        .iter()
        .filter_map(|candidate| {
            let distance = levenshtein_distance(target, candidate);
            if distance <= max_distance && distance > 0 {
                Some((candidate.clone(), distance))
            } else {
                None
            }
        })
        .collect();

    // Sort by distance (closest matches first)
    results.sort_by_key(|(_, dist)| *dist);

    // Return top 3 suggestions
    results.into_iter().map(|(s, _)| s).take(3).collect()
}

/// Calculate Levenshtein distance between two strings.
///
/// # Arguments
///
/// * `s1` - First string
/// * `s2` - Second string
///
/// # Returns
///
/// The minimum number of single-character edits required to change s1 into s2.
fn levenshtein_distance(s1: &str, s2: &str) -> usize {
    let len1 = s1.len();
    let len2 = s2.len();
    let mut matrix = vec![vec![0; len2 + 1]; len1 + 1];

    for (i, row) in matrix.iter_mut().enumerate().take(len1 + 1) {
        row[0] = i;
    }

    for (j, cell) in matrix[0].iter_mut().enumerate().take(len2 + 1) {
        *cell = j;
    }

    for (i, c1) in s1.chars().enumerate() {
        for (j, c2) in s2.chars().enumerate() {
            let cost = if c1 == c2 { 0 } else { 1 };
            matrix[i + 1][j + 1] = (matrix[i][j + 1] + 1)
                .min(matrix[i + 1][j] + 1)
                .min(matrix[i][j] + cost);
        }
    }

    matrix[len1][len2]
}

impl From<sled::Error> for StorageError {
    fn from(err: sled::Error) -> Self {
        match err {
            sled::Error::Io(io_err) => StorageError::open_error(io_err.to_string()),
            sled::Error::Corruption { .. } => {
                StorageError::read_error("Database corruption detected".to_string())
            }
            sled::Error::ReportableBug(msg) => StorageError::write_error(format!(
                "Internal error: {}. Please report this bug at https://github.com/aviralgarg05/NexumDB/issues",
                msg
            )),
            sled::Error::Unsupported(msg) => {
                StorageError::write_error(format!("Unsupported operation: {}", msg))
            }
            _ => StorageError::write_error(err.to_string()),
        }
    }
}

impl From<serde_json::Error> for StorageError {
    fn from(err: serde_json::Error) -> Self {
        let preview = format!("at line {} column {}", err.line(), err.column());
        StorageError::serialization_error(err.to_string(), preview)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_codes() {
        let err = StorageError::OpenError("test".to_string());
        assert_eq!(err.code(), ErrorCode::NxmStor101);
    }

    #[test]
    fn test_error_code_display() {
        assert_eq!(ErrorCode::NxmStor101.to_string(), "NXM-STOR-101");
        assert_eq!(ErrorCode::NxmStor104.to_string(), "NXM-STOR-104");
    }

    #[test]
    fn test_backward_compatibility() {
        // Test that old-style constructors still work
        let err1 = StorageError::OpenError("test".to_string());
        let err2 = StorageError::WriteError("test".to_string());
        let err3 = StorageError::ReadError("test".to_string());

        assert_eq!(err1.code(), ErrorCode::NxmStor101);
        assert_eq!(err2.code(), ErrorCode::NxmStor102);
        assert_eq!(err3.code(), ErrorCode::NxmStor103);
    }

    #[test]
    fn test_similar_keys() {
        let candidates = vec![
            "users".to_string(),
            "products".to_string(),
            "orders".to_string(),
        ];
        let similar = find_similar_keys("user", &candidates, 2);
        assert!(similar.contains(&"users".to_string()));
    }

    #[test]
    fn test_levenshtein_distance() {
        assert_eq!(levenshtein_distance("kitten", "sitting"), 3);
        assert_eq!(levenshtein_distance("users", "user"), 1);
        assert_eq!(levenshtein_distance("same", "same"), 0);
    }

    #[test]
    fn test_truncate_preview() {
        let short = "short text";
        assert_eq!(truncate_preview(short, 20), "short text");

        let long = "a".repeat(150);
        let truncated = truncate_preview(&long, 100);
        assert!(truncated.contains("... (truncated)"));
        assert!(truncated.len() < 120);
    }
}
