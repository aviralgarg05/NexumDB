pub mod engine; 
mod error;

// Re-exporting StorageEngine as the primary interface
pub use engine::StorageEngine; 
pub use error::{find_similar_keys, StorageError};
pub type Result<T> = std::result::Result<T, StorageError>;