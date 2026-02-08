mod engine;
mod error;

pub use engine::StorageEngine;
pub use error::{StorageError, find_similar_keys};
pub type Result<T> = std::result::Result<T, StorageError>;
