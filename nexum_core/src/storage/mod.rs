mod engine;
mod error;

pub use engine::StorageEngine;
pub use error::{find_similar_keys, StorageError};
pub type Result<T> = std::result::Result<T, StorageError>;
