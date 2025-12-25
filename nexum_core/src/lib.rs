/// Module providing natural-language translation and Python integration.
pub mod bridge;

/// Module managing the database catalog and metadata.
pub mod catalog;

/// Module responsible for executing queries.
pub mod executor;

/// Module for SQL parsing and related utilities.
pub mod sql;

/// Module handling data storage and low-level engine components.
pub mod storage;

/// Re-exports for natural language translator, Python bridge, semantic cache, and query explainer.
pub use bridge::{NLTranslator, PythonBridge, QueryExplainer, SemanticCache};

/// Re-export of the main Catalog used by Nexum.
pub use catalog::Catalog;

/// Re-export of the core Executor responsible for running queries.
pub use executor::Executor;

/// Re-export of the SQL Parser.
pub use sql::parser::Parser;

/// Re-export of the storage engine implementation.
pub use storage::StorageEngine;
