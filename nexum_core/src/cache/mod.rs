//! Result caching module for unchanged scenarios.
//!
//! This module provides file-based caching of query results to avoid
//! re-running queries when the underlying data hasn't changed.
//!
//! # Features
//!
//! - Hash-based cache keys from query + data state
//! - File-based storage in `.nexum/cache/` directory
//! - Automatic cache invalidation on data changes
//! - Support for `--no-cache` flag to force fresh runs
//!
//! # Usage
//!
//! ```rust
//! use nexum_core::cache::ResultCache;
//!
//! let cache = ResultCache::new(".nexum/cache")?;
//! 
//! // Try to get cached result
//! if let Some(result) = cache.get("SELECT * FROM users", &data_hash)? {
//!     return result;
//! }
//!
//! // Execute query and cache result
//! let result = execute_query();
//! cache.put("SELECT * FROM users", &data_hash, &result)?;
//! ```

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// Cached result entry with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry {
    /// The cached query result as JSON
    pub result: String,
    /// Hash of the data state when cached
    pub data_hash: u64,
    /// Timestamp when cached
    pub timestamp: u64,
    /// Original query string
    pub query: String,
}

/// File-based result cache for query execution
pub struct ResultCache {
    cache_dir: PathBuf,
    enabled: bool,
}

/// Per-table hash tracking for efficient cache invalidation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableHashEntry {
    pub table_name: String,
    pub data_hash: u64,
    pub last_updated: u64,
}

impl ResultCache {
    /// Create a new result cache with the specified directory
    pub fn new<P: AsRef<Path>>(cache_dir: P) -> Result<Self> {
        let cache_dir = cache_dir.as_ref().to_path_buf();
        
        // Create cache directory if it doesn't exist
        if !cache_dir.exists() {
            fs::create_dir_all(&cache_dir)?;
        }

        Ok(Self {
            cache_dir,
            enabled: true,
        })
    }

    /// Create a disabled cache (for --no-cache flag)
    pub fn disabled() -> Self {
        Self {
            cache_dir: PathBuf::new(),
            enabled: false,
        }
    }

    /// Check if caching is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Generate cache key from query and data hash
    fn cache_key(&self, query: &str, data_hash: u64) -> String {
        let mut hasher = DefaultHasher::new();
        query.hash(&mut hasher);
        data_hash.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }

    /// Get cache file path for a given key
    fn cache_file_path(&self, key: &str) -> PathBuf {
        self.cache_dir.join(format!("{}.json", key))
    }

    /// Get table hash file path
    fn table_hash_file_path(&self, table_name: &str) -> PathBuf {
        self.cache_dir.join(format!("table_hash_{}.json", table_name))
    }

    /// Get or compute table hash efficiently
    fn get_table_hash(&self, table_name: &str, storage_data: &[(Vec<u8>, Vec<u8>)]) -> u64 {
        if !self.enabled {
            return 0;
        }

        let hash_file = self.table_hash_file_path(table_name);
        
        // Try to read existing hash
        if let Ok(content) = fs::read_to_string(&hash_file) {
            if let Ok(hash_entry) = serde_json::from_str::<TableHashEntry>(&content) {
                // For now, we'll recompute the hash each time to ensure accuracy
                // In a production system, we could optimize this further with timestamps
                let current_hash = self.compute_table_hash(storage_data);
                
                // Update the hash file if it changed
                if current_hash != hash_entry.data_hash {
                    let _ = self.update_table_hash(table_name, current_hash);
                }
                
                return current_hash;
            }
        }
        
        // Compute and store new hash
        let hash = self.compute_table_hash(storage_data);
        let _ = self.update_table_hash(table_name, hash);
        hash
    }

    /// Compute hash for table data
    fn compute_table_hash(&self, storage_data: &[(Vec<u8>, Vec<u8>)]) -> u64 {
        let data_bytes: Vec<u8> = storage_data
            .iter()
            .flat_map(|(k, v)| [k.as_slice(), v.as_slice()].concat())
            .collect();
        calculate_data_hash(&data_bytes)
    }

    /// Update table hash file
    fn update_table_hash(&self, table_name: &str, hash: u64) -> Result<()> {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)?
            .as_secs();

        let hash_entry = TableHashEntry {
            table_name: table_name.to_string(),
            data_hash: hash,
            last_updated: timestamp,
        };

        let content = serde_json::to_string(&hash_entry)?;
        let hash_file = self.table_hash_file_path(table_name);
        fs::write(&hash_file, content)?;
        
        Ok(())
    }

    /// Get cached result if available and valid
    pub fn get(&self, query: &str, data_hash: u64) -> Result<Option<String>> {
        if !self.enabled {
            return Ok(None);
        }

        let key = self.cache_key(query, data_hash);
        let cache_file = self.cache_file_path(&key);

        if !cache_file.exists() {
            return Ok(None);
        }

        // Read and deserialize cache entry
        let content = fs::read_to_string(&cache_file)?;
        let entry: CacheEntry = serde_json::from_str(&content)?;

        // Validate data hash matches
        if entry.data_hash != data_hash {
            // Data has changed, invalidate cache
            let _ = fs::remove_file(&cache_file);
            return Ok(None);
        }

        println!("Cache hit for query: {}", query);
        Ok(Some(entry.result))
    }

    /// Get table data hash efficiently (public method for executor)
    pub fn get_table_data_hash(&self, table_name: &str, storage_data: &[(Vec<u8>, Vec<u8>)]) -> u64 {
        self.get_table_hash(table_name, storage_data)
    }

    /// Store result in cache
    pub fn put(&self, query: &str, data_hash: u64, result: &str) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }

        let key = self.cache_key(query, data_hash);
        let cache_file = self.cache_file_path(&key);

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)?
            .as_secs();

        let entry = CacheEntry {
            result: result.to_string(),
            data_hash,
            timestamp,
            query: query.to_string(),
        };

        let content = serde_json::to_string_pretty(&entry)?;
        fs::write(&cache_file, content)?;

        println!("Cached result for query: {}", query);
        Ok(())
    }

    /// Invalidate all cache entries
    pub fn clear(&self) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }

        if self.cache_dir.exists() {
            for entry in fs::read_dir(&self.cache_dir)? {
                let entry = entry?;
                if entry.path().extension().and_then(|s| s.to_str()) == Some("json") {
                    fs::remove_file(entry.path())?;
                }
            }
            println!("Cache cleared");
        }

        Ok(())
    }

    /// Invalidate cache entries for a specific table
    pub fn invalidate_table(&self, table_name: &str) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }

        if !self.cache_dir.exists() {
            return Ok(());
        }

        let mut removed_count = 0;

        for entry in fs::read_dir(&self.cache_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.extension().and_then(|s| s.to_str()) != Some("json") {
                continue;
            }

            // Read cache entry to check if it references the table
            if let Ok(content) = fs::read_to_string(&path) {
                if let Ok(cache_entry) = serde_json::from_str::<CacheEntry>(&content) {
                    // More precise table name matching to avoid over-invalidation
                    if self.query_references_table(&cache_entry.query, table_name) {
                        fs::remove_file(&path)?;
                        removed_count += 1;
                    }
                }
            }
        }

        if removed_count > 0 {
            println!("Invalidated {} cache entries for table: {}", removed_count, table_name);
        }

        Ok(())
    }

    /// Check if a query references a specific table name with word boundaries
    fn query_references_table(&self, query: &str, table_name: &str) -> bool {
        let query_lower = query.to_lowercase();
        let table_lower = table_name.to_lowercase();
        
        // Look for table name with word boundaries (spaces, commas, parentheses, etc.)
        let word_boundaries = [" ", "\t", "\n", "(", ")", ",", ";"];
        
        // Check if table name appears after FROM, JOIN, UPDATE, INSERT INTO, etc.
        let table_keywords = ["from ", "join ", "update ", "insert into ", "into "];
        
        for keyword in &table_keywords {
            if let Some(pos) = query_lower.find(&format!("{}{}", keyword, table_lower)) {
                let end_pos = pos + keyword.len() + table_lower.len();
                
                // Check if the character after the table name is a word boundary or end of string
                if end_pos >= query_lower.len() || 
                   word_boundaries.iter().any(|&boundary| query_lower[end_pos..].starts_with(boundary)) {
                    return true;
                }
            }
        }
        
        false
    }

    /// Get cache statistics
    pub fn stats(&self) -> Result<CacheStats> {
        if !self.enabled || !self.cache_dir.exists() {
            return Ok(CacheStats::default());
        }

        let mut total_entries = 0;
        let mut total_size = 0u64;
        let mut oldest_timestamp = u64::MAX;
        let mut newest_timestamp = 0u64;

        for entry in fs::read_dir(&self.cache_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.extension().and_then(|s| s.to_str()) != Some("json") {
                continue;
            }

            total_entries += 1;
            total_size += entry.metadata()?.len();

            if let Ok(content) = fs::read_to_string(&path) {
                if let Ok(cache_entry) = serde_json::from_str::<CacheEntry>(&content) {
                    oldest_timestamp = oldest_timestamp.min(cache_entry.timestamp);
                    newest_timestamp = newest_timestamp.max(cache_entry.timestamp);
                }
            }
        }

        Ok(CacheStats {
            total_entries,
            total_size_bytes: total_size,
            oldest_entry_timestamp: if oldest_timestamp == u64::MAX { None } else { Some(oldest_timestamp) },
            newest_entry_timestamp: if newest_timestamp == 0 { None } else { Some(newest_timestamp) },
        })
    }
}

/// Cache statistics
#[derive(Debug, Default, Clone)]
pub struct CacheStats {
    pub total_entries: usize,
    pub total_size_bytes: u64,
    pub oldest_entry_timestamp: Option<u64>,
    pub newest_entry_timestamp: Option<u64>,
}

/// Calculate hash of data state for cache invalidation
pub fn calculate_data_hash(data: &[u8]) -> u64 {
    let mut hasher = DefaultHasher::new();
    data.hash(&mut hasher);
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_cache_basic_operations() {
        let temp_dir = TempDir::new().unwrap();
        let cache = ResultCache::new(temp_dir.path()).unwrap();

        let query = "SELECT * FROM users WHERE age > 25";
        let data_hash = 12345u64;
        let result = r#"[{"id": 1, "name": "Alice"}]"#;

        // Initially no cache
        assert!(cache.get(query, data_hash).unwrap().is_none());

        // Store in cache
        cache.put(query, data_hash, result).unwrap();

        // Should retrieve from cache
        let cached = cache.get(query, data_hash).unwrap();
        assert_eq!(cached, Some(result.to_string()));
    }

    #[test]
    fn test_cache_invalidation_on_data_change() {
        let temp_dir = TempDir::new().unwrap();
        let cache = ResultCache::new(temp_dir.path()).unwrap();

        let query = "SELECT * FROM users";
        let data_hash_1 = 12345u64;
        let data_hash_2 = 67890u64;
        let result = r#"[{"id": 1}]"#;

        // Cache with first data hash
        cache.put(query, data_hash_1, result).unwrap();

        // Should not retrieve with different data hash
        assert!(cache.get(query, data_hash_2).unwrap().is_none());
    }

    #[test]
    fn test_disabled_cache() {
        let cache = ResultCache::disabled();
        assert!(!cache.is_enabled());

        let query = "SELECT * FROM test";
        let data_hash = 123u64;
        let result = "test";

        // Operations should succeed but do nothing
        cache.put(query, data_hash, result).unwrap();
        assert!(cache.get(query, data_hash).unwrap().is_none());
    }

    #[test]
    fn test_cache_clear() {
        let temp_dir = TempDir::new().unwrap();
        let cache = ResultCache::new(temp_dir.path()).unwrap();

        // Add multiple entries
        cache.put("SELECT 1", 111, "result1").unwrap();
        cache.put("SELECT 2", 222, "result2").unwrap();

        // Clear cache
        cache.clear().unwrap();

        // All entries should be gone
        assert!(cache.get("SELECT 1", 111).unwrap().is_none());
        assert!(cache.get("SELECT 2", 222).unwrap().is_none());
    }

    #[test]
    fn test_table_invalidation() {
        let temp_dir = TempDir::new().unwrap();
        let cache = ResultCache::new(temp_dir.path()).unwrap();

        cache.put("SELECT * FROM users", 111, "result1").unwrap();
        cache.put("SELECT * FROM orders", 222, "result2").unwrap();
        cache.put("SELECT * FROM user_logs", 333, "result3").unwrap(); // Should not be invalidated

        // Invalidate users table
        cache.invalidate_table("users").unwrap();

        // Users query should be invalidated
        assert!(cache.get("SELECT * FROM users", 111).unwrap().is_none());
        
        // Orders query should still be cached
        assert!(cache.get("SELECT * FROM orders", 222).unwrap().is_some());
        
        // user_logs should still be cached (precise matching)
        assert!(cache.get("SELECT * FROM user_logs", 333).unwrap().is_some());
    }

    #[test]
    fn test_precise_table_matching() {
        let temp_dir = TempDir::new().unwrap();
        let cache = ResultCache::new(temp_dir.path()).unwrap();

        // Test that "user" table invalidation doesn't affect "users" table
        assert!(!cache.query_references_table("SELECT * FROM users", "user"));
        assert!(cache.query_references_table("SELECT * FROM user", "user"));
        assert!(cache.query_references_table("SELECT * FROM user WHERE id = 1", "user"));
        assert!(cache.query_references_table("INSERT INTO user VALUES (1)", "user"));
        assert!(!cache.query_references_table("SELECT * FROM power_user", "user"));
    }

    #[test]
    fn test_cache_stats() {
        let temp_dir = TempDir::new().unwrap();
        let cache = ResultCache::new(temp_dir.path()).unwrap();

        // Empty cache
        let stats = cache.stats().unwrap();
        assert_eq!(stats.total_entries, 0);

        // Add entries
        cache.put("SELECT 1", 111, "result1").unwrap();
        cache.put("SELECT 2", 222, "result2").unwrap();

        let stats = cache.stats().unwrap();
        assert_eq!(stats.total_entries, 2);
        assert!(stats.total_size_bytes > 0);
    }
}
