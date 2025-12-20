use crate::bridge::SemanticCache;
use crate::cache::ResultCache;
use crate::catalog::Catalog;
use crate::sql::types::{Statement, Value};
use crate::storage::{Result, StorageEngine, StorageError};
use serde::{Deserialize, Serialize};
use std::time::Instant;

pub mod filter;
use filter::ExpressionEvaluator;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Row {
    pub values: Vec<Value>,
}

pub struct Executor {
    storage: StorageEngine,
    catalog: Catalog,
    cache: Option<SemanticCache>,
    result_cache: ResultCache,
}

impl Executor {
    pub fn new(storage: StorageEngine) -> Self {
        let catalog = Catalog::new(storage.clone());
        let result_cache = ResultCache::new(".nexum/cache").unwrap_or_else(|e| {
            println!("Warning: Could not initialize result cache: {}", e);
            ResultCache::disabled()
        });

        Self {
            storage,
            catalog,
            cache: None,
            result_cache,
        }
    }

    pub fn new_with_cache_disabled(storage: StorageEngine) -> Self {
        let catalog = Catalog::new(storage.clone());
        Self {
            storage,
            catalog,
            cache: None,
            result_cache: ResultCache::disabled(),
        }
    }

    pub fn with_cache(mut self) -> Self {
        match SemanticCache::new() {
            Ok(cache) => {
                self.cache = Some(cache);
                println!("Semantic cache enabled");
            }
            Err(e) => {
                println!("Warning: Could not initialize semantic cache: {}", e);
            }
        }
        self
    }

    pub fn execute(&self, statement: Statement) -> Result<ExecutionResult> {
        let start = Instant::now();

        let result = match statement {
            Statement::CreateTable { name, columns } => {
                self.catalog.create_table(&name, columns)?;
                Ok(ExecutionResult::Created { table: name })
            }
            Statement::Insert {
                table,
                columns: _,
                values,
            } => {
                let _schema = self
                    .catalog
                    .get_table(&table)?
                    .ok_or_else(|| StorageError::ReadError(format!("Table {} not found", table)))?;

                for row_values in &values {
                    let row = Row {
                        values: row_values.clone(),
                    };
                    let key = self.generate_row_key(&table);
                    let value = serde_json::to_vec(&row)?;
                    self.storage.set(&key, &value)?;
                }

                // Invalidate caches for this table since data changed
                if let Err(e) = self.result_cache.invalidate_table(&table) {
                    println!(
                        "Warning: Failed to invalidate cache for table {}: {}",
                        table, e
                    );
                    // Continue execution - cache invalidation failure shouldn't block inserts
                }

                Ok(ExecutionResult::Inserted {
                    table,
                    rows: values.len(),
                })
            }
            Statement::Select {
                table,
                columns,
                where_clause,
                order_by,
                limit,
            } => {
                // Generate query string for caching
                let query_str = format!(
                    "SELECT {:?} FROM {} WHERE {:?} ORDER BY {:?} LIMIT {:?}",
                    columns, table, where_clause, order_by, limit
                );

                // Calculate data hash for cache invalidation (optimized)
                let prefix = Self::table_data_prefix(&table);
                let all_data = self.storage.scan_prefix(&prefix)?;
                let data_hash = self.result_cache.get_table_data_hash(&table, &all_data);

                // Try result cache first
                if let Ok(Some(cached_result)) = self.result_cache.get(&query_str, data_hash) {
                    println!("Result cache hit for query: {}", query_str);
                    let rows: Vec<Row> =
                        serde_json::from_str(&cached_result).unwrap_or_else(|_| Vec::new());
                    return Ok(ExecutionResult::Selected { columns, rows });
                }

                // Fall back to semantic cache
                if let Some(cache) = &self.cache {
                    let simple_query_str = format!("SELECT {:?} FROM {}", columns, table);

                    if let Ok(Some(cached_result)) = cache.get(&simple_query_str) {
                        println!("Semantic cache hit for query: {}", simple_query_str);
                        let rows: Vec<Row> =
                            serde_json::from_str(&cached_result).unwrap_or_else(|_| Vec::new());
                        return Ok(ExecutionResult::Selected { columns, rows });
                    }
                }

                let schema = self
                    .catalog
                    .get_table(&table)?
                    .ok_or_else(|| StorageError::ReadError(format!("Table {} not found", table)))?;

                let all_rows = self.storage.scan_prefix(&prefix)?;

                let mut rows: Vec<Row> = all_rows
                    .iter()
                    .filter_map(|(_, v)| serde_json::from_slice::<Row>(v).ok())
                    .collect();

                if let Some(where_expr) = where_clause {
                    let column_names: Vec<String> =
                        schema.columns.iter().map(|c| c.name.clone()).collect();
                    let evaluator = ExpressionEvaluator::new(column_names);

                    rows.retain(|row| {
                        evaluator
                            .evaluate(&where_expr, &row.values)
                            .unwrap_or(false)
                    });

                    println!("Filtered {} rows using WHERE clause", rows.len());
                }

                if let Some(order_clauses) = order_by {
                    let column_names: Vec<String> =
                        schema.columns.iter().map(|c| c.name.clone()).collect();

                    for order_clause in order_clauses.iter().rev() {
                        if let Some(col_idx) =
                            column_names.iter().position(|c| c == &order_clause.column)
                        {
                            rows.sort_by(|a, b| {
                                let ordering = match (&a.values[col_idx], &b.values[col_idx]) {
                                    (Value::Integer(av), Value::Integer(bv)) => av.cmp(bv),
                                    (Value::Float(av), Value::Float(bv)) => {
                                        av.partial_cmp(bv).unwrap_or(std::cmp::Ordering::Equal)
                                    }
                                    (Value::Text(av), Value::Text(bv)) => av.cmp(bv),
                                    (Value::Boolean(av), Value::Boolean(bv)) => av.cmp(bv),
                                    _ => std::cmp::Ordering::Equal,
                                };

                                if order_clause.ascending {
                                    ordering
                                } else {
                                    ordering.reverse()
                                }
                            });
                        }
                    }

                    println!("Sorted {} rows using ORDER BY", rows.len());
                }

                if let Some(limit_count) = limit {
                    rows.truncate(limit_count);
                    println!("Limited to {} rows using LIMIT", limit_count);
                }

                // Cache the result
                match serde_json::to_string(&rows) {
                    Ok(result_json) => {
                        if let Err(e) = self.result_cache.put(&query_str, data_hash, &result_json) {
                            println!("Warning: Failed to cache query result: {}", e);
                        }
                    }
                    Err(e) => {
                        println!(
                            "Warning: Failed to serialize query result for caching: {}",
                            e
                        );
                    }
                }

                // Also cache in semantic cache for backward compatibility
                if let Some(cache) = &self.cache {
                    let simple_query_str = format!("SELECT {:?} FROM {}", columns, table);
                    match serde_json::to_string(&rows) {
                        Ok(cached_data) => {
                            if let Err(e) = cache.put(&simple_query_str, &cached_data) {
                                println!("Warning: Failed to cache in semantic cache: {}", e);
                            }
                        }
                        Err(e) => {
                            println!("Warning: Failed to serialize for semantic cache: {}", e);
                        }
                    }
                }

                Ok(ExecutionResult::Selected { columns, rows })
            }
        };

        let duration = start.elapsed();
        println!("Query executed in {:?}", duration);
        result
    }

    /// Generate a unique key for a row in a table
    fn generate_row_key(&self, table: &str) -> Vec<u8> {
        use std::time::{SystemTime, UNIX_EPOCH};
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();

        let mut key = Self::table_data_prefix(table);
        key.extend_from_slice(&timestamp.to_be_bytes());
        key
    }

    fn table_data_prefix(table: &str) -> Vec<u8> {
        format!("data:{}:", table).into_bytes()
    }

    /// Clear all result caches
    pub fn clear_cache(&self) -> Result<()> {
        self.result_cache
            .clear()
            .map_err(|e| StorageError::WriteError(e.to_string()))
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> Result<crate::cache::CacheStats> {
        self.result_cache
            .stats()
            .map_err(|e| StorageError::ReadError(e.to_string()))
    }

    /// Check if result caching is enabled
    pub fn is_result_cache_enabled(&self) -> bool {
        self.result_cache.is_enabled()
    }
}

impl Clone for StorageEngine {
    fn clone(&self) -> Self {
        StorageEngine::memory().unwrap()
    }
}

#[derive(Debug)]
pub enum ExecutionResult {
    Created {
        table: String,
    },
    Inserted {
        table: String,
        rows: usize,
    },
    Selected {
        columns: Vec<String>,
        rows: Vec<Row>,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sql::types::{Column, DataType};

    #[test]
    fn test_end_to_end_execution() {
        let storage = StorageEngine::memory().unwrap();
        let executor = Executor::new(storage);

        let create = Statement::CreateTable {
            name: "test_table".to_string(),
            columns: vec![
                Column {
                    name: "id".to_string(),
                    data_type: DataType::Integer,
                },
                Column {
                    name: "name".to_string(),
                    data_type: DataType::Text,
                },
            ],
        };

        let result = executor.execute(create).unwrap();
        match result {
            ExecutionResult::Created { table } => assert_eq!(table, "test_table"),
            _ => panic!("Expected Created result"),
        }

        let insert = Statement::Insert {
            table: "test_table".to_string(),
            columns: vec!["id".to_string(), "name".to_string()],
            values: vec![
                vec![Value::Integer(1), Value::Text("Alice".to_string())],
                vec![Value::Integer(2), Value::Text("Bob".to_string())],
            ],
        };

        let result = executor.execute(insert).unwrap();
        match result {
            ExecutionResult::Inserted { rows, .. } => assert_eq!(rows, 2),
            _ => panic!("Expected Inserted result"),
        }

        let select = Statement::Select {
            table: "test_table".to_string(),
            columns: vec!["*".to_string()],
            where_clause: None,
            order_by: None,
            limit: None,
        };
        let result = executor.execute(select).unwrap();

        match result {
            ExecutionResult::Selected { rows, .. } => {
                assert_eq!(rows.len(), 2);
            }
            _ => panic!("Expected selected"),
        }
    }
}
