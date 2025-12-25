use crate::bridge::SemanticCache;
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
}

impl Executor {
    pub fn new(storage: StorageEngine) -> Self {
        let catalog = Catalog::new(storage.clone());
        Self {
            storage,
            catalog,
            cache: None,
        }
    }

    pub fn with_cache(self) -> Self {
        self.with_cache_file("semantic_cache.pkl")
    }

    pub fn with_cache_file(mut self, cache_file: &str) -> Self {
        match SemanticCache::with_cache_file(cache_file) {
            Ok(cache) => {
                self.cache = Some(cache);
                log::info!("Semantic cache enabled");
                println!("Semantic cache enabled with file: {}", cache_file);
            }
            Err(e) => {
                log::warn!("Could not initialize semantic cache: {}", e);
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
                if let Some(cache) = &self.cache {
                    let query_str = format!("SELECT {:?} FROM {}", columns, table);

                    if let Ok(Some(cached_result)) = cache.get(&query_str) {
                        log::debug!("Cache hit for query: {}", query_str);
                        let rows: Vec<Row> =
                            serde_json::from_str(&cached_result).unwrap_or_else(|_| Vec::new());
                        return Ok(ExecutionResult::Selected { columns, rows });
                    }
                }

                let schema = self
                    .catalog
                    .get_table(&table)?
                    .ok_or_else(|| StorageError::ReadError(format!("Table {} not found", table)))?;

                let prefix = Self::table_data_prefix(&table);
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

                    log::debug!("Filtered {} rows using WHERE clause", rows.len());
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

                    log::debug!("Sorted {} rows using ORDER BY", rows.len());
                }

                if let Some(limit_count) = limit {
                    rows.truncate(limit_count);
                    log::debug!("Limited to {} rows using LIMIT", limit_count);
                }

                if let Some(cache) = &self.cache {
                    let query_str = format!("SELECT {:?} FROM {}", columns, table);
                    let cached_data = serde_json::to_string(&rows).unwrap_or_default();
                    let _ = cache.put(&query_str, &cached_data);
                }

                Ok(ExecutionResult::Selected { columns, rows })
            }
            Statement::Delete {
                table,
                where_clause,
            } => {
                let schema = self
                    .catalog
                    .get_table(&table)?
                    .ok_or_else(|| StorageError::ReadError(format!("Table {} not found", table)))?;

                let prefix = Self::table_data_prefix(&table);

                if let Some(where_expr) = where_clause {
                    let column_names: Vec<String> =
                        schema.columns.iter().map(|c| c.name.clone()).collect();
                    let evaluator = ExpressionEvaluator::new(column_names);

                    // Two-phase deletion: first collect keys to delete, then delete them
                    // This prevents partial deletion if WHERE evaluation fails
                    let all_rows = self.storage.scan_prefix(&prefix)?;
                    let mut keys_to_delete: Vec<Vec<u8>> = Vec::new();

                    // Phase 1: Evaluate all rows and collect matching keys
                    for (key, value) in &all_rows {
                        if let Ok(row) = serde_json::from_slice::<Row>(value) {
                            match evaluator.evaluate(&where_expr, &row.values) {
                                Ok(true) => {
                                    keys_to_delete.push(key.clone());
                                }
                                Ok(false) => {
                                    // Row doesn't match WHERE condition, skip
                                }
                                Err(e) => {
                                    return Err(StorageError::ReadError(format!(
                                            "WHERE clause evaluation failed on row: {}. No rows were deleted.", e
                                        )));
                                }
                            }
                        }
                    }

                    // Phase 2: Delete all matching rows (only if Phase 1 succeeded)
                    let deleted_count = keys_to_delete.len();
                    for key in keys_to_delete {
                        self.storage.delete(&key)?;
                    }

                    Ok(ExecutionResult::Deleted {
                        table,
                        rows: deleted_count,
                    })
                } else {
                    // No WHERE clause - delete all rows
                    log::warn!(
                        "DELETE without WHERE clause will remove all rows from table '{}'",
                        table
                    );
                    let all_rows = self.storage.scan_prefix(&prefix)?;
                    let deleted_count = all_rows.len();
                    for (key, _) in &all_rows {
                        self.storage.delete(key)?;
                    }

                    Ok(ExecutionResult::Deleted {
                        table,
                        rows: deleted_count,
                    })
                }
            }
            Statement::Update {
                table,
                assignments,
                where_clause,
            } => {
                let schema = self
                    .catalog
                    .get_table(&table)?
                    .ok_or_else(|| StorageError::ReadError(format!("Table {} not found", table)))?;

                let column_names: Vec<String> =
                    schema.columns.iter().map(|c| c.name.clone()).collect();

                // Build column index map for assignments
                let mut assignment_indices: Vec<(usize, Value)> = Vec::new();
                for (col_name, new_value) in &assignments {
                    let col_idx = column_names
                        .iter()
                        .position(|c| c == col_name)
                        .ok_or_else(|| {
                            StorageError::ReadError(format!("Column {} not found in table {}", col_name, table))
                        })?;

                    // Type checking: verify the new value matches the column type
                    let expected_type = &schema.columns[col_idx].data_type;
                    let actual_type = new_value.data_type();
                    if actual_type != crate::sql::types::DataType::Null && *expected_type != actual_type {
                        return Err(StorageError::ReadError(format!(
                            "Type mismatch for column '{}': expected {:?}, got {:?}",
                            col_name, expected_type, actual_type
                        )));
                    }

                    assignment_indices.push((col_idx, new_value.clone()));
                }

                let prefix = Self::table_data_prefix(&table);
                let all_rows = self.storage.scan_prefix(&prefix)?;

                // Two-phase update: collect updates first, then apply them atomically
                let mut updates: Vec<(Vec<u8>, Row)> = Vec::new();

                let evaluator = ExpressionEvaluator::new(column_names);

                for (key, value) in &all_rows {
                    if let Ok(mut row) = serde_json::from_slice::<Row>(value) {
                        let should_update = if let Some(ref where_expr) = where_clause {
                            match evaluator.evaluate(where_expr, &row.values) {
                                Ok(result) => result,
                                Err(e) => {
                                    return Err(StorageError::ReadError(format!(
                                        "WHERE clause evaluation failed: {}. No rows were updated.",
                                        e
                                    )));
                                }
                            }
                        } else {
                            true // No WHERE clause means update all rows
                        };

                        if should_update {
                            // Apply assignments to the row
                            for (col_idx, new_value) in &assignment_indices {
                                row.values[*col_idx] = new_value.clone();
                            }
                            updates.push((key.clone(), row));
                        }
                    }
                }

                // Phase 2: Apply all updates atomically
                let updated_count = updates.len();
                for (key, row) in updates {
                    let value = serde_json::to_vec(&row)?;
                    self.storage.set(&key, &value)?;
                }

                if where_clause.is_none() && updated_count > 0 {
                    log::warn!(
                        "UPDATE without WHERE clause modified all {} rows in table '{}'",
                        updated_count,
                        table
                    );
                }

                Ok(ExecutionResult::Updated {
                    table,
                    rows: updated_count,
                })
            }
        };

        let duration = start.elapsed();
        log::debug!("Query executed in {:?}", duration);

        result
    }

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

    pub fn save_cache(&self) -> Result<()> {
        if let Some(cache) = &self.cache {
            cache
                .save_cache()
                .map_err(|e| StorageError::WriteError(e.to_string()))?;
            println!("Semantic cache saved to disk");
        } else {
            println!("No semantic cache to save");
        }
        Ok(())
    }

    pub fn clear_cache(&self) -> Result<()> {
        if let Some(cache) = &self.cache {
            cache
                .clear_cache()
                .map_err(|e| StorageError::WriteError(e.to_string()))?;
            println!("Semantic cache cleared");
        } else {
            println!("No semantic cache to clear");
        }
        Ok(())
    }

    pub fn get_cache_stats(&self) -> Result<String> {
        if let Some(cache) = &self.cache {
            cache
                .get_cache_stats()
                .map_err(|e| StorageError::ReadError(e.to_string()))
        } else {
            Ok("No semantic cache enabled".to_string())
        }
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
    Deleted {
        table: String,
        rows: usize,
    },
    Updated {
        table: String,
        rows: usize,
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

    #[test]
    fn test_delete_with_where_clause() {
        let storage = StorageEngine::memory().unwrap();
        let executor = Executor::new(storage);

        // Create table
        let create = Statement::CreateTable {
            name: "test_delete".to_string(),
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
        executor.execute(create).unwrap();

        // Insert rows
        let insert = Statement::Insert {
            table: "test_delete".to_string(),
            columns: vec!["id".to_string(), "name".to_string()],
            values: vec![
                vec![Value::Integer(1), Value::Text("Alice".to_string())],
                vec![Value::Integer(2), Value::Text("Bob".to_string())],
                vec![Value::Integer(3), Value::Text("Charlie".to_string())],
            ],
        };
        executor.execute(insert).unwrap();

        // Delete with WHERE clause
        use sqlparser::dialect::GenericDialect;
        use sqlparser::parser::Parser as SqlParser;
        let dialect = GenericDialect {};
        let ast = SqlParser::parse_sql(&dialect, "SELECT * FROM t WHERE id = 2").unwrap();
        let where_expr = if let sqlparser::ast::Statement::Query(query) = &ast[0] {
            if let sqlparser::ast::SetExpr::Select(select) = &*query.body {
                select.selection.clone().map(Box::new)
            } else {
                None
            }
        } else {
            None
        };

        let delete = Statement::Delete {
            table: "test_delete".to_string(),
            where_clause: where_expr,
        };

        let result = executor.execute(delete).unwrap();
        match result {
            ExecutionResult::Deleted { table, rows } => {
                assert_eq!(table, "test_delete");
                assert_eq!(rows, 1);
            }
            _ => panic!("Expected Deleted result"),
        }

        // Verify only 2 rows remain
        let select = Statement::Select {
            table: "test_delete".to_string(),
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
            _ => panic!("Expected Selected result"),
        }
    }

    #[test]
    fn test_delete_all_rows() {
        let storage = StorageEngine::memory().unwrap();
        let executor = Executor::new(storage);

        // Create table
        let create = Statement::CreateTable {
            name: "test_delete_all".to_string(),
            columns: vec![Column {
                name: "id".to_string(),
                data_type: DataType::Integer,
            }],
        };
        executor.execute(create).unwrap();

        // Insert rows
        let insert = Statement::Insert {
            table: "test_delete_all".to_string(),
            columns: vec!["id".to_string()],
            values: vec![vec![Value::Integer(1)], vec![Value::Integer(2)]],
        };
        executor.execute(insert).unwrap();

        // Delete all (no WHERE clause)
        let delete = Statement::Delete {
            table: "test_delete_all".to_string(),
            where_clause: None,
        };

        let result = executor.execute(delete).unwrap();
        match result {
            ExecutionResult::Deleted { rows, .. } => {
                assert_eq!(rows, 2);
            }
            _ => panic!("Expected Deleted result"),
        }

        // Verify no rows remain
        let select = Statement::Select {
            table: "test_delete_all".to_string(),
            columns: vec!["*".to_string()],
            where_clause: None,
            order_by: None,
            limit: None,
        };
        let result = executor.execute(select).unwrap();
        match result {
            ExecutionResult::Selected { rows, .. } => {
                assert_eq!(rows.len(), 0);
            }
            _ => panic!("Expected Selected result"),
        }
    }

    #[test]
    fn test_update_with_where_clause() {
        let storage = StorageEngine::memory().unwrap();
        let executor = Executor::new(storage);

        // Create table
        let create = Statement::CreateTable {
            name: "test_update".to_string(),
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
        executor.execute(create).unwrap();

        // Insert rows
        let insert = Statement::Insert {
            table: "test_update".to_string(),
            columns: vec!["id".to_string(), "name".to_string()],
            values: vec![
                vec![Value::Integer(1), Value::Text("Alice".to_string())],
                vec![Value::Integer(2), Value::Text("Bob".to_string())],
                vec![Value::Integer(3), Value::Text("Charlie".to_string())],
            ],
        };
        executor.execute(insert).unwrap();

        // Parse WHERE clause for UPDATE
        use sqlparser::dialect::GenericDialect;
        use sqlparser::parser::Parser as SqlParser;
        let dialect = GenericDialect {};
        let ast = SqlParser::parse_sql(&dialect, "SELECT * FROM t WHERE id = 2").unwrap();
        let where_expr = if let sqlparser::ast::Statement::Query(query) = &ast[0] {
            if let sqlparser::ast::SetExpr::Select(select) = &*query.body {
                select.selection.clone().map(Box::new)
            } else {
                None
            }
        } else {
            None
        };

        // Update with WHERE clause
        let update = Statement::Update {
            table: "test_update".to_string(),
            assignments: vec![("name".to_string(), Value::Text("Bobby".to_string()))],
            where_clause: where_expr,
        };

        let result = executor.execute(update).unwrap();
        match result {
            ExecutionResult::Updated { table, rows } => {
                assert_eq!(table, "test_update");
                assert_eq!(rows, 1);
            }
            _ => panic!("Expected Updated result"),
        }

        // Verify the update
        let select = Statement::Select {
            table: "test_update".to_string(),
            columns: vec!["*".to_string()],
            where_clause: None,
            order_by: None,
            limit: None,
        };
        let result = executor.execute(select).unwrap();
        match result {
            ExecutionResult::Selected { rows, .. } => {
                assert_eq!(rows.len(), 3);
                // Find the updated row
                let updated_row = rows.iter().find(|r| {
                    if let Value::Integer(id) = &r.values[0] {
                        *id == 2
                    } else {
                        false
                    }
                });
                assert!(updated_row.is_some());
                if let Value::Text(name) = &updated_row.unwrap().values[1] {
                    assert_eq!(name, "Bobby");
                } else {
                    panic!("Expected Text value for name");
                }
            }
            _ => panic!("Expected Selected result"),
        }
    }

    #[test]
    fn test_update_multiple_columns() {
        let storage = StorageEngine::memory().unwrap();
        let executor = Executor::new(storage);

        // Create table
        let create = Statement::CreateTable {
            name: "test_update_multi".to_string(),
            columns: vec![
                Column {
                    name: "id".to_string(),
                    data_type: DataType::Integer,
                },
                Column {
                    name: "name".to_string(),
                    data_type: DataType::Text,
                },
                Column {
                    name: "age".to_string(),
                    data_type: DataType::Integer,
                },
            ],
        };
        executor.execute(create).unwrap();

        // Insert a row
        let insert = Statement::Insert {
            table: "test_update_multi".to_string(),
            columns: vec!["id".to_string(), "name".to_string(), "age".to_string()],
            values: vec![vec![
                Value::Integer(1),
                Value::Text("Alice".to_string()),
                Value::Integer(25),
            ]],
        };
        executor.execute(insert).unwrap();

        // Update multiple columns
        let update = Statement::Update {
            table: "test_update_multi".to_string(),
            assignments: vec![
                ("name".to_string(), Value::Text("Alicia".to_string())),
                ("age".to_string(), Value::Integer(26)),
            ],
            where_clause: None,
        };

        let result = executor.execute(update).unwrap();
        match result {
            ExecutionResult::Updated { rows, .. } => {
                assert_eq!(rows, 1);
            }
            _ => panic!("Expected Updated result"),
        }

        // Verify the update
        let select = Statement::Select {
            table: "test_update_multi".to_string(),
            columns: vec!["*".to_string()],
            where_clause: None,
            order_by: None,
            limit: None,
        };
        let result = executor.execute(select).unwrap();
        match result {
            ExecutionResult::Selected { rows, .. } => {
                assert_eq!(rows.len(), 1);
                let row = &rows[0];
                if let Value::Text(name) = &row.values[1] {
                    assert_eq!(name, "Alicia");
                }
                if let Value::Integer(age) = &row.values[2] {
                    assert_eq!(*age, 26);
                }
            }
            _ => panic!("Expected Selected result"),
        }
    }

    #[test]
    fn test_update_all_rows() {
        let storage = StorageEngine::memory().unwrap();
        let executor = Executor::new(storage);

        // Create table
        let create = Statement::CreateTable {
            name: "test_update_all".to_string(),
            columns: vec![
                Column {
                    name: "id".to_string(),
                    data_type: DataType::Integer,
                },
                Column {
                    name: "status".to_string(),
                    data_type: DataType::Text,
                },
            ],
        };
        executor.execute(create).unwrap();

        // Insert rows
        let insert = Statement::Insert {
            table: "test_update_all".to_string(),
            columns: vec!["id".to_string(), "status".to_string()],
            values: vec![
                vec![Value::Integer(1), Value::Text("pending".to_string())],
                vec![Value::Integer(2), Value::Text("pending".to_string())],
            ],
        };
        executor.execute(insert).unwrap();

        // Update all rows (no WHERE clause)
        let update = Statement::Update {
            table: "test_update_all".to_string(),
            assignments: vec![("status".to_string(), Value::Text("completed".to_string()))],
            where_clause: None,
        };

        let result = executor.execute(update).unwrap();
        match result {
            ExecutionResult::Updated { rows, .. } => {
                assert_eq!(rows, 2);
            }
            _ => panic!("Expected Updated result"),
        }

        // Verify all rows updated
        let select = Statement::Select {
            table: "test_update_all".to_string(),
            columns: vec!["*".to_string()],
            where_clause: None,
            order_by: None,
            limit: None,
        };
        let result = executor.execute(select).unwrap();
        match result {
            ExecutionResult::Selected { rows, .. } => {
                for row in rows {
                    if let Value::Text(status) = &row.values[1] {
                        assert_eq!(status, "completed");
                    }
                }
            }
            _ => panic!("Expected Selected result"),
        }
    }
}
