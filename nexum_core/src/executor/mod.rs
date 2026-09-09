use crate::bridge::SemanticCache;
use crate::catalog::Catalog;
use crate::sql::types::{Column, DataType, SelectItem, Statement, Value};
use crate::storage::{find_similar_keys, Result, StorageEngine, StorageError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard};
use std::time::Instant;
use std::time::{SystemTime, UNIX_EPOCH};

pub mod filter;
use filter::ExpressionEvaluator;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Row {
    pub values: Vec<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TxWalFile {
    tx_id: u64,
    #[serde(default)]
    committed: bool,
    #[serde(default)]
    snapshot: Option<Vec<(Vec<u8>, Vec<u8>)>>,
}

#[derive(Debug, Clone)]
struct TransactionState {
    tx_id: u64,
    snapshot: Vec<(Vec<u8>, Vec<u8>)>,
    write_count: usize,
}

pub struct Executor {
    storage: StorageEngine,
    catalog: Catalog,
    cache: Option<SemanticCache>,
    tx_state: Mutex<Option<TransactionState>>,
    tx_counter: AtomicU64,
}

impl Executor {
    pub fn new(storage: StorageEngine) -> Self {
        let catalog = Catalog::new(storage.clone());
        let executor = Self {
            storage,
            catalog,
            cache: None,
            tx_state: Mutex::new(None),
            tx_counter: AtomicU64::new(
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|d| d.as_millis() as u64)
                    .unwrap_or(1),
            ),
        };

        if let Err(e) = executor.recover_pending_transaction() {
            log::warn!("Transaction recovery failed: {}", e);
        }

        executor
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
            Statement::BeginTransaction => self.begin_transaction(),
            Statement::CommitTransaction => self.commit_transaction(),
            Statement::RollbackTransaction => self.rollback_transaction(),
            Statement::CreateTable { name, columns } => {
                self.catalog.create_table(&name, columns)?;
                self.invalidate_cache()?;
                self.record_transaction_write(0)?;
                Ok(ExecutionResult::Created { table: name })
            }
            Statement::Insert {
                table,
                columns,
                values,
            } => {
                let schema = self.catalog.get_table(&table)?.ok_or_else(|| {
                    let similar = self.catalog.list_tables().unwrap_or_default();
                    StorageError::key_not_found(
                        &table,
                        "INSERT query execution",
                        find_similar_keys(&table, &similar, 2),
                    )
                })?;
                let prepared_rows = Self::prepare_insert_rows(&schema, &table, &columns, &values)?;

                for row in prepared_rows {
                    let key = self.generate_row_key(&table);
                    let value = serde_json::to_vec(&row)?;
                    self.storage.set(&key, &value)?;
                }

                self.invalidate_cache()?;
                self.record_transaction_write(values.len())?;

                Ok(ExecutionResult::Inserted {
                    table,
                    rows: values.len(),
                })
            }
            Statement::Select {
                table,
                projection,
                where_clause,
                order_by,
                limit,
            } => {
                let schema = self.catalog.get_table(&table)?.ok_or_else(|| {
                    let similar = self.catalog.list_tables().unwrap_or_default();
                    StorageError::key_not_found(
                        &table,
                        "SELECT query execution",
                        find_similar_keys(&table, &similar, 2),
                    )
                })?;

                let (projection_indices, output_columns) =
                    Self::build_projection(&schema, &projection)?;

                let cache_key = Self::build_cache_key(
                    &table,
                    &projection,
                    where_clause.as_deref(),
                    &order_by,
                    limit,
                );

                if let Some(cache) = &self.cache {
                    if let Ok(Some(cached_result)) = cache.get(&cache_key) {
                        log::debug!("Cache hit for query: {}", cache_key);
                        let rows: Vec<Row> =
                            serde_json::from_str(&cached_result).unwrap_or_else(|_| Vec::new());
                        return Ok(ExecutionResult::Selected {
                            columns: output_columns,
                            rows,
                        });
                    }
                }

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

                if let Some(order_clauses) = &order_by {
                    let column_names: Vec<String> =
                        schema.columns.iter().map(|c| c.name.clone()).collect();

                    for order_clause in order_clauses.iter().rev() {
                        let col_idx = column_names
                            .iter()
                            .position(|c| c == &order_clause.column)
                            .ok_or_else(|| {
                                StorageError::ReadError(format!(
                                    "Column {} not found in table {}",
                                    order_clause.column, table
                                ))
                            })?;

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

                    log::debug!("Sorted {} rows using ORDER BY", rows.len());
                }

                if let Some(limit_count) = limit {
                    rows.truncate(limit_count);
                    log::debug!("Limited to {} rows using LIMIT", limit_count);
                }

                let projected_rows: Vec<Row> = rows
                    .into_iter()
                    .map(|row| Row {
                        values: projection_indices
                            .iter()
                            .map(|idx| row.values[*idx].clone())
                            .collect(),
                    })
                    .collect();

                if let Some(cache) = &self.cache {
                    let cached_data = serde_json::to_string(&projected_rows).unwrap_or_default();
                    let _ = cache.put(&cache_key, &cached_data);
                }

                Ok(ExecutionResult::Selected {
                    columns: output_columns,
                    rows: projected_rows,
                })
            }
            Statement::ShowTables => {
                let tables = self.catalog.list_tables()?;
                Ok(ExecutionResult::TableList { tables })
            }
            Statement::DescribeTable { name } => {
                let schema = self.catalog.get_table(&name)?.ok_or_else(|| {
                    let similar = self.catalog.list_tables().unwrap_or_default();
                    StorageError::key_not_found(
                        &name,
                        "DESCRIBE command",
                        find_similar_keys(&name, &similar, 2),
                    )
                })?;
                Ok(ExecutionResult::TableDescription {
                    table: name,
                    columns: schema.columns,
                })
            }
            Statement::DropTable { name, if_exists } => {
                let schema = self.catalog.get_table(&name)?;
                if schema.is_none() {
                    if if_exists {
                        return Ok(ExecutionResult::Deleted {
                            table: name,
                            rows: 0,
                        });
                    }
                    let similar = self.catalog.list_tables().unwrap_or_default();
                    return Err(StorageError::key_not_found(
                        &name,
                        "DROP TABLE command",
                        find_similar_keys(&name, &similar, 2),
                    ));
                }

                let prefix = Self::table_data_prefix(&name);
                let all_rows = self.storage.scan_prefix(&prefix)?;
                let deleted_count = all_rows.len();
                for (key, _) in &all_rows {
                    self.storage.delete(key)?;
                }

                self.catalog.drop_table(&name)?;
                self.invalidate_cache()?;
                self.record_transaction_write(deleted_count)?;

                Ok(ExecutionResult::Deleted {
                    table: name,
                    rows: deleted_count,
                })
            }
            Statement::Delete {
                table,
                where_clause,
            } => {
                let schema = self.catalog.get_table(&table)?.ok_or_else(|| {
                    let similar = self.catalog.list_tables().unwrap_or_default();
                    StorageError::key_not_found(
                        &table,
                        "DELETE query execution",
                        find_similar_keys(&table, &similar, 2),
                    )
                })?;

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

                    self.invalidate_cache()?;
                    self.record_transaction_write(deleted_count)?;
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

                    self.invalidate_cache()?;
                    self.record_transaction_write(deleted_count)?;
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
                let schema = self.catalog.get_table(&table)?.ok_or_else(|| {
                    let similar = self.catalog.list_tables().unwrap_or_default();
                    StorageError::key_not_found(
                        &table,
                        "UPDATE query execution",
                        find_similar_keys(&table, &similar, 2),
                    )
                })?;

                let column_names: Vec<String> =
                    schema.columns.iter().map(|c| c.name.clone()).collect();

                // Check for duplicate column assignments
                let mut seen_columns: std::collections::HashSet<&str> =
                    std::collections::HashSet::new();
                for (col_name, _) in &assignments {
                    if !seen_columns.insert(col_name.as_str()) {
                        return Err(StorageError::ReadError(format!(
                            "Duplicate column assignment for '{}'",
                            col_name
                        )));
                    }
                }

                // Build column index map for assignments
                let mut assignment_indices: Vec<(usize, Value)> = Vec::new();
                for (col_name, new_value) in &assignments {
                    let col_idx =
                        column_names
                            .iter()
                            .position(|c| c == col_name)
                            .ok_or_else(|| {
                                StorageError::ReadError(format!(
                                    "Column {} not found in table {}",
                                    col_name, table
                                ))
                            })?;

                    let expected_type = &schema.columns[col_idx].data_type;
                    let coerced_value = Self::coerce_value(col_name, expected_type, new_value)?;

                    assignment_indices.push((col_idx, coerced_value));
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
                            // Apply assignments to the row with bounds checking
                            for (col_idx, new_value) in &assignment_indices {
                                if let Some(value) = row.values.get_mut(*col_idx) {
                                    *value = new_value.clone();
                                } else {
                                    return Err(StorageError::ReadError(format!(
                                        "Row data corrupted: column index {} out of bounds (row has {} values)",
                                        col_idx,
                                        row.values.len()
                                    )));
                                }
                            }
                            updates.push((key.clone(), row));
                        }
                    }
                }

                // Phase 2: Apply all updates atomically using batch operation
                let updated_count = updates.len();
                if !updates.is_empty() {
                    let mut batch_operations: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();

                    // Serialize all rows first, fail early if any serialization fails
                    for (key, row) in updates {
                        let value = serde_json::to_vec(&row).map_err(|e| {
                            StorageError::WriteError(format!("Failed to serialize row: {}", e))
                        })?;
                        batch_operations.push((key, value));
                    }

                    // Only apply batch if all serializations succeeded
                    self.storage.batch_set(batch_operations)?;
                }

                if where_clause.is_none() && updated_count > 0 {
                    log::warn!(
                        "UPDATE without WHERE clause modified all {} rows in table '{}'",
                        updated_count,
                        table
                    );
                }

                self.invalidate_cache()?;
                self.record_transaction_write(updated_count)?;
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

    fn lock_tx_state(&self) -> Result<MutexGuard<'_, Option<TransactionState>>> {
        self.tx_state
            .lock()
            .map_err(|_| StorageError::ReadError("Transaction state lock poisoned".to_string()))
    }

    fn begin_transaction(&self) -> Result<ExecutionResult> {
        if self.lock_tx_state()?.is_some() {
            return Err(StorageError::WriteError(
                "A transaction is already active. Commit or rollback before BEGIN.".to_string(),
            ));
        }

        // NOTE: Current isolation is a single-active-transaction model per Executor.
        // Writes still apply directly to storage and are not MVCC-isolated across sessions.
        // TODO(issue#22-followup): This MVP WAL takes a full database snapshot via scan_all().
        // Replace with an incremental undo/redo mutation log for large datasets.
        let snapshot = self.storage.scan_all()?;
        let tx_id = self.tx_counter.fetch_add(1, Ordering::Relaxed);

        self.write_transaction_wal(&TxWalFile {
            tx_id,
            committed: false,
            snapshot: Some(snapshot.clone()),
        })?;

        *self.lock_tx_state()? = Some(TransactionState {
            tx_id,
            snapshot,
            write_count: 0,
        });

        Ok(ExecutionResult::TransactionBegan { tx_id })
    }

    fn commit_transaction(&self) -> Result<ExecutionResult> {
        let (tx_id, writes) = {
            let state_ref = self.lock_tx_state()?;
            let state = state_ref.as_ref().ok_or_else(|| {
                StorageError::ReadError("No active transaction. Use BEGIN first.".to_string())
            })?;
            (state.tx_id, state.write_count)
        };

        self.write_transaction_wal(&TxWalFile {
            tx_id,
            committed: true,
            snapshot: None,
        })?;
        self.clear_transaction_wal()?;
        *self.lock_tx_state()? = None;

        Ok(ExecutionResult::TransactionCommitted { tx_id, writes })
    }

    fn rollback_transaction(&self) -> Result<ExecutionResult> {
        let (tx_id, snapshot) = {
            let state_ref = self.lock_tx_state()?;
            let state = state_ref.as_ref().ok_or_else(|| {
                StorageError::ReadError("No active transaction. Use BEGIN first.".to_string())
            })?;
            (state.tx_id, state.snapshot.clone())
        };

        self.restore_snapshot(&snapshot)?;
        self.clear_transaction_wal()?;
        self.invalidate_cache()?;
        *self.lock_tx_state()? = None;

        Ok(ExecutionResult::TransactionRolledBack { tx_id })
    }

    fn record_transaction_write(&self, rows: usize) -> Result<()> {
        if rows == 0 {
            return Ok(());
        }

        if let Some(active) = self.lock_tx_state()?.as_mut() {
            active.write_count += rows;
        }
        Ok(())
    }

    fn recover_pending_transaction(&self) -> Result<()> {
        let Some(wal) = self.read_transaction_wal()? else {
            return Ok(());
        };

        if wal.committed {
            self.clear_transaction_wal()?;
            return Ok(());
        }

        let snapshot = wal.snapshot.ok_or_else(|| {
            StorageError::ReadError("WAL recovery failed: missing snapshot for transaction".into())
        })?;

        self.restore_snapshot(&snapshot)?;
        self.clear_transaction_wal()?;
        self.invalidate_cache()?;
        log::warn!(
            "Recovered and rolled back uncommitted transaction {} from WAL",
            wal.tx_id
        );
        Ok(())
    }

    fn read_transaction_wal(&self) -> Result<Option<TxWalFile>> {
        let Some(path) = self.storage.wal_path() else {
            return Ok(None);
        };

        if !path.exists() {
            return Ok(None);
        }

        let bytes = fs::read(path).map_err(|e| StorageError::ReadError(e.to_string()))?;
        let wal: TxWalFile =
            serde_json::from_slice(&bytes).map_err(|e| StorageError::ReadError(e.to_string()))?;
        Ok(Some(wal))
    }

    fn write_transaction_wal(&self, wal: &TxWalFile) -> Result<()> {
        let Some(path) = self.storage.wal_path() else {
            return Ok(());
        };

        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|e| StorageError::WriteError(e.to_string()))?;
        }

        let data = serde_json::to_vec(wal).map_err(|e| StorageError::WriteError(e.to_string()))?;
        let mut file = File::create(path).map_err(|e| StorageError::WriteError(e.to_string()))?;
        file.write_all(&data)
            .map_err(|e| StorageError::WriteError(e.to_string()))?;
        file.sync_all()
            .map_err(|e| StorageError::WriteError(e.to_string()))?;
        Ok(())
    }

    fn clear_transaction_wal(&self) -> Result<()> {
        let Some(path) = self.storage.wal_path() else {
            return Ok(());
        };

        match fs::remove_file(path) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(StorageError::WriteError(e.to_string())),
        }
    }

    fn restore_snapshot(&self, snapshot: &[(Vec<u8>, Vec<u8>)]) -> Result<()> {
        let current = self.storage.scan_all()?;
        let current_map: HashMap<Vec<u8>, Vec<u8>> = current.into_iter().collect();
        let snapshot_map: HashMap<Vec<u8>, Vec<u8>> = snapshot
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();

        let keys_to_delete: Vec<Vec<u8>> = current_map
            .keys()
            .filter(|key| !snapshot_map.contains_key(*key))
            .cloned()
            .collect();

        let entries_to_set: Vec<(Vec<u8>, Vec<u8>)> = snapshot_map
            .iter()
            .filter(|(key, value)| match current_map.get(*key) {
                Some(current_value) => current_value != *value,
                None => true,
            })
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect();

        if !keys_to_delete.is_empty() {
            self.storage.delete_keys(&keys_to_delete)?;
        }

        if !entries_to_set.is_empty() {
            self.storage.batch_set(entries_to_set)?;
        }

        self.storage.flush()?;
        Ok(())
    }

    fn invalidate_cache(&self) -> Result<()> {
        if let Some(cache) = &self.cache {
            cache
                .clear_cache()
                .map_err(|e| StorageError::WriteError(e.to_string()))?;
        }
        Ok(())
    }

    fn build_projection(
        schema: &crate::sql::types::TableSchema,
        projection: &[SelectItem],
    ) -> Result<(Vec<usize>, Vec<String>)> {
        if projection.is_empty() {
            return Err(StorageError::ReadError(
                "SELECT projection cannot be empty".to_string(),
            ));
        }

        let mut indices = Vec::new();
        let mut column_names = Vec::new();

        for item in projection {
            match item {
                SelectItem::Wildcard => {
                    for (idx, column) in schema.columns.iter().enumerate() {
                        indices.push(idx);
                        column_names.push(column.name.clone());
                    }
                }
                SelectItem::Column { name, alias } => {
                    let col_idx = schema
                        .columns
                        .iter()
                        .position(|c| c.name == *name)
                        .ok_or_else(|| {
                            StorageError::ReadError(format!(
                                "Column {} not found in table {}",
                                name, schema.name
                            ))
                        })?;

                    indices.push(col_idx);
                    column_names.push(alias.clone().unwrap_or_else(|| name.clone()));
                }
            }
        }

        Ok((indices, column_names))
    }

    fn prepare_insert_rows(
        schema: &crate::sql::types::TableSchema,
        table: &str,
        columns: &[String],
        values: &[Vec<Value>],
    ) -> Result<Vec<Row>> {
        let schema_len = schema.columns.len();

        if columns.is_empty() {
            let mut rows = Vec::with_capacity(values.len());
            for (row_idx, row_values) in values.iter().enumerate() {
                if row_values.len() != schema_len {
                    return Err(StorageError::WriteError(format!(
                        "INSERT row {} has {} values but table {} expects {} columns",
                        row_idx + 1,
                        row_values.len(),
                        table,
                        schema_len
                    )));
                }

                let mut coerced = Vec::with_capacity(schema_len);
                for (idx, value) in row_values.iter().enumerate() {
                    let column = &schema.columns[idx];
                    let coerced_value = Self::coerce_value(&column.name, &column.data_type, value)?;
                    coerced.push(coerced_value);
                }
                rows.push(Row { values: coerced });
            }
            return Ok(rows);
        }

        let mut seen_columns = std::collections::HashSet::new();
        for column in columns {
            if !seen_columns.insert(column.as_str()) {
                return Err(StorageError::WriteError(format!(
                    "Duplicate column '{}' in INSERT statement",
                    column
                )));
            }
        }

        let mut column_indices = Vec::with_capacity(columns.len());
        for column in columns {
            let col_idx = schema
                .columns
                .iter()
                .position(|c| c.name == *column)
                .ok_or_else(|| {
                    StorageError::WriteError(format!(
                        "Column {} not found in table {}",
                        column, table
                    ))
                })?;
            column_indices.push(col_idx);
        }

        let mut rows = Vec::with_capacity(values.len());
        for (row_idx, row_values) in values.iter().enumerate() {
            if row_values.len() != columns.len() {
                return Err(StorageError::WriteError(format!(
                    "INSERT row {} has {} values but {} columns were specified",
                    row_idx + 1,
                    row_values.len(),
                    columns.len()
                )));
            }

            let mut row = vec![Value::Null; schema_len];
            for (value_idx, value) in row_values.iter().enumerate() {
                let col_idx = column_indices[value_idx];
                let column = &schema.columns[col_idx];
                let coerced_value = Self::coerce_value(&column.name, &column.data_type, value)?;
                row[col_idx] = coerced_value;
            }
            rows.push(Row { values: row });
        }

        Ok(rows)
    }

    fn coerce_value(column_name: &str, expected: &DataType, value: &Value) -> Result<Value> {
        if matches!(value, Value::Null) {
            return Ok(Value::Null);
        }

        match expected {
            DataType::Integer => match value {
                Value::Integer(_) => Ok(value.clone()),
                Value::Float(f) => {
                    if f.fract() == 0.0 {
                        Ok(Value::Integer(*f as i64))
                    } else {
                        Err(StorageError::WriteError(format!(
                            "Type mismatch for column '{}': expected Integer, got Float",
                            column_name
                        )))
                    }
                }
                Value::Text(t) => t.parse::<i64>().map(Value::Integer).map_err(|_| {
                    StorageError::WriteError(format!(
                        "Type mismatch for column '{}': expected Integer, got Text",
                        column_name
                    ))
                }),
                Value::Boolean(b) => Ok(Value::Integer(if *b { 1 } else { 0 })),
                Value::Null => Ok(Value::Null),
            },
            DataType::Float => match value {
                Value::Float(_) => Ok(value.clone()),
                Value::Integer(i) => Ok(Value::Float(*i as f64)),
                Value::Text(t) => t.parse::<f64>().map(Value::Float).map_err(|_| {
                    StorageError::WriteError(format!(
                        "Type mismatch for column '{}': expected Float, got Text",
                        column_name
                    ))
                }),
                Value::Boolean(b) => Ok(Value::Float(if *b { 1.0 } else { 0.0 })),
                Value::Null => Ok(Value::Null),
            },
            DataType::Text => Ok(Value::Text(Self::format_value(value))),
            DataType::Boolean => match value {
                Value::Boolean(_) => Ok(value.clone()),
                Value::Integer(i) => match *i {
                    0 => Ok(Value::Boolean(false)),
                    1 => Ok(Value::Boolean(true)),
                    _ => Err(StorageError::WriteError(format!(
                        "Type mismatch for column '{}': expected Boolean, got Integer",
                        column_name
                    ))),
                },
                Value::Float(f) => {
                    if *f == 0.0 {
                        Ok(Value::Boolean(false))
                    } else if *f == 1.0 {
                        Ok(Value::Boolean(true))
                    } else {
                        Err(StorageError::WriteError(format!(
                            "Type mismatch for column '{}': expected Boolean, got Float",
                            column_name
                        )))
                    }
                }
                Value::Text(t) => {
                    let normalized = t.trim().to_lowercase();
                    match normalized.as_str() {
                        "true" | "1" => Ok(Value::Boolean(true)),
                        "false" | "0" => Ok(Value::Boolean(false)),
                        _ => Err(StorageError::WriteError(format!(
                            "Type mismatch for column '{}': expected Boolean, got Text",
                            column_name
                        ))),
                    }
                }
                Value::Null => Ok(Value::Null),
            },
            DataType::Null => Err(StorageError::WriteError(format!(
                "Column '{}' does not accept non-null values",
                column_name
            ))),
        }
    }

    fn format_value(value: &Value) -> String {
        value.to_string()
    }

    fn build_cache_key(
        table: &str,
        projection: &[SelectItem],
        where_clause: Option<&sqlparser::ast::Expr>,
        order_by: &Option<Vec<crate::sql::types::OrderByClause>>,
        limit: Option<usize>,
    ) -> String {
        let projection_str = projection
            .iter()
            .map(|item| match item {
                SelectItem::Wildcard => "*".to_string(),
                SelectItem::Column { name, alias } => alias
                    .as_ref()
                    .map(|a| format!("{} AS {}", name, a))
                    .unwrap_or_else(|| name.clone()),
            })
            .collect::<Vec<_>>()
            .join(", ");

        let mut key = format!("SELECT {} FROM {}", projection_str, table);

        if let Some(expr) = where_clause {
            key.push_str(&format!(" WHERE {}", expr));
        }

        if let Some(order_clauses) = order_by {
            let order_str = order_clauses
                .iter()
                .map(|clause| {
                    if clause.ascending {
                        format!("{} ASC", clause.column)
                    } else {
                        format!("{} DESC", clause.column)
                    }
                })
                .collect::<Vec<_>>()
                .join(", ");
            key.push_str(&format!(" ORDER BY {}", order_str));
        }

        if let Some(limit_count) = limit {
            key.push_str(&format!(" LIMIT {}", limit_count));
        }

        key
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

#[derive(Debug)]
pub enum ExecutionResult {
    TransactionBegan {
        tx_id: u64,
    },
    TransactionCommitted {
        tx_id: u64,
        writes: usize,
    },
    TransactionRolledBack {
        tx_id: u64,
    },
    Created {
        table: String,
    },
    TableList {
        tables: Vec<String>,
    },
    TableDescription {
        table: String,
        columns: Vec<Column>,
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
    use crate::sql::types::{Column, DataType, SelectItem};

    //added a test that checks if dropping a non-existent table without IF EXISTS fails correctly.
    #[test]
    fn test_drop_non_existent_table_fails() {
        let storage = StorageEngine::memory().unwrap();
        let executor = Executor::new(storage);

        let drop = Statement::DropTable {
            name: "imaginary_table".to_string(),
            if_exists: false,
        };
        
        let result = executor.execute(drop);
        assert!(result.is_err(), "Dropping a non-existent table should return an error");
    }

    // second test
    #[test]
    fn test_verify_physical_deletion() {
        // 1. Setup Engine
        let storage = StorageEngine::memory().unwrap();
        let executor = Executor::new(storage.clone());
        let table_name = "cleanup_test";

        // 2. Create and Insert Data
        let create = Statement::CreateTable {
            name: table_name.to_string(),
            columns: vec![Column { name: "id".to_string(), data_type: DataType::Integer }],
        };
        executor.execute(create).unwrap();

        let insert = Statement::Insert {
            table: table_name.to_string(),
            columns: vec!["id".to_string()],
            values: vec![vec![Value::Integer(1)], vec![Value::Integer(2)]],
        };
        executor.execute(insert).unwrap();

        // 3. Verify data exists in raw storage before DROP
        let prefix = format!("data:{}:", table_name).into_bytes();
        let rows_before = storage.scan_prefix(&prefix).unwrap();
        assert_eq!(rows_before.len(), 2, "Data should exist before DROP");

        // 4. Execute DROP
        let drop = Statement::DropTable {
            name: table_name.to_string(),
            if_exists: false,
        };
        executor.execute(drop).unwrap();

        // 5. THE CRITICAL CHECK: Scan raw storage again
        let rows_after = storage.scan_prefix(&prefix).unwrap();
        assert_eq!(rows_after.len(), 0, "PHYSICAL WIPE FAILED: Raw data still exists in storage after DROP!");
    }

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
            projection: vec![SelectItem::Wildcard],
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
            projection: vec![SelectItem::Wildcard],
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
            projection: vec![SelectItem::Wildcard],
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
            projection: vec![SelectItem::Wildcard],
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
            projection: vec![SelectItem::Wildcard],
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
            projection: vec![SelectItem::Wildcard],
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

    #[test]
    fn test_update_type_mismatch_error() {
        let storage = StorageEngine::memory().unwrap();
        let executor = Executor::new(storage);

        // Create table with integer column
        let create = Statement::CreateTable {
            name: "test_type_error".to_string(),
            columns: vec![
                Column {
                    name: "id".to_string(),
                    data_type: DataType::Integer,
                },
                Column {
                    name: "count".to_string(),
                    data_type: DataType::Integer,
                },
            ],
        };
        executor.execute(create).unwrap();

        // Insert a row
        let insert = Statement::Insert {
            table: "test_type_error".to_string(),
            columns: vec!["id".to_string(), "count".to_string()],
            values: vec![vec![Value::Integer(1), Value::Integer(10)]],
        };
        executor.execute(insert).unwrap();

        // Try to update integer column with text value - should fail
        let update = Statement::Update {
            table: "test_type_error".to_string(),
            assignments: vec![("count".to_string(), Value::Text("not a number".to_string()))],
            where_clause: None,
        };

        let result = executor.execute(update);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("Type mismatch"));
    }

    #[test]
    fn test_update_duplicate_column_error() {
        let storage = StorageEngine::memory().unwrap();
        let executor = Executor::new(storage);

        // Create table
        let create = Statement::CreateTable {
            name: "test_dup_col".to_string(),
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

        // Insert a row
        let insert = Statement::Insert {
            table: "test_dup_col".to_string(),
            columns: vec!["id".to_string(), "name".to_string()],
            values: vec![vec![Value::Integer(1), Value::Text("Alice".to_string())]],
        };
        executor.execute(insert).unwrap();

        // Try to update same column twice - should fail
        let update = Statement::Update {
            table: "test_dup_col".to_string(),
            assignments: vec![
                ("name".to_string(), Value::Text("Bob".to_string())),
                ("name".to_string(), Value::Text("Charlie".to_string())),
            ],
            where_clause: None,
        };

        let result = executor.execute(update);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("Duplicate column assignment"));
    }

    #[test]
    fn test_update_unknown_column_error() {
        let storage = StorageEngine::memory().unwrap();
        let executor = Executor::new(storage);

        // Create table
        let create = Statement::CreateTable {
            name: "test_unknown_col".to_string(),
            columns: vec![Column {
                name: "id".to_string(),
                data_type: DataType::Integer,
            }],
        };
        executor.execute(create).unwrap();

        // Insert a row
        let insert = Statement::Insert {
            table: "test_unknown_col".to_string(),
            columns: vec!["id".to_string()],
            values: vec![vec![Value::Integer(1)]],
        };
        executor.execute(insert).unwrap();

        // Try to update non-existent column - should fail
        let update = Statement::Update {
            table: "test_unknown_col".to_string(),
            assignments: vec![("nonexistent".to_string(), Value::Integer(42))],
            where_clause: None,
        };

        let result = executor.execute(update);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("not found"));
    }

    #[test]
    fn test_insert_column_mapping_and_nulls() {
        let storage = StorageEngine::memory().unwrap();
        let executor = Executor::new(storage);

        let create = Statement::CreateTable {
            name: "test_insert_map".to_string(),
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

        let insert = Statement::Insert {
            table: "test_insert_map".to_string(),
            columns: vec!["name".to_string(), "id".to_string()],
            values: vec![vec![Value::Text("Alice".to_string()), Value::Integer(1)]],
        };
        executor.execute(insert).unwrap();

        let select = Statement::Select {
            table: "test_insert_map".to_string(),
            projection: vec![SelectItem::Wildcard],
            where_clause: None,
            order_by: None,
            limit: None,
        };
        let result = executor.execute(select).unwrap();

        match result {
            ExecutionResult::Selected { rows, .. } => {
                assert_eq!(rows.len(), 1);
                assert_eq!(rows[0].values.len(), 3);
                assert_eq!(rows[0].values[0], Value::Integer(1));
                assert_eq!(rows[0].values[1], Value::Text("Alice".to_string()));
                assert_eq!(rows[0].values[2], Value::Null);
            }
            _ => panic!("Expected Selected result"),
        }
    }

    #[test]
    fn test_insert_and_update_coercion() {
        let storage = StorageEngine::memory().unwrap();
        let executor = Executor::new(storage);

        let create = Statement::CreateTable {
            name: "test_coercion".to_string(),
            columns: vec![
                Column {
                    name: "id".to_string(),
                    data_type: DataType::Integer,
                },
                Column {
                    name: "score".to_string(),
                    data_type: DataType::Float,
                },
                Column {
                    name: "active".to_string(),
                    data_type: DataType::Boolean,
                },
                Column {
                    name: "note".to_string(),
                    data_type: DataType::Text,
                },
            ],
        };
        executor.execute(create).unwrap();

        let insert = Statement::Insert {
            table: "test_coercion".to_string(),
            columns: vec![],
            values: vec![vec![
                Value::Text("42".to_string()),
                Value::Integer(7),
                Value::Text("true".to_string()),
                Value::Integer(9),
            ]],
        };
        executor.execute(insert).unwrap();

        let update = Statement::Update {
            table: "test_coercion".to_string(),
            assignments: vec![
                ("score".to_string(), Value::Text("3.5".to_string())),
                ("active".to_string(), Value::Integer(0)),
            ],
            where_clause: None,
        };
        executor.execute(update).unwrap();

        let select = Statement::Select {
            table: "test_coercion".to_string(),
            projection: vec![SelectItem::Wildcard],
            where_clause: None,
            order_by: None,
            limit: None,
        };
        let result = executor.execute(select).unwrap();

        match result {
            ExecutionResult::Selected { rows, .. } => {
                assert_eq!(rows.len(), 1);
                assert_eq!(rows[0].values[0], Value::Integer(42));
                assert_eq!(rows[0].values[1], Value::Float(3.5));
                assert_eq!(rows[0].values[2], Value::Boolean(false));
                assert_eq!(rows[0].values[3], Value::Text("9".to_string()));
            }
            _ => panic!("Expected Selected result"),
        }
    }

    #[test]
    fn test_drop_table_removes_data_and_schema() {
        let storage = StorageEngine::memory().unwrap();
        let executor = Executor::new(storage);

        let create = Statement::CreateTable {
            name: "test_drop".to_string(),
            columns: vec![Column {
                name: "id".to_string(),
                data_type: DataType::Integer,
            }],
        };
        executor.execute(create).unwrap();

        let insert = Statement::Insert {
            table: "test_drop".to_string(),
            columns: vec!["id".to_string()],
            values: vec![vec![Value::Integer(1)]],
        };
        executor.execute(insert).unwrap();

        let drop = Statement::DropTable {
            name: "test_drop".to_string(),
            if_exists: false,
        };
        let result = executor.execute(drop).unwrap();
        match result {
            ExecutionResult::Deleted { rows, .. } => {
                assert_eq!(rows, 1);
            }
            _ => panic!("Expected Deleted result"),
        }

        let show = Statement::ShowTables;
        let result = executor.execute(show).unwrap();
        match result {
            ExecutionResult::TableList { tables } => {
                assert!(tables.is_empty());
            }
            _ => panic!("Expected TableList result"),
        }
    }

    #[test]
    fn test_transaction_rollback_restores_previous_state() {
        let storage = StorageEngine::memory().unwrap();
        let executor = Executor::new(storage);

        executor
            .execute(Statement::CreateTable {
                name: "tx_users".to_string(),
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
            })
            .unwrap();

        executor.execute(Statement::BeginTransaction).unwrap();
        executor
            .execute(Statement::Insert {
                table: "tx_users".to_string(),
                columns: vec!["id".to_string(), "name".to_string()],
                values: vec![vec![Value::Integer(1), Value::Text("Alice".to_string())]],
            })
            .unwrap();

        executor.execute(Statement::RollbackTransaction).unwrap();

        let result = executor
            .execute(Statement::Select {
                table: "tx_users".to_string(),
                projection: vec![SelectItem::Wildcard],
                where_clause: None,
                order_by: None,
                limit: None,
            })
            .unwrap();

        match result {
            ExecutionResult::Selected { rows, .. } => assert!(rows.is_empty()),
            _ => panic!("Expected Selected result"),
        }
    }

    #[test]
    fn test_transaction_commit_persists_state() {
        let storage = StorageEngine::memory().unwrap();
        let executor = Executor::new(storage);

        executor
            .execute(Statement::CreateTable {
                name: "tx_products".to_string(),
                columns: vec![Column {
                    name: "id".to_string(),
                    data_type: DataType::Integer,
                }],
            })
            .unwrap();

        let begin = executor.execute(Statement::BeginTransaction).unwrap();
        match begin {
            ExecutionResult::TransactionBegan { .. } => {}
            _ => panic!("Expected TransactionBegan"),
        }

        executor
            .execute(Statement::Insert {
                table: "tx_products".to_string(),
                columns: vec!["id".to_string()],
                values: vec![vec![Value::Integer(1)]],
            })
            .unwrap();

        let commit = executor.execute(Statement::CommitTransaction).unwrap();
        match commit {
            ExecutionResult::TransactionCommitted { writes, .. } => assert_eq!(writes, 1),
            _ => panic!("Expected TransactionCommitted"),
        }

        let result = executor
            .execute(Statement::Select {
                table: "tx_products".to_string(),
                projection: vec![SelectItem::Wildcard],
                where_clause: None,
                order_by: None,
                limit: None,
            })
            .unwrap();
        match result {
            ExecutionResult::Selected { rows, .. } => assert_eq!(rows.len(), 1),
            _ => panic!("Expected Selected result"),
        }
    }

    #[test]
    fn test_transaction_recovery_from_wal() {
        let temp_dir = tempfile::tempdir().unwrap();
        let db_path = temp_dir.path().join("tx_recovery_db");

        {
            let storage = StorageEngine::new(&db_path).unwrap();
            let executor = Executor::new(storage);
            executor
                .execute(Statement::CreateTable {
                    name: "accounts".to_string(),
                    columns: vec![
                        Column {
                            name: "id".to_string(),
                            data_type: DataType::Integer,
                        },
                        Column {
                            name: "balance".to_string(),
                            data_type: DataType::Integer,
                        },
                    ],
                })
                .unwrap();

            executor.execute(Statement::BeginTransaction).unwrap();
            executor
                .execute(Statement::Insert {
                    table: "accounts".to_string(),
                    columns: vec!["id".to_string(), "balance".to_string()],
                    values: vec![vec![Value::Integer(1), Value::Integer(500)]],
                })
                .unwrap();
            // Simulate crash by dropping without commit/rollback.
        }

        {
            let storage = StorageEngine::new(&db_path).unwrap();
            let executor = Executor::new(storage);
            let result = executor
                .execute(Statement::Select {
                    table: "accounts".to_string(),
                    projection: vec![SelectItem::Wildcard],
                    where_clause: None,
                    order_by: None,
                    limit: None,
                })
                .unwrap();

            match result {
                ExecutionResult::Selected { rows, .. } => assert_eq!(rows.len(), 0),
                _ => panic!("Expected Selected result"),
            }
        }
    }
}
