use crate::sql::types::{Column, TableSchema};
use crate::storage::{Result, StorageEngine, StorageError};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CatalogEntry {
    name: String,
    columns: Vec<(String, String)>,
}

pub struct Catalog {
    storage: StorageEngine,
}

impl Catalog {
    const CATALOG_PREFIX: &'static [u8] = b"catalog:";

    pub fn new(storage: StorageEngine) -> Self {
        Self { storage }
    }

    pub fn create_table(&self, name: &str, columns: Vec<Column>) -> Result<()> {
        let key = Self::table_key(name);

        if self.storage.get(&key)?.is_some() {
            return Err(StorageError::LogicalWrite(format!(
                "Table {} already exists",
                name
            )));
        }

        let cols: Vec<(String, String)> = columns
            .iter()
            .map(|c| (c.name.clone(), format!("{:?}", c.data_type)))
            .collect();

        let entry = CatalogEntry {
            name: name.to_string(),
            columns: cols,
        };

        let value = serde_json::to_vec(&entry)?;
        self.storage.set(&key, &value)?;

        Ok(())
    }

    pub fn get_table(&self, name: &str) -> Result<Option<TableSchema>> {
        let key = Self::table_key(name);

        if let Some(data) = self.storage.get(&key)? {
            let entry: CatalogEntry = serde_json::from_slice(&data)?;
            let columns = entry
                .columns
                .iter()
                .map(|(name, dtype)| Column {
                    name: name.clone(),
                    data_type: Self::parse_data_type(dtype),
                })
                .collect();

            Ok(Some(TableSchema {
                name: entry.name,
                columns,
            }))
        } else {
            Ok(None)
        }
    }

    pub fn list_tables(&self) -> Result<Vec<String>> {
        let results = self.storage.scan_prefix(Self::CATALOG_PREFIX)?;
        let tables = results
            .iter()
            .filter_map(|(_, v)| {
                serde_json::from_slice::<CatalogEntry>(v)
                    .ok()
                    .map(|e| e.name)
            })
            .collect();
        Ok(tables)
    }

    pub fn drop_table(&self, name: &str) -> Result<()> {
        let key = Self::table_key(name);
        self.storage.delete(&key)?;
        Ok(())
    }

    fn table_key(name: &str) -> Vec<u8> {
        let mut key = Self::CATALOG_PREFIX.to_vec();
        key.extend_from_slice(name.as_bytes());
        key
    }

    fn parse_data_type(s: &str) -> crate::sql::types::DataType {
        use crate::sql::types::DataType;
        match s {
            "Integer" => DataType::Integer,
            "Float" => DataType::Float,
            "Text" => DataType::Text,
            "Boolean" => DataType::Boolean,
            _ => DataType::Null,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sql::types::DataType;

    #[test]
    fn test_catalog_operations() {
        let storage = StorageEngine::memory().unwrap();
        let catalog = Catalog::new(storage);

        let columns = vec![
            Column {
                name: "id".to_string(),
                data_type: DataType::Integer,
            },
            Column {
                name: "name".to_string(),
                data_type: DataType::Text,
            },
        ];

        catalog.create_table("users", columns).unwrap();

        let schema = catalog.get_table("users").unwrap();
        assert!(schema.is_some());

        let schema = schema.unwrap();
        assert_eq!(schema.name, "users");
        assert_eq!(schema.columns.len(), 2);

        let tables = catalog.list_tables().unwrap();
        assert_eq!(tables.len(), 1);
        assert_eq!(tables[0], "users");
    }
}
