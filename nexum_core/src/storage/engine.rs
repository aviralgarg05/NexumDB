use super::{Result, StorageError};
use sled::Db;
use std::path::Path;

pub struct StorageEngine {
    db: Db,
}

impl StorageEngine {
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let db = sled::open(path).map_err(|e| StorageError::Open{source:e})?;
        Ok(Self { db })
    }

    pub fn memory() -> Result<Self> {
        let config = sled::Config::new().temporary(true);
        let db = config
            .open()
            .map_err(|e| StorageError::Open{source: e})?;
        Ok(Self { db })
    }

    pub fn set(&self, key: &[u8], value: &[u8]) -> Result<()> {
        self.db.insert(key, value)?;
        self.db.flush()?;
        Ok(())
    }

    pub fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        match self.db.get(key)? {
            Some(ivec) => Ok(Some(ivec.to_vec())),
            None => Ok(None),
        }
    }

    pub fn delete(&self, key: &[u8]) -> Result<()> {
        self.db.remove(key)?;
        Ok(())
    }

    pub fn scan_prefix(&self, prefix: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let mut results = Vec::new();
        for item in self.db.scan_prefix(prefix) {
            let (k, v) = item?;
            results.push((k.to_vec(), v.to_vec()));
        }
        Ok(results)
    }

    pub fn flush(&self) -> Result<()> {
        self.db.flush()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_kv_operations() {
        let engine = StorageEngine::memory().unwrap();

        let key = b"test_key";
        let value = b"test_value";

        engine.set(key, value).unwrap();

        let retrieved = engine.get(key).unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap(), value);

        engine.delete(key).unwrap();
        let deleted = engine.get(key).unwrap();
        assert!(deleted.is_none());
    }

    #[test]
    fn test_persistence() {
        let temp_dir = tempfile::tempdir().unwrap();
        let db_path = temp_dir.path().join("test_db");

        {
            let engine = StorageEngine::new(&db_path).unwrap();
            engine.set(b"persist_key", b"persist_value").unwrap();
            engine.flush().unwrap();
        }

        {
            let engine = StorageEngine::new(&db_path).unwrap();
            let value = engine.get(b"persist_key").unwrap();
            assert!(value.is_some());
            assert_eq!(value.unwrap(), b"persist_value");
        }
    }

    #[test]
    fn test_scan_prefix() {
        let engine = StorageEngine::memory().unwrap();

        engine.set(b"user:1", b"alice").unwrap();
        engine.set(b"user:2", b"bob").unwrap();
        engine.set(b"item:1", b"laptop").unwrap();

        let users = engine.scan_prefix(b"user:").unwrap();
        assert_eq!(users.len(), 2);
    }
}
