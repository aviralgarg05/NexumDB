use anyhow::{anyhow, Result};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyModule};

pub struct PythonBridge {
    initialized: bool,
}

impl PythonBridge {
    pub fn new() -> Result<Self> {
        Ok(Self { initialized: false })
    }

    pub fn initialize(&mut self) -> Result<()> {
        Python::with_gil(|py| {
            let sys = py.import("sys")?;
            let path_attr = sys.getattr("path")?;
            let path = path_attr.downcast::<PyList>()?;

            let nexum_ai_pathbuf =
                std::env::current_dir().expect("Failed to get current directory");
            let nexum_ai_path = nexum_ai_pathbuf.to_str().expect("Invalid path");

            path.insert(0, nexum_ai_path)?;

            Ok::<(), PyErr>(())
        })?;

        self.initialized = true;
        Ok(())
    }

    pub fn vectorize(&self, text: &str) -> Result<Vec<f32>> {
        if !self.initialized {
            return Err(anyhow!("Python bridge not initialized"));
        }

        Python::with_gil(|py| {
            let nexum_ai = PyModule::import(py, "nexum_ai.optimizer")?;
            let semantic_cache = nexum_ai.getattr("SemanticCache")?;
            let cache_instance = semantic_cache.call0()?;

            let vector: Vec<f32> = cache_instance
                .call_method1("vectorize", (text,))?
                .extract()?;

            Ok(vector)
        })
        .map_err(|e: PyErr| anyhow!("Python error: {}", e))
    }

    pub fn test_integration(&self) -> Result<String> {
        Python::with_gil(|py| {
            let nexum_ai = PyModule::import(py, "nexum_ai.optimizer")?;
            let test_func = nexum_ai.getattr("test_vectorization")?;
            let result = test_func.call0()?;
            let result_str: String = result.str()?.extract()?;
            Ok(result_str)
        })
        .map_err(|e: PyErr| anyhow!("Python error: {}", e))
    }
}

pub struct SemanticCache {
    bridge: PythonBridge,
    cache: PyObject,
}

impl SemanticCache {
    pub fn new() -> Result<Self> {
        Self::with_cache_file("semantic_cache.pkl")
    }

    pub fn with_cache_file(cache_file: &str) -> Result<Self> {
        let mut bridge = PythonBridge::new()?;
        bridge.initialize()?;

        let cache = Python::with_gil(|py| {
            let nexum_ai = PyModule::import(py, "nexum_ai.optimizer")?;
            let semantic_cache_class = nexum_ai.getattr("SemanticCache")?;
            let cache_instance = semantic_cache_class.call1((0.95, cache_file))?;
            Ok::<PyObject, PyErr>(cache_instance.unbind())
        })?;

        Ok(Self { bridge, cache })
    }

    pub fn get(&self, query: &str) -> Result<Option<String>> {
        Python::with_gil(|py| {
            let cache_bound = self.cache.bind(py);
            let result = cache_bound.call_method1("get", (query,))?;

            if result.is_none() {
                Ok(None)
            } else {
                let value: String = result.extract()?;
                Ok(Some(value))
            }
        })
        .map_err(|e: PyErr| anyhow!("Python error: {}", e))
    }

    pub fn put(&self, query: &str, result: &str) -> Result<()> {
        Python::with_gil(|py| {
            let cache_bound = self.cache.bind(py);
            cache_bound.call_method1("put", (query, result))?;
            Ok::<(), PyErr>(())
        })
        .map_err(|e: PyErr| anyhow!("Python error: {}", e))
    }

    pub fn vectorize(&self, text: &str) -> Result<Vec<f32>> {
        self.bridge.vectorize(text)
    }

    pub fn save_cache(&self) -> Result<()> {
        Python::with_gil(|py| {
            let cache_bound = self.cache.bind(py);
            cache_bound.call_method0("save_cache")?;
            Ok::<(), PyErr>(())
        })
        .map_err(|e: PyErr| anyhow!("Python error: {}", e))
    }

    pub fn load_cache(&self) -> Result<()> {
        Python::with_gil(|py| {
            let cache_bound = self.cache.bind(py);
            cache_bound.call_method0("load_cache")?;
            Ok::<(), PyErr>(())
        })
        .map_err(|e: PyErr| anyhow!("Python error: {}", e))
    }

    pub fn clear_cache(&self) -> Result<()> {
        Python::with_gil(|py| {
            let cache_bound = self.cache.bind(py);
            cache_bound.call_method0("clear")?;
            Ok::<(), PyErr>(())
        })
        .map_err(|e: PyErr| anyhow!("Python error: {}", e))
    }

    pub fn get_cache_stats(&self) -> Result<String> {
        Python::with_gil(|py| {
            let cache_bound = self.cache.bind(py);
            let result = cache_bound.call_method0("get_cache_stats")?;
            let stats_str: String = result.str()?.extract()?;
            Ok(stats_str)
        })
        .map_err(|e: PyErr| anyhow!("Python error: {}", e))
    }

    pub fn explain_query(&self, query: &str) -> Result<String> {
        Python::with_gil(|py| {
            let cache_bound = self.cache.bind(py);
            let result = cache_bound.call_method1("explain_query", (query,))?;
            let explain_str: String = result.str()?.extract()?;
            Ok(explain_str)
        })
        .map_err(|e: PyErr| anyhow!("Python error: {}", e))
    }
}

pub struct NLTranslator {
    _bridge: PythonBridge,
    translator: PyObject,
}

impl NLTranslator {
    pub fn new() -> Result<Self> {
        let mut bridge = PythonBridge::new()?;
        bridge.initialize()?;

        let translator = Python::with_gil(|py| {
            let nexum_ai = PyModule::import(py, "nexum_ai.translator")?;
            let translator_class = nexum_ai.getattr("NLTranslator")?;
            let translator_instance = translator_class.call0()?;
            Ok::<PyObject, PyErr>(translator_instance.unbind())
        })?;

        Ok(Self {
            _bridge: bridge,
            translator,
        })
    }

    pub fn translate(&self, natural_query: &str, schema: &str) -> Result<String> {
        Python::with_gil(|py| {
            let translator_bound = self.translator.bind(py);
            let result = translator_bound.call_method1("translate", (natural_query, schema))?;

            let sql: String = result.extract()?;
            Ok(sql)
        })
        .map_err(|e: PyErr| anyhow!("Python error: {}", e))
    }
}

pub struct QueryExplainer {
    _bridge: PythonBridge,
}

impl QueryExplainer {
    pub fn new() -> Result<Self> {
        let mut bridge = PythonBridge::new()?;
        bridge.initialize()?;
        Ok(Self { _bridge: bridge })
    }

    pub fn explain(&self, query: &str) -> Result<String> {
        Python::with_gil(|py| {
            let nexum_ai = PyModule::import(py, "nexum_ai.optimizer")?;
            let explain_func = nexum_ai.getattr("explain_query_plan")?;
            let format_func = nexum_ai.getattr("format_explain_output")?;

            let result = explain_func.call1((query,))?;
            let formatted = format_func.call1((result,))?;
            let output: String = formatted.extract()?;
            Ok(output)
        })
        .map_err(|e: PyErr| anyhow!("Python error: {}", e))
    }

    pub fn explain_raw(&self, query: &str) -> Result<String> {
        Python::with_gil(|py| {
            let nexum_ai = PyModule::import(py, "nexum_ai.optimizer")?;
            let explain_func = nexum_ai.getattr("explain_query_plan")?;

            let result = explain_func.call1((query,))?;
            let output: String = result.str()?.extract()?;
            Ok(output)
        })
        .map_err(|e: PyErr| anyhow!("Python error: {}", e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_python_available() -> bool {
        let mut bridge = PythonBridge::new().unwrap();
        bridge.initialize().is_ok()
            && Python::with_gil(|py| PyModule::import(py, "nexum_ai.optimizer").is_ok())
    }

    #[test]
    fn test_python_bridge_initialization() {
        let mut bridge = PythonBridge::new().unwrap();
        bridge.initialize().unwrap();
        assert!(bridge.initialized);
    }

    #[test]
    fn test_vectorization() {
        if !check_python_available() {
            println!("Skipping test: Python environment not available");
            return;
        }

        let mut bridge = PythonBridge::new().unwrap();
        bridge.initialize().unwrap();

        let test_text = "SELECT * FROM users WHERE age > 25";
        let vector = bridge.vectorize(test_text).unwrap();

        assert!(!vector.is_empty());
        assert_eq!(vector.len(), 384);
    }

    #[test]
    fn test_semantic_cache_persistence() {
        if !check_python_available() {
            println!("Skipping test: Python environment not available");
            return;
        }

        let cache = SemanticCache::with_cache_file("test_rust_cache.pkl").unwrap();

        let query = "SELECT * FROM users WHERE name = 'test'";
        let result = "Test user data";

        // Put data in cache
        cache.put(query, result).unwrap();

        // Verify cache hit
        let cached = cache.get(query).unwrap();
        assert!(cached.is_some());
        assert_eq!(cached.unwrap(), result);

        // Test cache stats
        let stats = cache.get_cache_stats().unwrap();
        println!("Cache stats: {}", stats);

        // Test save/load cycle
        cache.save_cache().unwrap();

        // Create new cache instance and verify persistence
        let cache2 = SemanticCache::with_cache_file("test_rust_cache.pkl").unwrap();
        let cached2 = cache2.get(query).unwrap();
        assert!(cached2.is_some());
        assert_eq!(cached2.unwrap(), result);

        // Cleanup
        cache2.clear_cache().unwrap();
    }

    #[test]
    fn test_nl_translator() {
        if !check_python_available() {
            println!("Skipping test: Python environment not available");
            return;
        }

        let translator = NLTranslator::new().unwrap();

        let schema = "TABLE users (id INTEGER, name TEXT, age INTEGER)";
        let nl_query = "Show me all users named Alice";

        let sql = translator.translate(nl_query, schema).unwrap();

        println!("Translated: {} -> {}", nl_query, sql);
        assert!(sql.contains("SELECT"));
        assert!(sql.contains("users"));
    }

    #[test]
    fn test_query_explainer() {
        if !check_python_available() {
            println!("Skipping test: Python environment not available");
            return;
        }

        let explainer = super::QueryExplainer::new().unwrap();
        let query = "SELECT * FROM users WHERE age > 25";

        let plan = explainer.explain(query).unwrap();

        println!("Explain output:\n{}", plan);
        assert!(plan.contains("QUERY EXECUTION PLAN"));
        assert!(plan.contains("PARSING"));
        assert!(plan.contains("CACHE LOOKUP"));
        assert!(plan.contains("RL AGENT"));
        assert!(plan.contains("EXECUTION STRATEGY"));
    }
}
