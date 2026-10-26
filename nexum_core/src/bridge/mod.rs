pub mod error;
use crate::bridge::error::{BridgeError, Result};
use pyo3::prelude::*;
use pyo3::types::PyAny;
use pyo3::types::{PyList, PyModule};
use pyo3::Py;

pub struct PythonBridge {
    initialized: bool,
}
fn with_python<F, T>(f: F) -> Result<T>
where
    F: FnOnce(Python<'_>) -> PyResult<T>,
{
    Python::try_attach(f)
        .ok_or(BridgeError::InterpreterUnavailable)?
        .map_err(|e: PyErr| BridgeError::PythonError(e.to_string()))
}

impl PythonBridge {
    pub fn new() -> Result<Self> {
        Ok(Self { initialized: false })
    }

    pub fn initialize(&mut self) -> Result<()> {
        with_python(|py| {
            let sys = py.import("sys")?;
            let path_attr = sys.getattr("path")?;
            let path = path_attr.cast::<PyList>()?;

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
            return Err(BridgeError::NotInitialized);
        }

        with_python(|py| {
            let nexum_ai = PyModule::import(py, "nexum_ai.optimizer")?;
            let semantic_cache = nexum_ai.getattr("SemanticCache")?;
            let cache_instance = semantic_cache.call0()?;

            let vector: Vec<f32> = cache_instance
                .call_method1("vectorize", (text,))?
                .extract()?;

            Ok(vector)
        })
    }

    pub fn test_integration(&self) -> Result<String> {
        if !self.initialized {
            return Err(BridgeError::NotInitialized);
        }
        with_python(|py| {
            let nexum_ai = PyModule::import(py, "nexum_ai.optimizer")?;
            let test_func = nexum_ai.getattr("test_vectorization")?;
            let result = test_func.call0()?;
            let result_str: String = result.str()?.extract()?;
            Ok(result_str)
        })
    }
}

pub struct SemanticCache {
    bridge: PythonBridge,
    cache: Py<PyAny>,
}

impl SemanticCache {
    pub fn new() -> Result<Self> {
        Self::with_cache_file("semantic_cache.pkl")
    }

    pub fn with_cache_file(cache_file: &str) -> Result<Self> {
        let mut bridge = PythonBridge::new()?;
        bridge.initialize()?;

        let cache = with_python(|py| {
            let nexum_ai = PyModule::import(py, "nexum_ai.optimizer")?;
            let semantic_cache_class = nexum_ai.getattr("SemanticCache")?;
            let cache_instance = semantic_cache_class.call1((0.95, cache_file))?;
            Ok(cache_instance.unbind())
        })?;

        Ok(Self { bridge, cache })
    }

    pub fn get(&self, query: &str) -> Result<Option<String>> {
        with_python(|py| {
            let cache_bound = self.cache.bind(py);
            let result = cache_bound.call_method1("get", (query,))?;

            if result.is_none() {
                Ok(None)
            } else {
                let value: String = result.extract()?;
                Ok(Some(value))
            }
        })
    }

    pub fn put(&self, query: &str, result: &str) -> Result<()> {
        with_python(|py| {
            let cache_bound = self.cache.bind(py);
            cache_bound.call_method1("put", (query, result))?;
            Ok(())
        })
    }

    pub fn vectorize(&self, text: &str) -> Result<Vec<f32>> {
        self.bridge.vectorize(text)
    }

    pub fn save_cache(&self) -> Result<()> {
        with_python(|py| {
            let cache_bound = self.cache.bind(py);
            cache_bound.call_method0("save_cache")?;
            Ok(())
        })
    }

    pub fn load_cache(&self) -> Result<()> {
        with_python(|py| {
            let cache_bound = self.cache.bind(py);
            cache_bound.call_method0("load_cache")?;
            Ok(())
        })
    }

    pub fn clear_cache(&self) -> Result<()> {
        with_python(|py| {
            let cache_bound = self.cache.bind(py);
            cache_bound.call_method0("clear")?;
            Ok(())
        })
    }

    pub fn get_cache_stats(&self) -> Result<String> {
        with_python(|py| {
            let cache_bound = self.cache.bind(py);
            let result = cache_bound.call_method0("get_cache_stats")?;
            let stats_str: String = result.str()?.extract()?;
            Ok(stats_str)
        })
    }

    pub fn explain_query(&self, query: &str) -> Result<String> {
        with_python(|py| {
            let cache_bound = self.cache.bind(py);
            let result = cache_bound.call_method1("explain_query", (query,))?;
            let explain_str: String = result.str()?.extract()?;
            Ok(explain_str)
        })
    }
}

pub struct NLTranslator {
    _bridge: PythonBridge,
    translator: Py<PyAny>,
}

impl NLTranslator {
    pub fn new() -> Result<Self> {
        let mut bridge = PythonBridge::new()?;
        bridge.initialize()?;

        let translator = with_python(|py| {
            let nexum_ai = PyModule::import(py, "nexum_ai.translator")?;
            let translator_class = nexum_ai.getattr("NLTranslator")?;
            let translator_instance = translator_class.call0()?;
            Ok(translator_instance.unbind())
        })?;
        Ok(Self {
            _bridge: bridge,
            translator,
        })
    }

    pub fn translate(&self, natural_query: &str, schema: &str) -> Result<String> {
        with_python(|py| {
            let translator_bound = self.translator.bind(py);
            let result = translator_bound.call_method1("translate", (natural_query, schema))?;

            let sql: String = result.extract()?;
            Ok(sql)
        })
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
        with_python(|py| {
            let nexum_ai = PyModule::import(py, "nexum_ai.optimizer")?;
            let explain_func = nexum_ai.getattr("explain_query_plan")?;
            let format_func = nexum_ai.getattr("format_explain_output")?;

            let result = explain_func.call1((query,))?;
            let formatted = format_func.call1((result,))?;
            let output: String = formatted.extract()?;
            Ok(output)
        })
    }

    pub fn explain_raw(&self, query: &str) -> Result<String> {
        with_python(|py| {
            let nexum_ai = PyModule::import(py, "nexum_ai.optimizer")?;
            let explain_func = nexum_ai.getattr("explain_query_plan")?;

            let result = explain_func.call1((query,))?;
            let output: String = result.str()?.extract()?;
            Ok(output)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_python_available() -> bool {
        let mut bridge = PythonBridge::new().unwrap();
        bridge.initialize().is_ok()
            && Python::try_attach(|py| PyModule::import(py, "nexum_ai.optimizer").is_ok())
                .unwrap_or(false)
    }

    #[test]
    fn test_python_bridge_initialization() {
        if !check_python_available() {
            println!("Skipping test: Python environment not available");
            return;
        }
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

        let explainer = QueryExplainer::new().unwrap();
        let query = "SELECT * FROM users WHERE age > 25";

        let plan = explainer.explain(query).unwrap();

        println!("Explain output:\n{}", plan);
        assert!(plan.contains("QUERY EXECUTION PLAN"));
        assert!(plan.contains("PARSING"));
        assert!(plan.contains("CACHE LOOKUP"));
        assert!(plan.contains("RL AGENT"));
        assert!(plan.contains("EXECUTION STRATEGY"));
    }

    #[test]
    fn test_query_explainer_select_queries() {
        if !check_python_available() {
            println!("Skipping test: Python environment not available");
            return;
        }

        let explainer = QueryExplainer::new().unwrap();

        let test_queries = vec![
            "SELECT * FROM users",
            "SELECT id, name FROM users WHERE age > 25",
            "SELECT COUNT(*) FROM orders WHERE status = 'active'",
            "SELECT * FROM products ORDER BY price DESC LIMIT 10",
        ];

        for query in test_queries {
            let plan = explainer.explain(query).unwrap();

            // Verify all required sections are present
            assert!(
                plan.contains("PARSING"),
                "Missing PARSING section for: {}",
                query
            );
            assert!(
                plan.contains("CACHE LOOKUP"),
                "Missing CACHE LOOKUP section for: {}",
                query
            );
            assert!(
                plan.contains("RL AGENT"),
                "Missing RL AGENT section for: {}",
                query
            );
            assert!(
                plan.contains("EXECUTION STRATEGY"),
                "Missing EXECUTION STRATEGY section for: {}",
                query
            );

            // Verify query type detection
            if query.to_uppercase().starts_with("SELECT") {
                assert!(
                    plan.contains("SELECT"),
                    "Query type not detected for: {}",
                    query
                );
            }
        }
    }

    #[test]
    fn test_query_explainer_mutation_queries() {
        if !check_python_available() {
            println!("Skipping test: Python environment not available");
            return;
        }

        let explainer = QueryExplainer::new().unwrap();

        let test_queries = vec![
            "INSERT INTO users (name, age) VALUES ('John', 30)",
            "UPDATE users SET age = 31 WHERE name = 'John'",
            "DELETE FROM users WHERE id = 1",
        ];

        for query in test_queries {
            let plan = explainer.explain(query).unwrap();

            // All sections should be present
            assert!(plan.contains("PARSING"));
            assert!(plan.contains("CACHE LOOKUP"));
            assert!(plan.contains("EXECUTION STRATEGY"));

            // Detect mutation query types
            let upper = query.to_uppercase();
            if upper.starts_with("INSERT") {
                assert!(plan.contains("INSERT"));
            } else if upper.starts_with("UPDATE") {
                assert!(plan.contains("UPDATE"));
            } else if upper.starts_with("DELETE") {
                assert!(plan.contains("DELETE"));
            }
        }
    }

    #[test]
    fn test_query_explainer_raw_output() {
        if !check_python_available() {
            println!("Skipping test: Python environment not available");
            return;
        }

        let explainer = QueryExplainer::new().unwrap();
        let query = "SELECT * FROM users WHERE age > 25";

        let raw_plan = explainer.explain_raw(query).unwrap();

        // Raw output should be valid Python dict format (contains {} and expected keys)
        assert!(raw_plan.contains('{') || raw_plan.contains("["));
        // Validate structure by checking for actual expected keys present in output
        assert!(
            raw_plan.contains("parsing"),
            "Raw output should contain required structural key 'parsing'"
        );
        assert!(
            raw_plan.contains("cache_analysis"),
            "Raw output should contain required structural key 'cache_analysis'"
        );
        println!("Raw explain output:\n{}", raw_plan);
    }

    #[test]
    fn test_query_explainer_q_values_present() {
        if !check_python_available() {
            println!("Skipping test: Python environment not available");
            return;
        }

        let explainer = QueryExplainer::new().unwrap();
        let query = "SELECT * FROM products WHERE price BETWEEN 10 AND 100";

        let plan = explainer.explain(query).unwrap();

        // RL Agent section should contain Q-values information
        assert!(plan.contains("Q-values"));
        assert!(plan.contains("Best action"));
        println!("Q-values information present in plan");
    }
}
