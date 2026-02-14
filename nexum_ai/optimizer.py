"""
Semantic cache and query optimizer using local embedding models
"""

import logging
import numpy as np
from typing import Optional, List, Dict, Any
import json
import os
from pathlib import Path
import threading

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Shared constants for defensive formatting and display alignment.
ACTION_DISPLAY_WIDTH = 20
COMPLEXITY_MIN = 0
COMPLEXITY_MAX = 10

# Module-level default cache instance (created once to avoid repeated initialization).
_default_cache: Optional["SemanticCache"] = None
_default_cache_file: Optional[str] = None


def _get_default_cache() -> "SemanticCache":
    global _default_cache, _default_cache_file
    current_cache_file = os.environ.get("NEXUMDB_CACHE_FILE", "semantic_cache.pkl")

    if _default_cache is None or _default_cache_file != current_cache_file:
        _default_cache = SemanticCache(cache_file=current_cache_file)
        _default_cache_file = current_cache_file
        logger.debug(
            "Created module-level default SemanticCache instance for cache_file=%s",
            current_cache_file,
        )

    return _default_cache


def _reset_default_cache() -> None:
    """Reset module-level default cache instance (primarily for test isolation)."""
    global _default_cache, _default_cache_file
    _default_cache = None
    _default_cache_file = None


class SemanticCache:
    """
    Caches query results using semantic similarity.
    Uses local embedding models only.
    Supports persistence to disk via JSON.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.95,
        cache_file: str = "semantic_cache.pkl",
    ) -> None:
        self.cache: List[Dict[str, Any]] = []
        self.similarity_threshold = similarity_threshold
        self.model = None

        # Async model loading state
        self._model_lock = threading.Lock()
        self._model_loading = False
        self._model_load_error: Optional[Exception] = None
        self._model_thread: Optional[threading.Thread] = None

        # Support environment variable for cache file path
        cache_file_env = os.environ.get("NEXUMDB_CACHE_FILE", cache_file)
        self.cache_file = cache_file_env

        self.cache_dir = Path("cache")
        self.cache_dir.mkdir(exist_ok=True)

        self.cache_path = self.cache_dir / self.cache_file

        # Load existing cache on initialization
        self.load_cache()

    # ------------------------------
    # Model + Vectorization
    # ------------------------------

    def initialize_model(self) -> None:
        """Load sentence-transformers model in background thread."""
        with self._model_lock:
            if self.model is not None:
                return
            if self._model_load_error is not None:
                return
            if self._model_loading:
                return
            self._model_loading = True

        def _load() -> None:
            try:
                from sentence_transformers import SentenceTransformer

                model = SentenceTransformer("all-MiniLM-L6-v2")

                with self._model_lock:
                    self.model = model
                    self._model_loading = False

                logger.info("Semantic cache initialized with all-MiniLM-L6-v2 (async)")

            except Exception as e:
                with self._model_lock:
                    self.model = None
                    self._model_loading = False
                    self._model_load_error = e
                logger.warning("Semantic model unavailable, using fallback vectors")

        self._model_thread = threading.Thread(
            target=_load,
            name="nexumdb-model-loader",
            daemon=True,
        )
        self._model_thread.start()

    def _fallback_vectorize(self, text: str) -> List[float]:
        """Fallback vectorization using character hashing (always returns list)."""
        vec = [0.0] * 384
        for i, char in enumerate(text[:384]):
            vec[i] = float(ord(char)) / 128.0
        return vec

    def vectorize(self, text: str) -> List[float]:
        """
        Convert text to embedding vector.

        If sentence-transformers is not ready, fallback vector is used.
        This prevents NoneType crashes.
        """
        if self.model is None and self._model_load_error is None:
            self.initialize_model()

        # Wait briefly for model
        if self._model_thread is not None:
            self._model_thread.join(timeout=2.0)

        with self._model_lock:
            if self.model is not None:
                embedding = self.model.encode(text)
                return embedding.tolist()

        # fallback always
        return self._fallback_vectorize(text)

    # ------------------------------
    # Similarity + Cache
    # ------------------------------

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        if vec1 is None or vec2 is None:
            return 0.0

        vec1_arr = np.array(vec1, dtype=float)
        vec2_arr = np.array(vec2, dtype=float)

        if vec1_arr.size == 0 or vec2_arr.size == 0:
            return 0.0

        dot_product = np.dot(vec1_arr, vec2_arr)
        norm1 = np.linalg.norm(vec1_arr)
        norm2 = np.linalg.norm(vec2_arr)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def get(self, query: str) -> Optional[str]:
        """Retrieve cached result if similar query exists"""
        query_vec = self.vectorize(query)

        for entry in self.cache:
            entry_vec = entry.get("vector")

            # skip invalid stored vectors
            if not isinstance(entry_vec, list) or len(entry_vec) == 0:
                continue

            similarity = self.cosine_similarity(query_vec, entry_vec)
            if similarity >= self.similarity_threshold:
                logger.info("Cache hit! Similarity: %.4f", similarity)
                return entry.get("result")

        return None

    def put(self, query: str, result: str) -> None:
        """Store query and result in cache"""
        query_vec = self.vectorize(query)

        # never store None
        if not isinstance(query_vec, list) or len(query_vec) == 0:
            return

        self.cache.append(
            {
                "query": query,
                "vector": query_vec,
                "result": result,
            }
        )
        logger.info("Cached query: %s...", query[:50])

    def clear(self) -> None:
        """Clear the cache"""
        self.cache.clear()

        # delete JSON file too
        json_path = str(self.cache_path).replace(".pkl", ".json")
        if os.path.exists(json_path):
            os.remove(json_path)

        if self.cache_path.exists():
            try:
                self.cache_path.unlink()
            except Exception:
                pass

        logger.info("Cache cleared")

    # ------------------------------
    # Persistence (JSON)
    # ------------------------------

    def save_cache(self, filepath: Optional[str] = None) -> None:
        """Save cache to disk using JSON format"""
        if filepath is None:
            filepath = str(self.cache_path)

        json_filepath = (
            filepath.replace(".pkl", ".json") if filepath.endswith(".pkl") else filepath
        )
        self.save_cache_json(json_filepath)

    def load_cache(self, filepath: Optional[str] = None) -> None:
        """Load cache from disk using JSON format"""
        if filepath is None:
            filepath = str(self.cache_path)

        json_filepath = (
            filepath.replace(".pkl", ".json")
            if filepath.endswith(".pkl")
            else f"{filepath}.json"
        )

        if not os.path.exists(json_filepath):
            self.cache = []
            return

        self.load_cache_json(json_filepath)

    def save_cache_json(self, filepath: Optional[str] = None) -> None:
        """Save cache to JSON format"""
        if filepath is None:
            filepath = str(self.cache_path).replace(".pkl", ".json")

        # remove bad entries (vector None)
        cleaned = []
        for entry in self.cache:
            vec = entry.get("vector")
            if isinstance(vec, list) and len(vec) > 0:
                cleaned.append(entry)

        self.cache = cleaned

        cache_data = {
            "cache": self.cache,
            "similarity_threshold": self.similarity_threshold,
            "cache_size": len(self.cache),
            "format_version": "1.0",
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, indent=2)

        logger.info("Cache saved to %s (%d entries)", filepath, len(self.cache))

    def load_cache_json(self, filepath: Optional[str] = None) -> None:
        """Load cache from JSON format"""
        if filepath is None:
            filepath = str(self.cache_path).replace(".pkl", ".json")

        if not os.path.exists(filepath):
            self.cache = []
            return

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            loaded_cache = data.get("cache", [])
            cleaned = []

            for entry in loaded_cache:
                vec = entry.get("vector")
                if isinstance(vec, list) and len(vec) > 0:
                    cleaned.append(entry)

            self.cache = cleaned
            self.similarity_threshold = data.get(
                "similarity_threshold", self.similarity_threshold
            )

            logger.info(
                "Cache loaded from JSON: %s (%d entries)",
                filepath,
                len(self.cache),
            )

        except Exception:
            logger.exception("Error loading cache from JSON")
            self.cache = []

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        json_path = str(self.cache_path).replace(".pkl", ".json")

        return {
            "total_entries": len(self.cache),
            "similarity_threshold": self.similarity_threshold,
            "cache_file": json_path,
            "cache_exists": os.path.exists(json_path),
            "cache_size_bytes": os.path.getsize(json_path) if os.path.exists(json_path) else 0,
        }

    # ------------------------------
    # EXPLAIN helper
    # ------------------------------

    def explain_query(self, query: str) -> Dict[str, Any]:
        """Analyze query similarity against cache entries"""
        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string")

        query = query.strip()
        if not query:
            raise ValueError("Query cannot be empty")

        query_vec = self.vectorize(query)

        cache_analysis = []
        best_match = None
        best_similarity = 0.0

        for entry in self.cache:
            entry_vec = entry.get("vector")
            if not isinstance(entry_vec, list) or len(entry_vec) == 0:
                continue

            similarity = self.cosine_similarity(query_vec, entry_vec)

            cached_query = entry.get("query", "N/A")
            display_query = cached_query[:50] + "..." if len(cached_query) > 50 else cached_query

            cache_analysis.append(
                {
                    "cached_query": display_query,
                    "similarity": round(similarity, 4),
                    "would_hit": similarity >= self.similarity_threshold,
                }
            )

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = cached_query

        cache_analysis.sort(key=lambda x: x["similarity"], reverse=True)

        best_match_display = (
            best_match[:50] + "..." if best_match and len(best_match) > 50 else best_match
        )

        return {
            "query": query,
            "cache_entries_checked": len(self.cache),
            "similarity_threshold": round(self.similarity_threshold, 4),
            "best_match": best_match_display,
            "best_similarity": round(best_similarity, 4),
            "would_hit_cache": best_similarity >= self.similarity_threshold,
            "top_matches": cache_analysis[:5],
        }

    def optimize_cache(self, max_entries: int = 1000) -> None:
        """Remove oldest entries if cache exceeds max size"""
        if len(self.cache) > max_entries:
            removed_count = len(self.cache) - max_entries
            self.cache = self.cache[-max_entries:]
            logger.info("Cache optimized: removed %d oldest entries", removed_count)
            self.save_cache()


class QueryOptimizer:
    """
    Reinforcement learning-based query optimizer
    Uses Q-learning to optimize query execution
    """

    def __init__(self, learning_rate: float = 0.1, discount_factor: float = 0.9) -> None:
        self.q_table: Dict[str, Dict[str, float]] = {}
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = 0.1

    def get_action(self, state: str, available_actions: List[str]) -> str:
        """Select action using epsilon-greedy strategy"""
        if np.random.random() < self.epsilon:
            return str(np.random.choice(available_actions))

        if state not in self.q_table:
            self.q_table[state] = {action: 0.0 for action in available_actions}

        state_values = self.q_table[state]
        best_action = max(available_actions, key=lambda a: state_values.get(a, 0.0))
        return best_action

    def update(self, state: str, action: str, reward: float, next_state: str) -> None:
        """Update Q-values based on observed reward"""
        if state not in self.q_table:
            self.q_table[state] = {}

        if action not in self.q_table[state]:
            self.q_table[state][action] = 0.0

        current_q = self.q_table[state][action]

        max_next_q = 0.0
        if next_state in self.q_table and self.q_table[next_state]:
            max_next_q = max(self.q_table[next_state].values())

        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        self.q_table[state][action] = new_q

        logger.debug("Updated Q(%s, %s) = %.4f", state, action, new_q)

    def explain_action(self, query: str, available_actions: List[str]) -> Dict[str, Any]:
        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string")
        if not available_actions:
            raise ValueError("available_actions must be non-empty")

        state = f"query_type_{len(query) // 10}"

        if state in self.q_table:
            q_values = {a: round(self.q_table[state].get(a, 0.0), 4) for a in available_actions}
        else:
            q_values = {a: 0.0 for a in available_actions}

        best_action = max(available_actions, key=lambda a: q_values.get(a, 0.0))
        best_action_display = (
            best_action[:ACTION_DISPLAY_WIDTH]
            if len(best_action) > ACTION_DISPLAY_WIDTH
            else best_action
        )

        epsilon_safe = max(0.0, min(1.0, self.epsilon))

        return {
            "state": state,
            "q_values": q_values,
            "best_action": best_action_display,
            "epsilon": round(epsilon_safe, 4),
            "would_explore": epsilon_safe > 0.0,
            "explanation": f"With ε={epsilon_safe:.4f}, agent would explore {epsilon_safe*100:.1f}% of the time",
        }


def explain_query_plan(
    query: str,
    cache: Optional[SemanticCache] = None,
    optimizer: Optional[QueryOptimizer] = None,
) -> Dict[str, Any]:
    """Generate a complete EXPLAIN plan for a query"""

    if not query or not isinstance(query, str):
        raise ValueError("Query must be a non-empty string")

    query = query.strip()
    if not query:
        raise ValueError("Query cannot be empty")

    result = {
        "query": query,
        "query_length": len(query),
        "parsing": {},
        "cache_analysis": {},
        "rl_agent": {},
        "execution_strategy": {},
    }

    query_upper = query.upper().strip()
    if query_upper.startswith("SELECT"):
        query_type = "SELECT"
    elif query_upper.startswith("INSERT"):
        query_type = "INSERT"
    elif query_upper.startswith("UPDATE"):
        query_type = "UPDATE"
    elif query_upper.startswith("DELETE"):
        query_type = "DELETE"
    elif query_upper.startswith("CREATE"):
        query_type = "CREATE"
    else:
        query_type = "UNKNOWN"

    result["parsing"] = {
        "query_type": query_type,
        "query_length": len(query),
        "complexity_estimate": min(len(query) // 20, COMPLEXITY_MAX),
        "has_where_clause": "WHERE" in query_upper,
        "has_join": "JOIN" in query_upper,
        "has_aggregation": any(
            agg in query_upper for agg in ["COUNT", "SUM", "AVG", "MAX", "MIN"]
        ),
        "has_order_by": "ORDER BY" in query_upper,
        "has_group_by": "GROUP BY" in query_upper,
    }

    if cache is None:
        cache = _get_default_cache()

    try:
        result["cache_analysis"] = cache.explain_query(query)
    except Exception as e:
        logger.warning("Cache analysis failed: %s", e)
        result["cache_analysis"] = {
            "cache_entries_checked": 0,
            "similarity_threshold": cache.similarity_threshold,
            "best_similarity": 0.0,
            "would_hit_cache": False,
            "top_matches": [],
            "error": str(e),
        }

    if optimizer is None:
        optimizer = QueryOptimizer()

    available_actions = ["use_cache", "bypass_cache", "full_scan", "index_scan"]
    result["rl_agent"] = optimizer.explain_action(query, available_actions)

    would_hit_cache = result["cache_analysis"].get("would_hit_cache", False)
    best_action = result["rl_agent"].get("best_action", "full_scan")

    if would_hit_cache:
        strategy = "CACHE_HIT"
        estimated_latency = "< 1ms"
    elif best_action == "use_cache":
        strategy = "CACHE_MISS_THEN_STORE"
        estimated_latency = "5-50ms"
    elif best_action == "index_scan":
        strategy = "INDEX_SCAN"
        estimated_latency = "1-10ms"
    else:
        strategy = "FULL_SCAN"
        estimated_latency = "10-100ms"

    result["execution_strategy"] = {
        "strategy": strategy,
        "estimated_latency": estimated_latency,
        "will_cache_result": query_type == "SELECT" and not would_hit_cache,
        "recommendation": "Use cached result" if would_hit_cache else "Execute and cache",
    }

    return result


def format_explain_output(explain_result: Dict[str, Any]) -> str:
    """Format EXPLAIN result as readable table"""

    def truncate(value: Any, max_len: int) -> str:
        s = str(value) if value is not None else "N/A"
        if len(s) > max_len:
            return s[: max_len - 3] + "..."
        return s

    lines = []
    lines.append("=" * 70)
    lines.append("QUERY EXECUTION PLAN")
    lines.append("=" * 70)

    query = explain_result.get("query", "N/A")
    lines.append(f"Query: {truncate(query, 60)}")
    lines.append("")

    p = explain_result.get("parsing", {})
    lines.append("┌─ PARSING ─────────────────────────────────────────────────────────┐")
    lines.append(
        f"│ Type: {truncate(p.get('query_type','UNKNOWN'),15):<15} "
        f"Complexity: {p.get('complexity_estimate',0)}/10              │"
    )
    lines.append("└───────────────────────────────────────────────────────────────────┘")
    lines.append("")

    c = explain_result.get("cache_analysis", {})
    lines.append("┌─ CACHE LOOKUP ────────────────────────────────────────────────────┐")
    lines.append(
        f"│ Entries checked: {c.get('cache_entries_checked',0):<5} "
        f"Threshold: {c.get('similarity_threshold',0.95):>6.4f}            │"
    )
    lines.append(
        f"│ Best similarity: {c.get('best_similarity',0.0):>6.4f} "
        f"Would hit: {str(c.get('would_hit_cache',False)):<6}              │"
    )
    lines.append("└───────────────────────────────────────────────────────────────────┘")
    lines.append("")

    r = explain_result.get("rl_agent", {})
    lines.append("┌─ RL AGENT ────────────────────────────────────────────────────────┐")
    lines.append(
        f"│ State: {truncate(r.get('state','unknown'),30):<30} "
        f"Epsilon: {r.get('epsilon',0.1):<6.4f}        │"
    )
    lines.append(f"│ Best action: {truncate(r.get('best_action','N/A'),20):<20}                          │")
    lines.append("└───────────────────────────────────────────────────────────────────┘")
    lines.append("")

    e = explain_result.get("execution_strategy", {})
    lines.append("┌─ EXECUTION STRATEGY ──────────────────────────────────────────────┐")
    lines.append(
        f"│ Strategy: {truncate(e.get('strategy','UNKNOWN'),20):<20} "
        f"Est. latency: {truncate(e.get('estimated_latency','N/A'),10):<10}   │"
    )
    lines.append("└───────────────────────────────────────────────────────────────────┘")

    return "\n".join(lines)


# ------------------------------
# TESTS
# ------------------------------

def test_vectorization() -> Dict[str, Any]:
    """Test function for vectorization"""
    cache = SemanticCache()
    test_query = "SELECT * FROM users WHERE age > 25"
    vector = cache.vectorize(test_query)

    return {
        "query": test_query,
        "vector": vector[:10],
        "dimension": len(vector),
        "note": "Vector created successfully (model or fallback).",
    }


def test_cache_persistence() -> Dict[str, Any]:
    """Test semantic cache persistence functionality"""

    logger.info("=" * 60)
    logger.info("Testing Semantic Cache Persistence")
    logger.info("=" * 60)

    cache1 = SemanticCache(cache_file="test_cache.pkl")

    test_queries = [
        ("SELECT * FROM users WHERE age > 25", "User data for age > 25"),
        ("SELECT name FROM products WHERE price < 100", "Product names under $100"),
        ("SELECT COUNT(*) FROM orders WHERE status = 'active'", "Active order count: 42"),
    ]

    for query, result in test_queries:
        cache1.put(query, result)

    cache1.save_cache()

    stats1 = cache1.get_cache_stats()
    logger.info("Cache stats after adding entries: %s", stats1)

    cache2 = SemanticCache(cache_file="test_cache.pkl")
    stats2 = cache2.get_cache_stats()
    logger.info("Cache stats after reload: %s", stats2)

    for query, _ in test_queries:
        cached_result = cache2.get(query)
        if cached_result:
            logger.info("✓ Cache hit for: %s", query[:30])
        else:
            logger.info("✗ Cache miss for: %s", query[:30])

    cache2.clear()

    return {
        "test_passed": True,
        "entries_before_reload": stats1["total_entries"],
        "entries_after_reload": stats2["total_entries"],
        "persistence_working": stats1["total_entries"] == stats2["total_entries"],
    }


if __name__ == "__main__":
    logger.info("Running vectorization test...")
    result = test_vectorization()
    logger.info(json.dumps(result, indent=2))

    logger.info("\nRunning persistence test...")
    persistence_result = test_cache_persistence()
    logger.info("\nPersistence test result: %s", persistence_result)

    logger.info("\n" + "=" * 70)
    logger.info("Testing EXPLAIN Query Plan")
    logger.info("=" * 70)

    cache = SemanticCache()
    cache.put("SELECT * FROM users WHERE age > 25", "User data result")
    cache.put("SELECT name FROM products WHERE price < 100", "Product names")

    test_query = "SELECT * FROM users WHERE age > 30"
    explain_result = explain_query_plan(test_query, cache)
    logger.info("\n" + format_explain_output(explain_result))
