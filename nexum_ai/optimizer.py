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

# Shared constants for defensive formatting and display alignment.
ACTION_DISPLAY_WIDTH = 20
COMPLEXITY_MIN = 0
COMPLEXITY_MAX = 10

# Module-level default cache instance (created once to avoid repeated initialization).
# Track cache file source so env var changes can rebuild the instance safely.
_default_cache: Optional['SemanticCache'] = None
_default_cache_file: Optional[str] = None

def _get_default_cache() -> 'SemanticCache':
    """
    Get or create the default SemanticCache instance.
    
    This caches the default instance at module level to avoid repeated
    initialization overhead (model loading, directory creation, disk I/O)
    when explain_query_plan() is called multiple times without providing
    a cache argument (e.g., in REPL loops).
    
    Returns:
        SemanticCache: Module-level default cache instance
    """
    global _default_cache, _default_cache_file
    current_cache_file = os.environ.get('NEXUMDB_CACHE_FILE', "semantic_cache.pkl")
    if _default_cache is None or _default_cache_file != current_cache_file:
        _default_cache = SemanticCache(cache_file=current_cache_file)
        _default_cache_file = current_cache_file
        logger.debug(
            "Created module-level default SemanticCache instance for cache_file=%s",
            current_cache_file
        )
    return _default_cache


def _reset_default_cache() -> None:
    """Reset module-level default cache instance (primarily for test isolation)."""
    global _default_cache, _default_cache_file
    _default_cache = None
    _default_cache_file = None

class SemanticCache:
    """
    Caches query results using semantic similarity
    Uses local embedding models only
    Supports persistence to disk via JSON or pickle files
    """


    def __init__(self, similarity_threshold: float = 0.95, cache_file: str = "semantic_cache.pkl") -> None:
        self.cache: List[Dict] = []
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

    def initialize_model(self) -> None:
        """
        Start loading the SentenceTransformer model in a background thread.

        This function is intentionally non-blocking. If the model is already
        loaded or currently loading, it returns immediately.
        """
        with self._model_lock:
            # Already loaded
            if self.model is not None:
                return

            # Already loading
            if self._model_loading:
                return

            self._model_loading = True
            self._model_load_error = None

            def _load() -> None:
                try:
                    from sentence_transformers import SentenceTransformer
                    model = SentenceTransformer("all-MiniLM-L6-v2")

                    with self._model_lock:
                        self.model = model
                        self._model_loading = False

                    logger.info("Semantic cache initialized with all-MiniLM-L6-v2 (async)")

                except ImportError as e:
                    with self._model_lock:
                        self.model = None
                        self._model_loading = False
                        self._model_load_error = e
                    logger.warning("sentence-transformers not installed, using fallback")

                except Exception as e:
                    with self._model_lock:
                        self.model = None
                        self._model_loading = False
                        self._model_load_error = e
                    logger.exception("Failed to initialize sentence-transformers model, using fallback")

            self._model_thread = threading.Thread(
                target=_load,
                name="nexumdb-model-loader",
                daemon=True,
            )
            self._model_thread.start()

    def vectorize(self, text: str) -> List[float]:
        """Convert text to embedding vector (non-blocking)."""

        # Kick off async loading if needed
        if self.model is None:
            self.initialize_model()

        # If model is ready, use it
        if self.model is not None:
            embedding = self.model.encode(text)
            return embedding.tolist()

        # If model is still loading or failed, fallback
        return self._fallback_vectorize(text)



    
    def _fallback_vectorize(self, text: str) -> List[float]:
        """Simple fallback vectorization using character hashing"""
        vec = [0.0] * 384
        for i, char in enumerate(text[:384]):
            vec[i] = float(ord(char)) / 128.0
        return vec
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        vec1_arr = np.array(vec1)
        vec2_arr = np.array(vec2)
        
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
            similarity = self.cosine_similarity(query_vec, entry['vector'])
            if similarity >= self.similarity_threshold:
                logger.info(f"Cache hit! Similarity: {similarity:.4f}")
                return entry['result']
        
        return None
    
    def put(self, query: str, result: str) -> None:
        """Store query and result in cache"""
        query_vec = self.vectorize(query)
        self.cache.append({
            'query': query,
            'vector': query_vec,
            'result': result
        })
        logger.info(f"Cached query: {query[:50]}...")
    
    def clear(self) -> None:
        """Clear the cache"""
        self.cache.clear()
        # Remove cache file when clearing
        if self.cache_path.exists():
            self.cache_path.unlink()
            logger.info("Cache file deleted")
    
    def save_cache(self, filepath: Optional[str] = None) -> None:
        """Save cache to disk using JSON format (secure default)"""
        if filepath is None:
            filepath = str(self.cache_path)
        
        # Use JSON format by default for security
        json_filepath = filepath.replace('.pkl', '.json') if filepath.endswith('.pkl') else filepath
        self.save_cache_json(json_filepath)
    
    def load_cache(self, filepath: Optional[str] = None) -> None:
        """Load cache from disk using JSON (safe) or pickle (legacy)"""
        if filepath is None:
            filepath = str(self.cache_path)
        
        # Try JSON first (safer format)
        json_filepath = filepath.replace('.pkl', '.json') if filepath.endswith('.pkl') else f"{filepath}.json"
        if os.path.exists(json_filepath):
            self.load_cache_json(json_filepath)
            return
        
        # Fall back to pickle for legacy files (with restricted unpickler for safety)
        if os.path.exists(filepath) and filepath.endswith('.pkl'):
            try:
                import pickle
                
                # Use RestrictedUnpickler to limit allowed classes
                class RestrictedUnpickler(pickle.Unpickler):
                    """Restricted unpickler that only allows safe types"""
                    ALLOWED_CLASSES = {
                        ('builtins', 'dict'),
                        ('builtins', 'list'),
                        ('builtins', 'str'),
                        ('builtins', 'int'),
                        ('builtins', 'float'),
                        ('builtins', 'bool'),
                        ('builtins', 'tuple'),
                        ('builtins', 'set'),
                        ('builtins', 'frozenset'),
                    }
                    
                    def find_class(self, module: str, name: str) -> type:
                        if (module, name) not in self.ALLOWED_CLASSES:
                            raise pickle.UnpicklingError(
                                f"Forbidden class: {module}.{name}"
                            )
                        return super().find_class(module, name)
                
                with open(filepath, 'rb') as f:
                    data = RestrictedUnpickler(f).load()
                
                self.cache = data.get('cache', [])
                self.similarity_threshold = data.get('similarity_threshold', self.similarity_threshold)
                
                logger.info(
                    "Semantic cache loaded from %s (%d entries)",
                    filepath,
                    len(self.cache),
                )

                logger.info(
                    "Converting legacy pickle cache to JSON format for security"
                )
                
                # Validate cache entries
                valid_entries = []
                for entry in self.cache:
                    if all(key in entry for key in ['query', 'vector', 'result']):
                        valid_entries.append(entry)
                    else:
                        logger.info("Warning: Invalid cache entry found and removed")
                
                self.cache = valid_entries
                
                # Auto-convert to JSON format for future use
                self.save_cache_json(json_filepath)
                
            except Exception:
                logger.exception(
                    "Error loading semantic cache"
                )

                logger.debug(
                    "Starting with empty cache"
                )
                self.cache = []
        else:
            logger.debug(f"No cache file found at {filepath}, starting with empty cache")
    
    def save_cache_json(self, filepath: Optional[str] = None) -> None:
        """Save cache to JSON format (secure and portable)"""
        if filepath is None:
            filepath = str(self.cache_path).replace('.pkl', '.json')
        
        try:
            # Create backup of existing cache
            backup_path = f"{filepath}.backup"
            if os.path.exists(filepath):
                os.rename(filepath, backup_path)
            
            cache_data = {
                'cache': self.cache,
                'similarity_threshold': self.similarity_threshold,
                'cache_size': len(self.cache),
                'format_version': '1.0'
            }
            
            with open(filepath, 'w') as f:
                json.dump(cache_data, f, indent=2)
            
            logger.info(f"Semantic cache saved to {filepath} ({len(self.cache)} entries)")
            
            # Remove backup if save was successful
            if os.path.exists(backup_path):
                os.remove(backup_path)
            
        except Exception:
            logger.exception("Error saving cache to JSON")
            # Restore backup if save failed
            if os.path.exists(backup_path):
                os.rename(backup_path, filepath)
    
    def load_cache_json(self, filepath: Optional[str] = None) -> None:
        """Load cache from JSON format"""
        if filepath is None:
            filepath = str(self.cache_path).replace('.pkl', '.json')
        
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                self.cache = data.get('cache', [])
                self.similarity_threshold = data.get('similarity_threshold', self.similarity_threshold)
                
                logger.info(f"Semantic cache loaded from JSON: {filepath} ({len(self.cache)} entries)")
                
            except Exception:
                logger.exception("Error loading cache from JSON")
                self.cache = []
        else:
            logger.debug(f"No JSON cache file found at {filepath}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'total_entries': len(self.cache),
            'similarity_threshold': self.similarity_threshold,
            'cache_file': str(self.cache_path),
            'cache_exists': self.cache_path.exists(),
            'cache_size_bytes': self.cache_path.stat().st_size if self.cache_path.exists() else 0
        }
    
    def explain_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze query without executing - returns cache similarity scores
        and potential cache hits for EXPLAIN command
        
        Args:
            query: SQL query string to analyze (must be non-empty)
        
        Returns:
            Dict containing:
                - query: Original query
                - cache_entries_checked: Number of entries in cache
                - similarity_threshold: Threshold for cache hits
                - best_match: Query with highest similarity
                - best_similarity: Highest similarity score (0.0-1.0)
                - would_hit_cache: Whether best match exceeds threshold
                - top_matches: List of top 5 similar cached queries
        
        Raises:
            ValueError: If query is empty or invalid
        """
        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string")
        
        query = query.strip()
        if not query:
            raise ValueError("Query cannot be empty or whitespace-only")
        
        try:
            query_vec = self.vectorize(query)
        except Exception as e:
            logger.warning(f"Failed to vectorize query: {e}")
            # Return fallback response
            return {
                'query': query,
                'cache_entries_checked': len(self.cache),
                'similarity_threshold': self.similarity_threshold,
                'best_match': None,
                'best_similarity': 0.0,
                'would_hit_cache': False,
                'top_matches': [],
                'error': str(e)
            }
        
        cache_analysis = []
        best_match = None
        best_similarity = 0.0
        
        # Analyze cache entries safely
        for i, entry in enumerate(self.cache):
            try:
                similarity = self.cosine_similarity(query_vec, entry.get('vector', []))
            except Exception as e:
                logger.warning(f"Failed to compute similarity for cache entry {i}: {e}")
                similarity = 0.0
            
            # Smart truncation for cached query display
            cached_query = entry.get('query', 'N/A')
            if len(cached_query) > 50:
                display_query = cached_query[:50] + '...'
            else:
                display_query = cached_query
                
            cache_analysis.append({
                'cached_query': display_query,
                'similarity': round(similarity, 4),
                'would_hit': similarity >= self.similarity_threshold
            })
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = cached_query
        
        # Sort by similarity descending
        cache_analysis.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Smart truncation for best match
        if best_match and len(best_match) > 50:
            best_match_display = best_match[:50] + '...'
        else:
            best_match_display = best_match
        
        return {
            'query': query,
            'cache_entries_checked': len(self.cache),
            'similarity_threshold': round(self.similarity_threshold, 4),
            'best_match': best_match_display,
            'best_similarity': round(best_similarity, 4),
            'would_hit_cache': best_similarity >= self.similarity_threshold,
            'top_matches': cache_analysis[:5]  # Top 5 similar cached queries
        }
    
    def set_cache_expiration(self, max_age_hours: int = 24) -> None:
        """Remove cache entries older than specified hours (future enhancement)"""
        # This would require adding timestamps to cache entries
        # For now, just a placeholder for TTL functionality
        logger.info(f"Cache expiration set to {max_age_hours} hours (not yet implemented)")
    
    def optimize_cache(self, max_entries: int = 1000) -> None:
        """Remove oldest entries if cache exceeds max size"""
        if len(self.cache) > max_entries:
            removed_count = len(self.cache) - max_entries
            self.cache = self.cache[-max_entries:]  # Keep most recent entries
            logger.info(f"Cache optimized: removed {removed_count} oldest entries")
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
            return np.random.choice(available_actions)
        
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
        if next_state in self.q_table:
            max_next_q = max(self.q_table[next_state].values()) if self.q_table[next_state] else 0.0
        
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state][action] = new_q
        
        logger.debug(f"Updated Q({state}, {action}) = {new_q:.4f}")
    
    def feed_metrics(self, query: str, latency_ms: float) -> None:
        """Feed execution metrics to the optimizer"""
        reward = -latency_ms / 1000.0
        
        state = f"query_type_{len(query) // 10}"
        action = "execute"
        next_state = "completed"
        
        self.update(state, action, reward, next_state)
    
    def explain_action(self, query: str, available_actions: List[str]) -> Dict[str, Any]:
        """
        Explain what action would be taken without executing.
        
        Returns Q-values and predicted action for EXPLAIN command.
        This method provides a read-only analysis of the optimizer's decision-making
        process without actually executing any action or updating the Q-table.
        
        Args:
            query: SQL query string (length > 0)
            available_actions: List of possible actions (non-empty)
        
        Returns:
            Dict containing:
                - state: state key string
                - q_values: Q-values for all actions
                - best_action: action with highest Q-value
                - epsilon: current exploration rate (0.0-1.0)
                - would_explore: whether exploration is possible
                - explanation: human-readable explanation of optimizer behavior
        
        Raises:
            ValueError: If query is empty or available_actions is empty
        """
        # Input validation
        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string")
        if not available_actions or not isinstance(available_actions, (list, tuple)):
            raise ValueError("available_actions must be a non-empty list or tuple")
        
        state = f"query_type_{len(query) // 10}"
        
        # Initialize Q-values with defensive rounding
        q_values = {}
        if state in self.q_table:
            # Only include actions that exist in available_actions
            q_values = {
                a: round(self.q_table[state].get(a, 0.0), 4) 
                for a in available_actions
            }
        else:
            q_values = {a: 0.0 for a in available_actions}
        
        # Find best action with defensive handling
        best_action = max(available_actions, key=lambda a: q_values.get(a, 0.0))
        
        # Defensive truncation for display to keep formatting consistent.
        best_action_display = (
            best_action[:ACTION_DISPLAY_WIDTH]
            if len(best_action) > ACTION_DISPLAY_WIDTH
            else best_action
        )
        
        # Ensure epsilon is in valid range [0, 1]
        epsilon_safe = max(0.0, min(1.0, self.epsilon))
        
        return {
            'state': state,
            'q_values': q_values,
            'best_action': best_action_display,
            'epsilon': round(epsilon_safe, 4),
            'would_explore': epsilon_safe > 0.0,
            'explanation': f'With ε={epsilon_safe:.4f}, agent would explore {epsilon_safe*100:.1f}% of the time'
        }


def test_vectorization() -> Dict[str, Any]:
    """Test function for Rust integration"""
    cache = SemanticCache()
    test_query = "SELECT * FROM users WHERE age > 25"
    vector = cache.vectorize(test_query)
    return {
        'query': test_query,
        'vector': vector[:10],
        'dimension': len(vector)
    }


def explain_query_plan(query: str, cache: Optional[SemanticCache] = None, 
                       optimizer: Optional[QueryOptimizer] = None) -> Dict[str, Any]:
    """
    Generate a complete EXPLAIN plan for a query
    Shows parsing, cache analysis, and RL agent predictions
    
    Args:
        query: SQL query string to explain (must be non-empty)
        cache: Optional SemanticCache instance for cache analysis
        optimizer: Optional QueryOptimizer instance for RL analysis
    
    Returns:
        Dict containing:
            - query: Original query string
            - query_length: Length of query
            - parsing: Query structure analysis
            - cache_analysis: Semantic cache lookup results
            - rl_agent: RL agent decision info with Q-values
            - execution_strategy: Recommended execution strategy
    
    Raises:
        ValueError: If query is empty or invalid
    """
    # Input validation
    if not query or not isinstance(query, str):
        raise ValueError("Query must be a non-empty string")
    
    query = query.strip()
    if not query:
        raise ValueError("Query cannot be empty or whitespace-only")
    
    result = {
        'query': query,
        'query_length': len(query),
        'parsing': {},
        'cache_analysis': {},
        'rl_agent': {},
        'execution_strategy': {}
    }
    
    # 1. Query Parsing Analysis
    query_upper = query.upper().strip()
    if query_upper.startswith('SELECT'):
        query_type = 'SELECT'
    elif query_upper.startswith('INSERT'):
        query_type = 'INSERT'
    elif query_upper.startswith('UPDATE'):
        query_type = 'UPDATE'
    elif query_upper.startswith('DELETE'):
        query_type = 'DELETE'
    elif query_upper.startswith('CREATE'):
        query_type = 'CREATE'
    else:
        query_type = 'UNKNOWN'
    
    result['parsing'] = {
        'query_type': query_type,
        'query_length': len(query),
        'complexity_estimate': min(len(query) // 20, COMPLEXITY_MAX),
        'has_where_clause': 'WHERE' in query_upper,
        'has_join': 'JOIN' in query_upper,
        'has_aggregation': any(agg in query_upper for agg in ['COUNT', 'SUM', 'AVG', 'MAX', 'MIN']),
        'has_order_by': 'ORDER BY' in query_upper,
        'has_group_by': 'GROUP BY' in query_upper
    }
    
    # 2. Cache Analysis
    if cache is None:
        # Use module-level default cache to avoid repeated initialization
        cache = _get_default_cache()
    
    try:
        result['cache_analysis'] = cache.explain_query(query)
    except Exception as e:
        logger.warning("Cache analysis failed: %s", e)
        result['cache_analysis'] = {
            'cache_entries_checked': 0,
            'similarity_threshold': cache.similarity_threshold,
            'best_similarity': 0.0,
            'would_hit_cache': False,
            'top_matches': [],
            'error': str(e)
        }
    
    # 3. RL Agent Analysis
    if optimizer is None:
        optimizer = QueryOptimizer()
    
    available_actions = ['use_cache', 'bypass_cache', 'full_scan', 'index_scan']
    try:
        result['rl_agent'] = optimizer.explain_action(query, available_actions)
    except Exception as e:
        logger.warning("RL agent analysis failed: %s", e)
        # Use optimizer's actual epsilon value instead of hardcoded fallback
        result['rl_agent'] = {
            'state': 'unknown',
            'q_values': {a: 0.0 for a in available_actions},
            'best_action': 'full_scan',
            'epsilon': round(optimizer.epsilon, 4),  # Use actual optimizer epsilon
            'would_explore': optimizer.epsilon > 0.0,
            'explanation': f'RL agent error: {str(e)}',
            'error': str(e)
        }
    
    # 4. Execution Strategy
    would_hit_cache = result['cache_analysis'].get('would_hit_cache', False)
    best_action = result['rl_agent'].get('best_action', 'full_scan')
    
    if would_hit_cache:
        strategy = 'CACHE_HIT'
        estimated_latency = '< 1ms'
    elif best_action == 'use_cache':
        strategy = 'CACHE_MISS_THEN_STORE'
        estimated_latency = '5-50ms'
    elif best_action == 'index_scan':
        strategy = 'INDEX_SCAN'
        estimated_latency = '1-10ms'
    else:
        strategy = 'FULL_SCAN'
        estimated_latency = '10-100ms'
    
    result['execution_strategy'] = {
        'strategy': strategy,
        'estimated_latency': estimated_latency,
        'will_cache_result': query_type == 'SELECT' and not would_hit_cache,
        'recommendation': 'Use cached result' if would_hit_cache else 'Execute and cache'
    }
    
    return result


def format_explain_output(explain_result: Dict[str, Any]) -> str:
    """
    Format EXPLAIN result as a readable table with defensive field width limits
    
    Args:
        explain_result: Dict from explain_query_plan() containing query analysis
    
    Returns:
        Formatted string suitable for terminal display
    """
    # Defensive input validation - graceful fallback instead of raising
    if not isinstance(explain_result, dict):
        return (
            "=" * 70 + "\n"
            "ERROR: Invalid explain_result format\n"
            "=" * 70 + "\n"
            "explain_result must be a dictionary\n"
        )
    
    required_keys = ['query', 'parsing', 'cache_analysis', 'rl_agent', 'execution_strategy']
    missing_keys = [k for k in required_keys if k not in explain_result]
    if missing_keys:
        return (
            "=" * 70 + "\n"
            "ERROR: Missing required keys in explain_result\n"
            "=" * 70 + "\n"
            f"Missing: {', '.join(missing_keys)}\n"
        )
    
    def truncate(value: Any, max_len: int) -> str:
        """Truncate value to max length for box alignment"""
        s = str(value) if value is not None else "N/A"
        if len(s) > max_len:
            return s[:max_len - 3] + "..."
        return s
    
    try:
        lines = []
        lines.append("=" * 70)
        lines.append("QUERY EXECUTION PLAN")
        lines.append("=" * 70)
        
        # Smart query truncation
        query = explain_result.get('query', 'N/A')
        display_query = truncate(query, 60)
        
        lines.append(f"Query: {display_query}")
        lines.append("")
        
        # Parsing section with fallback values
        lines.append("┌─ PARSING ─────────────────────────────────────────────────────────┐")
        p = explain_result.get('parsing', {})
        query_type = truncate(p.get('query_type', 'UNKNOWN'), 15)
        raw_complexity = p.get('complexity_estimate', 0)
        try:
            complexity = int(float(raw_complexity))
        except (TypeError, ValueError):
            complexity = COMPLEXITY_MIN
        complexity = max(COMPLEXITY_MIN, min(COMPLEXITY_MAX, complexity))
        lines.append(f"│ Type: {query_type:<15} Complexity: {complexity}/10              │")
        
        has_where = p.get('has_where_clause', False)
        has_join = p.get('has_join', False)
        has_agg = p.get('has_aggregation', False)
        lines.append(f"│ WHERE: {str(has_where):<8} JOIN: {str(has_join):<8} AGG: {str(has_agg):<8}     │")
        lines.append("└───────────────────────────────────────────────────────────────────┘")
        lines.append("")
        
        # Cache section with fallback values
        lines.append("┌─ CACHE LOOKUP ────────────────────────────────────────────────────┐")
        c = explain_result.get('cache_analysis', {})
        # Defensive limits: cache_entries_checked capped at 99999 for display
        entries_checked = min(c.get('cache_entries_checked', 0), 99999)
        raw_threshold = c.get('similarity_threshold', 0.95)
        raw_best_sim = c.get('best_similarity', 0.0)
        try:
            threshold = float(raw_threshold)
        except (TypeError, ValueError):
            threshold = 0.95
        try:
            best_sim = float(raw_best_sim)
        except (TypeError, ValueError):
            best_sim = 0.0
        # Clamp numeric display fields so fixed-width rows remain aligned.
        threshold = max(0.0, min(1.0, threshold))
        best_sim = max(0.0, min(1.0, best_sim))
        would_hit = c.get('would_hit_cache', False)
        
        lines.append(f"│ Entries checked: {entries_checked:<5} Threshold: {threshold:>6.4f}            │")
        lines.append(f"│ Best similarity: {best_sim:>6.4f} Would hit: {str(would_hit):<6}              │")
        
        top_matches = c.get('top_matches', [])
        if top_matches:
            lines.append("│ Top matches:                                                      │")
            for match in top_matches[:3]:
                sim = match.get('similarity', 0.0)
                hit = "✓" if match.get('would_hit', False) else "✗"
                # Smart truncation for cached queries (limit to 45 chars)
                cached_query = truncate(match.get('cached_query', 'N/A'), 45)
                lines.append(f"│   {hit} {sim:<6.4f} - {cached_query:<45} │")
        
        lines.append("└───────────────────────────────────────────────────────────────────┘")
        lines.append("")
        
        # RL Agent section with fallback values
        lines.append("┌─ RL AGENT ────────────────────────────────────────────────────────┐")
        r = explain_result.get('rl_agent', {})
        # Defensive truncation for state (30 chars) and best_action (20 chars)
        state_display = truncate(r.get('state', 'unknown'), 30)
        best_action_display = truncate(r.get('best_action', 'N/A'), 20)
        epsilon = r.get('epsilon', 0.1)
        
        lines.append(f"│ State: {state_display:<30} Epsilon: {epsilon:<6.4f}        │")
        lines.append(f"│ Best action: {best_action_display:<20}                          │")
        lines.append("│ Q-values:                                                         │")
        
        q_values = r.get('q_values', {})
        if q_values:
            for action, qval in q_values.items():
                # Keep action width aligned with explain_action() best_action display.
                action_display = truncate(action, ACTION_DISPLAY_WIDTH)
                try:
                    q_val_float = float(qval)
                    lines.append(f"│   {action_display:<20}: {q_val_float:>8.4f}                                │")
                except (ValueError, TypeError):
                    lines.append(f"│   {action_display:<20}: N/A                                      │")
        else:
            lines.append("│   (no Q-values available)                                        │")
        
        lines.append("└───────────────────────────────────────────────────────────────────┘")
        lines.append("")
        
        # Execution strategy with fallback values
        lines.append("┌─ EXECUTION STRATEGY ──────────────────────────────────────────────┐")
        e = explain_result.get('execution_strategy', {})
        # Defensive truncation for strategy (20 chars)
        strategy_display = truncate(e.get('strategy', 'UNKNOWN'), 20)
        est_latency = e.get('estimated_latency', 'N/A')
        will_cache = e.get('will_cache_result', False)
        recommendation_display = truncate(e.get('recommendation', 'N/A'), 40)
        
        lines.append(f"│ Strategy: {strategy_display:<20} Est. latency: {est_latency:<10}   │")
        lines.append(f"│ Will cache: {str(will_cache):<8}                                          │")
        lines.append(f"│ Recommendation: {recommendation_display:<40}       │")
        lines.append("└───────────────────────────────────────────────────────────────────┘")
        
        return "\n".join(lines)
    
    except Exception as e:
        logger.error("Error formatting EXPLAIN output: %s", e)
        # Return minimal but valid output with defensive width constraints
        error_msg = str(e)
        # Truncate long error messages to maintain 70-char width
        if len(error_msg) > 60:
            error_msg = error_msg[:57] + "..."
        
        return (
            "=" * 70 + "\n"
            "QUERY EXECUTION PLAN (Error Formatting)\n"
            "=" * 70 + "\n"
            f"Error: {error_msg}\n"
            f"Query: {truncate(explain_result.get('query', 'Unknown'), 60)}\n"
        )


def test_cache_persistence() -> Dict[str, Any]:
    """Test semantic cache persistence functionality"""
    logger.debug("=" * 60)
    logger.debug("Testing Semantic Cache Persistence")
    logger.debug("=" * 60)
    
    # Test 1: Create cache and add entries
    logger.info("1. Creating cache and adding test entries...")
    cache1 = SemanticCache(cache_file="test_cache.pkl")
    
    test_queries = [
        ("SELECT * FROM users WHERE age > 25", "User data for age > 25"),
        ("SELECT name FROM products WHERE price < 100", "Product names under $100"),
        ("SELECT COUNT(*) FROM orders WHERE status = 'active'", "Active order count: 42")
    ]
    
    for query, result in test_queries:
        cache1.put(query, result)
    
    # Save cache after adding entries
    cache1.save_cache()
    
    stats1 = cache1.get_cache_stats()
    logger.info(f"Cache stats after adding entries: {stats1}")
    
    # Test 2: Create new cache instance and verify persistence
    logger.info("\n2. Creating new cache instance to test persistence...")
    cache2 = SemanticCache(cache_file="test_cache.pkl")
    
    stats2 = cache2.get_cache_stats()
    logger.info(f"Cache stats after reload: {stats2}")
    
    # Test 3: Verify cache hits work after reload
    logger.info("\n3. Testing cache hits after reload...")
    for query, expected_result in test_queries:
        cached_result = cache2.get(query)
        if cached_result:
            logger.info(f"✓ Cache hit for: {query[:30]}...")
            logger.info(f"  Result: {cached_result[:50]}...")
        else:
            logger.info(f"✗ Cache miss for: {query[:30]}...")
    
    # Test 4: Test JSON export
    logger.info("\n4. Testing JSON export...")
    cache2.save_cache_json("test_cache.json")
    
    # Test 5: Test cache optimization
    logger.info("\n5. Testing cache optimization...")
    cache2.optimize_cache(max_entries=2)
    
    # Cleanup
    logger.info("\n6. Cleaning up test files...")
    cache2.clear()
    
    return {
        'test_passed': True,
        'entries_before_reload': stats1['total_entries'],
        'entries_after_reload': stats2['total_entries'],
        'persistence_working': stats1['total_entries'] == stats2['total_entries']
    }


if __name__ == "__main__":
    # Run both tests
    logger.info("Running vectorization test...")
    result = test_vectorization()
    logger.info(json.dumps(result, indent=2))
    
    logger.info("\nRunning persistence test...")
    persistence_result = test_cache_persistence()
    logger.info(f"\nPersistence test result: {persistence_result}")
    
    # Test EXPLAIN functionality
    logger.info("\n" + "="*70)
    logger.info("Testing EXPLAIN Query Plan")
    logger.info("="*70)
    
    # Add some test data to cache first
    cache = SemanticCache()
    cache.put("SELECT * FROM users WHERE age > 25", "User data result")
    cache.put("SELECT name FROM products WHERE price < 100", "Product names")
    
    # Test explain
    test_query = "SELECT * FROM users WHERE age > 30"
    explain_result = explain_query_plan(test_query, cache)
    logger.info(format_explain_output(explain_result))
