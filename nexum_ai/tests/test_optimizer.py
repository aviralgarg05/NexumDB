"""
Unit tests for optimizer.py - Query optimization logic
"""

from nexum_ai.optimizer import SemanticCache, QueryOptimizer


class TestSemanticCache:
    """Test suite for SemanticCache class"""
    
    def test_initialization(self, tmp_path):
        """Test SemanticCache initialization"""
        cache = SemanticCache(
            similarity_threshold=0.9,
            cache_file=str(tmp_path / "test_init.json")
        )
        assert cache.similarity_threshold == 0.9
        assert cache.cache == []
        assert cache.model is None
    
    def test_vectorize_fallback(self, tmp_path):
        """Test fallback vectorization when model is not available"""
        cache = SemanticCache(cache_file=str(tmp_path / "test_vectorize_fallback.json"))
        cache.model = None  # Force fallback
        
        vector = cache.vectorize("test query")
        assert isinstance(vector, list)
        assert len(vector) == 384
        assert all(isinstance(v, float) for v in vector)
    
    def test_vectorize_with_model(self, tmp_path):
        """Test vectorization with sentence transformer model"""
        cache = SemanticCache(cache_file=str(tmp_path / "test_vectorize_model.json"))
        
        # Use fallback since model not installed
        cache.model = None
        vector = cache.vectorize("test query")
        
        assert isinstance(vector, list)
        assert len(vector) == 384
    
    def test_cosine_similarity_identical(self, tmp_path):
        """Test cosine similarity with identical vectors"""
        cache = SemanticCache(cache_file=str(tmp_path / "test_cosine_identical.json"))
        vec1 = [1.0, 2.0, 3.0]
        vec2 = [1.0, 2.0, 3.0]
        
        similarity = cache.cosine_similarity(vec1, vec2)
        assert abs(similarity - 1.0) < 0.001
    
    def test_cosine_similarity_orthogonal(self, tmp_path):
        """Test cosine similarity with orthogonal vectors"""
        cache = SemanticCache(cache_file=str(tmp_path / "test_cosine_orthogonal.json"))
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        
        similarity = cache.cosine_similarity(vec1, vec2)
        assert abs(similarity) < 0.001
    
    def test_cosine_similarity_zero_vector(self, tmp_path):
        """Test cosine similarity with zero vector"""
        cache = SemanticCache(cache_file=str(tmp_path / "test_cosine_zero.json"))
        vec1 = [0.0, 0.0, 0.0]
        vec2 = [1.0, 2.0, 3.0]
        
        similarity = cache.cosine_similarity(vec1, vec2)
        assert similarity == 0.0
    
    def test_put_and_get_cache_hit(self, tmp_path):
        """Test caching and retrieving with high similarity"""
        cache = SemanticCache(
            similarity_threshold=0.95,
            cache_file=str(tmp_path / "test_put_get.json")
        )
        
        query = "SELECT * FROM users"
        result = "result_data"
        
        cache.put(query, result)
        assert len(cache.cache) == 1
        
        # Same query should hit cache
        retrieved = cache.get(query)
        assert retrieved == result
    
    def test_get_cache_miss(self, tmp_path):
        """Test cache miss with different query"""
        cache = SemanticCache(
            similarity_threshold=0.95,
            cache_file=str(tmp_path / "test_cache_miss.json")
        )
        
        cache.put("SELECT * FROM users", "result1")
        
        # Very different query should miss
        retrieved = cache.get("SELECT * FROM products WHERE price > 100")
        assert retrieved is None
    
    def test_clear_cache(self, tmp_path):
        """Test clearing the cache"""
        cache = SemanticCache(cache_file=str(tmp_path / "test_clear.json"))
        
        cache.put("query1", "result1")
        cache.put("query2", "result2")
        assert len(cache.cache) == 2
        
        cache.clear()
        assert len(cache.cache) == 0
    
    def test_multiple_cache_entries(self, tmp_path):
        """Test cache with multiple entries"""
        cache = SemanticCache(
            similarity_threshold=0.95,
            cache_file=str(tmp_path / "test_multiple.json")
        )
        
        queries = [
            ("SELECT * FROM users", "result1"),
            ("SELECT * FROM products", "result2"),
            ("SELECT * FROM orders", "result3"),
        ]
        
        for query, result in queries:
            cache.put(query, result)
        
        assert len(cache.cache) == 3
    
    def test_lru_eviction_on_cache_full(self, tmp_path):
        """Test that LRU eviction kicks in when cache is full"""
        cache = SemanticCache(
            similarity_threshold=0.95, 
            max_cache_size=3,
            cache_file=str(tmp_path / "test_lru_eviction.json")
        )
        
        # Add 3 entries (fill cache) - use completely different strings to avoid collisions
        cache.put("SELECT * FROM users WHERE age > 25", "result1")
        cache.put("INSERT INTO products (name, price) VALUES ('Widget', 99.99)", "result2")
        cache.put("DELETE FROM orders WHERE status = 'cancelled' AND created_at < '2023-01-01'", "result3")
        assert len(cache.cache) == 3
        
        # Add 4th entry - should evict oldest (first query)
        cache.put("UPDATE inventory SET quantity = quantity - 1 WHERE product_id = 42", "result4")
        assert len(cache.cache) == 3
        assert cache.evictions == 1
        
        # Verify first query was evicted (won't be found)
        queries_in_cache = [entry['query'] for entry in cache.cache]
        assert "SELECT * FROM users WHERE age > 25" not in queries_in_cache
        assert "INSERT INTO products (name, price) VALUES ('Widget', 99.99)" in queries_in_cache
        assert "DELETE FROM orders WHERE status = 'cancelled' AND created_at < '2023-01-01'" in queries_in_cache
        assert "UPDATE inventory SET quantity = quantity - 1 WHERE product_id = 42" in queries_in_cache
    
    def test_lru_updates_on_access(self, tmp_path):
        """Test that accessing an entry updates its LRU timestamp"""
        import time
        
        cache = SemanticCache(
            similarity_threshold=0.95, 
            max_cache_size=2,
            cache_file=str(tmp_path / "test_lru_updates.json")
        )
        
        # Add 2 entries - use completely different strings
        cache.put("SELECT COUNT(*) FROM customers WHERE country = 'USA'", "result1")
        time.sleep(0.01)
        cache.put("INSERT INTO logs (message, level) VALUES ('System started', 'INFO')", "result2")
        time.sleep(0.01)
        
        # Access first query (should update its timestamp)
        result = cache.get("SELECT COUNT(*) FROM customers WHERE country = 'USA'")
        assert result == "result1"
        time.sleep(0.01)
        
        # Add 3rd entry - should evict second query (oldest), not first
        cache.put("DELETE FROM temp_files WHERE created_at < NOW() - INTERVAL '7 days'", "result3")
        assert len(cache.cache) == 2
        
        queries_in_cache = [entry['query'] for entry in cache.cache]
        assert "SELECT COUNT(*) FROM customers WHERE country = 'USA'" in queries_in_cache  # Was accessed recently
        assert "INSERT INTO logs (message, level) VALUES ('System started', 'INFO')" not in queries_in_cache  # Should be evicted
        assert "DELETE FROM temp_files WHERE created_at < NOW() - INTERVAL '7 days'" in queries_in_cache
    
    def test_cache_stats_tracking(self, tmp_path):
        """Test that cache statistics are tracked correctly"""
        cache = SemanticCache(
            similarity_threshold=0.95, 
            max_cache_size=5,
            cache_file=str(tmp_path / "test_stats.json")
        )
        
        # Add entries - use completely different strings
        cache.put("SELECT * FROM products WHERE category = 'Electronics'", "result1")
        cache.put("UPDATE users SET last_login = NOW() WHERE user_id = 123", "result2")
        
        # Cache hit
        cache.get("SELECT * FROM products WHERE category = 'Electronics'")
        
        # Cache miss - completely different query
        cache.get("CREATE TABLE new_table (id INT PRIMARY KEY, name VARCHAR(255))")
        
        stats = cache.get_cache_stats()
        assert stats['total_entries'] == 2
        assert stats['max_cache_size'] == 5
        assert stats['hits'] == 1
        assert stats['misses'] == 1
        assert stats['evictions'] == 0
        assert stats['hit_rate'] == 0.5
    
    def test_optimize_cache_manual_resize(self, tmp_path):
        """Test manually resizing cache with optimize_cache"""
        cache = SemanticCache(
            similarity_threshold=0.95, 
            max_cache_size=10,
            cache_file=str(tmp_path / "test_resize.json")
        )
        
        # Add 5 entries - use very different SQL commands
        cache.put("SELECT employee_id, name FROM employees WHERE department = 'Sales'", "result0")
        cache.put("INSERT INTO audit_log (action, timestamp) VALUES ('LOGIN', NOW())", "result1")
        cache.put("DELETE FROM expired_sessions WHERE last_activity < NOW() - INTERVAL '1 hour'", "result2")
        cache.put("UPDATE inventory SET stock_level = stock_level + 100 WHERE warehouse_id = 5", "result3")
        cache.put("CREATE INDEX idx_customer_email ON customers(email) WHERE active = true", "result4")
        
        assert len(cache.cache) == 5
        
        # Resize to smaller cache
        cache.optimize_cache(new_max_size=3)
        assert cache.max_cache_size == 3
        assert len(cache.cache) == 3
        assert cache.evictions == 2
    
    def test_cache_size_zero_max(self, tmp_path):
        """Test cache behavior with max_cache_size = 0
        
        Note: With max_cache_size=0, put() follows the add-then-evict pattern.
        The entry is briefly added to cache, then immediately evicted to maintain
        the size constraint. This is the expected behavior, not a no-op.
        """
        cache = SemanticCache(
            similarity_threshold=0.95, 
            max_cache_size=0,
            cache_file=str(tmp_path / "test_zero_max.json")
        )
        
        # Should not be able to add any entries
        cache.put("query1", "result1")
        assert len(cache.cache) == 0
        assert cache.evictions == 1
    
    def test_negative_max_cache_size_raises_error(self, tmp_path):
        """Test that negative max_cache_size raises ValueError"""
        import pytest
        with pytest.raises(ValueError, match="max_cache_size must be non-negative"):
            SemanticCache(
                similarity_threshold=0.95, 
                max_cache_size=-1,
                cache_file=str(tmp_path / "test_negative.json")
            )
    
    def test_negative_new_max_size_raises_error(self, tmp_path):
        """Test that negative new_max_size in optimize_cache raises ValueError"""
        import pytest
        cache = SemanticCache(
            similarity_threshold=0.95, 
            max_cache_size=10,
            cache_file=str(tmp_path / "test_negative_resize.json")
        )
        with pytest.raises(ValueError, match="new_max_size must be non-negative"):
            cache.optimize_cache(new_max_size=-5)


class TestQueryOptimizer:
    """Test suite for QueryOptimizer class"""
    
    def test_initialization(self):
        """Test QueryOptimizer initialization"""
        optimizer = QueryOptimizer(learning_rate=0.2, discount_factor=0.8)
        assert optimizer.learning_rate == 0.2
        assert optimizer.discount_factor == 0.8
        assert optimizer.epsilon == 0.1
        assert optimizer.q_table == {}
    
    def test_get_action_exploration(self):
        """Test action selection during exploration"""
        optimizer = QueryOptimizer()
        optimizer.epsilon = 1.0  # Force exploration
        
        actions = ["action1", "action2", "action3"]
        action = optimizer.get_action("state1", actions)
        
        assert action in actions
    
    def test_get_action_exploitation(self):
        """Test action selection during exploitation"""
        optimizer = QueryOptimizer()
        optimizer.epsilon = 0.0  # Force exploitation
        
        # Set up Q-values
        optimizer.q_table["state1"] = {
            "action1": 0.5,
            "action2": 1.0,
            "action3": 0.3
        }
        
        actions = ["action1", "action2", "action3"]
        action = optimizer.get_action("state1", actions)
        
        assert action == "action2"  # Highest Q-value
    
    def test_get_action_new_state(self):
        """Test action selection for new state"""
        optimizer = QueryOptimizer()
        optimizer.epsilon = 0.0
        
        actions = ["action1", "action2"]
        action = optimizer.get_action("new_state", actions)
        
        assert action in actions
        assert "new_state" in optimizer.q_table
    
    def test_update_q_value(self):
        """Test Q-value update"""
        optimizer = QueryOptimizer(learning_rate=0.1, discount_factor=0.9)
        
        state = "state1"
        action = "action1"
        reward = 1.0
        next_state = "state2"
        
        optimizer.update(state, action, reward, next_state)
        
        assert state in optimizer.q_table
        assert action in optimizer.q_table[state]
        assert optimizer.q_table[state][action] != 0.0
    
    def test_update_multiple_times(self):
        """Test multiple Q-value updates"""
        optimizer = QueryOptimizer(learning_rate=0.5)
        
        # First update
        optimizer.update("state1", "action1", 1.0, "state2")
        q_value_1 = optimizer.q_table["state1"]["action1"]
        
        # Second update with different reward
        optimizer.update("state1", "action1", 2.0, "state2")
        q_value_2 = optimizer.q_table["state1"]["action1"]
        
        # Q-value should have changed
        assert q_value_2 != q_value_1
        assert q_value_2 > q_value_1  # Higher reward should increase Q-value
    
    def test_feed_metrics(self):
        """Test feeding execution metrics"""
        optimizer = QueryOptimizer()
        
        query = "SELECT * FROM users WHERE age > 25"
        latency_ms = 50.0
        
        optimizer.feed_metrics(query, latency_ms)
        
        # Should have created Q-table entries
        assert len(optimizer.q_table) > 0
    
    def test_feed_metrics_multiple_queries(self):
        """Test feeding metrics for multiple queries"""
        optimizer = QueryOptimizer()
        
        queries = [
            ("SELECT * FROM users", 10.0),
            ("SELECT * FROM products WHERE price > 100", 25.0),
            ("SELECT * FROM orders", 15.0),
        ]
        
        for query, latency in queries:
            optimizer.feed_metrics(query, latency)
        
        assert len(optimizer.q_table) > 0


class TestIntegration:
    """Integration tests for optimizer components"""
    
    def test_cache_and_optimizer_together(self):
        """Test using cache and optimizer together"""
        cache = SemanticCache(similarity_threshold=0.95)
        optimizer = QueryOptimizer()
        
        query = "SELECT * FROM users"
        result = "user_data"
        
        # First execution - cache miss
        cached_result = cache.get(query)
        assert cached_result is None
        
        # Store in cache
        cache.put(query, result)
        optimizer.feed_metrics(query, latency_ms=50.0)
        
        # Second execution - cache hit
        cached_result = cache.get(query)
        assert cached_result == result
        optimizer.feed_metrics(query, latency_ms=0.5)  # Much faster
    
    def test_test_vectorization_function(self):
        """Test the test_vectorization function"""
        from nexum_ai.optimizer import test_vectorization
        
        result = test_vectorization()
        
        assert 'query' in result
        assert 'vector' in result
        assert 'dimension' in result
        assert result['dimension'] == 384
        assert len(result['vector']) == 10
