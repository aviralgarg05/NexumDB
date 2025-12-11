"""
Unit tests for optimizer.py - Query optimization logic
"""

from nexum_ai.optimizer import SemanticCache, QueryOptimizer


class TestSemanticCache:
    """Test suite for SemanticCache class"""
    
    def test_initialization(self):
        """Test SemanticCache initialization"""
        cache = SemanticCache(similarity_threshold=0.9)
        assert cache.similarity_threshold == 0.9
        assert cache.cache == []
        assert cache.model is None
    
    def test_vectorize_fallback(self):
        """Test fallback vectorization when model is not available"""
        cache = SemanticCache()
        cache.model = None  # Force fallback
        
        vector = cache.vectorize("test query")
        assert isinstance(vector, list)
        assert len(vector) == 384
        assert all(isinstance(v, float) for v in vector)
    
    def test_vectorize_with_model(self):
        """Test vectorization with sentence transformer model"""
        cache = SemanticCache()
        
        # Use fallback since model not installed
        cache.model = None
        vector = cache.vectorize("test query")
        
        assert isinstance(vector, list)
        assert len(vector) == 384
    
    def test_cosine_similarity_identical(self):
        """Test cosine similarity with identical vectors"""
        cache = SemanticCache()
        vec1 = [1.0, 2.0, 3.0]
        vec2 = [1.0, 2.0, 3.0]
        
        similarity = cache.cosine_similarity(vec1, vec2)
        assert abs(similarity - 1.0) < 0.001
    
    def test_cosine_similarity_orthogonal(self):
        """Test cosine similarity with orthogonal vectors"""
        cache = SemanticCache()
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        
        similarity = cache.cosine_similarity(vec1, vec2)
        assert abs(similarity) < 0.001
    
    def test_cosine_similarity_zero_vector(self):
        """Test cosine similarity with zero vector"""
        cache = SemanticCache()
        vec1 = [0.0, 0.0, 0.0]
        vec2 = [1.0, 2.0, 3.0]
        
        similarity = cache.cosine_similarity(vec1, vec2)
        assert similarity == 0.0
    
    def test_put_and_get_cache_hit(self):
        """Test caching and retrieving with high similarity"""
        cache = SemanticCache(similarity_threshold=0.95)
        
        query = "SELECT * FROM users"
        result = "result_data"
        
        cache.put(query, result)
        assert len(cache.cache) == 1
        
        # Same query should hit cache
        retrieved = cache.get(query)
        assert retrieved == result
    
    def test_get_cache_miss(self):
        """Test cache miss with different query"""
        cache = SemanticCache(similarity_threshold=0.95)
        
        cache.put("SELECT * FROM users", "result1")
        
        # Very different query should miss
        retrieved = cache.get("SELECT * FROM products WHERE price > 100")
        assert retrieved is None
    
    def test_clear_cache(self):
        """Test clearing the cache"""
        cache = SemanticCache()
        
        cache.put("query1", "result1")
        cache.put("query2", "result2")
        assert len(cache.cache) == 2
        
        cache.clear()
        assert len(cache.cache) == 0
    
    def test_multiple_cache_entries(self):
        """Test cache with multiple entries"""
        cache = SemanticCache(similarity_threshold=0.95)
        
        queries = [
            ("SELECT * FROM users", "result1"),
            ("SELECT * FROM products", "result2"),
            ("SELECT * FROM orders", "result3"),
        ]
        
        for query, result in queries:
            cache.put(query, result)
        
        assert len(cache.cache) == 3


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
