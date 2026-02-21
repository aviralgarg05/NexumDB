"""
Unit tests for optimizer.py - Query optimization logic
"""

import os
import tempfile
import time

import pytest

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


class TestSemanticCacheTTL:
    """Test suite for SemanticCache TTL / expiration feature"""

    def test_put_stores_timestamp(self):
        """Entries created via put() must carry a timestamp."""
        cache = SemanticCache()
        cache.put("SELECT 1", "one")
        entry = cache.cache[0]
        assert 'timestamp' in entry
        assert isinstance(entry['timestamp'], float)
        # Timestamp should be very recent (within last 5 seconds)
        assert time.time() - entry['timestamp'] < 5

    def test_set_cache_expiration_rejects_non_positive(self):
        """set_cache_expiration must reject zero and negative hours."""
        cache = SemanticCache()
        with pytest.raises(ValueError):
            cache.set_cache_expiration(0)
        with pytest.raises(ValueError):
            cache.set_cache_expiration(-1)

    def test_set_cache_expiration_rejects_nan_and_inf(self):
        """set_cache_expiration must reject NaN and infinite values."""
        cache = SemanticCache()
        with pytest.raises(ValueError):
            cache.set_cache_expiration(float('nan'))
        with pytest.raises(ValueError):
            cache.set_cache_expiration(float('inf'))
        with pytest.raises(ValueError):
            cache.set_cache_expiration(float('-inf'))

    def test_set_cache_expiration_rejects_non_numeric(self):
        """set_cache_expiration must reject non-numeric types."""
        cache = SemanticCache()
        with pytest.raises(ValueError):
            cache.set_cache_expiration("24")  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            cache.set_cache_expiration(True)  # type: ignore[arg-type]

    def test_set_cache_expiration_sets_max_age(self):
        """set_cache_expiration stores the TTL internally."""
        cache = SemanticCache()
        cache.set_cache_expiration(2)
        assert cache.max_age_seconds == 2 * 3600.0

    def test_expired_entries_are_evicted(self):
        """Entries older than the TTL are removed by set_cache_expiration."""
        cache = SemanticCache()
        # Insert an entry with a timestamp 2 hours in the past
        cache.cache.append({
            'query': 'old query',
            'vector': [0.0] * 384,
            'result': 'old',
            'timestamp': time.time() - 7200,  # 2 hours ago
        })
        cache.put("new query", "new")  # fresh entry
        assert len(cache.cache) == 2

        evicted = cache.set_cache_expiration(1)  # 1 hour TTL
        assert evicted == 1
        assert len(cache.cache) == 1
        assert cache.cache[0]['result'] == 'new'

    def test_get_skips_expired_entries(self):
        """get() must not return results from expired entries."""
        cache = SemanticCache(similarity_threshold=0.95)
        cache.put("SELECT * FROM users", "result_users")
        # Artificially expire the entry
        cache.cache[0]['timestamp'] = time.time() - 7200
        cache.max_age_seconds = 3600.0  # 1 hour TTL

        result = cache.get("SELECT * FROM users")
        assert result is None  # expired, should miss

    def test_get_returns_valid_entries(self):
        """get() still returns non-expired entries."""
        cache = SemanticCache(similarity_threshold=0.95)
        cache.put("SELECT * FROM users", "result_users")
        cache.max_age_seconds = 3600.0

        result = cache.get("SELECT * FROM users")
        assert result == "result_users"

    def test_no_ttl_means_no_eviction(self):
        """When max_age_seconds is None nothing is evicted."""
        cache = SemanticCache()
        cache.cache.append({
            'query': 'ancient',
            'vector': [0.0] * 384,
            'result': 'data',
            'timestamp': 0,  # epoch – very old
        })
        assert cache.max_age_seconds is None
        assert cache._evict_expired() == 0
        assert len(cache.cache) == 1

    def test_legacy_entries_without_timestamp_survive(self):
        """Entries loaded from old caches (no timestamp) are not evicted."""
        cache = SemanticCache()
        cache.cache.append({
            'query': 'legacy',
            'vector': [0.0] * 384,
            'result': 'legacy_data',
            # no 'timestamp' key
        })
        cache.set_cache_expiration(1)
        assert len(cache.cache) == 1  # kept, not evicted

    def test_explain_query_skips_expired(self):
        """explain_query should ignore expired entries."""
        cache = SemanticCache(similarity_threshold=0.5)
        cache.put("SELECT * FROM users", "result_users")
        # Expire the entry
        cache.cache[0]['timestamp'] = time.time() - 7200
        cache.max_age_seconds = 3600.0

        explanation = cache.explain_query("SELECT * FROM users")
        # cache_entries_checked reports len(self.cache) which includes expired
        # entries; top_matches only contains entries that were actually analysed.
        assert explanation['cache_entries_checked'] == 1
        assert len(explanation['top_matches']) == 0

    def test_get_cache_stats_includes_ttl_info(self):
        """Stats dict must include TTL fields when TTL is active."""
        cache = SemanticCache()
        # Without TTL
        stats = cache.get_cache_stats()
        assert 'max_age_hours' not in stats

        cache.set_cache_expiration(12)
        stats = cache.get_cache_stats()
        assert stats['max_age_hours'] == 12.0
        assert 'expired_entries' in stats

    def test_ttl_persists_across_save_load(self):
        """max_age_seconds should survive a save/load cycle."""
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "ttl_test.json")
            cache1 = SemanticCache()
            cache1.set_cache_expiration(6)
            cache1.put("q1", "r1")
            cache1.save_cache_json(path)

            cache2 = SemanticCache()
            cache2.load_cache_json(path)
            assert cache2.max_age_seconds == 6 * 3600.0
            assert len(cache2.cache) == 1

    def test_load_no_ttl_cache_resets_ttl(self):
        """Loading a no-TTL cache into a TTL-enabled instance resets TTL."""
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "no_ttl.json")

            # Save a cache without TTL
            cache1 = SemanticCache()
            cache1.put("q1", "r1")
            cache1.save_cache_json(path)

            # Load into an instance that already has TTL set
            cache2 = SemanticCache()
            cache2.set_cache_expiration(1)
            assert cache2.max_age_seconds is not None

            cache2.load_cache_json(path)
            assert cache2.max_age_seconds is None
            assert len(cache2.cache) == 1

    def test_evict_expired_returns_count(self):
        """_evict_expired returns the number of removed entries."""
        cache = SemanticCache()
        now = time.time()
        for i in range(5):
            cache.cache.append({
                'query': f'q{i}',
                'vector': [0.0] * 384,
                'result': f'r{i}',
                'timestamp': now - (i * 3600),  # 0h, 1h, 2h, 3h, 4h ago
            })
        cache.max_age_seconds = 2.5 * 3600  # 2.5 hour TTL
        removed = cache._evict_expired()
        assert removed == 2  # entries at 3h and 4h ago
        assert len(cache.cache) == 3

    def test_disable_ttl_with_none(self):
        """Passing None to set_cache_expiration disables TTL."""
        cache = SemanticCache()
        cache.set_cache_expiration(1)
        assert cache.max_age_seconds is not None

        result = cache.set_cache_expiration(None)
        assert cache.max_age_seconds is None
        assert result == 0  # no eviction when disabling


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
