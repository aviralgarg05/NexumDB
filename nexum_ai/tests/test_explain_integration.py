#!/usr/bin/env python3
"""
Comprehensive integration tests for EXPLAIN query plan feature (Issue #66)
Tests the complete flow: parsing -> cache analysis -> RL agent -> formatting
"""

import sys
import os
import tempfile
import json

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from nexum_ai.optimizer import (
    explain_query_plan,
    format_explain_output,
    SemanticCache,
    QueryOptimizer
)
from nexum_ai.rl_agent import QLearningAgent


class TestExplainIntegration:
    """Comprehensive integration tests for EXPLAIN feature"""
    
    def setup_method(self):
        """Setup: Create temporary directory for test caches to avoid disk artifacts"""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_cache_file = os.environ.get('NEXUMDB_CACHE_FILE')
        # Point cache operations to temporary directory for test isolation
        os.environ['NEXUMDB_CACHE_FILE'] = os.path.join(self.temp_dir.name, 'test_cache.pkl')
    
    def teardown_method(self):
        """Teardown: Clean up temporary directory and restore cache location"""
        # Clean up temporary directory and all cache artifacts
        self.temp_dir.cleanup()
        # Restore original cache location
        if self.original_cache_file is None:
            os.environ.pop('NEXUMDB_CACHE_FILE', None)
        else:
            os.environ['NEXUMDB_CACHE_FILE'] = self.original_cache_file
    
    def test_explain_select_query_with_where(self):
        """Test EXPLAIN output for SELECT with WHERE clause"""
        query = "SELECT * FROM users WHERE age > 25"
        result = explain_query_plan(query)
        
        # Verify all required components are present
        assert 'query' in result
        assert 'parsing' in result
        assert 'cache_analysis' in result
        assert 'rl_agent' in result
        assert 'execution_strategy' in result
        
        # Verify parsing information
        parsing = result['parsing']
        assert parsing['query_type'] == 'SELECT'
        assert parsing['has_where_clause'] is True
        assert parsing['complexity_estimate'] >= 0
        
        # Verify RL agent information includes Q-values
        rl_info = result['rl_agent']
        assert 'q_values' in rl_info
        assert 'best_action' in rl_info
        assert 'state' in rl_info
        
        # Verify execution strategy
        strategy = result['execution_strategy']
        assert 'strategy' in strategy
        assert 'estimated_latency' in strategy
        assert strategy['strategy'] in ['CACHE_HIT', 'CACHE_MISS_THEN_STORE', 'INDEX_SCAN', 'FULL_SCAN']
    
    def test_explain_insert_query(self):
        """Test EXPLAIN output for INSERT statement"""
        query = "INSERT INTO employees (name, salary) VALUES ('John', 50000)"
        result = explain_query_plan(query)
        
        parsing = result['parsing']
        assert parsing['query_type'] == 'INSERT'
        assert parsing['has_where_clause'] is False
        
        # INSERT shouldn't be cached
        strategy = result['execution_strategy']
        assert strategy['will_cache_result'] is False
    
    def test_explain_update_query(self):
        """Test EXPLAIN output for UPDATE statement"""
        query = "UPDATE products SET price = 99 WHERE id = 10"
        result = explain_query_plan(query)
        
        parsing = result['parsing']
        assert parsing['query_type'] == 'UPDATE'
        assert parsing['has_where_clause'] is True
        
        # UPDATE shouldn't be cached
        strategy = result['execution_strategy']
        assert strategy['will_cache_result'] is False
    
    def test_explain_delete_query(self):
        """Test EXPLAIN output for DELETE statement"""
        query = "DELETE FROM logs WHERE timestamp < '2024-01-01'"
        result = explain_query_plan(query)
        
        parsing = result['parsing']
        assert parsing['query_type'] == 'DELETE'
        assert parsing['has_where_clause'] is True
        
        # DELETE shouldn't be cached
        strategy = result['execution_strategy']
        assert strategy['will_cache_result'] is False
    
    def test_explain_create_table_query(self):
        """Test EXPLAIN output for CREATE TABLE statement"""
        query = "CREATE TABLE users (id INTEGER, name TEXT, age INTEGER)"
        result = explain_query_plan(query)
        
        parsing = result['parsing']
        assert parsing['query_type'] == 'CREATE'
    
    def test_explain_cache_hit_scenario(self):
        """Test EXPLAIN with cache hit scenario"""
        cache = SemanticCache(similarity_threshold=0.85)
        
        # Pre-populate cache
        query1 = "SELECT * FROM users WHERE age > 25"
        result1 = "User records for age > 25"
        cache.put(query1, result1)
        
        # Query similar to cached one
        query2 = "SELECT * FROM users WHERE age > 30"
        explain_result = explain_query_plan(query2, cache)
        
        # Should detect high similarity
        cache_analysis = explain_result['cache_analysis']
        assert cache_analysis['top_matches'] is not None
        assert len(cache_analysis['top_matches']) > 0
        
        # Formatted output should show cache information
        formatted = format_explain_output(explain_result)
        assert 'CACHE LOOKUP' in formatted
        assert 'PARSING' in formatted
    
    def test_explain_cache_miss_scenario(self):
        """Test EXPLAIN with cache miss scenario"""
        cache = SemanticCache(similarity_threshold=0.95)
        
        # Pre-populate cache with unrelated query
        cache.put("SELECT * FROM products WHERE price < 100", "Product data")
        
        # Query completely different from cached
        query = "SELECT COUNT(*) FROM orders WHERE status = 'active'"
        explain_result = explain_query_plan(query, cache)
        
        # Should not hit cache
        cache_analysis = explain_result['cache_analysis']
        assert cache_analysis['would_hit_cache'] is False
    
    def test_explain_output_formatting_completeness(self):
        """Test that formatted EXPLAIN output contains all required sections"""
        query = "SELECT * FROM users WHERE age > 25 ORDER BY name LIMIT 10"
        explain_result = explain_query_plan(query)
        formatted = format_explain_output(explain_result)
        
        # Verify all major sections are present
        required_sections = [
            'QUERY EXECUTION PLAN',
            'PARSING',
            'CACHE LOOKUP',
            'RL AGENT',
            'EXECUTION STRATEGY'
        ]
        
        for section in required_sections:
            assert section in formatted, f"Missing section: {section}"
        
        # Verify specific information is formatted
        assert 'SELECT' in formatted  # Query type
        assert 'Q-values' in formatted  # RL agent info
        
        # Verify no excessive truncation
        lines = formatted.split('\n')
        for line in lines:
            assert len(line) <= 75, f"Line too long: {line}"
    
    def test_explain_rl_agent_q_values_display(self):
        """Test that RL agent Q-values are properly displayed"""
        agent = QLearningAgent()
        query_length = 55  # "SELECT * FROM products WHERE price BETWEEN 50 AND 100"
        cache_hit = False
        complexity = 4
        
        explain_result = agent.explain_action(query_length, cache_hit, complexity)
        
        # Verify Q-values structure
        assert 'q_values' in explain_result
        assert isinstance(explain_result['q_values'], dict)
        
        # All actions should have Q-values
        for action in agent.actions:
            assert action in explain_result['q_values']
            assert isinstance(explain_result['q_values'][action], (int, float))
        
        # Best action should be the one with highest Q-value
        best_q = max(explain_result['q_values'].values())
        best_action_q = explain_result['q_values'][explain_result['best_action']]
        assert best_action_q == best_q
    
    def test_explain_query_complexity_estimation(self):
        """Test complexity estimation for various queries including boundary case"""
        # Complexity formula: min(len(query) // 20, 10)
        # Tests include: low, mid, high, and upper-bound boundary case
        test_cases = [
            ("SELECT *", "SELECT", 0),  # 8 chars → 0
            ("SELECT id FROM users WHERE active", "SELECT", 1),  # 34 chars → 1
            ("SELECT a, b, c FROM users_table WHERE condition_a = 1 AND condition_b = 2 AND condition_c = 3", "SELECT", 4),  # 97 chars → 4
            # Boundary case: 200+ char query should clamp to 10
            ("SELECT id, name, email, phone, address, city, state, zip FROM customers WHERE status = 'active' AND created_date > '2024-01-01' AND account_balance > 100.00 AND subscription_level IN ('premium', 'enterprise') ORDER BY created_date DESC LIMIT 50", "SELECT", 10),  # 232 chars → 10 (clamped)
        ]
        
        for query, expected_type, expected_complexity in test_cases:
            result = explain_query_plan(query)
            parsing = result['parsing']
            actual_complexity = parsing['complexity_estimate']
            
            assert parsing['query_type'] == expected_type, f"Expected type {expected_type}, got {parsing['query_type']}"
            assert actual_complexity == expected_complexity, f"Expected complexity {expected_complexity}, got {actual_complexity} for query ({len(query)} chars): {query}"
    
    def test_explain_aggregation_detection(self):
        """Test detection of aggregation functions"""
        agg_queries = [
            ("SELECT COUNT(*) FROM orders", True),
            ("SELECT SUM(price) FROM products", True),
            ("SELECT AVG(salary) FROM employees", True),
            ("SELECT MAX(date) FROM events", True),
            ("SELECT MIN(score) FROM results", True),
            ("SELECT * FROM users", False),
        ]
        
        for query, should_have_agg in agg_queries:
            result = explain_query_plan(query)
            parsing = result['parsing']
            assert parsing['has_aggregation'] == should_have_agg
    
    def test_explain_join_detection(self):
        """Test detection of JOIN clauses"""
        join_queries = [
            ("SELECT * FROM users JOIN orders ON users.id = orders.user_id", True),
            ("SELECT * FROM a LEFT JOIN b ON a.id = b.aid", True),
            ("SELECT * FROM products WHERE category = 'electronics'", False),
        ]
        
        for query, should_have_join in join_queries:
            result = explain_query_plan(query)
            parsing = result['parsing']
            assert parsing['has_join'] == should_have_join
    
    def test_explain_order_by_detection(self):
        """Test detection of ORDER BY clauses"""
        order_queries = [
            ("SELECT * FROM users ORDER BY name", True),
            ("SELECT * FROM products ORDER BY price DESC, date ASC", True),
            ("SELECT * FROM users WHERE active = true", False),
        ]
        
        for query, should_have_order in order_queries:
            result = explain_query_plan(query)
            parsing = result['parsing']
            assert parsing['has_order_by'] == should_have_order
    
    def test_explain_group_by_detection(self):
        """Test detection of GROUP BY clauses"""
        group_queries = [
            ("SELECT category, COUNT(*) FROM products GROUP BY category", True),
            ("SELECT user_id, SUM(amount) FROM transactions GROUP BY user_id", True),
            ("SELECT * FROM sales WHERE region = 'East'", False),
        ]
        
        for query, should_have_group in group_queries:
            result = explain_query_plan(query)
            parsing = result['parsing']
            assert parsing['has_group_by'] == should_have_group
    
    def test_explain_medication_queries_comprehensive(self):
        """Test EXPLAIN for medical/healthcare domain queries with RL Agent validation"""
        medication_query = "SELECT medication_id, drug_name, dosage FROM medications WHERE status = 'active' AND patient_count > 1000"
        result = explain_query_plan(medication_query)
        
        # Verify parsing section
        parsing = result['parsing']
        assert parsing['query_type'] == 'SELECT'
        assert parsing['has_where_clause'] is True
        assert 'complexity_estimate' in parsing
        
        # CRITICAL: Verify RL_AGENT section is properly populated (this was Issue #11)
        rl_agent = result['rl_agent']
        assert 'q_values' in rl_agent, "RL Agent must contain Q-values"
        assert 'best_action' in rl_agent, "RL Agent must have best action"
        assert 'state' in rl_agent, "RL Agent must include state information"
        assert isinstance(rl_agent['q_values'], dict), "Q-values must be a dictionary"
        assert len(rl_agent['q_values']) > 0, "Q-values dictionary cannot be empty"
        
        # CRITICAL: Verify EXECUTION_STRATEGY section is present and complete
        strategy = result['execution_strategy']
        assert 'strategy' in strategy, "Strategy must specify execution approach"
        assert 'estimated_latency' in strategy, "Strategy must include latency estimate"
        assert 'will_cache_result' in strategy, "Strategy must indicate caching intent"
        assert strategy['strategy'] in ['CACHE_HIT', 'CACHE_MISS_THEN_STORE', 'INDEX_SCAN', 'FULL_SCAN'], \
               f"Invalid strategy: {strategy['strategy']}"
    
    def test_explain_cache_stats_consistency(self):
        """Test that cache statistics are consistent across multiple EXPLAIN calls"""
        cache = SemanticCache()
        
        query1 = "SELECT * FROM users"
        query2 = "SELECT * FROM products"
        
        # First EXPLAIN
        result1 = explain_query_plan(query1, cache)
        stats1 = result1['cache_analysis']
        total_entries_1 = stats1['cache_entries_checked']
        
        # Add entry to cache
        cache.put(query1, "result1")
        
        # Second EXPLAIN
        result2 = explain_query_plan(query2, cache)
        stats2 = result2['cache_analysis']
        total_entries_2 = stats2['cache_entries_checked']
        
        # Should now have one cached entry
        assert total_entries_2 > total_entries_1
    
    def test_explain_long_query_truncation(self):
        """Test that very long queries are truncated in display output"""
        long_query = "SELECT * FROM users WHERE " + " AND ".join([f"field{i} = {i}" for i in range(50)])
        
        explain_result = explain_query_plan(long_query)
        formatted = format_explain_output(explain_result)
        
        # Query should be present but truncated
        lines = formatted.split('\n')
        query_lines = [l for l in lines if l.startswith('Query:')]
        
        assert len(query_lines) > 0
        # Formatted query in output should be truncated (not full length)
        assert len(query_lines[0]) < len(long_query)
    
    def test_explain_with_custom_optimizer(self):
        """Test EXPLAIN with custom QueryOptimizer instance"""
        optimizer = QueryOptimizer()
        query = "SELECT AVG(price) FROM products WHERE stock > 0"
        
        explain_result = explain_query_plan(query, optimizer=optimizer)
        
        # Should have RL agent information from optimizer
        assert 'rl_agent' in explain_result
        assert 'best_action' in explain_result['rl_agent']


def run_all_tests():
    """Run all integration tests"""
    test_class = TestExplainIntegration()
    test_methods = [method for method in dir(test_class) if method.startswith('test_')]
    
    print(f"\n{'=' * 70}")
    print(f"Running {len(test_methods)} EXPLAIN integration tests")
    print(f"{'=' * 70}\n")
    
    passed = 0
    failed = 0
    
    for method_name in test_methods:
        try:
            method = getattr(test_class, method_name)
            method()
            print(f"✓ {method_name}")
            passed += 1
        except AssertionError as e:
            print(f"✗ {method_name}: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {method_name}: Unexpected error: {e}")
            failed += 1
    
    print(f"\n{'=' * 70}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'=' * 70}\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
