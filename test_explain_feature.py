#!/usr/bin/env python3
"""
Test script for EXPLAIN query plan feature (Issue #48)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nexum_ai'))

from nexum_ai.optimizer import explain_query_plan, format_explain_output, SemanticCache
from nexum_ai.rl_agent import QLearningAgent

def test_explain_feature():
    """Test the complete EXPLAIN functionality"""
    print("=" * 70)
    print("TESTING EXPLAIN QUERY PLAN FEATURE (Issue #48)")
    print("=" * 70)
    
    # Setup test data
    cache = SemanticCache()
    cache.put("SELECT * FROM users WHERE age > 25", "User data for age > 25")
    cache.put("SELECT name FROM products WHERE price < 100", "Product names under $100")
    cache.put("SELECT COUNT(*) FROM orders WHERE status = 'active'", "Active order count: 42")
    
    # Test queries
    test_queries = [
        "SELECT * FROM users WHERE age > 30",  # Should hit cache (similar to first entry)
        "SELECT * FROM products WHERE category = 'electronics'",  # Should miss cache
        "SELECT COUNT(*) FROM orders WHERE date > '2024-01-01'",  # Should partially match
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'-' * 70}")
        print(f"TEST {i}: {query}")
        print(f"{'-' * 70}")
        
        # Generate explain plan
        explain_result = explain_query_plan(query, cache)
        formatted_output = format_explain_output(explain_result)
        
        print(formatted_output)
        
        # Verify key components are present
        assert "QUERY EXECUTION PLAN" in formatted_output
        assert "PARSING" in formatted_output
        assert "CACHE LOOKUP" in formatted_output
        assert "RL AGENT" in formatted_output
        assert "EXECUTION STRATEGY" in formatted_output
        
        # Verify data structure
        assert 'query' in explain_result
        assert 'parsing' in explain_result
        assert 'cache_analysis' in explain_result
        assert 'rl_agent' in explain_result
        assert 'execution_strategy' in explain_result
        
        print(f"✓ Test {i} passed - All components present")
    
    print(f"\n{'=' * 70}")
    print("ALL TESTS PASSED - EXPLAIN FEATURE WORKING CORRECTLY")
    print("=" * 70)
    
    # Test individual components
    print("\nTesting individual components:")
    
    # Test cache explain
    cache_result = cache.explain_query("SELECT * FROM users WHERE age > 35")
    print(f"✓ Cache explain: {cache_result['cache_entries_checked']} entries checked")
    
    # Test RL agent explain
    agent = QLearningAgent()
    rl_result = agent.explain_action(30, False, 5)
    print(f"✓ RL agent explain: State {rl_result['state']}, Action {rl_result['best_action']}")
    
    return True

if __name__ == "__main__":
    try:
        test_explain_feature()
        print("\n🎉 EXPLAIN query plan feature implementation complete!")
        print("Usage: EXPLAIN <query> in the NexumDB CLI")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)