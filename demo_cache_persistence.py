#!/usr/bin/env python3
"""
Demo script for semantic cache persistence functionality
Demonstrates cache persistence across application restarts
"""

import sys
import os
import time
from pathlib import Path

# Add nexum_ai to path
sys.path.insert(0, str(Path(__file__).parent))

from nexum_ai.optimizer import SemanticCache


def demo_cache_persistence():
    """Demonstrate semantic cache persistence functionality"""
    
    print("🚀 NexumDB Semantic Cache Persistence Demo")
    print("=" * 60)
    
    # Demo 1: First session - populate cache
    print("\n📝 Session 1: Populating semantic cache...")
    cache1 = SemanticCache(cache_file="demo_cache.pkl")
    
    # Simulate typical database queries
    demo_queries = [
        ("SELECT * FROM users WHERE age > 25", "Found 42 users older than 25"),
        ("SELECT name, email FROM customers WHERE city = 'New York'", "NYC customers: Alice, Bob, Charlie"),
        ("SELECT COUNT(*) FROM orders WHERE status = 'pending'", "Pending orders: 15"),
        ("SELECT product_name FROM inventory WHERE stock < 10", "Low stock items: Widget A, Gadget B"),
        ("SELECT AVG(price) FROM products WHERE category = 'electronics'", "Average electronics price: $299.99")
    ]
    
    for i, (query, result) in enumerate(demo_queries, 1):
        print(f"  {i}. Caching: {query[:40]}...")
        cache1.put(query, result)
        time.sleep(0.1)  # Simulate processing time
    
    # Save cache after adding all entries
    cache1.save_cache()
    
    stats1 = cache1.get_cache_stats()
    print(f"\n📊 Cache populated: {stats1['total_entries']} entries")
    print(f"   Cache file: {stats1['cache_file']}")
    print(f"   File size: {stats1['cache_size_bytes']} bytes")
    
    # Demo 2: Simulate application restart
    print("\n🔄 Simulating application restart...")
    print("   (Creating new cache instance)")
    del cache1  # Simulate application shutdown
    
    # Demo 3: Second session - load from disk
    print("\n📂 Session 2: Loading cache from disk...")
    cache2 = SemanticCache(cache_file="demo_cache.pkl")
    
    stats2 = cache2.get_cache_stats()
    print(f"   ✅ Cache loaded: {stats2['total_entries']} entries")
    
    # Demo 4: Test cache hits
    print("\n🎯 Testing cache hits after restart...")
    hit_count = 0
    
    for i, (query, expected_result) in enumerate(demo_queries, 1):
        cached_result = cache2.get(query)
        if cached_result:
            hit_count += 1
            print(f"  ✅ Hit {i}: {query[:35]}...")
            print(f"      Result: {cached_result}")
        else:
            print(f"  ❌ Miss {i}: {query[:35]}...")
    
    print(f"\n📈 Cache Performance:")
    print(f"   Cache hits: {hit_count}/{len(demo_queries)}")
    print(f"   Hit rate: {(hit_count/len(demo_queries)*100):.1f}%")
    
    # Demo 5: Test semantic similarity
    print("\n🧠 Testing semantic similarity...")
    similar_queries = [
        "SELECT * FROM users WHERE age > 25",  # Exact match
        "SELECT * FROM users WHERE age >= 26", # Similar query
        "Show me users older than 25",         # Natural language variant
    ]
    
    for query in similar_queries:
        result = cache2.get(query)
        if result:
            print(f"  ✅ Similar hit: {query}")
        else:
            print(f"  ❌ No match: {query}")
    
    # Demo 6: Export to JSON
    print("\n💾 Exporting cache to JSON format...")
    cache2.save_cache_json("demo_cache.json")
    
    # Demo 7: Cache management
    print("\n🔧 Cache management features:")
    print(f"   Current threshold: {cache2.similarity_threshold}")
    print("   Optimizing cache (keeping last 3 entries)...")
    cache2.optimize_cache(max_entries=3)
    
    final_stats = cache2.get_cache_stats()
    print(f"   Optimized cache: {final_stats['total_entries']} entries")
    
    # Demo 8: Environment variable configuration
    print("\n⚙️  Environment variable configuration:")
    print("   Set NEXUMDB_CACHE_FILE to customize cache location")
    print("   Example: export NEXUMDB_CACHE_FILE=my_custom_cache.pkl")
    
    # Cleanup
    print("\n🧹 Cleaning up demo files...")
    cache2.clear()
    
    # Remove JSON file if it exists
    json_file = Path("demo_cache.json")
    if json_file.exists():
        json_file.unlink()
        print("   Removed demo_cache.json")
    
    print("\n✨ Demo completed successfully!")
    print("\n💡 Key Benefits:")
    print("   • Cache persists across application restarts")
    print("   • Configurable cache file location")
    print("   • JSON export for debugging/analysis")
    print("   • Automatic cache optimization")
    print("   • Semantic similarity matching")


if __name__ == "__main__":
    try:
        demo_cache_persistence()
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        sys.exit(1)