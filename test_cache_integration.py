#!/usr/bin/env python3
"""
Integration test for semantic cache persistence
Tests the complete cache lifecycle without requiring Rust compilation
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path

# Add nexum_ai to path
sys.path.insert(0, str(Path(__file__).parent))

from nexum_ai.optimizer import SemanticCache


def test_cache_persistence_lifecycle():
    """Test complete cache persistence lifecycle"""
    
    print("🧪 Testing Semantic Cache Persistence Lifecycle")
    print("=" * 60)
    
    # Use temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_file = os.path.join(temp_dir, "test_cache.pkl")
        
        # Test 1: Create cache and populate
        print("\n1️⃣ Creating cache and adding entries...")
        cache1 = SemanticCache(cache_file=cache_file)
        
        test_data = [
            ("SELECT * FROM users WHERE active = true", "Active users: 150"),
            ("SELECT COUNT(*) FROM orders", "Total orders: 1,234"),
            ("SELECT name FROM products WHERE price > 100", "Premium products list")
        ]
        
        for query, result in test_data:
            cache1.put(query, result)
        
        # Save cache after adding entries
        cache1.save_cache()
        
        stats1 = cache1.get_cache_stats()
        print(f"   ✅ Added {stats1['total_entries']} entries")
        
        # Test 2: Verify file exists
        print("\n2️⃣ Verifying cache file creation...")
        assert os.path.exists(cache_file), "Cache file should exist"
        file_size = os.path.getsize(cache_file)
        print(f"   ✅ Cache file exists ({file_size} bytes)")
        
        # Test 3: Create new instance and verify loading
        print("\n3️⃣ Creating new cache instance...")
        cache2 = SemanticCache(cache_file=cache_file)
        
        stats2 = cache2.get_cache_stats()
        print(f"   ✅ Loaded {stats2['total_entries']} entries")
        
        assert stats1['total_entries'] == stats2['total_entries'], "Entry count should match"
        
        # Test 4: Verify cache hits
        print("\n4️⃣ Testing cache hits...")
        hit_count = 0
        for query, expected_result in test_data:
            cached_result = cache2.get(query)
            if cached_result == expected_result:
                hit_count += 1
                print(f"   ✅ Hit: {query[:30]}...")
            else:
                print(f"   ❌ Miss: {query[:30]}...")
        
        assert hit_count == len(test_data), f"Expected {len(test_data)} hits, got {hit_count}"
        
        # Test 5: Test JSON export
        print("\n5️⃣ Testing JSON export...")
        json_file = os.path.join(temp_dir, "test_cache.json")
        cache2.save_cache_json(json_file)
        
        assert os.path.exists(json_file), "JSON file should exist"
        print(f"   ✅ JSON export successful")
        
        # Test 6: Test cache optimization
        print("\n6️⃣ Testing cache optimization...")
        cache2.optimize_cache(max_entries=2)
        
        stats3 = cache2.get_cache_stats()
        assert stats3['total_entries'] == 2, "Should have 2 entries after optimization"
        print(f"   ✅ Cache optimized to {stats3['total_entries']} entries")
        
        # Test 7: Test cache clearing
        print("\n7️⃣ Testing cache clearing...")
        cache2.clear()
        
        assert not os.path.exists(cache_file), "Cache file should be deleted"
        print("   ✅ Cache cleared successfully")
        
        print("\n✨ All tests passed!")
        return True


def test_environment_variable_config():
    """Test environment variable configuration"""
    
    print("\n🌍 Testing Environment Variable Configuration")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        custom_cache = os.path.join(temp_dir, "env_cache.pkl")
        
        # Set environment variable
        os.environ['NEXUMDB_CACHE_FILE'] = custom_cache
        
        try:
            # Create cache (should use env variable)
            cache = SemanticCache()
            cache.put("SELECT 1", "test result")
            cache.save_cache()
            
            # Verify it used the custom path
            assert os.path.exists(custom_cache), "Should use environment variable path"
            print("   ✅ Environment variable configuration works")
            
            cache.clear()
            
        finally:
            # Clean up environment
            if 'NEXUMDB_CACHE_FILE' in os.environ:
                del os.environ['NEXUMDB_CACHE_FILE']
        
        return True


def test_error_handling():
    """Test error handling scenarios"""
    
    print("\n🛡️ Testing Error Handling")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test 1: Invalid cache file path
        print("\n1️⃣ Testing invalid cache file path...")
        try:
            invalid_path = os.path.join("/nonexistent/path", "cache.pkl")
            cache = SemanticCache(cache_file=invalid_path)
            cache.put("SELECT 1", "test")
            print("   ⚠️ Should handle invalid paths gracefully")
        except Exception as e:
            print(f"   ✅ Handled error gracefully: {type(e).__name__}")
        
        # Test 2: Corrupted cache file
        print("\n2️⃣ Testing corrupted cache file...")
        corrupted_file = os.path.join(temp_dir, "corrupted.pkl")
        
        # Create corrupted file
        with open(corrupted_file, 'w') as f:
            f.write("This is not a valid pickle file")
        
        cache = SemanticCache(cache_file=corrupted_file)
        # Should start with empty cache
        stats = cache.get_cache_stats()
        assert stats['total_entries'] == 0, "Should start with empty cache for corrupted file"
        print("   ✅ Handled corrupted cache file gracefully")
        
        cache.clear()
        
        return True


def main():
    """Run all integration tests"""
    
    print("🚀 NexumDB Semantic Cache Integration Tests")
    print("=" * 80)
    
    tests = [
        ("Cache Persistence Lifecycle", test_cache_persistence_lifecycle),
        ("Environment Variable Config", test_environment_variable_config),
        ("Error Handling", test_error_handling)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            print(f"\n📋 Running: {test_name}")
            if test_func():
                passed += 1
                print(f"✅ PASSED: {test_name}")
            else:
                failed += 1
                print(f"❌ FAILED: {test_name}")
        except Exception as e:
            failed += 1
            print(f"❌ FAILED: {test_name} - {e}")
    
    print("\n" + "=" * 80)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All integration tests passed!")
        return True
    else:
        print("💥 Some tests failed!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)