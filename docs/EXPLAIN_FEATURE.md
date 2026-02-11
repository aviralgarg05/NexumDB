# EXPLAIN Query Plan Feature

## Overview

The `EXPLAIN` command provides comprehensive visibility into how NexumDB executes and optimizes queries. It shows the parsed SQL structure, cache analysis, and RL agent decision-making process with Q-values.

**Status:** Issue #66 (Feature Implemented & Enhanced)  
**PR:** #131 (feat: Add EXPLAIN query plan command)

## Implementation Details

### Architecture

The EXPLAIN feature is implemented across multiple layers:

#### 1. Rust Layer (nexum_cli & nexum_bridges)
- **Location:** `nexum_cli/src/main.rs` (lines 162-180)
- **Component:** `QueryExplainer` struct in `nexum_core/src/bridge/mod.rs`
- **Responsibility:** Invoke Python explain functions and format output

```rust
// CLI Usage
if input.len() >= 8 && input[..8].eq_ignore_ascii_case("EXPLAIN ") {
    let query_to_explain = input[8..].trim();
    if let Some(ref explainer) = query_explainer {
        match explainer.explain(query_to_explain) {
            Ok(plan) => println!("{}", plan),
            Err(e) => print_error("Explain error", &e.to_string()),
        }
    }
}
```

#### 2. Python Layer (nexum_ai)
- **Files:**
  - `optimizer.py`: `explain_query_plan()` and `format_explain_output()`
  - `rl_agent.py`: `QLearningAgent.explain_action()`
  - `model_manager.py`: Model loading for semantic analysis

### Core Functions

#### `explain_query_plan(query, cache=None, optimizer=None)`

Generates complete query analysis combining:

**Inputs:**
- `query` (str): SQL query to explain
- `cache` (Optional[SemanticCache]): Semantic cache instance
- `optimizer` (Optional[QueryOptimizer]): RL optimizer instance

**Returns:** Dict with 5 major sections:

```python
{
    'query': 'SELECT * FROM users WHERE age > 25',
    'parsing': {
        'query_type': 'SELECT',              # Query type
        'query_length': 40,                  # Character count
        'complexity_estimate': 2,            # 0-10 scale
        'has_where_clause': True,            # Boolean
        'has_join': False,
        'has_aggregation': False,
        'has_order_by': False,
        'has_group_by': False
    },
    'cache_analysis': {
        'cache_entries_checked': 5,          # Total cache entries
        'similarity_threshold': 0.95,        # Matching threshold
        'best_similarity': 0.92,             # Best match score
        'would_hit_cache': False,            # Cache hit prediction
        'top_matches': [                     # Similar cached queries
            {
                'cached_query': 'SELECT * FROM users...',
                'similarity': 0.92,
                'would_hit': False
            }
        ]
    },
    'rl_agent': {
        'state': 'query_type_4',             # State for this query
        'q_values': {                        # Q-learning values
            'use_cache': 0.5,
            'bypass_cache': 1.5,
            'full_scan': 0.3,
            'index_scan': 0.8
        },
        'best_action': 'bypass_cache',       # Highest Q-value action
        'epsilon': 0.2,                      # Exploration rate
        'would_explore': True
    },
    'execution_strategy': {
        'strategy': 'FULL_SCAN',             # Chosen strategy
        'estimated_latency': '10-100ms',
        'will_cache_result': True,
        'recommendation': 'Execute and cache'
    }
}
```

#### `format_explain_output(explain_result)`

Formats the analysis into a terminal-friendly table with 5 sections:

```text
======================================================================
                       QUERY EXECUTION PLAN
======================================================================

Query: SELECT * FROM users WHERE age > 25

┌─ PARSING ──────────────────────────────────────────────────────────┐
│ Type: SELECT        Complexity: 2/10              │
│ WHERE: True     JOIN: False     AGG: False     │
└───────────────────────────────────────────────────────────────────┘

┌─ CACHE LOOKUP ────────────────────────────────────────────────────┐
│ Entries checked: 5     Threshold: 0.95            │
│ Best similarity: 0.92   Would hit: False              │
│ Top matches:                                                      │
│   ✗ 0.92   - SELECT * FROM users WHERE age > 30         │
└───────────────────────────────────────────────────────────────────┘

┌─ RL AGENT ────────────────────────────────────────────────────────┐
│ State: query_type_4            Epsilon: 0.2000        │
│ Best action: bypass_cache                              │
│ Q-values:                                                         │
│   use_cache       :     0.5000                                  │
│   bypass_cache    :     1.5000                                  │
│   full_scan       :     0.3000                                  │
│   index_scan      :     0.8000                                  │
└───────────────────────────────────────────────────────────────────┘

┌─ EXECUTION STRATEGY ───────────────────────────────────────────────┐
│ Strategy: FULL_SCAN            Est. latency: 10-100ms   │
│ Will cache: True                                                   │
│ Recommendation: Execute and cache                                  │
└───────────────────────────────────────────────────────────────────┘
```

### Output Sections

#### 1. PARSING
Shows SQL structure analysis:
- **Query Type:** SELECT, INSERT, UPDATE, DELETE, CREATE, or UNKNOWN
- **Complexity:** Estimated from query length (0-10 scale)
- **Clauses Detected:** WHERE, JOIN, aggregation functions, ORDER BY, GROUP BY

#### 2. CACHE LOOKUP
Semantic cache analysis:
- **Entries Checked:** Number of cached queries analyzed
- **Similarity Threshold:** Required score for cache hit (0.0-1.0)
- **Best Similarity:** Highest similarity score found
- **Would Hit Cache:** Boolean prediction of cache effectiveness
- **Top Matches:** List of most similar cached queries with similarity scores

#### 3. RL AGENT
Reinforcement learning decision information:
- **State:** Internal state representation (e.g., "query_type_4")
- **Best Action:** Action with highest Q-value (use_cache, bypass_cache, full_scan, index_scan)
- **Q-Values:** Q-learning values for all possible actions (higher = better)
- **Epsilon:** Exploration rate (0.0 = pure exploitation, 1.0 = pure exploration)
- **Would Explore:** Whether random action might be taken

#### 4. EXECUTION STRATEGY
Recommended execution plan:
- **Strategy:** CACHE_HIT, CACHE_MISS_THEN_STORE, INDEX_SCAN, or FULL_SCAN
- **Estimated Latency:** Expected execution time
- **Will Cache:** Whether result will be cached
- **Recommendation:** Human-readable guidance

## Usage Examples

### Interactive REPL

```bash
nexumdb> EXPLAIN SELECT * FROM users WHERE age > 25
[Shows complete query execution plan]

nexumdb> EXPLAIN INSERT INTO users (name, age) VALUES ('John', 30)
[Shows insert optimization strategy]

nexumdb> EXPLAIN SELECT COUNT(*) FROM orders WHERE status = 'active'
[Shows aggregation and cache analysis]
```

## Error Handling

### Input Validation
- Query must be non-empty string
- Empty/whitespace-only queries raise ValueError
- Invalid query types handled gracefully

### Fallback Mechanisms
- Missing Python environment: Returns informative error
- Cache unavailable: Continues with cache_analysis = {}
- RL Agent failure: Uses default actions with explanation
- Vectorization errors: Returns cache analysis without match scores

### Defensive Programming
- Field width limits prevent output overflow
- Query truncation at 60 characters
- Action names truncated to 15 characters
- Cache entries capped at 99,999 for display
- Epsilon value clamped to [0.0, 1.0] range

## Performance Characteristics

- **Vectorization:** O(D) where D = embedding dimension (~384)
- **Cache Lookup:** O(N) where N = number of cached queries
- **RL Agent Analysis:** O(1) Q-table lookup
- **Formatting:** O(M) where M = number of top matches (capped at 5)

**Total Time:** Typically 10-100ms per EXPLAIN call

## Testing

### Python Tests
- **File:** `nexum_ai/tests/test_explain_integration.py`
- **Coverage:** 17 comprehensive integration tests
  - Query type detection (SELECT, INSERT, UPDATE, DELETE, CREATE)
  - Cache hit/miss scenarios
  - Aggregation and JOIN detection
  - Query complexity estimation
  - Output formatting validation
  - Long query truncation
  - Custom optimizer instances
  - Q-values display and consistency

### Rust Tests
- **File:** `nexum_core/src/bridge/mod.rs`
- **Coverage:** 5 comprehensive tests
  - Basic QueryExplainer functionality
  - Multiple SELECT query variations
  - Mutation queries (INSERT, UPDATE, DELETE)
  - Raw JSON output validation
  - Q-values presence verification

### Integration Tests
- **File:** `tests/integration_test.rs`
- End-to-end CLI usage validation

## Requirements Met

✅ **Requirement 1: EXPLAIN Command**
- Implemented as CLI special command (handled before SQL parsing)
- Works with all query types

✅ **Requirement 2: Parsed SQL Structure**
- Shows query type (SELECT, INSERT, UPDATE, etc.)
- Detects clauses (WHERE, JOIN, ORDER BY, GROUP BY)
- Estimates complexity

✅ **Requirement 3: Cache Information**
- Displays cache entries checked
- Shows similarity scores (0.0-1.0)
- Lists top 5 most similar cached queries
- Predicts cache hits

✅ **Requirement 4: RL Agent Q-Values**
- Shows Q-values for all actions
- Displays best action (highest Q-value)
- Shows epsilon (exploration rate)
- Explains decision-making process

## Code Quality

### Error Handling
- Input validation on all Python functions
- try-except blocks with logging
- Fallback values for missing data
- Graceful degradation when Python unavailable

### Documentation
- Comprehensive docstrings with Args/Returns
- Inline comments explaining logic
- Type hints on function signatures
- Example outputs in documentation

### Testing
- 17 Python integration tests
- 5 Rust bridge tests
- Edge case coverage (empty queries, missing data, etc.)
- Performance validation

## Future Enhancements

### Planned Features
- [ ] JOIN optimization hints
- [ ] Subquery analysis
- [ ] Index recommendation engine
- [ ] Aggregate optimization strategies
- [ ] Parallel query plan suggestions
- [ ] Per-operator execution time breakdown

### Potential Improvements
- [ ] Interactive EXPLAIN mode (drill-down into specific sections)
- [ ] Historical query performance comparison
- [ ] What-if scenario analysis
- [ ] Cost-based query plan alternatives
- [ ] Detailed RL agent training history

## Troubleshooting

### "Query explainer not available"
**Cause:** Python environment not initialized  
**Fix:** Ensure sentence-transformers and torch are installed
```bash
pip install sentence-transformers torch numpy scikit-learn
```

### Empty Q-values in output
**Cause:** New RL agent with no training history  
**Expected:** Q-values default to 0, best_action is first in list

### Cache similarity showing 0.0
**Cause:** Few or no cached queries  
**Expected:** Cache grows as queries are executed

### Very long query truncated in output
**Expected:** Queries longer than 60 characters are automatically truncated for display

## References

- **Issue:** #66 (feat: Add EXPLAIN query plan command)
- **Related:** #45 (ASK for NL queries), #46 (RL optimization)
- **Architecture:** nexum_core/src/bridge/mod.rs
- **Tests:** nexum_ai/tests/test_explain_integration.py

---

**Status:** Released in PR #131  
**Maintainer:** NexumDB Core Team
