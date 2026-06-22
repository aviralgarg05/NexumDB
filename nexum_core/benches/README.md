# NexumDB Core Benchmarks

This directory contains comprehensive performance benchmarks for the `nexum_core` module using the [Criterion](https://github.com/bheisler/criterion.rs) benchmarking framework.

## Benchmark Categories
## Performance & Benchmarks (Feb 2026)
To maintain the high performance expected of a Rust-based engine, NexumDB is continuously benchmarked against SQLite using the `criterion` suite.

### Comparative Performance Results
| Operation | SQLite | NexumDB | Delta |
| :--- | :--- | :--- | :--- |
| **Single INSERT** | 15.18 ms | **7.48 ms** | NexumDB ~2x Faster |
| **Point SELECT (Cold)** | **140.5 µs** | 1.86 ms | SQLite Faster |
| **Point SELECT (Cached)**| **143.8 µs** | 1.87 ms | SQLite Faster |

### Analysis
![Benchmark Visualization](./nexum_core/benches/bench_results.png)

### Architectural Insights

#### 1. Write Throughput: The LSM-Tree Advantage
NexumDB’s storage engine (`sled`) utilizes a **Log-Structured Merge-tree (LSM-tree)**, whereas SQLite uses a traditional **B-tree**. 
* **LSM-tree (NexumDB):** Optimizes for writes by batching updates into immutable segments, leading to the 2x speedup observed in our `INSERT` benchmarks.
* **B-tree (SQLite):** Optimized for reads. Every write requires finding a leaf node on disk, which involves more synchronous I/O.



#### 2. Read Latency & AI Overhead
In small-scale point lookups (1,000 rows), SQLite's raw C-speed is superior. NexumDB's current ~1.8ms latency includes:
* **SQL Parsing**: Converting strings to `Statement` enums.
* **PyO3 Bridge**: The overhead of crossing the Rust-Python boundary for AI-native planning.
* **Semantic Caching**: The current benchmark dataset is too small to show the "skip-the-disk" benefits of semantic caching, which scale exponentially with query complexity and data volume.

---

## Architecture
* **Core System**: Rust-based storage engine using `sled`, with SQL parsing and intelligent execution.
* **AI Engine**: Python-based semantic caching, NL translation, and RL optimization via local models.
* **Integration**: PyO3 bindings for seamless Rust-Python interoperability.



## Features
### v0.4.0 - Core Correctness & Table Management
* **Projection-Correct SELECT**: Column/alias projection with schema validation.
* **Schema-Safe Writes**: INSERT/UPDATE validation with best-effort coercion.
* **Table Management**: SHOW TABLES, DESCRIBE, DROP TABLE (IF EXISTS).
* **Performance Suite**: Integrated benchmark framework for regression testing.

---

Tests the performance of the underlying storage engine operations:

- **Write Throughput**: Sequential write operations with different data sizes
- **Read Throughput**: Sequential read operations with different data sizes  
- **Mixed Workload**: Combined read/write operations with different ratios
- **Prefix Scanning**: Performance of prefix-based key scanning
- **Persistence**: Flush and durability operations

### 2. SQL Parser Benchmarks (`sql_bench.rs`)

Measures SQL parsing performance across different query types:

- **CREATE TABLE**: Simple and complex table definitions
- **INSERT**: Single and multi-row insert statements
- **SELECT**: Various SELECT queries with different complexity
- **Mixed Workload**: Typical application SQL statement patterns
- **Error Handling**: Performance of invalid SQL processing
- **Large Queries**: Performance with very large SQL statements

### 3. Query Executor Benchmarks (`executor_bench.rs`)

Tests the query execution engine performance:

- **Simple SELECT**: Basic table scanning operations
- **Filtered SELECT**: Queries with WHERE clauses
- **INSERT Operations**: Data insertion performance
- **CREATE TABLE**: Table creation overhead
- **Mixed Workload**: Realistic application usage patterns
- **Large Datasets**: Performance with substantial data volumes

### 4. Filter Evaluation Benchmarks (`filter_bench.rs`)

Focuses on WHERE clause evaluation performance:

- **Simple Comparisons**: Basic equality, inequality operations
- **Complex Expressions**: AND, OR, nested conditions
- **LIKE Patterns**: Pattern matching with wildcards
- **IN Lists**: Performance with different list sizes
- **BETWEEN Ranges**: Range-based filtering
- **Batch Evaluation**: Filter performance across large datasets

## Running Benchmarks

### Run All Benchmarks
```bash
cd nexum_core
cargo bench
```

### Run Specific Benchmark Suite
```bash
# Storage engine benchmarks
cargo bench --bench storage_bench

# SQL parser benchmarks  
cargo bench --bench sql_bench

# Query executor benchmarks
cargo bench --bench executor_bench

# Filter evaluation benchmarks
cargo bench --bench filter_bench
```

### Run Specific Benchmark
```bash
# Run only write throughput tests
cargo bench --bench storage_bench -- "storage_write"

# Run only SELECT parsing tests
cargo bench --bench sql_bench -- "parse_select"
```

### Generate HTML Reports
```bash
cargo bench
# Reports are generated in target/criterion/
```

## Benchmark Results

Criterion generates detailed reports including:

- **Performance Metrics**: Mean, median, standard deviation
- **Throughput Measurements**: Operations per second
- **Regression Detection**: Performance changes over time
- **HTML Reports**: Interactive charts and graphs
- **Statistical Analysis**: Confidence intervals and outlier detection

## CI Integration

Benchmarks run automatically on pull requests to:

- Compare performance against the main branch
- Detect performance regressions
- Generate performance reports
- Upload benchmark artifacts

## Performance Targets

### Storage Engine
- **Write Throughput**: >10,000 ops/sec for small records
- **Read Throughput**: >50,000 ops/sec for cached data
- **Mixed Workload**: Maintain >5,000 ops/sec combined

### SQL Parser
- **Simple Queries**: <1ms parsing time
- **Complex Queries**: <10ms parsing time
- **Large Queries**: Linear scaling with query size

### Query Executor
- **Table Scans**: >1,000 records/ms
- **Filtered Queries**: >500 records/ms after filtering
- **Insert Operations**: >1,000 records/sec

### Filter Evaluation
- **Simple Filters**: <1μs per row evaluation
- **Complex Filters**: <10μs per row evaluation
- **Batch Processing**: >100,000 rows/sec

## Optimization Guidelines

### Storage Layer
- Minimize disk I/O operations
- Optimize key encoding/decoding
- Implement efficient caching strategies
- Use batch operations where possible

### SQL Parser
- Cache parsed query plans
- Optimize AST construction
- Minimize string allocations
- Use efficient parsing algorithms

### Query Executor
- Implement query plan optimization
- Use indexed access when available
- Optimize memory usage patterns
- Implement parallel processing

### Filter Evaluation
- Short-circuit boolean expressions
- Optimize regex compilation
- Use SIMD operations where possible
- Implement predicate pushdown

## Adding New Benchmarks

When adding new benchmarks:

1. **Follow Naming Conventions**: Use descriptive function names
2. **Include Multiple Scenarios**: Test different data sizes and patterns
3. **Set Appropriate Throughput**: Use `Throughput::Elements` or `Throughput::Bytes`
4. **Use Realistic Data**: Generate representative test datasets
5. **Document Performance Targets**: Include expected performance ranges
6. **Test Edge Cases**: Include boundary conditions and error cases

### Example Benchmark Structure
```rust
fn my_new_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("my_feature");
    
    for size in [100, 1000, 10000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        
        group.bench_with_input(
            BenchmarkId::new("operation", size),
            size,
            |b, &size| {
                b.iter_batched(
                    || setup_test_data(size),
                    |data| black_box(perform_operation(data)),
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }
    
    group.finish();
}
```

## Troubleshooting

### Common Issues

1. **Benchmark Timeouts**: Reduce dataset sizes or increase measurement time
2. **Memory Usage**: Use `BatchSize::SmallInput` for large datasets
3. **Inconsistent Results**: Ensure stable system conditions
4. **Missing Dependencies**: Check that all required crates are available

### Performance Analysis

- Use `cargo flamegraph` for detailed profiling
- Monitor memory allocation patterns
- Check for unnecessary cloning or allocations
- Profile with different optimization levels

## Contributing

When contributing benchmarks:

1. Ensure benchmarks are deterministic and reproducible
2. Include documentation for new benchmark categories
3. Update performance targets if needed
4. Test benchmarks locally before submitting
5. Consider the impact on CI runtime (keep benchmarks reasonably fast)