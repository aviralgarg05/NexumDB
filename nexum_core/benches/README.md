# NexumDB Core Benchmarks

This directory contains comprehensive performance benchmarks for the `nexum_core` module using the [Criterion](https://github.com/bheisler/criterion.rs) benchmarking framework.

## Benchmark Categories

### 1. Database Comparison Benchmarks (`db_comparison.rs`)
**Comparative analysis between NexumDB and SQLite (via `rusqlite`):**
- **Cross-Engine Comparison**: Evaluates NexumDB's execution speed against the industry-standard SQLite baseline.
- **Semantic Cache Analysis**: Measures performance delta between "Cold" starts and "Cached" executor states.
- **Platform Stability**: Implements proper lifetime management for `NamedTempFile` and `TempDir` to ensure stable benchmarking on Windows and Unix alike.

### 2. Storage Engine Benchmarks (`storage_bench.rs`)
Tests the performance of the underlying storage engine operations:
- **Write Throughput**: Sequential write operations with different data sizes.
- **Read Throughput**: Sequential read operations with different data sizes.
- **Prefix Scanning**: Performance of prefix-based key scanning.

### 3. SQL Parser Benchmarks (`sql_bench.rs`)
Measures SQL parsing performance across different query types:
- **CREATE/INSERT/SELECT**: Parsing overhead for standard SQL statements.
- **Large Queries**: Performance scaling with complex SQL statements.

### 4. Query Executor & Filter Evaluation
- **Executor Bench**: Tests the query execution engine performance on substantial data volumes.
- **Filter Bench**: Focuses on `WHERE` clause evaluation (LIKE, IN, BETWEEN, and complex logic).

---

## Running Benchmarks

### Prerequisites
Ensure `rusqlite` is available in your dev-dependencies. If missing, run:

# Comparative benchmarks (Nexum vs SQLite)
```bash
cargo bench --bench db_comparison
```

# Storage engine benchmarks
```bash
cargo bench --bench storage_bench
```

# SQL parser benchmarks  
```bash
cargo bench --bench sql_bench
```

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

## Benchmark Comparative Results
The following results were captured on February 12, 2026, using 1,000 unique pre-populated records.

| Benchmark Case | Engine | Avg Latency | Comparison |
| :--- | :--- | :--- | :--- |
| **Point Lookup** | SQLite (Prepared) | **157.09 µs** | Baseline (1.0x) |
| **Point Lookup (Cold)** | NexumDB | **2.81 ms** | ~17.9x slower |
| **Point Lookup (Cached)**| NexumDB | **2.64 ms** | ~16.8x slower |


## Performance Analysis
- **SQLite Efficiency**: SQLite's performance serves as our high-bar baseline. Its C-based architecture and mature B-Tree implementation result in sub-millisecond latencies for prepared lookups.

- **NexumDB Cold vs. Cached**: We observe a 6.1% performance gain in cached results. This confirms the semantic cache is active; however, the proximity of "Cold" and "Cached" timings suggests that the primary bottleneck currently lies in the Executor Dispatch Pipeline rather than storage retrieval.

## CI Integration

Benchmarks run automatically on pull requests to:

- Compare performance against the main branch
- Detect performance regressions
- Generate performance reports
- Upload benchmark artifacts


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
- **Zero-Copy Parsing**: Reduce allocations during Parser::parse by using string references where possible.
- **Plan Caching**: Cache the execution plan alongside results to bypass the standard dispatch pipeline.
- **LSM Tuning**: Optimize the underlying storage parameters to reduce seek time for point lookups.

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