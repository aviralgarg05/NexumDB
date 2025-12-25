# NexumDB Core Performance Benchmarks

This document provides an overview of the comprehensive benchmarking suite implemented for `nexum_core`.

## Overview

The benchmarking suite uses [Criterion.rs](https://github.com/bheisler/criterion.rs) to provide:
- Statistical analysis of performance metrics
- Regression detection across code changes
- HTML reports with interactive charts
- CI integration for automated performance monitoring

## Quick Start

```bash
# Run all benchmarks
cd nexum_core
cargo bench

# Run specific benchmark suite
cargo bench --bench storage_bench

# Use the benchmark runner script (Linux/macOS)
./run_benchmarks.sh
```

## Benchmark Suites

### 1. Storage Engine (`storage_bench.rs`)
- **Write Throughput**: 100 to 100,000 sequential writes
- **Read Throughput**: 100 to 100,000 sequential reads
- **Mixed Workload**: Various read/write ratios (70/30, 50/50, 30/70)
- **Prefix Scanning**: Scan performance with different prefix sizes
- **Persistence**: Flush and durability operations

### 2. SQL Parser (`sql_bench.rs`)
- **CREATE TABLE**: Simple and complex table definitions
- **INSERT**: 1 to 1,000 row batch inserts
- **SELECT**: Simple to complex query parsing
- **Mixed Workload**: Realistic SQL statement patterns
- **Error Handling**: Invalid SQL processing performance
- **Large Queries**: 10,000+ row INSERT statements

### 3. Query Executor (`executor_bench.rs`)
- **Simple SELECT**: Table scanning with 100 to 10,000 records
- **Filtered SELECT**: WHERE clause evaluation performance
- **INSERT Operations**: 1 to 1,000 row batch inserts
- **CREATE TABLE**: Table creation overhead
- **Mixed Workload**: Typical application usage patterns
- **Large Datasets**: 50,000 to 100,000 record operations

### 4. Filter Evaluation (`filter_bench.rs`)
- **Simple Comparisons**: =, >, <, >=, <=, != operations
- **Complex Expressions**: AND, OR, nested conditions
- **LIKE Patterns**: %, _ wildcard matching
- **IN Lists**: 5 to 100 item list performance
- **BETWEEN Ranges**: Numeric and text range filtering
- **Batch Evaluation**: 10,000 row filter processing

## Performance Targets

| Component | Operation | Target Performance |
|-----------|-----------|-------------------|
| Storage | Write Throughput | >10,000 ops/sec |
| Storage | Read Throughput | >50,000 ops/sec |
| Storage | Mixed Workload | >5,000 ops/sec |
| SQL Parser | Simple Queries | <1ms parse time |
| SQL Parser | Complex Queries | <10ms parse time |
| Executor | Table Scans | >1,000 records/ms |
| Executor | Filtered Queries | >500 records/ms |
| Filters | Simple Filters | <1μs per row |
| Filters | Complex Filters | <10μs per row |

## CI Integration

Benchmarks run automatically on pull requests:

1. **Baseline Comparison**: Compare against main branch
2. **Regression Detection**: Identify performance degradations
3. **Report Generation**: Create performance summaries
4. **Artifact Upload**: Store detailed results

### CI Workflow
```yaml
benchmarks:
  name: Rust benchmarks
  runs-on: ubuntu-latest
  if: github.event_name == 'pull_request'
  # ... runs benchmarks and compares results
```

## Interpreting Results

### Criterion Output
```
storage_write/sequential_writes/1000
                        time:   [2.1234 ms 2.1456 ms 2.1678 ms]
                        thrpt:  [461.23 Kelem/s 466.12 Kelem/s 470.89 Kelem/s]
```

- **time**: Mean execution time with confidence interval
- **thrpt**: Throughput (elements per second)
- **change**: Performance change from previous run

### HTML Reports
- Interactive charts showing performance trends
- Statistical analysis with confidence intervals
- Comparison between different benchmark runs
- Detailed timing distributions

## Development Workflow

### Before Making Changes
```bash
# Establish baseline
cargo bench > baseline_results.txt
```

### After Making Changes
```bash
# Run benchmarks and compare
cargo bench
# Check for regressions in the output
```

### Adding New Benchmarks

1. **Create benchmark function**:
```rust
fn my_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("my_feature");
    // ... benchmark implementation
    group.finish();
}
```

2. **Add to criterion_group!**:
```rust
criterion_group!(benches, existing_bench, my_benchmark);
```

3. **Update documentation** with performance targets

## Optimization Guidelines

### Storage Layer
- Minimize allocations in hot paths
- Use batch operations for bulk data
- Optimize key encoding/decoding
- Implement efficient caching

### SQL Parser
- Cache compiled regex patterns
- Minimize string allocations
- Use efficient AST construction
- Implement parser combinators

### Query Executor
- Implement query plan caching
- Use vectorized operations
- Optimize memory access patterns
- Implement parallel processing

### Filter Evaluation
- Short-circuit boolean expressions
- Use SIMD for bulk operations
- Optimize regex compilation
- Implement predicate pushdown

## Troubleshooting

### Common Issues

**Inconsistent Results**
- Ensure stable system conditions
- Close other applications
- Run multiple times and average

**Memory Issues**
- Use `BatchSize::SmallInput` for large datasets
- Monitor memory allocation patterns
- Check for memory leaks

**Compilation Errors**
- Ensure all dependencies are available
- Check feature flags
- Verify Rust version compatibility

### Performance Analysis Tools

```bash
# Detailed profiling
cargo install flamegraph
cargo flamegraph --bench storage_bench

# Memory profiling
cargo install heaptrack
heaptrack cargo bench --bench storage_bench

# Assembly analysis
cargo asm nexum_core::storage::StorageEngine::set
```

## Contributing

When contributing benchmarks:

1. **Follow existing patterns** in benchmark structure
2. **Include realistic test data** representative of actual usage
3. **Document performance expectations** and targets
4. **Test locally** before submitting PRs
5. **Consider CI runtime** - keep benchmarks reasonably fast

### Benchmark Checklist
- [ ] Realistic test data and scenarios
- [ ] Multiple input sizes tested
- [ ] Appropriate throughput measurements
- [ ] Documentation updated
- [ ] Performance targets defined
- [ ] CI integration tested

## Resources

- [Criterion.rs Documentation](https://bheisler.github.io/criterion.rs/book/)
- [Rust Performance Book](https://nnethercote.github.io/perf-book/)
- [Benchmarking Best Practices](https://github.com/bheisler/criterion.rs/blob/master/book/src/user_guide/advanced_configuration.md)