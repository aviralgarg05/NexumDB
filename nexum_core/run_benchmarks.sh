#!/bin/bash

# NexumDB Core Benchmark Runner
# Runs all benchmarks and generates reports

set -e

echo "Running NexumDB Core Benchmarks"
echo "=================================="

# Check if we're in the right directory
if [ ! -f "Cargo.toml" ] || [ ! -d "benches" ]; then
    echo "Error: Please run this script from the nexum_core directory"
    exit 1
fi

# Create output directory
mkdir -p benchmark_results
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="benchmark_results/run_$TIMESTAMP"
mkdir -p "$RESULTS_DIR"

echo "Results will be saved to: $RESULTS_DIR"
echo ""

# Function to run a benchmark suite
run_benchmark() {
    local bench_name=$1
    local description=$2
    
    echo "Running $description..."
    echo "   Benchmark: $bench_name"
    
    # Run the benchmark and capture output
    if cargo bench --bench "$bench_name" 2>&1 | tee "$RESULTS_DIR/${bench_name}_output.log"; then
        echo "   $description completed successfully"
    else
        echo "   $description failed"
        return 1
    fi
    echo ""
}

# Run all benchmark suites
echo "Starting benchmark execution..."
echo ""

run_benchmark "storage_bench" "Storage Engine Benchmarks"
run_benchmark "sql_bench" "SQL Parser Benchmarks"
run_benchmark "executor_bench" "Query Executor Benchmarks"
run_benchmark "filter_bench" "Filter Evaluation Benchmarks"

# Copy criterion reports
if [ -d "target/criterion" ]; then
    echo "Copying detailed reports..."
    cp -r target/criterion "$RESULTS_DIR/"
    echo "   Criterion reports copied"
else
    echo "   No criterion reports found"
fi

# Generate summary
echo "Generating benchmark summary..."
cat > "$RESULTS_DIR/summary.md" << EOF
# NexumDB Core Benchmark Results

**Run Date:** $(date)
**Git Commit:** $(git rev-parse HEAD 2>/dev/null || echo "Unknown")
**Git Branch:** $(git branch --show-current 2>/dev/null || echo "Unknown")

## Benchmark Suites Executed

- Storage Engine Benchmarks
- SQL Parser Benchmarks  
- Query Executor Benchmarks
- Filter Evaluation Benchmarks

## Files Generated

- \`storage_bench_output.log\` - Storage engine benchmark results
- \`sql_bench_output.log\` - SQL parser benchmark results
- \`executor_bench_output.log\` - Query executor benchmark results
- \`filter_bench_output.log\` - Filter evaluation benchmark results
- \`criterion/\` - Detailed HTML reports and charts

## Viewing Results

### Command Line Results
View the \`.log\` files for detailed command-line output including:
- Performance metrics (mean, median, std dev)
- Throughput measurements
- Regression analysis

### HTML Reports
Open \`criterion/report/index.html\` in a web browser for:
- Interactive performance charts
- Historical trend analysis
- Detailed statistical analysis
- Performance comparisons

## Performance Analysis

### Key Metrics to Monitor
- **Storage Write Throughput**: Target >10,000 ops/sec
- **Storage Read Throughput**: Target >50,000 ops/sec  
- **SQL Parse Time**: Target <1ms for simple queries
- **Query Execution**: Target >1,000 records/ms
- **Filter Evaluation**: Target <1μs per row

### Regression Detection
Check for significant performance changes compared to previous runs.
Look for:
- Increased execution times
- Reduced throughput
- Higher variance in measurements

EOF

echo "   ✅ Summary generated: $RESULTS_DIR/summary.md"
echo ""

# Final summary
echo "🎉 All benchmarks completed successfully!"
echo ""
echo "📊 Results Summary:"
echo "   - Logs: $RESULTS_DIR/*.log"
echo "   - HTML Reports: $RESULTS_DIR/criterion/report/index.html"
echo "   - Summary: $RESULTS_DIR/summary.md"
echo ""
echo "💡 To view HTML reports:"
echo "   cd $RESULTS_DIR && python -m http.server 8000"
echo "   Then open: http://localhost:8000/criterion/report/index.html"
echo ""
echo "🔍 To compare with previous runs:"
echo "   ls benchmark_results/"
echo ""