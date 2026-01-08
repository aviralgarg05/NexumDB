# Logging Configuration Guide

## Overview

NexumDB now supports configurable logging levels and structured logging output, making it easier to debug issues and monitor production deployments.

## Features

- **Configurable Log Levels**: Control verbosity via environment variables
- **Structured Logging**: JSON output for production monitoring
- **Pretty Logging**: Human-readable format for development
- **Contextual Information**: Rich metadata with each log entry
- **Per-Module Filtering**: Fine-grained control over log sources

## Quick Start

### Basic Usage

```bash
# Default: INFO level with pretty formatting
cargo run --bin nexum

# Debug level for detailed information
RUST_LOG=debug cargo run --bin nexum

# JSON format for production
RUST_LOG=info NEXUM_LOG_FORMAT=json cargo run --bin nexum
```

## Environment Variables

### RUST_LOG

Controls the log level and filtering. Supports standard Rust log levels:

- `error` - Only errors
- `warn` - Warnings and errors
- `info` - Informational messages (default)
- `debug` - Detailed debugging information
- `trace` - Very verbose tracing

**Examples:**

```bash
# Set global log level
RUST_LOG=debug cargo run --bin nexum

# Filter by module
RUST_LOG=nexum_cli=debug,nexum_core=info cargo run --bin nexum

# Multiple filters
RUST_LOG=nexum_cli=trace,nexum_core::executor=debug cargo run --bin nexum

# Disable all logs except errors
RUST_LOG=error cargo run --bin nexum
```

### NEXUM_LOG_FORMAT

Controls the output format:

- `pretty` - Human-readable format with colors (default)
- `json` - Structured JSON for log aggregation

**Examples:**

```bash
# Pretty format (default)
NEXUM_LOG_FORMAT=pretty cargo run --bin nexum

# JSON format for production
NEXUM_LOG_FORMAT=json cargo run --bin nexum
```

## Log Levels Explained

### ERROR
Critical failures that prevent operations from completing:
```
ERROR Execution error: Table 'users' not found
ERROR Parse error: Unexpected token at position 15
```

### WARN
Non-critical issues or fallback scenarios:
```
WARN NL translator not available: Python module not found
WARN Query explainer not available: Model not loaded
```

### INFO (Default)
Important operational events:
```
INFO Starting NexumDB v0.2.0
INFO Storage engine initialized successfully
INFO Natural Language translator enabled
INFO Query executed successfully
```

### DEBUG
Detailed execution flow for debugging:
```
DEBUG Initializing storage engine at ./nexumdb_data
DEBUG Received input: SELECT * FROM users
DEBUG SQL parsed successfully
DEBUG Schema context: 245 bytes
```

### TRACE
Very verbose information (use sparingly):
```
TRACE Entering function: execute_query
TRACE Cache lookup: key=abc123, hit=true
TRACE Exiting function: execute_query, duration=15ms
```

## Production Deployment

### Recommended Configuration

```bash
# Production: INFO level with JSON output
RUST_LOG=info NEXUM_LOG_FORMAT=json ./nexum

# Staging: DEBUG level with JSON output
RUST_LOG=debug NEXUM_LOG_FORMAT=json ./nexum
```

### JSON Output Example

```json
{
  "timestamp": "2024-01-08T15:30:45.123Z",
  "level": "INFO",
  "target": "nexum_cli",
  "message": "Query executed successfully",
  "span": {
    "name": "execute_query"
  },
  "fields": {
    "query_type": "SELECT",
    "duration_ms": 15
  }
}
```

### Log Aggregation

JSON logs can be easily integrated with log aggregation systems:

- **ELK Stack**: Elasticsearch, Logstash, Kibana
- **Splunk**: Direct JSON ingestion
- **Datadog**: Log collection agent
- **CloudWatch**: AWS log streams
- **Loki**: Grafana's log aggregation

## Development Workflow

### Debugging a Specific Issue

```bash
# Enable debug logs for the entire application
RUST_LOG=debug cargo run --bin nexum

# Focus on specific modules
RUST_LOG=nexum_core::executor=trace cargo run --bin nexum

# Combine with pretty formatting
RUST_LOG=debug NEXUM_LOG_FORMAT=pretty cargo run --bin nexum
```

### Testing with Different Log Levels

```bash
# Silent mode (errors only)
RUST_LOG=error cargo test

# Verbose test output
RUST_LOG=trace cargo test -- --nocapture

# Debug specific test
RUST_LOG=debug cargo test test_query_execution -- --nocapture
```

## Performance Considerations

- **INFO level**: Minimal overhead (~1-2% in production)
- **DEBUG level**: Moderate overhead (~5-10%), use in staging
- **TRACE level**: Significant overhead (~20-30%), development only
- **JSON format**: Slightly faster than pretty format
- **Pretty format**: Better for human readability, slower serialization

## Best Practices

1. **Production**: Use `RUST_LOG=info` with `NEXUM_LOG_FORMAT=json`
2. **Staging**: Use `RUST_LOG=debug` with `NEXUM_LOG_FORMAT=json`
3. **Development**: Use `RUST_LOG=debug` with `NEXUM_LOG_FORMAT=pretty`
4. **Debugging**: Use `RUST_LOG=trace` temporarily, then reduce
5. **CI/CD**: Use `RUST_LOG=info` for test runs
6. **Log Rotation**: Implement external log rotation for production

## Troubleshooting

### Logs Not Appearing

```bash
# Ensure RUST_LOG is set
echo $RUST_LOG

# Try explicit level
RUST_LOG=info cargo run --bin nexum

# Check if logs are going to stderr
RUST_LOG=debug cargo run --bin nexum 2>&1 | less
```

### Too Much Output

```bash
# Reduce log level
RUST_LOG=warn cargo run --bin nexum

# Filter specific modules
RUST_LOG=nexum_cli=info,nexum_core=warn cargo run --bin nexum
```

### JSON Parsing Issues

```bash
# Validate JSON output
RUST_LOG=info NEXUM_LOG_FORMAT=json cargo run --bin nexum 2>&1 | jq .

# Pretty print JSON logs
RUST_LOG=info NEXUM_LOG_FORMAT=json cargo run --bin nexum 2>&1 | jq -C . | less -R
```

## Migration from Previous Versions

Previous versions used `println!` statements for logging. These have been replaced with structured logging:

**Before:**
```rust
println!("Storage engine initialized");
```

**After:**
```rust
info!("Storage engine initialized successfully");
```

User-facing output (prompts, results) remains unchanged. Only internal logging has been enhanced.

## Future Enhancements

- [ ] Log file output with rotation
- [ ] Performance metrics in logs
- [ ] Distributed tracing support
- [ ] Custom log formatters
- [ ] Log sampling for high-volume scenarios

## Contributing

When adding new logging:

1. Use appropriate log levels (error, warn, info, debug, trace)
2. Include contextual information in log messages
3. Avoid logging sensitive data (passwords, tokens)
4. Use structured fields for machine-readable data
5. Keep user-facing output separate from logs

## References

- [tracing documentation](https://docs.rs/tracing/)
- [tracing-subscriber documentation](https://docs.rs/tracing-subscriber/)
- [env_logger documentation](https://docs.rs/env_logger/)
- [Rust logging best practices](https://rust-lang-nursery.github.io/rust-cookbook/development_tools/debugging/log.html)
