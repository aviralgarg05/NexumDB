[![CI](https://github.com/aviralgarg05/NexumDB/actions/workflows/ci.yml/badge.svg)](https://github.com/aviralgarg05/NexumDB/actions/workflows/ci.yml)

# NexumDB - AI-Native Database

An innovative, open-source database that combines traditional SQL with AI-powered features including advanced query operators, natural language processing, semantic caching, and reinforcement learning-based query optimization.

## Architecture

- **Core System**: Rust-based storage engine using sled, with SQL parsing and intelligent execution
- **AI Engine**: Python-based semantic caching, NL translation, RL optimization, and model management using local models
- **Integration**: PyO3 bindings for seamless Rust-Python integration

## Features

### v0.3.0 - Advanced SQL & Persistent Learning
- **Advanced SQL Operators**: LIKE (pattern matching), IN (list membership), BETWEEN (range queries)
- **Query Modifiers**: ORDER BY (multi-column sorting), LIMIT (result truncation)
- **Persistent RL Agent**: Q-table saves to disk, learning survives restarts
- **Model Management**: Automatic LLM downloads from HuggingFace Hub

### v0.2.0 - Intelligent Query Engine
- **WHERE Clause Filtering**: Full support for comparison (=, >, <, >=, <=, !=) and logical operators (AND, OR)
- **Natural Language Queries**: ASK command for plain English queries with local LLM or rule-based fallback
- **Reinforcement Learning**: Q-Learning agent that optimizes query execution strategies
- **Expression Evaluator**: Type-safe WHERE clause evaluation with comprehensive operator support

### v0.1.0 - Foundation
- SQL support (CREATE TABLE, INSERT, SELECT)
- Semantic query caching using local embedding models (all-MiniLM-L6-v2)
- Self-optimizing query execution
- Local-only execution (no cloud dependencies)
- Persistent storage with sled
- Query performance instrumentation

## Project Structure

```
NexumDB/
├── nexum_core/          # Rust core database engine
│   └── src/
│       ├── storage/     # Storage layer (sled)
│       ├── sql/         # SQL parsing and planning
│       ├── catalog/     # Table metadata management
│       ├── executor/    # Query execution + caching
│       └── bridge/      # Python integration (PyO3)
├── nexum_cli/           # CLI REPL interface
├── nexum_ai/            # Python AI engine
│   └── optimizer.py     # Semantic cache and RL optimizer
└── tests/               # Integration tests
```

## Building

```bash
# Set PyO3 forward compatibility (for Python 3.14+)
export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1

# Build release binary
cargo build --release
```

## Build, run and stop the application using docker compose

### Build the application

```bash
$ docker compose build
```

### Run the application

```bash
$ docker compose up
```

### Run an interactive shell

```bash
$ docker compose up -d
$ docker exec -it nexumdb nexum
```
 
### Stop the application 

```bash
$ docker compose down
```

### Logs 

```bash 
$ docker compose logs
```

## Python Dependencies

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install AI dependencies
pip install -r nexum_ai/requirements.txt
```

## Running Tests

```bash
export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1
cargo test -- --test-threads=1
```

**Test Results**: 11/11 passing

## Usage

```bash
./target/release/nexum
```

### SQL Queries

```sql
CREATE TABLE users (id INTEGER, name TEXT, age INTEGER);
INSERT INTO users (id, name, age) VALUES (1, 'Alice', 30), (2, 'Bob', 25);

-- Simple query
SELECT * FROM users;

-- WHERE clause filtering (v0.2.0)
SELECT * FROM users WHERE age > 25;
SELECT * FROM users WHERE name = 'Alice' AND age >= 30;

-- Advanced operators (v0.3.0)
SELECT * FROM users WHERE name LIKE 'A%';  -- Pattern matching
SELECT * FROM users WHERE age BETWEEN 20 AND 30;  -- Range query
SELECT * FROM users WHERE name IN ('Alice', 'Bob');  -- List membership

-- Query modifiers (v0.3.0)
SELECT * FROM users ORDER BY age DESC;  -- Sort by age descending
SELECT * FROM users ORDER BY age ASC LIMIT 5;  -- Top 5 by age

-- Combined example
SELECT * FROM products 
WHERE price BETWEEN 100 AND 500 
  AND category IN ('electronics', 'accessories')
  AND name LIKE 'L%'
ORDER BY price DESC 
LIMIT 10;
```

### Natural Language Queries (v0.2.0+)

```
nexumdb> ASK Show me all users
Translating: 'Show me all users'
Generated SQL: SELECT * FROM users
[Results displayed]

nexumdb> ASK Find users older than 25
Translating: 'Find users older than 25'
Generated SQL: SELECT * FROM users WHERE age > 25
[Filtered results displayed]

nexumdb> ASK Show top 3 products under $100 sorted by price
Generated SQL: SELECT * FROM products WHERE price < 100 ORDER BY price ASC LIMIT 3
[Results displayed]
```

### Performance Examples

**Advanced SQL Operators (v0.3.0):**
```
Query: SELECT * FROM products WHERE name LIKE 'Test%'
Filtered 15 rows using WHERE clause
Query executed in 2.9ms

Query: SELECT * FROM items WHERE price BETWEEN 100 AND 500
Filtered 42 rows using WHERE clause  
Query executed in 3.1ms

Query: SELECT * FROM orders WHERE status IN ('active', 'pending')
Filtered 28 rows using WHERE clause
Query executed in 2.7ms
```

**Query Modifiers:**
```
Query: SELECT * FROM products ORDER BY price DESC LIMIT 5
Sorted 150 rows using ORDER BY
Limited to 5 rows using LIMIT
Query executed in 3.8ms
```

**Semantic Caching:**
```
First SELECT:  Query executed in 2.5ms  (cache miss)
Second SELECT: Query executed in 0.04ms (cache hit - 60x faster)
```

**RL Optimization (Automatic):**
```
The RL agent learns optimal strategies automatically.
Learning persists across restarts (v0.3.0).
No configuration needed - just use the database!
```

## Development Status

- **Phase 1**: Project Skeleton & Storage Layer - COMPLETE
- **Phase 2**: SQL Engine - COMPLETE  
- **Phase 3**: AI Bridge (PyO3) - COMPLETE
- **Phase 4**: Intelligent Features - COMPLETE
- **Phase 5**: Final Interface - IN PROGRESS

## Key Achievements

1. Fully functional SQL database with CREATE, INSERT, SELECT
2. Semantic caching using local embedding models
3. Successful Rust-Python integration via PyO3
4. 60x query speedup on cache hits
5. Comprehensive test suite (11 tests passing)
6. Query performance instrumentation
7. Production release build working

## Technical Highlights

- **Zero Cloud Dependencies**: All models run locally
- **High Performance**: Sub-millisecond query execution
- **AI-Powered**: Semantic caching using transformer embeddings
- **Type-Safe**: Rust core with comprehensive error handling
- **Well-Tested**: Full unit and integration test coverage

## License

MIT
