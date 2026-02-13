# Changelog

All notable changes to NexumDB will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Notes

* Transaction WAL currently uses a full `scan_all()` snapshot on `BEGIN` as an MVP implementation. A follow-up is planned to replace this with an incremental per-mutation undo/redo log for better scalability.
* Transaction isolation is currently a single-active-transaction model per `Executor` and does not implement MVCC across multiple sessions yet.

## [0.6.0](https://github.com/aviralgarg05/NexumDB/compare/v0.5.0...v0.6.0) (2026-02-06)


### Features

* align templates and README with OSCG'26 guidelines ([79bb239](https://github.com/aviralgarg05/NexumDB/commit/79bb239c88fe4a434d7dc0ecc04d700d5363c19a))
* enhance PR review automation with comprehensive workflows ([ad3fda1](https://github.com/aviralgarg05/NexumDB/commit/ad3fda1a4bc26bfc125f42870028acec06d02a05))


### Bug Fixes

* correct Release Please conditional syntax to single line ([d1382ba](https://github.com/aviralgarg05/NexumDB/commit/d1382ba086254a525a62f42d244a69c06b2833e4))
* release only on direct pushes, not PR merges ([8d4ab16](https://github.com/aviralgarg05/NexumDB/commit/8d4ab16ac393b3974436c00cce90fc9fff4d1bcc))

## [0.5.0](https://github.com/aviralgarg05/NexumDB/compare/v0.4.0...v0.5.0) (2026-02-06)


### Features

* add comprehensive OSS automation and community features ([cf22f4b](https://github.com/aviralgarg05/NexumDB/commit/cf22f4bff1996f9d2bc5b28a19d652c680b0b148))
* Add DELETE statement support ([17974b8](https://github.com/aviralgarg05/NexumDB/commit/17974b87cdda4544ebd102891c73d3c1a321464a)), closes [#45](https://github.com/aviralgarg05/NexumDB/issues/45)
* Add DELETE statement support with WHERE clause filtering ([44bedae](https://github.com/aviralgarg05/NexumDB/commit/44bedaed335c233b8b4effae52fc8bc2c19f5aa0))
* Add UPDATE statement support ([922afce](https://github.com/aviralgarg05/NexumDB/commit/922afcea9856a2f30de3276cc2bb34dc85d67cc8))
* Add UPDATE statement support ([96921be](https://github.com/aviralgarg05/NexumDB/commit/96921beb26c10380aae6cecae3c8445ee84ceffb)), closes [#46](https://github.com/aviralgarg05/NexumDB/issues/46)
* add UPDATE/DELETE support with cache invalidation (v0.4.0) ([777421c](https://github.com/aviralgarg05/NexumDB/commit/777421c2fe81c25f067b340c772f31d8d90e5455))
* Added .editorconfig for consistent formatting ([d659d8c](https://github.com/aviralgarg05/NexumDB/commit/d659d8cdc99c1808c6f0c5fe31e3ab541d830703))
* Added docker-compose.yml file ([c4a7b20](https://github.com/aviralgarg05/NexumDB/commit/c4a7b200d2a8d300b4a07d41b31ebd57b1088f36))
* App dockerization ([149d801](https://github.com/aviralgarg05/NexumDB/commit/149d8015b7d42c65ed56d1c80d7de84939fd4685))
* build and release docker image ([c09f564](https://github.com/aviralgarg05/NexumDB/commit/c09f564b5b92188366a4d2eb16f098ab87d8fff4))
* Cached Python deps in CI ([603cafa](https://github.com/aviralgarg05/NexumDB/commit/603cafa24ccedfc1a9e844b4b55d7238c81529da))
* Committed Cargo.lock for reproducible builds ([db4311e](https://github.com/aviralgarg05/NexumDB/commit/db4311ece06a103fdda71432928ad23a9dfa5d62))
* complete OSS infrastructure setup ([626d622](https://github.com/aviralgarg05/NexumDB/commit/626d6229dcf806f09319c67d246305e06f4cacbc))
* Implement semantic cache persistence to disk ([#47](https://github.com/aviralgarg05/NexumDB/issues/47)) ([0179452](https://github.com/aviralgarg05/NexumDB/commit/017945216d285ca51c89dbce59e8faec5fd5e890))
* implement semantic cache persistence to disk (issue [#47](https://github.com/aviralgarg05/NexumDB/issues/47)) ([78e2a4e](https://github.com/aviralgarg05/NexumDB/commit/78e2a4ed4f288e9c1e177a36a15f7b6e59506884))


### Bug Fixes

* Add boolean value assertion in test_parse_update_without_where ([be4e731](https://github.com/aviralgarg05/NexumDB/commit/be4e73146d544c4c23c628168e4d9f05b532555a))
* add defensive field width limits and improve explain_action docstrings ([763493e](https://github.com/aviralgarg05/NexumDB/commit/763493ed6c936bd7f5cb1d672a3efc3221633b0c))
* Address CodeRabbit feedback ([d596cb0](https://github.com/aviralgarg05/NexumDB/commit/d596cb09f8c1460687d68c4c6122ec1a2c92bc1b))
* Address CodeRabbit nitpicks ([19543f2](https://github.com/aviralgarg05/NexumDB/commit/19543f25151cb4bfe0e215c935e29d8f1d5c4d38))
* Address CodeRabbit review feedback ([a208673](https://github.com/aviralgarg05/NexumDB/commit/a2086732573d89271d027f6458e769604948db47))
* address CodeRabbit security concerns - Replace unsafe pickle.load() with JSON as default format - Add RestrictedUnpickler for legacy pickle files - Auto-convert legacy pickle caches to JSON on load - Add test cache files to .gitignore ([9d37b98](https://github.com/aviralgarg05/NexumDB/commit/9d37b9873d33ac255a9e674d0728704aa6b31dde))
* Apply cargo fmt formatting and improve UPDATE atomicity ([cbd62fc](https://github.com/aviralgarg05/NexumDB/commit/cbd62fc9b7d10dbc98a25cfd423f6eb2d9b59e09))
* cargo-audit installation issue in workflow ([16fd851](https://github.com/aviralgarg05/NexumDB/commit/16fd85113b86cfb21029dfc233ae532f5064c46c))
* cashing issue ([af501a3](https://github.com/aviralgarg05/NexumDB/commit/af501a361e5da998b90777257cec0b569f58c609))
* cashing problem ([d9fbf58](https://github.com/aviralgarg05/NexumDB/commit/d9fbf5831653f6db8a47b49eff8b454d6e8f160b))
* **ci:** correct YAML syntax in Python workflow job ([ccf5a6c](https://github.com/aviralgarg05/NexumDB/commit/ccf5a6ccfd03b2ca8622d8b35e3122416f1df6e0))
* **ci:** remove broken requirements-lock.txt dependency installation ([1b8edce](https://github.com/aviralgarg05/NexumDB/commit/1b8edce5ad18c8dc77d8fcaa3cf431200e182d28))
* final release-please workspace fix (remove workflow-level type override) ([ecb67d7](https://github.com/aviralgarg05/NexumDB/commit/ecb67d7a810f658151e1abcdd6ffcfe66f0dc095))
* fixed suggested changes ([dd83131](https://github.com/aviralgarg05/NexumDB/commit/dd831318ef9cf2bee50b27f107987a90ec710d2f))
* Handle serialization errors properly in batch UPDATE ([5a13d99](https://github.com/aviralgarg05/NexumDB/commit/5a13d99e087c8175106e04e1ec19c88ebb05bad8))
* Implement two-phase deletion to prevent partial deletion risk ([f184562](https://github.com/aviralgarg05/NexumDB/commit/f1845628260032a9262e8f71d4b1e775055d5a6c))
* reduce test iterations to prevent CI timeout ([5fe7919](https://github.com/aviralgarg05/NexumDB/commit/5fe79190325c4aa0b57a2e44b70029e4f1d057b7))
* remove auto-save from put() to prevent test hanging ([3b109c6](https://github.com/aviralgarg05/NexumDB/commit/3b109c6daa35476cf02a8c869629a06aa05336a0))
* remove unused io import ([01effa3](https://github.com/aviralgarg05/NexumDB/commit/01effa365638a9d11ff18bf306cc7c7cd7448728))
* remove unused mut from with_cache function ([4507d2f](https://github.com/aviralgarg05/NexumDB/commit/4507d2f03132543b011453e6c32a1cc49958eae5))
* replace deprecated PyO3 import_bound with import ([f60c483](https://github.com/aviralgarg05/NexumDB/commit/f60c483e0fe67c9979fadb096d3bcb5a1de09e94))
* Resolve merge conflicts in SQL module files ([ead1cf0](https://github.com/aviralgarg05/NexumDB/commit/ead1cf0cc7bc41d31b36e1df6c06dee61a740a86))
* resolve Rust formatting issues in benchmark files ([f2f6cf4](https://github.com/aviralgarg05/NexumDB/commit/f2f6cf4448dcab30b70f8d6dadded02da6f0b324))
* setup release-please for subpackage to bypass workspace parsing issue ([988281d](https://github.com/aviralgarg05/NexumDB/commit/988281da64c38c80b640a8d805c4a5fd9b20a759))
* suggested but coherant changes ([d1b9f1a](https://github.com/aviralgarg05/NexumDB/commit/d1b9f1a2a880df75b22acfdf2b90db9df35e4c4f))
* switch release-please to simple release-type for workspace support ([e23dd8a](https://github.com/aviralgarg05/NexumDB/commit/e23dd8a8c1dfb519875d1adcec3aa89f3b6cbdca))
* update bytes to 1.11.1 and improve DCO workflow ([a2374fe](https://github.com/aviralgarg05/NexumDB/commit/a2374fe2486d67494916decd3fb3b87bd461ff04))
* update release-please config for cargo workspace ([1bceda8](https://github.com/aviralgarg05/NexumDB/commit/1bceda802645e8c15bcab80415122393e367b394))
* upgraded pyo3 from  version 0.22 to version 0.24.1 ([67d5a47](https://github.com/aviralgarg05/NexumDB/commit/67d5a47466ceb12f5308cb283fbc0ef388c8e1b1))


### Reverts

* restore original test iterations ([ea703bd](https://github.com/aviralgarg05/NexumDB/commit/ea703bd89e80d72b52b1e5cf487ad74d3c221cf0))

## [0.4.0] - 2026-02-03

### Added
- Projection-aware SELECT with alias support
- SHOW TABLES, DESCRIBE, and DROP TABLE (IF EXISTS)
- Best-effort coercion for INSERT/UPDATE with schema validation
- Full cache invalidation on data writes
- Catalog persistence test coverage

### Changed
- StorageEngine clone now shares underlying sled database
- Semantic cache keys now include projection, WHERE, ORDER BY, LIMIT

### Tests
- Added projection, coercion, and table management coverage
- Extended integration tests for table lifecycle

## [0.3.0] - 2025-11-26

### Added
- Advanced SQL operators
  - LIKE operator for pattern matching (supports % and _ wildcards)
  - IN operator for list membership testing
  - BETWEEN operator for inclusive range queries
- Query modifiers
  - ORDER BY clause with multi-column sorting (ASC/DESC)
  - LIMIT clause for result truncation
- Persistent RL agent
  - save_state() and load_state() methods using joblib
  - Auto-load Q-table on agent initialization
  - Learning persists across database restarts
- Model management system
  - ModelManager class for automatic LLM downloads
  - HuggingFace Hub integration
  - Automatic download of phi-2.Q4_K_M.gguf model
  - Graceful fallback to rule-based translation
- 6 new unit tests for advanced operators
- 3 new integration tests for combined features

### Changed
- Updated Statement::Select to include order_by and limit fields
- Extended ExpressionEvaluator with LIKE, IN, BETWEEN support
- Modified NLTranslator to use ModelManager
- Enhanced parser to handle ORDER BY and LIMIT clauses
- Executor now performs sorting and limiting operations

### Dependencies
- Added: regex = "1.10" (Rust)
- Added: huggingface-hub>=0.20.0 (Python)

### Performance
- LIKE: ~100µs overhead per query
- IN (5 items): ~50µs overhead
- BETWEEN: ~40µs overhead
- ORDER BY (1000 rows): ~2ms
- LIMIT: ~1µs overhead
- RL save/load: ~5ms/~3ms

### Tests
- 21 tests passing (was 18)
- Zero regressions
- All existing functionality preserved

## [0.2.0] - 2025-11-25

### Added
- WHERE clause filtering with full expression evaluation
  - Comparison operators: =, >, <, >=, <=, !=
  - Logical operators: AND, OR
  - Support for Integer, Float, Text, Boolean types
- Natural Language Query interface (ASK command)
  - NLTranslator class with llama-cpp-python support
  - Fallback rule-based translator
  - Schema-aware translation
- Reinforcement Learning query optimizer
  - Q-Learning agent with state/action/reward
  - Epsilon-greedy exploration
  - Automatic learning from query performance
- ExpressionEvaluator for WHERE clause evaluation
- NLTranslator PyO3 bridge integration
- Schema introspection for NL context
- 3 new integration tests for WHERE filtering
- CLI ASK command mode

### Changed
- Updated CLI to v0.2.0 with enhanced command modes
- Extended Python AI engine with translator and RL agent
- Improved PyO3 bridge with NLTranslator export
- Enhanced README with v0.2.0 features

### Fixed
- Parser now correctly handles WHERE clause expressions
- Fixed unused variable warnings in executor

### Dependencies
- Added: llama-cpp-python>=0.2.0
- Added: diskcache (via llama-cpp-python)

### Tests
- 15 unit tests passing (was 11)
- 3 integration tests passing (new)
- Total: 18/18 tests passing

## [0.1.0] - 2025-11-25

### Added
- Initial release with core SQL database functionality
- Storage engine using sled
- SQL parser for CREATE TABLE, INSERT, SELECT
- Query executor with end-to-end workflow
- Catalog for table metadata management
- Semantic caching using sentence-transformers
- PyO3 Rust-Python integration
- CLI REPL interface
- 11 comprehensive unit tests

### Features
- Persistent KV storage
- SQL query execution
- AI-powered semantic caching
- 60x query speedup on cache hits
- Local-only execution (no cloud dependencies)

[0.4.0]: https://github.com/aviralgarg05/NexumDB/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/aviralgarg05/NexumDB/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/aviralgarg05/NexumDB/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/aviralgarg05/NexumDB/releases/tag/v0.1.0
