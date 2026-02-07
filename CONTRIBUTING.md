# Contributing to NexumDB

First off, thank you for considering contributing to NexumDB! It's people like you that make NexumDB such a great tool. We welcome contributions from everyone, whether you're fixing a typo, reporting a bug, proposing a new feature, or writing code.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Pull Request Process](#pull-request-process)
- [Style Guidelines](#style-guidelines)
- [Community](#community)

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

## Getting Started

### Understanding the Project

NexumDB is an AI-native database that combines:
- **Rust Core** (`nexum_core/`): Storage engine, SQL parsing, and execution
- **Python AI Engine** (`nexum_ai/`): Semantic caching, NL translation, RL optimization
- **CLI Interface** (`nexum_cli/`): Interactive REPL

Before diving in, we recommend:
1. Reading the [README.md](README.md) to understand the project's purpose
2. Running the demo: `./demo.sh`
3. Exploring the codebase structure

### Finding Something to Work On

- Look for issues labeled [`good first issue`](../../issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) for beginner-friendly tasks
- Check [`help wanted`](../../issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22) for issues where we need community help
- Review the [Development Status](#development-status) section in README for areas needing work

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates.

**When filing a bug report, include:**
- A clear, descriptive title
- Steps to reproduce the issue
- Expected behavior vs. actual behavior
- Your environment (OS, Rust version, Python version)
- Relevant logs or error messages
- Code samples if applicable

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion:

- Use a clear, descriptive title
- Provide a detailed description of the proposed functionality
- Explain why this enhancement would be useful
- List any alternatives you've considered

### Your First Code Contribution

Unsure where to begin? Start with:
1. **Documentation**: Improve docs, fix typos, add examples
2. **Tests**: Add test coverage for existing features
3. **Bug fixes**: Look for issues labeled `bug`

### Pull Requests

1. Fork the repo and create your branch from `main`
2. Follow the [Development Setup](#development-setup) guide
3. Make your changes following our [Style Guidelines](#style-guidelines)
4. Add tests for any new functionality
5. Ensure all tests pass
6. Update documentation as needed
7. Submit a pull request!

## Development Setup

### Prerequisites

- **Rust**: 1.70+ (install via [rustup](https://rustup.rs/))
- **Python**: 3.10+ with pip
- **Git**: For version control

### Setting Up Your Environment

```bash
# 1. Clone your fork
git clone https://github.com/YOUR_USERNAME/NexumDB.git
cd NexumDB

# 2. Add upstream remote
git remote add upstream https://github.com/aviralgarg05/NexumDB.git

# 3. Set PyO3 compatibility flag
export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1

# 4. Create Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 5. Install Python dependencies
pip install -r nexum_ai/requirements.txt

# 6. Build the project
cargo build

# 7. Run tests to verify setup
cargo test -- --test-threads=1
```

### Project Structure

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
│   ├── optimizer.py     # Semantic cache and RL optimizer
│   ├── translator.py    # NL to SQL translation
│   ├── rl_agent.py      # Reinforcement learning agent
│   └── model_manager.py # LLM model management
├── tests/               # Integration tests
└── .github/             # GitHub workflows and templates
```

### Running Tests

```bash
# Run all tests
cargo test -- --test-threads=1

# Run specific test
cargo test test_name -- --test-threads=1

# Run with verbose output
cargo test -- --test-threads=1 --nocapture
```

### Building for Release

```bash
cargo build --release
./target/release/nexum
```

## Pull Request Process

### Before Submitting

1. **Update your fork**: Sync with the upstream repository
   ```bash
   git fetch upstream
   git checkout main
   git merge upstream/main
   ```

2. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make focused commits**: Each commit should represent a single logical change

4. **Write meaningful commit messages**:
   ```
   feat: add LIKE operator support for pattern matching
   
   - Implement SQL LIKE operator with % and _ wildcards
   - Add filter module for pattern evaluation
   - Include comprehensive test coverage
   ```

### Commit Message Convention

We follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting, etc.)
- `refactor:` Code refactoring
- `test:` Adding or updating tests
- `chore:` Maintenance tasks

### Linting GitHub Actions

`actionlint` is a linter that checks GitHub Actions workflow files in `.github/workflows/*.yml` for syntax errors and common issues.

Run it locally using Docker:

```bash
docker run --rm -v "$(pwd):/repo" -w /repo ghcr.io/rhysd/actionlint:latest -color
```

If there is no output, it means no issues were found.


### After Submitting

1. **CI checks**: Ensure all automated checks pass
2. **Code review**: Address any feedback from reviewers
3. **Keep updated**: Rebase your branch if `main` has progressed
4. **Be patient**: Maintainers will review as soon as possible

## Style Guidelines

### Rust Code Style

- Follow the official [Rust Style Guide](https://doc.rust-lang.org/1.0.0/style/README.html)
- Use `cargo fmt` before committing
- Run `cargo clippy` to catch common mistakes
- Write documentation comments for public APIs

```rust
/// Executes a SQL query and returns results.
/// 
/// # Arguments
/// 
/// * `query` - The SQL query string to execute
/// 
/// # Returns
/// 
/// * `Result<QueryResult>` - The query results or an error
/// 
/// # Example
/// 
/// ```
/// let result = executor.execute("SELECT * FROM users")?;
/// ```
pub fn execute(&self, query: &str) -> Result<QueryResult> {
    // Implementation
}
```

### Python Code Style

- Follow [PEP 8](https://pep8.org/)
- Use type hints where appropriate
- Document functions with docstrings

```python
def translate_query(self, natural_language: str) -> str:
    """
    Translate natural language to SQL.
    
    Args:
        natural_language: The natural language query string
        
    Returns:
        The generated SQL query string
        
    Raises:
        TranslationError: If translation fails
    """
    # Implementation
```

### Documentation

- Keep README.md updated with any new features
- Add inline comments for complex logic
- Update CHANGELOG.md for notable changes

## Community

### Updating dependencies

1. Install pip-tools: `pip install pip-tools`
2. Update the lock file: `pip-compile requirements.txt -o requirements-lock.txt`
3. Commit the updated `requirements-lock.txt` to the repo


### Getting Help

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and general discussion

### Recognition

Contributors will be recognized in:
- The project's README
- Release notes when their contributions are included

## Thank You!

Your contributions make NexumDB better for everyone. We appreciate your time and effort in helping improve this project!

