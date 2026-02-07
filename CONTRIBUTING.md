# Contributing to NexumDB

First off, thank you for considering contributing to NexumDB! It's people like you that make NexumDB such a great tool. We welcome contributions from everyone, whether you're fixing a typo, reporting a bug, proposing a new feature, or writing code.

**🌟 OSCG'26 Participants Welcome!** This project is part of Open Source Connect Global (OSCG'26). We follow OSCG's collaborative approach focused on learning, quality, and sustainable open-source contribution.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Contribution Workflow](#contribution-workflow)
- [Pull Request Process](#pull-request-process)
- [Quality Standards](#quality-standards)
- [Communication Guidelines](#communication-guidelines)
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

## Contribution Workflow

**⚠️ Important**: This project follows standard open-source workflows. NEVER work directly on `main` branch.

### Step-by-Step Process

#### 1. Review the Repository
Before making any changes:
- Read the README.md thoroughly
- Check existing issues and open PRs
- Understand the project structure and setup
- Never assume how the project works

#### 2. Fork the Repository
Create your own copy using GitHub's Fork button. This ensures:
- Your changes don't affect the main project directly
- Proper review and tracking of contributions

#### 3. Clone Your Fork Locally
```bash
git clone https://github.com/<your-username>/NexumDB.git
cd NexumDB
```

#### 4. Add Upstream Remote (Recommended)
```bash
git remote add upstream https://github.com/aviralgarg05/NexumDB.git
git fetch upstream
```

#### 5. Create a Feature Branch (Mandatory)
**Never work on main or master branch**:
```bash
git checkout -b feature/short-descriptive-name
```

**Good branch names:**
- `feature/add-api-validation`
- `feature/update-docs`
- `feature/fix-login-bug`

#### 6. Make Changes Carefully
While working:
- Keep changes small and focused
- Avoid mixing unrelated fixes
- Follow existing code style and conventions
- Add comments where logic is non-obvious
- Keep documentation updated

#### 7. Test Your Changes Thoroughly
Before committing:
- Run all tests: `cargo test -- --test-threads=1`
- Test manually if needed
- Ensure no existing functionality is broken
- **Unverified changes may be rejected**

#### 8. Commit Your Work
```bash
git add .
git commit -m "Brief, clear description of the change"
```

**Good commit messages explain what changed and why.**

#### 9. Push Your Branch
```bash
git push origin feature/short-descriptive-name
```

#### 10. Create a Pull Request
When opening a PR:
- Use a clear, descriptive title
- Fill out the PR template completely
- Reference related issues if applicable
- Describe assumptions or limitations
- **A good PR description helps reviewers help you**

## Quality Standards

**Quality over quantity** - We value meaningful contributions that improve the project.

### ✅ Accepted Contributions
- **Feature improvements**: New functionality or enhancements
- **Bug fixes**: Resolving existing issues
- **Documentation updates**: Improving clarity and completeness
- **Refactoring**: Code improvements with clear justification
- **Testing**: Adding test coverage or improving test quality
- **Tooling improvements**: Better development or build processes

### ❌ Not Accepted
- **Cosmetic-only changes**: Without clear purpose or benefit
- **Copy-pasted code**: Without understanding or attribution
- **Auto-generated changes**: Without explanation or review
- **Spam or rushed PRs**: Low-quality submissions
- **Plagiarism**: Copying code without attribution

**🛑 Zero Tolerance**: Plagiarism results in immediate removal from the project.

## Communication Guidelines

### Where to Communicate
- **GitHub Issues** → Questions about bugs and feature requests
- **Pull Request comments** → Code reviews and implementation feedback
- **GitHub Discussions** → Open-ended topics and general questions

### Professional Communication Standards
**DO:**
- Be respectful and patient
- Ask thoughtful, well-researched questions
- Read documentation before asking for help
- Respond constructively to feedback
- Help other contributors when possible

**DON'T:**
- Repeatedly tag maintainers or reviewers
- Post "please merge" or "when will this be reviewed" comments
- Send multiple follow-ups within short time periods
- Message maintainers on LinkedIn for PR reviews or technical support
- Use LinkedIn for anything other than professional networking

**Remember**: Open source runs on mutual respect and patience. Maintainers are volunteers.

### Handling Reviews and Feedback
- Respond politely and clearly to all feedback
- Ask questions if reviewer suggestions are unclear
- Push fixes to the same branch (don't create new PRs)
- Don't take feedback personally - it's about the code, not you
- **PRs left inactive for extended periods may be closed**

### Learning Expectations
We encourage:
- Reading documentation thoroughly before asking questions
- Asking thoughtful questions that show you've tried to understand
- Helping other contributors when you can
- Taking ownership of your contributions

**Learning happens through participation, not shortcuts.**
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

### PR Requirements

**Before submitting your PR, ensure:**
- You've followed the [Contribution Workflow](#contribution-workflow) above
- All tests pass locally: `cargo test -- --test-threads=1`
- Code follows our [Style Guidelines](#style-guidelines)
- PR template is completely filled out
- Commits follow conventional commit format (see below)

### Commit Message Convention

We follow [Conventional Commits](https://www.conventionalcommits.org/):

**Format**: `type(scope): description`

**Types:**
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting, etc.)
- `refactor:` Code refactoring
- `test:` Adding or updating tests
- `chore:` Maintenance tasks

**Example:**
```
feat: add LIKE operator support for pattern matching

- Implement SQL LIKE operator with % and _ wildcards
- Add filter module for pattern evaluation  
- Include comprehensive test coverage
```

### PR Review Process

1. **Automated Checks**: CI must pass before human review
2. **Code Review**: Maintainers will review when available
3. **Address Feedback**: Make requested changes promptly
4. **Stay Updated**: Keep your branch updated with main if requested
5. **Be Patient**: Reviews happen as maintainer time allows

**Remember**: Maintainers are volunteers. Allow reasonable time for reviews.

### Linting GitHub Actions

`actionlint` checks workflow files for issues:

```bash
docker run --rm -v "$(pwd):/repo" -w /repo ghcr.io/rhysd/actionlint:latest -color
```

No output means no issues found.

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

## OSCG'26 Program

NexumDB is proud to be part of **Open Source Connect Global (OSCG'26)**, a community-driven initiative helping developers learn open source through real-world contributions.

### OSCG Values
- **Sustainable contribution** over quick fixes
- **Learning and growth** through collaboration  
- **Quality and impact** over quantity
- **Professional communication** and mutual respect

### Program Benefits
- Learn industry-standard open-source workflows
- Build your GitHub profile with meaningful contributions
- Connect with a global community of developers
- Gain experience with production-quality codebases

**Resources:**
- OSCG Website: [https://osconnect.org](https://osconnect.org/)
- Projects Directory: [https://www.osconnect.org/projects](https://www.osconnect.org/projects)
- Program Support: [hello@osconnect.org](mailto:hello@osconnect.org)

## Community

### Getting Help

**GitHub First - LinkedIn Never for Support:**
- **GitHub Issues**: For bugs, feature requests, and project questions
- **GitHub Discussions**: For general questions and open discussions
- **Pull Request Comments**: For code review discussions

**For OSCG Program Support:**
- Email: [hello@osconnect.org](mailto:hello@osconnect.org) with:
  - Your GitHub username
  - Project name (NexumDB) 
  - Repository link
  - Clear problem description

### Updating Dependencies

1. Install pip-tools: `pip install pip-tools`
2. Update the lock file: `pip-compile requirements.txt -o requirements-lock.txt`
3. Commit the updated `requirements-lock.txt` to the repo

### Recognition

Contributors are recognized in:
- Project README contributors list
- Release notes for significant contributions
- OSCG community showcases for exceptional contributions

### Code of Conduct Enforcement

We maintain a **zero-tolerance policy** for:
- Harassment or discrimination of any kind
- Toxic, aggressive, or disrespectful language  
- Spam, manipulation, or misrepresentation
- Plagiarism or uncredited code copying

Violations may result in removal from both the project and OSCG program.

## Final Reminders

**Open source is about long-term collaboration:**
- Take ownership of your contributions
- Respect the process and other contributors  
- Focus on learning and creating positive impact
- Be patient - meaningful work takes time

By contributing to NexumDB, you agree to follow these guidelines and help build a healthy, inclusive open-source community.

## Thank You!

Your contributions make NexumDB better for everyone. We appreciate your time, effort, and commitment to learning and growing together in the open-source community!

