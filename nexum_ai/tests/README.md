# NexumDB AI Test Suite

Comprehensive unit tests for the Python AI modules in `nexum_ai/`.

## Test Coverage

### 1. `test_model_manager.py` - Model Loading & Inference
- Model initialization and directory management
- Model file existence checking
- HuggingFace model download (mocked)
- Model listing functionality
- Error handling for missing models

### 2. `test_optimizer.py` - Query Optimization Logic
- **SemanticCache Tests:**
  - Vectorization (with and without model)
  - Cosine similarity calculations
  - Cache hit/miss scenarios
  - Cache clearing
  - Multiple cache entries
  
- **QueryOptimizer Tests:**
  - Q-learning initialization
  - Action selection (exploration vs exploitation)
  - Q-value updates
  - Metrics feeding
  - Multiple query optimization

### 3. `test_rl_agent.py` - RL Training Loop & State/Action Handling
- Agent initialization with hyperparameters
- State key generation and bucketing
- Action selection (epsilon-greedy)
- Q-value updates and learning
- Reward calculation
- Epsilon decay
- State persistence (save/load)
- Training statistics
- Integration tests with training loops

### 4. `test_translator.py` - NL to SQL Translation
- Translator initialization
- Fallback rule-based translation:
  - "Show all users" queries
  - "Show users named X" queries
  - "Show all products" queries
  - Price filter queries (less than, more than)
- Prompt building
- SQL cleaning and formatting
- LLM-based translation (mocked)
- Error handling and fallback

### 5. `test_expensive_ops.py` - Expensive Operations (Mocked)
- Model download without actual downloading
- LLM inference without loading models
- Embedding generation without GPU
- GPU fallback scenarios
- Large-scale operations:
  - Large cache operations
  - Large Q-table training
  - Batch vectorization
  - Memory efficiency tests

## Running Tests

### Run all tests:
```bash
cd nexum_ai
pytest
```

### Run with coverage:
```bash
pytest --cov=. --cov-report=html --cov-report=term-missing
```

### Run specific test file:
```bash
pytest tests/test_optimizer.py
```

### Run specific test class:
```bash
pytest tests/test_optimizer.py::TestSemanticCache
```

### Run specific test:
```bash
pytest tests/test_optimizer.py::TestSemanticCache::test_cosine_similarity_identical
```

### Run tests with verbose output:
```bash
pytest -v
```

### Run tests excluding slow tests:
```bash
pytest -m "not slow"
```

## Test Structure

```
nexum_ai/
├── conftest.py              # Pytest fixtures and configuration
├── pytest.ini               # Pytest settings
├── tests/
│   ├── __init__.py
│   ├── README.md            # This file
│   ├── requirements.txt     # Test dependencies
│   ├── test_model_manager.py
│   ├── test_optimizer.py
│   ├── test_rl_agent.py
│   ├── test_translator.py
│   └── test_expensive_ops.py
```

## Coverage Target

Target: **>80% coverage** for `nexum_ai` module

Current coverage includes:
- All public methods and functions
- Error handling paths
- Edge cases (empty inputs, None values, etc.)
- Integration scenarios

## Mocking Strategy

To avoid expensive operations during testing:
- **Model downloads**: Mocked using `unittest.mock.patch`
- **LLM inference**: Mocked Llama model responses
- **GPU operations**: Mocked to work on CPU-only CI
- **Embedding models**: Mocked SentenceTransformer

## CI Integration

Tests run automatically on:
- Push to `main` branch
- Pull requests to `main` branch

CI workflow includes:
1. Python environment setup
2. Dependency installation
3. Byte-compilation check
4. Pytest execution with coverage
5. Coverage report upload to Codecov

## Adding New Tests

When adding new functionality to `nexum_ai`:

1. Create test file: `tests/test_<module_name>.py`
2. Add test class: `class Test<ClassName>`
3. Add test methods: `def test_<functionality>(self)`
4. Use fixtures from `conftest.py` for common setup
5. Mock expensive operations (model loading, GPU, etc.)
6. Aim for >80% coverage of new code

## Dependencies

Test dependencies (in `tests/requirements.txt`):
- `pytest>=7.4.0` - Testing framework
- `pytest-cov>=4.1.0` - Coverage reporting
- `pytest-mock>=3.11.1` - Mocking utilities

## Notes

- Tests use temporary directories for file operations
- All model loading is mocked to avoid downloads
- GPU operations are mocked for CI compatibility
- Tests are designed to run quickly (<30 seconds total)
