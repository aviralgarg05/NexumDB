"""
Pytest configuration and fixtures for nexum_ai tests
"""

import pytest
import tempfile
import shutil


@pytest.fixture
def temp_models_dir():
    """Create a temporary directory for model files"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_schema():
    """Sample database schema for testing"""
    return """
    TABLE users (id INTEGER, name TEXT, age INTEGER, email TEXT)
    TABLE products (id INTEGER, name TEXT, price REAL, category TEXT)
    TABLE orders (id INTEGER, user_id INTEGER, product_id INTEGER, quantity INTEGER)
    """


@pytest.fixture
def sample_queries():
    """Sample SQL queries for testing"""
    return [
        "SELECT * FROM users WHERE age > 25",
        "SELECT name, price FROM products WHERE price < 100",
        "SELECT * FROM orders WHERE quantity > 5",
        "SELECT u.name, p.name FROM users u JOIN orders o ON u.id = o.user_id JOIN products p ON o.product_id = p.id",
    ]
