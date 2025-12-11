# test_optimizer.py

from optimizer.stats import analyze
from optimizer.optimizer import explain

# Dummy in-memory tables
class Table:
    def __init__(self, name, rows):
        self.name = name
        self.rows = rows

# Example tables
A = Table("A", [{"x": i} for i in range(100)])
B = Table("B", [{"y": i} for i in range(10)])

# Analyze tables
stats = {
    "A": analyze(A),
    "B": analyze(B)
}

# Fake query object
class Query:
    def __init__(self, tables, stats):
        self.tables = tables
        self.stats = stats

query = Query(["A", "B"], stats)

# Run EXPLAIN
explain(query)
