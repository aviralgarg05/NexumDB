# test_optimizer.py

from nexum_ai.stats import analyze
from nexum_ai.optimizer import explain, optimize



def test_explain_output():
    # Setup in-memory tables
    class Table:
        def __init__(self, name, rows):
            self.name = name
            self.rows = rows

    A = Table("A", [{"x": i} for i in range(100)])
    B = Table("B", [{"y": i} for i in range(10)])

    # Analyze tables
    stats = {"A": analyze(A), "B": analyze(B)}

    # Create a fake query object
    class Query:
        def __init__(self, tables, stats):
            self.tables = tables
            self.stats = stats

    query = Query(["A", "B"], stats)

    # Test optimize() returns correct types
    plan, cost = optimize(query)
    assert isinstance(plan, str)
    assert isinstance(cost, (int, float))

    # Print EXPLAIN output (optional for CLI/demo)
    explain(query)

# Run the test only if file executed directly
if __name__ == "__main__":
    test_explain_output()
