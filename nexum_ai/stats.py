

class TableStats:
    def __init__(self):
        self.row_count = 0
        self.distinct = {}  # column_name -> set of distinct values

def analyze(table):
    stats = TableStats()
    for row in table.rows:
        stats.row_count += 1
        for col, val in row.items():
            stats.distinct.setdefault(col, set()).add(val)
    return stats
