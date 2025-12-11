# cost_model.py
def cost_scan(rows):
    """Cost for scanning a table"""
    return rows

def cost_filter(rows):
    """Cost for filtering rows"""
    return rows
def cost_nested_loop(outer_rows, inner_rows):
    """
    Simple nested-loop cost model:
    - scan outer once
    - for each outer row, scan all inner rows
    => cost ~ outer_rows + outer_rows * inner_rows
    """
    return outer_rows + outer_rows * inner_rows
