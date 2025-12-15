from nexum_ai.stats import TableStats
from nexum_ai.optimizer import estimate_selectivity, estimate_cardinality

# --- Create fake table stats ---
table_stats = TableStats()
table_stats.row_count = 1000
table_stats.distinct = {"x": set(range(10)), "y": set(range(5))}

# --- Test estimate_selectivity ---
sel_x = estimate_selectivity(table_stats, "x")
sel_y = estimate_selectivity(table_stats, "y")
sel_missing = estimate_selectivity(table_stats, "z")  # column not in table

print(f"Selectivity x: {sel_x}")         # Expected: 0.1
print(f"Selectivity y: {sel_y}")         # Expected: 0.2
print(f"Selectivity missing: {sel_missing}")  # Expected: 0.1 fallback

# --- Test estimate_cardinality ---
card_x = estimate_cardinality(table_stats, sel_x)
print(f"Cardinality for x: {card_x}")    # Expected: 1000 * 0.1 = 100.0
