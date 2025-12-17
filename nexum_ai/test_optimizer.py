import unittest
from nexum_ai.stats import TableStats
from nexum_ai.optimizer import estimate_selectivity, estimate_cardinality


class TestOptimizerHelpers(unittest.TestCase):
    def setUp(self):
        """Create fake table stats for testing."""
        self.table_stats = TableStats()
        self.table_stats.row_count = 1000
        self.table_stats.distinct = {
            "x": set(range(10)),
            "y": set(range(5)),
        }

    def test_estimate_selectivity_with_distinct_values(self):
        sel_x = estimate_selectivity(self.table_stats, "x")
        sel_y = estimate_selectivity(self.table_stats, "y")

        self.assertAlmostEqual(sel_x, 0.1, places=6)
        self.assertAlmostEqual(sel_y, 0.2, places=6)

    def test_estimate_selectivity_fallback_for_missing_column(self):
        sel_missing = estimate_selectivity(self.table_stats, "z")
        self.assertEqual(sel_missing, 0.1)

    def test_estimate_cardinality(self):
        sel_x = estimate_selectivity(self.table_stats, "x")
        card_x = estimate_cardinality(self.table_stats, sel_x)

        self.assertAlmostEqual(card_x, 100.0, places=6)

    def test_optimizer_helpers_prefer_lower_cardinality(self):
        """
        Beginner-level higher test:
        lower selectivity should lead to lower estimated cardinality.
        """

        sel_x = estimate_selectivity(self.table_stats, "x")  # 0.1
        sel_y = estimate_selectivity(self.table_stats, "y")  # 0.2

        card_x = estimate_cardinality(self.table_stats, sel_x)
        card_y = estimate_cardinality(self.table_stats, sel_y)

        # Optimizer intuition: smaller cardinality is cheaper
        self.assertLess(card_x, card_y)

if __name__ == "__main__":
    unittest.main()
