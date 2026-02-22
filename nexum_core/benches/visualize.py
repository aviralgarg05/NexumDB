import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt

def parse_criterion_json(path: Path) -> float:
    """Parses mean execution time from Criterion estimates.json (returns µs)."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data['mean']['point_estimate'] / 1_000 # Convert ns to µs
    except (json.JSONDecodeError, KeyError, IOError):
        return 0.0

def find_benchmark_data(base_path: Path, targets: List[str]) -> Dict[str, float]:
    """Recursively finds benchmark data matching target keywords."""
    results = {}
    if not base_path.exists():
        return results

    for json_path in base_path.rglob("estimates.json"):
        path_str = str(json_path).lower()
        for target in targets:
            if target.lower() in path_str:
                val = parse_criterion_json(json_path)
                if val > 0:
                    results[target] = val
    return results

def plot_results(data: Dict[str, float], output_file: Optional[str] = None) -> None:
    if not data:
        print("No data found. Run 'cargo bench' first.")
        return

    labels = sorted(data.keys(), key=lambda x: "sqlite" not in x.lower())
    values = [data[label] for label in labels]

    plt.figure(figsize=(10, 6))
    colors = ['#4CAF50' if 'sqlite' in l.lower() else '#FF9800' for l in labels]
    plt.bar(labels, values, color=colors)
    plt.ylabel('Mean Latency (µs)')
    plt.title('Database Performance: SQLite vs NexumDB')
    plt.yscale('log') 
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file)
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Path is relative to Workspace Root
    parser.add_argument("--path", type=Path, default=Path("target/criterion"))
    parser.add_argument("--output", type=str, default="benchmark_results.png")
    args = parser.parse_args()

    # Keywords matching your folder names
    target_keywords = [
        "sqlite_single_insert",
        "nexumdb_single_insert",
        "sqlite_point_lookup",
        "nexumdb_point_lookup_cold",
        "nexumdb_point_lookup_cached"
    ]

    data = find_benchmark_data(args.path, target_keywords)
    plot_results(data, args.output)