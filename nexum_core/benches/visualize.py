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
        # mean point_estimate is in nanoseconds
        return data['mean']['point_estimate'] / 1_000
    except (json.JSONDecodeError, KeyError, IOError):
        return 0.0

def find_benchmark_data(base_path: Path, targets: List[str]) -> Dict[str, float]:
    """Recursively finds benchmark data by matching target substrings in paths."""
    results = {}
    print(f"Searching in: {base_path.absolute()}")

    if not base_path.exists():
        return results

    # Search for all estimates.json files recursively
    for json_path in base_path.rglob("estimates.json"):
        # Convert path to lowercase string for easier matching
        path_str = str(json_path).lower()
        
        for target in targets:
            # Match if the target name (e.g., 'nexumdb') is in the folder path
            if target.lower() in path_str:
                val = parse_criterion_json(json_path)
                if val > 0:
                    results[target] = val
                    print(f" Found match for '{target}': {val:.2f} µs")
    
    return results

def plot_results(data: Dict[str, float], output_file: Optional[str] = None) -> None:
    if not data:
        print("\n[!] No matching data found in target/criterion. Run 'cargo bench' first.")
        return

    # Sort: SQLite baseline first
    labels = sorted(data.keys(), key=lambda x: "sqlite" not in x.lower())
    values = [data[label] for label in labels]

    plt.figure(figsize=(10, 6))
    colors = ['#4CAF50' if 'sqlite' in l.lower() else '#FF9800' for l in labels]
    
    bars = plt.bar(labels, values, color=colors)
    plt.ylabel('Mean Latency (µs)')
    plt.title('Database Performance: SQLite vs NexumDB')
    plt.yscale('log') 
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:,.2f} µs', 
                 va='bottom', ha='center', fontweight='bold')

    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file)
        print(f"Graph saved to {output_file}")
    else:
        plt.show()

def main() -> None:
    parser = argparse.ArgumentParser()
    # Check current directory for target/criterion
    base = Path("target/criterion")
    parser.add_argument("--path", type=Path, default=base)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()

    # Simplified search terms based on your screenshot
    target_keywords = [
        "sqlite_single_insert",
        "nexumdb_single_insert"
    ]

    data = find_benchmark_data(args.path, target_keywords)
    plot_results(data, args.output)

if __name__ == "__main__":
    main()