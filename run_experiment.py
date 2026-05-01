#!/usr/bin/env python3
"""
CLI entry point for GP symbolic regression experiments.

Usage:
    uv run python run_experiment.py --benchmark koza-1
    uv run python run_experiment.py --all
    uv run python run_experiment.py --benchmark nguyen-5 --config configs/custom.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from src.benchmarks import list_benchmarks, BENCHMARKS
from src.experiment import run_experiment
from src.visualization import generate_all_plots, plot_benchmark_comparison


def load_config(path: str) -> dict:
    """Load YAML configuration file."""
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Symbolic Regression via Genetic Programming",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available benchmarks: {', '.join(list_benchmarks())}",
    )
    parser.add_argument(
        "--benchmark",
        "-b",
        type=str,
        help="Name of benchmark to run (e.g., electric-power, gravitational-pe)",
    )
    parser.add_argument(
        "--all",
        "-a",
        action="store_true",
        help="Run all registered benchmarks",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config file (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--results-dir",
        "-r",
        type=str,
        default="results",
        help="Directory for results output (default: results/)",
    )
    parser.add_argument(
        "--test-run",
        action="store_true",
        help="Only do a single run on a fixed seed for regression testing",
    )

    args = parser.parse_args()

    if not args.benchmark and not args.all:
        parser.print_help()
        print("\nError: specify --benchmark NAME or --all")
        sys.exit(1)

    config = load_config(args.config)

    if args.test_run:
        config["n_runs"] = 1
        config["seed"] = 42
        print(
            "\n  [TEST RUN] Executing a single run on a fixed seed for regression testing."
        )

    results_dir = Path(args.results_dir)

    # Determine which benchmarks to run
    if args.all:
        benchmark_names = list_benchmarks()
    else:
        benchmark_names = [args.benchmark]

    all_results = {}

    for name in benchmark_names:
        result = run_experiment(name, config, results_dir)
        all_results[name] = result

        # Generate per-benchmark plots
        print(f"\n  Generating plots for {name}...")
        generate_all_plots(result, results_dir)

    # Cross-benchmark comparison if multiple benchmarks
    if len(all_results) > 1:
        print("\n  Generating cross-benchmark comparison...")
        comp_path = results_dir / "figures" / "benchmark_comparison.png"
        plot_benchmark_comparison(all_results, comp_path)
        print(f"    ✓ {comp_path.name}")

    print(f"\n{'=' * 60}")
    print(f"  All results saved to: {results_dir.resolve()}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
