"""
Experiment runner — orchestrates multi-run GP experiments.

Handles seeding, evolution loops, result collection, and CSV logging.
"""

from __future__ import annotations

import csv
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
from deap import algorithms, tools

from src.benchmarks import Benchmark, get_benchmark
from src.primitives import build_primitive_set
from src.gp_engine import setup_toolbox, evaluate_individual, build_stats
from src.simplify import simplify_individual


def run_single(
    toolbox: tools.Toolbox,
    benchmark: Benchmark,
    config: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    """
    Run a single GP evolution and return results.

    Returns
    -------
    dict with keys:
        best_fitness, best_expr, best_size, best_depth,
        logbook, hof, elapsed_seconds
    """
    random.seed(seed)
    np.random.seed(seed)

    pop_size = config.get("population_size", 500)
    n_gen = config.get("n_generations", 50)
    cx_prob = config.get("crossover_prob", 0.9)
    mut_prob = config.get("mutation_prob", 0.1)

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    stats = build_stats()

    # Register evaluation with benchmark data
    toolbox.register(
        "evaluate",
        evaluate_individual,
        toolbox=toolbox,
        X=benchmark.X,
        y=benchmark.y,
        n_vars=benchmark.n_vars,
    )

    t0 = time.time()
    pop, logbook = algorithms.eaSimple(
        pop,
        toolbox,
        cxpb=cx_prob,
        mutpb=mut_prob,
        ngen=n_gen,
        stats=stats,
        halloffame=hof,
        verbose=False,
    )
    elapsed = time.time() - t0

    best = hof[0]
    return {
        "best_fitness": best.fitness.values[0],
        "best_expr": str(best),
        "best_size": len(best),
        "best_depth": best.height,
        "logbook": logbook,
        "hof": hof,
        "elapsed_seconds": elapsed,
    }


def run_experiment(
    benchmark_name: str,
    config: dict[str, Any],
    results_dir: Path,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Run a full multi-run experiment on a single benchmark.

    Parameters
    ----------
    benchmark_name : str
        Name of the benchmark to evaluate.
    config : dict
        Experiment hyperparameters.
    results_dir : Path
        Directory for saving results.
    verbose : bool
        Whether to print progress.

    Returns
    -------
    dict with keys:
        benchmark, config, runs (list of per-run results),
        summary (aggregate statistics)
    """
    benchmark = get_benchmark(benchmark_name)
    n_runs = config.get("n_runs", 30)

    pset = build_primitive_set(
        n_vars=benchmark.n_vars,
        include_trig=config.get("include_trig", True),
        include_exp_log=config.get("include_exp_log", True),
    )
    toolbox = setup_toolbox(pset, config)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Benchmark: {benchmark.name} — {benchmark.description}")
        print(f"  Runs: {n_runs} | Pop: {config.get('population_size', 500)}"
              f" | Gens: {config.get('n_generations', 50)}")
        print(f"{'='*60}")

    runs = []
    for i in range(n_runs):
        seed = 42 + i
        result = run_single(toolbox, benchmark, config, seed)
        runs.append(result)
        if verbose:
            print(
                f"  Run {i+1:3d}/{n_runs}: "
                f"fitness={result['best_fitness']:.6e}  "
                f"size={result['best_size']:3d}  "
                f"time={result['elapsed_seconds']:.1f}s"
            )

    # Aggregate statistics
    fitnesses = [r["best_fitness"] for r in runs]
    sizes = [r["best_size"] for r in runs]
    best_idx = int(np.argmin(fitnesses))

    # Simplify best expression with SymPy
    best_individual = runs[best_idx]["hof"][0]
    simp = simplify_individual(best_individual)

    summary = {
        "mean_fitness": float(np.mean(fitnesses)),
        "std_fitness": float(np.std(fitnesses)),
        "min_fitness": float(np.min(fitnesses)),
        "max_fitness": float(np.max(fitnesses)),
        "median_fitness": float(np.median(fitnesses)),
        "mean_size": float(np.mean(sizes)),
        "std_size": float(np.std(sizes)),
        "best_overall_expr": runs[best_idx]["best_expr"],
        "best_overall_fitness": runs[best_idx]["best_fitness"],
        "simplified_expr": simp["simplified_str"],
        "simplified_latex": simp["latex"],
        "simplified_complexity": simp["complexity"],
        "simplification_strategy": simp.get("strategy", "none"),
    }

    if verbose:
        print(f"\n  Summary: {summary['mean_fitness']:.6e} "
              f"± {summary['std_fitness']:.6e}")
        print(f"  Best: {summary['best_overall_fitness']:.6e}")
        print(f"  Raw:  {summary['best_overall_expr']}")
        print(f"  SymPy: {summary['simplified_expr']}")
        print(f"  LaTeX: {summary['simplified_latex']}")

    # Save per-run CSV log
    log_dir = results_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = log_dir / f"{benchmark.name}_runs.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run", "seed", "best_fitness", "best_size",
                "best_depth", "elapsed_seconds", "best_expr",
                "simplified_expr",
            ],
        )
        writer.writeheader()
        for i, r in enumerate(runs):
            simp_r = simplify_individual(r["hof"][0])
            writer.writerow({
                "run": i + 1,
                "seed": 42 + i,
                "best_fitness": r["best_fitness"],
                "best_size": r["best_size"],
                "best_depth": r["best_depth"],
                "elapsed_seconds": r["elapsed_seconds"],
                "best_expr": r["best_expr"],
                "simplified_expr": simp_r["simplified_str"],
            })

    if verbose:
        print(f"  Saved: {csv_path}")

    return {
        "benchmark": benchmark,
        "config": config,
        "runs": runs,
        "summary": summary,
        "toolbox": toolbox,
    }
