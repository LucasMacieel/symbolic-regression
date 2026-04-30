"""
Visualization module for GP symbolic regression experiments.

Provides publication-quality plots for convergence analysis, tree
structure, fitness-complexity trade-offs, and cross-benchmark comparison.
Uses Graphviz 'dot' engine (via pydot) for tree layout.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
from deap import gp

from src.benchmarks import Benchmark


# ──────────────────────────────────────────────────────────────────────────────
# Style setup
# ──────────────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "text.color": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "grid.color": "#21262d",
    "grid.alpha": 0.6,
    "legend.facecolor": "#161b22",
    "legend.edgecolor": "#30363d",
    "legend.labelcolor": "#c9d1d9",
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": "#0d1117",
})

# Color palette
COLORS = {
    "primary": "#58a6ff",
    "secondary": "#f78166",
    "accent": "#7ee787",
    "warning": "#d29922",
    "purple": "#bc8cff",
    "pink": "#f778ba",
    "gradient": ["#58a6ff", "#7ee787", "#f78166", "#bc8cff", "#d29922", "#f778ba"],
}


# ──────────────────────────────────────────────────────────────────────────────
# 1. Convergence plot
# ──────────────────────────────────────────────────────────────────────────────

def plot_convergence(
    runs: list[dict[str, Any]],
    benchmark: Benchmark,
    save_path: Path,
) -> None:
    """
    Plot per-generation best/avg fitness across all runs with std shading.
    """
    logbook0 = runs[0]["logbook"]
    n_gens = len(logbook0.chapters["fitness"])

    # Extract per-generation stats from logbook chapters
    all_min = np.zeros((len(runs), n_gens))
    all_avg = np.zeros((len(runs), n_gens))
    all_size = np.zeros((len(runs), n_gens))

    for i, run in enumerate(runs):
        fit_chapter = run["logbook"].chapters["fitness"]
        size_chapter = run["logbook"].chapters["size"]
        for g in range(n_gens):
            all_min[i, g] = fit_chapter[g]["min"]
            all_avg[i, g] = fit_chapter[g]["avg"]
            all_size[i, g] = size_chapter[g]["avg"]

    gens = np.arange(n_gens)
    min_mean = np.mean(all_min, axis=0)
    min_std = np.std(all_min, axis=0)
    avg_mean = np.mean(all_avg, axis=0)
    avg_std = np.std(all_avg, axis=0)
    size_mean = np.mean(all_size, axis=0)
    size_std = np.std(all_size, axis=0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Fitness convergence
    ax1.semilogy(gens, min_mean, color=COLORS["primary"], linewidth=2, label="Best (min)")
    ax1.fill_between(
        gens,
        np.maximum(min_mean - min_std, 1e-15),
        min_mean + min_std,
        alpha=0.2,
        color=COLORS["primary"],
    )
    ax1.semilogy(gens, avg_mean, color=COLORS["secondary"], linewidth=1.5, alpha=0.8, label="Average")
    ax1.fill_between(
        gens,
        np.maximum(avg_mean - avg_std, 1e-15),
        avg_mean + avg_std,
        alpha=0.15,
        color=COLORS["secondary"],
    )
    ax1.set_ylabel("Fitness (MSE)")
    ax1.set_title(f"Convergence — {benchmark.name}: {benchmark.description}")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Tree size evolution
    ax2.plot(gens, size_mean, color=COLORS["accent"], linewidth=2, label="Avg tree size")
    ax2.fill_between(
        gens,
        np.maximum(size_mean - size_std, 0),
        size_mean + size_std,
        alpha=0.2,
        color=COLORS["accent"],
    )
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Tree Size (nodes)")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Tree visualization (Graphviz dot layout via pydot)
# ──────────────────────────────────────────────────────────────────────────────

def plot_tree(
    individual: gp.PrimitiveTree,
    benchmark: Benchmark,
    save_path: Path,
    pset: gp.PrimitiveSet | None = None,
) -> None:
    """
    Render a GP expression tree with color-coded nodes.

    Uses Graphviz 'dot' engine for hierarchical tree layout.
    Colors: operators=blue, variables=green, constants=orange.
    """
    from networkx.drawing.nx_pydot import graphviz_layout

    nodes, edges, labels = gp.graph(individual)

    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    # Classify node types for coloring
    node_colors = []
    for n in nodes:
        label = labels[n]
        if isinstance(label, str) and label in ("x", "y", "z", "w"):
            node_colors.append(COLORS["accent"])  # variables
        elif isinstance(label, (int, float)):
            node_colors.append(COLORS["warning"])  # constants
        else:
            try:
                float(label)
                node_colors.append(COLORS["warning"])  # numeric string constants
            except (ValueError, TypeError):
                node_colors.append(COLORS["primary"])  # operators

    # Graphviz dot layout — produces clean hierarchical tree positioning
    pos = graphviz_layout(G, prog="dot", root=nodes[0])

    fig, ax = plt.subplots(figsize=(max(10, len(nodes) * 0.5), max(6, individual.height * 1.5)))

    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#30363d", width=1.5)
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=900,
        edgecolors="#0d1117",
        linewidths=2,
    )

    # Format labels
    display_labels = {}
    for n, label in labels.items():
        if isinstance(label, float):
            display_labels[n] = f"{label:.2f}"
        else:
            s = str(label)
            # Shorten function names
            s = s.replace("protectedDiv", "÷")
            s = s.replace("protectedLog", "ln")
            s = s.replace("protectedExp", "exp")
            s = s.replace("protectedSqrt", "√")
            display_labels[n] = s

    nx.draw_networkx_labels(
        G, pos, display_labels, ax=ax,
        font_size=9, font_color="#0d1117", font_weight="bold",
    )

    # Legend
    legend_items = [
        mpatches.Patch(color=COLORS["primary"], label="Operator"),
        mpatches.Patch(color=COLORS["accent"], label="Variable"),
        mpatches.Patch(color=COLORS["warning"], label="Constant"),
    ]
    ax.legend(handles=legend_items, loc="upper left", fontsize=9)

    from src.simplify import format_expression
    simplified_str = format_expression(individual)
    if len(simplified_str) > 80:
        simplified_str = simplified_str[:77] + "..."
    ax.set_title(
        f"Best Tree — {benchmark.name}\nf(x) = {simplified_str}",
        fontsize=11,
        pad=15,
    )
    ax.axis("off")

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# 3. Fitness vs. Complexity scatter
# ──────────────────────────────────────────────────────────────────────────────

def plot_fitness_vs_complexity(
    runs: list[dict[str, Any]],
    benchmark: Benchmark,
    save_path: Path,
) -> None:
    """Scatter plot of best fitness vs. tree size across runs."""
    fitnesses = [r["best_fitness"] for r in runs]
    sizes = [r["best_size"] for r in runs]

    fig, ax = plt.subplots(figsize=(8, 6))

    scatter = ax.scatter(
        sizes, fitnesses,
        c=fitnesses, cmap="cool", s=80, alpha=0.8,
        edgecolors="#0d1117", linewidths=1,
        zorder=3,
    )
    cbar = fig.colorbar(scatter, ax=ax, label="Fitness (MSE)")

    ax.set_xlabel("Tree Size (nodes)")
    ax.set_ylabel("Fitness (MSE)")
    ax.set_yscale("log")
    ax.set_title(f"Fitness vs. Complexity — {benchmark.name}")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# 4. Prediction vs. ground truth
# ──────────────────────────────────────────────────────────────────────────────

def plot_prediction(
    individual: gp.PrimitiveTree,
    toolbox: Any,
    benchmark: Benchmark,
    save_path: Path,
) -> None:
    """Overlay best individual's predictions on the ground-truth function."""
    if benchmark.n_vars != 1:
        return  # Only plot 1D benchmarks

    func = toolbox.compile(expr=individual)
    x_dense = np.linspace(benchmark.domain[0], benchmark.domain[1], 200)
    y_true = np.array([benchmark.func(xi) for xi in x_dense])

    y_pred = []
    for xi in x_dense:
        try:
            val = func(float(xi))
            if not math.isfinite(val):
                val = np.nan
        except Exception:
            val = np.nan
        y_pred.append(val)
    y_pred = np.array(y_pred)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(x_dense, y_true, color=COLORS["primary"], linewidth=2.5, label="Ground truth", zorder=2)
    ax.plot(x_dense, y_pred, color=COLORS["secondary"], linewidth=2, linestyle="--", label="GP prediction", zorder=3)
    ax.scatter(
        benchmark.X, benchmark.y,
        color=COLORS["accent"], s=40, zorder=4,
        label="Training points", edgecolors="#0d1117", linewidths=0.5,
    )

    # Use SymPy-simplified expression if available
    from src.simplify import format_expression
    simplified_str = format_expression(individual)
    if len(simplified_str) > 80:
        simplified_str = simplified_str[:77] + "..."
    ax.text(
        0.02, 0.98, f"f(x) = {simplified_str}",
        transform=ax.transAxes, fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#21262d", edgecolor="#30363d", alpha=0.9),
    )

    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_title(f"Prediction vs. Ground Truth — {benchmark.name}: {benchmark.description}")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# 5. Cross-benchmark comparison (box plot)
# ──────────────────────────────────────────────────────────────────────────────

def plot_benchmark_comparison(
    all_results: dict[str, dict[str, Any]],
    save_path: Path,
) -> None:
    """Box plot comparing fitness distributions across benchmarks."""
    names = sorted(all_results.keys())
    data = []
    for name in names:
        fitnesses = [r["best_fitness"] for r in all_results[name]["runs"]]
        data.append(fitnesses)

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.5), 6))

    bp = ax.boxplot(
        data, labels=names, patch_artist=True,
        medianprops=dict(color="#f0f6fc", linewidth=2),
        whiskerprops=dict(color="#8b949e"),
        capprops=dict(color="#8b949e"),
        flierprops=dict(markerfacecolor=COLORS["secondary"], markeredgecolor=COLORS["secondary"], markersize=4),
    )

    for i, box in enumerate(bp["boxes"]):
        color = COLORS["gradient"][i % len(COLORS["gradient"])]
        box.set_facecolor(color)
        box.set_alpha(0.7)
        box.set_edgecolor("#f0f6fc")

    ax.set_yscale("log")
    ax.set_ylabel("Best Fitness (MSE)")
    ax.set_title("Cross-Benchmark Comparison")
    ax.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=30, ha="right")

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Master plot generator
# ──────────────────────────────────────────────────────────────────────────────

def generate_all_plots(
    experiment_result: dict[str, Any],
    results_dir: Path,
) -> list[Path]:
    """Generate all plots for a single benchmark experiment. Returns saved paths."""
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    benchmark = experiment_result["benchmark"]
    runs = experiment_result["runs"]
    toolbox = experiment_result["toolbox"]
    best_idx = int(np.argmin([r["best_fitness"] for r in runs]))
    best_ind = runs[best_idx]["hof"][0]

    paths = []

    # 1. Convergence
    p = fig_dir / f"{benchmark.name}_convergence.png"
    plot_convergence(runs, benchmark, p)
    paths.append(p)
    print(f"    ✓ {p.name}")

    # 2. Tree
    p = fig_dir / f"{benchmark.name}_tree.png"
    plot_tree(best_ind, benchmark, p)
    paths.append(p)
    print(f"    ✓ {p.name}")

    # 3. Fitness vs. complexity
    p = fig_dir / f"{benchmark.name}_fitness_vs_complexity.png"
    plot_fitness_vs_complexity(runs, benchmark, p)
    paths.append(p)
    print(f"    ✓ {p.name}")

    # 4. Prediction
    if benchmark.n_vars == 1:
        p = fig_dir / f"{benchmark.name}_prediction.png"
        plot_prediction(best_ind, toolbox, benchmark, p)
        paths.append(p)
        print(f"    ✓ {p.name}")

    return paths
