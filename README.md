# Symbolic Regression via Genetic Programming

Experimental framework for symbolic regression using [DEAP](https://deap.readthedocs.io/)'s genetic programming module. The framework evaluates GP performance on the rediscovery of fundamental physical laws from noiseless simulation data, featuring multi-run statistical analysis, robust SymPy-based algebraic simplification, and publication-quality visualizations.

## Quick Start

```bash
# Install dependencies
uv sync

# Run a single benchmark
uv run python run_experiment.py --benchmark mass-energy-equivalence

# Run all benchmarks
uv run python run_experiment.py --all

# Custom config
uv run python run_experiment.py --benchmark kinetic-energy --config configs/custom.yaml
```

## Benchmarks

The benchmark suite targets the rediscovery of fundamental physical laws.

| Name | Target Function | Domain | Variables |
|------|-----------------|--------|-----------|
| `mass-energy-equivalence` | m·c² (x=m, c=3.0) | [0.1, 10.0] | 1 |
| `newtons-second-law` | m·a | [0.1, 10.0] | 2 |
| `kinetic-energy` | ½·m·v² | [0.1, 10.0] | 2 |
| `newtons-gravity` | G·m₁·m₂/r² (G=1) | [0.1, 5.0] | 3 |
| `keplers-third-law` | 2π·√(a³/GM) (G=1) | [0.1, 10.0] | 2 |
| `stefan-boltzmann` | σ·A·T⁴ (σ=5.67) | [0.1, 5.0] | 2 |
| `projectile-range` | v₀²·sin(2θ)/g (g=9.81) | [0.1, 5.0] | 2 |
| `bernoullis-equation` | P₁ + ½ρ(v₁² - v₂²) + ρg(h₁ - h₂) | [0.1, 5.0] | 6 |

## Configuration

Edit `configs/default.yaml` to adjust hyperparameters:

```yaml
population_size: 500
n_generations: 50
crossover_prob: 0.9
mutation_prob: 0.1
tournament_size: 7
parsimony_pressure: 1.4
max_tree_depth: 17
min_init_depth: 2
max_init_depth: 6
n_runs: 30
```

## Features & Visualizations

- **SymPy Integration** — Multi-strategy algebraic simplification of GP trees with variable mapping to human-readable symbols.
- **Convergence curves** — Per-generation best/avg fitness with ±1σ bands.
- **Expression trees** — Color-coded tree structure of the best simplified solution.
- **Fitness vs. complexity** — Scatter plot for bloat analysis, correctly visualizing individual run data.
- **Prediction overlay** — GP prediction vs. ground truth.
- **Benchmark comparison** — Box plot across all benchmarks.

## Project Structure

```text
├── configs/default.yaml       # Experiment hyperparameters
├── run_experiment.py          # CLI entry point
├── src/
│   ├── benchmarks.py          # Physics benchmark functions registry
│   ├── primitives.py          # Protected operators & primitive sets
│   ├── gp_engine.py           # DEAP toolbox configuration
│   ├── experiment.py          # Multi-run experiment runner
│   ├── simplify.py            # SymPy algebraic simplification
│   └── visualization.py       # Plotting, mathematical notation & tree visualization
└── results/                   # Auto-generated outputs
    ├── figures/
    └── logs/
```
