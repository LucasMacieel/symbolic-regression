# Symbolic Regression via Genetic Programming

Experimental framework for symbolic regression using [DEAP](https://deap.readthedocs.io/)'s genetic programming module. Evaluates GP performance on classic benchmark functions with multi-run statistical analysis and publication-quality visualizations.

## Quick Start

```bash
# Install dependencies
uv sync

# Run a single benchmark
uv run python run_experiment.py --benchmark koza-1

# Run all benchmarks
uv run python run_experiment.py --all

# Custom config
uv run python run_experiment.py --benchmark nguyen-5 --config configs/custom.yaml
```

## Benchmarks

| Name | Target Function | Domain | Variables |
|------|----------------|--------|-----------|
| `koza-1` | x⁴ + x³ + x² + x | [-1, 1] | 1 |
| `nguyen-1` | x³ + x² + x | [-1, 1] | 1 |
| `nguyen-4` | x⁶ + x⁵ + x⁴ + x³ + x² + x | [-1, 1] | 1 |
| `nguyen-5` | sin(x²)·cos(x) − 1 | [-1, 1] | 1 |
| `nguyen-7` | ln(x+1) + ln(x²+1) | [0, 2] | 1 |
| `keijzer-4` | x³·e⁻ˣ·cos(x)·sin(x)·(sin²(x)·cos(x)−1) | [0, 10] | 1 |
| `pagie-1` | 1/(1+x⁻⁴) + 1/(1+y⁻⁴) | [-5, 5]² | 2 |

## Configuration

Edit `configs/default.yaml` to adjust hyperparameters:

```yaml
population_size: 500
n_generations: 50
crossover_prob: 0.9
mutation_prob: 0.1
tournament_size: 7
max_tree_depth: 17
n_runs: 30
```

## Visualizations

Generated in `results/figures/`:

- **Convergence curves** — Per-generation best/avg fitness with ±1σ bands
- **Expression trees** — Color-coded tree structure of best solution
- **Fitness vs. complexity** — Scatter plot for bloat analysis
- **Prediction overlay** — GP prediction vs. ground truth
- **Benchmark comparison** — Box plot across all benchmarks

## Project Structure

```
├── configs/default.yaml       # Experiment hyperparameters
├── run_experiment.py          # CLI entry point
├── src/
│   ├── benchmarks.py          # Benchmark function registry
│   ├── primitives.py          # Protected operators & primitive sets
│   ├── gp_engine.py           # DEAP toolbox configuration
│   ├── experiment.py          # Multi-run experiment runner
│   └── visualization.py       # All plotting & tree visualization
└── results/                   # Auto-generated outputs
    ├── figures/
    └── logs/
```
