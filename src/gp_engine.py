"""
Genetic programming engine — DEAP toolbox configuration.

Sets up the DEAP creator types, registers all genetic operators,
and provides safe evaluation with bloat control.
"""

from __future__ import annotations

import operator
import signal
import math
from typing import Any

import numpy as np
from deap import base, creator, gp, tools


def _ensure_creator() -> None:
    """Create DEAP fitness/individual types if they don't already exist."""
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)


def setup_toolbox(pset: gp.PrimitiveSet, config: dict[str, Any]) -> base.Toolbox:
    """
    Configure a DEAP Toolbox for symbolic regression.

    Parameters
    ----------
    pset : gp.PrimitiveSet
        The primitive set defining the GP language.
    config : dict
        Hyperparameters dict.

    Returns
    -------
    base.Toolbox
        Fully configured toolbox.
    """
    _ensure_creator()
    toolbox = base.Toolbox()

    min_d = config.get("min_init_depth", 2)
    max_d = config.get("max_init_depth", 6)
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=min_d, max_=max_d)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    tournsize = config.get("tournament_size", 7)
    toolbox.register("select", tools.selTournament, tournsize=tournsize)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2, pset=pset)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    max_depth = config.get("max_tree_depth", 17)
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_depth))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_depth))

    return toolbox


class _TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _TimeoutError()


def evaluate_individual(
    individual: gp.PrimitiveTree,
    toolbox: base.Toolbox,
    X: np.ndarray,
    y: np.ndarray,
    n_vars: int = 1,
    timeout_seconds: float = 1.0,
) -> tuple[float]:
    """Evaluate individual's fitness as MSE with timeout & error protection."""
    PENALTY = 1e10

    try:
        func = toolbox.compile(expr=individual)
    except Exception:
        return (PENALTY,)

    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, timeout_seconds)

    try:
        se_sum = 0.0
        n = len(y)
        if n_vars == 1:
            for i in range(n):
                pred = func(float(X[i]))
                if not math.isfinite(pred):
                    return (PENALTY,)
                se_sum += (pred - y[i]) ** 2
        else:
            for i in range(n):
                pred = func(*[float(v) for v in X[i]])
                if not math.isfinite(pred):
                    return (PENALTY,)
                se_sum += (pred - y[i]) ** 2
        mse = se_sum / n
        if not math.isfinite(mse):
            return (PENALTY,)
        return (mse,)
    except _TimeoutError:
        return (PENALTY,)
    except Exception:
        return (PENALTY,)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def build_stats() -> tools.MultiStatistics:
    """Build MultiStatistics tracking fitness and tree size."""
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)
    return mstats
