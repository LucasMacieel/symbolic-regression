"""
Protected operators and primitive set construction for genetic programming.

Provides safe mathematical operators that gracefully handle edge cases
(division by zero, sqrt of negatives, overflow, etc.) and a factory
function to build DEAP PrimitiveSets with configurable operator suites.

Physics-focused primitive set
------------------------------
The default configuration includes only the operators required by the
physics benchmarks in this project:

    Core arithmetic : add, sub, mul, protectedDiv, neg
    Square root     : protectedSqrt  (pendulum period)
"""

from __future__ import annotations

import math
import operator
import random

import numpy as np
from deap import gp


def _rand_const() -> float:
    """Generate a random ephemeral constant in [-1, 1]."""
    return round(random.uniform(-1, 1), 4)


# ──────────────────────────────────────────────────────────────────────────────
# Protected operators
# ──────────────────────────────────────────────────────────────────────────────


def protectedDiv(left, right):
    """Division protected against zero-division (vectorized)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        if np.isscalar(right):
            if abs(right) < 1e-10:
                return 1.0
            return left / right

        right_abs = np.abs(right)
        mask = right_abs < 1e-10
        safe_right = np.where(mask, 1.0, right)
        res = left / safe_right
        return np.where(mask, 1.0, res)


def protectedSqrt(x):
    """Square root protected against negative arguments (vectorized)."""
    if np.isscalar(x):
        return math.sqrt(abs(x))
    return np.sqrt(np.abs(x))


def protectedSin(x):
    """Sine function (vectorized)."""
    if np.isscalar(x):
        return math.sin(x)
    return np.sin(x)


# ──────────────────────────────────────────────────────────────────────────────
# Primitive set factory
# ──────────────────────────────────────────────────────────────────────────────

_VAR_NAMES = ["x", "y", "z", "u", "v", "w", "q"]


def build_primitive_set(
    n_vars: int = 1,
) -> gp.PrimitiveSet:
    """
    Build a DEAP PrimitiveSet with configurable operator suites.

    Parameters
    ----------
    n_vars : int
        Number of input variables.

    Returns
    -------
    gp.PrimitiveSet
        Fully configured primitive set ready for GP.
    """
    pset = gp.PrimitiveSet("MAIN", arity=n_vars)

    # Core arithmetic (always included)
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(protectedDiv, 2)
    pset.addPrimitive(operator.neg, 1)

    # Square root — needed for pendulum period
    pset.addPrimitive(protectedSqrt, 1)

    # Trigonometry — needed for projectile range
    pset.addPrimitive(protectedSin, 1)

    # Ephemeral random constant in [-1, 1]
    pset.addEphemeralConstant("rand_const", _rand_const)

    # Rename variables from ARG0, ARG1, ... to x, y, ...
    rename_map = {}
    for i in range(n_vars):
        rename_map[f"ARG{i}"] = _VAR_NAMES[i] if i < len(_VAR_NAMES) else f"x{i}"
    pset.renameArguments(**rename_map)

    return pset
