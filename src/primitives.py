"""
Protected operators and primitive set construction for genetic programming.

Provides safe mathematical operators that gracefully handle edge cases
(division by zero, log of negatives, overflow, etc.) and a factory
function to build DEAP PrimitiveSets with configurable operator suites.
"""

from __future__ import annotations

import math
import operator
import random

from deap import gp


def _rand_const() -> float:
    """Generate a random ephemeral constant in [-1, 1]."""
    return round(random.uniform(-1, 1), 4)


# ──────────────────────────────────────────────────────────────────────────────
# Protected operators
# ──────────────────────────────────────────────────────────────────────────────

def protectedDiv(left: float, right: float) -> float:
    """Division protected against zero-division."""
    if abs(right) < 1e-10:
        return 1.0
    return left / right


def protectedLog(x: float) -> float:
    """Natural logarithm protected against non-positive arguments."""
    if abs(x) < 1e-10:
        return 0.0
    return math.log(abs(x))


def protectedSqrt(x: float) -> float:
    """Square root protected against negative arguments."""
    return math.sqrt(abs(x))


def protectedExp(x: float) -> float:
    """Exponential clamped to avoid overflow."""
    try:
        return math.exp(min(x, 100.0))
    except OverflowError:
        return math.exp(100.0)


# ──────────────────────────────────────────────────────────────────────────────
# Primitive set factory
# ──────────────────────────────────────────────────────────────────────────────

_VAR_NAMES = ["x", "y", "z", "w"]


def build_primitive_set(
    n_vars: int = 1,
    include_trig: bool = True,
    include_exp_log: bool = True,
) -> gp.PrimitiveSet:
    """
    Build a DEAP PrimitiveSet with configurable operator suites.

    Parameters
    ----------
    n_vars : int
        Number of input variables (1 or 2 typically).
    include_trig : bool
        Whether to include sin/cos primitives.
    include_exp_log : bool
        Whether to include exp/log/sqrt primitives.

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

    # Trigonometric
    if include_trig:
        pset.addPrimitive(math.sin, 1)
        pset.addPrimitive(math.cos, 1)

    # Transcendental
    if include_exp_log:
        pset.addPrimitive(protectedLog, 1)
        pset.addPrimitive(protectedExp, 1)
        pset.addPrimitive(protectedSqrt, 1)

    # Ephemeral random constant in [-1, 1]
    pset.addEphemeralConstant("rand_const", _rand_const)

    # Rename variables from ARG0, ARG1, ... to x, y, ...
    rename_map = {}
    for i in range(n_vars):
        rename_map[f"ARG{i}"] = _VAR_NAMES[i] if i < len(_VAR_NAMES) else f"x{i}"
    pset.renameArguments(**rename_map)

    return pset
