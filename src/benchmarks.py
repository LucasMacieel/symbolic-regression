"""
Benchmark target functions for symbolic regression.

Each benchmark is a dataclass containing the ground-truth function,
sampling domain, number of variables, and pre-generated training data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class Benchmark:
    """A symbolic regression benchmark problem."""

    name: str
    func: Callable  # ground-truth f(x) or f(x,y)
    domain: tuple[float, float]
    n_points: int
    n_vars: int
    description: str = ""
    X: np.ndarray = field(init=False, repr=False)
    y: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Generate training data from the ground-truth function."""
        if self.n_vars == 1:
            self.X = np.linspace(self.domain[0], self.domain[1], self.n_points)
            self.y = np.array([self.func(x) for x in self.X])
        elif self.n_vars == 2:
            n_side = int(math.isqrt(self.n_points))
            xs = np.linspace(self.domain[0], self.domain[1], n_side)
            ys = np.linspace(self.domain[0], self.domain[1], n_side)
            grid_x, grid_y = np.meshgrid(xs, ys)
            self.X = np.column_stack([grid_x.ravel(), grid_y.ravel()])
            self.y = np.array(
                [self.func(row[0], row[1]) for row in self.X]
            )
        else:
            raise ValueError(f"n_vars={self.n_vars} not supported (use 1 or 2)")


# ──────────────────────────────────────────────────────────────────────────────
# Benchmark definitions
# ──────────────────────────────────────────────────────────────────────────────

def _koza1(x: float) -> float:
    """x⁴ + x³ + x² + x"""
    return x**4 + x**3 + x**2 + x


def _nguyen1(x: float) -> float:
    """x³ + x² + x"""
    return x**3 + x**2 + x


def _nguyen4(x: float) -> float:
    """x⁶ + x⁵ + x⁴ + x³ + x² + x"""
    return x**6 + x**5 + x**4 + x**3 + x**2 + x


def _nguyen5(x: float) -> float:
    """sin(x²)·cos(x) − 1"""
    return math.sin(x**2) * math.cos(x) - 1


def _nguyen7(x: float) -> float:
    """ln(x + 1) + ln(x² + 1)"""
    return math.log(x + 1) + math.log(x**2 + 1)


def _keijzer4(x: float) -> float:
    """x³·e⁻ˣ·cos(x)·sin(x)·(sin²(x)·cos(x) − 1)"""
    return (
        x**3
        * math.exp(-x)
        * math.cos(x)
        * math.sin(x)
        * (math.sin(x) ** 2 * math.cos(x) - 1)
    )


def _pagie1(x: float, y: float) -> float:
    """1/(1 + x⁻⁴) + 1/(1 + y⁻⁴)"""
    # Protected against x=0 or y=0
    x_term = 1.0 / (1.0 + abs(x) ** (-4)) if x != 0 else 0.0
    y_term = 1.0 / (1.0 + abs(y) ** (-4)) if y != 0 else 0.0
    return x_term + y_term


# ──────────────────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────────────────

BENCHMARKS: dict[str, Benchmark] = {
    "nguyen-1": Benchmark(
        name="nguyen-1",
        func=_nguyen1,
        domain=(-1.0, 1.0),
        n_points=20,
        n_vars=1,
        description="x³ + x² + x",
    ),
    "nguyen-5": Benchmark(
        name="nguyen-5",
        func=_nguyen5,
        domain=(-1.0, 1.0),
        n_points=20,
        n_vars=1,
        description="sin(x²)·cos(x) - 1",
    ),
    "nguyen-7": Benchmark(
        name="nguyen-7",
        func=_nguyen7,
        domain=(0.0, 2.0),
        n_points=20,
        n_vars=1,
        description="ln(x + 1) + ln(x² + 1)",
    )
}


def get_benchmark(name: str) -> Benchmark:
    """Retrieve a benchmark by name (case-insensitive)."""
    key = name.lower().strip()
    if key not in BENCHMARKS:
        available = ", ".join(sorted(BENCHMARKS.keys()))
        raise KeyError(f"Unknown benchmark '{name}'. Available: {available}")
    return BENCHMARKS[key]


def list_benchmarks() -> list[str]:
    """Return sorted list of available benchmark names."""
    return sorted(BENCHMARKS.keys())
