"""
Benchmark target functions for symbolic regression.

All benchmarks target the rediscovery of fundamental physical laws from
noiseless simulation data.  Each benchmark uses random uniform sampling over
its domain so that the approach scales to three or more input variables
without a combinatorial explosion in sample count.

Variable naming convention follows primitives._VAR_NAMES: x, y, z, w.
The docstring of every physics helper maps those names to their physical
meaning (e.g. x → mass, y → velocity for kinetic energy).

Primitive set used
------------------
Only the primitives required by the physics formulas are included:
  add, sub, mul, protectedDiv, neg, protectedSqrt
  (sin, cos, log, exp are excluded as no physics benchmark needs them)
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
    func: Callable  # ground-truth f(*args) accepting n_vars positional args
    domain: tuple[float, float]
    n_points: int
    n_vars: int
    description: str = ""
    var_names: list[str] = field(default_factory=list)  # human-readable variable names
    X: np.ndarray = field(init=False, repr=False)
    y: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Generate training data from the ground-truth function.

        * 1-var benchmarks: uniform linspace over the domain.
        * 2-var benchmarks: regular grid.
        * 3+-var benchmarks: random uniform samples over the domain.
          Random sampling avoids an exponential blow-up while still
          providing good coverage for GP fitness evaluation.
        """
        rng = np.random.default_rng(seed=42)  # reproducible

        if self.n_vars == 1:
            self.X = np.linspace(self.domain[0], self.domain[1], self.n_points).reshape(-1, 1)
            self.y = np.array([self.func(x) for x in self.X[:, 0]])

        elif self.n_vars == 2:
            n_side = int(math.isqrt(self.n_points))
            xs = np.linspace(self.domain[0], self.domain[1], n_side)
            ys = np.linspace(self.domain[0], self.domain[1], n_side)
            grid_x, grid_y = np.meshgrid(xs, ys)
            self.X = np.column_stack([grid_x.ravel(), grid_y.ravel()])
            self.y = np.array([self.func(row[0], row[1]) for row in self.X])

        else:
            # Random uniform sampling for 3+ variables
            self.X = rng.uniform(self.domain[0], self.domain[1], size=(self.n_points, self.n_vars))
            self.y = np.array([self.func(*row) for row in self.X])

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def input_for_gp(self) -> np.ndarray:
        """Return X shaped (n_points, n_vars) regardless of n_vars."""
        if self.n_vars == 1:
            return self.X  # already (n, 1) after __post_init__ reshape
        return self.X

    def summary(self) -> str:
        """One-line summary with key statistics."""
        return (
            f"{self.name}: {self.description} | "
            f"vars={self.n_vars}, pts={self.n_points}, "
            f"y∈[{self.y.min():.3g}, {self.y.max():.3g}]"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Physics benchmark functions
# ──────────────────────────────────────────────────────────────────────────────
# Each function maps symbolic variable names (x, y, z, u, v, w, q) to physical
# quantities.  The docstring states the full mapping.
#
# Physical constants (G, k, c, R) are embedded as numeric values so the GP
# algorithm must discover them as part of the expression tree.  Some
# benchmarks use unit-normalised variants to keep outputs in a numerically
# friendly range for GP.
# ──────────────────────────────────────────────────────────────────────────────

def _newtons_second_law(x: float, y: float) -> float:
    """F = m·a  |  x → mass m (kg), y → acceleration a (m/s²)"""
    return x * y

def _kinetic_energy(x: float, y: float) -> float:
    """E = ½·m·v²  |  x → mass m (kg), y → velocity v (m/s)"""
    return 0.5 * x * y**2

def _newtons_gravity(x: float, y: float, z: float) -> float:
    """F = G·m₁·m₂/r²  |  x → m₁ (kg), y → m₂ (kg), z → r (m), unit G=1"""
    return x * y / max(z**2, 1e-10)

def _keplers_third_law(x: float, y: float) -> float:
    """T = 2π·√(a³/GM)  |  x → semi-major axis a (m), y → mass M (kg), unit G=1"""
    return 2 * math.pi * math.sqrt(max(x**3, 0.0) / max(abs(y), 1e-10))

def _stefan_boltzmann(x: float, y: float) -> float:
    """P = σ·A·T⁴  |  x → area A (m²), y → temperature T (K), σ=5.67"""
    return 5.67 * x * y**4

def _projectile_range(x: float, y: float) -> float:
    """R = v₀²·sin(2θ)/g  |  x → v₀ (m/s), y → θ (rad), g=9.81"""
    return x**2 * math.sin(2 * y) / 9.81

def _bernoullis_equation(x: float, y: float, z: float, u: float, v: float, w: float) -> float:
    """P₂ = P₁ + ½ρ(v₁² - v₂²) + ρg(h₁ - h₂)
    x → P₁ (Pa), y → ρ (kg/m³), z → v₁ (m/s), u → v₂ (m/s), v → h₁ (m), w → h₂ (m), g=9.81
    """
    return x + 0.5 * y * (z**2 - u**2) + y * 9.81 * (v - w)


# ──────────────────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────────────────

BENCHMARKS: dict[str, Benchmark] = {
    "newtons-second-law": Benchmark(
        name="newtons-second-law",
        func=_newtons_second_law,
        domain=(0.1, 10.0),
        n_points=100,
        n_vars=2,
        description="m·a  (x=m, y=a)",
        var_names=["m (kg)", "a (m/s²)"],
    ),
    "kinetic-energy": Benchmark(
        name="kinetic-energy",
        func=_kinetic_energy,
        domain=(0.1, 10.0),
        n_points=100,
        n_vars=2,
        description="½·m·v²  (x=m, y=v)",
        var_names=["m (kg)", "v (m/s)"],
    ),
    "newtons-gravity": Benchmark(
        name="newtons-gravity",
        func=_newtons_gravity,
        domain=(0.1, 5.0),
        n_points=200,
        n_vars=3,
        description="G·m₁·m₂/r²  (x=m₁, y=m₂, z=r, unit G=1)",
        var_names=["m1 (kg)", "m2 (kg)", "r (m)"],
    ),
    "keplers-third-law": Benchmark(
        name="keplers-third-law",
        func=_keplers_third_law,
        domain=(0.1, 10.0),
        n_points=100,
        n_vars=2,
        description="2π·√(a³/GM)  (x=a, y=M, unit G=1)",
        var_names=["a (m)", "M (kg)"],
    ),
    "stefan-boltzmann": Benchmark(
        name="stefan-boltzmann",
        func=_stefan_boltzmann,
        domain=(0.1, 5.0),
        n_points=100,
        n_vars=2,
        description="σ·A·T⁴  (x=A, y=T, σ=5.67)",
        var_names=["A (m²)", "T (K)"],
    ),
    "projectile-range": Benchmark(
        name="projectile-range",
        func=_projectile_range,
        domain=(0.1, 5.0),
        n_points=100,
        n_vars=2,
        description="v₀²·sin(2θ)/g  (x=v₀, y=θ, g=9.81)",
        var_names=["v0 (m/s)", "θ (rad)"],
    ),
    "bernoullis-equation": Benchmark(
        name="bernoullis-equation",
        func=_bernoullis_equation,
        domain=(0.1, 5.0),
        n_points=500,
        n_vars=6,
        description="P₁ + ½ρ(v₁² - v₂²) + ρg(h₁ - h₂)  (x=P₁, y=ρ, z=v₁, u=v₂, v=h₁, w=h₂, g=9.81)",
        var_names=["P1 (Pa)", "ρ (kg/m³)", "v1 (m/s)", "v2 (m/s)", "h1 (m)", "h2 (m)"],
    ),
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
