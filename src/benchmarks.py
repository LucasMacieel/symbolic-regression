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
# Each function maps symbolic variable names (x, y, z, w) to physical
# quantities.  The docstring states the full mapping.
#
# Physical constants (G, k, c, R) are embedded as numeric values so the GP
# algorithm must discover them as part of the expression tree.  Some
# benchmarks use unit-normalised variants to keep outputs in a numerically
# friendly range for GP.
# ──────────────────────────────────────────────────────────────────────────────

# ── Mechanics ─────────────────────────────────────────────────────────────────

def _kinetic_energy(x: float, y: float) -> float:
    """KE = ½·m·v²  |  x → mass (kg), y → velocity (m/s)"""
    return 0.5 * x * y**2


def _gravitational_pe(x: float, y: float) -> float:
    """PE = m·g·h  |  x → mass (kg), y → height (m), g = 9.81 baked in"""
    return x * 9.81 * y


def _newtons_gravity(x: float, y: float, z: float) -> float:
    """F = m₁·m₂/r²  |  x → m₁, y → m₂, z → r  (unit G = 1)"""
    return x * y / max(z**2, 1e-10)


def _hookes_law(x: float, y: float) -> float:
    """F = −k·Δx  |  x → spring constant k, y → displacement"""
    return -x * y


def _momentum(x: float, y: float) -> float:
    """p = m·v  |  x → mass (kg), y → velocity (m/s)"""
    return x * y


# ── Thermodynamics ────────────────────────────────────────────────────────────

def _ideal_gas_pressure(x: float, y: float, z: float) -> float:
    """P = nRT/V  |  x → n (mol), y → T (K), z → V (m³), R = 8.314"""
    return x * 8.314 * y / max(abs(z), 1e-6)


def _heat_conduction(x: float, y: float) -> float:
    """Q = k·ΔT  |  x → thermal conductivity k, y → temperature diff ΔT"""
    return x * y


# ── Electromagnetism ──────────────────────────────────────────────────────────

def _coulombs_law(x: float, y: float, z: float) -> float:
    """F = q₁·q₂/r²  |  x → q₁, y → q₂, z → r  (unit k = 1)"""
    return x * y / max(z**2, 1e-10)


def _ohms_law(x: float, y: float) -> float:
    """V = I·R  |  x → current I (A), y → resistance R (Ω)"""
    return x * y


def _electric_power(x: float, y: float) -> float:
    """P = I²·R  |  x → current I (A), y → resistance R (Ω)"""
    return x**2 * y


# ── Waves & Optics ────────────────────────────────────────────────────────────

def _wave_speed(x: float, y: float) -> float:
    """v = f·λ  |  x → frequency f (Hz), y → wavelength λ (m)"""
    return x * y


def _pendulum_period(x: float, y: float) -> float:
    """T = 2π·√(L/g)  |  x → length L (m), y → gravity g (m/s²)"""
    return 2 * math.pi * math.sqrt(abs(x) / max(abs(y), 1e-6))


# ── Relativity ────────────────────────────────────────────────────────────────

def _rest_energy(x: float) -> float:
    """E = m·c²  |  x → mass m  (natural units: c = 1, so E = m)"""
    return x


def _lorentz_factor(x: float) -> float:
    """γ = 1/√(1−β²)  |  x → β = v/c ∈ [0, 0.99]"""
    return 1.0 / math.sqrt(1.0 - min(x**2, 1.0 - 1e-9))


# ──────────────────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────────────────

BENCHMARKS: dict[str, Benchmark] = {
    # ── Mechanics ─────────────────────────────────────────────────────────────
    "kinetic-energy": Benchmark(
        name="kinetic-energy",
        func=_kinetic_energy,
        domain=(0.1, 10.0),
        n_points=100,
        n_vars=2,
        description="½·m·v²  (x=mass, y=velocity)",
        var_names=["mass (kg)", "velocity (m/s)"],
    ),
    "gravitational-pe": Benchmark(
        name="gravitational-pe",
        func=_gravitational_pe,
        domain=(0.1, 10.0),
        n_points=100,
        n_vars=2,
        description="m·g·h  (x=mass, y=height, g=9.81 m/s²)",
        var_names=["mass (kg)", "height (m)"],
    ),
    "newtons-gravity": Benchmark(
        name="newtons-gravity",
        func=_newtons_gravity,
        domain=(0.1, 5.0),
        n_points=200,
        n_vars=3,
        description="m₁·m₂/r²  (x=m₁, y=m₂, z=r, unit G)",
        var_names=["m1 (kg)", "m2 (kg)", "r (m)"],
    ),
    "hookes-law": Benchmark(
        name="hookes-law",
        func=_hookes_law,
        domain=(0.1, 10.0),
        n_points=100,
        n_vars=2,
        description="-k·x  (x=spring constant, y=displacement)",
        var_names=["k (N/m)", "displacement (m)"],
    ),
    "momentum": Benchmark(
        name="momentum",
        func=_momentum,
        domain=(0.1, 10.0),
        n_points=100,
        n_vars=2,
        description="m·v  (x=mass, y=velocity)",
        var_names=["mass (kg)", "velocity (m/s)"],
    ),

    # ── Thermodynamics ────────────────────────────────────────────────────────
    "ideal-gas": Benchmark(
        name="ideal-gas",
        func=_ideal_gas_pressure,
        domain=(0.1, 5.0),
        n_points=300,
        n_vars=3,
        description="nRT/V  (x=n mol, y=T K, z=V m³, R=8.314)",
        var_names=["n (mol)", "T (K)", "V (m³)"],
    ),
    "heat-conduction": Benchmark(
        name="heat-conduction",
        func=_heat_conduction,
        domain=(0.1, 10.0),
        n_points=100,
        n_vars=2,
        description="k·ΔT  (x=conductivity, y=temp diff)",
        var_names=["k (W/m·K)", "ΔT (K)"],
    ),

    # ── Electromagnetism ──────────────────────────────────────────────────────
    "coulombs-law": Benchmark(
        name="coulombs-law",
        func=_coulombs_law,
        domain=(0.1, 5.0),
        n_points=200,
        n_vars=3,
        description="q₁·q₂/r²  (x=q₁, y=q₂, z=r, unit k)",
        var_names=["q1 (C)", "q2 (C)", "r (m)"],
    ),
    "ohms-law": Benchmark(
        name="ohms-law",
        func=_ohms_law,
        domain=(0.1, 10.0),
        n_points=100,
        n_vars=2,
        description="I·R  (x=current, y=resistance)",
        var_names=["I (A)", "R (Ω)"],
    ),
    "electric-power": Benchmark(
        name="electric-power",
        func=_electric_power,
        domain=(0.1, 10.0),
        n_points=100,
        n_vars=2,
        description="I²·R  (x=current, y=resistance)",
        var_names=["I (A)", "R (Ω)"],
    ),

    # ── Waves & Optics ────────────────────────────────────────────────────────
    "wave-speed": Benchmark(
        name="wave-speed",
        func=_wave_speed,
        domain=(1.0, 100.0),
        n_points=100,
        n_vars=2,
        description="f·λ  (x=frequency Hz, y=wavelength m)",
        var_names=["f (Hz)", "λ (m)"],
    ),
    "pendulum-period": Benchmark(
        name="pendulum-period",
        func=_pendulum_period,
        domain=(0.1, 5.0),
        n_points=100,
        n_vars=2,
        description="2π·√(L/g)  (x=length m, y=gravity m/s²)",
        var_names=["L (m)", "g (m/s²)"],
    ),

    # ── Relativity ────────────────────────────────────────────────────────────
    "rest-energy": Benchmark(
        name="rest-energy",
        func=_rest_energy,
        domain=(0.001, 1.0),
        n_points=50,
        n_vars=1,
        description="E = m·c²  (natural units, c=1; x=mass)",
        var_names=["mass (kg)"],
    ),
    "lorentz-factor": Benchmark(
        name="lorentz-factor",
        func=_lorentz_factor,
        domain=(0.0, 0.99),
        n_points=50,
        n_vars=1,
        description="γ = 1/√(1−β²)  (x=β=v/c)",
        var_names=["β = v/c"],
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
