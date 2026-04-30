"""
SymPy-based simplification for DEAP GP expression trees.

Converts GP prefix-notation trees into SymPy expressions, applies
algebraic simplification, and returns clean mathematical formulas.
"""

from __future__ import annotations

from typing import Any

import sympy as sp
from deap import gp


# ──────────────────────────────────────────────────────────────────────────────
# SymPy symbol registry
# ──────────────────────────────────────────────────────────────────────────────

_SYMBOLS = {name: sp.Symbol(name) for name in ("x", "y", "z", "w")}


def _get_symbol(name: str) -> sp.Symbol:
    """Get or create a SymPy symbol by name."""
    if name not in _SYMBOLS:
        _SYMBOLS[name] = sp.Symbol(name)
    return _SYMBOLS[name]


# ──────────────────────────────────────────────────────────────────────────────
# GP primitive → SymPy mapping
# ──────────────────────────────────────────────────────────────────────────────

def _build_sympy_map() -> dict[str, Any]:
    """Map GP primitive names to SymPy constructors."""
    return {
        # Arithmetic
        "add": lambda a, b: a + b,
        "sub": lambda a, b: a - b,
        "mul": lambda a, b: a * b,
        "protectedDiv": lambda a, b: a / b,
        "neg": lambda a: -a,
        # Trigonometric
        "sin": sp.sin,
        "cos": sp.cos,
        # Transcendental
        "protectedLog": sp.log,
        "protectedExp": sp.exp,
        "protectedSqrt": sp.sqrt,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Tree → SymPy conversion
# ──────────────────────────────────────────────────────────────────────────────

def _tree_to_sympy(tree: gp.PrimitiveTree, pset: gp.PrimitiveSet | None = None) -> sp.Expr:
    """
    Recursively convert a DEAP PrimitiveTree into a SymPy expression.

    Parameters
    ----------
    tree : gp.PrimitiveTree
        The GP expression tree.
    pset : gp.PrimitiveSet, optional
        The primitive set (used for context, not strictly required).

    Returns
    -------
    sp.Expr
        Equivalent SymPy expression.
    """
    sympy_map = _build_sympy_map()
    stack: list[sp.Expr] = []

    # Walk the tree in reverse (postfix evaluation of prefix tree)
    for node in reversed(tree):
        if isinstance(node, gp.Primitive):
            func_name = node.name
            arity = node.arity
            args = [stack.pop() for _ in range(arity)]

            if func_name in sympy_map:
                result = sympy_map[func_name](*args)
            else:
                # Fallback: create a generic SymPy function
                f = sp.Function(func_name)
                result = f(*args)
            stack.append(result)

        elif isinstance(node, gp.Terminal):
            if node.name.startswith("ARG") or node.name in _SYMBOLS:
                # Map DEAP's ARG0/ARG1/... to human-readable x/y/z/w
                name = node.name
                if name.startswith("ARG"):
                    idx = int(name[3:])
                    var_names = ("x", "y", "z", "w")
                    name = var_names[idx] if idx < len(var_names) else f"x{idx}"
                stack.append(_get_symbol(name))
            else:
                # It's a constant (ephemeral or literal)
                try:
                    val = float(node.value)
                    # Use rounded Float for readable output
                    stack.append(sp.Float(round(val, 4)))
                except (ValueError, TypeError):
                    stack.append(_get_symbol(str(node.name)))
        else:
            # Shouldn't happen, but handle gracefully
            stack.append(_get_symbol(str(node)))

    if len(stack) != 1:
        raise ValueError(f"Tree conversion error: stack has {len(stack)} elements (expected 1)")

    return stack[0]


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

_STRATEGIES = [
    ("simplify", sp.simplify),
    ("expand", sp.expand),
    ("factor", sp.factor),
    ("cancel", sp.cancel),
    ("trigsimp", sp.trigsimp),
    ("powsimp", sp.powsimp),
    ("radsimp", sp.radsimp),
]


def simplify_individual(
    individual: gp.PrimitiveTree,
    pset: gp.PrimitiveSet | None = None,
) -> dict[str, Any]:
    """
    Convert a GP individual to SymPy and apply multiple simplification strategies.

    Parameters
    ----------
    individual : gp.PrimitiveTree
        The GP expression tree to simplify.
    pset : gp.PrimitiveSet, optional
        The primitive set used to build the tree.

    Returns
    -------
    dict with keys:
        raw_expr, simplified, raw_str, simplified_str, latex, complexity, strategy
    """
    try:
        raw_expr = _tree_to_sympy(individual, pset)
    except Exception as e:
        return {
            "raw_expr": None,
            "simplified": None,
            "raw_str": str(individual),
            "simplified_str": f"[conversion error: {e}]",
            "latex": "",
            "complexity": -1,
        }

    # Try multiple simplification strategies, pick the shortest
    candidates = []

    for name, strategy in _STRATEGIES:
        try:
            result = strategy(raw_expr)
            complexity = sp.count_ops(result)
            candidates.append((result, complexity, name))
        except Exception:
            continue

    if not candidates:
        # All strategies failed, return raw
        simplified = raw_expr
        best_strategy = "none"
    else:
        # Pick the simplest result (fewest operations)
        candidates.sort(key=lambda x: x[1])
        simplified, _, best_strategy = candidates[0]

    # Round small floating-point coefficients for cleaner display
    simplified = _round_floats(simplified)

    return {
        "raw_expr": raw_expr,
        "simplified": simplified,
        "raw_str": str(raw_expr),
        "simplified_str": str(simplified),
        "latex": sp.latex(simplified),
        "complexity": int(sp.count_ops(simplified)),
        "strategy": best_strategy,
    }


def _round_floats(expr: sp.Expr, decimals: int = 4) -> sp.Expr:
    """Round floating-point numbers in a SymPy expression to N decimal places."""
    for atom in expr.atoms(sp.Float):
        rounded = sp.Float(round(float(atom), decimals))
        expr = expr.subs(atom, rounded)
    return expr


def format_expression(individual: gp.PrimitiveTree, pset: gp.PrimitiveSet | None = None) -> str:
    """
    Quick helper: return the simplified expression as a clean string.

    Falls back to the raw DEAP string if simplification fails.
    """
    result = simplify_individual(individual, pset)
    return result["simplified_str"]
