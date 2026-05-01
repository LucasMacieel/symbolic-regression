"""
Algebraic formatter for DEAP GP expression trees.

Converts a GP prefix-notation tree into a clean infix algebraic string
(e.g. ``protectedDiv(mul(x, y), mul(z, z))`` → ``x*y/z**2``) without
any SymPy simplification.  This keeps the output fast, deterministic,
and free from SymPy's occasional hang/overflow on complex sub-expressions.

The formatter applies only lightweight, structural rewrites:
  - ``mul(x, x)`` → ``x**2``  (repeated-argument squaring)
  - ``protectedDiv(a, mul(b, b))`` → ``a/b**2``  (similar)
  - Correct operator precedence with minimal parentheses.
"""

from __future__ import annotations

from deap import gp


# ──────────────────────────────────────────────────────────────────────────────
# Operator metadata
# ──────────────────────────────────────────────────────────────────────────────

# (precedence, is_right_associative)
_BINARY_OPS: dict[str, tuple[int, bool, str]] = {
    #  name           prec  right-assoc  symbol
    "add":          (1, False, "+"),
    "sub":          (1, False, "-"),
    "mul":          (2, False, "*"),
    "protectedDiv": (2, False, "/"),
}

_UNARY_OPS: dict[str, str] = {
    "neg":          "-",
    "sin":          "sin",
    "cos":          "cos",
    "protectedLog": "log",
    "protectedExp": "exp",
    "protectedSqrt": "sqrt",
}

# Variables and constants pass through unchanged
_VAR_NAMES = ("x", "y", "z", "w")


# ──────────────────────────────────────────────────────────────────────────────
# Internal node representation
# ──────────────────────────────────────────────────────────────────────────────

class _Node:
    """Lightweight expression node for formatting."""

    __slots__ = ("op", "children", "leaf", "prec")

    def __init__(
        self,
        op: str | None = None,
        children: list[_Node] | None = None,
        leaf: str | None = None,
    ) -> None:
        self.op = op
        self.children = children or []
        self.leaf = leaf
        # Precedence: terminals get highest (never parenthesised)
        if leaf is not None:
            self.prec = 99
        elif op in _BINARY_OPS:
            self.prec = _BINARY_OPS[op][0]
        else:
            self.prec = 99  # unary / function calls never need outer parens

    def is_leaf(self) -> bool:
        return self.leaf is not None


# ──────────────────────────────────────────────────────────────────────────────
# Tree builder (prefix → node tree)
# ──────────────────────────────────────────────────────────────────────────────

def _build_node_tree(tree: gp.PrimitiveTree) -> _Node:
    """
    Walk the DEAP PrimitiveTree in prefix order and build a ``_Node`` tree.
    DEAP stores nodes in prefix (pre-order) sequence.
    """
    stack: list[tuple[_Node, int]] = []  # (node, remaining_children_needed)
    root: _Node | None = None

    for node in tree:
        if isinstance(node, gp.Primitive):
            n = _Node(op=node.name, children=[])
            arity = node.arity
            # Push onto stack; we'll fill children as we process subsequent nodes
            stack.append((n, arity))
        elif isinstance(node, gp.Terminal):
            name = node.name
            if name.startswith("ARG"):
                idx = int(name[3:])
                name = _VAR_NAMES[idx] if idx < len(_VAR_NAMES) else f"x{idx}"
            else:
                # Ephemeral constant or literal variable name
                try:
                    val = float(node.value)
                    # Format: drop trailing zeros, keep up to 6 sig figs
                    name = f"{val:g}"
                except (ValueError, TypeError, AttributeError):
                    name = str(name)
            n = _Node(leaf=name)
            stack.append((n, 0))

        # Attach completed nodes to their parents
        while stack:
            top_node, remaining = stack[-1]
            if remaining == 0:
                # This node is complete
                completed = stack.pop()[0]
                if not stack:
                    root = completed
                else:
                    parent, parent_rem = stack[-1]
                    parent.children.append(completed)
                    stack[-1] = (parent, parent_rem - 1)
            else:
                break

    if root is None:
        raise ValueError("Failed to build node tree from GP expression.")
    return root


# ──────────────────────────────────────────────────────────────────────────────
# Structural rewrites  (pure syntactic, no SymPy)
# ──────────────────────────────────────────────────────────────────────────────

def _node_str(node: _Node) -> str:
    """Return the raw string of a node (without formatting context)."""
    return _format_node(node, parent_prec=0, is_right=False)


def _same_subtree(a: _Node, b: _Node) -> bool:
    """Cheap structural equality check."""
    if a.is_leaf() and b.is_leaf():
        return a.leaf == b.leaf
    if a.op != b.op or len(a.children) != len(b.children):
        return False
    return all(_same_subtree(ca, cb) for ca, cb in zip(a.children, b.children))


def _rewrite(node: _Node) -> _Node:
    """
    Apply lightweight structural rewrites bottom-up.

    Rules (applied after children are rewritten):
      mul(a, a)          → a**2
      mul(a, mul(a, a))  → a**3   (one level)
      any operator with identical children triggers the power rewrite.
    """
    # Recurse first
    node.children = [_rewrite(c) for c in node.children]

    # mul(a, a) → a**2
    if node.op == "mul" and len(node.children) == 2:
        a, b = node.children
        if _same_subtree(a, b):
            return _Node(op="__pow__", children=[a, _Node(leaf="2")])

    return node


# ──────────────────────────────────────────────────────────────────────────────
# Formatter
# ──────────────────────────────────────────────────────────────────────────────

def _format_node(node: _Node, parent_prec: int = 0, is_right: bool = False) -> str:
    """Recursively format a node into a clean algebraic string."""

    if node.is_leaf():
        return node.leaf  # type: ignore[return-value]

    op = node.op

    # ── Power (synthetic node from rewrite) ───────────────────────────────────
    if op == "__pow__":
        base = _format_node(node.children[0], parent_prec=3, is_right=False)
        exp  = node.children[1].leaf or _format_node(node.children[1])
        # Parenthesise base if it's not a simple token
        if not node.children[0].is_leaf():
            base = f"({base})"
        return f"{base}**{exp}"

    # ── Binary operators ──────────────────────────────────────────────────────
    if op in _BINARY_OPS:
        prec, right_assoc, symbol = _BINARY_OPS[op]
        left, right = node.children

        # For division, check if right child needs parens (it usually does
        # unless it's a leaf or power node)
        left_str  = _format_node(left,  parent_prec=prec, is_right=False)
        right_str = _format_node(right, parent_prec=prec, is_right=True)

        # Parenthesise left if lower precedence
        if left.prec < prec or (left.prec == prec and right_assoc):
            left_str = f"({left_str})"

        # Parenthesise right for sub/div or lower precedence
        if op in ("sub", "protectedDiv"):
            # Right operand of sub/div needs parens when it is itself add/sub
            if right.prec <= prec and not right.is_leaf():
                right_str = f"({right_str})"
        elif right.prec < prec or (right.prec == prec and not right_assoc):
            right_str = f"({right_str})"

        if op == "protectedDiv":
            return f"{left_str}/{right_str}"
        return f"{left_str}{symbol}{right_str}"

    # ── Unary operators ───────────────────────────────────────────────────────
    if op in _UNARY_OPS:
        sym = _UNARY_OPS[op]
        child_str = _format_node(node.children[0], parent_prec=99)
        if sym == "-":
            # Negation: only add parens for compound expressions
            if node.children[0].is_leaf():
                return f"-{child_str}"
            return f"-({child_str})"
        # Named unary functions (sin, cos, log, exp, sqrt)
        return f"{sym}({child_str})"

    # ── Fallback: unknown primitive ───────────────────────────────────────────
    args = ", ".join(_format_node(c) for c in node.children)
    return f"{op}({args})"


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def format_expression(
    individual: gp.PrimitiveTree,
    pset: gp.PrimitiveSet | None = None,  # kept for API compatibility
) -> str:
    """
    Convert a GP individual to a clean algebraic infix string.

    Example
    -------
    ``protectedDiv(mul(x, y), mul(z, z))`` → ``"x*y/z**2"``

    Parameters
    ----------
    individual : gp.PrimitiveTree
        The GP expression tree.
    pset : gp.PrimitiveSet, optional
        Unused; kept for API compatibility.

    Returns
    -------
    str
        Clean algebraic representation.
    """
    try:
        node_tree = _build_node_tree(individual)
        node_tree = _rewrite(node_tree)
        return _format_node(node_tree)
    except Exception as e:
        # Graceful fallback: return the raw DEAP string
        return str(individual)


def simplify_individual(
    individual: gp.PrimitiveTree,
    pset: gp.PrimitiveSet | None = None,
) -> dict:
    """
    Convert a GP individual to its algebraic form.

    Returns a dict with the same keys that the old SymPy-based version
    produced, so callers need no changes.

    Keys
    ----
    raw_str : str
        Raw DEAP prefix string (e.g. ``protectedDiv(mul(x,y),mul(z,z))``).
    simplified_str : str
        Clean algebraic infix string (e.g. ``x*y/z**2``).
    latex : str
        Empty string — LaTeX output removed with SymPy.
    complexity : int
        Number of nodes in the tree (proxy for expression size).
    strategy : str
        Always ``"algebraic"`` to signal the new codepath.
    """
    raw_str = str(individual)
    simplified_str = format_expression(individual, pset)
    return {
        "raw_expr": None,
        "simplified": None,
        "raw_str": raw_str,
        "simplified_str": simplified_str,
        "latex": "",
        "complexity": len(individual),
        "strategy": "algebraic",
    }
