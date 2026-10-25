"""
cad/expression.py

Safe math expression evaluator for CAD input fields.

parse_expr(text, unit, names=None) -> float (mm, or degrees for unit="deg")

Supports:
  - Arithmetic:  + - * / // % ** ( )
  - Constants:   pi  tau  e
  - Unit suffix: 10mm  2.5in  1ft  3cm  0.5m  90deg
                 (suffix overrides the *unit* argument for that term)
  - Parameters:  any name in *names* (see cad/params.py)
  - Active unit: bare numbers are interpreted in *unit* (default mm)

Examples:
  parse_expr("10 + 5/2", "mm")   -> 12.5
  parse_expr("1in + 5mm", "mm")  -> 30.4
  parse_expr("2 * pi * 3", "mm") -> 18.849...
  parse_expr("  50  ", "cm")     -> 500.0  (50 cm in mm)
  parse_expr("w*2", "in", {"w": Quantity(20.0, 1)}) -> 40.0  (w is already mm)

Units are tracked as a power of length per sub-expression, so a parameter
(already in mm) is never re-scaled by the active unit, while bare numbers
added to a length are read in the active unit: "w + 1" in inches is
w + 25.4mm.  The result is only scaled by the active unit when the whole
expression is unitless ("2*pi*3").
"""

from __future__ import annotations
import ast
import math
import re
from typing import NamedTuple
from cad.units import UNITS


class Quantity(NamedTuple):
    """A value with a length dimension.

    dim: power of length (1 = mm, 0 = plain number / angle), or None for a
    *bare* number that hasn't been given a unit yet and adopts whatever it is
    combined with.
    """
    value: float
    dim:   int | None


# ---------------------------------------------------------------------------
# Allowed names in expressions
# ---------------------------------------------------------------------------

_SAFE_NAMES: dict[str, float] = {
    "pi":  math.pi,
    "tau": math.tau,
    "e":   math.e,
}

# Unit suffixes are rewritten to these private names: "10mm" -> "10*_u_mm".
_UNIT_PREFIX = "_u_"


def _unit_quantity(unit: str) -> Quantity:
    return Quantity(UNITS[unit], 0 if unit == "deg" else 1)


# ---------------------------------------------------------------------------
# Pre-processing — replace unit suffixes before parsing
# ---------------------------------------------------------------------------

# Matches a number (int or float) immediately followed by a unit suffix.
# e.g. "10mm", "2.5in", "1ft", "3cm", "0.5m".  The lookbehind keeps digits
# inside identifiers ("h2in") from being read as a unit literal.
_UNIT_RE = re.compile(
    r'(?<![\w.])(\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)\s*(' +
    '|'.join(sorted(UNITS.keys(), key=len, reverse=True)) +  # longest first
    r')\b'
)


def _preprocess(text: str) -> str:
    """
    Replace unit suffixes with multiplication by a private unit name.

    e.g.  "10mm + 1in"  ->  "10*_u_mm + 1*_u_in"
          "2/3in"        ->  "2/3*_u_in"   (correct: (2/3)*25.4 = 16.93mm)
    """
    return _UNIT_RE.sub(lambda m: f"{m.group(1)}*{_UNIT_PREFIX}{m.group(2)}",
                        text)


# ---------------------------------------------------------------------------
# AST evaluator — whitelist only safe node types
# ---------------------------------------------------------------------------

_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Name,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.UAdd, ast.USub,
)


def _dim_name(d: int | None) -> str:
    return {None: "number", 0: "number", 1: "length"}.get(d, f"length^{d}")


class _Evaluator:
    def __init__(self, unit: str, names: dict[str, Quantity]):
        self._unit_factor = UNITS.get(unit, 1.0)
        self._names = names

    def _promote(self, q: Quantity, dim: int | None) -> Quantity:
        """Give a bare number the dimension it's being combined with."""
        if q.dim is not None or dim is None:
            return q
        if dim == 1:
            return Quantity(q.value * self._unit_factor, 1)
        return Quantity(q.value, dim)

    def _additive(self, a: Quantity, b: Quantity, what: str):
        a = self._promote(a, b.dim)
        b = self._promote(b, a.dim)
        if a.dim != b.dim:
            raise ValueError(f"Can't {what} a {_dim_name(a.dim)} and a "
                             f"{_dim_name(b.dim)}")
        return a, b

    def eval(self, node: ast.AST) -> Quantity:
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(f"Unsupported expression element: {type(node).__name__}")

        if isinstance(node, ast.Expression):
            return self.eval(node.body)

        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
                raise ValueError(f"Non-numeric constant: {node.value!r}")
            return Quantity(float(node.value), None)

        if isinstance(node, ast.Name):
            if node.id.startswith(_UNIT_PREFIX) and node.id[len(_UNIT_PREFIX):] in UNITS:
                return _unit_quantity(node.id[len(_UNIT_PREFIX):])
            if node.id in self._names:
                return self._names[node.id]
            if node.id in _SAFE_NAMES:
                return Quantity(_SAFE_NAMES[node.id], None)
            raise ValueError(f"Unknown name: {node.id!r}")

        if isinstance(node, ast.UnaryOp):
            q = self.eval(node.operand)
            if isinstance(node.op, ast.UAdd):
                return q
            if isinstance(node.op, ast.USub):
                return Quantity(-q.value, q.dim)
            raise ValueError("Unsupported unary operator")

        if isinstance(node, ast.BinOp):
            a  = self.eval(node.left)
            b  = self.eval(node.right)
            op = node.op
            if isinstance(op, ast.Add):
                a, b = self._additive(a, b, "add")
                return Quantity(a.value + b.value, a.dim)
            if isinstance(op, ast.Sub):
                a, b = self._additive(a, b, "subtract")
                return Quantity(a.value - b.value, a.dim)
            if isinstance(op, ast.Mod):
                a, b = self._additive(a, b, "take the modulo of")
                if b.value == 0:
                    raise ValueError("Division by zero")
                return Quantity(a.value % b.value, a.dim)
            if isinstance(op, ast.Mult):
                dim = None if a.dim is None and b.dim is None \
                    else (a.dim or 0) + (b.dim or 0)
                return Quantity(a.value * b.value, dim)
            if isinstance(op, (ast.Div, ast.FloorDiv)):
                if b.value == 0:
                    raise ValueError("Division by zero")
                dim = None if a.dim is None and b.dim is None \
                    else (a.dim or 0) - (b.dim or 0)
                v = a.value / b.value if isinstance(op, ast.Div) \
                    else float(a.value // b.value)
                return Quantity(v, dim)
            if isinstance(op, ast.Pow):
                if b.dim not in (None, 0):
                    raise ValueError("Exponent must be a plain number")
                dim = a.dim
                if a.dim not in (None, 0):
                    if b.value != int(b.value):
                        raise ValueError("A length can only be raised to a "
                                         "whole-number power")
                    dim = a.dim * int(b.value)
                return Quantity(a.value ** b.value, dim)
            raise ValueError("Unsupported binary operator")

        raise ValueError(f"Unhandled node type: {type(node).__name__}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _parse(text: str) -> ast.Expression:
    text = text.strip()
    if not text:
        raise ValueError("Empty expression")
    try:
        return ast.parse(_preprocess(text), mode='eval')
    except SyntaxError as ex:
        raise ValueError(f"Syntax error: {ex.msg}") from ex


def eval_quantity(text: str, unit: str = "mm",
                  names: dict[str, Quantity] | None = None) -> Quantity:
    """Evaluate *text* keeping its dimension (see Quantity).  Bare numbers
    combined with lengths are read in *unit*; a fully bare result is
    returned with dim=None, unscaled."""
    tree = _parse(text)
    try:
        return _Evaluator(unit, names or {}).eval(tree)
    except (ZeroDivisionError, OverflowError) as ex:
        raise ValueError(str(ex)) from ex


def finalize(q: Quantity, unit: str) -> float:
    """Turn a Quantity into a field value: mm for length units, plain
    degrees/number for "deg".  Bare numbers adopt *unit*."""
    if unit == "deg":
        if q.dim not in (None, 0):
            raise ValueError(f"Expected an angle, got a {_dim_name(q.dim)}")
        return float(q.value)
    if q.dim in (None, 0):
        return float(q.value * UNITS.get(unit, 1.0))
    if q.dim != 1:
        raise ValueError(f"Expected a length, got a {_dim_name(q.dim)}")
    return float(q.value)


def parse_expr(text: str, unit: str = "mm",
               names: dict[str, Quantity] | None = None) -> float:
    """
    Parse and evaluate *text* as a math expression.

    Bare numbers are interpreted in *unit* and converted to mm.
    Unit-suffixed numbers (e.g. "1in") are always converted correctly
    regardless of *unit*.  *names* maps parameter names to Quantity values.

    Returns the result in millimetres (degrees for unit="deg"), or raises
    ValueError with a human-readable message on any error.
    """
    return finalize(eval_quantity(text, unit, names), unit)


def expr_refs(text: str) -> set[str]:
    """Parameter names referenced by *text* (excludes constants and units).
    Raises ValueError on a syntax error."""
    return {n.id for n in ast.walk(_parse(text))
            if isinstance(n, ast.Name)
            and n.id not in _SAFE_NAMES
            and not n.id.startswith(_UNIT_PREFIX)}


def rename_ref(text: str, old: str, new: str) -> str:
    """Rewrite references to parameter *old* as *new*, leaving everything else
    (spacing, unit suffixes) untouched.  Token-based, not regex: a parameter
    named "w" doesn't touch "width" or "2w"-style text."""
    import io
    import tokenize
    toks = list(tokenize.generate_tokens(io.StringIO(text).readline))
    out, prev_end = [], (1, 0)
    for tok in toks:
        if tok.type == tokenize.ENDMARKER:
            break
        # tokens are single-line here; re-insert the gap before each one
        out.append(text[prev_end[1]:tok.start[1]])
        s = tok.string
        if tok.type == tokenize.NAME and s == old:
            s = new
        out.append(s)
        prev_end = tok.end
    out.append(text[prev_end[1]:])
    return "".join(out)
