"""
cad/params.py

Document-level named parameters ("width = 40mm", "wall = width/20").

Parameters live on History (replay needs them) and are referenced from:
  - op fields:          entry.params["exprs"] = {"distance": {"expr": "w/2", "unit": "mm"}}
  - sketch dimensions:  SketchConstraint.expr / .expr_unit

The baked float stays in params / constraint.value exactly as before, so a
file is still readable by code that knows nothing about parameters; the
binding only says "recompute this float from the expression on replay".

A binding stores the unit that was active when it was typed: bare numbers
in "w + 5" mean 5 of *that* unit forever, not whatever the unit preference
is switched to later.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, asdict
from cad.expression import (Quantity, eval_quantity, finalize, parse_expr,
                            expr_refs, rename_ref, _SAFE_NAMES)
from cad.units import UNITS

KINDS = ("length", "angle", "number")

_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


@dataclass
class Param:
    name:    str
    expr:    str
    kind:    str = "length"     # one of KINDS
    unit:    str = "mm"         # unit bare numbers in expr are read in
    comment: str = ""


class ParamError(ValueError):
    pass


def validate_name(name: str) -> None:
    if not _NAME_RE.match(name or ""):
        raise ParamError(f"'{name}' isn't a valid name — use letters, digits "
                         "and _, starting with a letter")
    if name in _SAFE_NAMES or name in UNITS:
        raise ParamError(f"'{name}' is reserved")


class ParamTable:
    def __init__(self):
        self.params: dict[str, Param] = {}

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self) -> tuple[dict[str, Quantity], dict[str, str]]:
        """Evaluate every parameter in dependency order.

        Returns (values, errors).  A parameter that fails (bad syntax, unknown
        name, cycle, depends on a failed one) is in *errors* and absent from
        *values*; the rest still evaluate.
        """
        values: dict[str, Quantity] = {}
        errors: dict[str, str] = {}

        def visit(name: str, stack: list[str]):
            if name in values or name in errors:
                return
            if name in stack:
                cyc = stack[stack.index(name):] + [name]
                msg = "Circular reference: " + " → ".join(cyc)
                for n in cyc[:-1]:
                    errors.setdefault(n, msg)
                return
            p = self.params[name]
            try:
                refs = expr_refs(p.expr)
            except ValueError as ex:
                errors[name] = str(ex)
                return
            for dep in sorted(refs):
                if dep not in self.params:
                    errors[name] = f"Unknown name: '{dep}'"
                    return
                visit(dep, stack + [name])
                if name in errors:          # we're part of a cycle
                    return
                if dep in errors:
                    errors[name] = f"Depends on '{dep}', which has an error"
                    return
            try:
                values[name] = self._as_quantity(p, values)
            except ValueError as ex:
                errors[name] = str(ex)

        for name in self.params:
            visit(name, [])
        return values, errors

    @staticmethod
    def _as_quantity(p: Param, values: dict[str, Quantity]) -> Quantity:
        q = eval_quantity(p.expr, p.unit, values)
        if p.kind == "length":
            return Quantity(finalize(q, p.unit), 1)
        # angle (degrees) / plain number
        return Quantity(finalize(q, "deg"), 0)

    # ------------------------------------------------------------------
    # Editing
    # ------------------------------------------------------------------

    def set(self, p: Param) -> None:
        """Add or replace a parameter.  Validates the name and that the whole
        table still evaluates without introducing an error on *p*."""
        validate_name(p.name)
        if p.kind not in KINDS:
            raise ParamError(f"Unknown kind '{p.kind}'")
        old = self.params.get(p.name)
        self.params[p.name] = p
        _, errors = self.evaluate()
        if p.name in errors:
            if old is None:
                del self.params[p.name]
            else:
                self.params[p.name] = old
            raise ParamError(errors[p.name])

    def remove(self, name: str) -> None:
        users = [n for n, p in self.params.items()
                 if n != name and name in _safe_refs(p.expr)]
        if users:
            raise ParamError(f"'{name}' is used by {', '.join(users)}")
        self.params.pop(name, None)

    def rename(self, old: str, new: str) -> None:
        """Rename inside the table (other params' expressions included).
        Callers also rewrite bindings — see History.rename_param."""
        validate_name(new)
        if new in self.params:
            raise ParamError(f"'{new}' already exists")
        self.params = {
            (new if n == old else n): Param(
                new if n == old else n, rename_ref(p.expr, old, new),
                p.kind, p.unit, p.comment)
            for n, p in self.params.items()}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_list(self) -> list[dict]:
        return [asdict(p) for p in self.params.values()]

    @classmethod
    def from_list(cls, items: list[dict] | None) -> "ParamTable":
        t = cls()
        for d in items or []:
            p = Param(**{k: d[k] for k in ("name", "expr", "kind", "unit",
                                           "comment") if k in d})
            t.params[p.name] = p    # no validation: keep even if broken
        return t


def _safe_refs(text: str) -> set[str]:
    try:
        return expr_refs(text)
    except ValueError:
        return set()


# ---------------------------------------------------------------------------
# Bindings (op fields)
# ---------------------------------------------------------------------------

def make_binding(expr: str, unit: str) -> dict:
    return {"expr": expr, "unit": unit}


def eval_binding(b: dict, values: dict[str, Quantity]) -> float:
    return parse_expr(b["expr"], b.get("unit", "mm"), values)


def binding_refs(b: dict) -> set[str]:
    return _safe_refs(b.get("expr", ""))
