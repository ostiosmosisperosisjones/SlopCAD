"""
cad/sketch_tools/chain.py

Chain selection: from one sketch entity, collect the entities joined to it
end-to-end.

`chain_from(entities, seed)` walks only through plain joints (exactly two
entity ends meet), stopping at T/X junctions and open ends, the way
Fusion/Onshape chain-select does.  With `whole=True` it ignores junctions and
returns the entire connected group.  Lines, arcs and splines take part;
points and includes don't, and the chain never crosses between construction
and normal geometry.
"""
from __future__ import annotations

from collections import defaultdict

_TOL = 1e-3


def _chainable(e) -> bool:
    from cad.sketch import LineEntity, ArcEntity, SplineEntity
    return isinstance(e, (LineEntity, ArcEntity, SplineEntity))


def _end_nodes(entities, idxs, tol):
    """Merge endpoints closer than *tol* into shared nodes (grid hash with a
    3x3 neighbourhood).  Returns ends[i] = (node_a, node_b), node → [entity
    idx per end touching it]."""
    cells: dict = {}
    ends: dict[int, tuple[int, int]] = {}
    at: dict[int, list[int]] = defaultdict(list)

    def node(p):
        cx, cy = round(float(p[0]) / tol), round(float(p[1]) / tol)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                n = cells.get((cx + dx, cy + dy))
                if n is not None:
                    return n
        n = cells[(cx, cy)] = len(cells)
        return n

    for i in idxs:
        e = entities[i]
        a, b = node(e.p0), node(e.p1)
        ends[i] = (a, b)
        at[a].append(i)
        at[b].append(i)
    return ends, at


def chain_from(entities, seed: int, whole: bool = False,
               tol: float = _TOL) -> list[int]:
    """Entity indices chained to *seed* (seed included), sorted.  [seed] if
    the seed can't chain; [] if it isn't a chainable entity."""
    if not (0 <= seed < len(entities)) or not _chainable(entities[seed]):
        return []
    cons = bool(getattr(entities[seed], "construction", False))
    idxs = [i for i, e in enumerate(entities)
            if _chainable(e) and bool(getattr(e, "construction", False)) == cons]
    ends, at = _end_nodes(entities, idxs, tol)

    found = {seed}
    stack = [seed]
    while stack:
        i = stack.pop()
        for n in ends[i]:
            touching = at[n]
            if not whole and len(touching) != 2:
                continue                         # open end or junction
            for j in touching:
                if j not in found:
                    found.add(j)
                    stack.append(j)
    return sorted(found)
