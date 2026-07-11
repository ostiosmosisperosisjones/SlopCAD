"""
cad/trace/walk.py

Stage 2c of the V3 (inside-out / stroke) tracer: turn a pruned skeleton into
ordered pixel polylines that fit.py can fit, exactly like contours.py's boundary
loops — but for centerlines.

The skeleton is a graph: pixels of degree 1 are endpoints, degree 2 are simple
path interior, degree >= 3 are junctions.  A naive walk that STOPS at every
junction shatters a shape (e.g. a wall crossed by a tick mark breaks into three
stubs).  So walk.py STITCHES straight-through junctions:

  at a junction, pair the two most tangent-collinear incident edges (they arrive
  from ~opposite directions) into one continuous polyline; the odd branch (the
  tick) departs on its own.

This is what keeps the outer wall a single continuous polyline through the
interior T-marks — the #1 risk from the V3 plan, resolved here.

Output
------
walk_skeleton(skel) -> list[Polyline]
    Polyline.points : (N,2) float array of (x, y) = (col, row) pixel coords,
                      ordered along the branch.  x=col, y=row matches fit.py.
    Polyline.closed : True if the branch is a closed loop (first ~ last).

Junctions are handled so straight-through paths stay whole; genuine branch
points (the odd edge) start a new polyline.  Feed each Polyline.points to
fit.py's fit_loop (closed) or _fit_run (open).
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from cad.trace.skeleton import neighbour_count, _nbrs


# Minimum |dot| of arrival tangents for two edges to be stitched straight
# through a junction.  -1 = perfectly collinear (opposite arrival dirs); we
# require them to arrive at least this "oppositely" to fuse.
_STRAIGHT_DOT = -0.5

# Pixels back from the node used to estimate an edge's arrival tangent.
_TANGENT_SPAN = 4


@dataclass
class Polyline:
    points: np.ndarray        # (N,2) float (x, y)
    closed: bool = False


def _extract_edges(skel: np.ndarray):
    """Split the skeleton into edges: each a degree-2 chain between two nodes
    (endpoint or junction).  Also returns the set of node coords.

    Returns (edges, node_set) where each edge is a list of (r, c) from one node
    to another (inclusive), and node_set is the set of (r, c) nodes.
    """
    deg = neighbour_count(skel)
    nodes = (deg != 2) & skel
    node_set = set(zip(*(a.tolist() for a in np.where(nodes))))
    visited = set()
    edges = []
    # Degree-2 pixels consumed by node-anchored edges; the pure-loop pass below
    # must skip these or it re-walks a chain that already became an edge
    # (which would duplicate a straight stroke into path + phantom loop).
    consumed = np.zeros_like(skel)

    def ekey(a, b):
        return (a, b) if a <= b else (b, a)

    for start in node_set:
        for first in _nbrs(skel, *start):
            if ekey(start, first) in visited:
                continue
            path = [start, first]
            visited.add(ekey(start, first))
            prev, cur = start, first
            while cur not in node_set:
                consumed[cur] = True
                nb = [p for p in _nbrs(skel, *cur) if p != prev]
                if not nb:
                    break
                nxt = nb[0]
                visited.add(ekey(cur, nxt))
                path.append(nxt)
                prev, cur = cur, nxt
            edges.append(path)

    # Pure loops (no node at all: every pixel degree 2).  Skip pixels already
    # consumed by a node-anchored edge above.
    seen = consumed.copy()
    for r, c in zip(*np.where(skel & (deg == 2))):
        if seen[r, c]:
            continue
        path = [(r, c)]
        seen[r, c] = True
        prev, cur = None, (r, c)
        while True:
            nb = [p for p in _nbrs(skel, *cur) if p != prev and not seen[p]]
            if not nb:
                break
            nxt = nb[0]
            seen[nxt] = True
            path.append(nxt)
            prev, cur = cur, nxt
        if len(path) >= 4:
            path.append(path[0])          # close the loop
            edges.append(path)
    return edges, node_set


def _arrival_dir(path, node) -> np.ndarray:
    """Unit tangent of an edge as it ARRIVES at `node` (an endpoint of path).

    Oriented so the vector points *into* the node from a few px back, giving a
    stable direction even on a staircased skeleton."""
    pts = np.array(path, dtype=np.float64)
    if node == tuple(path[0]):
        pts = pts[::-1]                   # orient so the node is last
    k = min(_TANGENT_SPAN, len(pts) - 1)
    d = pts[-1] - pts[-(k + 1)]
    n = np.hypot(d[0], d[1])
    return d / n if n > 1e-9 else d


def _stitch_junctions(edges, node_set):
    """Decide, per junction, which incident edges continue straight through each
    other.  Returns a dict {(edge_idx, node): partner_edge_idx}."""
    incident = {}                         # node -> [edge indices]
    for i, e in enumerate(edges):
        for end in (tuple(e[0]), tuple(e[-1])):
            if end in node_set:
                incident.setdefault(end, []).append(i)

    through = {}
    for node, eids in incident.items():
        if len(eids) < 3:
            continue                      # deg<=2: nothing to disambiguate
        dirs = {i: _arrival_dir(edges[i], node) for i in eids}
        cand = list(eids)
        while len(cand) >= 2:
            best = None                   # (dot, ia, ib) most collinear
            for a in range(len(cand)):
                for b in range(a + 1, len(cand)):
                    ia, ib = cand[a], cand[b]
                    dot = float(dirs[ia] @ dirs[ib])
                    if best is None or dot < best[0]:
                        best = (dot, ia, ib)
            dot, ia, ib = best
            if dot < _STRAIGHT_DOT:        # collinear enough: stitch through
                through[(ia, node)] = ib
                through[(ib, node)] = ia
            cand.remove(ia)
            cand.remove(ib)
        # A leftover odd edge simply has no through-partner (it departs alone).
    return through


def walk_skeleton(skel: np.ndarray) -> list:
    """Skeleton (bool HxW) → ordered Polylines, stitching straight-through
    junctions so crossing strokes don't shatter continuous paths."""
    edges, node_set = _extract_edges(skel)
    if not edges:
        return []
    through = _stitch_junctions(edges, node_set)

    def other_end(e, node):
        return tuple(e[-1]) if tuple(e[0]) == node else tuple(e[0])

    def oriented(e, from_node):
        return e if tuple(e[0]) == from_node else e[::-1]

    used = set()
    out = []
    for i0 in range(len(edges)):
        if i0 in used:
            continue
        e0 = edges[i0]
        # Pre-closed pure loops emit directly.
        if tuple(e0[0]) == tuple(e0[-1]) and len(e0) >= 4:
            used.add(i0)
            pts = np.array([(c, r) for (r, c) in e0], dtype=np.float64)
            out.append(Polyline(points=pts, closed=True))
            continue

        start_node = tuple(e0[0])
        chain = list(oriented(e0, start_node))
        used.add(i0)

        # Extend forward through stitched junctions.
        cur_edge, cur_node = i0, other_end(e0, start_node)
        while True:
            partner = through.get((cur_edge, cur_node))
            if partner is None or partner in used:
                break
            pe = oriented(edges[partner], cur_node)
            chain.extend(pe[1:])          # skip the shared node pixel
            used.add(partner)
            cur_edge, cur_node = partner, other_end(edges[partner], cur_node)

        # Extend backward from the start node too.
        back_edge, back_node = i0, start_node
        while True:
            partner = through.get((back_edge, back_node))
            if partner is None or partner in used:
                break
            pe = oriented(edges[partner], back_node)   # starts at back_node
            chain = list(pe[::-1][:-1]) + chain        # prepend, skip shared
            used.add(partner)
            back_edge, back_node = partner, other_end(edges[partner], back_node)

        if len(chain) < 2:
            continue
        pts = np.array([(c, r) for (r, c) in chain], dtype=np.float64)
        closed = bool(np.linalg.norm(pts[0] - pts[-1]) <= 2.5 and len(pts) >= 8)
        out.append(Polyline(points=pts, closed=closed))
    return out
