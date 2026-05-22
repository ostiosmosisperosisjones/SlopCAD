"""
cad/cut_all.py

Helpers for "no-target cut" — the operation that cuts every workspace body
the tool intersects.  Pure helpers (bbox prefilter, fan-out) go here so
viewport code stays thin and the same fan-out logic serves both extrude-
and revolve-based cuts.
"""

from __future__ import annotations
from typing import Callable, Any


def bboxes_overlap(a, b) -> bool:
    """
    True when two axis-aligned bounding boxes (build123d BoundingBox or
    anything with .min.{X,Y,Z} and .max.{X,Y,Z}) overlap on every axis.

    Used as a cheap prefilter to skip bodies the cut tool can't intersect
    before paying for a boolean operation.
    """
    return (a.min.X <= b.max.X and a.max.X >= b.min.X and
            a.min.Y <= b.max.Y and a.max.Y >= b.min.Y and
            a.min.Z <= b.max.Z and a.max.Z >= b.min.Z)


def fan_out_cut(viewport, tool_solid, build_op: Callable[[str], Any],
                extra: dict | None, op_label: str = "Cut") -> int:
    """
    Fan out a "no-target cut" across every body whose bbox overlaps the tool.

    viewport   : has .workspace and .history
    tool_solid : the cutting solid (build123d Compound) — already built
    build_op   : callable (body_id) → Op that cuts that body
    extra      : extra params forwarded to op.commit()
    op_label   : tag used in console output ("Cut" / "Revolve cut" / …)

    Returns the number of bodies that were actually cut (i.e. their
    volume changed).  Entries pushed for bodies that didn't change
    are rolled back so the history panel stays clean.
    """
    tool_bbox = tool_solid.bounding_box()
    candidates: list[str] = []
    for bid in viewport.workspace.bodies.keys():
        bshape = viewport.workspace.current_shape(bid)
        if bshape is None:
            continue
        try:
            if bboxes_overlap(bshape.bounding_box(), tool_bbox):
                candidates.append(bid)
        except Exception:
            candidates.append(bid)   # bbox failed — let the cut decide

    if not candidates:
        print(f"[{op_label}] No-target: tool doesn't intersect any body.")
        return 0

    n_cut = 0
    for bid in candidates:
        shape_before = viewport.workspace.current_shape(bid)
        vol_before   = (sum(s.volume for s in shape_before.solids())
                        if shape_before is not None else 0.0)

        op = build_op(bid)
        op.commit(viewport, extra)

        # Roll back if the bbox prefilter was a false positive.
        shape_after = viewport.workspace.current_shape(bid)
        vol_after   = (sum(s.volume for s in shape_after.solids())
                       if shape_after is not None else 0.0)
        if abs(vol_after - vol_before) < 1e-3 and viewport.history.entries:
            last_idx = len(viewport.history.entries) - 1
            if viewport.history.entries[last_idx].body_id == bid:
                viewport.history.delete(last_idx)
        else:
            n_cut += 1

    if n_cut == 0:
        print(f"[{op_label}] No-target: tool didn't actually intersect any body.")
    else:
        print(f"[{op_label}] No-target cut applied to {n_cut} body/bodies.")
    return n_cut
