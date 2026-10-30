"""
cad/cut_all.py

Helpers for "no-target cut" — the operation that cuts every workspace body
the tool intersects.  Pure helpers (bbox prefilter, fan-out) go here so
viewport code stays thin and the same fan-out logic serves both extrude-
and revolve-based cuts.
"""

from __future__ import annotations
from typing import Callable, Any


def bboxes_overlap(a, b, tol: float = 0.0) -> bool:
    """
    True when two axis-aligned bounding boxes (build123d BoundingBox or
    anything with .min.{X,Y,Z} and .max.{X,Y,Z}) overlap on every axis.

    Used as a cheap prefilter to skip bodies the cut tool can't intersect
    before paying for a boolean operation.

    tol expands each box before comparing — pass a small positive value when
    abutting must count as overlap: two bodies in exact face-to-face contact
    can have bboxes separated by float noise (~1e-7 mm), which a zero-tol
    check reads as disjoint.
    """
    return (a.min.X - tol <= b.max.X and a.max.X + tol >= b.min.X and
            a.min.Y - tol <= b.max.Y and a.max.Y + tol >= b.min.Y and
            a.min.Z - tol <= b.max.Z and a.max.Z + tol >= b.min.Z)


def fan_out_cut(viewport, tool_solid, build_op: Callable[[str], Any],
                extra: dict | None, op_label: str = "Cut",
                last: str | None = None) -> int:
    """
    Fan out a "no-target cut" across every body whose bbox overlaps the tool.

    viewport   : has .workspace and .history
    tool_solid : the cutting solid (build123d Compound) — already built
    build_op   : callable (body_id) → Op that cuts that body
    extra      : extra params forwarded to op.commit()
    op_label   : tag used in console output ("Cut" / "Revolve cut" / …)
    last       : body to cut after all others (a face-driven tool's source)

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

    if last in candidates:
        candidates.remove(last)
        candidates.append(last)

    if not candidates:
        print(f"[{op_label}] No-target: tool doesn't intersect any body.")
        return 0

    # Every per-body entry carries the same fanout_id, so the history panel,
    # delete, seek and edit treat the whole cut as one action (History.op_group).
    import uuid
    fanout_id = uuid.uuid4().hex
    history = viewport.history

    n_cut = 0
    for bid in candidates:
        shape_before = viewport.workspace.current_shape(bid)
        vol_before   = (sum(s.volume for s in shape_before.solids())
                        if shape_before is not None else 0.0)

        ids_before = {e.entry_id for e in history.entries}
        op = build_op(bid)
        op.commit(viewport, extra)
        entries = history.entries
        pushed = next((i for i, e in enumerate(entries)
                       if e.entry_id not in ids_before and e.body_id == bid
                       and e.operation != "import"), None)

        # Roll back if the bbox prefilter was a false positive.
        shape_after = viewport.workspace.current_shape(bid)
        vol_after   = (sum(s.volume for s in shape_after.solids())
                       if shape_after is not None else 0.0)
        if abs(vol_after - vol_before) < 1e-3:
            if pushed is not None:
                history.delete(pushed)
                if history.cursor >= pushed:     # mid-history (edit) push
                    history.seek(pushed - 1)
        else:
            n_cut += 1
            if pushed is not None:
                entries[pushed].params["fanout_id"] = fanout_id

    if n_cut == 0:
        print(f"[{op_label}] No-target: tool didn't actually intersect any body.")
    else:
        print(f"[{op_label}] No-target cut applied to {n_cut} body/bodies.")
    return n_cut
