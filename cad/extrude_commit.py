"""
cad/extrude_commit.py

Which op an extrude-panel commit becomes, and the no-target cut fan-out.
The GUI (viewer/vp_extrude.py) and the scripting API (cad/api.py) both go
through these, so a script builds exactly the entries a user's clicks do.

  extrude_op(profiles, dist, direction, merge_body_id, extra)
      dist < 0 with a target body → CrossBodyCutOp; anything else (extrude,
      merge, new body "__new_body__", self-cut with no target) → ExtrudeOp.
  cut_all_intersecting(viewport, dist, direction, profiles, extra)
      dist < 0 and no target: one CrossBodyCutOp per body the tool hits
      (cad.cut_all.fan_out_cut), the face-driven tool's own body last.

*viewport* is anything with the attributes op.commit() reads (the real
Viewport, or cad.api's headless host).  *extra* is the panel's extra params
(target_vertex, vertex_source, vertex_side, direction_source, start/end
offset, draft_angle, exprs).
"""
from __future__ import annotations

NEW_BODY = "__new_body__"


def _common(direction, extra) -> dict:
    ex = extra or {}
    return dict(
        direction        = (list(map(float, direction))
                            if direction is not None else None),
        target_vertex    = ex.get('target_vertex'),
        vertex_source    = ex.get('vertex_source'),
        vertex_side      = int(ex.get('vertex_side', 0)),
        direction_source = ex.get('direction_source'),
        start_offset     = float(ex.get('start_offset', 0.0)),
        end_offset       = float(ex.get('end_offset', 0.0)),
        draft_angle      = float(ex.get('draft_angle', 0.0)))


def extrude_op(profiles, dist, direction, merge_body_id, extra):
    """The op for a panel commit: a cut into a chosen body is a
    CrossBodyCutOp; everything else (extrude / self-cut / merge / new
    body) an ExtrudeOp.  merge_body_id NEW_BODY → new body."""
    from cad.op_types import ExtrudeOp, CrossBodyCutOp
    common = _common(direction, extra)
    new_body = merge_body_id == NEW_BODY
    if dist < 0 and merge_body_id is not None and not new_body:
        return CrossBodyCutOp(cut_body_id=merge_body_id, profiles=profiles,
                              distance=abs(dist), **common)
    return ExtrudeOp(profiles=profiles, distance=dist, force_new_body=new_body,
                     merge_body_id=None if new_body else merge_body_id,
                     **common)


def cut_all_intersecting(viewport, dist, direction, profiles, extra) -> int:
    """No-target cut: build the extrude tool solid once, then fan out one
    CrossBodyCutOp per workspace body it intersects (see cad.cut_all).  Each
    op gets its own copy of the (now baked) profiles.  Returns the number of
    bodies cut (0 also when the tool could not be built)."""
    import numpy as np
    from cad.op_types import CrossBodyCutOp
    from cad.profile_source import BodyFaceProfile
    from cad.cut_all import fan_out_cut

    common = _common(direction, extra)
    try:
        tool_faces = []
        for p in profiles:
            tool_faces += p.resolve_live(viewport, "[Cut]")
    except Exception as ex:
        print(f"[Cut] No-target cut: {ex}")
        return 0
    if not tool_faces:
        print("[Cut] No-target cut: no tool faces.")
        return 0

    # Tool for the bbox prefilter only (a hair deeper than the cut).
    probe = CrossBodyCutOp(cut_body_id="", profiles=profiles,
                           distance=abs(dist) + 0.01, **common)
    try:
        tool_solid = probe.tool_solid(
            tool_faces, normal=(np.asarray(direction, dtype=float)
                                if direction is not None else None))
    except Exception as ex:
        print(f"[Cut] Tool construction failed: {ex}")
        return 0

    def build_op(bid: str):
        return CrossBodyCutOp(cut_body_id=bid,
                              profiles=[p.copy() for p in profiles],
                              distance=abs(dist), **common)

    # A face-driven tool's source body goes last — cutting it first would
    # change the face the later ops re-find.
    faces = [p for p in profiles if isinstance(p, BodyFaceProfile)]
    return fan_out_cut(viewport, tool_solid, build_op, extra, op_label="Cut",
                       last=faces[0].body_id if faces else None)
