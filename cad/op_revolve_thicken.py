"""
cad/op_revolve_thicken.py

Revolve and thicken operation types:
  ThickenOp, RevolveOp (sketch loops and/or body faces), CrossBodyRevolveCutOp
"""

from __future__ import annotations
from cad.solid_ref import as_compound
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from cad.op_base import Op, _push_result

if TYPE_CHECKING:
    from cad.history import History


# ---------------------------------------------------------------------------
# ThickenOp  —  uniform outward offset of a body
# ---------------------------------------------------------------------------

@dataclass
class ThickenOp(Op):
    """
    Grow/shrink one or more faces on *source_body_id* by *thickness* mm.

    source_body_id : body whose faces are selected
    face_indices   : face indices at commit time (fallback for old entries)
    face_refs      : AnyFaceRef fingerprints — used for stable replay
    thickness      : mm — positive = grow outward, negative = cut inward
    """
    source_body_id: str
    face_indices:   list
    thickness:      float
    face_refs:      list = None   # list of AnyFaceRef, populated at commit

    def __post_init__(self):
        if self.face_refs is None:
            self.face_refs = []

    # Legacy single-face convenience
    @property
    def face_idx(self) -> int:
        return self.face_indices[0] if self.face_indices else 0

    def _resolve_faces(self, shape, history=None, entry_index=None):
        """Return list of (idx, b3d_face) using face_refs when available
        (provenance first)."""
        all_faces = list(shape.faces())
        resolved = []
        if self.face_refs:
            at = (self.source_body_id, entry_index) if entry_index is not None else None
            for ref in self.face_refs:
                idx, face = ref.resolve(history, shape, at)
                if idx is None:
                    raise RuntimeError(
                        f"ThickenOp: could not re-locate face "
                        f"(centroid={ref.centroid}, area={ref.area:.3f})")
                resolved.append((idx, face))
        else:
            for idx in self.face_indices:
                if idx >= len(all_faces):
                    raise RuntimeError(f"ThickenOp: face_idx {idx} out of range")
                resolved.append((idx, all_faces[idx]))
            if resolved:
                from cad import resolve_ledger
                resolve_ledger.note(resolve_ledger.FUZZY, "thicken face", "by index")
        return resolved

    def execute(self, shape: Any, history: "History", entry_index: int) -> Any:
        from cad.operations.thicken import thicken_face
        if shape is None:
            src = history._shape_for_body_at(self.source_body_id, entry_index)
            if src is None:
                raise RuntimeError(
                    f"ThickenOp: no shape for source body '{self.source_body_id}'")
            shape = src
        resolved = self._resolve_faces(shape, history, entry_index)
        result = shape
        for _idx, face in resolved:
            result = thicken_face(result.wrapped, face.wrapped, self.thickness)
        return result

    def commit(self, viewport: Any, extra_params: dict | None = None) -> Any:
        compute, finalize = self._split_commit(viewport, extra_params)
        try:
            shape_after = compute()
        except Exception as ex:
            print(f"[Op] FAILED: {ex}")
            shape_after = None
            viewport._pending_op_error = str(ex)
        else:
            viewport._pending_op_error = None
        try:
            finalize(shape_after)
        finally:
            viewport._pending_op_error = None
        return shape_after

    def _split_commit(self, viewport: Any, extra_params: dict | None = None):
        from cad.operations.thicken import thicken_face
        from cad.face_ref import AnyFaceRef

        shape_before = viewport.workspace.current_shape(self.source_body_id)
        if shape_before is None:
            raise RuntimeError(f"[Thicken] No shape for body {self.source_body_id}")
        all_faces = list(shape_before.faces())
        for idx in self.face_indices:
            if idx >= len(all_faces):
                raise RuntimeError(f"[Thicken] face_idx {idx} out of range")

        h = viewport.history
        self.face_refs = [
            AnyFaceRef.for_face(h, self.source_body_id, all_faces[idx].wrapped,
                                h.cursor + 1)
            for idx in self.face_indices
        ]

        body_occ   = shape_before.wrapped
        face_occs  = [all_faces[idx].wrapped for idx in self.face_indices]
        thickness  = self.thickness
        op_params  = self.to_params()
        if extra_params:
            op_params.update(extra_params)
        original_solid_count = len(list(shape_before.solids()))

        def compute():
            current_body = body_occ
            for face_occ in face_occs:
                result = thicken_face(current_body, face_occ, thickness)
                current_body = result.wrapped
            return result

        def finalize(shape_after):
            _push_result(viewport, "thicken", op_params, self.source_body_id,
                         None, shape_before, shape_after, original_solid_count)

        return compute, finalize

    def reopen(self, viewport: Any, history_idx: int) -> None:
        entry = viewport.history.entries[history_idx]
        vp = viewport
        if history_idx > 0:
            vp.history.seek(history_idx - 1)
            vp._rebuild_body_mesh(self.source_body_id)
        vp.history_changed.emit()
        # Saved face_indices are commit-time positions; upstream edits since
        # then renumber faces (replay re-finds them by face_refs).  Re-find
        # them the same way here, or the edit commits against stale indices.
        indices = list(self.face_indices)
        shape = vp.workspace.current_shape(self.source_body_id)
        if shape is not None and self.face_refs:
            try:
                indices = [i for i, _f in self._resolve_faces(shape, vp.history,
                                                               history_idx)]
            except RuntimeError as ex:
                print(f"[Thicken] reopen: {ex}; using saved face indices")
        vp._thicken_face_indices = indices
        vp._show_thicken_panel(self.source_body_id, indices, editing_entry=entry)
        panel = getattr(vp, '_thicken_panel', None)
        if panel is not None:
            panel.set_thickness(self.thickness)
            panel._spinbox.set_binding(entry.params.get("exprs", {}).get("thickness"),
                                       abs(self.thickness))
            panel._emit_preview()

    def to_params(self) -> dict:
        p: dict = {
            "source_body_id": self.source_body_id,
            "face_indices":   self.face_indices,
            "thickness":      self.thickness,
        }
        if self.face_refs:
            p["face_refs"] = [r.to_dict() for r in self.face_refs]
        return p

    @classmethod
    def _from_params(cls, params: dict, sign: int = 1) -> "ThickenOp":
        from cad.face_ref import AnyFaceRef
        if "face_indices" in params:
            indices = list(params["face_indices"])
        else:
            indices = [int(params.get("face_idx", 0))]
        raw_refs = params.get("face_refs", [])
        face_refs = [AnyFaceRef.from_dict(r) for r in raw_refs]
        return cls(
            source_body_id = params.get("source_body_id", ""),
            face_indices   = indices,
            thickness      = float(params.get("thickness", 0)),
            face_refs      = face_refs,
        )


# ---------------------------------------------------------------------------
# Revolve helpers (shared by RevolveOp and CrossBodyRevolveCutOp)
# ---------------------------------------------------------------------------

def _own_child_shape(entry, child_body_ids, assigned, shape):
    """What execute() returns for a face op whose result became new child
    bodies: the entry is tagged to its primary child (see finalize), so its
    shape is that child's solid — returning the (source) input shape left the
    child empty on replay/reload and broke every downstream op on it."""
    if entry.body_id in child_body_ids:
        own = assigned[child_body_ids.index(entry.body_id)]
        if own is None:
            raise RuntimeError("could not assign a solid to the primary child body")
        return own
    return shape


def _add_child_bodies(ws, solids, preserved_body_ids):
    """One new body per solid.  On an edit re-commit, solid i reuses
    preserved_body_ids[i] (and that body's name) so downstream entries that
    reference the old child bodies keep resolving."""
    bodies = []
    for i, solid in enumerate(solids):
        preserved = preserved_body_ids[i] if i < len(preserved_body_ids) else None
        existing = ws.bodies.get(preserved) if preserved else None
        name = existing.name if existing else ws.next_part_name()
        bodies.append(ws.add_body(name, as_compound(solid), body_id=preserved))
    return bodies


def _sketch_frame(se):
    import numpy as np
    return (np.asarray(se.plane_origin, dtype=float),
            np.asarray(se.plane_x_axis, dtype=float),
            np.asarray(se.plane_y_axis, dtype=float))


def _axis_line_ends(se, uid):
    """UV ends of the straight line (or straight include) with this uid."""
    import numpy as np
    from cad.sketch import LineEntity
    for ent in se.entities:
        if getattr(ent, "uid", None) != uid:
            continue
        if isinstance(ent, LineEntity):
            return np.asarray(ent.p0, dtype=float), np.asarray(ent.p1, dtype=float)
        pts = getattr(ent, "points", None)
        if pts is None or len(pts) < 2:
            return None
        pts = np.asarray(pts, dtype=float)
        a, b = pts[0], pts[-1]
        d = b - a
        L = float(np.linalg.norm(d))
        if L < 1e-9:
            return None
        off = np.abs((pts - a) @ np.array([-d[1], d[0]]) / L)
        return (a, b) if float(off.max()) < 1e-6 * max(1.0, L) else None
    return None


def axis_in_sketch(se, axis_pt, axis_dir, line_uid=None):
    """Sketch-relative form of a world axis, so it follows the sketch when an
    upstream edit moves the sketch's face (a world axis stays behind, as the
    extrude direction once did).  Returns (axis_local, axis_line_uid):
    axis_local = [u, v, du, dv] when the axis lies in the sketch plane;
    axis_line_uid kept when it names a straight line of this sketch."""
    import numpy as np
    if se is None or axis_pt is None or axis_dir is None:
        return None, None
    o, x, y = _sketch_frame(se)
    n = np.cross(x, y)
    p = np.asarray(axis_pt, dtype=float)
    d = np.asarray(axis_dir, dtype=float)
    d = d / np.linalg.norm(d)
    if line_uid is not None and _axis_line_ends(se, line_uid) is None:
        line_uid = None
    r = p - o
    if abs(float(d @ n)) > 1e-6 or abs(float(r @ n)) > 1e-6 * max(1.0, float(np.linalg.norm(r))):
        return None, line_uid
    return [float(r @ x), float(r @ y), float(d @ x), float(d @ y)], line_uid


def sketch_axis(se, axis_point, axis_dir, axis_local=None, axis_line_uid=None):
    """The world axis now: from the named sketch line, else the sketch-plane
    coordinates, else the stored world axis (older files, off-plane axes)."""
    import numpy as np
    if se is not None and (axis_line_uid or axis_local):
        o, x, y = _sketch_frame(se)
        ends = _axis_line_ends(se, axis_line_uid) if axis_line_uid else None
        if ends is not None:
            a, b = ends
            pt = o + a[0] * x + a[1] * y
            d = (b[0] - a[0]) * x + (b[1] - a[1]) * y
            if axis_local:
                ref = axis_local[2] * x + axis_local[3] * y
                if float(d @ ref) < 0:     # keep the authored rotation sense
                    d = -d
            return pt, d / np.linalg.norm(d)
        if axis_local:
            u, v, du, dv = axis_local
            d = du * x + dv * y
            return o + u * x + v * y, d / np.linalg.norm(d)
    return (np.array(axis_point, dtype=float), np.array(axis_dir, dtype=float))


def _axis_fields(viewport, sketch_id, se, axis_pt, axis_dir):
    """Commit-time axis_local / axis_line_uid; the line uid comes from the
    viewport's last axis pick when that was a line of this same sketch."""
    ref = getattr(viewport, "_revolve_axis_ref", None)
    uid = ref[1] if ref and ref[0] == sketch_id else None
    return axis_in_sketch(se, axis_pt, axis_dir, uid)

def _revolve_faces_to_tool(faces, axis_pt, axis_dir, angle_deg, side: int = 1):
    """Revolve each face and fuse the results into one tool solid (build123d
    Compound). Shared by commit and replay so the two can't drift."""
    from cad.operations.revolve import revolve_face_direct
    from build123d import Compound
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
    from OCP.collections import List_TopoDS_Shape as TopTools_ListOfShape

    tool = None
    for face in faces:
        s = revolve_face_direct(None, face, axis_pt, axis_dir, angle_deg, side)
        if tool is None:
            tool = s
        else:
            lst_a = TopTools_ListOfShape(); lst_a.Append(tool.wrapped)
            lst_b = TopTools_ListOfShape(); lst_b.Append(s.wrapped)
            fu = BRepAlgoAPI_Fuse()
            fu.SetArguments(lst_a); fu.SetTools(lst_b)
            fu.SetRunParallel(True); fu.Build()
            if fu.IsDone():
                tool = as_compound(fu.Shape())
    return tool


def _fuse_tool_into_target(tool, target_occ):
    """Fuse a revolved tool solid into a target body's shape. Returns a
    build123d Compound. Shared by commit and replay."""
    from build123d import Compound
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
    from OCP.collections import List_TopoDS_Shape as TopTools_ListOfShape

    lst_a = TopTools_ListOfShape(); lst_a.Append(target_occ)
    lst_b = TopTools_ListOfShape(); lst_b.Append(tool.wrapped)
    fu = BRepAlgoAPI_Fuse()
    fu.SetArguments(lst_a); fu.SetTools(lst_b)
    fu.SetRunParallel(True); fu.Build()
    if not fu.IsDone():
        raise RuntimeError("Revolve fuse failed.")
    return as_compound(fu.Shape())



def edge_axis(occ_edge):
    """(point, unit direction) of the axis an edge pick gives: the edge's
    parameter midpoint and its tangent there.  None for a degenerate edge."""
    import numpy as np
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    adp = BRepAdaptor_Curve(occ_edge)
    mid = (adp.FirstParameter() + adp.LastParameter()) * 0.5
    pt, d = adp.Value(mid), adp.DN(mid, 1)
    axis_dir = np.array([d.X(), d.Y(), d.Z()], dtype=float)
    norm = float(np.linalg.norm(axis_dir))
    if norm < 1e-10:
        return None
    return np.array([pt.X(), pt.Y(), pt.Z()], dtype=float), axis_dir / norm


def _follow_axis_edge(op, history, entry_index):
    """An axis picked from a body edge follows that edge: re-find it, take
    its midpoint + tangent (turned to the sense it was picked with, see
    oriented_tangent: a re-found edge can run the other way).  A lost edge is an error, not the stale
    axis.  Updates op and entry params; returns (point, dir)."""
    import numpy as np
    from cad.serializer import _edge_source_from_dict, _edge_source_to_dict
    from cad.op_extrude import refind_picked_edge, oriented_tangent, _with_sense
    before = history.fanout_members(entry_index)[0]
    src = _edge_source_from_dict(op.axis_source)
    try:
        occ = refind_picked_edge(src, history, before, "revolve axis edge")
        ax = edge_axis(occ) if occ is not None else None
    except Exception as ex:
        raise RuntimeError(f"revolve axis edge is gone, edit the op and pick "
                           f"the axis again ({ex})") from ex
    if ax is None:
        raise RuntimeError("revolve axis edge has no tangent")
    pt, d = ax
    last = np.asarray(op.axis_dir, dtype=float) if op.axis_dir is not None else None
    d, sense = oriented_tangent(op.axis_source, src, d, last, history, before)
    op.axis_point, op.axis_dir = pt.tolist(), d.tolist()
    op.axis_source = _with_sense(_edge_source_to_dict(src), sense)
    history._entries[entry_index].params.update(
        axis_point=op.axis_point, axis_dir=op.axis_dir, axis_source=op.axis_source)
    return pt, d


_ADOPT_AXIS_TOL = 1e-6    # mm: an edge pick's axis is that edge's midpoint


def _adopt_axis_edge(op, history, entry_index):
    """A revolve saved before edge axes were followed (or by a build that
    dropped the pick) has only a frozen world axis.  When that axis is
    exactly an edge pick's axis (the edge's midpoint, to 1e-6 mm, and its
    line) on a body as it stood before the revolve, it was picked from that
    edge: bake the edge as axis_source so the axis follows it from now on.
    Only ever while the geometry is still the one it was picked on (a moved
    edge no longer matches, and nothing is guessed)."""
    import numpy as np
    from cad.edge_ref import EdgeRef, BodyEdgeSource
    from cad.serializer import _edge_source_to_dict
    try:
        before = history.fanout_members(entry_index)[0]
        P = np.asarray(op.axis_point, dtype=float)
        D = np.asarray(op.axis_dir, dtype=float)
        D = D / np.linalg.norm(D)
        bids = []
        for e in history._entries[:before]:
            for b in [e.body_id] + list(e.params.get("child_body_ids") or []):
                if b and b not in bids:
                    bids.append(b)
        hits = []
        for bid in bids:
            shape = history._shape_for_body_at(bid, before)
            if shape is None:
                continue
            occ = getattr(shape, "wrapped", shape)
            for ed in shape.edges():
                ax = edge_axis(ed.wrapped)
                if ax is None:
                    continue
                if (np.linalg.norm(ax[0] - P) <= _ADOPT_AXIS_TOL
                        and abs(abs(float(ax[1] @ D)) - 1.0) < 1e-9):
                    hits.append((bid, ed.wrapped, occ))
        if len(hits) != 1:                     # none, or ambiguous (coincident edges)
            return
        bid, occ_edge, occ = hits[0]
        ref = EdgeRef.for_edge(history, bid, occ_edge, occ, before)
        if ref is None:
            return
        op.axis_source = _edge_source_to_dict(BodyEdgeSource(bid, ref, ref.origin))
        history._entries[entry_index].params["axis_source"] = op.axis_source
        print("[Revolve] axis recognised as a body edge pick; it follows that edge now")
    except Exception as ex:
        print(f"[Revolve] axis edge adoption skipped: {ex}")


def _revolve_axis(op, history, entry_index=None):
    """The op's world axis now.  An axis picked from a body edge follows the
    edge.  With a sketch profile the stored sketch-relative axis
    (axis_line_uid / axis_local, anchored to the first sketch) follows that
    sketch; otherwise the world axis is used as stored.  Call after the
    profiles resolved (they reproject the sketch)."""
    if (not op.axis_source and entry_index is not None and not op.axis_local
            and not op.axis_line_uid and op.axis_point is not None and op.axis_dir is not None):
        _adopt_axis_edge(op, history, entry_index)
    if op.axis_source and entry_index is not None:
        return _follow_axis_edge(op, history, entry_index)
    from cad.profile_source import sketch_profiles
    sp = sketch_profiles(op.profiles)
    se = sp[0].sketch_entry(history) if sp else None
    return sketch_axis(se, op.axis_point, op.axis_dir,
                       op.axis_local, op.axis_line_uid)


def _bake_revolve_axis(op, viewport):
    """Commit-time axis_source (the body edge the axis was picked from, when
    the panel's axis is still that pick) and axis_local / axis_line_uid
    against the first sketch."""
    import numpy as np
    from cad.profile_source import sketch_profiles
    op.axis_source = None
    picked = getattr(viewport, "_revolve_axis_edge", None)
    if picked is not None and op.axis_point is not None and op.axis_dir is not None:
        src, ppt, pdir = picked
        same = (np.linalg.norm(np.asarray(op.axis_point, dtype=float) - ppt) < 1e-6
                and abs(abs(float(np.dot(op.axis_dir, pdir))) - 1.0) < 1e-9)
        if same:
            op.axis_source = src
    sp = sketch_profiles(op.profiles)
    if not sp:
        op.axis_local, op.axis_line_uid = None, None
        return
    se = sp[0].sketch_entry(viewport.history)
    op.axis_local, op.axis_line_uid = _axis_fields(
        viewport, sp[0].sketch_id, se,
        np.array(op.axis_point, dtype=float), np.array(op.axis_dir, dtype=float))


def _revolve_axis_params(op, p: dict) -> dict:
    if getattr(op, "axis_side", 1) == -1:
        p["axis_side"] = -1
    if op.axis_local is not None:
        p["axis_local"] = op.axis_local
    if op.axis_line_uid is not None:
        p["axis_line_uid"] = op.axis_line_uid
    if op.axis_source is not None:
        p["axis_source"] = op.axis_source
    return p


def _restore_revolve_panel_profiles(vp, op, entry):
    """Open the revolve panel on the op's profiles (sketch loops first, then
    body faces re-found on the current shapes) and set the axis + angle."""
    from cad.profile_source import sketch_profiles
    sk = sketch_profiles(op.profiles)
    sketch_idx = vp.history.id_to_index(sk[0].sketch_id) if sk else None
    pairs = vp._live_face_pairs(op.profiles)
    if sk:
        vp._selected_sketch_entry = sketch_idx
        vp._selected_sketch_face = (list(sk[0].face_indices)
                                    if sk[0].face_indices else None)
    first_face = pairs[0] if pairs and not sk else (None, None)
    vp._show_revolve_panel(sketch_idx=sketch_idx, body_id=first_face[0],
                           face_idx=first_face[1], editing_entry=entry)
    panel = getattr(vp, "_revolve_panel", None)
    if panel is None:
        return None
    vp._revolve_sketch_idx = sketch_idx
    vp._revolve_face_pairs = list(pairs)
    vp._refresh_revolve_rows()
    vp._revolve_axis_edge = None
    if op.axis_point is not None and op.axis_dir is not None:
        se = sk[0].sketch_entry(vp.history) if sk else None
        if op.axis_line_uid and sk:
            vp._revolve_axis_ref = (sk[0].sketch_id, op.axis_line_uid)
        if op.axis_source:
            # Replay left the followed edge's axis in op.axis_point/dir.
            import numpy as np
            pt = np.array(op.axis_point, dtype=float)
            d = np.array(op.axis_dir, dtype=float)
            vp._revolve_axis_edge = (op.axis_source, pt, d)
            panel.set_axis(pt, d)
        else:
            panel.set_axis(*sketch_axis(se, op.axis_point, op.axis_dir,
                                        op.axis_local, op.axis_line_uid))
    panel.set_axis_side(getattr(op, "axis_side", 1))
    panel._angle_spinbox.set_binding(
        entry.params.get("exprs", {}).get("angle_deg"), abs(op.angle_deg))
    return panel


# ---------------------------------------------------------------------------
# RevolveOp  —  revolve sketch loops and/or body faces around an axis
# ---------------------------------------------------------------------------

@dataclass
class RevolveOp(Op):
    """
    Revolve profiles (sketch loops and/or body faces, see
    cad/profile_source.py) around an axis.  The result is new body/bodies,
    or is fused into *merge_body_id*.

    profiles       : ordered ProfileSource list
    angle_deg      : revolution arc in degrees (positive = CCW from axis_dir tip)
    axis_point     : world-space point on the rotation axis
    axis_dir       : unit direction of the rotation axis
    merge_body_id  : body to fuse into, or None for new body/bodies
    axis_local     : [u, v, du, dv] in the first sketch's plane — the axis
                     follows that sketch when an upstream edit moves it
    axis_line_uid  : the sketch line the axis follows
    axis_side      : when the axis runs through a profile, the side of it
                     revolved: +1 = axis_dir x profile normal, -1 = other
    """
    profiles:      list               = field(default_factory=list)
    angle_deg:     float              = 360.0
    axis_point:    list[float] | None = None
    axis_dir:      list[float] | None = None
    merge_body_id: str | None         = None
    axis_local:    list[float] | None = None
    axis_line_uid: str | None         = None
    axis_source:   dict | None        = None
    axis_side:     int                = 1

    def __post_init__(self):
        from cad.profile_source import as_profile
        self.profiles = [as_profile(p) for p in self.profiles]
        if self.merge_body_id == "__new_body__":
            self.merge_body_id = None

    @classmethod
    def from_sketch(cls, from_sketch_id: str, face_indices=None,
                    face_centroids=None, face_boundaries=None, **kw) -> "RevolveOp":
        from cad.profile_source import SketchProfile
        prof = SketchProfile(sketch_id=from_sketch_id,
                             face_indices=None if face_indices is None else list(face_indices),
                             face_centroids=face_centroids,
                             face_boundaries=face_boundaries)
        return cls(profiles=[prof], **kw)

    @classmethod
    def from_face(cls, source_body_id: str, face_idx: int, face_ref=None,
                  **kw) -> "RevolveOp":
        from cad.profile_source import BodyFaceProfile
        return cls(profiles=[BodyFaceProfile(source_body_id, int(face_idx), face_ref)], **kw)

    @property
    def from_sketch_ids(self) -> list[str]:
        from cad.profile_source import sketch_profiles
        return [p.sketch_id for p in sketch_profiles(self.profiles)]

    @property
    def from_sketch_id(self) -> str | None:
        ids = self.from_sketch_ids
        return ids[0] if ids else None

    @property
    def source_body_id(self) -> str | None:
        from cad.profile_source import BodyFaceProfile
        fp = [p for p in self.profiles if isinstance(p, BodyFaceProfile)]
        return fp[0].body_id if fp else None

    @property
    def force_new_body(self) -> bool:
        return self.merge_body_id is None

    @property
    def sketch_style(self) -> bool:
        from cad.profile_source import SketchProfile
        return bool(self.profiles) and isinstance(self.profiles[0], SketchProfile)

    def execute(self, shape: Any, history: "History", entry_index: int) -> Any:
        import numpy as np
        from cad.profile_source import faces_of
        from cad.op_extrude import _child_result, _largest_solid_with_imports
        name = "RevolveOp"
        faces = faces_of(self.profiles, history, entry_index, name)
        if not faces:
            raise RuntimeError(f"{name}: no profile faces")
        axis_pt, axis_dir = _revolve_axis(self, history, entry_index)
        tool = _revolve_faces_to_tool(faces, axis_pt, axis_dir, self.angle_deg,
                                      self.axis_side)
        if tool is None:
            raise RuntimeError(f"{name}: revolve produced no tool")

        # Merge: the entry is tagged to merge_body_id, so `shape` IS the
        # target — reproduce the commit's fuse instead of replacing it.
        if self.merge_body_id is not None:
            if shape is None:
                raise RuntimeError(f"{name}: merge target has no shape on replay")
            return _fuse_tool_into_target(tool, shape.wrapped)

        entry = history._entries[entry_index]
        own = _child_result(entry, history, entry_index, tool, shape,
                            self.sketch_style, name)
        if own is not None:
            return own
        if not self.sketch_style:
            return shape
        return _largest_solid_with_imports(entry, history, entry_index, tool, name)

    def commit(self, viewport: Any, extra_params: dict | None = None) -> Any:
        try:
            compute, finalize = self._split_commit(viewport, extra_params)
        except Exception as ex:
            print(f"[Op] FAILED: {ex}")
            self._push_failed_entry(viewport, str(ex), extra_params)
            return None
        try:
            shape_after = compute()
        except Exception as ex:
            print(f"[Op] FAILED: {ex}")
            shape_after = None
            viewport._pending_op_error = str(ex)
        else:
            viewport._pending_op_error = None
        try:
            finalize(shape_after)
        finally:
            viewport._pending_op_error = None
        return shape_after

    def _split_commit(self, viewport: Any, extra_params: dict | None = None):
        import numpy as np
        from cad.profile_source import faces_live
        from cad.op_extrude import (_apply_live_sketch_selection, _entry_face_ref,
                                    _clear_sketch_selection, _push_child_bodies)
        if not self.profiles:
            raise RuntimeError("[Revolve] No profile selected.")
        _apply_live_sketch_selection(self.profiles, viewport)
        faces = faces_live(self.profiles, viewport, "[Revolve]")
        _bake_revolve_axis(self, viewport)
        axis_pt  = np.array(self.axis_point, dtype=float)
        axis_dir = np.array(self.axis_dir,   dtype=float)
        angle    = self.angle_deg
        side     = self.axis_side
        face_ref = _entry_face_ref(viewport, self.profiles)
        host     = self.profiles[0].owner_body_id(viewport.history)
        profiles = list(self.profiles)

        op_params = self.to_params()
        if extra_params:
            op_params.update(extra_params)
        preserved_body_ids = list(op_params.pop("_preserved_body_ids", []))

        def tool():
            t = _revolve_faces_to_tool(faces, axis_pt, axis_dir, angle, side)
            if t is None:
                raise RuntimeError("Revolve produced no tool.")
            return t

        if self.merge_body_id is not None:
            target_shape = viewport.workspace.current_shape(self.merge_body_id)
            if target_shape is None:
                raise RuntimeError("[Revolve] Merge target has no shape.")
            target_occ = target_shape.wrapped
            target     = self.merge_body_id
            count      = len(list(target_shape.solids()))

            def compute():
                return _fuse_tool_into_target(tool(), target_occ)

            def finalize(merged):
                _push_result(viewport, "revolve", op_params, target,
                             None, target_shape, merged, count)
                _clear_sketch_selection(viewport, profiles)

            return compute, finalize

        # A face revolve records its source body's shape as shape_before.
        shape_before = (None if self.sketch_style or host is None
                        else viewport.workspace.current_shape(host))

        def finalize(shape_after):
            if shape_after is None:
                _push_result(viewport, "revolve", op_params, host, face_ref,
                             shape_before, None, 0)
            else:
                _push_child_bodies(viewport, "revolve", op_params, shape_after,
                                   preserved_body_ids, host, face_ref,
                                   shape_before=shape_before, log="[Revolve]")
            _clear_sketch_selection(viewport, profiles)

        return tool, finalize

    def reopen(self, viewport: Any, history_idx: int) -> None:
        vp = viewport
        entry = vp.history.entries[history_idx]
        if history_idx > 0:
            vp.history.seek(history_idx - 1)
            vp._rebuild_all_meshes()
        vp.history_changed.emit()
        panel = _restore_revolve_panel_profiles(vp, self, entry)
        if panel is None:
            return
        if self.merge_body_id and self.merge_body_id in vp.workspace.bodies:
            panel.set_merge_body(self.merge_body_id,
                                 vp.workspace.bodies[self.merge_body_id].name)
            panel._radio_merge.setChecked(True)
            panel._on_op_changed(1)
        panel._emit_preview()

    def to_params(self) -> dict:
        from cad.profile_source import profiles_to_dicts
        p: dict[str, Any] = {
            "angle_deg":  self.angle_deg,
            "profiles":   profiles_to_dicts(self.profiles),
            "axis_point": self.axis_point,
            "axis_dir":   self.axis_dir,
        }
        if self.merge_body_id is not None:
            p["merge_body_id"] = self.merge_body_id
        return _revolve_axis_params(self, p)

    @classmethod
    def _from_params(cls, params: dict) -> "RevolveOp":
        """Older saves: SketchRevolveOp keys (from_sketch_id + loop
        selection) or FaceRevolveOp keys (source_body_id + face_idx +
        face_ref)."""
        from cad.profile_source import (profiles_from_dicts, BodyFaceProfile,
                                        face_ref_from_dict)
        from cad.op_extrude import _legacy_sketch_profile
        if "profiles" in params:
            profiles = profiles_from_dicts(params["profiles"])
        elif "from_sketch_id" in params:
            profiles = [_legacy_sketch_profile(params["from_sketch_id"], params)]
        else:
            profiles = [BodyFaceProfile(
                body_id=params["source_body_id"],
                face_idx=int(params.get("face_idx", 0)),
                face_ref=face_ref_from_dict(params.get("face_ref")))]
        return cls(
            profiles      = profiles,
            angle_deg     = float(params.get("angle_deg", 360.0)),
            axis_point    = params.get("axis_point"),
            axis_dir      = params.get("axis_dir"),
            merge_body_id = params.get("merge_body_id"),
            axis_local    = params.get("axis_local"),
            axis_line_uid = params.get("axis_line_uid"),
            axis_source   = params.get("axis_source"),
            axis_side     = int(params.get("axis_side", 1)),
        )


# ---------------------------------------------------------------------------
# CrossBodyRevolveCutOp  —  revolve profiles and cut another body
# ---------------------------------------------------------------------------

@dataclass
class CrossBodyRevolveCutOp(Op):
    """
    Revolve profiles (sketch loops and/or body faces) around an axis into a
    tool solid and cut it from *cut_body_id*.  Mirrors CrossBodyCutOp.

    cut_body_id : body being cut
    profiles    : ProfileSource list
    angle_deg, axis_point, axis_dir, axis_local, axis_line_uid, axis_side :
                  see RevolveOp
    """
    cut_body_id:   str
    profiles:      list               = field(default_factory=list)
    angle_deg:     float              = 360.0
    axis_point:    list[float] | None = None
    axis_dir:      list[float] | None = None
    axis_local:    list[float] | None = None
    axis_line_uid: str | None         = None
    axis_source:   dict | None        = None
    axis_side:     int                = 1

    # Used by Op._push_failed_entry to tag the failed history entry correctly
    # (the base class's fallback otherwise produces "crossbodyrevolvecut").
    _op_str = "revolve_cut"

    def __post_init__(self):
        from cad.profile_source import as_profile
        self.profiles = [as_profile(p) for p in self.profiles]

    @classmethod
    def from_source(cls, cut_body_id: str, source_body_id: str | None = None,
                    source_sketch_id: str | None = None,
                    source_face_idx: int | None = None, face_indices=None,
                    face_centroids=None, face_boundaries=None,
                    **kw) -> "CrossBodyRevolveCutOp":
        """The pre-profile signature: a sketch (+ loop selection) or one face."""
        from cad.profile_source import SketchProfile, BodyFaceProfile
        if source_sketch_id is not None:
            prof = SketchProfile(sketch_id=source_sketch_id,
                                 face_indices=None if face_indices is None else list(face_indices),
                                 face_centroids=face_centroids,
                                 face_boundaries=face_boundaries)
        else:
            if source_face_idx is None:
                raise ValueError("CrossBodyRevolveCutOp: give a sketch or a source face")
            prof = BodyFaceProfile(body_id=source_body_id, face_idx=int(source_face_idx))
        return cls(cut_body_id=cut_body_id, profiles=[prof], **kw)

    @property
    def from_sketch_ids(self) -> list[str]:
        from cad.profile_source import sketch_profiles
        return [p.sketch_id for p in sketch_profiles(self.profiles)]

    @property
    def source_sketch_id(self) -> str | None:
        ids = self.from_sketch_ids
        return ids[0] if ids else None

    @property
    def body_id(self) -> str:
        return self.cut_body_id

    def _tool(self, faces, axis_pt, axis_dir):
        import numpy as np
        t = _revolve_faces_to_tool(faces, np.asarray(axis_pt, dtype=float),
                                   np.asarray(axis_dir, dtype=float), self.angle_deg,
                                   self.axis_side)
        if t is None:
            raise RuntimeError("CrossBodyRevolveCutOp: revolve produced no tool")
        return t

    def execute(self, shape: Any, history: "History", entry_index: int) -> Any:
        from cad.profile_source import faces_of
        from cad.operations.boolean import bool_op
        from cad.op_extrude import _fill_legacy_face_refs
        _fill_legacy_face_refs(self.profiles, history._entries[entry_index])
        faces = faces_of(self.profiles, history, entry_index, "CrossBodyRevolveCutOp")
        axis_pt, axis_dir = _revolve_axis(self, history, entry_index)
        result = bool_op("cut", [shape], [self._tool(faces, axis_pt, axis_dir)])
        if result is None:
            raise RuntimeError("CrossBodyRevolveCutOp: boolean cut failed")
        return result

    def commit(self, viewport: Any, extra_params: dict | None = None) -> Any:
        try:
            compute, finalize = self._split_commit(viewport, extra_params)
        except Exception as ex:
            print(f"[Op] FAILED: {ex}")
            self._push_failed_entry(viewport, str(ex), extra_params)
            return None
        try:
            shape_after = compute()
        except Exception as ex:
            print(f"[Op] FAILED: {ex}")
            shape_after = None
            viewport._pending_op_error = str(ex)
        else:
            viewport._pending_op_error = None
        try:
            finalize(shape_after)
        finally:
            viewport._pending_op_error = None
        return shape_after

    def _split_commit(self, viewport: Any, extra_params: dict | None = None):
        from cad.profile_source import faces_live, BodyFaceProfile
        from cad.operations.boolean import bool_op
        from cad.op_extrude import _apply_live_sketch_selection, _clear_sketch_selection
        if not self.profiles:
            raise RuntimeError("[Revolve cut] No profile selected.")
        # Baked loop indices (fan-out / edit) win over the viewport selection,
        # which the first finalize clears.
        _apply_live_sketch_selection(self.profiles, viewport)
        faces = faces_live(self.profiles, viewport, "[Revolve cut]")
        _bake_revolve_axis(self, viewport)
        axis_pt, axis_dir = self.axis_point, self.axis_dir
        first = self.profiles[0]
        face_ref = first.face_ref if isinstance(first, BodyFaceProfile) else None

        target_shape = viewport.workspace.current_shape(self.cut_body_id)
        if target_shape is None:
            raise RuntimeError("[Revolve cut] Target body has no shape.")
        count      = len(list(target_shape.solids()))
        target_occ = target_shape.wrapped
        profiles   = list(self.profiles)

        op_params = self.to_params()
        if extra_params:
            op_params.update(extra_params)

        def compute():
            result = bool_op("cut", [target_occ], [self._tool(faces, axis_pt, axis_dir)])
            if result is None:
                raise RuntimeError("Revolve cut failed.")
            return result

        def finalize(result):
            _push_result(viewport, "revolve_cut", op_params, self.cut_body_id,
                         face_ref, target_shape, result, count,
                         split_key="split_from")
            _clear_sketch_selection(viewport, profiles)

        return compute, finalize

    def reopen(self, viewport: Any, history_idx: int) -> None:
        vp = viewport
        entry = vp.history.entries[history_idx]
        if history_idx > 0:
            vp.history.seek(history_idx - 1)
            vp._rebuild_all_meshes()
        vp.history_changed.emit()
        panel = _restore_revolve_panel_profiles(vp, self, entry)
        if panel is None:
            return
        panel._radio_cut.setChecked(True)
        panel._on_mode_changed(1)
        # A fan-out member's cut_body_id is just one of the bodies it hit:
        # leave the target empty so the edit re-fans (no-target cut).
        if (self.cut_body_id in vp.workspace.bodies
                and "fanout_id" not in entry.params):
            panel.set_merge_body(self.cut_body_id,
                                 vp.workspace.bodies[self.cut_body_id].name)
        panel._emit_preview()

    def to_params(self) -> dict:
        from cad.profile_source import profiles_to_dicts
        p: dict[str, Any] = {
            "angle_deg":   self.angle_deg,
            "cut_body_id": self.cut_body_id,
            "profiles":    profiles_to_dicts(self.profiles),
            "axis_point":  self.axis_point,
            "axis_dir":    self.axis_dir,
        }
        return _revolve_axis_params(self, p)

    @classmethod
    def _from_params(cls, params: dict) -> "CrossBodyRevolveCutOp":
        """Older saves: source_sketch_id (+ loop selection), or
        source_body_id + source_face_idx with the face's FaceRef on the entry."""
        from cad.profile_source import profiles_from_dicts, BodyFaceProfile
        from cad.op_extrude import _legacy_sketch_profile
        if "profiles" in params:
            profiles = profiles_from_dicts(params["profiles"])
        elif params.get("source_sketch_id") is not None:
            profiles = [_legacy_sketch_profile(params["source_sketch_id"], params)]
        else:
            profiles = [BodyFaceProfile(body_id=params.get("source_body_id") or "",
                                        face_idx=int(params.get("source_face_idx") or 0))]
        return cls(
            cut_body_id   = params.get("cut_body_id", ""),
            profiles      = profiles,
            angle_deg     = float(params.get("angle_deg", 360.0)),
            axis_point    = params.get("axis_point"),
            axis_dir      = params.get("axis_dir"),
            axis_local    = params.get("axis_local"),
            axis_line_uid = params.get("axis_line_uid"),
            axis_source   = params.get("axis_source"),
            axis_side     = int(params.get("axis_side", 1)),
        )
