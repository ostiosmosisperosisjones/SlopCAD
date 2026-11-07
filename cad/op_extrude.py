"""
cad/op_extrude.py

Extrude and sketch operation types:
  ExtrudeOp (sketch loops and/or body faces), CrossBodyCutOp, SketchOp, ImportOp
"""

from __future__ import annotations
import math
from cad.solid_ref import as_compound
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from cad.op_base import Op, _push_result

if TYPE_CHECKING:
    from cad.history import History


# ---------------------------------------------------------------------------
# Shared merge helpers — build a tool solid from faces and fuse into a target.
# Used by both the commit and replay paths of the "merge with part" extrude
# flavors so the two can't drift (same family as the revolve merge fix).
# ---------------------------------------------------------------------------

def _extrude_faces_to_tool(faces, distance, direction,
                           start_offset, end_offset, draft_angle):
    """Extrude each face and fuse the results into one tool solid (build123d
    Compound)."""
    from cad.operations.extrude import _do_extrude_solid
    from build123d import Compound
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
    from OCP.collections import List_TopoDS_Shape as TopTools_ListOfShape

    tool = None
    for face in faces:
        s = _do_extrude_solid(face, distance, direction,
                              start_offset=start_offset, end_offset=end_offset,
                              draft_angle=draft_angle)
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
    """Fuse a tool solid into a target body's shape. Returns a build123d
    Compound."""
    from build123d import Compound
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
    from OCP.collections import List_TopoDS_Shape as TopTools_ListOfShape

    lst_a = TopTools_ListOfShape(); lst_a.Append(target_occ)
    lst_b = TopTools_ListOfShape(); lst_b.Append(tool.wrapped)
    fu = BRepAlgoAPI_Fuse()
    fu.SetArguments(lst_a); fu.SetTools(lst_b)
    fu.SetRunParallel(True); fu.Build()
    if not fu.IsDone():
        raise RuntimeError("Extrude fuse failed.")
    return as_compound(fu.Shape())


# ---------------------------------------------------------------------------
# Shared profile-op plumbing (extrude, cut, revolve, revolve-cut)
# ---------------------------------------------------------------------------

def _apply_live_sketch_selection(profiles, viewport) -> None:
    """A lone sketch profile with no baked loop selection takes the viewport's
    picked loops (callers that construct the op before baking).  Ops built
    for an edit or a fan-out already carry their indices and are untouched."""
    from cad.profile_source import SketchProfile
    sk = [p for p in profiles if isinstance(p, SketchProfile)]
    if len(sk) != 1 or sk[0].face_indices is not None:
        return
    sel = getattr(viewport, "_selected_sketch_face", None)
    if sel is None:
        return
    # Only the picked sketch's loops (an unset entry = no sketch tracked).
    idx = viewport.history.id_to_index(sk[0].sketch_id)
    if getattr(viewport, "_selected_sketch_entry", None) in (None, idx):
        sk[0].face_indices = list(sel)


def _clear_sketch_selection(viewport, profiles) -> None:
    from cad.profile_source import SketchProfile
    if any(isinstance(p, SketchProfile) for p in profiles):
        viewport._selected_sketch_entry = None
        viewport._selected_sketch_face  = None


def _sketch_parent_face_ref(viewport, se):
    """FaceRef of the body face a sketch sits on (the entry's face_ref for a
    sketch-driven op), or None for world / datum planes."""
    from cad.face_ref import FaceRef
    mesh = viewport._meshes.get(se.body_id) if se.body_id else None
    try:
        return FaceRef.from_b3d_face(mesh.occt_faces[se.face_idx]) if mesh else None
    except Exception:
        return None


def _entry_face_ref(viewport, profiles):
    """face_ref recorded on the pushed entry: the first face profile's ref
    when the op is face-driven, else the sketch's parent face."""
    from cad.profile_source import BodyFaceProfile, SketchProfile
    first = profiles[0]
    if isinstance(first, BodyFaceProfile):
        return first.face_ref
    se = first.sketch_entry(viewport.history)
    return _sketch_parent_face_ref(viewport, se) if se is not None else None


def _fill_legacy_face_refs(profiles, entry) -> None:
    """Face-driven cuts saved before profiles kept their face's FaceRef on the
    history entry, not the op — hand it to the (single) face profile."""
    from cad.profile_source import BodyFaceProfile
    fp = [p for p in profiles if isinstance(p, BodyFaceProfile)]
    ref = getattr(entry, "face_ref", None)
    if len(fp) == 1 and fp[0].face_ref is None and ref is not None:
        fp[0].face_ref = ref


def _push_child_bodies(viewport, op_str, op_params, shape_after,
                       preserved_body_ids, fallback_body_id, face_ref,
                       shape_before=None, log="[Extrude]"):
    """force_new_body finalize: one new body per solid (reusing preserved ids
    on an edit re-commit), a SolidRef per child so replay re-finds each
    segment by shape, and the parent entry tagged to the primary child."""
    from cad.units import format_op_label as _lbl
    from cad.solid_ref import solid_refs_to_dicts
    from cad.op_revolve_thicken import _add_child_bodies
    solids = list(shape_after.solids())
    op_params["child_solid_refs"] = solid_refs_to_dicts(solids)
    new_bodies = _add_child_bodies(viewport.workspace, solids, preserved_body_ids)
    op_params["child_body_ids"] = [b.id for b in new_bodies]
    for b in new_bodies:
        print(f"{log} New body '{b.name}'")
    tag = new_bodies[0].id if new_bodies else fallback_body_id
    # The entry is tagged to the primary child, so it carries that child's
    # solid only — what replay returns (_child_result).  The whole tool here
    # would show every solid on the primary child until the next replay.
    own = new_bodies[0].source_shape if new_bodies else shape_after
    parent = viewport.history.push(
        label=_lbl(op_str, op_params), operation=op_str, params=op_params,
        body_id=tag, face_ref=face_ref,
        shape_before=shape_before, shape_after=own)
    for b in new_bodies:
        b.created_at_entry_id = parent.entry_id
    viewport._rebuild_bodies({b.id for b in new_bodies})
    viewport.history_changed.emit()


def _child_result(entry, history, entry_index, result, shape, sketch_style, name):
    """force_new_body replay mirror: assign the result's solids to the child
    bodies by SolidRef (OCCT solid order isn't reproducible) and return the
    entry's own child.  None when the entry recorded no children."""
    child_body_ids = entry.params.get("child_body_ids", [])
    if not child_body_ids or history._workspace is None:
        return None
    from cad.solid_ref import assign_solids_to_children, solid_refs_from_dicts
    if not list(result.solids()):
        raise RuntimeError(f"{name}: result contains no solids")
    refs = solid_refs_from_dicts(entry.params.get("child_solid_refs"))
    assigned = assign_solids_to_children(result, child_body_ids, refs,
                                         history._workspace, history, entry_index)
    if sketch_style:
        if assigned[0] is None:
            raise RuntimeError(
                f"{name}: could not assign a solid to the primary child body")
        return assigned[0]
    from cad.op_revolve_thicken import _own_child_shape
    return _own_child_shape(entry, child_body_ids, assigned, shape)


def _largest_solid_with_imports(entry, history, entry_index, result, name):
    """Sketch-style in-place result: the body keeps its largest solid; split
    pieces go to the import entries this op produced (by solid_index)."""
    solids = sorted(result.solids(), key=lambda s: s.volume, reverse=True)
    if not solids:
        raise RuntimeError(f"{name}: result contains no solids")
    this_id = entry.entry_id
    for j in range(entry_index + 1, len(history._entries)):
        e = history._entries[j]
        if e.operation == "import" and e.params.get("source_entry_id") == this_id:
            k = e.params.get("solid_index", 1)
            e.shape_after = solids[k] if k < len(solids) else None
    return solids[0]


def _profile_common_kwargs(params: dict) -> dict:
    return dict(
        direction      = params.get("direction"),
        target_vertex  = params.get("target_vertex"),
        vertex_source  = params.get("vertex_source"),
        vertex_side    = int(params.get("vertex_side", 0)),
        direction_source = params.get("direction_source"),
        start_offset   = float(params.get("start_offset", 0.0)),
        end_offset     = float(params.get("end_offset", 0.0)),
        draft_angle    = float(params.get("draft_angle", 0.0)),
    )


def _find_vertex_source(history, point, before_index, entry):
    """A save from before vertex sources: bind the stored world point to the
    body vertex sitting exactly there, or None if none does (the part moved
    on since that point was stored, so nothing says which vertex it was)."""
    from cad.edge_ref import vertex_source_at
    if history._workspace is None:
        return None
    bids = [entry.body_id] + [b for b in history._workspace.bodies if b != entry.body_id]
    for bid in bids:
        shape = history._shape_for_body_at(bid, before_index)
        if shape is not None:
            src = vertex_source_at(history, bid, shape, point, before_index)
            if src is not None:
                return src
    return None


def edge_tangent(occ_edge):
    """Unit tangent at the middle of an edge (the panel's edge pick), or None."""
    import numpy as np
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    adp = BRepAdaptor_Curve(occ_edge)
    d = adp.DN((adp.FirstParameter() + adp.LastParameter()) * 0.5, 1)
    vec = np.array([d.X(), d.Y(), d.Z()], dtype=float)
    norm = float(np.linalg.norm(vec))
    return None if norm < 1e-10 else vec / norm


def _follow_direction_edge(op, history, entry_index) -> None:
    """A direction picked from an edge follows that edge: re-find it and take
    its tangent, turned to the side of the last direction (a re-found edge can
    run the other way).  A lost edge is an error, not the stale vector."""
    import numpy as np
    from cad.serializer import _edge_source_from_dict, _edge_source_to_dict
    if not op.direction_source or op.direction is None:
        return
    before = history.fanout_members(entry_index)[0]
    src = _edge_source_from_dict(op.direction_source)
    try:
        if src.origin is not None:
            # Named by its faces: found exactly or not at all.  The fuzzy
            # fallback could hand back a neighbour and turn the op silently.
            shape = history._shape_for_body_at(src.body_id, before)
            if shape is None:
                raise RuntimeError("its body has no shape")
            # EdgeOrigin settles for one face when the other is missing.
            if any(f.find_in(history, shape)[1] is None
                   for f in getattr(src.origin, "faces", [])):
                raise RuntimeError("a face it lies on is gone")
            found = src.origin.find_in(history, shape)
            if found is None:
                raise RuntimeError("its faces no longer meet")
            occ = [found]
        else:
            occ = src.resolve(history, before)[1]
        t = edge_tangent(occ[0]) if occ else None
    except Exception as ex:
        raise RuntimeError(f"direction edge is gone, edit the op and pick the "
                           f"direction edge again ({ex})") from ex
    if t is None:
        raise RuntimeError("direction edge has no tangent")
    if float(np.dot(t, np.asarray(op.direction, dtype=float))) < 0:
        t = -t
    op.direction = t.tolist()
    op.direction_source = _edge_source_to_dict(src)
    entry = history._entries[entry_index]
    entry.params.update(direction=op.direction, direction_source=op.direction_source)


def _follow_target_vertex(op, history, entry_index, face) -> None:
    """Re-find an up-to-vertex target and re-measure the depth to it, the way
    the panel does: along the direction until the end plane (parallel to the
    profile *face*) passes through the vertex.  A vertex that is gone is an
    error rather than a depth nobody asked for; a save that never recorded
    which vertex it was keeps its depth, with a warning."""
    import numpy as np
    from build123d import Plane
    from cad.serializer import _edge_source_from_dict, _edge_source_to_dict
    if op.target_vertex is None:
        return
    entry = history._entries[entry_index]
    # All of a fan-out measures on the part as it was before the first cut.
    before = history.fanout_members(entry_index)[0]
    src = (_edge_source_from_dict(op.vertex_source) if op.vertex_source
           else _find_vertex_source(history, op.target_vertex, before, entry))
    if src is None:
        entry.warning = ("Up-to-vertex target not found on the part; depth kept "
                         "as saved. Edit the op and pick the vertex again.")
        print(f"[Extrude] Entry {entry_index}: {entry.warning}")
        return
    try:
        v = np.asarray(src.resolve(history, before)[0][0], dtype=float)
    except Exception as ex:
        raise RuntimeError(f"up-to-vertex target is gone, edit the op and pick "
                           f"the vertex again ({ex})") from ex
    plane = Plane(face)
    n = np.array([plane.z_dir.X, plane.z_dir.Y, plane.z_dir.Z], dtype=float)
    o = np.array([plane.origin.X, plane.origin.Y, plane.origin.Z], dtype=float)
    d = op._direction(history)
    d = n if d is None else d / np.linalg.norm(d)
    cos = float(np.dot(d, n))
    if abs(cos) < 1e-6:
        raise RuntimeError("up-to-vertex: direction runs along the profile plane")
    reach = float(np.dot(v - o, n)) / cos
    side = op.vertex_side or (1 if reach >= 0 else -1)
    if reach * side < 0.001:
        raise RuntimeError("up-to-vertex target is no longer ahead of the profile")
    op.distance = math.copysign(reach * side, op.distance)
    op.target_vertex = v.tolist()
    op.vertex_source = _edge_source_to_dict(src)
    op.vertex_side = side
    entry.params.update(distance=abs(op.distance), target_vertex=op.target_vertex,
                        vertex_source=op.vertex_source, vertex_side=side)


def _legacy_sketch_profile(sketch_id, params: dict):
    from cad.profile_source import SketchProfile
    fi = params.get("face_indices")
    return SketchProfile(sketch_id=sketch_id,
                         face_indices=list(fi) if fi is not None else None,
                         face_centroids=params.get("face_centroids"),
                         face_boundaries=params.get("face_boundaries"))


# ---------------------------------------------------------------------------
# ExtrudeOp  —  extrude / self-cut sketch loops and/or body faces
# ---------------------------------------------------------------------------

@dataclass
class ExtrudeOp(Op):
    """
    Extrude (distance > 0) or cut (distance < 0) one or more profiles —
    sketch loops and body faces, mixed freely (cad/profile_source.py).

    profiles       : ordered ProfileSource list; the first decides the host
                     body (a sketch's parent body / the face's body) and the
                     replay style (sketch-first: the host keeps its largest
                     solid, split pieces go to import children; face-first:
                     the whole boolean result)
    distance       : signed mm — positive = extrude, negative = cut
    direction      : unit vector or None (None → each face's normal)
    target_vertex  : world-space point the depth is measured to, or None
    vertex_source  : serialized BodyVertexSource for it; replay re-finds the
                     vertex and re-measures distance (_follow_target_vertex)
    vertex_side    : +1 direction heads at the vertex, -1 away from it (the
                     panel's flip); 0 → not recorded, taken at first replay
    direction_source : serialized BodyEdgeSource when `direction` was picked
                     from an edge; replay re-reads the edge's tangent
                     (_follow_direction_edge)
    start_offset   : mm offset applied to the near end of the tool solid
    end_offset     : mm offset applied to the far end
    draft_angle    : taper angle in degrees (signed)
    force_new_body : every resulting solid becomes a new body
    merge_body_id  : fuse the extruded tool into this body instead
    merged_from    : the host body when merging (reopen UI fallback, legacy)
    along_normal   : ±1 → `direction` is the first sketch's normal (signed)
                     and follows it when an upstream edit turns the sketch
                     plane; 0 → `direction` is a fixed world vector.  Bound
                     once by bind_along_normal() while the normal is still
                     the authored one.

    In place, a face profile on the host body is re-found on the running
    shape and extruded glued to it (extrude_face); every other profile is
    extruded as a free face and fused / cut (extrude_face_direct).
    """
    profiles:       list               = field(default_factory=list)
    distance:       float              = 0.0
    direction:      list[float] | None = None
    target_vertex:  list[float] | None = None
    vertex_source:  dict | None = None
    vertex_side:    int   = 0
    direction_source: dict | None = None
    start_offset:   float = 0.0
    end_offset:     float = 0.0
    draft_angle:    float = 0.0
    force_new_body: bool  = False
    merge_body_id:  str | None = None
    merged_from:    str | None = None
    along_normal:   int   = 0

    def __post_init__(self):
        from cad.profile_source import as_profile
        self.profiles = [as_profile(p) for p in self.profiles]

    # -- construction ------------------------------------------------------

    @classmethod
    def from_sketch(cls, from_sketch_id: str, distance: float = 0.0,
                    face_indices=None, face_centroids=None, face_boundaries=None,
                    **kw) -> "ExtrudeOp":
        from cad.profile_source import SketchProfile
        prof = SketchProfile(sketch_id=from_sketch_id,
                             face_indices=None if face_indices is None else list(face_indices),
                             face_centroids=face_centroids,
                             face_boundaries=face_boundaries)
        return cls(profiles=[prof], distance=distance, **kw)

    @classmethod
    def from_faces(cls, source_body_id: str | None = None, face_idx: int = 0,
                   distance: float = 0.0, face_pairs=None, face_indices=None,
                   face_refs=None, **kw) -> "ExtrudeOp":
        """Body-face extrude: face_pairs [(body_id, face_idx)], else
        source_body_id + face_indices / face_idx; face_refs align with them."""
        from cad.profile_source import BodyFaceProfile
        if face_pairs is None:
            face_pairs = [(source_body_id, fi) for fi in (face_indices or [face_idx])]
        refs = list(face_refs or [])
        profs = [BodyFaceProfile(body_id=b, face_idx=int(fi),
                                 face_ref=refs[i] if i < len(refs) else None)
                 for i, (b, fi) in enumerate(face_pairs)]
        return cls(profiles=profs, distance=distance, **kw)

    # -- views over the profiles ------------------------------------------

    @property
    def from_sketch_ids(self) -> list[str]:
        from cad.profile_source import sketch_profiles
        return [p.sketch_id for p in sketch_profiles(self.profiles)]

    @property
    def from_sketch_id(self) -> str | None:
        ids = self.from_sketch_ids
        return ids[0] if ids else None

    @from_sketch_id.setter
    def from_sketch_id(self, sid: str) -> None:
        from cad.profile_source import sketch_profiles
        sp = sketch_profiles(self.profiles)
        if sp:
            sp[0].sketch_id = sid

    @property
    def face_pairs(self) -> list[tuple]:
        from cad.profile_source import BodyFaceProfile
        return [(p.body_id, p.face_idx) for p in self.profiles
                if isinstance(p, BodyFaceProfile)]

    @property
    def source_body_id(self) -> str | None:
        """First face profile's body (failed-entry tagging, reopen)."""
        pairs = self.face_pairs
        return pairs[0][0] if pairs else None

    @property
    def sketch_style(self) -> bool:
        from cad.profile_source import SketchProfile
        return bool(self.profiles) and isinstance(self.profiles[0], SketchProfile)

    def host_body_id(self, history) -> str | None:
        return self.profiles[0].owner_body_id(history) if self.profiles else None

    def creates_body_from_nothing(self, history: "History", entry_index: int) -> bool:
        """True when this extrude builds its body from scratch: new-body /
        merge mode, or a first profile that is a world-plane sketch (no prior
        shape; extrude_face_direct handles shape=None)."""
        if self.force_new_body or self.merge_body_id is not None:
            return True
        if not self.sketch_style:
            return False
        se = self.profiles[0].sketch_entry(history)
        if se is None:
            return False
        from cad.plane_ref import WorldPlaneSource
        return isinstance(se.plane_source, WorldPlaneSource)

    def _mode(self) -> str:
        if self.force_new_body:
            return "new"
        if self.merge_body_id is not None:
            return "merge"
        return "inplace"

    def _direction(self, history):
        """Effective direction once sketches are reprojected (along_normal
        re-reads the first sketch's normal)."""
        import numpy as np
        from cad.profile_source import sketch_profiles
        d = (np.array(self.direction, dtype=float)
             if self.direction is not None else None)
        sp = sketch_profiles(self.profiles)
        if self.along_normal and d is not None and sp:
            n = sp[0].plane_normal(history)
            if n is not None:
                d = self.along_normal * np.asarray(n, dtype=float)
        return d

    def _split_glued(self, host, mode):
        """(glued face profiles, other profiles): in place, faces on the host
        body extrude glued to it; everything else is a free tool face."""
        from cad.profile_source import BodyFaceProfile
        glued, free = [], []
        for p in self.profiles:
            if (mode == "inplace" and host is not None
                    and isinstance(p, BodyFaceProfile) and p.body_id == host):
                glued.append(p)
            else:
                free.append(p)
        return glued, free

    def _in_place(self, shape, glued, free_faces, direction):
        """Glued host faces first (re-found after every step), then free
        faces fused / cut into the running shape."""
        from cad.operations.extrude import extrude_face, extrude_face_direct
        kw = dict(direction=direction, start_offset=self.start_offset,
                  end_offset=self.end_offset, draft_angle=self.draft_angle)
        current = shape
        for p in glued:
            fi, _f = p.locate(current, "ExtrudeOp")
            current = extrude_face(current, fi, self.distance, **kw)
        for face in free_faces:
            current = extrude_face_direct(current, face, self.distance, **kw)
        return current

    def _tool(self, faces, direction):
        tool = _extrude_faces_to_tool(faces, self.distance, direction,
                                      self.start_offset, self.end_offset,
                                      self.draft_angle)
        if tool is None:
            raise RuntimeError("ExtrudeOp: extrude produced no tool")
        return tool

    # -- replay ------------------------------------------------------------

    def execute(self, shape: Any, history: "History", entry_index: int) -> Any:
        name = "ExtrudeOp"
        if not self.profiles:
            raise RuntimeError(f"{name}: no profiles")
        entry = history._entries[entry_index]
        mode = self._mode()
        host = self.host_body_id(history)
        glued, free = self._split_glued(host, mode)
        faces = []
        for p in free:
            faces.extend(p.resolve(history, entry_index, name))
        _follow_direction_edge(self, history, entry_index)
        direction = self._direction(history)
        if self.target_vertex is not None:
            first = faces[0] if faces else glued[0].locate(shape, name)[1]
            _follow_target_vertex(self, history, entry_index, first)

        # Merge: the entry is tagged to merge_body_id, so `shape` IS the
        # target's current shape — rebuild the tool and fuse it in, exactly
        # as the commit did.
        if mode == "merge":
            if shape is None:
                raise RuntimeError(f"{name}: merge target has no shape on replay")
            return _fuse_tool_into_target(self._tool(faces, direction), shape.wrapped)

        if mode == "new":
            result = self._tool(faces, direction)
            own = _child_result(entry, history, entry_index, result, shape,
                               self.sketch_style, name)
            if own is not None:
                return own
            if not self.sketch_style:
                return shape                    # legacy: source body unchanged
            return _largest_solid_with_imports(entry, history, entry_index,
                                               result, name)

        if glued and shape is None:
            raise RuntimeError(f"{name}: host body has no shape")
        result = self._in_place(shape, glued, faces, direction)
        if self.sketch_style:
            return _largest_solid_with_imports(entry, history, entry_index,
                                               result, name)
        return result

    # -- commit ------------------------------------------------------------

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
        from cad.profile_source import BodyFaceProfile
        if not self.profiles:
            raise RuntimeError("[Extrude] Nothing to extrude.")
        ws = viewport.workspace
        _apply_live_sketch_selection(self.profiles, viewport)
        mode = self._mode()
        host = self.host_body_id(viewport.history)
        glued, free = self._split_glued(host, mode)
        faces = []
        for p in self.profiles:
            got = p.resolve_live(viewport, "[Extrude]")
            if isinstance(p, BodyFaceProfile) and not p.is_planar():
                raise RuntimeError(
                    f"[Extrude] Face {p.face_idx} on body '{p.body_id}' is not planar.")
            if p in free:
                faces.extend(got)
        direction = self._direction(viewport.history)
        face_ref  = _entry_face_ref(viewport, self.profiles)
        op_str    = "cut" if self.distance < 0 else "extrude"
        profiles  = list(self.profiles)

        if mode == "merge":
            self.merged_from = host
        op_params = self.to_params()
        # Transient hint from the edit path; consumed below, never stored.
        preserved_body_ids = list((extra_params or {}).get("_preserved_body_ids", []))
        if extra_params:
            op_params.update(extra_params)
        op_params.pop("_preserved_body_ids", None)

        if mode == "merge":
            target_shape = ws.current_shape(self.merge_body_id)
            if target_shape is None:
                raise RuntimeError("[Extrude] Merge target has no shape.")
            target_occ = target_shape.wrapped
            count      = len(list(target_shape.solids()))
            target     = self.merge_body_id

            def compute():
                return _fuse_tool_into_target(self._tool(faces, direction), target_occ)

            def finalize(merged):
                _push_result(viewport, op_str, op_params, target,
                             None, target_shape, merged, count)
                _clear_sketch_selection(viewport, profiles)

            return compute, finalize

        if mode == "new":
            def compute():
                return self._tool(faces, direction)

            def finalize(shape_after):
                if shape_after is None:
                    _push_result(viewport, op_str, op_params, host, face_ref,
                                 None, None, 0)
                else:
                    _push_child_bodies(viewport, op_str, op_params, shape_after,
                                       preserved_body_ids, host, face_ref)
                _clear_sketch_selection(viewport, profiles)

            return compute, finalize

        shape_before = ws.current_shape(host) if host else None
        if glued and shape_before is None:
            raise RuntimeError(f"[Extrude] No shape for body {host}")
        count = len(list(shape_before.solids())) if shape_before else 0

        def compute():
            out = self._in_place(shape_before, glued, faces, direction)
            if out is None or not list(out.solids()):
                raise RuntimeError("Extrude produced no solids.")
            return out

        def finalize(shape_after):
            _push_result(viewport, op_str, op_params, host,
                         face_ref, shape_before, shape_after, count)
            _clear_sketch_selection(viewport, profiles)

        return compute, finalize

    # -- edit --------------------------------------------------------------

    def reopen(self, viewport: Any, history_idx: int) -> None:
        import numpy as np
        from cad.profile_source import sketch_profiles
        entry = viewport.history.entries[history_idx]
        vp = viewport

        if history_idx > 0:
            vp.history.seek(history_idx - 1)
            vp._rebuild_all_meshes()
        vp.history_changed.emit()

        sk = sketch_profiles(self.profiles)
        sketch_idx = vp.history.id_to_index(sk[0].sketch_id) if sk else None
        pairs = vp._live_face_pairs(self.profiles)
        if sk:
            vp._selected_sketch_entry = sketch_idx
            vp._selected_sketch_face = (list(sk[0].face_indices)
                                        if sk[0].face_indices else None)
        first_face = pairs[0] if pairs and not sk else (None, None)
        vp._show_extrude_panel(sketch_idx=sketch_idx, body_id=first_face[0],
                               face_idx=first_face[1], editing_entry=entry)
        panel = vp._extrude_panel
        if panel is None:
            return
        vp._extrude_sketch_idx = sketch_idx
        vp._extrude_face_pairs = list(pairs)
        vp._refresh_extrude_rows()

        if self.distance < 0:
            panel._radio_cut.setChecked(True)
            panel._on_mode_changed(1)

        # Restore the picked merge TARGET; merged_from is only a fallback for
        # legacy sketch entries saved before merge_body_id was persisted.
        if not self.force_new_body:
            merge_target = (self.merge_body_id
                            if self.merge_body_id in vp.workspace.bodies
                            else (self.merged_from if self.sketch_style else None))
            if merge_target and merge_target in vp.workspace.bodies:
                panel.set_merge_body(merge_target,
                                     vp.workspace.bodies[merge_target].name)
                panel._radio_merge.setChecked(True)
                panel._on_op_changed(1)

        _restore_extrude_fields(panel, entry, self)

    # -- params ------------------------------------------------------------

    def to_params(self) -> dict:
        from cad.profile_source import profiles_to_dicts
        p: dict[str, Any] = {"distance": abs(self.distance),
                             "profiles": profiles_to_dicts(self.profiles)}
        if self.direction is not None:
            p["direction"] = list(self.direction)
        if self.target_vertex is not None:
            p["target_vertex"] = self.target_vertex
        if self.vertex_source is not None:
            p["vertex_source"] = self.vertex_source
        if self.vertex_side:
            p["vertex_side"] = self.vertex_side
        if self.direction_source is not None:
            p["direction_source"] = self.direction_source
        if self.start_offset:
            p["start_offset"] = self.start_offset
        if self.end_offset:
            p["end_offset"] = self.end_offset
        if self.draft_angle:
            p["draft_angle"] = self.draft_angle
        if self.force_new_body:
            p["force_new_body"] = True
        if self.merge_body_id is not None:
            p["merge_body_id"] = self.merge_body_id
        if self.merged_from is not None:
            p["merged_from"] = self.merged_from
        if self.along_normal:
            p["along_normal"] = self.along_normal
        return p

    @classmethod
    def _from_params(cls, params: dict, sign: int = 1) -> "ExtrudeOp":
        """Current saves carry "profiles"; older ones the SketchExtrudeOp
        keys (from_sketch_id + face_indices/centroids/boundaries) or the
        FaceExtrudeOp keys (face_pairs / face_indices / face_idx +
        source_body_id, face_refs aligned with them)."""
        from cad.profile_source import (profiles_from_dicts, BodyFaceProfile,
                                        face_ref_from_dict)
        if "profiles" in params:
            profiles = profiles_from_dicts(params["profiles"])
        elif "from_sketch_id" in params:
            profiles = [_legacy_sketch_profile(params["from_sketch_id"], params)]
        else:
            src = params.get("source_body_id", "")
            fi = int(params.get("face_idx", 0))
            if params.get("face_pairs"):
                pairs = [(str(q[0]), int(q[1])) for q in params["face_pairs"]]
            elif params.get("face_indices"):
                pairs = [(src, int(i)) for i in params["face_indices"]]
            else:
                pairs = [(src, fi)]
            refs = [face_ref_from_dict(r) for r in params.get("face_refs") or []]
            profiles = [BodyFaceProfile(body_id=b, face_idx=i,
                                        face_ref=refs[k] if k < len(refs) else None)
                        for k, (b, i) in enumerate(pairs)]
        return cls(
            profiles       = profiles,
            distance       = float(params.get("distance", 0)) * sign,
            force_new_body = bool(params.get("force_new_body", False)),
            # Legacy face merges stored only "merged_from" (the source), never
            # the target: they load with merge_body_id=None.
            merge_body_id  = params.get("merge_body_id"),
            merged_from    = params.get("merged_from"),
            along_normal   = int(params.get("along_normal", 0)),
            **_profile_common_kwargs(params),
        )


def _restore_extrude_fields(panel, entry, op) -> None:
    """Direction / offsets / draft / depth back into a reopened extrude panel."""
    import numpy as np
    if op.direction is not None:
        panel.set_direction(np.array(op.direction, dtype=float), op.direction_source)
    exprs = entry.params.get("exprs", {})
    if op.start_offset or "start_offset" in exprs:
        panel._start_offset.set_binding(exprs.get("start_offset"), op.start_offset)
    if op.end_offset or "end_offset" in exprs:
        panel._end_offset.set_binding(exprs.get("end_offset"), op.end_offset)
    if op.draft_angle:
        panel.set_draft(op.draft_angle)
    if op.target_vertex is not None:
        panel._radio_vertex.setChecked(True)
        panel._on_target_mode_changed(1)
        if op.vertex_side < 0:          # authored flipped: keep heading away
            panel._flip_btn.setChecked(True)
            panel._on_flip(True)
        panel.set_vertex_target(np.array(op.target_vertex, dtype=float),
                                op.vertex_source)
    else:
        panel._spinbox.set_binding(exprs.get("distance"), abs(op.distance))
    panel._emit_preview()


# ---------------------------------------------------------------------------
# CrossBodyCutOp  —  cut a target body with extruded profiles
# ---------------------------------------------------------------------------

@dataclass
class CrossBodyCutOp(Op):
    """
    Extrude profiles (sketch loops and/or body faces) into a tool solid and
    cut it from *cut_body_id*.

    cut_body_id  : body being cut
    profiles     : ProfileSource list (see ExtrudeOp)
    distance     : depth of the tool solid (always positive; direction
                   encodes side)
    direction    : unit vector the tool extends along (same convention as the
                   preview: outward from the picked face, or whatever the user
                   flipped to).  None → each face's inward normal.
    target_vertex, vertex_source, vertex_side, start_offset, end_offset,
    draft_angle : as ExtrudeOp
    along_normal : ±1 → `direction` follows the first sketch's normal; see
                   ExtrudeOp.along_normal / bind_along_normal().
    """
    cut_body_id:   str
    profiles:      list               = field(default_factory=list)
    distance:      float              = 0.0
    direction:     list[float] | None = None
    target_vertex: list[float] | None = None
    vertex_source: dict | None = None
    vertex_side:   int   = 0
    direction_source: dict | None = None
    start_offset:  float = 0.0
    end_offset:    float = 0.0
    draft_angle:   float = 0.0
    along_normal:  int   = 0

    def __post_init__(self):
        from cad.profile_source import as_profile
        self.profiles = [as_profile(p) for p in self.profiles]

    @classmethod
    def from_source(cls, cut_body_id: str, source_body_id: str | None = None,
                    source_face_idx: int | None = None,
                    source_sketch_id: str | None = None,
                    face_indices=None, **kw) -> "CrossBodyCutOp":
        """The pre-profile signature: a sketch (source_sketch_id + optional
        loop indices) or one body face (source_body_id + source_face_idx)."""
        from cad.profile_source import SketchProfile, BodyFaceProfile
        if source_sketch_id is not None:
            prof = SketchProfile(sketch_id=source_sketch_id,
                                 face_indices=None if face_indices is None else list(face_indices))
        else:
            if source_face_idx is None:
                raise ValueError("CrossBodyCutOp: give a sketch or a source face")
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

    def tool_solid(self, faces, history_or_none=None, normal=None):
        """Fused tool solid for *faces* (direction resolved by the caller)."""
        import numpy as np
        from build123d import Plane
        from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
        from OCP.collections import List_TopoDS_Shape as TopTools_ListOfShape
        from cad.operations.extrude import _do_extrude_solid
        tool = None
        for face in faces:
            if normal is None:
                z = Plane(face).z_dir
                fd = -np.array([z.X, z.Y, z.Z], dtype=float)
            else:
                fd = normal
            s = _do_extrude_solid(face, self.distance, fd,
                                  start_offset=self.start_offset,
                                  end_offset=self.end_offset,
                                  draft_angle=self.draft_angle)
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

    def _direction(self, history):
        return ExtrudeOp._direction(self, history)

    def execute(self, shape: Any, history: "History", entry_index: int) -> Any:
        from cad.profile_source import faces_of
        from cad.operations.boolean import bool_op
        _fill_legacy_face_refs(self.profiles, history._entries[entry_index])
        faces = faces_of(self.profiles, history, entry_index, "CrossBodyCutOp")
        _follow_direction_edge(self, history, entry_index)
        if faces:
            _follow_target_vertex(self, history, entry_index, faces[0])
        tool = self.tool_solid(faces, normal=self._direction(history))
        result = bool_op("cut", [shape], [tool])
        if result is None:
            raise RuntimeError("CrossBodyCutOp: boolean cut failed")
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
        if not self.profiles:
            raise RuntimeError("[Cut] Nothing to cut with.")
        # Baked indices win over the mutable viewport selection — a fan-out
        # commits this op once per body and the first finalize clears it.
        _apply_live_sketch_selection(self.profiles, viewport)
        faces = faces_live(self.profiles, viewport, "[Cut]")
        if not faces:
            raise RuntimeError("[Cut] Sketch has no faces.")
        normal = self._direction(viewport.history)
        first = self.profiles[0]
        source_face_ref = first.face_ref if isinstance(first, BodyFaceProfile) else None

        target_shape = viewport.workspace.current_shape(self.cut_body_id)
        if target_shape is None:
            raise RuntimeError("[Cut] Target body has no shape.")
        count      = len(list(target_shape.solids()))
        target_occ = target_shape.wrapped
        profiles   = list(self.profiles)

        op_params = self.to_params()
        if extra_params:
            op_params.update(extra_params)

        def compute():
            result = bool_op("cut", [target_occ], [self.tool_solid(faces, normal=normal)])
            if result is None:
                raise RuntimeError("Boolean cut failed")
            return result

        def finalize(result):
            _push_result(viewport, "cut", op_params, self.cut_body_id,
                         source_face_ref, target_shape, result, count,
                         split_key="split_from")
            _clear_sketch_selection(viewport, profiles)

        return compute, finalize

    def reopen(self, viewport: Any, history_idx: int) -> None:
        from cad.profile_source import sketch_profiles
        entry = viewport.history.entries[history_idx]
        vp = viewport
        sk = sketch_profiles(self.profiles)
        sketch_idx = vp.history.id_to_index(sk[0].sketch_id) if sk else None

        if history_idx > 0:
            vp.history.seek(history_idx - 1)
            vp._rebuild_all_meshes()
        vp.history_changed.emit()

        pairs = vp._live_face_pairs(self.profiles)
        if sk:
            vp._selected_sketch_entry = sketch_idx
            vp._selected_sketch_face = (list(sk[0].face_indices)
                                        if sk[0].face_indices else None)
        first_face = pairs[0] if pairs and not sk else (None, None)
        vp._show_extrude_panel(sketch_idx=sketch_idx, body_id=first_face[0],
                               face_idx=first_face[1], editing_entry=entry)
        panel = vp._extrude_panel
        if panel is None:
            return
        vp._extrude_sketch_idx = sketch_idx
        vp._extrude_face_pairs = list(pairs)
        vp._refresh_extrude_rows()

        panel._radio_cut.setChecked(True)
        panel._on_mode_changed(1)

        # A fan-out member's cut_body_id is just one of the bodies it hit:
        # leave the target empty so the edit re-fans (no-target cut).
        if (self.cut_body_id in vp.workspace.bodies
                and "fanout_id" not in entry.params):
            panel.set_merge_body(self.cut_body_id,
                                 vp.workspace.bodies[self.cut_body_id].name)

        _restore_extrude_fields(panel, entry, self)

    def to_params(self) -> dict:
        from cad.profile_source import profiles_to_dicts
        p: dict[str, Any] = {"distance": self.distance,
                             "cut_body_id": self.cut_body_id,
                             "profiles": profiles_to_dicts(self.profiles)}
        if self.direction is not None:
            p["direction"] = list(self.direction)
        if self.target_vertex is not None:
            p["target_vertex"] = self.target_vertex
        if self.vertex_source is not None:
            p["vertex_source"] = self.vertex_source
        if self.vertex_side:
            p["vertex_side"] = self.vertex_side
        if self.direction_source is not None:
            p["direction_source"] = self.direction_source
        if self.start_offset:
            p["start_offset"] = self.start_offset
        if self.end_offset:
            p["end_offset"] = self.end_offset
        if self.draft_angle:
            p["draft_angle"] = self.draft_angle
        if self.along_normal:
            p["along_normal"] = self.along_normal
        return p

    @classmethod
    def _from_params(cls, params: dict, sign: int = 1) -> "CrossBodyCutOp":
        """Older saves: source_sketch_id (+ face_indices), or source_body_id +
        source_face_idx with the face's FaceRef on the history entry."""
        from cad.profile_source import profiles_from_dicts, BodyFaceProfile
        if "profiles" in params:
            profiles = profiles_from_dicts(params["profiles"])
        elif params.get("source_sketch_id") is not None:
            profiles = [_legacy_sketch_profile(params["source_sketch_id"], params)]
        else:
            profiles = [BodyFaceProfile(body_id=params.get("source_body_id", ""),
                                        face_idx=int(params.get("source_face_idx") or 0))]
        return cls(
            cut_body_id  = params.get("cut_body_id", ""),
            profiles     = profiles,
            distance     = float(params.get("distance", 0)),
            along_normal = int(params.get("along_normal", 0)),
            **_profile_common_kwargs(params),
        )


# ---------------------------------------------------------------------------
# SketchOp  —  sketch creation (geometry no-op, stores SketchEntry)
# ---------------------------------------------------------------------------

@dataclass
class SketchOp(Op):
    """Stores the SketchEntry; replay re-projects parametric references."""
    sketch_entry: Any  # cad.sketch.SketchEntry — avoid circular import

    def execute(self, shape: Any, history: "History", entry_index: int) -> Any:
        from cad.history import _replay_sketch_entry
        se = self.sketch_entry
        entry = (history._entries[entry_index]
                 if 0 <= entry_index < len(history._entries) else None)
        if entry is not None:
            entry.warning = ""
        if se is not None and se.plane_source is not None:
            ok, err = _replay_sketch_entry(se, history, before_index=entry_index, own=True)
            if not ok:
                raise RuntimeError(err)
            stale = getattr(se, "stale_includes", [])
            if stale and entry is not None:
                entry.warning = (
                    f"{len(stale)} included edge(s) couldn't be found after an upstream "
                    f"change and kept their last position (entity "
                    f"{', '.join(str(i) for i, _ in stale)}): {stale[0][1]}")
                print(f"[Sketch] Entry {entry_index}: {entry.warning}")
        return shape

    def to_params(self) -> dict:
        return {"sketch_entry": self.sketch_entry}

    @classmethod
    def _from_params(cls, params: dict, sign: int = 1) -> "SketchOp":
        return cls(sketch_entry=params.get("sketch_entry"))


# ---------------------------------------------------------------------------
# ImportOp  —  records a split solid or imported body
# ---------------------------------------------------------------------------

@dataclass
class ImportOp(Op):
    """
    Records a solid that was split off from a parent boolean operation,
    or an externally imported body.

    source_entry_id : entry_id of the parent op that produced the split
    split_from      : body_id of the parent body
    solid_index     : which solid in the parent result this body corresponds to
    """
    source_entry_id: str  | None = None
    split_from:      str  | None = None
    solid_index:     int  = 1

    def execute(self, shape: Any, history: "History", entry_index: int) -> Any:
        """Import ops carry their shape_after directly; nothing to recompute."""
        entry = history._entries[entry_index]
        return entry.shape_after

    def to_params(self) -> dict:
        p: dict[str, Any] = {}
        if self.source_entry_id is not None:
            p["source_entry_id"] = self.source_entry_id
        if self.split_from is not None:
            p["split_from"] = self.split_from
        if self.solid_index != 1:
            p["solid_index"] = self.solid_index
        return p

    @classmethod
    def _from_params(cls, params: dict, sign: int = 1) -> "ImportOp":
        return cls(
            source_entry_id = params.get("source_entry_id"),
            split_from      = params.get("split_from"),
            solid_index     = int(params.get("solid_index", 1)),
        )


def bind_along_normal(history, entry) -> None:
    """Record whether an extrude/cut's stored direction is just its first
    sketch's normal (the panel's default) — see ExtrudeOp.along_normal.
    Call while the sketch plane is still the one the op was authored
    against: at push, or on load before the first replay.  No-op once bound
    (the "along_normal" key is present) or for ops without a sketch profile."""
    import numpy as np
    from cad.profile_source import sketch_profiles
    op = entry.op
    if not isinstance(op, (ExtrudeOp, CrossBodyCutOp)):
        return
    sp = sketch_profiles(op.profiles)
    if not sp or "along_normal" in entry.params:
        return
    sign = 0
    if op.direction is not None:
        n = sp[0].plane_normal(history)
        if n is not None:
            d = np.asarray(op.direction, dtype=float)
            dot = float(np.dot(d / np.linalg.norm(d), np.asarray(n, dtype=float)))
            if abs(abs(dot) - 1.0) < 1e-9:
                sign = 1 if dot > 0 else -1
    op.along_normal = sign
    entry.params["along_normal"] = sign
