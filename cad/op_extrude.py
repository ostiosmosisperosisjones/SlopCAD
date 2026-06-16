"""
cad/op_extrude.py

Extrude and sketch operation types:
  FaceExtrudeOp, CrossBodyCutOp, SketchExtrudeOp, SketchOp, ImportOp
"""

from __future__ import annotations
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
    from OCP.TopTools import TopTools_ListOfShape

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
                tool = Compound(fu.Shape())
    return tool


def _fuse_tool_into_target(tool, target_occ):
    """Fuse a tool solid into a target body's shape. Returns a build123d
    Compound."""
    from build123d import Compound
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
    from OCP.TopTools import TopTools_ListOfShape

    lst_a = TopTools_ListOfShape(); lst_a.Append(target_occ)
    lst_b = TopTools_ListOfShape(); lst_b.Append(tool.wrapped)
    fu = BRepAlgoAPI_Fuse()
    fu.SetArguments(lst_a); fu.SetTools(lst_b)
    fu.SetRunParallel(True); fu.Build()
    if not fu.IsDone():
        raise RuntimeError("Extrude fuse failed.")
    return Compound(fu.Shape())


# ---------------------------------------------------------------------------
# FaceExtrudeOp  —  extrude or self-cut from a body face
# ---------------------------------------------------------------------------

@dataclass
class FaceExtrudeOp(Op):
    """
    Extrude (distance > 0) or self-cut (distance < 0) from a face on a body.

    source_body_id : body whose face is selected
    face_idx       : index of the face at op-creation time (FaceRef handles
                     parametric re-resolution during replay)
    distance       : signed mm — positive = extrude, negative = cut
    direction      : unit vector or None (None → use face normal)
    target_vertex  : world-space point the extrusion depth is measured to, or None
    start_offset   : mm offset applied to the near end of the tool solid
    end_offset     : mm offset applied to the far end
    force_new_body : True when the result should be a new body instead of
                     modifying the source body
    """
    source_body_id: str
    face_idx:       int
    distance:       float
    direction:      list[float] | None = None
    target_vertex:  list[float] | None = None
    start_offset:   float = 0.0
    end_offset:     float = 0.0
    draft_angle:    float = 0.0     # taper angle in degrees (signed)
    force_new_body: bool  = False
    face_indices:   list  = None   # face indices (single body, compat)
    face_refs:      list  = None   # FaceRef per face, built at commit
    face_pairs:     list  = None   # list of (body_id, face_idx) — multi-body support
    merge_body_id:  str | None = None  # fuse the extruded tool into this body

    def __post_init__(self):
        if self.face_pairs is None:
            if self.face_indices:
                self.face_pairs = [(self.source_body_id, fi) for fi in self.face_indices]
            else:
                self.face_pairs = [(self.source_body_id, self.face_idx)]
        if self.face_indices is None:
            self.face_indices = [fi for _, fi in self.face_pairs
                                 if _ == self.source_body_id]
        if self.face_refs is None:
            self.face_refs = []

    def _resolve_faces_on(self, shape, refs_subset, indices_subset):
        """Resolve a subset of faces on a given shape. Returns list of (idx, b3d_face)."""
        resolved = []
        if refs_subset:
            for ref in refs_subset:
                idx, face = ref.find_in(shape)
                if idx is None:
                    raise RuntimeError(
                        f"FaceExtrudeOp: could not re-locate face "
                        f"(normal={ref.normal}, area={ref.area:.3f})")
                resolved.append((idx, face))
        else:
            all_faces = list(shape.faces())
            for fi in indices_subset:
                if fi >= len(all_faces):
                    raise RuntimeError(f"FaceExtrudeOp: face_idx {fi} out of range")
                resolved.append((fi, all_faces[fi]))
        return resolved

    def execute(self, shape: Any, history: "History", entry_index: int) -> Any:
        """Replay via face_ref resolution + extrude."""
        import numpy as np
        from cad.operations.extrude import extrude_face, _do_extrude_solid
        from build123d import Compound
        direction = (np.array(self.direction, dtype=float)
                     if self.direction is not None else None)

        # Merge-into-target case: the entry is tagged to merge_body_id, so the
        # `shape` passed in by replay IS that target body's current shape. The
        # faces being extruded live on the SOURCE body, not the target, so we
        # resolve them on the source shape, build the tool, and fuse it into
        # `shape`. Without this, replay would self-extrude faces off the target
        # (which don't exist there) or drop the merge entirely.
        if self.merge_body_id is not None and not self.force_new_body:
            if shape is None:
                raise RuntimeError(
                    "FaceExtrudeOp: merge target has no shape on replay")
            refs = self.face_refs if self.face_refs else []
            faces = []
            for i, (bid, fi) in enumerate(self.face_pairs):
                src_shape = history._shape_for_body_at(bid, entry_index)
                if src_shape is None:
                    raise RuntimeError(
                        f"FaceExtrudeOp (merge): no shape for source body '{bid}'")
                if i < len(refs):
                    _, face_obj = refs[i].find_in(src_shape)
                    if face_obj is None:
                        raise RuntimeError(
                            f"FaceExtrudeOp: could not re-locate face on body '{bid}'")
                else:
                    all_f = list(src_shape.faces())
                    if fi >= len(all_f):
                        raise RuntimeError(f"FaceExtrudeOp: face_idx {fi} out of range")
                    face_obj = all_f[fi]
                faces.append(face_obj)
            tool = _extrude_faces_to_tool(faces, self.distance, direction,
                                          self.start_offset, self.end_offset,
                                          self.draft_angle)
            if tool is None:
                raise RuntimeError("FaceExtrudeOp: extrude produced no tool")
            return _fuse_tool_into_target(tool, shape.wrapped)

        if self.force_new_body:
            from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
            from OCP.TopTools import TopTools_ListOfShape

            result_solid = None
            refs = self.face_refs if self.face_refs else []
            for i, (bid, fi) in enumerate(self.face_pairs):
                src_shape = history._shape_for_body_at(bid, entry_index)
                if src_shape is None:
                    raise RuntimeError(
                        f"FaceExtrudeOp (force_new_body): no shape for body '{bid}'")
                if i < len(refs):
                    _, face_obj = refs[i].find_in(src_shape)
                    if face_obj is None:
                        raise RuntimeError(
                            f"FaceExtrudeOp: could not re-locate face on body '{bid}'")
                else:
                    all_f = list(src_shape.faces())
                    if fi >= len(all_f):
                        raise RuntimeError(f"FaceExtrudeOp: face_idx {fi} out of range")
                    face_obj = all_f[fi]
                extruded = _do_extrude_solid(face_obj, self.distance, direction,
                                             start_offset=self.start_offset,
                                             end_offset=self.end_offset,
                                             draft_angle=self.draft_angle)
                if result_solid is None:
                    result_solid = extruded.wrapped
                else:
                    lst_a = TopTools_ListOfShape(); lst_a.Append(result_solid)
                    lst_b = TopTools_ListOfShape(); lst_b.Append(extruded.wrapped)
                    fuse = BRepAlgoAPI_Fuse()
                    fuse.SetArguments(lst_a); fuse.SetTools(lst_b)
                    fuse.SetRunParallel(True); fuse.Build()
                    if fuse.IsDone():
                        result_solid = fuse.Shape()

            result = Compound(result_solid)
            entry = history._entries[entry_index]
            child_body_ids = entry.params.get("child_body_ids", [])
            if child_body_ids and history._workspace is not None:
                from cad.solid_ref import (assign_solids_to_children,
                                           solid_refs_from_dicts)
                refs = solid_refs_from_dicts(entry.params.get("child_solid_refs"))
                assign_solids_to_children(result, child_body_ids, refs,
                                          history._workspace, history, entry_index)
            return shape  # source body (shape) is unchanged

        current = shape
        refs = list(self.face_refs) if self.face_refs else []
        same_body_indices = [fi for _, fi in self.face_pairs]
        n = len(refs) if refs else len(same_body_indices)
        for i in range(n):
            if refs:
                fi, _ = refs[i].find_in(current)
                if fi is None:
                    raise RuntimeError(
                        f"FaceExtrudeOp: could not re-locate face "
                        f"(normal={refs[i].normal}, area={refs[i].area:.3f})")
                if i + 1 < len(refs):
                    refs = refs
            else:
                all_f = list(current.faces())
                fi = same_body_indices[i]
                if fi >= len(all_f):
                    raise RuntimeError(f"FaceExtrudeOp: face_idx {fi} out of range")
            current = extrude_face(current, fi, self.distance,
                                   direction=direction,
                                   start_offset=self.start_offset,
                                   end_offset=self.end_offset,
                                   draft_angle=self.draft_angle)
        return current

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
        from cad.operations.extrude import extrude_face, _do_extrude_solid
        from cad.face_ref import FaceRef
        from build123d import Compound

        mesh = viewport._meshes.get(self.source_body_id)
        if mesh is None:
            raise RuntimeError(f"[Extrude] No mesh for body {self.source_body_id}")
        shape_before = viewport.workspace.current_shape(self.source_body_id)
        if shape_before is None:
            raise RuntimeError(f"[Extrude] No shape for body {self.source_body_id}")

        self.face_refs = []
        face_objs = []
        for bid, fi in self.face_pairs:
            bshape = (shape_before if bid == self.source_body_id
                      else viewport.workspace.current_shape(bid))
            if bshape is None:
                raise RuntimeError(f"[Extrude] No shape for body '{bid}'")
            all_f = list(bshape.faces())
            if fi >= len(all_f):
                raise RuntimeError(f"[Extrude] face_idx {fi} out of range on body '{bid}'")
            ref = FaceRef.from_b3d_face(all_f[fi])
            if ref is None:
                raise RuntimeError(f"[Extrude] Face {fi} on body '{bid}' is not planar.")
            self.face_refs.append(ref)
            face_objs.append(all_f[fi])
        face_ref = self.face_refs[0]

        direction    = (np.array(self.direction, dtype=float)
                        if self.direction is not None else None)
        op_str       = "cut" if self.distance < 0 else "extrude"
        op_params    = self.to_params()
        # Transient hint from the edit path; consumed below, never stored.
        preserved_body_ids = list((extra_params or {}).get("_preserved_body_ids", []))
        if extra_params:
            op_params.update(extra_params)
        op_params.pop("_preserved_body_ids", None)
        original_solid_count = len(list(shape_before.solids()))

        # Merge into another body: build the tool from the source face(s) and
        # fuse it into the target. Tagged to merge_body_id so replay (which
        # honors merge_body_id) reproduces the same fuse. Single-body picks
        # only — multi-body picks are forced to new-body mode upstream.
        if (self.merge_body_id is not None and not self.force_new_body
                and len(self.face_pairs) <= 1):
            target_shape = viewport.workspace.current_shape(self.merge_body_id)
            if target_shape is None:
                raise RuntimeError("[Extrude] Merge target has no shape.")
            target_occ   = target_shape.wrapped
            merge_target = self.merge_body_id
            orig_merge_count = len(list(target_shape.solids()))
            distance     = self.distance
            start_offset = self.start_offset
            end_offset   = self.end_offset
            draft_angle  = self.draft_angle
            # merged_from records the source body for the reopen/edit UI.
            op_params["merged_from"] = self.source_body_id

            def compute():
                tool = _extrude_faces_to_tool(face_objs, distance, direction,
                                              start_offset, end_offset, draft_angle)
                if tool is None:
                    raise RuntimeError("Extrude produced no tool.")
                return _fuse_tool_into_target(tool, target_occ)

            def finalize(merged):
                _push_result(viewport, op_str, op_params, merge_target,
                             None, target_shape, merged, orig_merge_count)

            return compute, finalize

        if self.force_new_body:
            distance     = self.distance
            body_id      = self.source_body_id
            start_offset = self.start_offset
            end_offset   = self.end_offset
            draft_angle  = self.draft_angle

            def compute():
                from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
                from OCP.TopTools import TopTools_ListOfShape
                result_occ = None
                for fo in face_objs:
                    extruded = _do_extrude_solid(fo, distance, direction,
                                                 start_offset=start_offset,
                                                 end_offset=end_offset,
                                                 draft_angle=draft_angle)
                    if result_occ is None:
                        result_occ = extruded.wrapped
                    else:
                        lst_a = TopTools_ListOfShape(); lst_a.Append(result_occ)
                        lst_b = TopTools_ListOfShape(); lst_b.Append(extruded.wrapped)
                        fuse = BRepAlgoAPI_Fuse()
                        fuse.SetArguments(lst_a); fuse.SetTools(lst_b)
                        fuse.SetRunParallel(True); fuse.Build()
                        if fuse.IsDone():
                            result_occ = fuse.Shape()
                return Compound(result_occ)

            def finalize(shape_after):
                from viewer.vp_extrude import _strip_split_suffix, _next_split_name
                from cad.units import format_op_label as _lbl
                label     = _lbl(op_str, op_params)
                from cad.solid_ref import solid_refs_to_dicts
                root_name = _strip_split_suffix(viewport.workspace.bodies[body_id].name)
                solids    = list(shape_after.solids())
                op_params["child_body_ids"] = []
                # Per-child geometric fingerprint so replay re-locates each
                # segment by shape, not by OCCT solid-iteration index.
                op_params["child_solid_refs"] = solid_refs_to_dicts(solids)
                new_bodies = []
                for i, solid in enumerate(solids):
                    new_name = _next_split_name(root_name, viewport.workspace)
                    preserved = (preserved_body_ids[i]
                                 if i < len(preserved_body_ids) else None)
                    new_body = viewport.workspace.add_body(
                        new_name, Compound(solid.wrapped), body_id=preserved)
                    new_bodies.append(new_body)
                    op_params["child_body_ids"].append(new_body.id)
                    print(f"[Extrude] New body '{new_name}' ({len(list(solid.faces()))} faces)")
                primary_body = new_bodies[0] if new_bodies else None
                tag_body_id  = primary_body.id if primary_body else body_id
                parent_entry = viewport.history.push(
                    label=label, operation=op_str, params=op_params,
                    body_id=tag_body_id, face_ref=face_ref,
                    shape_before=None, shape_after=shape_after)
                for new_body in new_bodies:
                    new_body.created_at_entry_id = parent_entry.entry_id
                viewport._rebuild_bodies({b.id for b in new_bodies})
                viewport.history_changed.emit()

            return compute, finalize

        distance     = self.distance
        start_offset = self.start_offset
        end_offset   = self.end_offset
        draft_angle  = self.draft_angle
        face_refs_snap = list(self.face_refs)

        def compute():
            current = shape_before
            for ref in face_refs_snap:
                fi, _ = ref.find_in(current)
                if fi is None:
                    raise RuntimeError(
                        f"FaceExtrudeOp compute: could not locate face "
                        f"(normal={ref.normal}, area={ref.area:.3f})")
                current = extrude_face(current, fi, distance, direction=direction,
                                       start_offset=start_offset, end_offset=end_offset,
                                       draft_angle=draft_angle)
            if not list(current.solids()):
                raise RuntimeError("Boolean result has no solids.")
            return current

        def finalize(shape_after):
            _push_result(viewport, op_str, op_params, self.source_body_id,
                         face_ref, shape_before, shape_after, original_solid_count)

        return compute, finalize

    def reopen(self, viewport: Any, history_idx: int) -> None:
        import numpy as np
        entry = viewport.history.entries[history_idx]
        vp = viewport

        if history_idx > 0:
            vp.history.seek(history_idx - 1)
            vp._rebuild_all_meshes()
        vp.history_changed.emit()

        vp._show_extrude_panel(
            sketch_idx    = None,
            body_id       = self.source_body_id,
            face_idx      = self.face_indices[0] if self.face_indices else self.face_idx,
            editing_entry = entry)

        vp._extrude_face_pairs = list(self.face_pairs)

        panel = vp._extrude_panel
        if panel is None:
            return

        if len(self.face_pairs) > 1:
            panel.clear_face_entries()
            for bid, fi in self.face_pairs:
                body = vp.workspace.bodies.get(bid)
                name = body.name if body else bid
                panel.add_face_entry(bid, fi, f"{name}  ·  face {fi}")
            vp._update_panel_mode_lock(panel, self.face_pairs)

        if self.distance < 0:
            panel._radio_cut.setChecked(True)
            panel._on_mode_changed(1)

        # Restore the picked merge target so re-entering the panel keeps the
        # user's body choice (only single-body picks can merge — multi-body is
        # locked to new-body mode above).
        if (self.merge_body_id and not self.force_new_body
                and len(self.face_pairs) <= 1
                and self.merge_body_id in vp.workspace.bodies):
            panel.set_merge_body(
                self.merge_body_id,
                vp.workspace.bodies[self.merge_body_id].name)
            panel._radio_merge.setChecked(True)
            panel._on_op_changed(1)

        if self.direction is not None:
            panel.set_direction(np.array(self.direction, dtype=float))

        if self.start_offset:
            panel._start_offset.set_mm(self.start_offset)
        if self.end_offset:
            panel._end_offset.set_mm(self.end_offset)

        if self.draft_angle:
            panel.set_draft(self.draft_angle)

        if self.target_vertex is not None:
            panel._radio_vertex.setChecked(True)
            panel._on_target_mode_changed(1)
            panel.set_vertex_target(np.array(self.target_vertex, dtype=float))
        else:
            panel._spinbox.set_mm(abs(self.distance))

        panel._emit_preview()

    def to_params(self) -> dict:
        p: dict[str, Any] = {
            "distance":       abs(self.distance),
            "face_idx":       self.face_idx,
            "face_indices":   list(self.face_indices),
            "face_pairs":     [[bid, fi] for bid, fi in self.face_pairs],
            "source_body_id": self.source_body_id,
        }
        if self.face_refs:
            p["face_refs"] = [
                {"normal": list(r.normal), "area": r.area,
                 "centroid_perp": list(r.centroid_perp),
                 "centroid_along": r.centroid_along}
                for r in self.face_refs
            ]
        if self.direction is not None:
            p["direction"] = self.direction
        if self.target_vertex is not None:
            p["target_vertex"] = self.target_vertex
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
        return p

    @classmethod
    def _from_params(cls, params: dict, sign: int = 1) -> "FaceExtrudeOp":
        from cad.face_ref import FaceRef
        dist     = float(params.get("distance", 0)) * sign
        fi       = int(params.get("face_idx", 0))
        src_body = params.get("source_body_id", "")
        # "merged_from" is the legacy key the old face-merge commit path wrote;
        # it stored the SOURCE body there and tagged the entry to the target,
        # but never recorded the target id in params — so old entries can't be
        # replayed as merges. New entries store "merge_body_id" (the target).
        merge_body_id = params.get("merge_body_id")
        raw_pairs   = params.get("face_pairs")
        raw_indices = params.get("face_indices")
        if raw_pairs:
            face_pairs = [(str(p[0]), int(p[1])) for p in raw_pairs]
        elif raw_indices:
            face_pairs = [(src_body, int(i)) for i in raw_indices]
        else:
            face_pairs = [(src_body, fi)]
        face_indices = [i for _, i in face_pairs if _ == src_body] or [fi]
        raw_refs = params.get("face_refs", [])
        face_refs = [
            FaceRef(normal=tuple(r["normal"]), area=r["area"],
                    centroid_perp=tuple(r["centroid_perp"]),
                    centroid_along=r["centroid_along"])
            for r in raw_refs
        ]
        return cls(
            source_body_id = src_body,
            face_idx       = fi,
            face_indices   = face_indices,
            face_pairs     = face_pairs,
            face_refs      = face_refs if face_refs else None,
            distance       = dist,
            direction      = params.get("direction"),
            target_vertex  = params.get("target_vertex"),
            start_offset   = float(params.get("start_offset", 0.0)),
            end_offset     = float(params.get("end_offset", 0.0)),
            draft_angle    = float(params.get("draft_angle", 0.0)),
            force_new_body = bool(params.get("force_new_body", False)),
            merge_body_id  = merge_body_id,
        )


# ---------------------------------------------------------------------------
# CrossBodyCutOp  —  cut one body using a face from another body (or sketch)
# ---------------------------------------------------------------------------

@dataclass
class CrossBodyCutOp(Op):
    """
    Use a face (or sketch profile) extruded as a tool solid to cut a target body.

    cut_body_id      : body being cut
    source_body_id   : body whose face defines the tool profile
    source_face_idx  : face index on source_body (None if driven by sketch)
    source_sketch_id: entry_id of the driving sketch entry (None if face-driven)
    distance         : depth of the tool solid (always positive; direction encodes side)
    direction        : unit vector the tool extends along (same convention
                       as the preview: outward from the face the user picked,
                       or whatever the user flipped to via the panel).  When
                       None, the tool extrudes along the face's inward normal.
    target_vertex    : world-space point the tool depth is measured to, or None
    start_offset     : mm offset on near end
    end_offset       : mm offset on far end
    """
    cut_body_id:      str
    source_body_id:   str
    source_face_idx:  int  | None = None
    source_sketch_id: str  | None = None
    distance:          float = 0.0
    direction:         list[float] | None = None
    target_vertex:     list[float] | None = None
    start_offset:      float = 0.0
    end_offset:        float = 0.0
    draft_angle:       float = 0.0   # taper angle in degrees (signed)

    def execute(self, shape: Any, history: "History", entry_index: int) -> Any:
        import numpy as np
        from build123d import Compound, Plane
        from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
        from OCP.TopTools import TopTools_ListOfShape
        from cad.operations.extrude import _do_extrude_solid

        # stored direction IS the tool direction (matches preview convention).
        # When None, the per-face fallback below extrudes along -face_normal
        # (into the body the face is on) as a default for face-driven cuts.
        direction = (np.array(self.direction, dtype=float)
                     if self.direction is not None else None)
        tool_dir = direction

        if self.source_sketch_id is not None:
            sketch_idx = history.id_to_index(self.source_sketch_id)
            if sketch_idx is None:
                raise RuntimeError(
                    f"CrossBodyCutOp: sketch entry '{self.source_sketch_id}' not found")
            sketch_rec = history._entries[sketch_idx]
            if sketch_rec.error:
                raise RuntimeError(
                    "CrossBodyCutOp: source sketch entry is in an error state")
            sketch_entry = sketch_rec.params.get("sketch_entry")
            if sketch_entry is None:
                raise RuntimeError(
                    f"CrossBodyCutOp: no sketch_entry at id '{self.source_sketch_id}'")
            # Reproject the sketch's plane + reference entities against the
            # shape state as of *this* op's history position — not the sketch's
            # own position.  The sketch's plane_source is the abstract identity
            # (e.g. "body A's top face"); its geometric realisation must follow
            # any intervening operations that moved that face.  Otherwise the
            # tool solid lands at the sketch's original plane, producing
            # garbage cuts (or wholesale deletion of the target).
            if sketch_entry.plane_source is not None:
                from cad.history import reproject_consumed_sketch
                ok, err = reproject_consumed_sketch(sketch_entry, history,
                                                    entry_index, sketch_rec)
                if not ok:
                    raise RuntimeError(
                        f"CrossBodyCutOp: sketch reprojection failed: {err}")
            all_faces, _ = sketch_entry.build_faces()
            if not all_faces:
                raise RuntimeError("CrossBodyCutOp: sketch has no closed loops")
            faces = all_faces
        else:
            src_shape = history._shape_for_body_at(self.source_body_id, entry_index)
            if src_shape is None:
                raise RuntimeError(
                    f"CrossBodyCutOp: no shape for source body '{self.source_body_id}'")
            entry = history._entries[entry_index]
            face_ref = getattr(entry, "face_ref", None)
            if face_ref is not None:
                _, face_obj = face_ref.find_in(src_shape)
                if face_obj is None:
                    raise RuntimeError(
                        f"CrossBodyCutOp: could not locate source face "
                        f"(normal={face_ref.normal}, area={face_ref.area:.3f})")
                faces = [face_obj]
            else:
                if self.source_face_idx is None:
                    raise RuntimeError("CrossBodyCutOp: no source_face_idx")
                faces = [list(src_shape.faces())[self.source_face_idx]]

        tool_dist = self.distance + 0.01

        tool_solid = None
        for face in faces:
            if tool_dir is None:
                z = Plane(face).z_dir
                fd = -np.array([z.X, z.Y, z.Z], dtype=float)
            else:
                fd = tool_dir
            s = _do_extrude_solid(face, tool_dist, fd,
                                  start_offset=self.start_offset, end_offset=self.end_offset,
                                  draft_angle=self.draft_angle)
            if tool_solid is None:
                tool_solid = s
            else:
                lst_a = TopTools_ListOfShape(); lst_a.Append(tool_solid.wrapped)
                lst_b = TopTools_ListOfShape(); lst_b.Append(s.wrapped)
                fuse = BRepAlgoAPI_Fuse()
                fuse.SetArguments(lst_a); fuse.SetTools(lst_b)
                fuse.SetRunParallel(True); fuse.Build()
                if fuse.IsDone():
                    tool_solid = Compound(fuse.Shape())

        lst_a = TopTools_ListOfShape(); lst_a.Append(shape.wrapped)
        lst_b = TopTools_ListOfShape(); lst_b.Append(tool_solid.wrapped)
        op = BRepAlgoAPI_Cut()
        op.SetArguments(lst_a); op.SetTools(lst_b)
        op.SetRunParallel(True); op.Build()
        if not op.IsDone():
            raise RuntimeError("CrossBodyCutOp: boolean cut failed")
        return Compound(op.Shape())

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
        from build123d import Compound, Plane
        from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
        from OCP.TopTools import TopTools_ListOfShape
        from cad.operations.extrude import _do_extrude_solid

        direction = (np.array(self.direction, dtype=float)
                     if self.direction is not None else None)
        # See note in execute(): stored direction is the tool direction.
        tool_dir = direction

        source_face_ref = None
        if self.source_sketch_id is not None:
            sketch_idx = viewport.history.id_to_index(self.source_sketch_id)
            if sketch_idx is None:
                raise RuntimeError(f"[Cut] Sketch entry '{self.source_sketch_id}' not found.")
            entries = viewport.history.entries
            se = entries[sketch_idx].params.get("sketch_entry")
            if se is None:
                raise RuntimeError("[Cut] No sketch_entry found.")
            all_faces = viewport._sketch_faces.get(sketch_idx, [])
            if not all_faces:
                raise RuntimeError("[Cut] Sketch has no faces.")
            fidx = viewport._selected_sketch_face
            if fidx is not None:
                faces = [all_faces[i][0] for i in fidx if 0 <= i < len(all_faces)]
            else:
                faces = [f[0] for f in all_faces]
        else:
            from cad.face_ref import FaceRef, AnyFaceRef
            src_shape = viewport.workspace.current_shape(self.source_body_id)
            if src_shape is None:
                raise RuntimeError("[Cut] Source body has no shape.")
            if self.source_face_idx is None:
                raise RuntimeError("[Cut] No source_face_idx.")
            src_faces_list = list(src_shape.faces())
            if self.source_face_idx >= len(src_faces_list):
                raise RuntimeError("[Cut] source_face_idx out of range.")
            src_face = src_faces_list[self.source_face_idx]
            source_face_ref = (FaceRef.from_occ_face(src_face.wrapped)
                               or AnyFaceRef.from_occ_face(src_face.wrapped))
            faces = [src_face]

        target_shape = viewport.workspace.current_shape(self.cut_body_id)
        if target_shape is None:
            raise RuntimeError("[Cut] Target body has no shape.")
        original_solid_count = len(list(target_shape.solids()))
        target_occ   = target_shape.wrapped
        tool_dist    = self.distance + 0.01
        start_offset = self.start_offset
        end_offset   = self.end_offset
        draft_angle  = self.draft_angle

        op_params = self.to_params()
        if extra_params:
            op_params.update(extra_params)

        def compute():
            tool_solid = None
            for face in faces:
                if tool_dir is not None:
                    fd = tool_dir
                else:
                    z  = Plane(face).z_dir
                    fd = -np.array([z.X, z.Y, z.Z], dtype=float)
                s = _do_extrude_solid(face, tool_dist, fd,
                                      start_offset=start_offset, end_offset=end_offset,
                                      draft_angle=draft_angle)
                if tool_solid is None:
                    tool_solid = s
                else:
                    lst_a = TopTools_ListOfShape(); lst_a.Append(tool_solid.wrapped)
                    lst_b = TopTools_ListOfShape(); lst_b.Append(s.wrapped)
                    fuse = BRepAlgoAPI_Fuse()
                    fuse.SetArguments(lst_a); fuse.SetTools(lst_b)
                    fuse.SetRunParallel(True); fuse.Build()
                    if fuse.IsDone():
                        tool_solid = Compound(fuse.Shape())
            lst_a = TopTools_ListOfShape(); lst_a.Append(target_occ)
            lst_b = TopTools_ListOfShape(); lst_b.Append(tool_solid.wrapped)
            op = BRepAlgoAPI_Cut()
            op.SetArguments(lst_a); op.SetTools(lst_b)
            op.SetRunParallel(True); op.Build()
            if not op.IsDone():
                raise RuntimeError("Boolean cut failed")
            return Compound(op.Shape())

        def finalize(result):
            _push_result(viewport, "cut", op_params, self.cut_body_id,
                         source_face_ref, target_shape, result, original_solid_count,
                         split_key="split_from")
            viewport._selected_sketch_entry = None
            viewport._selected_sketch_face  = None

        return compute, finalize

    def reopen(self, viewport: Any, history_idx: int) -> None:
        import numpy as np
        entry      = viewport.history.entries[history_idx]
        vp         = viewport
        sketch_idx = (vp.history.id_to_index(self.source_sketch_id)
                      if self.source_sketch_id is not None else None)

        if history_idx > 0:
            vp.history.seek(history_idx - 1)
            rebuild_id = (self.source_body_id if sketch_idx is None
                          else entry.body_id)
            vp._rebuild_body_mesh(rebuild_id)
        vp.history_changed.emit()

        if sketch_idx is not None:
            vp._selected_sketch_entry = sketch_idx
            vp._selected_sketch_face  = None

        vp._show_extrude_panel(
            sketch_idx    = sketch_idx,
            body_id       = self.source_body_id if sketch_idx is None else None,
            face_idx      = self.source_face_idx if sketch_idx is None else None,
            editing_entry = entry)

        panel = vp._extrude_panel
        if panel is None:
            return

        panel._radio_cut.setChecked(True)
        panel._on_mode_changed(1)

        if self.cut_body_id in vp.workspace.bodies:
            panel.set_merge_body(
                self.cut_body_id,
                vp.workspace.bodies[self.cut_body_id].name)

        if self.direction is not None:
            panel.set_direction(np.array(self.direction, dtype=float))

        if self.start_offset:
            panel._start_offset.set_mm(self.start_offset)
        if self.end_offset:
            panel._end_offset.set_mm(self.end_offset)

        if self.draft_angle:
            panel.set_draft(self.draft_angle)

        if self.target_vertex is not None:
            panel._radio_vertex.setChecked(True)
            panel._on_target_mode_changed(1)
            panel.set_vertex_target(np.array(self.target_vertex, dtype=float))
        else:
            panel._spinbox.set_mm(self.distance)

        panel._emit_preview()

    def to_params(self) -> dict:
        p: dict[str, Any] = {
            "distance":       self.distance,
            "cut_body_id":    self.cut_body_id,
            "source_body_id": self.source_body_id,
        }
        if self.source_face_idx is not None:
            p["source_face_idx"] = self.source_face_idx
        if self.source_sketch_id is not None:
            p["source_sketch_id"] = self.source_sketch_id
        if self.direction is not None:
            p["direction"] = self.direction
        if self.target_vertex is not None:
            p["target_vertex"] = self.target_vertex
        if self.start_offset:
            p["start_offset"] = self.start_offset
        if self.end_offset:
            p["end_offset"] = self.end_offset
        if self.draft_angle:
            p["draft_angle"] = self.draft_angle
        return p

    @classmethod
    def _from_params(cls, params: dict, sign: int = 1) -> "CrossBodyCutOp":
        return cls(
            cut_body_id      = params.get("cut_body_id", ""),
            source_body_id   = params.get("source_body_id", ""),
            source_face_idx  = params.get("source_face_idx"),
            source_sketch_id = params.get("source_sketch_id"),
            distance         = float(params.get("distance", 0)),
            direction        = params.get("direction"),
            target_vertex    = params.get("target_vertex"),
            start_offset     = float(params.get("start_offset", 0.0)),
            end_offset       = float(params.get("end_offset", 0.0)),
            draft_angle      = float(params.get("draft_angle", 0.0)),
        )


# ---------------------------------------------------------------------------
# SketchExtrudeOp  —  extrude or cut a sketch profile
# ---------------------------------------------------------------------------

@dataclass
class SketchExtrudeOp(Op):
    """
    Extrude (or cut) a closed-loop profile built from a SketchEntry.

    from_sketch_id  : entry_id of the sketch entry
    distance        : signed mm — positive = extrude, negative = cut
    merge_body_id   : body to fuse/cut into, or None for a new body
    face_indices    : which faces of the sketch to extrude (None = all)
    direction       : unit vector or None
    target_vertex   : world-space depth target, or None
    start_offset    : mm
    end_offset      : mm
    force_new_body  : True → always create a new body
    merged_from     : source body id when fusing into another body (legacy key)
    """
    from_sketch_id: str
    distance:        float
    merge_body_id:   str  | None       = None
    face_indices:    list[int] | None  = None
    face_centroids:  list[list[float]] | None = None
    direction:       list[float] | None = None
    target_vertex:   list[float] | None = None
    start_offset:    float = 0.0
    end_offset:      float = 0.0
    draft_angle:     float = 0.0       # taper angle in degrees (signed)
    force_new_body:  bool  = False
    merged_from:     str   | None = None

    def creates_body_from_nothing(self, history: "History", entry_index: int) -> bool:
        """
        True when this sketch extrude is rooted on a world plane and has no
        prior shape on its body — i.e. it's creating the body from scratch.
        extrude_face_direct() handles shape=None for exactly this case.
        """
        if self.force_new_body or self.merge_body_id is not None:
            return True
        sketch_idx = history.id_to_index(self.from_sketch_id)
        if sketch_idx is None:
            return False
        se = history._entries[sketch_idx].params.get("sketch_entry")
        if se is None:
            return False
        from cad.plane_ref import WorldPlaneSource
        return isinstance(se.plane_source, WorldPlaneSource)

    def execute(self, shape: Any, history: "History", entry_index: int) -> Any:
        from cad.operations.extrude import extrude_face_direct
        from build123d import Compound

        sketch_idx = history.id_to_index(self.from_sketch_id)
        if sketch_idx is None:
            raise RuntimeError(
                f"SketchExtrudeOp: sketch entry '{self.from_sketch_id}' not found")
        if sketch_idx >= entry_index:
            raise RuntimeError(
                f"SketchExtrudeOp: sketch entry is after this entry — invalid reorder")

        sketch_entry_rec = history._entries[sketch_idx]
        if sketch_entry_rec.error:
            raise RuntimeError(f"SketchExtrudeOp: sketch entry is in an error state")

        se = sketch_entry_rec.params.get("sketch_entry")
        if se is None:
            raise RuntimeError(
                f"SketchExtrudeOp: no sketch_entry at id '{self.from_sketch_id}'")

        if se.plane_source is not None:
            from cad.history import reproject_consumed_sketch
            # Reproject at *this* op's history position so the sketch's plane
            # follows any intervening operations on its parent body.  Using
            # sketch_idx would freeze the plane at the sketch's creation time
            # and produce wrong geometry when something moved the parent face
            # between the sketch and this extrude.  Falls back to the authored
            # plane if an intervening op destroyed the anchor face.
            ok, err = reproject_consumed_sketch(se, history, entry_index,
                                                sketch_entry_rec)
            if not ok:
                # No trustworthy plane (sketch entry itself errored) — fail
                # loudly instead of extruding against stale geometry.
                raise RuntimeError(
                    f"SketchExtrudeOp: sketch reprojection failed: {err}")

        all_faces, all_regions = se.build_faces()
        if not all_faces:
            raise RuntimeError("SketchExtrudeOp: sketch has no closed loops")

        if self.face_centroids:
            import numpy as np
            from cad.op_base import _match_face_by_centroid
            centroids = np.array(self.face_centroids, dtype=float)
            faces = [_match_face_by_centroid(t, all_faces, all_regions,
                                             "SketchExtrudeOp")
                     for t in centroids]
        elif self.face_indices is not None:
            faces = [all_faces[i] for i in self.face_indices if i < len(all_faces)]
            if not faces:
                raise RuntimeError("SketchExtrudeOp: stored face indices out of range")
        else:
            faces = all_faces

        signed_dist = -abs(self.distance) if self.distance < 0 else abs(self.distance)

        import numpy as np
        direction = (np.array(self.direction, dtype=float)
                     if self.direction is not None else None)

        # Merge-into-target case: the entry is tagged to merge_body_id, so the
        # `shape` passed in by replay IS that target body's current shape. Build
        # the tool from the sketch profile and fuse it in, reproducing the
        # commit. Without this, replay would diverge — dropping disjoint solids
        # (only solids[0] is returned below) or reordering them by volume so the
        # split-import children no longer line up.
        if self.merge_body_id is not None and not self.force_new_body:
            if shape is None:
                raise RuntimeError(
                    "SketchExtrudeOp: merge target has no shape on replay")
            tool = _extrude_faces_to_tool(faces, signed_dist, direction,
                                          self.start_offset, self.end_offset,
                                          self.draft_angle)
            if tool is None:
                raise RuntimeError("SketchExtrudeOp: extrude produced no tool")
            return _fuse_tool_into_target(tool, shape.wrapped)

        result = shape
        for face in faces:
            result = extrude_face_direct(result, face, signed_dist,
                                         direction=direction,
                                         start_offset=self.start_offset,
                                         end_offset=self.end_offset,
                                         draft_angle=self.draft_angle)

        entry = history._entries[entry_index]
        child_body_ids = entry.params.get("child_body_ids", [])

        # force_new_body split: each disjoint profile becomes its own child
        # body. The commit path (finalize() in _split_commit) creates those
        # bodies; this branch is its replay mirror — without it, reload leaves
        # the non-primary children with no shape. Assign by SolidRef so each
        # child gets *its* segment regardless of OCCT solid-iteration order
        # (positional index is not reproducible across rebuilds). The parent
        # entry is tagged to the primary child, so we return that child's solid.
        if self.force_new_body and child_body_ids:
            from cad.solid_ref import assign_solids_to_children, solid_refs_from_dicts
            if not list(result.solids()):
                raise RuntimeError("SketchExtrudeOp: result contains no solids")
            refs = solid_refs_from_dicts(entry.params.get("child_solid_refs"))
            assigned = assign_solids_to_children(
                result, child_body_ids, refs, history._workspace,
                history, entry_index)
            if assigned[0] is None:
                raise RuntimeError(
                    "SketchExtrudeOp: could not assign a solid to the primary "
                    "child body")
            return assigned[0]

        solids = list(result.solids())
        solids.sort(key=lambda s: s.volume, reverse=True)
        if not solids:
            raise RuntimeError("SketchExtrudeOp: result contains no solids")

        this_id = entry.entry_id
        for j in range(entry_index + 1, len(history._entries)):
            e = history._entries[j]
            if (e.operation == "import" and
                    e.params.get("source_entry_id") == this_id):
                solid_idx = e.params.get("solid_index", 1)
                e.shape_after = solids[solid_idx] if solid_idx < len(solids) else None

        return solids[0]

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
        from cad.operations.extrude import extrude_face_direct, _do_extrude_solid
        from build123d import Compound
        from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
        from OCP.TopTools import TopTools_ListOfShape
        from cad.face_ref import FaceRef

        entries    = viewport.history.entries
        sketch_idx = viewport.history.id_to_index(self.from_sketch_id)
        if sketch_idx is None:
            raise RuntimeError(f"[Extrude] Sketch entry '{self.from_sketch_id}' not found.")
        se = entries[sketch_idx].params.get("sketch_entry")
        if se is None:
            raise RuntimeError("[Extrude] No sketch entry.")
        all_faces = viewport._sketch_faces.get(sketch_idx, [])
        if not all_faces:
            raise RuntimeError("[Extrude] Sketch has no closed loops.")

        fidx = viewport._selected_sketch_face
        if fidx is not None:
            face_indices = [i for i in fidx if 0 <= i < len(all_faces)]
        else:
            face_indices = list(range(len(all_faces)))
        faces = [all_faces[i][0] for i in face_indices]

        def _uv_centroid(outer_uvs):
            arr = np.array(outer_uvs)
            return arr.mean(axis=0).tolist()
        face_centroids = [_uv_centroid(all_faces[i][1]) for i in face_indices]

        body_id      = se.body_id
        shape_before = viewport.workspace.current_shape(body_id)
        direction    = (np.array(self.direction, dtype=float)
                        if self.direction is not None else None)
        op_str    = "cut" if self.distance < 0 else "extrude"
        op_params = {"distance": abs(self.distance),
                     "from_sketch_id": self.from_sketch_id,
                     "face_indices": face_indices,
                     "face_centroids": face_centroids}
        if self.force_new_body:
            op_params["force_new_body"] = True
        if direction is not None:
            op_params["direction"] = direction.tolist()
        # Transient hint from the edit path; consumed below, never stored.
        preserved_body_ids = list((extra_params or {}).get("_preserved_body_ids", []))
        if extra_params:
            op_params.update(extra_params)
        op_params.pop("_preserved_body_ids", None)
        mesh     = viewport._meshes.get(body_id)
        face_ref = (FaceRef.from_b3d_face(mesh.occt_faces[se.face_idx]) if mesh else None)
        distance     = self.distance
        start_offset = self.start_offset
        end_offset   = self.end_offset
        draft_angle  = self.draft_angle

        if self.merge_body_id is not None and not self.force_new_body:
            target_shape = viewport.workspace.current_shape(self.merge_body_id)
            if target_shape is None:
                raise RuntimeError("[Extrude] Merge target has no shape.")
            target_occ = target_shape.wrapped
            original_solid_count = len(list(target_shape.solids()))
            # Store merge_body_id (the target) so replay can reproduce the
            # fuse. merged_from (the source) is kept for the reopen/edit UI.
            merge_params = {"distance": abs(distance), "merged_from": body_id,
                            "merge_body_id": self.merge_body_id,
                            "from_sketch_id": self.from_sketch_id,
                            "face_indices": face_indices,
                            "face_centroids": face_centroids}
            if direction is not None:
                merge_params["direction"] = direction.tolist()
            if extra_params:
                merge_params.update(extra_params)
            merge_body_id = self.merge_body_id

            def compute():
                tool_solid = _extrude_faces_to_tool(
                    faces, distance, direction,
                    start_offset, end_offset, draft_angle)
                if tool_solid is None:
                    raise RuntimeError("Extrude produced no tool.")
                return _fuse_tool_into_target(tool_solid, target_occ)

            def finalize(merged):
                _push_result(viewport, "extrude", merge_params, merge_body_id,
                             None, target_shape, merged, original_solid_count)
                viewport._selected_sketch_entry = None
                viewport._selected_sketch_face  = None

            return compute, finalize

        original_solid_count = len(list(shape_before.solids())) if shape_before else 0
        force_new = self.force_new_body

        def compute():
            result = None if force_new else shape_before
            for face in faces:
                result = extrude_face_direct(result, face, distance, direction=direction,
                                             start_offset=start_offset, end_offset=end_offset,
                                             draft_angle=draft_angle)
            if not force_new and not list(result.solids()):
                raise RuntimeError("Sketch extrude produced no solids.")
            return result

        def finalize(shape_after):
            if force_new:
                from viewer.vp_extrude import _strip_split_suffix, _next_split_name
                from cad.units import format_op_label as _lbl
                from cad.solid_ref import solid_refs_to_dicts
                root_name  = _strip_split_suffix(viewport.workspace.bodies[body_id].name)
                solids     = list(shape_after.solids())
                op_params["child_body_ids"] = []
                # Geometric fingerprint per child, aligned with child_body_ids,
                # so replay re-locates each segment by shape instead of by
                # OCCT's (non-reproducible) solid-iteration index.
                op_params["child_solid_refs"] = solid_refs_to_dicts(solids)
                new_bodies = []
                for i, solid in enumerate(solids):
                    new_name = _next_split_name(root_name, viewport.workspace)
                    # Reuse the original body_id if the edit path supplied one
                    # for this solid index — keeps downstream ops pointing at
                    # the same body across parametric edits.
                    preserved = (preserved_body_ids[i]
                                 if i < len(preserved_body_ids) else None)
                    new_body = viewport.workspace.add_body(
                        new_name, Compound(solid.wrapped), body_id=preserved)
                    new_bodies.append(new_body)
                    op_params["child_body_ids"].append(new_body.id)
                    print(f"[Extrude] New body '{new_name}' ({len(list(solid.faces()))} faces)")
                primary_body = new_bodies[0] if new_bodies else None
                tag_body_id  = primary_body.id if primary_body else body_id
                parent_entry = viewport.history.push(
                    label=_lbl(op_str, op_params), operation=op_str,
                    params=op_params, body_id=tag_body_id, face_ref=face_ref,
                    shape_before=None, shape_after=shape_after)
                for new_body in new_bodies:
                    new_body.created_at_entry_id = parent_entry.entry_id
                viewport._rebuild_bodies({b.id for b in new_bodies})
                viewport.history_changed.emit()
            else:
                _push_result(viewport, op_str, op_params, body_id,
                             face_ref, shape_before, shape_after, original_solid_count)
            viewport._selected_sketch_entry = None
            viewport._selected_sketch_face  = None

        return compute, finalize

    def reopen(self, viewport: Any, history_idx: int) -> None:
        import numpy as np
        entry      = viewport.history.entries[history_idx]
        vp         = viewport
        sketch_idx = vp.history.id_to_index(self.from_sketch_id)

        if history_idx > 0:
            vp.history.seek(history_idx - 1)
            vp._rebuild_all_meshes()
        vp.history_changed.emit()

        vp._selected_sketch_entry = sketch_idx
        face_indices = self.face_indices
        vp._selected_sketch_face  = list(face_indices) if face_indices else None

        vp._show_extrude_panel(
            sketch_idx    = sketch_idx,
            body_id       = None,
            face_idx      = None,
            editing_entry = entry)

        panel = vp._extrude_panel
        if panel is None:
            return

        # Replace generic "Sketch N" entry with one row per stored face index.
        if face_indices:
            panel.clear_face_entries()
            for i in face_indices:
                panel.add_face_entry(None, None, f"Sketch {sketch_idx}  ·  face {i}")

        if self.distance < 0:
            panel._radio_cut.setChecked(True)
            panel._on_mode_changed(1)

        # Restore the picked merge TARGET. merge_body_id is the body the user
        # chose to merge into; merged_from is only a fallback for legacy entries
        # saved before merge_body_id was persisted (it held the sketch's parent
        # body, which is the target only when they happen to coincide).
        merge_target = (self.merge_body_id
                        if self.merge_body_id in vp.workspace.bodies
                        else self.merged_from)
        if merge_target and merge_target in vp.workspace.bodies:
            panel.set_merge_body(
                merge_target,
                vp.workspace.bodies[merge_target].name)
            panel._radio_merge.setChecked(True)
            panel._on_op_changed(1)

        if self.direction is not None:
            panel.set_direction(np.array(self.direction, dtype=float))

        if self.start_offset:
            panel._start_offset.set_mm(self.start_offset)
        if self.end_offset:
            panel._end_offset.set_mm(self.end_offset)

        if self.draft_angle:
            panel.set_draft(self.draft_angle)

        if self.target_vertex is not None:
            panel._radio_vertex.setChecked(True)
            panel._on_target_mode_changed(1)
            panel.set_vertex_target(np.array(self.target_vertex, dtype=float))
        else:
            panel._spinbox.set_mm(abs(self.distance))

        panel._emit_preview()

    def to_params(self) -> dict:
        p: dict[str, Any] = {
            "distance":       abs(self.distance),
            "from_sketch_id": self.from_sketch_id,
        }
        if self.merge_body_id is not None:
            p["merge_body_id"] = self.merge_body_id
        if self.merged_from is not None:
            p["merged_from"] = self.merged_from
        if self.face_indices is not None:
            p["face_indices"] = self.face_indices
        if self.face_centroids is not None:
            p["face_centroids"] = self.face_centroids
        if self.direction is not None:
            p["direction"] = self.direction
        if self.target_vertex is not None:
            p["target_vertex"] = self.target_vertex
        if self.start_offset:
            p["start_offset"] = self.start_offset
        if self.end_offset:
            p["end_offset"] = self.end_offset
        if self.draft_angle:
            p["draft_angle"] = self.draft_angle
        if self.force_new_body:
            p["force_new_body"] = True
        return p

    @classmethod
    def _from_params(cls, params: dict, sign: int = 1) -> "SketchExtrudeOp":
        dist = float(params.get("distance", 0)) * sign
        return cls(
            from_sketch_id = params["from_sketch_id"],
            distance       = dist,
            merge_body_id  = params.get("merge_body_id"),
            face_indices   = params.get("face_indices"),
            face_centroids = params.get("face_centroids"),
            direction      = params.get("direction"),
            target_vertex  = params.get("target_vertex"),
            start_offset   = float(params.get("start_offset", 0.0)),
            end_offset     = float(params.get("end_offset", 0.0)),
            draft_angle    = float(params.get("draft_angle", 0.0)),
            force_new_body = bool(params.get("force_new_body", False)),
            merged_from    = params.get("merged_from"),
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
        if se is not None and se.plane_source is not None:
            ok, err = _replay_sketch_entry(se, history, before_index=entry_index)
            if not ok:
                raise RuntimeError(err)
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
