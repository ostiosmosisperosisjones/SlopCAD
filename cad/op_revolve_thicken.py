"""
cad/op_revolve_thicken.py

Revolve and thicken operation types:
  ThickenOp, FaceRevolveOp, SketchRevolveOp, CrossBodyRevolveCutOp
"""

from __future__ import annotations
from dataclasses import dataclass
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

    def _resolve_faces(self, shape):
        """Return list of (idx, b3d_face) using face_refs when available."""
        from cad.face_ref import AnyFaceRef
        all_faces = list(shape.faces())
        resolved = []
        if self.face_refs:
            for ref in self.face_refs:
                idx, face = ref.find_in(shape)
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
        return resolved

    def execute(self, shape: Any, history: "History", entry_index: int) -> Any:
        from cad.operations.thicken import thicken_face
        if shape is None:
            src = history._shape_for_body_at(self.source_body_id, entry_index)
            if src is None:
                raise RuntimeError(
                    f"ThickenOp: no shape for source body '{self.source_body_id}'")
            shape = src
        resolved = self._resolve_faces(shape)
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

        self.face_refs = [
            AnyFaceRef.from_occ_face(all_faces[idx].wrapped)
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
        vp._show_thicken_panel(self.source_body_id, self.face_indices, editing_entry=entry)
        panel = getattr(vp, '_thicken_panel', None)
        if panel is not None:
            panel.set_thickness(self.thickness)
            panel._emit_preview()

    def to_params(self) -> dict:
        p: dict = {
            "source_body_id": self.source_body_id,
            "face_indices":   self.face_indices,
            "thickness":      self.thickness,
        }
        if self.face_refs:
            p["face_refs"] = [
                {"centroid": list(r.centroid),
                 "area":     r.area,
                 "category": r.category}
                for r in self.face_refs
            ]
        return p

    @classmethod
    def _from_params(cls, params: dict, sign: int = 1) -> "ThickenOp":
        from cad.face_ref import AnyFaceRef
        if "face_indices" in params:
            indices = list(params["face_indices"])
        else:
            indices = [int(params.get("face_idx", 0))]
        raw_refs = params.get("face_refs", [])
        face_refs = [
            AnyFaceRef(centroid=tuple(r["centroid"]),
                       area=r["area"],
                       category=r.get("category", ""))
            for r in raw_refs
        ]
        return cls(
            source_body_id = params.get("source_body_id", ""),
            face_indices   = indices,
            thickness      = float(params.get("thickness", 0)),
            face_refs      = face_refs,
        )


# ---------------------------------------------------------------------------
# FaceRevolveOp  —  revolve a body face around an axis (new body result)
# ---------------------------------------------------------------------------

@dataclass
class FaceRevolveOp(Op):
    """
    Revolve a planar face on a body around a world-space axis.

    source_body_id : body whose face is revolved
    face_idx       : face index at commit time (face_ref used for replay)
    angle_deg      : degrees of revolution
    axis_point     : world-space point on the rotation axis
    axis_dir       : unit direction of the rotation axis
    face_ref       : FaceRef fingerprint for stable re-resolution during replay
    force_new_body : always True — revolve produces a new standalone solid
    """
    source_body_id : str
    face_idx       : int
    angle_deg      : float
    axis_point     : list[float]
    axis_dir       : list[float]
    face_ref       : Any = None
    force_new_body : bool = True

    def execute(self, shape: Any, history: "History", entry_index: int) -> Any:
        import numpy as np
        from cad.operations.revolve import revolve_face_direct
        from build123d import Compound

        src_shape = history._shape_for_body_at(self.source_body_id, entry_index)
        if src_shape is None:
            raise RuntimeError(
                f"FaceRevolveOp: no shape for body '{self.source_body_id}'")

        if self.face_ref is not None:
            _, face_obj = self.face_ref.find_in(src_shape)
            if face_obj is None:
                raise RuntimeError("FaceRevolveOp: could not re-locate face")
        else:
            all_f = list(src_shape.faces())
            if self.face_idx >= len(all_f):
                raise RuntimeError(f"FaceRevolveOp: face_idx {self.face_idx} out of range")
            face_obj = all_f[self.face_idx]

        axis_pt  = np.array(self.axis_point, dtype=float)
        axis_dir = np.array(self.axis_dir,   dtype=float)
        result = revolve_face_direct(None, face_obj, axis_pt, axis_dir, self.angle_deg)

        entry = history._entries[entry_index]
        child_body_ids = entry.params.get("child_body_ids", [])
        if child_body_ids and history._workspace is not None:
            from cad.solid_ref import (assign_solids_to_children,
                                       solid_refs_from_dicts)
            refs = solid_refs_from_dicts(entry.params.get("child_solid_refs"))
            assign_solids_to_children(result, child_body_ids, refs,
                                      history._workspace, history, entry_index)
        return shape

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
        from cad.operations.revolve import revolve_face_direct
        from cad.face_ref import FaceRef
        from build123d import Compound
        from cad.units import format_op_label as _lbl

        body_id     = self.source_body_id
        shape       = viewport.workspace.current_shape(body_id)
        if shape is None:
            raise RuntimeError(f"[Revolve] Body '{body_id}' has no shape.")

        all_faces = list(shape.faces())
        if self.face_idx >= len(all_faces):
            raise RuntimeError(f"[Revolve] face_idx {self.face_idx} out of range.")
        face_obj = all_faces[self.face_idx]

        mesh     = viewport._meshes.get(body_id)
        face_ref = FaceRef.from_b3d_face(mesh.occt_faces[self.face_idx]) if mesh else None
        self.face_ref = face_ref

        axis_pt  = np.array(self.axis_point, dtype=float)
        axis_dir = np.array(self.axis_dir,   dtype=float)
        angle    = self.angle_deg

        op_params = self.to_params()
        if extra_params:
            op_params.update(extra_params)

        def compute():
            return revolve_face_direct(None, face_obj, axis_pt, axis_dir, angle)

        def finalize(shape_after):
            if shape_after is None:
                err = getattr(viewport, '_pending_op_error', None) or "Revolve failed"
                entry = viewport.history.push(
                    label=_lbl("revolve", op_params), operation="revolve",
                    params=op_params, body_id=body_id, face_ref=face_ref,
                    shape_before=shape, shape_after=None)
                entry.error = True; entry.error_msg = err
                viewport._post_push_cascade(body_id)
                viewport.history_changed.emit()
                return

            from cad.solid_ref import solid_refs_to_dicts
            solids = list(shape_after.solids())
            new_bodies = []
            for solid in solids:
                new_name = viewport.workspace.next_part_name()
                new_body = viewport.workspace.add_body(
                    new_name, Compound(solid.wrapped))
                new_bodies.append(new_body)
            op_params["child_body_ids"] = [b.id for b in new_bodies]
            # Per-child geometric fingerprint, aligned with child_body_ids.
            op_params["child_solid_refs"] = solid_refs_to_dicts(solids)

            tag_body = new_bodies[0] if new_bodies else None
            tag_id   = tag_body.id if tag_body else body_id
            parent_entry = viewport.history.push(
                label=_lbl("revolve", op_params), operation="revolve",
                params=op_params, body_id=tag_id, face_ref=face_ref,
                shape_before=shape, shape_after=shape_after)
            for nb in new_bodies:
                nb.created_at_entry_id = parent_entry.entry_id
            viewport._rebuild_bodies({b.id for b in new_bodies})
            viewport.history_changed.emit()

        return compute, finalize

    def reopen(self, viewport: Any, history_idx: int) -> None:
        import numpy as np
        entry = viewport.history.entries[history_idx]
        if history_idx > 0:
            viewport.history.seek(history_idx - 1)
            viewport._rebuild_body_mesh(self.source_body_id)
        viewport.history_changed.emit()
        viewport._show_revolve_panel(
            body_id=self.source_body_id, face_idx=self.face_idx,
            editing_entry=entry)
        panel = getattr(viewport, '_revolve_panel', None)
        if panel is None:
            return
        panel.set_axis(
            np.array(self.axis_point, dtype=float),
            np.array(self.axis_dir,   dtype=float))
        panel._angle_spinbox.set_mm(self.angle_deg)
        panel._emit_preview()

    def to_params(self) -> dict:
        p: dict[str, Any] = {
            "angle_deg":      self.angle_deg,
            "source_body_id": self.source_body_id,
            "face_idx":       self.face_idx,
            "axis_point":     self.axis_point,
            "axis_dir":       self.axis_dir,
        }
        if self.face_ref is not None:
            r = self.face_ref
            p["face_ref"] = {"normal": list(r.normal), "area": r.area,
                             "centroid_perp": list(r.centroid_perp),
                             "centroid_along": r.centroid_along}
        return p

    @classmethod
    def _from_params(cls, params: dict) -> "FaceRevolveOp":
        face_ref = None
        if "face_ref" in params:
            from cad.face_ref import FaceRef
            r = params["face_ref"]
            face_ref = FaceRef(normal=tuple(r["normal"]), area=r["area"],
                               centroid_perp=tuple(r["centroid_perp"]),
                               centroid_along=r["centroid_along"])
        return cls(
            source_body_id = params["source_body_id"],
            face_idx       = int(params.get("face_idx", 0)),
            angle_deg      = float(params.get("angle_deg", 360.0)),
            axis_point     = params["axis_point"],
            axis_dir       = params["axis_dir"],
            face_ref       = face_ref,
        )


# ---------------------------------------------------------------------------
# SketchRevolveOp  —  revolve a sketch profile around an axis
# ---------------------------------------------------------------------------

def _revolve_faces_to_tool(faces, axis_pt, axis_dir, angle_deg):
    """Revolve each face and fuse the results into one tool solid (build123d
    Compound). Shared by commit and replay so the two can't drift."""
    from cad.operations.revolve import revolve_face_direct
    from build123d import Compound
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
    from OCP.TopTools import TopTools_ListOfShape

    tool = None
    for face in faces:
        s = revolve_face_direct(None, face, axis_pt, axis_dir, angle_deg)
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
    """Fuse a revolved tool solid into a target body's shape. Returns a
    build123d Compound. Shared by commit and replay."""
    from build123d import Compound
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
    from OCP.TopTools import TopTools_ListOfShape

    lst_a = TopTools_ListOfShape(); lst_a.Append(target_occ)
    lst_b = TopTools_ListOfShape(); lst_b.Append(tool.wrapped)
    fu = BRepAlgoAPI_Fuse()
    fu.SetArguments(lst_a); fu.SetTools(lst_b)
    fu.SetRunParallel(True); fu.Build()
    if not fu.IsDone():
        raise RuntimeError("Revolve fuse failed.")
    return Compound(fu.Shape())


@dataclass
class SketchRevolveOp(Op):
    """
    Revolve a closed-loop profile built from a SketchEntry around an axis.

    from_sketch_id : entry_id of the sketch entry
    angle_deg      : revolution arc in degrees (positive = CCW from axis_dir tip)
    axis_point     : world-space point on the rotation axis
    axis_dir       : unit direction of the rotation axis
    merge_body_id  : body to fuse into, or None for a new body
    face_indices   : which faces of the sketch to revolve (None = all)
    face_centroids : UV centroids for stable face matching across sketch edits
    """
    from_sketch_id : str
    angle_deg      : float
    axis_point     : list[float]
    axis_dir       : list[float]
    merge_body_id  : str | None       = None
    face_indices   : list[int] | None = None
    face_centroids : list[list[float]] | None = None
    force_new_body : bool = True   # revolves always produce a standalone solid

    def execute(self, shape: Any, history: "History", entry_index: int) -> Any:
        import numpy as np
        from cad.operations.revolve import revolve_face_direct
        from build123d import Compound

        sketch_idx = history.id_to_index(self.from_sketch_id)
        if sketch_idx is None:
            raise RuntimeError(
                f"SketchRevolveOp: sketch entry '{self.from_sketch_id}' not found")
        if sketch_idx >= entry_index:
            raise RuntimeError(
                "SketchRevolveOp: sketch entry is after this entry — invalid reorder")

        sketch_entry_rec = history._entries[sketch_idx]
        if sketch_entry_rec.error:
            raise RuntimeError("SketchRevolveOp: sketch entry is in an error state")

        se = sketch_entry_rec.params.get("sketch_entry")
        if se is None:
            raise RuntimeError(
                f"SketchRevolveOp: no sketch_entry at id '{self.from_sketch_id}'")

        if se.plane_source is not None:
            from cad.history import reproject_consumed_sketch
            # Reproject at this op's position (not the sketch's) so the sketch
            # follows intervening ops on its parent body; fall back to the
            # authored plane if the anchor face was destroyed.
            ok, err = reproject_consumed_sketch(se, history, entry_index,
                                                sketch_entry_rec)
            if not ok:
                raise RuntimeError(
                    f"SketchRevolveOp: sketch reprojection failed: {err}")

        all_faces, all_regions = se.build_faces()
        if not all_faces:
            raise RuntimeError("SketchRevolveOp: sketch has no closed loops")

        if self.face_centroids:
            from cad.op_base import _match_face_by_centroid
            centroids = np.array(self.face_centroids, dtype=float)
            faces = [_match_face_by_centroid(t, all_faces, all_regions,
                                             "SketchRevolveOp")
                     for t in centroids]
        elif self.face_indices is not None:
            faces = [all_faces[i] for i in self.face_indices if i < len(all_faces)]
            if not faces:
                raise RuntimeError("SketchRevolveOp: stored face indices out of range")
        else:
            faces = all_faces

        axis_pt  = np.array(self.axis_point, dtype=float)
        axis_dir = np.array(self.axis_dir,   dtype=float)

        # Merge-into-target case: the entry is tagged to merge_body_id, so the
        # `shape` passed in by replay IS that target body's current shape. We
        # must reproduce the same fuse the commit did (build the tool, fuse it
        # into the target) — otherwise replay would replace the target body
        # with the bare revolve and silently drop the merge.
        force_new = self.merge_body_id is None or self.merge_body_id == "__new_body__"
        if not force_new:
            if shape is None:
                raise RuntimeError(
                    "SketchRevolveOp: merge target has no shape on replay")
            tool = _revolve_faces_to_tool(faces, axis_pt, axis_dir, self.angle_deg)
            if tool is None:
                raise RuntimeError("SketchRevolveOp: revolve produced no tool")
            return _fuse_tool_into_target(tool, shape.wrapped)

        result = None
        for face in faces:
            result = revolve_face_direct(result, face, axis_pt, axis_dir, self.angle_deg)

        entry = history._entries[entry_index]
        child_body_ids = entry.params.get("child_body_ids", [])

        # force_new_body split mirror — see SketchExtrudeOp.execute. Without it
        # reload leaves non-primary children shapeless; SolidRef keeps the
        # child→segment mapping stable across OCCT solid-ordering churn.
        if self.force_new_body and child_body_ids:
            from cad.solid_ref import assign_solids_to_children, solid_refs_from_dicts
            if not list(result.solids()):
                raise RuntimeError("SketchRevolveOp: result contains no solids")
            refs = solid_refs_from_dicts(entry.params.get("child_solid_refs"))
            assigned = assign_solids_to_children(
                result, child_body_ids, refs, history._workspace,
                history, entry_index)
            if assigned[0] is None:
                raise RuntimeError(
                    "SketchRevolveOp: could not assign a solid to the primary "
                    "child body")
            return assigned[0]

        solids = list(result.solids())
        solids.sort(key=lambda s: s.volume, reverse=True)
        if not solids:
            raise RuntimeError("SketchRevolveOp: result contains no solids")

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
        from cad.operations.revolve import revolve_face_direct
        from build123d import Compound
        from cad.face_ref import FaceRef

        entries    = viewport.history.entries
        sketch_idx = viewport.history.id_to_index(self.from_sketch_id)
        if sketch_idx is None:
            raise RuntimeError(f"[Revolve] Sketch entry '{self.from_sketch_id}' not found.")
        se = entries[sketch_idx].params.get("sketch_entry")
        if se is None:
            raise RuntimeError("[Revolve] No sketch entry.")
        all_faces = viewport._sketch_faces.get(sketch_idx, [])
        if not all_faces:
            raise RuntimeError("[Revolve] Sketch has no closed loops.")

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
        axis_pt      = np.array(self.axis_point, dtype=float)
        axis_dir     = np.array(self.axis_dir,   dtype=float)
        angle_deg    = self.angle_deg

        op_params = {
            "angle_deg":      angle_deg,
            "from_sketch_id": self.from_sketch_id,
            "face_indices":   face_indices,
            "face_centroids": face_centroids,
            "axis_point":     self.axis_point,
            "axis_dir":       self.axis_dir,
        }
        if extra_params:
            op_params.update(extra_params)

        mesh     = viewport._meshes.get(body_id)
        face_ref = (FaceRef.from_b3d_face(mesh.occt_faces[se.face_idx]) if mesh else None)

        original_solid_count = len(list(shape_before.solids())) if shape_before else 0
        force_new = self.merge_body_id is None or self.merge_body_id == "__new_body__"

        if not force_new:
            target_shape = viewport.workspace.current_shape(self.merge_body_id)
            if target_shape is None:
                raise RuntimeError("[Revolve] Merge target has no shape.")
            target_occ      = target_shape.wrapped
            merge_body_id   = self.merge_body_id
            orig_merge_count = len(list(target_shape.solids()))
            merge_params = dict(op_params)
            merge_params["merge_body_id"] = merge_body_id

            def compute():
                tool = _revolve_faces_to_tool(faces, axis_pt, axis_dir, angle_deg)
                if tool is None:
                    raise RuntimeError("Revolve produced no tool.")
                return _fuse_tool_into_target(tool, target_occ)

            def finalize(merged):
                _push_result(viewport, "revolve", merge_params, merge_body_id,
                             None, target_shape, merged, orig_merge_count)
                viewport._selected_sketch_entry = None
                viewport._selected_sketch_face  = None

            return compute, finalize

        def compute():
            result = None if force_new else shape_before
            for face in faces:
                result = revolve_face_direct(result, face, axis_pt, axis_dir, angle_deg)
            return result

        def finalize(shape_after):
            if force_new:
                from cad.units import format_op_label as _lbl
                from cad.solid_ref import solid_refs_to_dicts
                solids    = list(shape_after.solids())
                op_params["child_body_ids"] = []
                op_params["child_solid_refs"] = solid_refs_to_dicts(solids)
                new_bodies = []
                for solid in solids:
                    new_name = viewport.workspace.next_part_name()
                    new_body = viewport.workspace.add_body(new_name, Compound(solid.wrapped))
                    new_bodies.append(new_body)
                    op_params["child_body_ids"].append(new_body.id)
                primary_body = new_bodies[0] if new_bodies else None
                tag_body_id  = primary_body.id if primary_body else body_id
                parent_entry = viewport.history.push(
                    label=_lbl("revolve", op_params), operation="revolve",
                    params=op_params, body_id=tag_body_id, face_ref=face_ref,
                    shape_before=None, shape_after=shape_after)
                for nb in new_bodies:
                    nb.created_at_entry_id = parent_entry.entry_id
                viewport._rebuild_bodies({b.id for b in new_bodies})
                viewport.history_changed.emit()
            else:
                _push_result(viewport, "revolve", op_params, body_id,
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

        vp._show_revolve_panel(sketch_idx=sketch_idx, editing_entry=entry)

        panel = vp._revolve_panel
        if panel is None:
            return

        panel.set_axis(
            np.array(self.axis_point, dtype=float),
            np.array(self.axis_dir,   dtype=float))
        panel._angle_spinbox.set_mm(abs(self.angle_deg))

        if self.merge_body_id and self.merge_body_id in vp.workspace.bodies:
            panel.set_merge_body(
                self.merge_body_id,
                vp.workspace.bodies[self.merge_body_id].name)
            panel._radio_merge.setChecked(True)
            panel._on_op_changed(1)

        panel._emit_preview()

    def to_params(self) -> dict:
        p: dict[str, Any] = {
            "angle_deg":      self.angle_deg,
            "from_sketch_id": self.from_sketch_id,
            "axis_point":     self.axis_point,
            "axis_dir":       self.axis_dir,
        }
        if self.merge_body_id is not None:
            p["merge_body_id"] = self.merge_body_id
        if self.face_indices is not None:
            p["face_indices"] = self.face_indices
        if self.face_centroids is not None:
            p["face_centroids"] = self.face_centroids
        return p

    @classmethod
    def _from_params(cls, params: dict) -> "SketchRevolveOp":
        return cls(
            from_sketch_id = params["from_sketch_id"],
            angle_deg      = float(params.get("angle_deg", 360.0)),
            axis_point     = params["axis_point"],
            axis_dir       = params["axis_dir"],
            merge_body_id  = params.get("merge_body_id"),
            face_indices   = params.get("face_indices"),
            face_centroids = params.get("face_centroids"),
        )


# ---------------------------------------------------------------------------
# CrossBodyRevolveCutOp  —  revolve a sketch profile and cut another body
# ---------------------------------------------------------------------------

@dataclass
class CrossBodyRevolveCutOp(Op):
    """
    Revolve a sketch profile around an axis to form a tool solid, then
    BRepAlgoAPI_Cut it from *cut_body_id*.  Mirrors CrossBodyCutOp for the
    revolve operation.

    cut_body_id      : body being cut
    source_sketch_id : entry_id of the sketch entry whose profile is revolved
                       (sketch-driven cuts only — face-driven revolve cuts go
                       through this same op with source_sketch_id=None and a
                       source_body_id+source_face_idx pair, for symmetry with
                       CrossBodyCutOp; unimplemented for now)
    source_body_id   : sketch's parent body (informational; not used for cut)
    angle_deg        : revolution arc in degrees
    axis_point       : world-space point on the rotation axis
    axis_dir         : unit direction of the rotation axis
    """
    cut_body_id      : str
    source_body_id   : str  | None       = None
    source_sketch_id : str  | None       = None
    angle_deg        : float = 360.0
    axis_point       : list[float] | None = None
    axis_dir         : list[float] | None = None
    face_indices     : list[int]   | None = None
    face_centroids   : list[list[float]] | None = None

    # Used by Op._push_failed_entry to tag the failed history entry correctly
    # (the base class's fallback otherwise produces "crossbodyrevolvecut").
    _op_str = "revolve_cut"

    def execute(self, shape: Any, history: "History", entry_index: int) -> Any:
        import numpy as np
        from cad.operations.revolve import _do_revolve_solid
        from build123d import Compound
        from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
        from OCP.TopTools import TopTools_ListOfShape

        if self.source_sketch_id is None:
            raise RuntimeError("CrossBodyRevolveCutOp: no source_sketch_id")
        sketch_idx = history.id_to_index(self.source_sketch_id)
        if sketch_idx is None:
            raise RuntimeError(
                f"CrossBodyRevolveCutOp: sketch entry '{self.source_sketch_id}' not found")
        sketch_rec = history._entries[sketch_idx]
        if sketch_rec.error:
            raise RuntimeError(
                "CrossBodyRevolveCutOp: source sketch entry is in an error state")
        se = sketch_rec.params.get("sketch_entry")
        if se is None:
            raise RuntimeError(
                f"CrossBodyRevolveCutOp: no sketch_entry at id '{self.source_sketch_id}'")

        # Reproject the sketch's plane at *this* op's history position so
        # intervening edits on the parent face move the profile with it; fall
        # back to the sketch's authored plane if an intervening op destroyed the
        # anchor face (see reproject_consumed_sketch).
        if se.plane_source is not None:
            from cad.history import reproject_consumed_sketch
            ok, err = reproject_consumed_sketch(se, history, entry_index,
                                                sketch_rec)
            if not ok:
                raise RuntimeError(
                    f"CrossBodyRevolveCutOp: sketch reprojection failed: {err}")

        all_faces, all_regions = se.build_faces()
        if not all_faces:
            raise RuntimeError("CrossBodyRevolveCutOp: sketch has no closed loops")

        if self.face_centroids:
            from cad.op_base import _match_face_by_centroid
            centroids = np.array(self.face_centroids, dtype=float)
            faces = [_match_face_by_centroid(t, all_faces, all_regions,
                                             "CrossBodyRevolveCutOp")
                     for t in centroids]
        elif self.face_indices is not None:
            faces = [all_faces[i] for i in self.face_indices if i < len(all_faces)]
            if not faces:
                raise RuntimeError("CrossBodyRevolveCutOp: stored face indices out of range")
        else:
            faces = all_faces

        axis_pt  = np.array(self.axis_point, dtype=float)
        axis_dir = np.array(self.axis_dir,   dtype=float)

        tool = None
        for face in faces:
            s = _do_revolve_solid(face, axis_pt, axis_dir, self.angle_deg)
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

        lst_a = TopTools_ListOfShape(); lst_a.Append(shape.wrapped)
        lst_b = TopTools_ListOfShape(); lst_b.Append(tool.wrapped)
        op = BRepAlgoAPI_Cut()
        op.SetArguments(lst_a); op.SetTools(lst_b)
        op.SetRunParallel(True); op.Build()
        if not op.IsDone():
            raise RuntimeError("CrossBodyRevolveCutOp: boolean cut failed")
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
        from cad.operations.revolve import _do_revolve_solid
        from build123d import Compound
        from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
        from OCP.TopTools import TopTools_ListOfShape

        if self.source_sketch_id is None:
            raise RuntimeError("[Revolve cut] No source_sketch_id.")
        entries = viewport.history.entries
        sketch_idx = viewport.history.id_to_index(self.source_sketch_id)
        if sketch_idx is None:
            raise RuntimeError(
                f"[Revolve cut] Sketch entry '{self.source_sketch_id}' not found.")
        se = entries[sketch_idx].params.get("sketch_entry")
        if se is None:
            raise RuntimeError("[Revolve cut] No sketch_entry.")
        all_sketch = viewport._sketch_faces.get(sketch_idx, [])
        if not all_sketch:
            raise RuntimeError("[Revolve cut] Sketch has no faces.")

        # Prefer indices already baked into the op (fan-out / replay) over the
        # mutable viewport selection, which the first finalize clears.
        if self.face_indices is not None:
            face_indices = [i for i in self.face_indices
                            if 0 <= i < len(all_sketch)]
        elif viewport._selected_sketch_face is not None:
            face_indices = [i for i in viewport._selected_sketch_face
                            if 0 <= i < len(all_sketch)]
        else:
            face_indices = list(range(len(all_sketch)))
        faces = [all_sketch[i][0] for i in face_indices]

        def _uv_centroid(outer_uvs):
            arr = np.array(outer_uvs)
            return arr.mean(axis=0).tolist()
        face_centroids = [_uv_centroid(all_sketch[i][1]) for i in face_indices]

        target_shape = viewport.workspace.current_shape(self.cut_body_id)
        if target_shape is None:
            raise RuntimeError("[Revolve cut] Target body has no shape.")
        original_solid_count = len(list(target_shape.solids()))
        target_occ = target_shape.wrapped

        axis_pt  = np.array(self.axis_point, dtype=float)
        axis_dir = np.array(self.axis_dir,   dtype=float)
        angle    = self.angle_deg

        op_params = self.to_params()
        op_params["face_indices"]   = face_indices
        op_params["face_centroids"] = face_centroids
        if extra_params:
            op_params.update(extra_params)

        def compute():
            tool = None
            for face in faces:
                s = _do_revolve_solid(face, axis_pt, axis_dir, angle)
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
            lst_a = TopTools_ListOfShape(); lst_a.Append(target_occ)
            lst_b = TopTools_ListOfShape(); lst_b.Append(tool.wrapped)
            op = BRepAlgoAPI_Cut()
            op.SetArguments(lst_a); op.SetTools(lst_b)
            op.SetRunParallel(True); op.Build()
            if not op.IsDone():
                raise RuntimeError("Revolve cut failed.")
            return Compound(op.Shape())

        def finalize(result):
            _push_result(viewport, "revolve_cut", op_params, self.cut_body_id,
                         None, target_shape, result, original_solid_count,
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
            vp._rebuild_all_meshes()
        vp.history_changed.emit()

        if sketch_idx is not None:
            vp._selected_sketch_entry = sketch_idx
            vp._selected_sketch_face  = (list(self.face_indices)
                                          if self.face_indices else None)

        vp._show_revolve_panel(sketch_idx=sketch_idx, editing_entry=entry)

        panel = vp._revolve_panel
        if panel is None:
            return

        if self.axis_point is not None and self.axis_dir is not None:
            panel.set_axis(
                np.array(self.axis_point, dtype=float),
                np.array(self.axis_dir,   dtype=float))
        panel._angle_spinbox.set_mm(abs(self.angle_deg))

        panel._radio_cut.setChecked(True)
        panel._on_mode_changed(1)

        if self.cut_body_id in vp.workspace.bodies:
            panel.set_merge_body(
                self.cut_body_id,
                vp.workspace.bodies[self.cut_body_id].name)

        panel._emit_preview()

    def to_params(self) -> dict:
        p: dict[str, Any] = {
            "angle_deg":   self.angle_deg,
            "cut_body_id": self.cut_body_id,
            "axis_point":  self.axis_point,
            "axis_dir":    self.axis_dir,
        }
        if self.source_body_id is not None:
            p["source_body_id"] = self.source_body_id
        if self.source_sketch_id is not None:
            p["source_sketch_id"] = self.source_sketch_id
        if self.face_indices is not None:
            p["face_indices"] = self.face_indices
        if self.face_centroids is not None:
            p["face_centroids"] = self.face_centroids
        return p

    @classmethod
    def _from_params(cls, params: dict) -> "CrossBodyRevolveCutOp":
        return cls(
            cut_body_id      = params.get("cut_body_id", ""),
            source_body_id   = params.get("source_body_id"),
            source_sketch_id = params.get("source_sketch_id"),
            angle_deg        = float(params.get("angle_deg", 360.0)),
            axis_point       = params.get("axis_point"),
            axis_dir         = params.get("axis_dir"),
            face_indices     = params.get("face_indices"),
            face_centroids   = params.get("face_centroids"),
        )
