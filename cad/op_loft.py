"""
cad/op_loft.py

SketchLoftOp — loft between 2+ ordered sketch profiles to form a solid,
optionally fused into or cut from another body.

Each profile sketch must produce exactly one closed-loop face; multi-loop
profiles are rejected with a clear error.  Profile order in
from_sketch_ids defines the loft sweep direction.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from cad.op_base import Op, _push_result

if TYPE_CHECKING:
    from cad.history import History


@dataclass
class SketchLoftOp(Op):
    """
    Loft (or cut-loft) between ordered sketch profiles.

    from_profiles    : ordered [(sketch_entry_id, face_idx_or_None)] tuples.
                       face_idx=None means "use the first/only face of the
                       sketch" (preserves v1 single-loop behavior).  Otherwise
                       use that specific face of the sketch — supports lofting
                       between disconnected loops on the same sketch.
    merge_body_id    : body to fuse the result into (mutually exclusive
                       with cut_body_id and force_new_body)
    cut_body_id      : body to subtract the loft from (defines a loft_cut)
    force_new_body   : True → always create a new body, ignore merge target
    ruled            : True → ruled loft (straight walls), False → smooth
    continuity       : "C0" (creases allowed), "C1" (smooth, default), or "C2"
                       (extra smooth — matches curvature too).  Only affects
                       multi-section lofts; with two sections C0/C1/C2 give
                       the same shape but different surface parameterization.
    """
    _CONTINUITY_DEFAULT = "C1"
    _CONTINUITY_VALID   = ("C0", "C1", "C2")

    from_profiles:   list                     = field(default_factory=list)
    merge_body_id:   str  | None              = None
    cut_body_id:     str  | None              = None
    force_new_body:  bool                     = False
    ruled:           bool                     = False
    continuity:      str                      = "C1"
    merged_from:     str  | None              = None   # source body id (loft into merge target)

    def __post_init__(self):
        # Normalize: a stored from_sketch_ids (legacy) becomes (sid, None) pairs.
        # Tuples may also come back from JSON as lists — coerce to tuples.
        self.from_profiles = [
            (p[0], p[1] if len(p) > 1 else None) if isinstance(p, (list, tuple))
            else (p, None)
            for p in self.from_profiles
        ]
        # Tolerate junk continuity values (e.g. from a corrupt save) by
        # clamping to the default instead of raising.
        if self.continuity not in self._CONTINUITY_VALID:
            self.continuity = self._CONTINUITY_DEFAULT

    def _build_loft(self, faces: list):
        """Build a solid via OCCT BRepOffsetAPI_ThruSections so we control
        continuity + ruled at the OCCT level.  Returns a build123d Solid."""
        from OCP.BRepOffsetAPI import BRepOffsetAPI_ThruSections
        from OCP.GeomAbs import GeomAbs_Shape
        from build123d import Solid

        _CONT_MAP = {
            "C0": GeomAbs_Shape.GeomAbs_C0,
            "C1": GeomAbs_Shape.GeomAbs_C1,
            "C2": GeomAbs_Shape.GeomAbs_C2,
        }
        cont = _CONT_MAP.get(self.continuity, GeomAbs_Shape.GeomAbs_C1)
        is_solid = True

        ts = BRepOffsetAPI_ThruSections(is_solid, bool(self.ruled))
        ts.SetContinuity(cont)
        for face in faces:
            ts.AddWire(face.outer_wire().wrapped)
        ts.Build()
        if not ts.IsDone():
            raise RuntimeError("BRepOffsetAPI_ThruSections failed to build")
        return Solid(ts.Shape())

    @property
    def from_sketch_ids(self) -> list[str]:
        """Back-compat read accessor — returns just the sketch ids."""
        return [sid for sid, _fi in self.from_profiles]

    def creates_body_from_nothing(self, history: "History", entry_index: int) -> bool:
        # A loft creating a fresh body has no source shape requirement.
        return (self.force_new_body
                or self.merge_body_id is not None
                or self.cut_body_id is None)

    # ------------------------------------------------------------------
    # Profile resolution
    # ------------------------------------------------------------------

    def _resolve_profile_faces(self, history: "History", entry_index: int) -> list:
        """Resolve every profile to its build123d Face.

        Each entry in from_profiles is (sketch_id, face_idx_or_None).  When
        face_idx is None the sketch must produce exactly one face (otherwise
        the loft would be ambiguous).
        """
        from cad.history import _replay_sketch_entry
        if len(self.from_profiles) < 2:
            raise RuntimeError("SketchLoftOp: loft requires at least 2 profiles")
        faces = []
        for n, (sid, face_idx) in enumerate(self.from_profiles, start=1):
            idx = history.id_to_index(sid)
            if idx is None:
                raise RuntimeError(
                    f"SketchLoftOp: profile #{n} sketch '{sid}' not found")
            if idx >= entry_index:
                raise RuntimeError(
                    f"SketchLoftOp: profile #{n} sketch is after this op — "
                    f"invalid reorder")
            rec = history._entries[idx]
            if rec.error:
                raise RuntimeError(
                    f"SketchLoftOp: profile #{n} sketch is in an error state")
            se = rec.params.get("sketch_entry")
            if se is None:
                raise RuntimeError(
                    f"SketchLoftOp: profile #{n} has no sketch_entry")
            if se.plane_source is not None:
                ok, err = _replay_sketch_entry(se, history, before_index=entry_index)
                if not ok:
                    raise RuntimeError(
                        f"SketchLoftOp: profile #{n} reprojection failed: {err}")
            all_faces, _ = se.build_faces()
            if not all_faces:
                raise RuntimeError(
                    f"SketchLoftOp: profile #{n} has no closed loops")
            if face_idx is None:
                if len(all_faces) != 1:
                    raise RuntimeError(
                        f"SketchLoftOp: profile #{n} has {len(all_faces)} faces; "
                        f"pick a specific face for this profile")
                faces.append(all_faces[0])
            else:
                if face_idx < 0 or face_idx >= len(all_faces):
                    raise RuntimeError(
                        f"SketchLoftOp: profile #{n} face index {face_idx} "
                        f"out of range (sketch has {len(all_faces)} faces)")
                faces.append(all_faces[face_idx])
        return faces

    # ------------------------------------------------------------------
    # Execute (replay)
    # ------------------------------------------------------------------

    def execute(self, shape: Any, history: "History", entry_index: int) -> Any:
        from build123d import Compound
        from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse, BRepAlgoAPI_Cut
        from OCP.TopTools import TopTools_ListOfShape

        faces = self._resolve_profile_faces(history, entry_index)
        try:
            loft_part = self._build_loft(faces)
        except Exception as ex:
            raise RuntimeError(f"SketchLoftOp: loft failed: {ex}")

        loft_solid = Compound(loft_part.wrapped)

        # ---- Cut path ----
        if self.cut_body_id is not None:
            target = history._shape_for_body_at(self.cut_body_id, entry_index)
            if target is None:
                raise RuntimeError(
                    f"SketchLoftOp: no shape for cut target '{self.cut_body_id}'")
            lst_a = TopTools_ListOfShape(); lst_a.Append(target.wrapped)
            lst_b = TopTools_ListOfShape(); lst_b.Append(loft_solid.wrapped)
            op = BRepAlgoAPI_Cut()
            op.SetArguments(lst_a); op.SetTools(lst_b)
            op.SetRunParallel(True); op.Build()
            if not op.IsDone():
                raise RuntimeError("SketchLoftOp: boolean cut failed")
            return Compound(op.Shape())

        # ---- Merge path ----
        if self.merge_body_id is not None and not self.force_new_body:
            target = history._shape_for_body_at(self.merge_body_id, entry_index)
            if target is None:
                raise RuntimeError(
                    f"SketchLoftOp: no shape for merge target '{self.merge_body_id}'")
            lst_a = TopTools_ListOfShape(); lst_a.Append(target.wrapped)
            lst_b = TopTools_ListOfShape(); lst_b.Append(loft_solid.wrapped)
            op = BRepAlgoAPI_Fuse()
            op.SetArguments(lst_a); op.SetTools(lst_b)
            op.SetRunParallel(True); op.Build()
            if not op.IsDone():
                raise RuntimeError("SketchLoftOp: boolean fuse failed")
            return Compound(op.Shape())

        # ---- New body ----
        solids = list(loft_solid.solids())
        if not solids:
            raise RuntimeError("SketchLoftOp: result contains no solids")
        return solids[0]

    # ------------------------------------------------------------------
    # Commit (first run + after edit)
    # ------------------------------------------------------------------

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
        from build123d import Compound
        from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse, BRepAlgoAPI_Cut
        from OCP.TopTools import TopTools_ListOfShape

        # Resolve profiles up-front so the panel-side error message is precise.
        faces = self._resolve_profile_faces(viewport.history,
                                            viewport.history.cursor + 1)

        op_params = self.to_params()
        if extra_params:
            op_params.update(extra_params)
        build_loft = self._build_loft   # capture method ref for closures

        # ---- Cut path ----
        if self.cut_body_id is not None:
            target_shape = viewport.workspace.current_shape(self.cut_body_id)
            if target_shape is None:
                raise RuntimeError(
                    f"[Loft] No shape for cut target '{self.cut_body_id}'")
            target_occ           = target_shape.wrapped
            original_solid_count = len(list(target_shape.solids()))
            cut_body_id          = self.cut_body_id

            def compute():
                loft_part = build_loft(faces)
                lst_a = TopTools_ListOfShape(); lst_a.Append(target_occ)
                lst_b = TopTools_ListOfShape(); lst_b.Append(loft_part.wrapped)
                op = BRepAlgoAPI_Cut()
                op.SetArguments(lst_a); op.SetTools(lst_b)
                op.SetRunParallel(True); op.Build()
                if not op.IsDone():
                    raise RuntimeError("Boolean cut failed")
                return Compound(op.Shape())

            def finalize(result):
                _push_result(viewport, "loft_cut", op_params, cut_body_id,
                             None, target_shape, result, original_solid_count,
                             split_key="split_from")
                viewport._selected_sketch_entry = None
                viewport._selected_sketch_face  = None

            return compute, finalize

        # ---- Merge path ----
        if self.merge_body_id is not None and not self.force_new_body:
            target_shape = viewport.workspace.current_shape(self.merge_body_id)
            if target_shape is None:
                raise RuntimeError("[Loft] Merge target has no shape.")
            target_occ           = target_shape.wrapped
            original_solid_count = len(list(target_shape.solids()))
            merge_body_id        = self.merge_body_id

            def compute():
                loft_part = build_loft(faces)
                lst_a = TopTools_ListOfShape(); lst_a.Append(target_occ)
                lst_b = TopTools_ListOfShape(); lst_b.Append(loft_part.wrapped)
                op = BRepAlgoAPI_Fuse()
                op.SetArguments(lst_a); op.SetTools(lst_b)
                op.SetRunParallel(True); op.Build()
                if not op.IsDone():
                    raise RuntimeError("Boolean fuse failed")
                return Compound(op.Shape())

            def finalize(merged):
                _push_result(viewport, "loft", op_params, merge_body_id,
                             None, target_shape, merged, original_solid_count)
                viewport._selected_sketch_entry = None
                viewport._selected_sketch_face  = None

            return compute, finalize

        # ---- New body ----
        first_sketch_id = self.from_profiles[0][0]

        def compute():
            loft_part = build_loft(faces)
            return Compound(loft_part.wrapped)

        preserved_body_ids = list((extra_params or {}).get("_preserved_body_ids", []))
        if extra_params:
            op_params.pop("_preserved_body_ids", None)

        def finalize(shape_after):
            from viewer.vp_extrude import _next_split_name
            from cad.units import format_op_label as _lbl
            ws = viewport.workspace
            # Name the new body after the first profile sketch's body, if any.
            first_idx = viewport.history.id_to_index(first_sketch_id)
            base_name = "Loft"
            if first_idx is not None:
                se = viewport.history.entries[first_idx].params.get("sketch_entry")
                if se is not None and se.body_id and se.body_id in ws.bodies:
                    base_name = ws.bodies[se.body_id].name
            solids = list(shape_after.solids())
            if not solids:
                raise RuntimeError("Loft produced no solids.")
            op_params["child_body_ids"] = []
            new_bodies = []
            for i, solid in enumerate(solids):
                new_name = _next_split_name(base_name, ws)
                preserved = preserved_body_ids[i] if i < len(preserved_body_ids) else None
                new_body = ws.add_body(new_name, Compound(solid.wrapped), body_id=preserved)
                new_bodies.append(new_body)
                op_params["child_body_ids"].append(new_body.id)
                print(f"[Loft] New body '{new_name}'")
            primary_body = new_bodies[0] if new_bodies else None
            tag_body_id  = primary_body.id if primary_body else None
            parent_entry = viewport.history.push(
                label=_lbl("loft", op_params), operation="loft",
                params=op_params, body_id=tag_body_id, face_ref=None,
                shape_before=None, shape_after=shape_after)
            for new_body in new_bodies:
                new_body.created_at_entry_id = parent_entry.entry_id
            viewport._rebuild_bodies({b.id for b in new_bodies})
            viewport.history_changed.emit()

        return compute, finalize

    # ------------------------------------------------------------------
    # Params round-trip
    # ------------------------------------------------------------------

    def to_params(self) -> dict:
        # Serialize profiles as a list of [sid, face_idx_or_null] lists so
        # they round-trip cleanly through JSON.
        p: dict[str, Any] = {
            "from_profiles": [[sid, fi] for sid, fi in self.from_profiles],
        }
        if self.merge_body_id is not None:
            p["merge_body_id"] = self.merge_body_id
        if self.cut_body_id is not None:
            p["cut_body_id"] = self.cut_body_id
        if self.force_new_body:
            p["force_new_body"] = True
        if self.ruled:
            p["ruled"] = True
        if self.continuity != self._CONTINUITY_DEFAULT:
            p["continuity"] = self.continuity
        if self.merged_from is not None:
            p["merged_from"] = self.merged_from
        return p

    @classmethod
    def _from_params(cls, params: dict, sign: int = 1) -> "SketchLoftOp":
        raw = params.get("from_profiles")
        if raw is None:
            # Legacy save: only sketch ids stored, treat each as a "whole sketch"
            # (face_idx=None) profile.
            raw = [(sid, None) for sid in params.get("from_sketch_ids", [])]
        return cls(
            from_profiles  = list(raw),
            merge_body_id  = params.get("merge_body_id"),
            cut_body_id    = params.get("cut_body_id"),
            force_new_body = bool(params.get("force_new_body", False)),
            ruled          = bool(params.get("ruled", False)),
            continuity     = params.get("continuity", cls._CONTINUITY_DEFAULT),
            merged_from    = params.get("merged_from"),
        )
