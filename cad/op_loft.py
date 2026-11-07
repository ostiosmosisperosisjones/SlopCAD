"""
cad/op_loft.py

SketchLoftOp — loft between 2+ ordered profiles to form a solid, optionally
fused into or cut from another body.

A profile is a ProfileSource (cad/profile_source.py): one loop of a sketch,
or one face of a body.  Each must resolve to exactly one face; a multi-loop
sketch given without a loop index is rejected with a clear error.  Profile
order in from_profiles defines the loft direction.
"""

from __future__ import annotations
from cad.solid_ref import as_compound
from cad.kernel_pool import offload
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from cad.op_base import Op, _push_result

if TYPE_CHECKING:
    from cad.history import History


@dataclass
class SketchLoftOp(Op):
    """
    Loft (or cut-loft) between ordered profiles.

    from_profiles    : ordered ProfileSource list — SketchProfile (one loop of
                       a sketch; face_indices=None means "the sketch's only
                       loop") or BodyFaceProfile (a body face).  Anything
                       cad.profile_source.as_profile accepts is normalized on
                       construction, incl. the legacy (sketch_id, face_idx)
                       tuples and bare sketch ids of older saves.
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
        from cad.profile_source import as_profile
        self.from_profiles = [as_profile(p) for p in self.from_profiles]
        # Tolerate junk continuity values (e.g. from a corrupt save) by
        # clamping to the default instead of raising.
        if self.continuity not in self._CONTINUITY_VALID:
            self.continuity = self._CONTINUITY_DEFAULT

    # Subclass hooks (SketchSweepOp): history operation name ("<op>_cut" for
    # the cut variant), fewest profiles, and the kernel task.
    _OP           = "loft"
    _MIN_PROFILES = 2

    def _task(self, faces: list, target=None, mode=None):
        """The solid through `faces`, or target with it cut / fused (mode
        "cut" / "fuse"); runs in a kernel worker."""
        return _loft_task(faces, self.ruled, self.continuity, target, mode)

    def _build_loft(self, faces: list):
        """The loft solid through `faces` (built in a kernel worker)."""
        return self._task(faces)

    @property
    def from_sketch_ids(self) -> list[str]:
        """The sketch ids among the profiles (body-face profiles skipped)."""
        from cad.profile_source import SketchProfile
        return [p.sketch_id for p in self.from_profiles
                if isinstance(p, SketchProfile)]

    def profile_keys(self) -> list[tuple]:
        """Panel keys of the profiles: (sketch_id, loop_idx_or_None) for a
        sketch loop, ("face", body_id, face_idx) for a body face."""
        from cad.profile_source import SketchProfile
        return [(p.sketch_id, p.single_index()) if isinstance(p, SketchProfile)
                else p.key() for p in self.from_profiles]

    def creates_body_from_nothing(self, history: "History", entry_index: int) -> bool:
        # A loft creating a fresh body has no source shape requirement.
        return (self.force_new_body
                or self.merge_body_id is not None
                or self.cut_body_id is None)

    # ------------------------------------------------------------------
    # Profile resolution
    # ------------------------------------------------------------------

    def _resolve_profile_faces(self, history: "History", entry_index: int) -> list:
        """Resolve every profile to its build123d Face — exactly one each
        (a multi-loop sketch needs a loop index, or the loft is ambiguous)."""
        if len(self.from_profiles) < self._MIN_PROFILES:
            raise RuntimeError(f"{type(self).__name__}: {self._OP} requires at "
                               f"least {self._MIN_PROFILES} profile(s)")
        faces = []
        for n, prof in enumerate(self.from_profiles, start=1):
            got = prof.resolve(history, entry_index,
                               f"{type(self).__name__}: profile #{n}")
            if len(got) != 1:
                raise RuntimeError(
                    f"{type(self).__name__}: profile #{n} has {len(got)} faces; "
                    f"pick a specific face for this profile")
            faces.append(got[0])
        return faces

    def _bake_profiles(self, history: "History", entry_index: int) -> None:
        """Commit-time: fingerprint each profile (sketch loop centroids +
        boundaries, body face refs) so replay re-finds it after edits."""
        for prof in self.from_profiles:
            prof.bake(history, entry_index)

    # ------------------------------------------------------------------
    # Execute (replay)
    # ------------------------------------------------------------------

    def execute(self, shape: Any, history: "History", entry_index: int) -> Any:
        faces = self._resolve_profile_faces(history, entry_index)

        # ---- Cut path ----
        if self.cut_body_id is not None:
            target = history._shape_for_body_at(self.cut_body_id, entry_index)
            if target is None:
                raise RuntimeError(
                    f"SketchLoftOp: no shape for cut target '{self.cut_body_id}'")
            try:
                return self._task(faces, target, "cut")
            except RuntimeError as ex:
                raise RuntimeError(f"SketchLoftOp: {ex}")

        # ---- Merge path ----
        if self.merge_body_id is not None and not self.force_new_body:
            target = history._shape_for_body_at(self.merge_body_id, entry_index)
            if target is None:
                raise RuntimeError(
                    f"SketchLoftOp: no shape for merge target '{self.merge_body_id}'")
            try:
                return self._task(faces, target, "fuse")
            except RuntimeError as ex:
                raise RuntimeError(f"SketchLoftOp: {ex}")

        # ---- New body ----
        try:
            loft_solid = as_compound(self._build_loft(faces))
        except Exception as ex:
            raise RuntimeError(f"SketchLoftOp: loft failed: {ex}")
        solids = list(loft_solid.solids())
        if not solids:
            raise RuntimeError("SketchLoftOp: result contains no solids")

        # entry_index can point past the end when an op is executed outside a
        # pushed history slot (e.g. preview/tests); guard the lookup.
        entry = (history._entries[entry_index]
                 if 0 <= entry_index < len(history._entries) else None)
        child_body_ids = entry.params.get("child_body_ids", []) if entry else []
        # force_new_body split mirror — see ExtrudeOp.execute. Assign each
        # loft lump to its child body by SolidRef so the mapping survives reload.
        if self.force_new_body and child_body_ids:
            from cad.solid_ref import assign_solids_to_children, solid_refs_from_dicts
            refs = solid_refs_from_dicts(entry.params.get("child_solid_refs"))
            assigned = assign_solids_to_children(
                loft_solid, child_body_ids, refs, history._workspace,
                history, entry_index)
            if assigned[0] is None:
                raise RuntimeError(
                    "SketchLoftOp: could not assign a solid to the primary "
                    "child body")
            return assigned[0]

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
        # Resolve profiles up-front so the panel-side error message is precise.
        faces = self._resolve_profile_faces(viewport.history,
                                            viewport.history.cursor + 1)
        self._bake_profiles(viewport.history, viewport.history.cursor + 1)

        op_params = self.to_params()
        if extra_params:
            op_params.update(extra_params)

        # ---- Cut path ----
        if self.cut_body_id is not None:
            target_shape = viewport.workspace.current_shape(self.cut_body_id)
            if target_shape is None:
                raise RuntimeError(
                    f"[Loft] No shape for cut target '{self.cut_body_id}'")
            original_solid_count = len(list(target_shape.solids()))
            cut_body_id          = self.cut_body_id

            def compute():
                return self._task(faces, target_shape, "cut")

            def finalize(result):
                _push_result(viewport, self._OP + "_cut", op_params, cut_body_id,
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
            original_solid_count = len(list(target_shape.solids()))
            merge_body_id        = self.merge_body_id

            def compute():
                return self._task(faces, target_shape, "fuse")

            def finalize(merged):
                _push_result(viewport, self._OP, op_params, merge_body_id,
                             None, target_shape, merged, original_solid_count)
                viewport._selected_sketch_entry = None
                viewport._selected_sketch_face  = None

            return compute, finalize

        # ---- New body ----
        def compute():
            return as_compound(self._task(faces))

        preserved_body_ids = list((extra_params or {}).get("_preserved_body_ids", []))
        if extra_params:
            op_params.pop("_preserved_body_ids", None)

        def finalize(shape_after):
            from cad.units import format_op_label as _lbl
            ws = viewport.workspace
            from cad.solid_ref import solid_refs_to_dicts
            # Failed compute: the error is already printed, and a new-body
            # entry with no body has nothing to show or replay.
            solids = list(shape_after.solids()) if shape_after is not None else []
            if not solids:
                if shape_after is not None:
                    print(f"[Op] FAILED: {self._OP.capitalize()} produced no solids.")
                viewport.history_changed.emit()
                return
            op_params["child_body_ids"] = []
            op_params["child_solid_refs"] = solid_refs_to_dicts(solids)
            new_bodies = []
            for i, solid in enumerate(solids):
                preserved = preserved_body_ids[i] if i < len(preserved_body_ids) else None
                # Fresh "Part N" for a new body; keep existing name on re-commit.
                existing = ws.bodies.get(preserved) if preserved else None
                new_name = existing.name if existing else ws.next_part_name()
                new_body = ws.add_body(new_name, as_compound(solid), body_id=preserved)
                new_bodies.append(new_body)
                op_params["child_body_ids"].append(new_body.id)
                print(f"[Loft] New body '{new_name}'")
            primary_body = new_bodies[0] if new_bodies else None
            tag_body_id  = primary_body.id if primary_body else None
            parent_entry = viewport.history.push(
                label=_lbl(self._OP, op_params), operation=self._OP,
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
        from cad.profile_source import profiles_to_dicts
        p: dict[str, Any] = {
            "from_profiles": profiles_to_dicts(self.from_profiles),
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
            raw = list(params.get("from_sketch_ids", []))
        return cls(
            from_profiles  = list(raw),
            merge_body_id  = params.get("merge_body_id"),
            cut_body_id    = params.get("cut_body_id"),
            force_new_body = bool(params.get("force_new_body", False)),
            ruled          = bool(params.get("ruled", False)),
            continuity     = params.get("continuity", cls._CONTINUITY_DEFAULT),
            merged_from    = params.get("merged_from"),
        )


def build_loft_solid(faces: list, ruled: bool, continuity: str):
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
    cont = _CONT_MAP.get(continuity, GeomAbs_Shape.GeomAbs_C1)
    is_solid = True

    ts = BRepOffsetAPI_ThruSections(is_solid, bool(ruled))
    ts.SetContinuity(cont)
    for face in faces:
        ts.AddWire(face.outer_wire().wrapped)
    ts.Build()
    if not ts.IsDone():
        raise RuntimeError("BRepOffsetAPI_ThruSections failed to build")
    return Solid(ts.Shape())


@offload()
def _loft_task(faces, ruled, continuity, target=None, mode=None):
    """The loft solid, or (mode "cut" / "fuse") target with it cut / fused."""
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse, BRepAlgoAPI_Cut
    from OCP.collections import List_TopoDS_Shape as TopTools_ListOfShape
    loft_part = build_loft_solid(faces, ruled, continuity)
    if mode is None:
        return loft_part
    lst_a = TopTools_ListOfShape(); lst_a.Append(target.wrapped)
    lst_b = TopTools_ListOfShape(); lst_b.Append(loft_part.wrapped)
    op = BRepAlgoAPI_Cut() if mode == "cut" else BRepAlgoAPI_Fuse()
    op.SetArguments(lst_a); op.SetTools(lst_b)
    op.SetRunParallel(True); op.Build()
    if not op.IsDone():
        raise RuntimeError(f"Boolean {mode} failed")
    return as_compound(op.Shape())
