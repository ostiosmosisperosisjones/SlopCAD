"""
cad/op_base.py

Base Op class and shared _push_result() commit helper.
"""

from __future__ import annotations
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from cad.history import History


def _compound_of(solids):
    """Build one raw TopoDS_Compound from a list of build123d solids.

    Used to store a split op's *retained* solids (the parent's own pieces)
    without dragging along the split-away children.  Wrap the return in
    build123d's Compound(...) at the call site.
    """
    from OCP.TopoDS import TopoDS_Compound
    from OCP.BRep import BRep_Builder
    comp = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(comp)
    for s in solids:
        builder.Add(comp, s.wrapped)
    return comp


class Op:
    """Marker base class for all operation types."""

    def execute(self, shape: Any, history: "History", entry_index: int) -> Any:
        raise NotImplementedError

    def creates_body_from_nothing(self, history: "History", entry_index: int) -> bool:
        """
        True if this op is the first one on its body chain and produces a
        shape with no prior input — e.g. a sketch extrude on a world plane.
        Replay uses this to skip the "no input shape" guard.

        Distinct from force_new_body: that flag means "spawn a child body
        alongside the source body".  This predicate means "no source shape
        is required at all".
        """
        return getattr(self, 'force_new_body', False)

    def commit(self, viewport: Any, extra_params: dict | None = None) -> Any:
        """
        Run the operation for the first time (or after an edit), push history
        entries, and trigger mesh rebuild.  Returns result shape or None.
        """
        raise NotImplementedError

    def commit_async(self, viewport: Any, extra_params: dict | None = None) -> None:
        """
        Async variant of commit: run the heavy OCC compute in a background
        thread, then push history and rebuild on the main thread.

        Subclasses override _compute_shape() and _finalize() to opt in.
        Falls back to synchronous commit() if not overridden.
        """
        try:
            compute_fn, finalize_fn = self._split_commit(viewport, extra_params)
        except NotImplementedError:
            self.commit(viewport, extra_params)
            return
        except Exception as ex:
            print(f"[Op] FAILED: {ex}")
            self._push_failed_entry(viewport, str(ex), extra_params)
            return
        viewport.run_op_async(type(self).__name__.replace("Op", ""), compute_fn, finalize_fn)

    def _push_failed_entry(self, viewport: Any, error_msg: str,
                           extra_params: dict | None = None) -> None:
        """Push a red history entry when the op fails before compute starts."""
        try:
            dist = getattr(self, 'distance', None)
            if dist is not None:
                op_str = "cut" if dist < 0 else "extrude"
            else:
                op_str = getattr(self, '_op_str', None) or type(self).__name__.replace("Op", "").lower()
            body_id   = getattr(self, 'source_body_id', None) or getattr(self, 'body_id', None)
            op_params = self.to_params()
            if extra_params:
                op_params.update(extra_params)
            shape_before = viewport.workspace.current_shape(body_id) if body_id else None
            from cad.units import format_op_label as _lbl
            label = _lbl(op_str, op_params)
            entry = viewport.history.push(
                label=label, operation=op_str, params=op_params,
                body_id=body_id, face_ref=None,
                shape_before=shape_before, shape_after=None)
            entry.error     = True
            entry.error_msg = error_msg
            viewport._post_push_cascade(body_id)
            viewport.history_changed.emit()
        except Exception as ex2:
            print(f"[Op] Could not push failed entry: {ex2}")

    def _split_commit(self, viewport: Any, extra_params: dict | None):
        """
        Return (compute_fn, finalize_fn) where:
          compute_fn()             — pure OCC, runs in thread, returns shape_after
          finalize_fn(shape_after) — Qt/history, runs on main thread
        Raise NotImplementedError to fall back to synchronous commit().
        """
        raise NotImplementedError

    def reopen(self, viewport: Any, history_idx: int) -> None:
        """
        Restore panel UI state for editing this entry.
        Called by reopen_extrude() after seek/rebuild; op-specific branching
        lives here rather than in the viewport.
        """
        raise NotImplementedError

    def to_params(self) -> dict:
        raise NotImplementedError

    @classmethod
    def from_params(cls, operation: str, params: dict) -> "Op | None":
        """
        Reconstruct the correct Op subclass from an operation string + params
        dict.  Returns None if the operation type is not recognised.
        """
        from cad.op_types import _FROM_PARAMS
        factory = _FROM_PARAMS.get(operation)
        if factory is None:
            return None
        return factory(operation, params)


# Maximum UV-space distance between a stored sketch-face centroid and the
# nearest current region centroid before we declare the original face gone.
# 5mm is generous enough to absorb realistic sketch edits (moves, small
# resizes, fillet/trim re-topology) but tight enough to catch deletions —
# practical sketches live at 1–100mm scales so a 5mm centroid shift means
# the face's identity is no longer recognisable.
_FACE_CENTROID_MATCH_TOL = 5.0


def _match_face_by_centroid(target, all_faces: list, all_regions: list,
                            op_name: str, tol: float = _FACE_CENTROID_MATCH_TOL):
    """
    Find the face in *all_faces* whose region centroid is closest to *target*.
    Raises RuntimeError when the closest match is further than *tol* mm —
    i.e. the original face has been deleted or moved beyond recognition.

    target      : (u, v) numpy array — the stored centroid from the original op
    all_faces   : faces returned by SketchEntry.build_faces()[0]
    all_regions : regions returned by SketchEntry.build_faces()[1]
    op_name     : operation name to embed in the error message
    """
    import numpy as np
    if not all_regions:
        raise RuntimeError(f"{op_name}: sketch has no regions to match against")
    best_i, best_d = -1, float('inf')
    for i, region in enumerate(all_regions):
        outer_uvs = region[0]
        c = np.array(outer_uvs).mean(axis=0)
        d = float(np.linalg.norm(c - target))
        if d < best_d:
            best_d, best_i = d, i
    if best_d > tol:
        raise RuntimeError(
            f"{op_name}: stored sketch face no longer exists "
            f"(nearest region centroid is {best_d:.3f}mm from stored "
            f"target, threshold is {tol}mm)")
    return all_faces[best_i]


def _push_result(viewport, op_str: str, op_params: dict, body_id: str,
                 face_ref: Any, shape_before: Any, shape_after: Any,
                 original_solid_count: int, split_key: str = "split_from"):
    """
    Push history entry/entries for a boolean result and trigger mesh rebuild.

    If shape_after is None the op failed — push a single diverged entry so it
    appears red in the history panel instead of silently disappearing.

    If the result has more solids than the original (a split), each extra solid
    gets its own 'import' entry and workspace body.  Otherwise a single entry
    updates the source body.
    """
    from cad.units import format_op_label as _lbl
    from build123d import Compound

    label = _lbl(op_str, op_params)

    if shape_after is None:
        err = getattr(viewport, '_pending_op_error', None) or f"{op_str} failed"
        entry = viewport.history.push(
            label=label, operation=op_str, params=op_params,
            body_id=body_id, face_ref=face_ref,
            shape_before=shape_before, shape_after=None)
        entry.error     = True
        entry.error_msg = err
        viewport._post_push_cascade(body_id)
        viewport.history_changed.emit()
        return set()

    solids = list(shape_after.solids())
    is_split = len(solids) > original_solid_count

    if is_split:
        # A split: the parent body keeps only its own solid(s); the extra
        # solids become new child bodies.  Previously the parent entry stored
        # the FULL multi-solid compound, so the parent body rendered every
        # disconnected piece — "two disjoint objects still treated as one".
        # Trim the parent to its retained solids and fingerprint them with
        # SolidRefs so replay re-selects the same pieces (OCCT solid order is
        # not stable across a rebuild).
        from cad.solid_ref import solid_refs_to_dicts
        retained = solids[:original_solid_count]
        op_params["retained_solid_refs"] = solid_refs_to_dicts(retained)
        parent_shape = Compound(_compound_of(retained))
    else:
        parent_shape = shape_after

    parent_entry = viewport.history.push(
        label=label, operation=op_str, params=op_params,
        body_id=body_id, face_ref=face_ref,
        shape_before=shape_before, shape_after=parent_shape)

    if is_split:
        # New solids produced — add import entries for extras.
        from cad.op_types import ImportOp
        extra_solids = solids[original_solid_count:]
        for i, solid in enumerate(extra_solids):
            ws   = viewport.workspace
            name = ws.next_part_name()
            new_body = ws.add_body(name, Compound(solid.wrapped))
            new_body.created_at_entry_id = parent_entry.entry_id
            # Fingerprint the child's solid so replay re-derives it from the
            # parent's full result by geometry, not fragile solid-iteration
            # order (the baked shape isn't persisted across save/load).
            from cad.solid_ref import SolidRef
            _r = SolidRef.from_b3d_solid(solid)
            imp_params = {
                "source_entry_id": parent_entry.entry_id,
                split_key:         body_id,
                "solid_index":     original_solid_count + i,
                "solid_ref":       {"volume": _r.volume, "com": list(_r.com),
                                    "extents": list(_r.extents)},
            }
            viewport.history.push(
                label=f"import  {name}", operation="import",
                params=imp_params, body_id=new_body.id, face_ref=None,
                shape_before=None, shape_after=Compound(solid.wrapped))

    viewport._post_push_cascade(body_id)
    print(f"[{op_str.capitalize()}] body='{viewport.workspace.bodies[body_id].name}' "
          f"({len(solids)} solid(s))")
    viewport.history_changed.emit()
    return {body_id}
