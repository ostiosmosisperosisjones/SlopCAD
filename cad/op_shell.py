"""
cad/op_shell.py

ShellOp — hollow a body into walls of uniform thickness, the picked faces
left open.  Kernel call: cad/operations/shell.py (runs in a kernel worker).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from cad.op_base import Op, _push_result

if TYPE_CHECKING:
    from cad.history import History


@dataclass
class ShellOp(Op):
    """
    source_body_id : body to hollow
    face_indices   : faces removed (open) at commit time — fallback for replay
    face_refs      : AnyFaceRef fingerprints of those faces — stable replay
    thickness      : wall thickness, mm (> 0)
    outside        : grow walls out from the current skin instead of in
    """
    source_body_id: str
    face_indices:   list
    thickness:      float
    outside:        bool = False
    face_refs:      list = field(default_factory=list)

    def _resolve_faces(self, shape) -> list:
        """[(idx, b3d_face)] — by face_refs when saved, else by index."""
        all_faces = list(shape.faces())
        if self.face_refs:
            out = []
            for ref in self.face_refs:
                idx, face = ref.find_in(shape)
                if idx is None:
                    raise RuntimeError(
                        f"ShellOp: could not re-locate face "
                        f"(centroid={ref.centroid}, area={ref.area:.3f})")
                out.append((idx, face))
            return out
        for idx in self.face_indices:
            if idx >= len(all_faces):
                raise RuntimeError(f"ShellOp: face_idx {idx} out of range")
        return [(idx, all_faces[idx]) for idx in self.face_indices]

    def execute(self, shape: Any, history: "History", entry_index: int) -> Any:
        from cad.operations.shell import shell_body
        if shape is None:
            shape = history._shape_for_body_at(self.source_body_id, entry_index)
            if shape is None:
                raise RuntimeError(
                    f"ShellOp: no shape for body '{self.source_body_id}'")
        faces = [f.wrapped for _i, f in self._resolve_faces(shape)]
        return shell_body(shape.wrapped, faces, self.thickness, self.outside)

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
        from cad.operations.shell import shell_body
        from cad.face_ref import AnyFaceRef

        shape_before = viewport.workspace.current_shape(self.source_body_id)
        if shape_before is None:
            raise RuntimeError(f"[Shell] No shape for body {self.source_body_id}")
        all_faces = list(shape_before.faces())
        for idx in self.face_indices:
            if idx >= len(all_faces):
                raise RuntimeError(f"[Shell] face_idx {idx} out of range")
        face_occs = [all_faces[i].wrapped for i in self.face_indices]
        self.face_refs = [AnyFaceRef.from_occ_face(f) for f in face_occs]

        op_params = self.to_params()
        if extra_params:
            op_params.update(extra_params)
        original_solid_count = len(list(shape_before.solids()))
        body_occ, thickness, outside = shape_before.wrapped, self.thickness, self.outside

        def compute():
            return shell_body(body_occ, face_occs, thickness, outside)

        def finalize(shape_after):
            _push_result(viewport, "shell", op_params, self.source_body_id,
                         None, shape_before, shape_after, original_solid_count)

        return compute, finalize

    def reopen(self, viewport: Any, history_idx: int) -> None:
        vp = viewport
        if history_idx > 0:
            vp.history.seek(history_idx - 1)
            vp._rebuild_body_mesh(self.source_body_id)
        vp.history_changed.emit()
        # Saved indices are commit-time positions; re-find them by face_refs
        # in case upstream edits renumbered faces.
        indices = list(self.face_indices)
        shape = vp.workspace.current_shape(self.source_body_id)
        if shape is not None and self.face_refs:
            try:
                indices = [i for i, _f in self._resolve_faces(shape)]
            except RuntimeError as ex:
                print(f"[Shell] reopen: {ex}; using saved face indices")
        entry = vp.history.entries[history_idx]
        vp._show_shell_panel(self.source_body_id, indices, editing_entry=entry)
        panel = getattr(vp, '_shell_panel', None)
        if panel is not None:
            panel.set_shell(self.thickness, self.outside)
            panel._spinbox.set_binding(
                entry.params.get("exprs", {}).get("thickness"), self.thickness)
            panel._emit_preview()

    def to_params(self) -> dict:
        p: dict = {
            "source_body_id": self.source_body_id,
            "face_indices":   list(self.face_indices),
            "thickness":      self.thickness,
        }
        if self.outside:
            p["outside"] = True
        if self.face_refs:
            p["face_refs"] = [
                {"centroid": list(r.centroid),
                 "area":     r.area,
                 "category": r.category,
                 **({"normal": list(r.normal)} if r.normal else {})}
                for r in self.face_refs
            ]
        return p

    @classmethod
    def _from_params(cls, params: dict) -> "ShellOp":
        from cad.face_ref import AnyFaceRef
        refs = [AnyFaceRef(centroid=tuple(r["centroid"]), area=r["area"],
                           category=r.get("category", ""),
                           normal=tuple(r["normal"]) if r.get("normal") else None)
                for r in params.get("face_refs", [])]
        return cls(
            source_body_id = params.get("source_body_id", ""),
            face_indices   = list(params.get("face_indices", [])),
            thickness      = float(params.get("thickness", 1.0)),
            outside        = bool(params.get("outside", False)),
            face_refs      = refs,
        )
