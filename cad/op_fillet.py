"""
cad/op_fillet.py

FaceFilletOp — fillet selected edges/faces of a body.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from cad.op_base import Op, _push_result

if TYPE_CHECKING:
    from cad.history import History


@dataclass
class FaceFilletOp(Op):
    """
    Fillet edges of *source_body_id* by *radius* mm.

    face_indices : edges of these faces are all filleted
    edge_indices : specific mesh edge indices (topo_edges_occ) to fillet
                   — transient; only valid at commit time
    edge_refs    : EdgeRef fingerprints for edge_indices, used to re-find
                   edges during replay after upstream edits
    radius       : mm
    """
    source_body_id: str
    face_indices:   list
    edge_indices:   list  = field(default_factory=list)
    radius:         float = 1.0
    edge_refs:      list  = field(default_factory=list)

    def _resolve_edge_occs(self, viewport):
        """Look up TopoDS_Edge objects for stored edge_indices via the mesh."""
        if not self.edge_indices:
            return []
        mesh = viewport._meshes.get(self.source_body_id)
        if mesh is None:
            return []
        result = []
        for ei in self.edge_indices:
            if ei < len(mesh.topo_edges_occ):
                result.append(mesh.topo_edges_occ[ei])
        return result

    def _resolve_edge_refs(self, shape) -> list:
        """Re-find edges via EdgeRef geometry fingerprints (replay-safe)."""
        if not self.edge_refs:
            return []
        out = []
        for ref in self.edge_refs:
            _, occ, _ = ref.find_in(shape)
            if occ is None:
                raise RuntimeError(
                    f"FaceFilletOp: could not relocate edge "
                    f"(midpoint={ref.midpoint}, length={ref.length:.3f})")
            out.append(occ)
        return out

    def execute(self, shape: Any, history: "History", entry_index: int) -> Any:
        from cad.operations.fillet import fillet_edges
        if shape is None:
            src = history._shape_for_body_at(self.source_body_id, entry_index)
            if src is None:
                raise RuntimeError(
                    f"FaceFilletOp: no shape for body '{self.source_body_id}'")
            shape = src
        # edge_indices are commit-time mesh handles; replay uses edge_refs
        # (geometry fingerprints) to relocate the same edges after upstream
        # edits perturb the topology.
        edge_occs = self._resolve_edge_refs(shape)
        return fillet_edges(shape, self.face_indices, edge_occs, self.radius)

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
        from cad.operations.fillet import fillet_edges
        from cad.edge_ref import EdgeRef

        shape_before = viewport.workspace.current_shape(self.source_body_id)
        if shape_before is None:
            raise RuntimeError(f"[Fillet] No shape for body {self.source_body_id}")

        all_faces = list(shape_before.faces())
        for idx in self.face_indices:
            if idx >= len(all_faces):
                raise RuntimeError(f"[Fillet] face_idx {idx} out of range")

        edge_occs = self._resolve_edge_occs(viewport)

        # Capture replay-stable fingerprints for each picked edge, including
        # adjacent-face signatures for topology-aware fallback when geometric
        # matching fails after an upstream edit (e.g. draft applied to extrude).
        # Skip any edge that can't be fingerprinted (returns None).
        parent_occ = shape_before.wrapped
        self.edge_refs = [
            r for r in (EdgeRef.from_occ_edge(e, parent_shape=parent_occ)
                        for e in edge_occs)
            if r is not None
        ]

        op_params = self.to_params()
        if extra_params:
            op_params.update(extra_params)
        original_solid_count = len(list(shape_before.solids()))
        face_indices = list(self.face_indices)
        radius = self.radius

        def compute():
            return fillet_edges(shape_before, face_indices, edge_occs, radius)

        def finalize(shape_after):
            _push_result(viewport, "fillet", op_params, self.source_body_id,
                         None, shape_before, shape_after, original_solid_count)

        return compute, finalize

    def reopen(self, viewport: Any, history_idx: int) -> None:
        viewport.reopen_fillet(history_idx)

    def to_params(self) -> dict:
        p: dict = {
            "source_body_id": self.source_body_id,
            "face_indices":   self.face_indices,
            "edge_indices":   self.edge_indices,
            "radius":         self.radius,
        }
        if self.edge_refs:
            p["edge_refs"] = [
                {"midpoint":  list(r.midpoint),
                 "length":    r.length,
                 "tangent":   list(r.tangent),
                 "face_sigs": list(r.face_sigs)}
                for r in self.edge_refs
            ]
        return p

    @classmethod
    def _from_params(cls, params: dict) -> "FaceFilletOp":
        from cad.edge_ref import EdgeRef
        if "face_indices" in params:
            face_indices = list(params["face_indices"])
        else:
            face_indices = [int(params.get("face_idx", 0))]
        edge_refs = [
            EdgeRef(midpoint=tuple(r["midpoint"]),
                    length=float(r["length"]),
                    tangent=tuple(r["tangent"]),
                    face_sigs=list(r.get("face_sigs", [])))
            for r in params.get("edge_refs", [])
        ]
        return cls(
            source_body_id = params.get("source_body_id", ""),
            face_indices   = face_indices,
            edge_indices   = list(params.get("edge_indices", [])),
            radius         = float(params.get("radius", 1.0)),
            edge_refs      = edge_refs,
        )
