"""
cad/op_chamfer.py

ChamferOp — chamfer selected edges/faces of a body with distance + angle.

Mirrors FaceFilletOp's structure (face_indices for replay-safe bulk picks,
edge_indices for live commit-time picks) but uses BRepFilletAPI_MakeChamfer
with the AddDA signature instead of fillet's Add(radius).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from cad.op_base import Op, _push_result

if TYPE_CHECKING:
    from cad.history import History


@dataclass
class ChamferOp(Op):
    """
    Chamfer edges of *source_body_id*.

    face_indices         : edges of these faces are all chamfered (replay-safe)
    edge_indices         : specific mesh edge indices to chamfer (commit-only)
    edge_refs            : EdgeRef fingerprints for edge_indices, used to re-find
                           edges during replay after upstream edits
    distance             : mm measured along the reference face
    angle_deg            : degrees, measured from the reference face (0–89.99)
    flip_reference_face  : pick the other adjacent face for angle measurement
    """
    source_body_id:      str
    face_indices:        list  = field(default_factory=list)
    edge_indices:        list  = field(default_factory=list)
    distance:            float = 1.0
    angle_deg:           float = 45.0
    flip_reference_face: bool  = False
    edge_refs:           list  = field(default_factory=list)

    def _resolve_edge_occs(self, viewport):
        """Look up TopoDS_Edge objects for stored edge_indices via the mesh."""
        if not self.edge_indices:
            return []
        mesh = viewport._meshes.get(self.source_body_id)
        if mesh is None:
            return []
        out = []
        for ei in self.edge_indices:
            if ei < len(mesh.topo_edges_occ):
                out.append(mesh.topo_edges_occ[ei])
        return out

    def _resolve_edge_refs(self, shape, history=None) -> list:
        """Re-find edges via EdgeRef provenance/fingerprints (replay-safe)."""
        if not self.edge_refs:
            return []
        out = []
        for ref in self.edge_refs:
            occ = ref.resolve(history, shape)
            if occ is None:
                if history is not None and ref.origin is not None \
                        and ref.origin.faces_apart(history, shape):
                    raise RuntimeError(
                        "ChamferOp: a selected edge no longer exists — an upstream "
                        "change removed it (e.g. an earlier fillet now runs over "
                        "it); edit the chamfer and reselect")
                raise RuntimeError(
                    f"ChamferOp: could not relocate edge "
                    f"(midpoint={ref.midpoint}, length={ref.length:.3f})")
            out.append(occ)
        return out

    def _collect_face_edges(self, shape) -> list:
        """Return raw TopoDS_Edge objects from every face in face_indices."""
        from cad.operations.chamfer import _build_edge_to_faces_map, smooth_edges
        edge_face_map = _build_edge_to_faces_map(shape.wrapped)
        out = []
        all_faces = list(shape.faces())
        for fi in self.face_indices:
            if fi >= len(all_faces):
                raise RuntimeError(f"Chamfer: face_idx {fi} out of range")
            face_edges = [edge.wrapped for edge in all_faces[fi].edges()]
            # A picked face means its corners, not boundaries it already
            # blends into (e.g. where an earlier fillet meets it).
            skip = set(smooth_edges(face_edges, edge_face_map))
            out += [e for k, e in enumerate(face_edges) if k not in skip]
        return _unique_edges(out)

    def execute(self, shape: Any, history: "History", entry_index: int) -> Any:
        from cad.operations.chamfer import chamfer_edges
        if shape is None:
            src = history._shape_for_body_at(self.source_body_id, entry_index)
            if src is None:
                raise RuntimeError(
                    f"ChamferOp: no shape for body '{self.source_body_id}'")
            shape = src
        # Combine face-derived edges with EdgeRef-resolved edges (dedup by id).
        face_edges = self._collect_face_edges(shape)
        ref_edges  = self._resolve_edge_refs(shape, history)
        edges = _unique_edges(face_edges + ref_edges)
        if not edges:
            raise RuntimeError("ChamferOp: no edges to chamfer on replay "
                                "(face_indices and edge_refs both empty)")
        return chamfer_edges(shape, edges, self.distance, self.angle_deg,
                              self.flip_reference_face)

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
        from cad.operations.chamfer import chamfer_edges
        from cad.edge_ref import EdgeRef

        shape_before = viewport.workspace.current_shape(self.source_body_id)
        if shape_before is None:
            raise RuntimeError(f"[Chamfer] No shape for body {self.source_body_id}")

        # Combine face_indices and direct edge picks.
        face_edges = self._collect_face_edges(shape_before)
        direct_edges = self._resolve_edge_occs(viewport)
        edges = _unique_edges(face_edges + direct_edges)

        # Capture replay-stable fingerprints for the directly-picked edges,
        # including adjacent-face signatures for topology-aware fallback so
        # downstream replay survives upstream edits (e.g. draft applied to
        # extrude) that move the edge geometrically.
        parent_occ = shape_before.wrapped
        hist = viewport.history
        self.edge_refs = [
            r for r in (EdgeRef.for_edge(hist, self.source_body_id, e, parent_occ,
                                         hist.cursor + 1)
                        for e in direct_edges)
            if r is not None
        ]

        op_params = self.to_params()
        if extra_params:
            op_params.update(extra_params)
        original_solid_count = len(list(shape_before.solids()))
        distance            = self.distance
        angle_deg           = self.angle_deg
        flip_reference_face = self.flip_reference_face

        def compute():
            return chamfer_edges(shape_before, edges, distance, angle_deg,
                                  flip_reference_face)

        def finalize(shape_after):
            _push_result(viewport, "chamfer", op_params, self.source_body_id,
                         None, shape_before, shape_after, original_solid_count)

        return compute, finalize

    def reopen(self, viewport: Any, history_idx: int) -> None:
        viewport.reopen_chamfer(history_idx)

    def to_params(self) -> dict:
        p: dict[str, Any] = {
            "source_body_id":      self.source_body_id,
            "face_indices":        list(self.face_indices),
            "edge_indices":        list(self.edge_indices),
            "distance":            self.distance,
            "angle_deg":           self.angle_deg,
        }
        if self.flip_reference_face:
            p["flip_reference_face"] = True
        if self.edge_refs:
            p["edge_refs"] = [r.to_dict() for r in self.edge_refs]
        return p

    @classmethod
    def _from_params(cls, params: dict) -> "ChamferOp":
        from cad.edge_ref import EdgeRef
        edge_refs = [EdgeRef.from_dict(r) for r in params.get("edge_refs", [])]
        return cls(
            source_body_id      = params.get("source_body_id", ""),
            face_indices        = list(params.get("face_indices", [])),
            edge_indices        = list(params.get("edge_indices", [])),
            distance            = float(params.get("distance", 1.0)),
            angle_deg           = float(params.get("angle_deg", 45.0)),
            flip_reference_face = bool(params.get("flip_reference_face", False)),
            edge_refs           = edge_refs,
        )


def _unique_edges(edges: list) -> list:
    """Drop repeats by topology (IsSame), not Python id: every .edges() call
    makes new wrappers, so an edge shared by two picked faces has two ids."""
    out = []
    for e in edges:
        if not any(e.IsSame(o) for o in out):
            out.append(e)
    return out
