"""
cad/op_sweep.py

SketchSweepOp — sweep one sketch profile along a path of sketch edges,
as a new body or fused into / cut from another.  Body handling (new body,
merge, cut, split children) is SketchLoftOp's; this adds the path.

The path is a set of drawn entities of one sketch, named by entity uid so
that edits renumbering the path sketch don't lose it — or a helix entry
(path_uids == [HELIX_PATH]), swept with the helix axis as binormal so the
profile doesn't roll.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from cad.op_loft import SketchLoftOp

if TYPE_CHECKING:
    from cad.history import History

HELIX_PATH = "helix"     # path_uids marker: path_sketch_id is a helix entry


@dataclass
class SketchSweepOp(SketchLoftOp):
    """
    from_profiles  : [(sketch_entry_id, face_idx_or_None)] — exactly one
    path_sketch_id : sketch entry holding the path
    path_uids      : uids of that sketch's entities forming the path
    frenet         : True → the profile turns with the path (corrected
                     Frenet); False → it keeps its orientation
    """
    _OP           = "sweep"
    _MIN_PROFILES = 1

    path_sketch_id: str        = ""
    path_uids:      list       = field(default_factory=list)
    frenet:         bool       = True

    def _task(self, faces: list, target=None, mode=None):
        from cad.operations.sweep import sweep_task
        if len(faces) != 1:
            raise RuntimeError("SketchSweepOp: sweep takes one profile")
        return sweep_task(faces, self._path_edges, self.frenet, target, mode,
                          binormal=getattr(self, "_binormal", None))

    def _resolve_profile_faces(self, history: "History", entry_index: int) -> list:
        # Resolve the path alongside the profile: every execute / commit path
        # calls this first, before _task.
        self._path_edges = self._resolve_path(history, entry_index)
        return super()._resolve_profile_faces(history, entry_index)

    def _resolve_path(self, history: "History", entry_index: int) -> list:
        from cad.history import _replay_sketch_entry
        idx = history.id_to_index(self.path_sketch_id)
        if idx is None:
            raise RuntimeError("SketchSweepOp: path sketch not found")
        if idx >= entry_index:
            raise RuntimeError("SketchSweepOp: path sketch is after this op — "
                               "invalid reorder")
        rec = history._entries[idx]
        self._binormal = None
        if rec.operation == "helix":
            if rec.error or rec.op is None:
                raise RuntimeError("SketchSweepOp: path helix is in an error state")
            edge, self._binormal, _turns = rec.op.edge(history, idx)
            return [edge.wrapped]
        se = rec.params.get("sketch_entry")
        if se is None or rec.error:
            raise RuntimeError("SketchSweepOp: path sketch is in an error state")
        if se.plane_source is not None:
            ok, err = _replay_sketch_entry(se, history, before_index=entry_index)
            if not ok:
                raise RuntimeError(f"SketchSweepOp: path sketch reprojection failed: {err}")
        by_uid = {getattr(e, "uid", None): e for e in se.entities}
        missing = [u for u in self.path_uids if u not in by_uid]
        if missing or not self.path_uids:
            raise RuntimeError(
                f"SketchSweepOp: {len(missing) or 'no'} path edge(s) missing "
                f"from the path sketch")
        edges = [se.entity_edge(by_uid[u]) for u in self.path_uids]
        if any(e is None for e in edges):
            raise RuntimeError("SketchSweepOp: path edges must be lines, arcs "
                               "or splines")
        return edges

    def to_params(self) -> dict:
        p = super().to_params()
        p["path_sketch_id"] = self.path_sketch_id
        p["path_uids"] = list(self.path_uids)
        if not self.frenet:
            p["frenet"] = False
        return p

    @classmethod
    def _from_params(cls, params: dict, sign: int = 1) -> "SketchSweepOp":
        return cls(
            from_profiles  = list(params.get("from_profiles", [])),
            merge_body_id  = params.get("merge_body_id"),
            cut_body_id    = params.get("cut_body_id"),
            force_new_body = bool(params.get("force_new_body", False)),
            merged_from    = params.get("merged_from"),
            path_sketch_id = params.get("path_sketch_id", ""),
            path_uids      = list(params.get("path_uids", [])),
            frenet         = bool(params.get("frenet", True)),
        )
