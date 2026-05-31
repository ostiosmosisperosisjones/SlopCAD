"""
viewer/hover.py

HoverState — screen-space projection cache and hover queries.

Occlusion testing uses ray casting (same Möller–Trumbore code as face
picking) rather than depth buffer sampling.  The depth buffer approach
fails when the far clip plane is very large (all depths compress to ~0.999),
making the per-pixel tolerance meaningless.  Ray casting is exact regardless
of clip plane range.

Committed sketch line entities are projected alongside mesh edges so they
participate in hover and selection naturally.  Their body_id key has the
form  "__sketch__{history_idx}__{entity_idx}"  which the viewport parses
to produce a SketchEdgeSel.
"""

from __future__ import annotations
import numpy as np
from OpenGL.GL import *


VERTEX_HOVER_RADIUS = 12   # screen pixels
EDGE_HOVER_RADIUS   = 6    # screen pixels

# Prefix used for synthetic sketch-edge keys in the hover cache
_SKETCH_KEY_PREFIX = "__sketch__"

_SKETCHVTX_KEY_PREFIX = "__sketchvtx__"

def sketch_vtx_key(history_idx: int) -> str:
    return f"{_SKETCHVTX_KEY_PREFIX}{history_idx}"

def parse_sketch_vtx_key(key: str) -> int | None:
    """Return history_idx if key is a sketch-vertex key, else None."""
    if not key.startswith(_SKETCHVTX_KEY_PREFIX):
        return None
    try:
        return int(key[len(_SKETCHVTX_KEY_PREFIX):])
    except ValueError:
        return None


def _sketch_key(history_idx: int, entity_idx: int) -> str:
    return f"{_SKETCH_KEY_PREFIX}{history_idx}__{entity_idx}"


def parse_sketch_key(key: str) -> tuple[int, int] | None:
    """
    If key is a sketch edge hover key, return (history_idx, entity_idx).
    Otherwise return None.
    """
    if not key.startswith(_SKETCH_KEY_PREFIX):
        return None
    rest = key[len(_SKETCH_KEY_PREFIX):]
    parts = rest.split("__")
    if len(parts) != 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def _ray_hits_anything(eye: np.ndarray, target: np.ndarray,
                        meshes: dict, workspace) -> bool:
    """
    Return True if any mesh triangle blocks the line of sight from eye
    to target.
    """
    from cad.picker import pick_face

    direction = eye - target
    dist      = float(np.linalg.norm(direction))
    if dist < 1e-9:
        return False
    ray_dir = direction / dist

    origin = target + ray_dir * 0.05

    for body_id, mesh in meshes.items():
        body = workspace.bodies.get(body_id)
        if body and not body.visible:
            continue
        result = pick_face(mesh, origin, ray_dir, return_t=True)
        if result is None:
            continue
        _, t = result
        if t < dist - 0.05:
            return True
    return False


class HoverState:
    """
    Holds cached screen-space projections and answers hover queries.

    Usage
    -----
    After paintGL (geometry drawn, matrices captured):
        hover.rebuild(meshes, workspace, modelview, projection, viewport, dpr,
                      history=history)   ← pass history for sketch edges

    On mouseMoveEvent:
        body_id, vert_idx = hover.vertex_at(x, y)
        body_id, edge_idx = hover.edge_at(x, y)
            body_id may be a sketch key — use parse_sketch_key() to detect.
    """

    def __init__(self):
        self._sv:    dict[str, np.ndarray]       = {}
        self._sv3d:  dict[str, np.ndarray]       = {}
        self._se:    dict[str, list[np.ndarray]] = {}
        self._se3d:  dict[str, list[np.ndarray]] = {}
        self._se_fn: dict[str, list[np.ndarray]] = {}  # adjacent face normals per edge
        self._eye:   np.ndarray | None           = None
        self._meshes    = None
        self._workspace = None
        self._ready     = False

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def rebuild(self, meshes, workspace, modelview, projection, viewport, dpr,
                camera_eye: np.ndarray = None, history=None,
                active_sketch=None, editing_sketch_idx=None):
        """
        Project all topo verts/edges, committed sketch line entities,
        and active sketch line entities to screen space.

        history            : History | None
        active_sketch      : SketchMode | None — the live sketch session if any
        editing_sketch_idx : int | None — history index of the sketch currently
            being edited.  Its committed edges are skipped so they don't shadow
            the active sketch's (-1) edges in the hover cache, which would make
            picks resolve to the committed entry and break active-sketch
            selection (mirror/construction toggle) after re-entry.
        """
        self._ready = False
        if modelview is None:
            return

        mv  = np.array(modelview,  dtype=np.float64).reshape(4, 4)
        prj = np.array(projection, dtype=np.float64).reshape(4, 4)
        mvp = mv @ prj
        vw  = float(viewport[2])
        vh  = float(viewport[3])

        self._eye       = camera_eye.copy() if camera_eye is not None \
                          else np.zeros(3)
        self._meshes    = meshes
        self._workspace = workspace

        def _project(pts: np.ndarray) -> np.ndarray:
            """(N,3) world → (N,2) logical widget pixels."""
            n      = len(pts)
            ones   = np.ones((n, 1), dtype=np.float64)
            clip   = np.hstack([pts.astype(np.float64), ones]) @ mvp
            w      = clip[:, 3]
            safe_w = np.where(np.abs(w) > 1e-9, w, 1e-9)
            ndcx   = clip[:, 0] / safe_w
            ndcy   = clip[:, 1] / safe_w
            sx     = (ndcx * 0.5 + 0.5) * vw
            sy     = (ndcy * 0.5 + 0.5) * vh
            wx     = sx / dpr
            wy     = (vh - sy) / dpr
            return np.stack([wx, wy], axis=1)

        self._sv.clear();   self._sv3d.clear()
        self._se.clear();   self._se3d.clear();  self._se_fn.clear()

        # ------------------------------------------------------------------
        # Mesh topo verts and edges
        # ------------------------------------------------------------------
        for body_id, mesh in meshes.items():
            body = workspace.bodies.get(body_id)
            if body and not body.visible:
                continue

            tv = mesh.topo_verts
            self._sv[body_id]   = _project(tv) if len(tv) > 0 \
                                   else np.zeros((0, 2))
            self._sv3d[body_id] = tv

            se_list, se3d_list, sefn_list = [], [], []
            fn_list = getattr(mesh, 'topo_edge_face_normals', [])
            for i, edge_pts in enumerate(mesh.topo_edges):
                proj = _project(edge_pts)
                se_list.append(proj)
                se3d_list.append(edge_pts)
                sefn_list.append(fn_list[i] if i < len(fn_list) else
                                 np.zeros((0, 3), dtype=np.float32))
                if not np.all(np.isfinite(proj)):
                    mid3d = edge_pts[len(edge_pts)//2]
                    print(f"[DBG] body={body_id} edge {i} projects to non-finite: 3d={mid3d.tolist()} screen={proj}")
            self._se[body_id]   = se_list
            self._se3d[body_id] = se3d_list
            self._se_fn[body_id] = sefn_list

        # ------------------------------------------------------------------
        # Committed sketch line entities
        # Each LineEntity becomes a 2-point edge in the hover cache,
        # keyed by the synthetic sketch key so the viewport can identify it.
        # ------------------------------------------------------------------
        if history is not None:
            self._add_sketch_edges(history, _project,
                                   skip_idx=editing_sketch_idx)

        if active_sketch is not None:
            self._add_active_sketch_edges(active_sketch, _project)

        self._ready = True

    def _add_sketch_edges(self, history, project_fn, skip_idx=None):
        """Project committed sketch LineEntity/ArcEntity objects into the hover cache."""
        from cad.sketch import LineEntity, ArcEntity, SketchEntry
        cursor = history.cursor
        for i, entry in enumerate(history.entries):
            if i > cursor:
                break
            if entry.operation != "sketch":
                continue
            # The sketch being edited is represented by the active (-1) cache;
            # skip its committed copy so picks don't resolve to the wrong key.
            if skip_idx is not None and i == skip_idx:
                continue
            se = entry.params.get("sketch_entry")
            if se is None or not se.visible:
                continue
            endpoints = []
            for j, ent in enumerate(se.entities):
                def _uv_to_world(uv):
                    return (se.plane_origin
                            + float(uv[0]) * se.plane_x_axis
                            + float(uv[1]) * se.plane_y_axis)
                if isinstance(ent, LineEntity):
                    pts3d = np.array([_uv_to_world(ent.p0),
                                      _uv_to_world(ent.p1)], dtype=np.float32)
                elif isinstance(ent, ArcEntity):
                    pts3d = np.array([_uv_to_world(p)
                                      for p in ent.tessellate(64)], dtype=np.float32)
                else:
                    continue
                key = _sketch_key(i, j)
                self._se[key]   = [project_fn(pts3d)]
                self._se3d[key] = [pts3d]

                if isinstance(ent, LineEntity):
                    endpoints.append(_uv_to_world(ent.p0))
                    endpoints.append(_uv_to_world(ent.p1))
                elif isinstance(ent, ArcEntity):
                    tess = ent.tessellate(2)
                    endpoints.append(_uv_to_world(tess[0]))
                    endpoints.append(_uv_to_world(tess[-1]))

            if endpoints:
                vtx_3d = np.array(endpoints, dtype=np.float32)
                vtx_2d = project_fn(vtx_3d)
                self._sv[sketch_vtx_key(i)]   = vtx_2d
                self._sv3d[sketch_vtx_key(i)] = vtx_3d

    def _add_active_sketch_edges(self, sketch, project_fn):
        """
        Project LineEntity/ArcEntity objects from the active (uncommitted) sketch
        into the hover cache.  Uses history_idx = -1 as a sentinel so
        parse_sketch_key returns (-1, entity_idx) and the viewport can
        distinguish active vs committed sketch edges.
        """
        from cad.sketch import LineEntity, ArcEntity
        for j, ent in enumerate(sketch.entities):
            if isinstance(ent, LineEntity):
                p0_world = sketch.plane.to_3d(float(ent.p0[0]), float(ent.p0[1]))
                p1_world = sketch.plane.to_3d(float(ent.p1[0]), float(ent.p1[1]))
                pts3d = np.array([p0_world, p1_world], dtype=np.float32)
            elif isinstance(ent, ArcEntity):
                pts3d = np.array([sketch.plane.to_3d(float(p[0]), float(p[1]))
                                  for p in ent.tessellate(64)], dtype=np.float32)
            else:
                continue
            key = _sketch_key(-1, j)
            self._se[key]   = [project_fn(pts3d)]
            self._se3d[key] = [pts3d]

    # ------------------------------------------------------------------
    # Occlusion
    # ------------------------------------------------------------------

    def _visible(self, world_pt: np.ndarray) -> bool:
        if self._eye is None or self._meshes is None:
            return True
        return not _ray_hits_anything(self._eye, world_pt,
                                      self._meshes, self._workspace)

    def _edge_visible(self, body_id: str, ei: int, world_pt: np.ndarray) -> bool:
        """
        Visibility check for edges. Uses adjacent face normals as a fast reject:
        if every adjacent face points away from the camera, the edge is on the
        back of its own body and cannot be visible. Otherwise we still ray-cast,
        because a front-facing edge on this body can be occluded by other bodies.
        """
        if self._eye is None:
            return True

        fn_list = self._se_fn.get(body_id)
        if fn_list is not None and ei < len(fn_list):
            face_normals = fn_list[ei]
            if len(face_normals) > 0:
                to_eye = self._eye - world_pt
                dist = float(np.linalg.norm(to_eye))
                if dist > 1e-9:
                    to_eye_n = to_eye / dist
                    dots = face_normals.astype(np.float64) @ to_eye_n
                    if float(dots.max()) < -0.01:
                        return False

        return self._visible(world_pt)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def vertex_at(self, x: float, y: float) -> tuple[str | None, int | None]:
        """Closest visible topo vertex within VERTEX_HOVER_RADIUS pixels."""
        if not self._ready:
            return None, None

        candidates: list[tuple[float, str, int]] = []
        for body_id, sv in self._sv.items():
            if len(sv) == 0:
                continue
            dists = np.hypot(sv[:, 0] - x, sv[:, 1] - y)
            mask  = dists < VERTEX_HOVER_RADIUS
            for i in np.where(mask)[0]:
                candidates.append((float(dists[i]), body_id, int(i)))

        if not candidates:
            return None, None

        candidates.sort()
        for dist, body_id, i in candidates:
            wp = self._sv3d[body_id][i].astype(np.float64)
            if self._visible(wp):
                return body_id, i

        return None, None

    def edge_at(self, x: float, y: float) -> tuple[str | None, int | None]:
        """
        Closest visible edge within EDGE_HOVER_RADIUS pixels.

        The returned body_id may be a sketch key — use parse_sketch_key()
        to detect and decode it.  The edge_idx is always 0 for sketch edges
        (each sketch edge has its own key with a single segment).
        """
        if not self._ready:
            return None, None

        candidates: list[tuple[float, str, int, np.ndarray]] = []

        for body_id, sedges in self._se.items():
            for ei, se in enumerate(sedges):
                if len(se) < 2:
                    continue
                a      = se[:-1]
                b      = se[1:]
                ab     = b - a
                len_sq = (ab * ab).sum(axis=1)
                t      = ((np.array([[x, y]]) - a) * ab).sum(axis=1) / \
                         np.where(len_sq > 1e-9, len_sq, 1e-9)
                t       = np.clip(t, 0.0, 1.0)
                closest = a + t[:, np.newaxis] * ab
                dists   = np.hypot(closest[:, 0] - x, closest[:, 1] - y)
                d       = float(dists.min())
                if d < EDGE_HOVER_RADIUS:
                    best_seg = int(np.argmin(dists))
                    pts3d    = self._se3d[body_id][ei]
                    t_val = float(t[best_seg])
                    if best_seg + 1 < len(pts3d):
                        wp3d = (pts3d[best_seg].astype(np.float64) * (1 - t_val) +
                                pts3d[best_seg + 1].astype(np.float64) * t_val)
                    else:
                        wp3d = pts3d[best_seg].astype(np.float64)
                    candidates.append((d, body_id, ei, wp3d))

        if not candidates:
            return None, None

        candidates.sort(key=lambda c: c[0])
        for d, body_id, ei, wp3d in candidates:
            if self._edge_visible(body_id, ei, wp3d):
                return body_id, ei

        return None, None

    def vertex_world_pos(self, body_id: str, vertex_idx: int) -> np.ndarray | None:
        """Return the 3D world position of a cached vertex, or None."""
        pts = self._sv3d.get(body_id)
        if pts is None or vertex_idx >= len(pts):
            return None
        return pts[vertex_idx].astype(np.float64)

    def clear(self):
        self._sv.clear();   self._sv3d.clear()
        self._se.clear();   self._se3d.clear();  self._se_fn.clear()
        self._eye       = None
        self._meshes    = None
        self._workspace = None
        self._ready     = False
