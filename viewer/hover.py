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

# Arc/spline tessellation for hover picking only. Deliberately decoupled from
# prefs.sketch_curve_segments (a *display* fidelity setting that can be 256+):
# a 32-segment chord approximation is well inside EDGE_HOVER_RADIUS, and the
# hover cache is rebuilt every frame so point count directly costs FPS.
HOVER_CURVE_SEGMENTS = 32

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
        # Flat batch of all sketch-entity segments, so edge_at() can answer
        # with one vectorized query instead of a per-entity Python loop.
        self._sk_stage: list[tuple[str, np.ndarray]] = []
        self._skb_keys: list[str] = []
        self._skb_a2 = np.zeros((0, 2))
        self._skb_b2 = np.zeros((0, 2))
        self._skb_a3 = np.zeros((0, 3), dtype=np.float32)
        self._skb_b3 = np.zeros((0, 3), dtype=np.float32)
        self._skb_seg_ent = np.zeros(0, dtype=np.intp)
        # Signature of the last rebuild's inputs. rebuild() re-projects
        # everything (thousands of edge polylines) which is expensive; on a
        # static scene the projection is identical frame-to-frame, so we skip
        # the work when nothing that affects it has changed.
        self._sig = None

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    @staticmethod
    def _sketch_sig(history, active_sketch, editing_sketch_idx):
        """Cheap signature of the sketch/history inputs to rebuild().

        Catches committed-sketch visibility toggles, cursor moves, and live
        edits to the active sketch (drags mutate entity coords in place, so we
        fold endpoint coordinates in, not just the entity count)."""
        parts = []
        if history is not None:
            parts.append(("cur", history.cursor, editing_sketch_idx))
            for i, entry in enumerate(history.entries):
                if i > history.cursor or entry.operation != "sketch":
                    continue
                se = entry.params.get("sketch_entry")
                if se is None:
                    continue
                parts.append((i, se.visible, len(se.entities)))
        if active_sketch is not None:
            ents = active_sketch.entities
            parts.append(("active", id(active_sketch), len(ents)))
            # Fold a coordinate digest so in-place drags invalidate the cache.
            acc = 0.0
            for e in ents:
                for attr in ("p0", "p1", "center", "radius",
                             "start_angle", "end_angle"):
                    v = getattr(e, attr, None)
                    if v is None:
                        continue
                    if isinstance(v, (tuple, list, np.ndarray)):
                        acc += float(v[0]) + float(v[1])
                    else:
                        acc += float(v)
            parts.append(("acc", round(acc, 6)))
        return tuple(parts)

    def rebuild(self, meshes, workspace, modelview, projection, viewport, dpr,
                camera_eye: np.ndarray = None, history=None,
                active_sketch=None, editing_sketch_idx=None,
                occlusion_fn=None):
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
        self._occlusion_fn = occlusion_fn

        if modelview is None:
            self._ready = False
            return

        mv  = np.array(modelview,  dtype=np.float64).reshape(4, 4)
        prj = np.array(projection, dtype=np.float64).reshape(4, 4)

        # Skip the (expensive) re-projection when nothing that affects it has
        # changed since the last rebuild. The matrices capture every camera
        # move; the mesh id/visibility set captures body add/remove/hide; the
        # sketch digest captures committed toggles and live drags.
        # Per-body: (id, visibility, mesh object identity). The mesh id changes
        # whenever a body is re-meshed (edit/undo), which reprojects its edges;
        # visibility gates whether it's projected at all.
        mesh_sig = tuple(sorted(
            (bid,
             bool(workspace.bodies.get(bid).visible)
                  if workspace.bodies.get(bid) is not None else True,
             id(mesh))
            for bid, mesh in meshes.items()))
        sig = (mv.tobytes(), prj.tobytes(),
               float(viewport[2]), float(viewport[3]), float(dpr),
               mesh_sig,
               self._sketch_sig(history, active_sketch, editing_sketch_idx))
        if self._ready and sig == self._sig:
            # Occlusion queries need a current eye even when we skip rebuild.
            if camera_eye is not None:
                self._eye = camera_eye.copy()
            return
        self._sig = sig

        self._ready = False
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

            se3d_list = list(mesh.topo_edges)
            fn_list   = getattr(mesh, 'topo_edge_face_normals', [])
            sefn_list = [fn_list[i] if i < len(fn_list)
                         else np.zeros((0, 3), dtype=np.float32)
                         for i in range(len(se3d_list))]
            # Project every edge of this body in a single batched matmul rather
            # than one _project() call per edge — with thousands of imported
            # edges the per-call overhead dominated rebuild().
            if se3d_list:
                counts  = [len(e) for e in se3d_list]
                all_pts = np.concatenate(se3d_list)
                all_2d  = _project(all_pts)
                offs    = np.concatenate([[0], np.cumsum(counts)])
                se_list = [all_2d[offs[i]:offs[i + 1]]
                           for i in range(len(se3d_list))]
            else:
                se_list = []
            self._se[body_id]   = se_list
            self._se3d[body_id] = se3d_list
            self._se_fn[body_id] = sefn_list

        # ------------------------------------------------------------------
        # Committed sketch line entities
        # Each LineEntity becomes a 2-point edge in the hover cache,
        # keyed by the synthetic sketch key so the viewport can identify it.
        # ------------------------------------------------------------------
        self._sk_stage = []
        if history is not None:
            self._add_sketch_edges(history, _project,
                                   skip_idx=editing_sketch_idx)

        if active_sketch is not None:
            self._add_active_sketch_edges(active_sketch, _project)

        self._finalize_sketch_batch(_project)

        self._ready = True

    @staticmethod
    def _entity_uv_polys(entities):
        """Return [(entity_idx, (P,2) uv polyline)] for hoverable entities.

        All arcs are tessellated in one batched trig evaluation — with traced
        sketches (hundreds of arcs) per-arc tessellation dominates rebuild().
        """
        from cad.sketch import LineEntity, ArcEntity, SplineEntity
        n_seg = HOVER_CURVE_SEGMENTS
        out = []
        arc_js, arc_params = [], []
        for j, ent in enumerate(entities):
            if isinstance(ent, LineEntity):
                out.append((j, np.array([ent.p0, ent.p1], dtype=np.float64)))
            elif isinstance(ent, ArcEntity):
                arc_js.append(j)
                arc_params.append((ent.center[0], ent.center[1], ent.radius,
                                   ent.start_angle, ent.end_angle))
            elif isinstance(ent, SplineEntity):
                out.append((j, ent.tessellate_np(n_seg)))
        if arc_js:
            P  = np.array(arc_params, dtype=np.float64)
            t  = np.linspace(0.0, 1.0, n_seg + 1)
            ang = P[:, 3:4] + (P[:, 4:5] - P[:, 3:4]) * t     # (A, n+1)
            r   = P[:, 2:3]
            polys = np.stack([P[:, 0:1] + r * np.cos(ang),
                              P[:, 1:2] + r * np.sin(ang)], axis=2)
            out.extend(zip(arc_js, polys))
        return out

    def _stage_entry_edges(self, history_idx, entities,
                           origin, x_axis, y_axis):
        """Convert one sketch's entities to world polylines and stage them
        for the batched projection pass.  Returns per-line/arc endpoint world
        points (for the sketch-vertex cache)."""
        from cad.sketch import LineEntity, ArcEntity
        polys = list(self._entity_uv_polys(entities))
        if not polys:
            return []
        counts  = [len(p) for _, p in polys]
        all_uv  = np.concatenate([p for _, p in polys])
        world   = (origin
                   + all_uv[:, 0:1] * x_axis
                   + all_uv[:, 1:2] * y_axis).astype(np.float32)
        offsets = np.concatenate([[0], np.cumsum(counts)])
        endpoints = []
        for k, (j, _) in enumerate(polys):
            pts3d = world[offsets[k]:offsets[k + 1]]
            self._sk_stage.append((_sketch_key(history_idx, j), pts3d))
            if isinstance(entities[j], (LineEntity, ArcEntity)):
                endpoints.append(pts3d[0])
                endpoints.append(pts3d[-1])
        return endpoints

    def _finalize_sketch_batch(self, project_fn):
        """Project every staged sketch polyline in one batch and build the
        flat segment arrays edge_at() queries."""
        stage = self._sk_stage
        self._skb_keys = []
        if not stage:
            self._skb_a2 = np.zeros((0, 2))
            self._skb_b2 = np.zeros((0, 2))
            self._skb_a3 = np.zeros((0, 3), dtype=np.float32)
            self._skb_b3 = np.zeros((0, 3), dtype=np.float32)
            self._skb_seg_ent = np.zeros(0, dtype=np.intp)
            return
        counts  = np.array([len(p) for _, p in stage])
        pts3d   = np.concatenate([p for _, p in stage])
        pts2d   = project_fn(pts3d)
        offsets = np.concatenate([[0], np.cumsum(counts)])
        ent_id  = np.repeat(np.arange(len(stage)), counts)
        for k, (key, _) in enumerate(stage):
            s, e = offsets[k], offsets[k + 1]
            self._se[key]   = [pts2d[s:e]]
            self._se3d[key] = [pts3d[s:e]]
            self._skb_keys.append(key)
        # Segments between consecutive points of the same entity.
        valid = ent_id[:-1] == ent_id[1:]
        self._skb_a2 = pts2d[:-1][valid]
        self._skb_b2 = pts2d[1:][valid]
        self._skb_a3 = pts3d[:-1][valid]
        self._skb_b3 = pts3d[1:][valid]
        self._skb_seg_ent = ent_id[:-1][valid]

    def _add_sketch_edges(self, history, project_fn, skip_idx=None):
        """Stage committed sketch LineEntity/ArcEntity objects for the hover cache."""
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
            endpoints = self._stage_entry_edges(
                i, se.entities,
                se.plane_origin, se.plane_x_axis, se.plane_y_axis)
            if endpoints:
                vtx_3d = np.array(endpoints, dtype=np.float32)
                vtx_2d = project_fn(vtx_3d)
                self._sv[sketch_vtx_key(i)]   = vtx_2d
                self._sv3d[sketch_vtx_key(i)] = vtx_3d

    def _add_active_sketch_edges(self, sketch, project_fn):
        """
        Stage LineEntity/ArcEntity objects from the active (uncommitted) sketch
        for the hover cache.  Uses history_idx = -1 as a sentinel so
        parse_sketch_key returns (-1, entity_idx) and the viewport can
        distinguish active vs committed sketch edges.
        """
        plane = sketch.plane
        self._stage_entry_edges(-1, sketch.entities,
                                plane.origin, plane.x_axis, plane.y_axis)

    # ------------------------------------------------------------------
    # Occlusion
    # ------------------------------------------------------------------

    def _visible(self, world_pt: np.ndarray) -> bool:
        if self._eye is None or self._meshes is None:
            return True
        # Prefer a GPU depth-buffer occlusion test if the viewport supplied one
        # (O(1) vs the CPU ray/triangle sweep over every body).
        occ = getattr(self, "_occlusion_fn", None)
        if occ is not None:
            res = occ(world_pt)
            if res is not None:
                return res
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

        # Sketch entities: one vectorized query over the flat segment batch.
        if len(self._skb_a2):
            a  = self._skb_a2
            ab = self._skb_b2 - a
            len_sq = (ab * ab).sum(axis=1)
            t = ((np.array([x, y]) - a) * ab).sum(axis=1) / \
                np.where(len_sq > 1e-9, len_sq, 1e-9)
            t = np.clip(t, 0.0, 1.0)
            closest = a + t[:, np.newaxis] * ab
            dists = np.hypot(closest[:, 0] - x, closest[:, 1] - y)
            hit_ents: dict[int, tuple[float, int]] = {}
            for si in np.where(dists < EDGE_HOVER_RADIUS)[0]:
                eid = int(self._skb_seg_ent[si])
                if eid not in hit_ents or dists[si] < hit_ents[eid][0]:
                    hit_ents[eid] = (float(dists[si]), int(si))
            for eid, (d, si) in hit_ents.items():
                t_val = float(t[si])
                wp3d = (self._skb_a3[si].astype(np.float64) * (1 - t_val) +
                        self._skb_b3[si].astype(np.float64) * t_val)
                candidates.append((d, self._skb_keys[eid], 0, wp3d))

        for body_id, sedges in self._se.items():
            if body_id.startswith(_SKETCH_KEY_PREFIX):
                continue
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
        self._sk_stage  = []
        self._skb_keys  = []
        self._skb_a2 = np.zeros((0, 2))
        self._skb_b2 = np.zeros((0, 2))
        self._skb_a3 = np.zeros((0, 3), dtype=np.float32)
        self._skb_b3 = np.zeros((0, 3), dtype=np.float32)
        self._skb_seg_ent = np.zeros(0, dtype=np.intp)
        self._sig = None
