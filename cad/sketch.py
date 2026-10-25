"""
cad/sketch.py

Sketch data model.  All coordinates in world millimetres.

Classes
-------
SketchTool      — enum of available tools (add new values here when adding tools)
LineEntity      — a committed 2-D line segment
ReferenceEntity — a projected reference edge/vertex from existing geometry
SketchPlane     — wraps a build123d Plane; ray intersection + coord conversion
SketchMode      — the live sketch session (plane, entities, active tool instance)
SketchEntry     — immutable committed snapshot stored in history

Tool logic lives in cad/sketch_tools/.
"""

from __future__ import annotations
import copy
import numpy as np
from collections import OrderedDict
from enum import Enum, auto
from build123d import Plane

# Memoized SketchEntry.build_faces() results, keyed by (geometry fingerprint,
# tol).  Entries hold OCCT faces; keep the cache small.
_FACE_BUILD_CACHE: OrderedDict = OrderedDict()
_FACE_BUILD_CACHE_MAX = 8


# ---------------------------------------------------------------------------
# Tool enum  — add a value here when adding a new tool
# ---------------------------------------------------------------------------

class SketchTool(Enum):
    NONE      = auto()
    LINE      = auto()
    CIRCLE    = auto()
    ARC       = auto()
    ARC3      = auto()   # 3-point arc
    TRIM      = auto()
    DIVIDE    = auto()
    POINT     = auto()
    OFFSET    = auto()
    FILLET    = auto()
    DIMENSION = auto()
    GEOMETRIC = auto()
    SQUARE    = auto()
    MIRROR    = auto()
    PATTERN_LINEAR   = auto()
    PATTERN_CIRCULAR = auto()
    SPLINE    = auto()
    IMPORT_IMAGE = auto()   # one-shot: trace a raster into LineEntity chains


# ---------------------------------------------------------------------------
# Entity types
# ---------------------------------------------------------------------------
#
# Every entity carries a `uid`: persisted, kept by copy.deepcopy (edit
# snapshots), fresh for anything newly constructed (trim/divide pieces get
# new ones).  Used for region identity — see SketchEntry.build_face_boundaries.

def _new_uid() -> str:
    import uuid
    return uuid.uuid4().hex[:12]


class LineEntity:
    """A 2-D line segment in sketch (u, v) coordinates (world mm)."""
    def __init__(self, p0: tuple[float, float], p1: tuple[float, float],
                 construction: bool = False):
        self.uid = _new_uid()
        self.p0 = np.array(p0, dtype=np.float64)
        self.p1 = np.array(p1, dtype=np.float64)
        # Snap metadata: (entity_idx, SnapType) if endpoint was snapped to a
        # non-endpoint position (midpoint, center, nearest, tangent). Used by
        # the constraint solver to emit coincidence constraints.
        self.p0_snap: tuple[int, object] | None = None
        self.p1_snap: tuple[int, object] | None = None
        # Construction (reference) geometry: drawn and snappable like normal
        # geometry, but excluded from profile/face building.  Toggled after
        # the fact via the construction tool — never drawn as construction.
        self.construction = bool(construction)


class PointEntity:
    """A 2-D construction point in sketch (u, v) coordinates."""
    def __init__(self, pos: tuple[float, float]):
        self.uid = _new_uid()
        self.pos = np.array(pos, dtype=np.float64)

    @property
    def p0(self) -> np.ndarray:
        return self.pos

    @property
    def p1(self) -> np.ndarray:
        return self.pos


class ArcEntity:
    """
    A 2-D circular arc in sketch (u, v) coordinates (world mm).

    Stored as center + radius + angles so the solver can own these params later.
    Angles are in radians, CCW from the U-axis.  start_angle < end_angle always
    (the arc sweeps CCW from start_angle to end_angle).
    """
    def __init__(self, center: tuple[float, float], radius: float,
                 start_angle: float, end_angle: float,
                 construction: bool = False):
        self.uid         = _new_uid()
        self.center      = np.array(center, dtype=np.float64)
        self.radius      = float(radius)
        self.start_angle = float(start_angle)
        self.end_angle   = float(end_angle)
        # Construction (reference) geometry — excluded from profile/face
        # building.  See LineEntity.construction.
        self.construction = bool(construction)

    @property
    def p0(self) -> np.ndarray:
        """Start point of the arc."""
        return self.center + self.radius * np.array(
            [np.cos(self.start_angle), np.sin(self.start_angle)])

    @property
    def p1(self) -> np.ndarray:
        """End point of the arc."""
        return self.center + self.radius * np.array(
            [np.cos(self.end_angle), np.sin(self.end_angle)])

    @property
    def mid_point(self) -> np.ndarray:
        """True mid-arc point."""
        mid = (self.start_angle + self.end_angle) * 0.5
        return self.center + self.radius * np.array([np.cos(mid), np.sin(mid)])

    def tessellate_np(self, n: int = 64) -> np.ndarray:
        """Return an (n+1, 2) array of points evenly distributed along the arc."""
        angles = np.linspace(self.start_angle, self.end_angle, n + 1)
        return self.center + self.radius * np.stack(
            [np.cos(angles), np.sin(angles)], axis=1)

    def tessellate(self, n: int = 64) -> list[np.ndarray]:
        """Return n+1 points evenly distributed along the arc."""
        return list(self.tessellate_np(n))


class SplineEntity:
    """
    A 2-D spline that interpolates (passes through) a sequence of fit points,
    in sketch (u, v) coordinates (world mm).

    points : list of (u, v) — the interpolation points, in order.
    closed : if True, the curve is periodic and returns to points[0].

    The actual curve is an OCCT B-spline fit through the points; we keep only
    the fit points as the parametric data (the solver can own them later) and
    derive geometry on demand.
    """
    def __init__(self, points, closed: bool = False,
                 construction: bool = False):
        self.uid    = _new_uid()
        self.points = [np.array(p, dtype=np.float64) for p in points]
        self.closed = bool(closed)
        self.construction = bool(construction)

    @property
    def p0(self) -> np.ndarray:
        return self.points[0]

    @property
    def p1(self) -> np.ndarray:
        # A closed spline ends where it began.
        return self.points[0] if self.closed else self.points[-1]

    def _occ_curve(self):
        """Build (and return) the Geom_BSplineCurve through the fit points in the
        sketch UV plane (z=0).  Returns None if it can't be built."""
        from OCP.TColgp import TColgp_Array1OfPnt
        from OCP.gp import gp_Pnt
        from OCP.GeomAPI import GeomAPI_PointsToBSpline
        pts = self.points
        if self.closed and len(pts) >= 1:
            pts = list(pts) + [pts[0]]
        if len(pts) < 2:
            return None
        arr = TColgp_Array1OfPnt(1, len(pts))
        for i, p in enumerate(pts, start=1):
            arr.SetValue(i, gp_Pnt(float(p[0]), float(p[1]), 0.0))
        try:
            return GeomAPI_PointsToBSpline(arr).Curve()
        except Exception:
            return None

    def tessellate(self, n: int = 64) -> list[np.ndarray]:
        """Return ~n+1 points along the spline (UV).  Falls back to the raw
        fit-point polyline if the B-spline can't be built."""
        curve = self._occ_curve()
        if curve is None:
            pts = list(self.points)
            if self.closed and pts:
                pts = pts + [pts[0]]
            return pts
        u0, u1 = curve.FirstParameter(), curve.LastParameter()
        out = []
        for t in np.linspace(u0, u1, max(n, 2)):
            p = curve.Value(float(t))
            out.append(np.array([p.X(), p.Y()], dtype=np.float64))
        return out

    def tessellate_np(self, n: int = 64) -> np.ndarray:
        """tessellate() as an (m, 2) array."""
        return np.asarray(self.tessellate(n), dtype=np.float64)


class ReferenceEntity:
    """
    A projected reference polyline — created by the Include tool.

    points      : list of (u, v) np.ndarray — projected onto sketch plane
                  This is a baked cache; updated during parametric replay.
    source_type : 'edge' | 'vertex' | 'sketch_edge'
    occ_edges   : list of TopoDS_Edge | None
                  Original OCCT edges preserved for exact face construction.
    source      : EdgeSource | None
                  Stable parametric reference used to re-derive points and
                  occ_edges during replay.  None for vertices and legacy data.
    """
    def __init__(self, points: list[np.ndarray], source_type: str = 'edge',
                 occ_edges=None, source=None):
        self.uid         = _new_uid()
        self.points      = [np.array(p, dtype=np.float64) for p in points]
        self.source_type = source_type
        self.occ_edges   = occ_edges
        self.source      = source          # EdgeSource | None

    def __deepcopy__(self, memo):
        # TopoDS_Edge is not picklable — share by reference (OCCT shapes are
        # immutable value types so aliasing is safe).
        import copy
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        result.uid         = self.uid
        result.points      = copy.deepcopy(self.points, memo)
        result.source_type = self.source_type
        result.occ_edges   = self.occ_edges   # shared reference, not copied
        result.source      = copy.deepcopy(self.source, memo)
        return result


# ---------------------------------------------------------------------------
# SketchPlane
# ---------------------------------------------------------------------------

class SketchPlane:
    """
    Wraps a build123d Plane.  Exposes origin and axes in world mm space,
    matching the coordinate system used by mesh VBOs and the camera.
    """

    def __init__(self, b3d_plane: Plane):
        self._plane = b3d_plane

        wo = b3d_plane.origin
        self.origin = np.array([wo.X, wo.Y, wo.Z], dtype=np.float64)

        xd = b3d_plane.x_dir
        yd = b3d_plane.y_dir
        zd = b3d_plane.z_dir

        self.x_axis = np.array([xd.X, xd.Y, xd.Z], dtype=np.float64)
        self.y_axis = np.array([yd.X, yd.Y, yd.Z], dtype=np.float64)
        self.normal = np.array([zd.X, zd.Y, zd.Z], dtype=np.float64)

        self.x_axis /= np.linalg.norm(self.x_axis)
        self.y_axis /= np.linalg.norm(self.y_axis)
        self.normal  /= np.linalg.norm(self.normal)

    def ray_intersect(self, ray_origin: np.ndarray, ray_dir: np.ndarray
                      ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """
        Intersect a ray with this plane (world mm).
        Returns (pt3d, pt2d) or (None, None) if parallel / behind.
        """
        denom = float(np.dot(ray_dir, self.normal))
        if abs(denom) < 1e-9:
            return None, None
        t = float(np.dot(self.origin - ray_origin, self.normal)) / denom
        if t < 0:
            return None, None
        pt3d  = ray_origin + t * ray_dir
        delta = pt3d - self.origin
        u = float(np.dot(delta, self.x_axis))
        v = float(np.dot(delta, self.y_axis))
        return pt3d, np.array([u, v], dtype=np.float64)

    def to_3d(self, u: float, v: float) -> np.ndarray:
        """Sketch (u, v) in mm → world 3-D point."""
        return self.origin + u * self.x_axis + v * self.y_axis

    def project_point(self, world_pt: np.ndarray) -> np.ndarray:
        """Orthographic projection of a world point → (u, v) sketch coords."""
        delta = np.array(world_pt, dtype=np.float64) - self.origin
        u = float(np.dot(delta, self.x_axis))
        v = float(np.dot(delta, self.y_axis))
        return np.array([u, v], dtype=np.float64)


# ---------------------------------------------------------------------------
# SketchMode
# ---------------------------------------------------------------------------

class SketchMode:
    """
    Live sketch session.  Created on double-click, discarded on ESC/commit.

    Owns a SnapEngine — snap resolution happens here so every tool gets
    it automatically.  Tools receive SnapResult, never raw pt2d.
    """

    def __init__(self, b3d_plane: Plane, body_id: str, face_idx: int,
                 plane_source=None):
        self.plane        = SketchPlane(b3d_plane)
        self.body_id      = body_id
        self.face_idx     = face_idx
        self.plane_source = plane_source   # SketchPlaneSource | None
        self.entities: list = []
        self.constraints: list = []

        # Per-sketch undo stack — each entry is a deep-copy of (entities, constraints)
        # taken before a mutation.  Ctrl+Z pops and restores.
        self._entity_snapshots: list = []
        self._redo_snapshots:   list = []

        # Active tool enum value (for status bar / overlay checks)
        self.tool = SketchTool.NONE
        # Active tool instance — None when no tool selected
        self._active_tool = None

        # Snap engine — shared across all tools in this session
        from cad.sketch_tools.snap import SnapEngine
        import numpy as np
        self.snap = SnapEngine()
        # World-plane sketches: origin (0,0) in UV space is always snappable
        if isinstance(plane_source, __import__('cad.plane_ref',
                fromlist=['WorldPlaneSource']).WorldPlaneSource):
            self.snap.extra_points = [np.array([0.0, 0.0])]

        # Last resolved snap result — read by overlay for indicator drawing
        self.last_snap: object = None   # SnapResult | None

        # Entity indices the next tool should operate on (set by the viewport
        # when activating a selection-driven tool like Mirror).  Consumed and
        # cleared on tool activation.
        self.pending_selection: list[int] = []

    # ------------------------------------------------------------------
    # Tool management
    # ------------------------------------------------------------------

    def set_tool(self, tool_enum: SketchTool):
        """Activate a tool by enum value.  Creates a fresh tool instance.

        If pending_selection was set (a selection-driven tool like Mirror), it
        is handed to the new tool via on_activate() and then cleared.
        """
        from cad.sketch_tools import TOOLS
        self.tool = tool_enum
        cls = TOOLS.get(tool_enum)
        self._active_tool = cls() if cls is not None else None
        if self._active_tool is not None and self.pending_selection:
            on_activate = getattr(self._active_tool, "on_activate", None)
            if on_activate is not None:
                on_activate(self, list(self.pending_selection))
        self.pending_selection = []

    def cancel_tool(self):
        """ESC within a tool — discard in-progress state and return to NONE.
        Stays in sketch mode so the user can select geometry, measure, etc."""
        if self._active_tool is not None:
            self._active_tool.cancel()
        self.tool = SketchTool.NONE
        self._active_tool = None
        self.last_snap = None
        self.snap.anchor_pt = None

    # ------------------------------------------------------------------
    # Per-sketch undo
    # ------------------------------------------------------------------

    def push_undo_snapshot(self):
        """Save entities + constraints before a mutation so Ctrl+Z can restore."""
        import copy
        self._entity_snapshots.append(
            (copy.deepcopy(self.entities), copy.deepcopy(self.constraints))
        )
        self._redo_snapshots.clear()

    def undo_entity(self) -> bool:
        """Restore the most recent undo snapshot. Returns True if anything was undone."""
        if not self._entity_snapshots:
            return False
        import copy
        self._redo_snapshots.append(
            (copy.deepcopy(self.entities), copy.deepcopy(self.constraints))
        )
        snapshot = self._entity_snapshots.pop()
        if isinstance(snapshot, tuple):
            self.entities, self.constraints = snapshot
        else:
            self.entities = snapshot  # backwards compat with old single-list snapshots
        if self._active_tool is not None:
            self._active_tool.cancel()
        return True

    def redo_entity(self) -> bool:
        """Re-apply the most recently undone snapshot. Returns True if anything was redone."""
        if not self._redo_snapshots:
            return False
        import copy
        self._entity_snapshots.append(
            (copy.deepcopy(self.entities), copy.deepcopy(self.constraints))
        )
        snapshot = self._redo_snapshots.pop()
        if isinstance(snapshot, tuple):
            self.entities, self.constraints = snapshot
        else:
            self.entities = snapshot
        return True

    # ------------------------------------------------------------------
    # Input forwarding — viewport calls these every frame
    # ------------------------------------------------------------------

    def handle_mouse_move(self, ray_origin: np.ndarray, ray_dir: np.ndarray,
                          camera_distance: float = 100.0):
        _, pt2d = self.plane.ray_intersect(ray_origin, ray_dir)
        if pt2d is None:
            self.last_snap = None
            return
        snap_result = self.snap.resolve(pt2d, self.entities,
                                        self.plane, camera_distance)
        self.last_snap = snap_result
        if self._active_tool is not None:
            self._active_tool.handle_mouse_move(snap_result, self)

    def handle_click(self, ray_origin: np.ndarray, ray_dir: np.ndarray,
                     camera_distance: float = 100.0) -> bool:
        _, pt2d = self.plane.ray_intersect(ray_origin, ray_dir)
        if pt2d is None or self._active_tool is None:
            return False
        if (self.last_snap is not None and
                np.linalg.norm(self.last_snap.cursor_raw - pt2d) < 1e-6):
            snap_result = self.last_snap
        else:
            snap_result = self.snap.resolve(pt2d, self.entities,
                                            self.plane, camera_distance)
            self.last_snap = snap_result
        result = self._active_tool.handle_click(snap_result, self)
        self.snap.consume_declared()
        return result

    # ------------------------------------------------------------------
    # Overlay helpers — read by sketch_overlay.py
    # ------------------------------------------------------------------

    @property
    def _cursor_2d(self) -> np.ndarray | None:
        """Snapped cursor position for the overlay."""
        if self._active_tool is not None:
            return self._active_tool.cursor_2d
        return None

    @property
    def _line_start(self) -> np.ndarray | None:
        """Line preview start point — only valid when LINE tool is active."""
        from cad.sketch_tools.line import LineTool
        if isinstance(self._active_tool, LineTool):
            return self._active_tool.line_start
        return None

    @property
    def is_empty(self) -> bool:
        return len(self.entities) == 0


# ---------------------------------------------------------------------------
# UV-space helpers
# ---------------------------------------------------------------------------

def _uv_point_in_loop(pt: tuple[float, float],
                      loop: list[tuple[float, float]]) -> bool:
    """
    2-D ray-casting point-in-polygon test.
    Returns True if `pt` is strictly inside the polygon defined by `loop`.
    """
    x, y = pt
    n = len(loop)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = loop[i]
        xj, yj = loop[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


# ---------------------------------------------------------------------------
# SketchConstraint
# ---------------------------------------------------------------------------

class SketchConstraint:
    """
    One parametric constraint on sketch entities.

    type         : 'distance' | 'diameter' | 'parallel' | ... etc.
    indices      : tuple of entity indices the constraint applies to
    value        : numeric value (mm for distance/diameter; 0 for geometric)
    label_offset : for 'distance'  — perpendicular offset of label from line (mm)
                   for 'diameter'  — radial offset of label beyond circle edge (mm)
    label_angle  : for 'diameter' only — angle (radians) of the diameter chord
                   measured from the +U axis.  None = default (0 = horizontal).
    expr         : parameter expression driving *value* ("width/2"), or None.
                   value stays the baked result; replay recomputes it.
    expr_unit    : unit bare numbers in expr are read in (see cad/params.py).
    """
    def __init__(self, type: str, indices: tuple, value: float,
                 label_offset: float | None = None,
                 label_angle:  float | None = None,
                 expr: str | None = None, expr_unit: str = "mm"):
        self.type         = type
        self.indices      = tuple(indices)
        self.value        = float(value)
        self.label_offset: float | None = label_offset
        self.label_angle:  float | None = label_angle
        self.expr:         str | None   = expr
        self.expr_unit:    str          = expr_unit


# Constraints that pin a line's *length*. Meaningless once the line is split
# into pieces, so splice_entities drops them rather than forcing one piece
# back to the original length.
# ---------------------------------------------------------------------------
# Included geometry follow-up
# ---------------------------------------------------------------------------
#
# Includes (ReferenceEntity) enter the solver as FIXED geometry (solver
# group 1, never solved for): user constraints may reference them
# (parallel / perpendicular / equal / symmetric axis on a straight include),
# and drawn points touching them are held there by hard coincidences.
#
# Touching is inferred, not stored: include_attachments() reads where each
# drawn point sits on an include.  While editing, that's the current
# coordinates.  On replay the includes move first, so replay infers before
# re-projecting and hands the result to the solve (attachment_targets gives
# the new spots, used as solver start points).  Needs no new metadata, so
# files saved before this existed get the same behaviour.

_ATTACH_TOL = 1e-3


def _drawn_points(entities):
    for i, ent in enumerate(entities):
        if isinstance(ent, LineEntity):
            yield (i, 'p0'), ent.p0
            yield (i, 'p1'), ent.p1
        elif isinstance(ent, ArcEntity):
            yield (i, 'center'), ent.center
            yield (i, 'p0'), ent.p0
            yield (i, 'p1'), ent.p1


def _fit_circle(pts: np.ndarray):
    """(center, radius) if pts lie on a circle (tessellated circle include),
    else None.  Algebraic least-squares fit, then a residual check."""
    if len(pts) < 5:
        return None
    A = np.column_stack([pts[:, 0], pts[:, 1], np.ones(len(pts))])
    b = (pts ** 2).sum(axis=1)
    try:
        (a, c, d), *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    center = np.array([a / 2, c / 2])
    r2 = d + center @ center
    if r2 <= 0:
        return None
    r = float(np.sqrt(r2))
    if np.abs(np.linalg.norm(pts - center, axis=1) - r).max() > 1e-6 * max(r, 1.0):
        return None
    return center, r


def _ref_circle(se, ref):
    """(center_uv, radius) of a circular include, or None.  Arc/circle
    includes keep only their endpoints in `points`; the curve is in
    occ_edges (projected with se's current plane).  Tessellated circles
    (no single OCC circle) are fitted from the polyline."""
    edges = [e for e in (ref.occ_edges or []) if e is not None]
    if len(edges) == 1:
        from OCP.BRepAdaptor import BRepAdaptor_Curve
        from OCP.GeomAbs import GeomAbs_Circle
        a = BRepAdaptor_Curve(edges[0])
        if a.GetType() == GeomAbs_Circle:
            circ = a.Circle()
            ax = circ.Axis().Direction()
            if abs(abs(float(np.dot([ax.X(), ax.Y(), ax.Z()], se.plane_normal))) - 1) > 1e-6:
                return None                  # tilted: projects to an ellipse
            loc = circ.Location()
            d = np.array([loc.X(), loc.Y(), loc.Z()]) - se.plane_origin
            return (np.array([d @ se.plane_x_axis, d @ se.plane_y_axis]),
                    float(circ.Radius()))
    return _fit_circle(np.asarray(ref.points, dtype=np.float64))


def include_attachments(se) -> list:
    """Drawn points lying on include geometry, as
    [(key, ref_idx, kind, data, n_points)] with kind:
      'v' vertex k of the include polyline          data = k
      's' on segment k..k+1 at fraction t           data = (k, t)
      'c' centre of a circular include              data = None
      'a' on a circular include                     data = UV angle
    """
    refs = [(ri, r) for ri, r in enumerate(se.entities)
            if isinstance(r, ReferenceEntity) and len(r.points) >= 1]
    if not refs:
        return []
    circles = {}
    out = []
    for key, pt in _drawn_points(se.entities):
        for ri, ref in refs:
            pts = np.asarray(ref.points, dtype=np.float64)
            # Circular includes attach by centre / angle only: their
            # tessellation seam moves when OCC rebuilds the edge, so a
            # vertex index would land somewhere else after replay.
            if ri not in circles:
                circles[ri] = _ref_circle(se, ref) if len(pts) >= 2 else None
            if circles[ri] is not None:
                hit = _attach_on_circle(pt, circles[ri])
            else:
                hit = _attach_on_polyline(pt, pts)
            if hit is not None:
                out.append((key, ri, *hit, len(pts)))
                break
    return out


def _attach_on_circle(pt, circ):
    if circ is None:
        return None
    center, r = circ
    d = float(np.linalg.norm(pt - center))
    if d < _ATTACH_TOL:
        return 'c', None
    if abs(d - r) < _ATTACH_TOL:
        return 'a', float(np.arctan2(pt[1] - center[1], pt[0] - center[0]))
    return None


def _attach_on_polyline(pt, pts):
    tol = _ATTACH_TOL
    d = np.linalg.norm(pts - pt, axis=1)
    k = int(d.argmin())
    if d[k] < tol:
        return 'v', k
    if len(pts) >= 2:
        a, b = pts[:-1], pts[1:]
        ab = b - a
        L2 = (ab ** 2).sum(axis=1)
        t = np.where(L2 > 1e-24, ((pt - a) * ab).sum(axis=1) / np.maximum(L2, 1e-24), 0.0)
        t = t.clip(0.0, 1.0)
        dist = np.linalg.norm(a + t[:, None] * ab - pt, axis=1)
        k = int(dist.argmin())
        if dist[k] < tol:
            return 's', (k, float(t[k]))
    return None


def attachment_targets(se, attachments) -> dict:
    """{key: new uv} for each attachment, against the re-projected includes.
    Attachments whose include changed point count (different tessellation)
    or stopped being a circle are dropped — no follow-up, as before."""
    targets = {}
    circles = {}
    for key, ri, kind, data, n_old in attachments:
        ref = se.entities[ri]
        pts = np.asarray(ref.points, dtype=np.float64)
        if kind in ('v', 's'):
            if len(pts) != n_old:
                continue
            if kind == 'v':
                new = pts[data]
            else:
                k, t = data
                new = pts[k] + t * (pts[k + 1] - pts[k])
        else:
            if ri not in circles:
                circles[ri] = _ref_circle(se, ref)
            if circles[ri] is None:
                continue
            center, r = circles[ri]
            new = center if kind == 'c' else \
                center + r * np.array([np.cos(data), np.sin(data)])
        targets[key] = np.asarray(new, dtype=np.float64)
    return targets


_SPLIT_FUZZ = 1e-4      # mm; build_faces gap tolerance (see _build_faces_impl)


def ref_line(ent):
    """(first, last) point of a straight include, else None.  These become a
    solver line that constraints can reference.  Body-edge includes are
    tessellated (a straight edge can carry dozens of points), so
    straightness comes from the OCC edge, or collinearity without one."""
    if not isinstance(ent, ReferenceEntity) or len(ent.points) < 2:
        return None
    edges = [e for e in (ent.occ_edges or []) if e is not None]
    if edges:
        from OCP.BRepAdaptor import BRepAdaptor_Curve
        from OCP.GeomAbs import GeomAbs_Line
        if len(edges) != 1 or BRepAdaptor_Curve(edges[0]).GetType() != GeomAbs_Line:
            return None          # arc include: points are just its endpoints
    else:
        a, b = ent.points[0], ent.points[-1]
        n = float(np.linalg.norm(b - a))
        if n < 1e-9 or any(_dist_to_line(p, a, b) > 1e-6 * max(1.0, n)
                           for p in ent.points[1:-1]):
            return None
    return ent.points[0], ent.points[-1]


def line_ends(ent):
    """(p0, p1) of a drawn line or straight include, else None."""
    if isinstance(ent, LineEntity):
        return ent.p0, ent.p1
    return ref_line(ent)


_LENGTH_CONSTRAINTS = ('distance', 'equal')

# Constraints with no value: safe to drop when SolveSpace calls them
# redundant, provided they still hold afterwards (see apply_constraints).
_GEOMETRIC_CONSTRAINTS = ('horizontal', 'vertical', 'parallel',
                          'perpendicular', 'equal')


_OVERHEAD_DOF: int | None = None


def _solver_overhead_dof() -> int:
    """DOF of an empty system as _build_solver_system starts one (workplane
    + the 3-D normal arcs need) — not user freedom.  Was hard-coded as 6,
    missing the normal's 3, so no sketch ever read as fully constrained."""
    global _OVERHEAD_DOF
    if _OVERHEAD_DOF is None:
        from python_solvespace import SolverSystem
        s = SolverSystem()
        s.create_2d_base()
        s.add_normal_3d(1.0, 0.0, 0.0, 0.0)
        s.solve()
        _OVERHEAD_DOF = s.dof()
    return _OVERHEAD_DOF


def _dist_to_line(p, a, b) -> float:
    d = b - a
    n = float(np.linalg.norm(d))
    if n < 1e-12:
        return float(np.linalg.norm(p - a))
    return abs(float(d[0] * (p - a)[1] - d[1] * (p - a)[0])) / n


def splice_entities(sketch, start: int, stop: int, replacement: list) -> None:
    """
    Replace sketch.entities[start:stop] with *replacement* and keep every
    index reference valid: constraint indices and line p0_snap/p1_snap.

    Constraints and snaps are positional, so any tool that inserts, removes
    or splits entities must go through here — otherwise they silently point
    at whatever slid into the old slot.

    Refs to a replaced entity are carried to the replacement piece nearest
    the referenced geometry (constraints: the piece sharing the old p0, else
    the first piece); length constraints on split entities are dropped; refs
    with no surviving piece are dropped.
    """
    ents  = sketch.entities
    old   = ents[start:stop]
    delta = len(replacement) - (stop - start)

    def _piece_near(old_ent, pt):
        """Index (into replacement) of the piece nearest pt, or None."""
        best, best_d = None, None
        for k, p in enumerate(replacement):
            if not hasattr(p, 'p0'):
                continue
            d = min(float(np.linalg.norm(p.p0 - pt)),
                    float(np.linalg.norm(p.p1 - pt)))
            if best_d is None or d < best_d:
                best, best_d = k, d
        return best

    def _remap(j, pt=None):
        if j < start:
            return j
        if j >= stop:
            return j + delta
        old_ent = old[j - start]
        if pt is None:
            pt = getattr(old_ent, 'p0', None)
        k = _piece_near(old_ent, pt) if pt is not None else None
        return None if k is None else start + k

    # Pieces of a split line inherit the snap on whichever original endpoint
    # they still sit on.
    for o in old:
        if not isinstance(o, LineEntity):
            continue
        for p in replacement:
            if not isinstance(p, LineEntity):
                continue
            if o.p0_snap and p.p0_snap is None and np.allclose(p.p0, o.p0):
                p.p0_snap = o.p0_snap
            if o.p1_snap and p.p1_snap is None and np.allclose(p.p1, o.p1):
                p.p1_snap = o.p1_snap

    ents[start:stop] = replacement

    new_cons = []
    for con in getattr(sketch, 'constraints', []):
        touched = any(start <= j < stop for j in con.indices)
        if touched and con.type in _LENGTH_CONSTRAINTS and \
                (len(replacement) != 1 or stop - start != 1):
            continue
        idx = tuple(_remap(j) for j in con.indices)
        if None in idx:
            continue
        con.indices = idx
        new_cons.append(con)
    if hasattr(sketch, 'constraints'):
        sketch.constraints[:] = new_cons

    for ent in ents:
        if not isinstance(ent, LineEntity):
            continue
        for which in ('p0', 'p1'):
            snap = getattr(ent, f'{which}_snap')
            if snap is None:
                continue
            split = start <= snap[0] < stop and len(replacement) > 1
            if split and getattr(snap[1], 'name', '') == 'MIDPOINT':
                # Midpoint of the *original* line no longer exists.
                setattr(ent, f'{which}_snap', None)
                continue
            j = _remap(snap[0], getattr(ent, which))
            setattr(ent, f'{which}_snap', None if j is None else (j, snap[1]))


# ---------------------------------------------------------------------------
# SketchEntry  — immutable snapshot of a committed sketch
# ---------------------------------------------------------------------------

class SketchEntry:
    """
    Immutable record of a committed sketch stored on a HistoryEntry.
    shape_before == shape_after (sketches don't mutate geometry).
    """

    def __init__(self, plane_origin, plane_x_axis, plane_y_axis, plane_normal,
                 entities, body_id, face_idx, visible=True, plane_source=None,
                 constraints=None, undo_snapshots=None):
        self.plane_origin    = np.array(plane_origin, dtype=np.float64)
        self.plane_x_axis    = np.array(plane_x_axis, dtype=np.float64)
        self.plane_y_axis    = np.array(plane_y_axis, dtype=np.float64)
        self.plane_normal    = np.array(plane_normal, dtype=np.float64)
        self.entities        = entities
        self.body_id         = body_id
        self.face_idx        = face_idx
        self.visible         = visible
        self.plane_source    = plane_source   # SketchPlaneSource | None
        self.constraints     = constraints or []
        # Per-sketch undo history — list of (entities, constraints) snapshots.
        # Populated from SketchMode on commit; restored on re-entry so the user
        # can undo individual sketch operations after re-opening the sketch.
        self.undo_snapshots  = undo_snapshots or []

    @classmethod
    def from_sketch_mode(cls, sketch: SketchMode) -> SketchEntry:
        return cls(
            plane_origin    = sketch.plane.origin.copy(),
            plane_x_axis    = sketch.plane.x_axis.copy(),
            plane_y_axis    = sketch.plane.y_axis.copy(),
            plane_normal    = sketch.plane.normal.copy(),
            entities        = copy.deepcopy(sketch.entities),
            body_id         = sketch.body_id,
            face_idx        = sketch.face_idx,
            plane_source    = sketch.plane_source,
            constraints     = copy.deepcopy(getattr(sketch, 'constraints', [])),
            undo_snapshots  = copy.deepcopy(sketch._entity_snapshots),
        )

    # ------------------------------------------------------------------
    # Constraint solver
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Solver helpers
    # ------------------------------------------------------------------

    def _build_solver_system(self, pins: dict | None = None):
        """
        Build and return a populated SolverSystem with all entities registered.

        pins: {(entity_idx, which): uv} — the canonical point of each key
        starts at uv instead of its stored position (callers drag it there;
        see apply_constraints).

        Returns (sys, wp, nm3d, canon_map, slvs_pts, slvs_lines, slvs_arcs,
                 arc_center_pts, arc_start_pts, arc_end_pts, arc_dist_ents,
                 ref_geo)
        or None if python-solvespace is unavailable.

        canon_map  : (entity_idx, 'p0'|'p1'|'center') → canonical key
                     Lines use 'p0'/'p1'; arcs use 'center'/'p0'/'p1'.
        slvs_pts   : canonical key → solver point entity
        slvs_lines : line entity index → solver line entity
        slvs_arcs  : arc entity index → solver arc entity
        arc_center_pts : arc idx → solver point for arc center
        arc_start_pts  : arc idx → solver point for arc start
        arc_end_pts    : arc idx → solver point for arc end
        arc_dist_ents  : arc idx → solver distance entity (radius param) — only
                         for full circles (start_angle==0, end_angle==2π).
        ref_geo    : include idx → {'verts': [pt], 'segs': [line],
                     'center': pt|None, 'circle': circle|None}, all fixed.
        """
        try:
            from python_solvespace import SolverSystem
        except ImportError:
            return None

        slvs = SolverSystem()
        wp   = slvs.create_2d_base()
        # Quaternion for XY-plane 3-D normal (identity rotation).
        nm3d = slvs.add_normal_3d(1.0, 0.0, 0.0, 0.0)

        tol = 1e-3

        # Build canon_map: deduplicate coincident points across lines AND arcs.
        # Keys: (entity_idx, 'p0') for line p0, (entity_idx, 'p1') for line p1,
        #       (entity_idx, 'center') for arc center,
        #       (entity_idx, 'p0') for arc start, (entity_idx, 'p1') for arc end.
        canon_map: dict[tuple, tuple] = {}

        def _register_pt(key, pt):
            for k, ck in canon_map.items():
                if ck != k:
                    continue
                other_key = ck
                # Retrieve the stored point for this canonical key.
                ei, wh = other_key
                ent = self.entities[ei]
                if wh == 'p0':
                    op = ent.p0
                elif wh == 'p1':
                    op = ent.p1
                else:  # 'center'
                    op = ent.center
                if np.linalg.norm(pt - op) < tol:
                    canon_map[key] = ck
                    return
            canon_map[key] = key

        for i, ent in enumerate(self.entities):
            if isinstance(ent, LineEntity):
                _register_pt((i, 'p0'), ent.p0)
                _register_pt((i, 'p1'), ent.p1)
            elif isinstance(ent, ArcEntity):
                _register_pt((i, 'center'), ent.center)
                _register_pt((i, 'p0'), ent.p0)
                _register_pt((i, 'p1'), ent.p1)

        start_at = {}
        for key, uv in (pins or {}).items():
            ck = canon_map.get(key)
            if ck is not None:
                start_at.setdefault(ck, uv)

        # Add solver points for each canonical key.
        slvs_pts: dict[tuple, object] = {}
        for i, ent in enumerate(self.entities):
            if isinstance(ent, LineEntity):
                keys_pts = [((i, 'p0'), ent.p0), ((i, 'p1'), ent.p1)]
            elif isinstance(ent, ArcEntity):
                keys_pts = [((i, 'center'), ent.center),
                            ((i, 'p0'),     ent.p0),
                            ((i, 'p1'),     ent.p1)]
            else:
                continue
            for key, pt in keys_pts:
                ck = canon_map[key]
                if ck not in slvs_pts:
                    pt = start_at.get(ck, pt)
                    slvs_pts[ck] = slvs.add_point_2d(float(pt[0]), float(pt[1]), wp)

        def _pt(ei, which):
            return slvs_pts[canon_map[(ei, which)]]

        # Register lines.
        slvs_lines: dict[int, object] = {}
        for i, ent in enumerate(self.entities):
            if isinstance(ent, LineEntity):
                slvs_lines[i] = slvs.add_line_2d(_pt(i, 'p0'), _pt(i, 'p1'), wp)

        # Register arcs and full circles.
        TWO_PI = 2.0 * np.pi
        slvs_arcs:      dict[int, object] = {}
        arc_center_pts: dict[int, object] = {}
        arc_start_pts:  dict[int, object] = {}
        arc_end_pts:    dict[int, object] = {}
        arc_dist_ents:  dict[int, object] = {}  # radius distance entity for circles

        for i, ent in enumerate(self.entities):
            if not isinstance(ent, ArcEntity):
                continue
            cp = _pt(i, 'center')
            sp = _pt(i, 'p0')
            ep = _pt(i, 'p1')
            arc_center_pts[i] = cp
            arc_start_pts[i]  = sp
            arc_end_pts[i]    = ep

            is_full_circle = abs(ent.end_angle - ent.start_angle - TWO_PI) < 1e-6
            if is_full_circle:
                dist_ent = slvs.add_distance(float(ent.radius), wp)
                arc_dist_ents[i] = dist_ent
                slvs_arcs[i] = slvs.add_circle(nm3d, cp, dist_ent, wp)
            else:
                slvs_arcs[i] = slvs.add_arc(nm3d, cp, sp, ep, wp)

        # Includes: fixed geometry in group 1 — the solver only solves the
        # current group (0), so their params are constants.  A straight
        # include (ref_line) also gets a first→last line in slvs_lines so
        # parallel / perpendicular / equal / symmetric constraints can name it.
        ref_geo: dict[int, dict] = {}
        slvs.set_group(1)
        for i, ent in enumerate(self.entities):
            if not isinstance(ent, ReferenceEntity) or not ent.points:
                continue
            g = {'verts': [slvs.add_point_2d(float(p[0]), float(p[1]), wp)
                           for p in ent.points],
                 'segs': [], 'center': None, 'circle': None}
            circ = _ref_circle(self, ent) if len(ent.points) >= 2 else None
            if circ is not None:
                c, r = circ
                g['center'] = slvs.add_point_2d(float(c[0]), float(c[1]), wp)
                g['circle'] = slvs.add_circle(
                    nm3d, g['center'], slvs.add_distance(float(r), wp), wp)
            else:
                g['segs'] = [slvs.add_line_2d(a, b, wp)
                             for a, b in zip(g['verts'], g['verts'][1:])]
                if ref_line(ent) is not None:
                    slvs_lines[i] = (g['segs'][0] if len(g['segs']) == 1 else
                                     slvs.add_line_2d(g['verts'][0], g['verts'][-1], wp))
            ref_geo[i] = g
        slvs.set_group(0)

        return (slvs, wp, nm3d, canon_map, slvs_pts,
                slvs_lines, slvs_arcs,
                arc_center_pts, arc_start_pts, arc_end_pts, arc_dist_ents,
                ref_geo)

    def _apply_constraints_to_solver(self, slvs, wp, canon_map,
                                     slvs_pts, slvs_lines, slvs_arcs,
                                     arc_dist_ents, constraints):
        """Emit all constraint calls onto an already-built solver system."""
        def _pt(ei, which):
            return slvs_pts[canon_map[(ei, which)]]

        for con in constraints:
            i = con.indices[0] if con.indices else None
            if con.type == 'distance':
                if i is not None and i < len(self.entities) and \
                        isinstance(self.entities[i], LineEntity):
                    slvs.distance(_pt(i, 'p0'), _pt(i, 'p1'), con.value, wp)
            elif con.type == 'diameter':
                if i is not None and i < len(self.entities) and \
                        isinstance(self.entities[i], ArcEntity):
                    if i in slvs_arcs:
                        slvs.diameter(slvs_arcs[i], con.value)
            elif con.type == 'parallel':
                ref, mov = con.indices
                if ref in slvs_lines and mov in slvs_lines:
                    slvs.parallel(slvs_lines[ref], slvs_lines[mov], wp)
            elif con.type == 'perpendicular':
                ref, mov = con.indices
                if ref in slvs_lines and mov in slvs_lines:
                    slvs.perpendicular(slvs_lines[ref], slvs_lines[mov], wp)
            elif con.type == 'horizontal':
                if i in slvs_lines:
                    slvs.horizontal(slvs_lines[i], wp)
            elif con.type == 'vertical':
                if i in slvs_lines:
                    slvs.vertical(slvs_lines[i], wp)
            elif con.type == 'equal':
                a, b = con.indices
                if a in slvs_lines and b in slvs_lines:
                    slvs.equal(slvs_lines[a], slvs_lines[b], wp)
            elif con.type == 'symmetric':
                # (src_idx, copy_idx, axis_line_idx): reflect each corresponding
                # point of src/copy across the axis line.
                if len(con.indices) != 3:
                    continue
                src_i, cpy_i, axis_i = con.indices
                if (src_i >= len(self.entities) or cpy_i >= len(self.entities)
                        or axis_i not in slvs_lines):
                    continue
                src_e, cpy_e = self.entities[src_i], self.entities[cpy_i]
                if type(src_e) is not type(cpy_e):
                    continue
                axis_line = slvs_lines[axis_i]
                if isinstance(src_e, LineEntity):
                    whichs = ('p0', 'p1')
                elif isinstance(src_e, ArcEntity):
                    whichs = ('center', 'p0', 'p1')
                else:
                    continue
                for which in whichs:
                    try:
                        slvs.symmetric(_pt(src_i, which), _pt(cpy_i, which),
                                       axis_line, wp)
                    except Exception:
                        pass

    def compute_constraint_status(self) -> tuple[str, int]:
        """
        Return (status, user_dof) without touching entity coordinates.

        status values: 'none' | 'under' | 'fully' | 'over'
        user_dof      : meaningful DOFs remaining (0 = fully constrained).
                        -1 when status is 'none' or 'over'.
        """
        if not self.constraints:
            return ('none', -1)

        has_geometry = any(isinstance(e, (LineEntity, ArcEntity))
                           for e in self.entities)
        if not has_geometry:
            return ('none', -1)

        def _solve(with_attachments: bool):
            built = self._build_solver_system()
            if built is None:
                return None, None
            slvs, wp, nm3d, canon_map, slvs_pts, slvs_lines, slvs_arcs, \
                arc_center_pts, arc_start_pts, arc_end_pts, arc_dist_ents, \
                ref_geo = built
            # Measure unconstrained DOF before adding any constraints.
            if slvs.solve() != 0:
                return None, None
            if with_attachments:
                self._emit_attachments(slvs, wp, canon_map, slvs_pts, ref_geo,
                                       include_attachments(self))
            self._apply_constraints_to_solver(slvs, wp, canon_map, slvs_pts,
                                              slvs_lines, slvs_arcs,
                                              arc_dist_ents, self.constraints)
            return slvs.solve(), slvs

        result, slvs = _solve(with_attachments=True)
        if result is None:
            return ('none', -1)
        if result != 0:
            # Points held by includes can be redundant with user constraints
            # (apply_constraints copes by dropping one); don't report that as
            # over-constrained — judge the user's constraints alone.
            result, slvs = _solve(with_attachments=False)
            if result is None:
                return ('none', -1)
        if result != 0:
            return ('over', -1)

        user_dof = slvs.dof() - _solver_overhead_dof()
        if user_dof <= 0:
            return ('fully', 0)
        return ('under', user_dof)

    def _add_snap_coincidences(self, slvs, wp, slvs_pts, slvs_lines,
                               slvs_arcs, canon_map, user_constraints=None,
                               implicit_lengths=True, pinned=frozenset()):
        """Add geometric constraints for non-endpoint snap anchors.

        MIDPOINT snap on a line  → midpoint(ep, line)
        NEAREST  snap on a line  → coincident(ep, line)
        ENDPOINT snap on an arc  → already deduplicated via canon_map (shared pt)
        CENTER   snap on an arc  → coincident(ep, arc_center_pt) via zero-distance
        NEAREST/TANGENT on arc   → pin endpoint to stored coordinate (fallback)

        implicit_lengths: if False, skip adding distance() for line length
          preservation. Use False when re-solving a fully-established sketch
          (no_pin=True) to avoid redundant distance constraints.

        pinned: (entity_idx, which) keys the caller pins to a new target
          (include follow-up, see apply_constraints) — skip the stored-
          coordinate fallback for those or the two pins fight.

        Returns a set of canonical keys that have coincident(point, circle)
        constraints — these must NOT be pinned with dragged() in the caller.
        """
        circle_coincident_cks: set = set()
        from cad.sketch_tools.snap import SnapType

        # Indices of lines that already have a user-placed distance constraint —
        # don't add an implicit length for those or we'll over-constrain.
        user_distanced = set()
        if user_constraints:
            for con in user_constraints:
                if con.type == 'distance':
                    user_distanced.add(con.indices[0])

        for i, ent in enumerate(self.entities):
            if not isinstance(ent, LineEntity):
                continue
            for which, snap_meta in (('p0', getattr(ent, 'p0_snap', None)),
                                     ('p1', getattr(ent, 'p1_snap', None))):
                if snap_meta is None:
                    continue
                src_idx, snap_type = snap_meta
                if src_idx >= len(self.entities):
                    continue
                src = self.entities[src_idx]
                ck = canon_map.get((i, which))
                ep_sp = slvs_pts.get(ck) if ck is not None else None
                if ep_sp is None:
                    continue

                # Check whether the OTHER endpoint of this line also has a
                # non-endpoint snap — if so, both ends are geometrically
                # constrained and adding an implicit length would over-constrain.
                other_which = 'p1' if which == 'p0' else 'p0'
                other_snap = getattr(ent, f'{other_which}_snap', None)
                other_also_snapped = (other_snap is not None)

                def _add_implicit_length():
                    """Pin this line's length to keep it rigid during solve."""
                    if not implicit_lengths:
                        return  # caller opted out (re-solve of established sketch)
                    if i in user_distanced:
                        return  # user already constrains this line's length
                    if other_also_snapped:
                        return  # both ends constrained — length is implicit
                    other_ck = canon_map.get((i, other_which))
                    other_sp = slvs_pts.get(other_ck) if other_ck else None
                    if other_sp is not None:
                        cur_len = float(np.linalg.norm(ent.p1 - ent.p0))
                        if cur_len > 1e-6:
                            slvs.distance(ep_sp, other_sp, cur_len, wp)

                if snap_type == SnapType.MIDPOINT and isinstance(src, LineEntity):
                    if src_idx in slvs_lines:
                        slvs.midpoint(ep_sp, slvs_lines[src_idx], wp)
                        _add_implicit_length()

                elif snap_type == SnapType.NEAREST and isinstance(src, LineEntity):
                    if src_idx in slvs_lines:
                        slvs.coincident(ep_sp, slvs_lines[src_idx], wp)
                        _add_implicit_length()

                elif isinstance(src, ArcEntity) and src_idx in slvs_arcs:
                    # Endpoint snapped to arc start/end: already shared via
                    # canon_map deduplication — no extra constraint needed.
                    # Center snap: enforce coincidence with arc center point.
                    if snap_type == SnapType.CENTER:
                        center_ck = canon_map.get((src_idx, 'center'))
                        center_sp = slvs_pts.get(center_ck) if center_ck else None
                        if center_sp is not None:
                            slvs.distance(ep_sp, center_sp, 0.0, wp)
                            _add_implicit_length()
                    elif snap_type in (SnapType.NEAREST, SnapType.TANGENT,
                                       SnapType.ENDPOINT):
                        TWO_PI = 2.0 * np.pi
                        is_full_circle = abs(src.end_angle - src.start_angle
                                             - TWO_PI) < 1e-6
                        if is_full_circle and src_idx in slvs_arcs:
                            # coincident(point, circle) keeps the point on the
                            # circle surface — works even when diameter changes.
                            slvs.coincident(ep_sp, slvs_arcs[src_idx], wp)
                            circle_coincident_cks.add(ck)
                            _add_implicit_length()
                        elif (i, which) not in pinned:
                            # Partial arc — pin to stored coordinate (fallback).
                            ref_uv = (ent.p0 if which == 'p0' else ent.p1).copy()
                            ref_sp = slvs.add_point_2d(float(ref_uv[0]),
                                                       float(ref_uv[1]), wp)
                            slvs.dragged(ref_sp, wp)
                            slvs.distance(ep_sp, ref_sp, 0.0, wp)

                elif snap_type in (SnapType.CENTER, SnapType.NEAREST,
                                   SnapType.TANGENT, SnapType.MIDPOINT) \
                        and (i, which) not in pinned:
                    # Reference entity or unregistered arc — pin coordinate.
                    ref_uv = (ent.p0 if which == 'p0' else ent.p1).copy()
                    ref_sp = slvs.add_point_2d(float(ref_uv[0]),
                                               float(ref_uv[1]), wp)
                    slvs.dragged(ref_sp, wp)
                    slvs.distance(ep_sp, ref_sp, 0.0, wp)

        return circle_coincident_cks

    def apply_last_constraint(self) -> bool:
        """Solve all constraints with no pinning. Alias for apply_all_constraints."""
        return self.apply_all_constraints()

    def apply_all_constraints(self, pins: dict | None = None,
                              attachments: list | None = None) -> bool:
        """Solve all constraints simultaneously with no pinning.
        pins / attachments: see apply_constraints."""
        return self.apply_constraints(free_for_last=False, no_pin=True,
                                      pins=pins, attachments=attachments)

    def _free_cks_for_constraint(self, con, canon_map) -> set:
        """Return canonical keys that must be FREE when applying constraint `con`."""
        free = set()
        if con.type == 'distance':
            i = con.indices[0]
            if i < len(self.entities) and isinstance(self.entities[i], LineEntity):
                free.add(canon_map[(i, 'p1')])
        elif con.type == 'diameter':
            # Diameter only changes radius, not point positions — free nothing.
            pass
        elif con.type in ('parallel', 'perpendicular', 'equal'):
            mov = con.indices[1]
            if mov < len(self.entities) and isinstance(self.entities[mov], LineEntity):
                free.add(canon_map[(mov, 'p0')])
                free.add(canon_map[(mov, 'p1')])
        elif con.type in ('horizontal', 'vertical'):
            i = con.indices[0]
            if i < len(self.entities) and isinstance(self.entities[i], LineEntity):
                free.add(canon_map[(i, 'p1')])
        return free

    def _free_cks_for_last(self, canon_map) -> set:
        """Canonical keys free to move for free_for_last: what the last
        constraint needs, plus whatever earlier constraints and arc snaps
        tie to those."""
        free_cks = self._free_cks_for_constraint(self.constraints[-1], canon_map)

        for con in self.constraints[:-1]:
            if con.type == 'distance':
                i = con.indices[0]
                if i < len(self.entities) and isinstance(self.entities[i], LineEntity):
                    free_cks.add(canon_map[(i, 'p1')])

        # Propagate freedom through geometric constraints.
        changed = True
        while changed:
            changed = False
            for con in self.constraints[:-1]:
                if con.type in ('parallel', 'perpendicular', 'equal'):
                    ref, mov = con.indices
                    involved = {canon_map.get((ref, 'p0')), canon_map.get((ref, 'p1')),
                                canon_map.get((mov, 'p0')), canon_map.get((mov, 'p1'))} - {None}
                    if involved & free_cks:
                        before = len(free_cks)
                        free_cks |= involved
                        changed = changed or len(free_cks) > before
                elif con.type in ('horizontal', 'vertical'):
                    i = con.indices[0]
                    p0 = canon_map.get((i, 'p0'))
                    p1 = canon_map.get((i, 'p1'))
                    if p0 in free_cks or p1 in free_cks:
                        before = len(free_cks)
                        free_cks |= {p0, p1} - {None}
                        changed = changed or len(free_cks) > before

        # Propagate freedom to arc centers: if any line endpoint that is
        # snap-connected to an arc is free, the arc center must also be
        # free so the arc can move with the line system instead of being
        # pinned in place by dragged(center_sp).
        for i, ent in enumerate(self.entities):
            if not isinstance(ent, LineEntity):
                continue
            for which, snap_meta in (('p0', getattr(ent, 'p0_snap', None)),
                                     ('p1', getattr(ent, 'p1_snap', None))):
                if snap_meta is None:
                    continue
                src_idx, _snap_type = snap_meta
                if src_idx >= len(self.entities):
                    continue
                if not isinstance(self.entities[src_idx], ArcEntity):
                    continue
                ep_ck = canon_map.get((i, which))
                center_ck = canon_map.get((src_idx, 'center'))
                if ep_ck in free_cks and center_ck is not None:
                    free_cks.add(center_ck)
        return free_cks

    def _emit_attachments(self, slvs, wp, canon_map, slvs_pts, ref_geo,
                          attachments, drop=frozenset(), pinned=frozenset()):
        """Hard-constrain drawn points to the include geometry they sit on.

        A point on a segment or circle would be free to slide along it; when
        replay moved the include (key in *pinned*, the point starts at its
        old fraction / angle) it is held there with dragged() instead, so
        it keeps its relative position.

        Returns ({attachment idx: check(slvs) -> bool}, attached canonical
        keys).  Attachments whose include changed point count or stopped
        being a circle are skipped (nothing to hold them to)."""
        checks, attached, seen, dragged = {}, set(), set(), set()

        def _uv(sp):
            return np.array(slvs.params(sp.params)[:2], dtype=np.float64)

        for n, (key, ri, kind, data, n_pts) in enumerate(attachments):
            g, ck = ref_geo.get(ri), canon_map.get(key)
            if g is None or ck is None or ck not in slvs_pts:
                continue
            pt, target, check = slvs_pts[ck], None, None
            if kind == 'v' and len(g['verts']) == n_pts and data < n_pts:
                target = g['verts'][data]
                check = (lambda s, p=pt, t=target:
                         np.linalg.norm(_uv(p) - _uv(t)) < 1e-6)
            elif kind == 's' and len(g['verts']) == n_pts and data[0] < len(g['segs']):
                k = data[0]
                target = g['segs'][k]
                a, b = g['verts'][k], g['verts'][k + 1]
                check = (lambda s, p=pt, a=a, b=b:
                         _dist_to_line(_uv(p), _uv(a), _uv(b)) < 1e-6)
            elif kind == 'c' and g['center'] is not None:
                target = g['center']
                check = (lambda s, p=pt, t=target:
                         np.linalg.norm(_uv(p) - _uv(t)) < 1e-6)
            elif kind == 'a' and g['circle'] is not None:
                target = g['circle']
                r = _ref_circle(self, self.entities[ri])[1]
                check = (lambda s, p=pt, c=g['center'], r=r:
                         abs(np.linalg.norm(_uv(p) - _uv(c)) - r) < 1e-6)
            if target is None:
                continue
            attached.add(ck)
            if (ck, id(target)) in seen or n in drop:
                continue
            seen.add((ck, id(target)))
            if kind in ('s', 'a') and key in pinned:
                if ck in dragged:
                    continue
                dragged.add(ck)
                slvs.dragged(pt, wp)
            else:
                slvs.coincident(pt, target, wp)
            checks[n] = check
        return checks, attached

    def _constraint_holds(self, con, slvs, canon_map, slvs_pts, ref_geo) -> bool:
        """Whether a geometric constraint is satisfied by the solved points."""
        def ends(i):
            if i in ref_geo:
                if ref_line(self.entities[i]) is None:
                    return None
                sps = [ref_geo[i]['verts'][0], ref_geo[i]['verts'][-1]]
            elif (i, 'p0') in canon_map:
                sps = [slvs_pts[canon_map[(i, 'p0')]], slvs_pts[canon_map[(i, 'p1')]]]
            else:
                return None
            return [np.array(slvs.params(sp.params)[:2]) for sp in sps]

        lines = [ends(i) for i in con.indices]
        if any(l is None for l in lines):
            return True               # not a line constraint the solver saw
        d = [b - a for a, b in lines]
        n = [np.linalg.norm(v) for v in d]
        if min(n) < 1e-9:
            return True
        u = [v / m for v, m in zip(d, n)]
        tol = 1e-6
        if con.type == 'horizontal':
            return abs(u[0][1]) < tol
        if con.type == 'vertical':
            return abs(u[0][0]) < tol
        if con.type == 'parallel':
            return abs(u[0][0] * u[1][1] - u[0][1] * u[1][0]) < tol
        if con.type == 'perpendicular':
            return abs(float(u[0] @ u[1])) < tol
        if con.type == 'equal':
            return abs(n[0] - n[1]) < tol * max(1.0, n[0])
        return True

    def apply_constraints(self, free_for_last: bool = False,
                          no_pin: bool = False, pins: dict | None = None,
                          attachments: list | None = None) -> bool:
        """
        Run the SolveSpace solver over all entities + constraints.
        Updates entity coordinates in-place.  Returns True on success.

        free_for_last: pin everything except what the last constraint needs to move.
        no_pin: skip all pinning — solver finds minimum motion (use when editing).
        pins: {(entity_idx, which): uv} start positions for points attached
              to moved includes (replay).
        attachments: include_attachments() result; inferred from the current
              coordinates when None.  Replay passes the ones it measured
              before re-projecting the includes.
        """
        if attachments is None:
            attachments = include_attachments(self)
        if not self.constraints and not pins and not attachments:
            return True
        pins = pins or {}

        def _attempt(implicit_lengths: bool, drop_att=frozenset(),
                     drop_con=frozenset()):
            """Build, constrain and solve once.  Returns (result, state)."""
            built = self._build_solver_system(pins)
            if built is None:
                return None, None
            slvs, wp, nm3d, canon_map, slvs_pts, slvs_lines, slvs_arcs, \
                arc_center_pts, arc_start_pts, arc_end_pts, arc_dist_ents, \
                ref_geo = built
            att_checks, attached_cks = self._emit_attachments(
                slvs, wp, canon_map, slvs_pts, ref_geo, attachments, drop_att,
                pinned=frozenset(pins))
            # Points held by an include are fully or partly determined by
            # it; a stored-coordinate pin or dragged() on top is redundant,
            # which SolveSpace rejects.
            circ_cks = self._add_snap_coincidences(
                slvs, wp, slvs_pts, slvs_lines, slvs_arcs, canon_map,
                self.constraints, implicit_lengths=implicit_lengths,
                pinned=frozenset(pins) | {k for k, *_ in attachments})
            no_drag = set(attached_cks)
            # Pinned points without a hard attachment: dragged() keeps them
            # at their start (the target) when it can.
            for key in pins:
                ck = canon_map.get(key)
                if ck is None or ck not in slvs_pts or ck in no_drag:
                    continue
                slvs.dragged(slvs_pts[ck], wp)
                no_drag.add(ck)

            if free_for_last and self.constraints:
                free_cks = self._free_cks_for_last(canon_map)
                for ck, sp in slvs_pts.items():
                    if ck not in free_cks and ck not in circ_cks and ck not in no_drag:
                        slvs.dragged(sp, wp)
            elif no_pin:
                # Arcs have no implicit anchor — pin arc centers so the
                # solver doesn't drift circles away from their snap-connected
                # line endpoints.  coincident(point, circle) constraints still
                # let endpoints slide along the circle.  Concentric circles
                # share one canonical center; dragging it twice makes
                # SolveSpace return result=1, so dedupe.
                for i, ent in enumerate(self.entities):
                    if not isinstance(ent, ArcEntity):
                        continue
                    ck = canon_map.get((i, 'center'))
                    if ck and ck in slvs_pts and ck not in no_drag:
                        slvs.dragged(slvs_pts[ck], wp)
                        no_drag.add(ck)
            else:
                for con in self.constraints:
                    if con.type == 'distance':
                        i = con.indices[0]
                        if i < len(self.entities) and isinstance(self.entities[i], LineEntity):
                            ck = canon_map[(i, 'p0')]
                            if ck not in no_drag:
                                slvs.dragged(slvs_pts[ck], wp)
                                no_drag.add(ck)

            cons = [c for k, c in enumerate(self.constraints) if k not in drop_con]
            self._apply_constraints_to_solver(slvs, wp, canon_map, slvs_pts,
                                              slvs_lines, slvs_arcs,
                                              arc_dist_ents, cons)
            result = slvs.solve()
            state = (slvs, wp, canon_map, slvs_pts, slvs_lines, slvs_arcs,
                     arc_dist_ents, ref_geo, att_checks)
            return result, state

        def _dropped_hold(state, drop_att, drop_con) -> bool:
            """A constraint dropped as redundant must still hold after the
            solve — otherwise it was a real conflict, not a redundancy."""
            slvs, _, canon_map, slvs_pts, _, _, _, ref_geo, att_checks = state
            ok = all(chk(slvs) for k, chk in att_checks.items() if k in drop_att)
            for k in drop_con:
                ok = ok and self._constraint_holds(
                    self.constraints[k], slvs, canon_map, slvs_pts, ref_geo)
            return ok

        result, state = _attempt(implicit_lengths=True)
        if result is None:
            return False
        if result != 0 and no_pin:
            # Implicit length constraints may be redundant when re-solving a
            # fully-constrained sketch (e.g., after re-entering). Retry without
            # them — the starting positions already satisfy the snap topology so
            # minimum-motion will preserve it without explicit length enforcement.
            r2, s2 = _attempt(implicit_lengths=False)
            if r2 == 0:
                result, state = r2, s2
        if result != 0 and attachments:
            # Include attachments are often redundant with the user's own
            # constraints (a line between two include vertices that is also
            # marked horizontal) and SolveSpace rejects redundancy.  Drop one
            # attachment or geometric constraint at a time; accept the first
            # solve in which the dropped one still holds.
            implicit = not no_pin
            cands = [('a', k) for k in sorted(state[-1], reverse=True)] + \
                    [('c', k) for k, c in enumerate(self.constraints)
                     if c.type in _GEOMETRIC_CONSTRAINTS]
            for kind, k in cands:
                da = frozenset([k]) if kind == 'a' else frozenset()
                dc = frozenset([k]) if kind == 'c' else frozenset()
                r3, s3 = _attempt(implicit, da, dc)
                if r3 == 0 and _dropped_hold(s3, da, dc):
                    result, state = r3, s3
                    break

        slvs, wp, canon_map, slvs_pts, slvs_lines, slvs_arcs, arc_dist_ents, \
            _ref_geo, _att_checks = state
        if result != 0:
            print(f"[Solver] FAILED result={result} dof={slvs.dof()} "
                  f"constraints={[c.type for c in self.constraints]} "
                  f"no_pin={no_pin} free_for_last={free_for_last}")
            return False

        # Write back line endpoint coordinates from solver points.
        for i, ent in enumerate(self.entities):
            if not isinstance(ent, LineEntity):
                continue
            for which in ('p0', 'p1'):
                ck = canon_map[(i, which)]
                sp = slvs_pts.get(ck)
                if sp is None:
                    continue
                uv = slvs.params(sp.params)
                if which == 'p0':
                    ent.p0 = np.array([uv[0], uv[1]], dtype=np.float64)
                else:
                    ent.p1 = np.array([uv[0], uv[1]], dtype=np.float64)

        # Write back arc coordinates.
        TWO_PI = 2.0 * np.pi
        for i, ent in enumerate(self.entities):
            if not isinstance(ent, ArcEntity):
                continue
            ck_center = canon_map.get((i, 'center'))
            if ck_center and ck_center in slvs_pts:
                uv = slvs.params(slvs_pts[ck_center].params)
                ent.center = np.array([uv[0], uv[1]], dtype=np.float64)
            is_full_circle = abs(ent.end_angle - ent.start_angle - TWO_PI) < 1e-6
            if is_full_circle:
                if i in arc_dist_ents:
                    r_val = slvs.params(arc_dist_ents[i].params)
                    ent.radius = float(r_val[0])
            else:
                ck0 = canon_map.get((i, 'p0'))
                ck1 = canon_map.get((i, 'p1'))
                if ck0 and ck0 in slvs_pts:
                    uv = slvs.params(slvs_pts[ck0].params)
                    d  = np.array([uv[0], uv[1]]) - ent.center
                    ent.radius      = float(np.linalg.norm(d))
                    ent.start_angle = float(np.arctan2(d[1], d[0]))
                if ck1 and ck1 in slvs_pts:
                    uv = slvs.params(slvs_pts[ck1].params)
                    d  = np.array([uv[0], uv[1]]) - ent.center
                    ent.end_angle = float(np.arctan2(d[1], d[0]))
                    if ent.end_angle <= ent.start_angle:
                        ent.end_angle += TWO_PI

        # Second pass: fix line endpoints that share a canonical key with an arc
        # point.  For full circles the radius changed but the solver point wasn't
        # repositioned (the circle entity owns its radius separately), so any
        # line endpoint that was merged with arc.p0 / arc.p1 / arc.center must be
        # recomputed from the now-updated arc geometry.
        arc_canonical_coords: dict[tuple, np.ndarray] = {}
        for i, ent in enumerate(self.entities):
            if not isinstance(ent, ArcEntity):
                continue
            ck_c  = canon_map.get((i, 'center'))
            ck_p0 = canon_map.get((i, 'p0'))
            ck_p1 = canon_map.get((i, 'p1'))
            if ck_c:
                arc_canonical_coords[ck_c] = ent.center.copy()
            if ck_p0:
                arc_canonical_coords[ck_p0] = ent.p0.copy()
            if ck_p1:
                arc_canonical_coords[ck_p1] = ent.p1.copy()

        for i, ent in enumerate(self.entities):
            if not isinstance(ent, LineEntity):
                continue
            for which in ('p0', 'p1'):
                ck = canon_map[(i, which)]
                if ck in arc_canonical_coords:
                    coord = arc_canonical_coords[ck]
                    if which == 'p0':
                        ent.p0 = coord.copy()
                    else:
                        ent.p1 = coord.copy()

        return True

    # ------------------------------------------------------------------
    # Derived geometry — built on demand, never cached
    # ------------------------------------------------------------------

    def line_segments(self) -> list[tuple[np.ndarray, np.ndarray]]:
        return [(e.p0.copy(), e.p1.copy())
                for e in self.entities if isinstance(e, LineEntity)]

    def arc_entities(self) -> list:
        return [e for e in self.entities if isinstance(e, ArcEntity)]

    def reference_chains(self) -> list[list[tuple[float, float]]]:
        return [[(float(p[0]), float(p[1])) for p in e.points]
                for e in self.entities
                if isinstance(e, ReferenceEntity) and len(e.points) >= 2]

    def closed_loops(self, tol: float = 1e-3) -> list[list[tuple[float, float]]]:
        """All closed chains from LineEntity/ArcEntity segments, ReferenceEntity
        polylines, and closed SplineEntity curves."""
        closed = []
        for _chain, uv_pts in self._collect_drawn_loops_with_arcs(tol=tol):
            closed.append(uv_pts)
        for chain in self.reference_chains():
            if np.linalg.norm(np.array(chain[-1]) - np.array(chain[0])) < tol:
                closed.append(chain)
        for ent in self.entities:
            if (isinstance(ent, SplineEntity)
                    and not getattr(ent, "construction", False)):
                is_closed = ent.closed or (len(ent.points) >= 3 and
                    np.linalg.norm(ent.points[-1] - ent.points[0]) < tol)
                if is_closed:
                    uv = [(float(p[0]), float(p[1]))
                          for p in ent.tessellate(64)]
                    closed.append(uv)
        return closed

    def has_closed_loop(self, tol: float = 1e-3) -> bool:
        return len(self.closed_loops(tol=tol)) > 0

    def face_regions(self, tol: float = 1e-3):
        """
        Return UV region descriptors parallel to build_faces().
        Each entry is (outer_uvs, [hole_uvs, ...]) — the outer polygon and
        the direct-child polygons that are punched as holes.
        Used by the picker for exact point-in-region tests.
        """
        uv_loops = self._collect_uv_loops(tol)
        if not uv_loops:
            return []

        def _area(uvs):
            a = 0.0
            for k in range(len(uvs)):
                x0, y0 = uvs[k]; x1, y1 = uvs[(k + 1) % len(uvs)]
                a += x0 * y1 - x1 * y0
            return abs(a) * 0.5

        areas = [_area(uv) for uv in uv_loops]
        n = len(uv_loops)

        def _loop_contains(j, i):
            if areas[j] <= areas[i]:
                return False
            return all(_uv_point_in_loop(pt, uv_loops[j]) for pt in uv_loops[i])

        parent = [None] * n
        for i in range(n):
            containers = [j for j in range(n) if _loop_contains(j, i)]
            if containers:
                parent[i] = min(containers, key=lambda j: areas[j])

        return [
            (uv_loops[i], [uv_loops[j] for j in range(n) if parent[j] == i])
            for i in range(n)
        ]

    def _collect_uv_loops(self, tol: float = 1e-3):
        """Collect closed UV polygon lists from all entities (shared by build_faces and face_regions)."""
        uv_loops = []
        for ref in self.entities:
            if not isinstance(ref, ReferenceEntity) or len(ref.points) < 3:
                continue
            pts_close = np.linalg.norm(ref.points[-1] - ref.points[0]) < tol
            edge_closed = False
            if not pts_close and ref.occ_edges:
                try:
                    from OCP.BRepAdaptor import BRepAdaptor_Curve as _BAC
                    for _e in ref.occ_edges:
                        _a = _BAC(_e)
                        if _a.IsClosed() or _a.IsPeriodic():
                            edge_closed = True
                            break
                except Exception:
                    pass
            if not pts_close and not edge_closed:
                continue
            raw = ref.points[:-1] if pts_close else ref.points
            uv = [(float(p[0]), float(p[1])) for p in raw]
            if len(uv) >= 3:
                uv_loops.append(uv)

        # Closed splines are standalone profile loops.
        for ent in self.entities:
            if (isinstance(ent, SplineEntity)
                    and not getattr(ent, "construction", False)):
                is_closed = ent.closed or (len(ent.points) >= 3 and
                    np.linalg.norm(ent.points[-1] - ent.points[0]) < tol)
                if not is_closed:
                    continue
                tess = ent.tessellate(64)
                uv = [(float(p[0]), float(p[1])) for p in tess]
                # Drop a duplicated closing point if present.
                if len(uv) >= 2 and abs(uv[0][0] - uv[-1][0]) < tol \
                        and abs(uv[0][1] - uv[-1][1]) < tol:
                    uv = uv[:-1]
                if len(uv) >= 3:
                    uv_loops.append(uv)

        for _chain, uv_pts in self._collect_drawn_loops_with_arcs(tol=tol):
            if len(uv_pts) >= 3:
                uv_loops.append(uv_pts)
        return uv_loops

    def _ref_uv_polyline(self, ref) -> list:
        """
        UV polyline a ReferenceEntity should chain on for loop detection.

        Prefers the OCCT edge geometry (sampled and projected onto the sketch
        plane) so the endpoints match exactly what the snap engine offers — the
        snap engine reports OCC curve endpoints, not the mesh-tessellated
        ref.points.  Falls back to ref.points when no usable occ_edge exists.
        """
        if ref.occ_edges:
            try:
                from OCP.BRepAdaptor import BRepAdaptor_Curve as _BAC
                from OCP.GCPnts import GCPnts_QuasiUniformAbscissa
                pts = []
                for occ_edge in ref.occ_edges:
                    if occ_edge is None:
                        continue
                    adp = _BAC(occ_edge)
                    sampler = GCPnts_QuasiUniformAbscissa()
                    sampler.Initialize(adp, 33)
                    for j in range(1, sampler.NbPoints() + 1):
                        p = adp.Value(sampler.Parameter(j))
                        d = np.array([p.X(), p.Y(), p.Z()]) - self.plane_origin
                        pts.append(np.array([float(np.dot(d, self.plane_x_axis)),
                                             float(np.dot(d, self.plane_y_axis))]))
                if len(pts) >= 2:
                    return pts
            except Exception:
                pass
        return [np.array(p, dtype=np.float64) for p in ref.points]

    def _collect_drawn_loops_with_arcs(self, tol: float = 1e-3):
        """
        Chain LineEntity and ArcEntity into closed loops, preserving entity refs.
        Returns list of (seg_loop, uv_pts) where:
          seg_loop : [(entity, p0_uv, p1_uv), ...]  in chained order
          uv_pts   : tessellated [(u,v), ...] for UV area tests

        Arcs are pre-split at any line endpoint that lands on their circumference
        (tangent/nearest snaps), so the chainer sees proper endpoint connections.
        """
        import math as _math

        # Collect all line endpoints that land on each arc's circumference.
        # These become split points that break the arc into sub-arcs.
        def _angle_on_arc(pt, arc):
            """Return the angle of pt on arc if within tol, else None."""
            d = pt - arc.center
            if abs(float(np.linalg.norm(d)) - arc.radius) > tol:
                return None
            angle = float(_math.atan2(float(d[1]), float(d[0])))
            # Normalise into [start_angle, end_angle]
            span = arc.end_angle - arc.start_angle
            rel  = (angle - arc.start_angle) % (2 * _math.pi)
            if rel > span + tol:
                return None
            return arc.start_angle + rel

        # Build (entity, p0, p1) list, splitting arcs where lines touch them.
        segs = []
        for e in self.entities:
            # Construction geometry never participates in a profile loop.
            if getattr(e, "construction", False):
                continue
            if isinstance(e, LineEntity):
                segs.append((e, e.p0.copy(), e.p1.copy()))
            elif isinstance(e, ReferenceEntity):
                # Open included polylines (e.g. an included straight edge) can
                # form part of a mixed drawn+included loop, so feed their
                # segments to the chainer.  Self-closed refs (circles, included
                # rectangles) are handled wholesale in _collect_uv_loops, so
                # skip them here to avoid double-counting.
                #
                # Use the OCCT-edge UV polyline when available so the endpoints
                # match exactly what the snap engine offers (snap uses the OCC
                # curve endpoints, NOT the mesh-tessellated ref.points).  If we
                # chained on ref.points instead, an arc/line that snapped to the
                # OCC endpoint wouldn't meet the tessellated endpoint within tol
                # and the loop would silently fail to close.
                poly = self._ref_uv_polyline(e)
                if len(poly) < 2:
                    continue
                if np.linalg.norm(poly[-1] - poly[0]) < tol:
                    continue
                for k in range(len(poly) - 1):
                    segs.append((e, poly[k].copy(), poly[k + 1].copy()))

        # For each arc, collect all split angles from line endpoints
        for arc in self.entities:
            if not isinstance(arc, ArcEntity):
                continue
            if getattr(arc, "construction", False):
                continue
            split_angles = set()
            for e in self.entities:
                if not isinstance(e, LineEntity):
                    continue
                for pt in (e.p0, e.p1):
                    ang = _angle_on_arc(pt, arc)
                    if ang is not None:
                        # Avoid duplicating the arc's own endpoints
                        if (abs(ang - arc.start_angle) > tol and
                                abs(ang - arc.end_angle) > tol):
                            split_angles.add(ang)

            # Build ordered list of boundary angles: start → splits → end
            angles = sorted([arc.start_angle] + list(split_angles) + [arc.end_angle])

            for k in range(len(angles) - 1):
                a0 = angles[k]
                a1 = angles[k + 1]
                if abs(a1 - a0) < tol:
                    continue
                p0 = arc.center + arc.radius * np.array([_math.cos(a0), _math.sin(a0)])
                p1 = arc.center + arc.radius * np.array([_math.cos(a1), _math.sin(a1)])
                segs.append((arc, p0, p1))

        if not segs:
            return []

        remaining = list(segs)
        loops = []

        while remaining:
            ent0, a0, b0 = remaining.pop(0)
            chain = [(ent0, a0, b0)]
            changed = True
            while changed:
                changed = False
                tail = chain[-1][2]
                head = chain[0][1]
                for i, (ent, a, b) in enumerate(remaining):
                    # Always orient each appended segment so its first endpoint
                    # is the connection point — for both lines and arcs — so the
                    # tessellated polygon stays in continuous boundary order.
                    if np.linalg.norm(a - tail) < tol:
                        chain.append((ent, a, b)); remaining.pop(i); changed = True; break
                    if np.linalg.norm(b - tail) < tol:
                        chain.append((ent, b, a)); remaining.pop(i); changed = True; break
                    if np.linalg.norm(b - head) < tol:
                        chain.insert(0, (ent, a, b)); remaining.pop(i); changed = True; break
                    if np.linalg.norm(a - head) < tol:
                        chain.insert(0, (ent, b, a)); remaining.pop(i); changed = True; break

            # Only keep closed loops
            if np.linalg.norm(chain[-1][2] - chain[0][1]) > tol:
                continue

            # Tessellate for UV area/containment tests
            uv_pts = []
            for ent, ep0, ep1 in chain:
                if isinstance(ent, ArcEntity):
                    # Walk the sub-arc from ep0 to ep1 along the arc's own
                    # stored CCW sweep [start_angle, end_angle].  The chainer
                    # may hand us the segment reversed (ep0 == arc end), so we
                    # measure each endpoint as a sweep fraction and walk in the
                    # direction that actually connects them.  Using the stored
                    # angles (not raw atan2 deltas) keeps major arcs (span > π)
                    # and reversed traversal correct.
                    sweep = ent.end_angle - ent.start_angle

                    def _frac(pt):
                        rel = (float(_math.atan2(float(pt[1] - ent.center[1]),
                                                 float(pt[0] - ent.center[0])))
                               - ent.start_angle) % (2 * _math.pi)
                        # Clamp tiny over/undershoot from float error.
                        if rel > sweep and rel - 2 * _math.pi > -tol:
                            rel = 0.0
                        return rel

                    f0 = _frac(ep0)
                    f1 = _frac(ep1)
                    if abs(f1 - f0) < tol:          # full circle as a sub-arc
                        f1 = f0 + sweep
                    a0 = ent.start_angle + f0
                    a1 = ent.start_angle + f1
                    angles = np.linspace(a0, a1, 17)
                    uv_pts.extend(
                        (float(ent.center[0] + ent.radius * _math.cos(ang)),
                         float(ent.center[1] + ent.radius * _math.sin(ang)))
                        for ang in angles[:-1]
                    )
                else:
                    uv_pts.append((float(ep0[0]), float(ep0[1])))
            if len(uv_pts) >= 3:
                loops.append((chain, uv_pts))

        return loops

    def _collect_uv_loops_drawn(self, tol: float = 1e-3):
        """Collect closed UV polygon loops from drawn LineEntity/ArcEntity segments."""
        uv_loops = []
        segs = self.line_segments()
        # Add arc endpoint pairs so the chainer can close loops containing arcs
        for arc in self.arc_entities():
            segs.append((arc.p0.copy(), arc.p1.copy()))
        if segs:
            for loop in _chain_segments(segs, tol=tol):
                if np.linalg.norm(np.array(loop[-1]) - np.array(loop[0])) > tol:
                    continue
                open_loop = [(float(u), float(v)) for u, v in loop[:-1]]
                if len(open_loop) >= 3:
                    uv_loops.append(open_loop)
        return uv_loops

    def build_wire(self, tol: float = 1e-3):
        """Convert LineEntity segments into a build123d Wire (for DXF export etc.)."""
        from build123d import Polyline, Vector, Compound
        segs = self.line_segments()
        if not segs:
            raise ValueError("Sketch has no line segments to build a wire from.")
        wires = []
        for chain in _chain_segments(segs, tol=tol):
            pts = [Vector(*(self.plane_origin
                            + u * self.plane_x_axis
                            + v * self.plane_y_axis).tolist())
                   for u, v in chain]
            if len(pts) >= 2:
                wires.append(Polyline(*pts))
        if not wires:
            raise ValueError("Could not build any wires.")
        return wires[0] if len(wires) == 1 else Compound(children=wires)

    def _profile_fingerprint(self) -> bytes:
        """Hash of everything _build_faces_impl() reads: the plane and all
        non-construction profile geometry.  ReferenceEntity occ_edges are
        represented by their baked `points` cache — the same replay that
        changes the edges re-bakes the points."""
        import hashlib
        h = hashlib.blake2b(digest_size=16)
        for arr in (self.plane_origin, self.plane_x_axis,
                    self.plane_y_axis, self.plane_normal):
            h.update(np.asarray(arr, dtype=np.float64).tobytes())
        for ent in self.entities:
            if getattr(ent, "construction", False):
                continue
            if isinstance(ent, LineEntity):
                h.update(b'L')
                h.update(ent.p0.tobytes())
                h.update(ent.p1.tobytes())
            elif isinstance(ent, ArcEntity):
                h.update(b'A')
                h.update(np.array([ent.center[0], ent.center[1], ent.radius,
                                   ent.start_angle, ent.end_angle]).tobytes())
            elif isinstance(ent, SplineEntity):
                h.update(b'Sc' if ent.closed else b'So')
                for p in ent.points:
                    h.update(np.asarray(p, dtype=np.float64).tobytes())
            elif isinstance(ent, ReferenceEntity):
                h.update(b'R')
                h.update(len(ent.occ_edges or []).to_bytes(4, 'little'))
                for p in ent.points:
                    h.update(np.asarray(p, dtype=np.float64).tobytes())
        return h.digest()

    def build_faces(self, tol: float = 1e-3) -> list:
        """
        Cached front-end for _build_faces_impl().

        The splitter run costs ~0.5 s on a traced sketch and the result is
        requested repeatedly with identical geometry (face overlay rebuild on
        every history mutation, plus each extrude/revolve/loft replay), so
        results are memoized module-wide keyed by geometry fingerprint.
        """
        faces, regions, _ = self._build_faces_cached(tol)
        return list(faces), list(regions)

    def build_face_boundaries(self, tol: float = 1e-3) -> list[frozenset]:
        """Per face from build_faces(): the uids of the entities bounding it.

        This is a region's identity across parameter changes and solver
        re-solves, where centroids drift.  Ops store it next to
        face_centroids (see op_base._match_sketch_faces).
        """
        _, _, bounds = self._build_faces_cached(tol)
        profile = self._profile_entities()
        return [frozenset(profile[k].uid for k in b) for b in bounds]

    def _profile_entities(self) -> list:
        """The entities _profile_fingerprint() hashes, in its order."""
        return [e for e in self.entities
                if isinstance(e, (LineEntity, ArcEntity, SplineEntity, ReferenceEntity))
                and not getattr(e, "construction", False)]

    def _build_faces_cached(self, tol: float):
        key = (self._profile_fingerprint(), float(tol))
        hit = _FACE_BUILD_CACHE.get(key)
        if hit is not None:
            _FACE_BUILD_CACHE.move_to_end(key)
            return hit
        faces, regions, bounds = self._build_faces_impl(tol)
        # Cache boundaries as positions among the fingerprinted entities, not
        # raw indices: points/construction aren't in the key, so two sketches
        # sharing a cache entry can number their entities differently.
        ordinal = {id(e): k for k, e in enumerate(self._profile_entities())}
        bounds = [tuple(ordinal[id(self.entities[i])] for i in b) for b in bounds]
        hit = (faces, regions, bounds)
        _FACE_BUILD_CACHE[key] = hit
        while len(_FACE_BUILD_CACHE) > _FACE_BUILD_CACHE_MAX:
            _FACE_BUILD_CACHE.popitem(last=False)
        return hit

    def _build_faces_impl(self, tol: float = 1e-3) -> list:
        """
        Build one planar Face per enclosed region in the sketch.

        Strategy: feed every drawn edge individually as a wire tool into
        BRepAlgoAPI_Splitter against a large planar face.  OCCT figures out
        all topology — no Python loop detection needed.  ReferenceEntity closed
        loops are also added as wire tools.  The largest output face (the
        unbounded remainder of the big plane) is discarded; all others are kept.

        Returns (faces, regions, bounds) where regions[k] = (outer_uvs,
        [hole_uvs]) and bounds[k] = sorted indices into self.entities of the
        entities whose edges bound faces[k], all parallel to faces[k].
        """
        import math as _math
        from build123d import Face
        from OCP.BRepBuilderAPI import (BRepBuilderAPI_MakeFace,
                                        BRepBuilderAPI_MakeWire,
                                        BRepBuilderAPI_MakeEdge)
        from OCP.BRepAlgoAPI import BRepAlgoAPI_Splitter
        from OCP.BRepTools import BRepTools_WireExplorer
        from OCP.BRepGProp import BRepGProp
        from OCP.GProp import GProp_GProps
        from OCP.Geom import Geom_Circle as _GCircle
        from OCP.gp import gp_Pln, gp_Ax3, gp_Ax2, gp_Pnt, gp_Dir
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopAbs import TopAbs_FACE, TopAbs_WIRE
        from OCP.TopTools import TopTools_ListOfShape
        from OCP.TopoDS import TopoDS

        plane_pln = gp_Pln(gp_Ax3(
            gp_Pnt(float(self.plane_origin[0]),
                   float(self.plane_origin[1]),
                   float(self.plane_origin[2])),
            gp_Dir(float(self.plane_normal[0]),
                   float(self.plane_normal[1]),
                   float(self.plane_normal[2])),
            gp_Dir(float(self.plane_x_axis[0]),
                   float(self.plane_x_axis[1]),
                   float(self.plane_x_axis[2])),
        ))

        # Big planar face — large enough to contain all sketch geometry
        _BIG = 1e5
        big_face = BRepBuilderAPI_MakeFace(plane_pln, -_BIG, _BIG, -_BIG, _BIG).Face()

        def _uv3d(u, v):
            return gp_Pnt(*(self.plane_origin
                            + float(u) * self.plane_x_axis
                            + float(v) * self.plane_y_axis).tolist())

        # (tool edge, entity index) — lets each output face report which
        # entities bound it (see build_face_boundaries).
        tool_edges: list = []

        def _single_edge_wire(edge, ei):
            wm = BRepBuilderAPI_MakeWire()
            wm.Add(edge)
            if not wm.IsDone():
                return None
            tool_edges.append((edge, ei))
            return wm.Wire()

        tools_list = TopTools_ListOfShape()

        # ── Drawn LineEntity edges — each edge is its own wire tool ──────
        for ei, ent in enumerate(self.entities):
            # Construction geometry is reference-only — never part of a profile.
            if getattr(ent, "construction", False):
                continue
            if isinstance(ent, LineEntity):
                e = BRepBuilderAPI_MakeEdge(_uv3d(ent.p0[0], ent.p0[1]),
                                            _uv3d(ent.p1[0], ent.p1[1])).Edge()
                w = _single_edge_wire(e, ei)
                if w is not None:
                    tools_list.Append(w)

            elif isinstance(ent, ArcEntity):
                try:
                    cx, cy = float(ent.center[0]), float(ent.center[1])
                    c3d = (self.plane_origin
                           + cx * self.plane_x_axis
                           + cy * self.plane_y_axis)
                    ax2 = gp_Ax2(
                        gp_Pnt(*c3d.tolist()),
                        gp_Dir(float(self.plane_normal[0]),
                               float(self.plane_normal[1]),
                               float(self.plane_normal[2])),
                        gp_Dir(float(self.plane_x_axis[0]),
                               float(self.plane_x_axis[1]),
                               float(self.plane_x_axis[2])),
                    )
                    geom_circ = _GCircle(ax2, ent.radius)
                    span = ent.end_angle - ent.start_angle
                    if abs(span - 2 * _math.pi) < 1e-9 or abs(span) < 1e-9:
                        arc_edge = BRepBuilderAPI_MakeEdge(geom_circ).Edge()
                    else:
                        arc_edge = BRepBuilderAPI_MakeEdge(
                            geom_circ, ent.start_angle, ent.end_angle).Edge()
                    w = _single_edge_wire(arc_edge, ei)
                    if w is not None:
                        tools_list.Append(w)
                except Exception as ex:
                    print(f"[Sketch] build_faces: arc edge error — {ex}")

            elif isinstance(ent, SplineEntity):
                try:
                    from OCP.TColgp import TColgp_Array1OfPnt
                    from OCP.GeomAPI import GeomAPI_PointsToBSpline
                    pts = list(ent.points)
                    if ent.closed and pts:
                        pts = pts + [pts[0]]
                    if len(pts) >= 2:
                        arr = TColgp_Array1OfPnt(1, len(pts))
                        for i, p in enumerate(pts, start=1):
                            arr.SetValue(i, _uv3d(p[0], p[1]))
                        curve = GeomAPI_PointsToBSpline(arr).Curve()
                        sp_edge = BRepBuilderAPI_MakeEdge(curve).Edge()
                        w = _single_edge_wire(sp_edge, ei)
                        if w is not None:
                            tools_list.Append(w)
                except Exception as ex:
                    print(f"[Sketch] build_faces: spline edge error — {ex}")

        # ── ReferenceEntity closed loops — add as wire tools ─────────────
        def _project_ref_edge(edge):
            from OCP.BRepAdaptor import BRepAdaptor_Curve as _BAC
            from OCP.GeomAbs import GeomAbs_Circle, GeomAbs_Line
            adp = _BAC(edge)
            if adp.GetType() == GeomAbs_Line:
                # Project both endpoints onto the sketch plane so the included
                # line is coplanar with the drawn geometry.  Without this an
                # edge that lies even slightly off the plane (e.g. included from
                # another body) floats above/below the big planar face and the
                # splitter can't use it to bound a region — no face is built.
                p0 = adp.Value(adp.FirstParameter())
                p1 = adp.Value(adp.LastParameter())
                def _to_uv(p):
                    d = np.array([p.X(), p.Y(), p.Z()]) - self.plane_origin
                    return (float(np.dot(d, self.plane_x_axis)),
                            float(np.dot(d, self.plane_y_axis)))
                u0, v0 = _to_uv(p0)
                u1, v1 = _to_uv(p1)
                em = BRepBuilderAPI_MakeEdge(_uv3d(u0, v0), _uv3d(u1, v1))
                return em.Edge() if em.IsDone() else None

            if adp.GetType() == GeomAbs_Circle:
                circ = adp.Circle()
                # Check if circle is viewed edge-on (its normal is perpendicular
                # to the sketch normal) — if so, project as a line segment.
                circ_ax = circ.Axis().Direction()
                circ_normal = np.array([circ_ax.X(), circ_ax.Y(), circ_ax.Z()])
                dot_normals = abs(float(np.dot(circ_normal, self.plane_normal)))
                if dot_normals < 1e-3:
                    # Edge-on: the circle projects to a line of length 2*radius.
                    # The axis of the projected line is the cross product of
                    # circ_normal and sketch_normal, normalised to sketch UV.
                    line_dir_3d = np.cross(circ_normal, self.plane_normal)
                    ld_len = float(np.linalg.norm(line_dir_3d))
                    if ld_len < 1e-9:
                        return None
                    line_dir_3d /= ld_len
                    c3d = circ.Location()
                    cw  = np.array([c3d.X(), c3d.Y(), c3d.Z()])
                    delta = cw - self.plane_origin
                    cu = float(np.dot(delta, self.plane_x_axis))
                    cv = float(np.dot(delta, self.plane_y_axis))
                    # line_dir in UV space
                    ldu = float(np.dot(line_dir_3d, self.plane_x_axis))
                    ldv = float(np.dot(line_dir_3d, self.plane_y_axis))
                    r = circ.Radius()
                    p0 = _uv3d(cu - r * ldu, cv - r * ldv)
                    p1 = _uv3d(cu + r * ldu, cv + r * ldv)
                    em = BRepBuilderAPI_MakeEdge(p0, p1)
                    return em.Edge() if em.IsDone() else None

                c3d  = circ.Location()
                cw   = np.array([c3d.X(), c3d.Y(), c3d.Z()])
                delta = cw - self.plane_origin
                cu = float(np.dot(delta, self.plane_x_axis))
                cv = float(np.dot(delta, self.plane_y_axis))
                c_on = (self.plane_origin
                        + cu * self.plane_x_axis
                        + cv * self.plane_y_axis)
                ax2 = gp_Ax2(
                    gp_Pnt(*c_on.tolist()),
                    gp_Dir(float(self.plane_normal[0]),
                           float(self.plane_normal[1]),
                           float(self.plane_normal[2])),
                    gp_Dir(float(self.plane_x_axis[0]),
                           float(self.plane_x_axis[1]),
                           float(self.plane_x_axis[2])),
                )
                new_circ = _GCircle(ax2, circ.Radius())
                u1, u2 = adp.FirstParameter(), adp.LastParameter()
                if abs(u2 - u1 - 2 * _math.pi) < 1e-9:
                    return BRepBuilderAPI_MakeEdge(new_circ).Edge()
                return BRepBuilderAPI_MakeEdge(new_circ, u1, u2).Edge()
            return None

        for ei, ref in enumerate(self.entities):
            if not isinstance(ref, ReferenceEntity):
                continue
            valid_occ = [e for e in (ref.occ_edges or []) if e is not None]
            if valid_occ:
                for occ_edge in valid_occ:
                    projected = _project_ref_edge(occ_edge)
                    if projected is not None:
                        w = _single_edge_wire(projected, ei)
                        if w is not None:
                            tools_list.Append(w)
                    else:
                        # Polyline fallback for non-circle reference edges
                        try:
                            from OCP.BRepAdaptor import BRepAdaptor_Curve as _BAC2
                            from OCP.GCPnts import GCPnts_QuasiUniformAbscissa
                            adp = _BAC2(occ_edge)
                            sampler = GCPnts_QuasiUniformAbscissa()
                            sampler.Initialize(adp, 33)
                            pts_3d = []
                            for j in range(1, sampler.NbPoints() + 1):
                                p = adp.Value(sampler.Parameter(j))
                                pts_3d.append(gp_Pnt(p.X(), p.Y(), p.Z()))
                            for j in range(len(pts_3d) - 1):
                                em = BRepBuilderAPI_MakeEdge(pts_3d[j], pts_3d[j+1])
                                if em.IsDone():
                                    w = _single_edge_wire(em.Edge(), ei)
                                    if w is not None:
                                        tools_list.Append(w)
                        except Exception:
                            pass
            elif len(ref.points) >= 2:
                # Polyline from tessellated points
                pts = ref.points
                for j in range(len(pts) - 1):
                    em = BRepBuilderAPI_MakeEdge(_uv3d(pts[j][0], pts[j][1]),
                                                 _uv3d(pts[j+1][0], pts[j+1][1]))
                    if em.IsDone():
                        w = _single_edge_wire(em.Edge(), ei)
                        if w is not None:
                            tools_list.Append(w)

        if tools_list.IsEmpty():
            return [], [], []

        # ── Splitter: big face cut by all edge-wires ──────────────────────
        args_list = TopTools_ListOfShape()
        args_list.Append(big_face)
        splitter = BRepAlgoAPI_Splitter()
        splitter.SetArguments(args_list)
        splitter.SetTools(tools_list)
        # Drawn ends meet includes only to solver precision (~1e-7, or ~1e-5
        # for includes snapped to float32 mesh points in older files); an
        # exact splitter leaves those gaps open and the region vanishes.
        splitter.SetFuzzyValue(_SPLIT_FUZZ)
        splitter.Build()

        if not splitter.IsDone():
            print("[Sketch] build_faces: Splitter failed")
            return [], [], []

        # Every output edge → the entity whose tool edge it came from.
        from OCP.TopTools import TopTools_IndexedMapOfShape
        from OCP.TopAbs import TopAbs_EDGE
        edge_map = TopTools_IndexedMapOfShape()
        edge_owner: list[int] = []
        for tool_edge, ei in tool_edges:
            pieces = splitter.Modified(tool_edge)
            for piece in (pieces if pieces.Size() else [tool_edge]):
                if edge_map.Add(piece) > len(edge_owner):
                    edge_owner.append(ei)

        def _face_entities(occ_face):
            out = set()
            ex = TopExp_Explorer(occ_face, TopAbs_EDGE)
            while ex.More():
                k = edge_map.FindIndex(ex.Current())
                if k:
                    out.add(edge_owner[k - 1])
                ex.Next()
            return tuple(sorted(out))

        # ── Extract faces, discard the unbounded remainder ────────────────
        def _wire_uv_pts(wire):
            from OCP.BRepAdaptor import BRepAdaptor_Curve as _BAC2
            from OCP.GCPnts import GCPnts_QuasiUniformAbscissa
            from OCP.TopAbs import TopAbs_REVERSED
            pts = []
            we = BRepTools_WireExplorer(wire)
            while we.More():
                adp = _BAC2(we.Current())
                sampler = GCPnts_QuasiUniformAbscissa()
                sampler.Initialize(adp, 33)
                # WireExplorer yields edges in connected boundary order, but
                # Value() samples along the edge's own parameter direction.
                # When the edge sits REVERSED in the wire, emit its samples in
                # reverse so the polygon stays boundary-ordered — otherwise the
                # outline self-intersects and the shoelace area/containment test
                # (used for picking) is wrong for arcs/reversed segments.
                idx = range(1, sampler.NbPoints())
                if we.Orientation() == TopAbs_REVERSED:
                    idx = reversed(idx)
                for j in idx:
                    p3d = adp.Value(sampler.Parameter(j))
                    world = np.array([p3d.X(), p3d.Y(), p3d.Z()])
                    delta = world - self.plane_origin
                    pts.append((float(np.dot(delta, self.plane_x_axis)),
                                float(np.dot(delta, self.plane_y_axis))))
                we.Next()
            return pts

        def _face_area(occ_face):
            props = GProp_GProps()
            BRepGProp.SurfaceProperties_s(occ_face, props)
            return props.Mass()

        def _uv_area(uvs):
            a = 0.0
            for k in range(len(uvs)):
                x0, y0 = uvs[k]; x1, y1 = uvs[(k + 1) % len(uvs)]
                a += x0 * y1 - x1 * y0
            return abs(a) * 0.5

        # Collect all output faces with their areas
        raw_faces = []
        exp = TopExp_Explorer(splitter.Shape(), TopAbs_FACE)
        while exp.More():
            try:
                occ_face = TopoDS.Face_s(exp.Current())
                raw_faces.append((occ_face, _face_area(occ_face)))
            except Exception as ex:
                print(f"[Sketch] build_faces: face extraction error — {ex}")
            exp.Next()

        if not raw_faces:
            return [], [], []

        # Discard the largest face (unbounded big-plane remainder)
        raw_faces.sort(key=lambda x: x[1], reverse=True)
        bounded_faces = raw_faces[1:]  # drop the biggest

        faces   = []
        regions = []
        bounds  = []
        for occ_face, _ in bounded_faces:
            try:
                wire_uvs = []
                wexp = TopExp_Explorer(occ_face, TopAbs_WIRE)
                while wexp.More():
                    uvs = _wire_uv_pts(TopoDS.Wire_s(wexp.Current()))
                    if len(uvs) >= 3:
                        wire_uvs.append(uvs)
                    wexp.Next()
                if not wire_uvs:
                    continue
                wire_uvs.sort(key=_uv_area, reverse=True)
                faces.append(Face(occ_face))
                regions.append((wire_uvs[0], wire_uvs[1:]))
                bounds.append(_face_entities(occ_face))
            except Exception as ex:
                print(f"[Sketch] build_faces: region extraction error — {ex}")

        return faces, regions, bounds


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _chain_segments(
    segs: list[tuple[np.ndarray, np.ndarray]],
    tol: float = 1e-3,
) -> list[list[tuple[float, float]]]:
    """Greedily chain (p0, p1) segments into ordered polylines."""
    remaining = [(p0.copy(), p1.copy()) for p0, p1 in segs]
    chains: list[list[tuple[float, float]]] = []

    while remaining:
        p0, p1 = remaining.pop(0)
        chain = [(float(p0[0]), float(p0[1])),
                 (float(p1[0]), float(p1[1]))]
        changed = True
        while changed:
            changed = False
            for i, (a, b) in enumerate(remaining):
                tail = np.array(chain[-1])
                head = np.array(chain[0])
                if np.linalg.norm(a - tail) < tol:
                    chain.append((float(b[0]), float(b[1])))
                    remaining.pop(i); changed = True; break
                if np.linalg.norm(b - tail) < tol:
                    chain.append((float(a[0]), float(a[1])))
                    remaining.pop(i); changed = True; break
                if np.linalg.norm(b - head) < tol:
                    chain.insert(0, (float(a[0]), float(a[1])))
                    remaining.pop(i); changed = True; break
                if np.linalg.norm(a - head) < tol:
                    chain.insert(0, (float(b[0]), float(b[1])))
                    remaining.pop(i); changed = True; break
        chains.append(chain)

    return chains
