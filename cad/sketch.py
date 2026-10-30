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
# tol).  Entries hold OCCT faces.  Must exceed a document's sketch count:
# every history change rebuilds all sketches' faces in order, and a cyclic
# walk over more sketches than slots evicts each one before its reuse (8 slots
# vs headphones' 18 sketches → 100% misses, ~200 ms main-thread stall per op).
_FACE_BUILD_CACHE: OrderedDict = OrderedDict()
_FACE_BUILD_CACHE_MAX = 128


# ---------------------------------------------------------------------------
# Tool enum  — add a value here when adding a new tool
# ---------------------------------------------------------------------------

# apply_constraints_local: entity hops to widen through before the global solve.
_LOCAL_SOLVE_HOPS = 4


class SketchTool(Enum):
    NONE      = auto()
    LINE      = auto()
    CIRCLE    = auto()
    ARC       = auto()
    ARC3      = auto()   # 3-point arc
    TRIM      = auto()
    DIVIDE    = auto()
    JOIN      = auto()
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
        from OCP.collections import Array1_gp_Pnt as TColgp_Array1OfPnt
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
                   'gap' (ref, mov): distance between two parallel lines
                   'pt_distance' (a, b): distance from a point of a to a
                   point of b, or to line b (see points)
                   'edge_gap' (line, arc): line to the arc/circle's edge
                   'equal' (a, b): equal length — or radius, for two arcs
                   'concentric' (a, b): two arcs/circles share a centre
    indices      : tuple of entity indices the constraint applies to
    value        : numeric value (mm for distance/diameter; 0 for geometric)
    label_offset : for 'distance'  — perpendicular offset of label from line (mm)
                   for 'gap'       — slide of the label along the lines (mm)
                   for 'diameter'  — radial offset of label beyond circle edge (mm)
    label_angle  : for 'diameter' only — angle (radians) of the diameter chord
                   measured from the +U axis.  None = default (0 = horizontal).
    expr         : parameter expression driving *value* ("width/2"), or None.
                   value stays the baked result; replay recomputes it.
    expr_unit    : unit bare numbers in expr are read in (see cad/params.py).
    points       : for 'pt_distance' — which point of each entity:
                   'p0' | 'p1' | 'center' | 'mid', or None for "the line
                   itself" (perpendicular distance; second entity only).
    """
    def __init__(self, type: str, indices: tuple, value: float,
                 label_offset: float | None = None,
                 label_angle:  float | None = None,
                 expr: str | None = None, expr_unit: str = "mm",
                 points: tuple | None = None):
        self.type         = type
        self.indices      = tuple(indices)
        self.value        = float(value)
        self.label_offset: float | None = label_offset
        self.label_angle:  float | None = label_angle
        self.expr:         str | None   = expr
        self.expr_unit:    str          = expr_unit
        self.points:       tuple | None = tuple(points) if points else None


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


_LENGTH_CONSTRAINTS = ('distance', 'equal', 'sharp_distance')

# Constraints with no value: safe to drop when SolveSpace calls them
# redundant, provided they still hold afterwards (see apply_constraints).
_GEOMETRIC_CONSTRAINTS = ('horizontal', 'vertical', 'parallel',
                          'perpendicular', 'equal', 'concentric')


def sharp_corner(entities, i: int, n: int):
    """Where line i meets line n extended (the virtual sharp of a fillet),
    or None if n isn't a line or they're parallel."""
    if not (0 <= n < len(entities)) or not isinstance(entities[n], LineEntity):
        return None
    a, b = entities[i], entities[n]
    d, e = a.p1 - a.p0, b.p1 - b.p0
    den = float(d[0] * e[1] - d[1] * e[0])
    if abs(den) < 1e-12 * max(float(np.dot(d, d)) * float(np.dot(e, e)), 1e-24):
        return None
    w = b.p0 - a.p0
    t = float(w[0] * e[1] - w[1] * e[0]) / den
    return a.p0 + t * d


def sharp_ends(entities, con):
    """The two points a 'distance' / 'sharp_distance' measures between."""
    i = con.indices[0]
    ends = [entities[i].p0, entities[i].p1]
    if con.type == 'sharp_distance':
        for k, n in enumerate(con.indices[1:3]):
            v = sharp_corner(entities, i, n)
            if v is not None:
                ends[k] = v
    return ends


def ref_point(entities, i: int, which):
    """2-D position of point *which* ('p0' / 'p1' / 'center' / 'mid') of
    entity i, or None."""
    if not 0 <= i < len(entities):
        return None
    e = entities[i]
    if which == 'center':
        return np.asarray(e.center, float)[:2] if isinstance(e, ArcEntity) else None
    if which == 'mid':
        return ((np.asarray(e.p0, float) + np.asarray(e.p1, float)) / 2)[:2] \
            if isinstance(e, LineEntity) else None
    if which in ('p0', 'p1') and isinstance(e, (LineEntity, ArcEntity)):
        return np.asarray(getattr(e, which), float)[:2]
    return None


def pt_distance_ends(entities, con):
    """(A, B) a 'pt_distance' measures between — B is the foot on the
    line for a point-to-line distance — or None."""
    (a, b), (wa, wb) = con.indices, con.points or (None, None)
    A = ref_point(entities, a, wa)
    if A is None:
        return None
    if wb is None:
        if not (0 <= b < len(entities)) or not isinstance(entities[b], LineEntity):
            return None
        p0, p1 = np.asarray(entities[b].p0, float)[:2], np.asarray(entities[b].p1, float)[:2]
        d = p1 - p0
        t = float(np.dot(A - p0, d) / max(float(np.dot(d, d)), 1e-24))
        return A, p0 + t * d
    B = ref_point(entities, b, wb)
    return None if B is None else (A, B)


def edge_gap_ends(entities, con):
    """(foot on the line, nearest point of the circle) an 'edge_gap'
    measures between, or None."""
    li, ai = con.indices
    if not (0 <= li < len(entities) and 0 <= ai < len(entities)) or not \
            isinstance(entities[li], LineEntity) or \
            not isinstance(entities[ai], ArcEntity):
        return None
    L, A = entities[li], entities[ai]
    p0, p1 = np.asarray(L.p0, float)[:2], np.asarray(L.p1, float)[:2]
    c = np.asarray(A.center, float)[:2]
    d = p1 - p0
    foot = p0 + d * float(np.dot(c - p0, d) / max(float(np.dot(d, d)), 1e-24))
    v = foot - c
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return None
    return foot, c + v / n * A.radius


def _firm_lock(sketch, i, group) -> bool:
    """A lock kept in preference to lone-line direction locks: a tied
    group of lines, or an arc's radius."""
    return len(group) > 1 or isinstance(sketch.entities[i], ArcEntity)


def _is_full_circle(arc) -> bool:
    return abs(arc.end_angle - arc.start_angle - 2.0 * np.pi) < 1e-6


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
        pts = getattr(con, 'points', None)
        if pts and touched:
            idx = []
            for j, w in zip(con.indices, pts):
                if not start <= j < stop:
                    idx.append(_remap(j))
                elif w == 'center':
                    idx.append(_remap(j))
                elif w in ('p0', 'p1'):
                    k = _piece_near(old[j - start], getattr(old[j - start], w))
                    idx.append(None if k is None else start + k)
                else:   # 'mid' / the line itself: only if it wasn't split
                    idx.append(_remap(j) if len(replacement) == 1 and
                               stop - start == 1 else None)
            idx = tuple(idx)
        else:
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
            j = _remap(snap[0], getattr(ent, which))
            kind = getattr(snap[1], 'name', '')
            if j is not None and start <= snap[0] < stop and \
                    kind in ('MIDPOINT', 'NEAREST') and \
                    isinstance(ents[j], LineEntity):
                # The line was cut (split, or trimmed to one piece): the snap
                # means what it says only if it still holds on the new piece.
                kind = _snap_on_piece(getattr(ent, which), ents[j], kind)
                if kind is None:
                    j = None
                else:
                    from cad.sketch_tools.snap import SnapType
                    snap = (snap[0], SnapType[kind])
            setattr(ent, f'{which}_snap', None if j is None else (j, snap[1]))


def join_lines(sketch, a: int, b: int) -> int:
    """Replace lines *a* and *b* with one line between their far ends (the
    near ends — the pair closest together — are dropped).  A gap closes and
    a small angle difference averages out, while whatever the far ends are
    attached to stays put.  Line *a* is kept (same uid) and carries *b*'s
    constraints and the snaps that pointed at either; lengths on them are
    dropped, and a snap that no longer holds on the joined line goes (see
    _snap_on_piece).  Returns the joined line's index."""
    from cad.sketch_tools.snap import SnapType
    ents = sketch.entities
    A, B = ents[a], ents[b]
    ends = [(np.linalg.norm(pa - pb), wa, wb)
            for wa, pa in (('p0', A.p0), ('p1', A.p1))
            for wb, pb in (('p0', B.p0), ('p1', B.p1))]
    _d, near_a, near_b = min(ends, key=lambda t: t[0])
    far_a = 'p1' if near_a == 'p0' else 'p0'
    far_b = 'p1' if near_b == 'p0' else 'p0'
    p0, s0 = getattr(A, far_a).copy(), getattr(A, f'{far_a}_snap')
    p1, s1 = getattr(B, far_b).copy(), getattr(B, f'{far_b}_snap')
    A.p0, A.p1, A.p0_snap, A.p1_snap = p0, p1, s0, s1

    cons, seen = [], set()
    for con in getattr(sketch, 'constraints', []):
        if a in con.indices or b in con.indices:
            if con.type in _LENGTH_CONSTRAINTS or getattr(con, 'points', None):
                continue        # lengths / joined lines' points: meaningless now
            con.indices = tuple(a if j == b else j for j in con.indices)
            if len(con.indices) > 1 and len(set(con.indices)) < len(con.indices):
                continue
        key = (con.type, con.indices, getattr(con, 'points', None))
        if key in seen:
            continue
        seen.add(key)
        cons.append(con)
    if hasattr(sketch, 'constraints'):
        sketch.constraints[:] = cons

    # Lines attached along a or b stay attached to the joined line: their
    # end moves onto it (with every end sharing that point) — the joined
    # line may sit slightly off where a and b were.
    seg = A.p1 - A.p0
    seg_len2 = max(float(np.dot(seg, seg)), 1e-24)
    moves = []
    for k, ent in enumerate(ents):
        if not isinstance(ent, LineEntity) or k in (a, b):
            continue
        for which in ('p0', 'p1'):
            snap = getattr(ent, f'{which}_snap')
            if snap is None or snap[0] not in (a, b):
                continue
            kind = getattr(snap[1], 'name', '')
            if kind not in ('MIDPOINT', 'NEAREST'):
                setattr(ent, f'{which}_snap', (a, snap[1]))
                continue
            pt = getattr(ent, which)
            if kind == 'MIDPOINT' and snap[0] == a and \
                    np.linalg.norm(pt - (A.p0 + A.p1) / 2) < 1e-3:
                new = (A.p0 + A.p1) / 2
            else:
                kind = 'NEAREST'
                t = float(np.clip(np.dot(pt - A.p0, seg) / seg_len2, 0.0, 1.0))
                new = A.p0 + t * seg
            if min(np.linalg.norm(new - A.p0), np.linalg.norm(new - A.p1)) < 1e-3:
                setattr(ent, f'{which}_snap', None)   # landed on an end
            else:
                setattr(ent, f'{which}_snap', (a, SnapType[kind]))
            moves.append((pt.copy(), new))
    for old_pt, new in moves:
        for e in ents:
            if isinstance(e, LineEntity) and e is not A:
                for which in ('p0', 'p1'):
                    if np.linalg.norm(getattr(e, which) - old_pt) < 1e-3:
                        setattr(e, which, new.copy())

    splice_entities(sketch, b, b + 1, [])        # no refs to b remain
    return a if a < b else a - 1


def merge_residual_joints(sketch, near, tol: float = 1e-4) -> int:
    """Merge two collinear lines meeting at a bare joint into one line.

    A joint is "residual" when it only existed because something crossed
    there (Divide, or Trim splitting at the crossing) and that something is
    gone: exactly two drawn lines end at it, they are collinear, and nothing
    else touches it — no other end, no line or arc passing through, no
    point, no include.  Only joints on *near* (a callable pt → bool, e.g.
    "on the piece just trimmed away") are considered, so collinear segments
    the user drew on purpose elsewhere are left alone.  Returns merges done."""
    from cad.sketch_tools.snap import _nearest_on_segment, _nearest_on_arc
    merged = 0
    while True:
        ents = sketch.entities
        nodes = {}
        for i, e in enumerate(ents):
            if isinstance(e, LineEntity):
                for w in ('p0', 'p1'):
                    nodes.setdefault(tuple(np.round(getattr(e, w) / tol)), []).append(i)
        done = False
        for key, at in nodes.items():
            if len(at) != 2 or at[0] == at[1]:
                continue
            i, j = at
            A, B = ents[i], ents[j]
            pt = np.array(key) * tol
            if A.construction != B.construction or not near(pt):
                continue
            da, db = A.p1 - A.p0, B.p1 - B.p0
            la, lb = np.linalg.norm(da), np.linalg.norm(db)
            if la < tol or lb < tol or \
                    abs(float(da[0] * db[1] - da[1] * db[0])) / (la * lb) > 1e-6:
                continue
            touched = False
            for k, e in enumerate(ents):
                if k in (i, j):
                    continue
                if isinstance(e, LineEntity):
                    q = _nearest_on_segment(pt, e.p0, e.p1)
                elif isinstance(e, ArcEntity):
                    q, _ = _nearest_on_arc(pt, e)
                elif isinstance(e, PointEntity):
                    q = e.pos
                else:
                    r = ref_line(e) if isinstance(e, ReferenceEntity) else None
                    q = None if r is None else _nearest_on_segment(pt, *r)
                    if q is None and hasattr(e, 'points'):
                        touched = touched or any(np.linalg.norm(
                            np.asarray(p)[:2] - pt) < 1e-3 for p in e.points)
                        continue
                    if q is None:
                        continue
                if np.linalg.norm(np.asarray(q)[:2] - pt) < 1e-3:
                    touched = True
                    break
            if touched:
                continue
            join_lines(sketch, i, j)
            merged += 1
            done = True
            break
        if not done:
            return merged


def _snap_on_piece(pt, piece, kind, tol: float = 1e-3) -> str | None:
    """What a MIDPOINT / NEAREST snap at *pt* becomes on a cut line *piece*:
    'MIDPOINT' if pt is still its midpoint, 'NEAREST' if pt lies along it,
    None if pt is at one of its ends (the shared endpoint already joins
    them) or off it."""
    pt = np.asarray(pt, float)[:2]
    a, b = np.asarray(piece.p0, float)[:2], np.asarray(piece.p1, float)[:2]
    if min(np.linalg.norm(pt - a), np.linalg.norm(pt - b)) < tol:
        return None
    if kind == 'MIDPOINT' and np.linalg.norm(pt - (a + b) / 2) < tol:
        return 'MIDPOINT'
    d = b - a
    t = float(np.dot(pt - a, d) / max(float(np.dot(d, d)), 1e-24))
    if 0.0 < t < 1.0 and np.linalg.norm(a + t * d - pt) < tol:
        return 'NEAREST'
    return None


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

        # Grid hash of canonical points (cell = tol, 3x3 neighbourhood): same
        # result as scanning them all in registration order — the earliest
        # canonical point within tol wins — without the O(n²) scan that cost
        # seconds per solve on traced sketches.
        cells: dict[tuple, list] = {}
        n_canon = [0]

        def _register_pt(key, pt):
            cx, cy = int(np.floor(pt[0] / tol)), int(np.floor(pt[1] / tol))
            best = None
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for order, ck, op in cells.get((cx + dx, cy + dy), ()):
                        if (best is None or order < best[0]) and \
                                np.hypot(pt[0] - op[0], pt[1] - op[1]) < tol:
                            best = (order, ck)
            if best is not None:
                canon_map[key] = best[1]
                return
            canon_map[key] = key
            cells.setdefault((cx, cy), []).append((n_canon[0], key, pt))
            n_canon[0] += 1

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

        implied = {id(self.constraints[k]) for k in self._implied_direction_cons()}
        for con in constraints:
            if id(con) in implied:
                continue                      # repeat of other constraints
            i = con.indices[0] if con.indices else None
            if con.type == 'distance':
                if i is not None and i < len(self.entities) and \
                        isinstance(self.entities[i], LineEntity):
                    slvs.distance(_pt(i, 'p0'), _pt(i, 'p1'), con.value, wp)
            elif con.type == 'gap':
                ref, mov = con.indices
                if ref in slvs_lines and mov in slvs_lines:
                    slvs.distance(_pt(mov, 'p0'), slvs_lines[ref],
                                  self._gap_sign(ref, mov) * con.value, wp)
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
                elif a in slvs_arcs and b in slvs_arcs:
                    from python_solvespace import Constraint as K, Entity as E
                    slvs.add_constraint(K.EQUAL_RADIUS, wp, 0.0, E.NONE,
                                        E.NONE, slvs_arcs[a], slvs_arcs[b])
            elif con.type == 'concentric':
                a, b = con.indices
                if a in slvs_arcs and b in slvs_arcs and \
                        canon_map[(a, 'center')] != canon_map[(b, 'center')]:
                    slvs.coincident(_pt(a, 'center'), _pt(b, 'center'), wp)
            elif con.type == 'edge_gap':
                self._emit_edge_gap(slvs, wp, con, _pt, slvs_lines, slvs_arcs)
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
            elif con.type == 'pt_distance':
                self._emit_pt_distance(slvs, wp, con, _pt, slvs_lines)
            elif con.type == 'tangent':
                self._emit_tangent(slvs, wp, con, canon_map, slvs_pts,
                                   slvs_lines, slvs_arcs)
            elif con.type == 'sharp_distance':
                # Length to the virtual sharp corner(s) a fillet cut away:
                # a hidden point where the line meets its old neighbour.
                if i not in slvs_lines:
                    continue
                pts = []
                for w, n in zip(('p0', 'p1'), con.indices[1:]):
                    v = sharp_corner(self.entities, i, n) \
                        if n in slvs_lines else None
                    if v is None:
                        pts.append(_pt(i, w))
                        continue
                    V = slvs.add_point_2d(float(v[0]), float(v[1]), wp)
                    slvs.coincident(V, slvs_lines[i], wp)
                    slvs.coincident(V, slvs_lines[n], wp)
                    pts.append(V)
                slvs.distance(pts[0], pts[1], con.value, wp)

    def _emit_edge_gap(self, slvs, wp, con, _pt, slvs_lines, slvs_arcs) -> None:
        """('edge_gap', (line, arc)): a hidden point T on the circle, its
        radius ⊥ the line, T *value* from the line on the centre's side."""
        from python_solvespace import Constraint as K, Entity as E
        li, ai = con.indices
        ends = edge_gap_ends(self.entities, con)
        if ends is None or li not in slvs_lines or ai not in slvs_arcs:
            return
        L, c = self.entities[li], self.entities[ai].center
        t = ends[1]
        T = slvs.add_point_2d(float(t[0]), float(t[1]), wp)
        slvs.add_constraint(K.PT_ON_CIRCLE, wp, 0.0, T, E.NONE,
                            slvs_arcs[ai], E.NONE)
        slvs.perpendicular(slvs.add_line_2d(_pt(ai, 'center'), T, wp),
                           slvs_lines[li], wp)
        cross = (L.p1[0] - L.p0[0]) * (c[1] - L.p0[1]) - \
            (L.p1[1] - L.p0[1]) * (c[0] - L.p0[0])
        slvs.distance(T, slvs_lines[li],
                      (-1.0 if cross > 0 else 1.0) * con.value, wp)

    def _emit_pt_distance(self, slvs, wp, con, _pt, slvs_lines) -> None:
        """('pt_distance', (a, b), points=(wa, wb)): point to point, or
        point to line b when wb is None (kept on the side it is on now)."""
        if pt_distance_ends(self.entities, con) is None:
            return
        (a, b), (wa, wb) = con.indices, con.points

        def point(i, w):
            if w != 'mid':
                return _pt(i, w)
            if i not in slvs_lines:
                return None
            m = ref_point(self.entities, i, 'mid')
            M = slvs.add_point_2d(float(m[0]), float(m[1]), wp)
            slvs.midpoint(M, slvs_lines[i], wp)
            return M

        A = point(a, wa)
        if A is None:
            return
        if wb is None:
            if b not in slvs_lines:
                return
            L = self.entities[b]
            p = ref_point(self.entities, a, wa)
            cross = (L.p1[0] - L.p0[0]) * (p[1] - L.p0[1]) - \
                (L.p1[1] - L.p0[1]) * (p[0] - L.p0[0])
            slvs.distance(A, slvs_lines[b],
                          (-1.0 if cross > 0 else 1.0) * con.value, wp)
            return
        B = point(b, wb)
        if B is not None:
            slvs.distance(A, B, con.value, wp)

    def _emit_tangent(self, slvs, wp, con, canon_map, slvs_pts,
                      slvs_lines, slvs_arcs) -> None:
        """('tangent', (arc, other)) — other a line or arc/circle.

        Where the two touch decides the form (the python_solvespace wrappers
        always use the arc's START end, so the raw calls pick the end):
          arc end on the line (shared end or tied) → ARC_LINE_TANGENT
          shared end of two arcs                  → CURVE_CURVE_TANGENT
          line end tied onto the circle           → radius-to-that-end ⊥ line
          no contact point                        → hidden touch point T on
                                                     both; radius ⊥ line (or
                                                     the two radii parallel)"""
        from python_solvespace import Constraint as K, Entity as E
        a, b = con.indices
        ents = self.entities
        if not (0 <= a < len(ents) and 0 <= b < len(ents)) or a not in slvs_arcs:
            return
        if not (b in slvs_lines or b in slvs_arcs):
            return
        A, B = ents[a], ents[b]
        ck = lambda i, w: canon_map.get((i, w))
        a_full = _is_full_circle(A)
        b_arc = isinstance(B, ArcEntity)
        b_full = b_arc and _is_full_circle(B)

        def tied(i, which, target):
            snap = getattr(ents[i], f'{which}_snap', None) \
                if isinstance(ents[i], LineEntity) else None
            return bool(snap) and snap[0] == target and \
                getattr(snap[1], 'name', '') in ('NEAREST', 'TANGENT')

        def on_circle(pt_ent, arc_i):
            slvs.add_constraint(K.PT_ON_CIRCLE, wp, 0.0, pt_ent, E.NONE,
                                slvs_arcs[arc_i], E.NONE)

        center_a = slvs_pts[ck(a, 'center')]
        if not b_arc:
            line_cks = (ck(b, 'p0'), ck(b, 'p1'))
            if not a_full:
                for k, w in enumerate(('p0', 'p1')):
                    if ck(a, w) in line_cks:
                        slvs.add_constraint(K.ARC_LINE_TANGENT, wp, 0.0,
                                            E.NONE, E.NONE, slvs_arcs[a],
                                            slvs_lines[b], other=bool(k))
                        return
            for w in ('p0', 'p1'):
                if tied(b, w, a):
                    radius = slvs.add_line_2d(center_a, slvs_pts[ck(b, w)], wp)
                    slvs.perpendicular(radius, slvs_lines[b], wp)
                    return
            # Touch point: the foot of the centre on the line, out on the circle.
            p, q = B.p0, B.p1
            d = q - p
            t = float(np.dot(A.center - p, d) / max(float(np.dot(d, d)), 1e-24))
            foot = p + t * d
            n = foot - A.center
            nn = float(np.linalg.norm(n))
            n = n / nn if nn > 1e-12 else np.array([-d[1], d[0]]) / np.linalg.norm(d)
            t0 = A.center + n * A.radius
            T = slvs.add_point_2d(float(t0[0]), float(t0[1]), wp)
            slvs.coincident(T, slvs_lines[b], wp)
            on_circle(T, a)
            radius = slvs.add_line_2d(center_a, T, wp)
            slvs.perpendicular(radius, slvs_lines[b], wp)
            return

        center_b = slvs_pts[ck(b, 'center')]
        if not a_full and not b_full:
            for ka, wa in enumerate(('p0', 'p1')):
                for kb, wb in enumerate(('p0', 'p1')):
                    if ck(a, wa) == ck(b, wb):
                        slvs.add_constraint(K.CURVE_CURVE_TANGENT, wp, 0.0,
                                            E.NONE, E.NONE, slvs_arcs[a],
                                            slvs_arcs[b], other=bool(ka),
                                            other2=bool(kb))
                        return
        # Two circles: touch point on both, centres in line with it.
        n = B.center - A.center
        nn = float(np.linalg.norm(n))
        n = n / nn if nn > 1e-12 else np.array([1.0, 0.0])
        t0 = A.center + n * A.radius
        T = slvs.add_point_2d(float(t0[0]), float(t0[1]), wp)
        on_circle(T, a)
        on_circle(T, b)
        slvs.parallel(slvs.add_line_2d(center_a, T, wp),
                      slvs.add_line_2d(center_b, T, wp), wp)

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
            # Snap anchors (midpoint / on-line / center) are real constraints
            # in every solve; leaving them out overcounted DOF.
            self._add_snap_coincidences(slvs, wp, slvs_pts, slvs_lines,
                                        slvs_arcs, canon_map, self.constraints,
                                        implicit_lengths=False)
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
                               implicit_lengths=True, pinned=frozenset(),
                               held=None):
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
                if con.type in ('distance', 'sharp_distance'):
                    user_distanced.add(con.indices[0])

        # Two line ends sharing a point (a divided line's joint) can both be
        # snapped onto the same line: tie the point once, midpoint first.
        on_line_kind: dict = {}
        for i, ent in enumerate(self.entities):
            if not isinstance(ent, LineEntity):
                continue
            for which in ('p0', 'p1'):
                snap_meta = getattr(ent, f'{which}_snap', None)
                if snap_meta is None or snap_meta[1] not in (
                        SnapType.MIDPOINT, SnapType.NEAREST):
                    continue
                key = (canon_map.get((i, which)), snap_meta[0])
                if on_line_kind.get(key) != SnapType.MIDPOINT:
                    on_line_kind[key] = snap_meta[1]
        on_line_done: set = set()

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
                if held is not None and ck in held and all(
                        c in held for (j, _w), c in canon_map.items()
                        if j == src_idx):
                    continue        # all held: already satisfied, redundant
                if isinstance(src, LineEntity) and snap_type in (
                        SnapType.MIDPOINT, SnapType.NEAREST) and ck in (
                        canon_map.get((src_idx, 'p0')),
                        canon_map.get((src_idx, 'p1'))):
                    # The point IS an end of the line it is "on": a midpoint
                    # there is impossible, on-line is a no-op.  A stale snap
                    # (line trimmed after the snap was made) — skip it.
                    continue
                if snap_type in (SnapType.MIDPOINT, SnapType.NEAREST):
                    key = (ck, src_idx)
                    if key in on_line_done or on_line_kind.get(key) != snap_type:
                        continue
                    on_line_done.add(key)

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
                    if held is not None and ck in held and other_ck in held:
                        return  # both ends held: redundant
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
                        if ck in (canon_map.get((src_idx, 'p0')),
                                  canon_map.get((src_idx, 'p1'))):
                            continue    # it IS the arc's end: shared already
                        if snap_type == SnapType.ENDPOINT:
                            continue    # end moved off since: nothing to tie
                        # On the arc's circle (python_solvespace's coincident
                        # takes circles only; the raw call takes arcs too).
                        from python_solvespace import Constraint as _K, Entity as _E
                        slvs.add_constraint(_K.PT_ON_CIRCLE, wp, 0.0, ep_sp,
                                            _E.NONE, slvs_arcs[src_idx], _E.NONE)
                        circle_coincident_cks.add(ck)
                        _add_implicit_length()

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

    def apply_constraints_local(self, seed: set[int] | None = None,
                                pins: dict | None = None,
                                attachments: list | None = None) -> bool:
        """Solve after adding/editing a constraint the way a user expects:
        lines keep their angles and far-away geometry stays put.

        A plain minimum-motion solve spreads a change over every free point,
        and tilting or rotating things is often the "smallest" motion — an
        underconstrained box (perpendiculars, no horizontal) twists when its
        width dimension changes, instead of getting narrower.  So, in order:

          1. lines hold their current direction (one lock per group of lines
             tied by parallel/perpendicular — see _direction_locks), and
             only the seed's points move; widen one hop of connected points
             at a time
          2. the same widening without direction locks
          3. the global minimum-motion solve

        seed: entity indices the change is about (default: the last
        constraint's)."""
        if seed is None:
            seed = set(self.constraints[-1].indices) if self.constraints else set()
        seed = {i for i in seed if 0 <= i < len(self.entities)}
        self.solve_error = None
        last = len(self.constraints) - 1
        if last >= 0 and last in self._implied_direction_cons():
            self.solve_error = ("Constraint refused: it is already implied by "
                                "(or contradicts) other direction constraints")
            print(f"[Sketch] {self.solve_error}")
            return False
        # Lines the user tied with perpendicular/parallel keep their angle
        # unless the edit is about them: a value that can only be reached by
        # turning such a group is refused, not met by bending the sketch.
        guarded = {j for _i, g in self._direction_locks() if len(g) > 1
                   and not g & seed for j in g}
        dirs = {j: self._line_dir(j) for j in guarded}
        # Lines and arcs in tangent chains (fillets) must not turn inside
        # out: a dimension smaller than the fillets would reverse a line.
        tan = {j for c in self.constraints if c.type == 'tangent'
               for j in c.indices}
        tan_dirs = {j: self._line_dir(j) for j in tan}
        sweeps = {j: self.entities[j].end_angle - self.entities[j].start_angle
                  for j in tan if isinstance(self.entities[j], ArcEntity)}
        saved = [copy.deepcopy(e.__dict__) for e in self.entities]
        if not self._solve_local_stages(seed, pins, attachments):
            if self.solve_error is None:
                self.solve_error = ("Constraint refused: it conflicts with "
                                    "(or repeats) other constraints")
                print(f"[Sketch] {self.solve_error}")
            return False
        def _sin(d, e):
            return abs(float(d[0] * e[1] - d[1] * e[0]))
        turned = sorted(j for j, d in dirs.items() if d is not None
                        and _sin(d, self._line_dir(j)) > 1e-4)
        collapsed = [i for i, e in enumerate(self.entities)
                     if isinstance(e, ArcEntity) and e.radius < 1e-6
                     and saved[i].get('radius', 0.0) >= 1e-6]
        flipped = sorted(
            [j for j, d in tan_dirs.items() if d is not None and
             (self._line_dir(j) is None or float(d @ self._line_dir(j)) < 0)] +
            [j for j, sw in sweeps.items() if abs(
                (self.entities[j].end_angle - self.entities[j].start_angle) - sw)
             > np.pi])
        if collapsed:
            self.solve_error = (
                "Constraint refused: solving it shrinks arc "
                f"{', '.join(map(str, collapsed))} to nothing")
        elif flipped:
            self.solve_error = (
                "Constraint refused: it turns "
                f"{', '.join(map(str, flipped))} inside out (too small "
                "for the fillets/arcs on it)")
        elif turned:
            self.solve_error = (
                "Constraint refused: solving it turns lines "
                f"{', '.join(map(str, turned))}, which are tied by "
                "perpendicular/parallel constraints")
        elif self.compute_constraint_status()[0] == 'over':
            # A local attempt leaves out constraints among held points, so a
            # redundancy through the rest of the sketch only shows globally.
            self.solve_error = ("Constraint refused: it over-constrains the "
                                "sketch (already implied by other constraints)")
        else:
            return True
        for e, st in zip(self.entities, saved):
            e.__dict__.update(st)
        print(f"[Sketch] {self.solve_error}")
        return False

    def _line_dir(self, i):
        e = self.entities[i]
        if not isinstance(e, LineEntity):
            return None
        d = np.asarray(e.p1, float)[:2] - np.asarray(e.p0, float)[:2]
        n = float(np.linalg.norm(d))
        return d / n if n > 1e-12 else None

    def _solve_local_stages(self, seed, pins, attachments) -> bool:
        built = self._build_solver_system(pins) if seed else None
        if built is not None:
            canon_map = built[3]
            adj = self._point_adjacency(canon_map)
            start = {ck for (i, _w), ck in canon_map.items() if i in seed}
            rings = [set(start)]                 # rings[k]: free points at hop k
            for _hop in range(1, _LOCAL_SOLVE_HOPS):
                grown = rings[-1] | {q for p in rings[-1] for q in adj.get(p, ())}
                if grown == rings[-1]:
                    break
                rings.append(grown)
            locks = self._independent_locks(canon_map, rings, pins, attachments)
            # Lone lines' locks are the first to give way, then groups' and
            # arc radii.
            group_locks = [lk for lk in locks if _firm_lock(self, *lk)]
            tiers = [locks] + ([group_locks] if len(group_locks) < len(locks)
                               else []) + [None]
            for lock in tiers:
                for free in rings:
                    if self.apply_constraints(free_points=free, lock_lines=lock,
                                              pins=pins, attachments=attachments):
                        return True
        return self.apply_all_constraints(pins=pins, attachments=attachments)

    def _independent_locks(self, canon_map, rings, pins, attachments) -> list:
        """Direction locks for the lines the local solve may move, minus any
        that are implied by the others through a loop of constraints (two
        horizontal connectors between two rectangles) or that the new value
        makes impossible.  Found by dry-run solving with every ring point
        free: add locks far-from-the-edit first (distant geometry keeps its
        angles in preference) and bisect for the first one that breaks the
        solve, drop it, repeat.  A dropped redundant lock is implied by the
        kept ones, so nothing is lost."""
        region = rings[-1]
        level = {}
        for k, ring in enumerate(rings):
            for p in ring:
                level.setdefault(p, k)
        ent_level = {}
        for (i, _w), ck in canon_map.items():
            if ck in level:
                ent_level[i] = min(ent_level.get(i, 99), level[ck])
        cands = []
        for i, group in self._direction_locks() + self._radius_locks():
            near = [ent_level[j] for j in group if j in ent_level]
            if near:
                cands.append((min(near), i, group))
        # Groups the user tied with perpendicular/parallel, and arc radii,
        # are kept before lone lines: a box keeps its orientation and a
        # fillet its size; the plain connectors between boxes tilt instead.
        cands = [(i, g) for _lvl, i, g in
                 sorted(cands, key=lambda c: (not _firm_lock(self, c[1], c[2]),
                                              -c[0]))]

        def ok(lk):
            return self.apply_constraints(free_points=region, lock_lines=lk,
                                          pins=pins, attachments=attachments,
                                          write_back=False)
        if not cands or not ok([]):
            return []
        kept, rest = [], cands
        while rest and not ok(kept + rest):
            lo, hi = 0, len(rest)       # kept + rest[:lo] solves, [:hi] doesn't
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if ok(kept + rest[:mid]):
                    lo = mid
                else:
                    hi = mid
            kept += rest[:hi - 1]
            rest = rest[hi:]
        return kept + rest

    def _radius_locks(self) -> list[tuple[int, set[int]]]:
        """Arcs whose radius to hold in apply_constraints_local (same shape
        as _direction_locks).  Without it the minimum-motion solve happily
        resizes a fillet — or collapses it to a point — instead of moving
        lines.  Arcs with a diameter dimension are already held."""
        arcs = {i for i, e in enumerate(self.entities)
                if isinstance(e, ArcEntity) and e.radius > 1e-9}
        group = {i: {i} for i in arcs}          # equal radius → one size
        for c in self.constraints:
            if c.type == 'equal' and set(c.indices) <= arcs:
                a, b = c.indices
                if group[a] is not group[b]:
                    merged = group[a] | group[b]
                    for k in merged:
                        group[k] = merged
        sized = {c.indices[0] for c in self.constraints if c.type == 'diameter'}
        out, seen = [], set()
        for i in sorted(arcs):
            g = group[i]
            if id(g) in seen or g & sized:
                continue
            seen.add(id(g))
            out.append((i, set(g)))
        return out

    def _point_adjacency(self, canon_map) -> dict:
        """Canonical-point graph for apply_constraints_local: points of one
        entity, of two entities tied by a constraint, and a snapped endpoint
        with the entity it is snapped onto."""
        from collections import defaultdict
        keys: dict[int, set] = defaultdict(set)
        for (i, _w), ck in canon_map.items():
            keys[i].add(ck)
        adj: dict = defaultdict(set)

        def clique(pts):
            for a in pts:
                adj[a] |= pts - {a}

        for pts in keys.values():
            clique(pts)
        for c in self.constraints:
            if len(c.indices) >= 2:
                clique(set().union(*(keys[j] for j in c.indices if j >= 0)))
        for i, e in enumerate(self.entities):
            for which in ('p0', 'p1'):
                snap = getattr(e, f'{which}_snap', None)
                ck = canon_map.get((i, which))
                if snap is not None and ck is not None and snap[0] in keys:
                    clique({ck} | keys[snap[0]])
        return adj

    def _implied_direction_cons(self) -> set[int]:
        """Indices of direction constraints (horizontal / vertical /
        parallel / perpendicular) already implied by earlier ones — e.g.
        vertical(5) + perpendicular(5, 6) make horizontal(6) a repeat.

        SolveSpace's own redundancy test is numeric and unreliable on
        exactly-satisfied geometry: such a repeat sometimes passes, then
        poisons every later solve.  This is exact: union-find over lines
        tied by parallel/perpendicular, a group's direction fixed once it
        holds a horizontal/vertical or an include line.  A constraint inside
        one group, between two fixed groups, or fixing a fixed group adds
        nothing (or contradicts) — so it is never sent to the solver."""
        parent = {i: i for i in range(len(self.entities))}

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        fixed = {i for i, e in enumerate(self.entities)
                 if isinstance(e, ReferenceEntity)}
        implied = set()
        for k, c in enumerate(self.constraints):
            if any(not 0 <= i < len(self.entities) for i in c.indices):
                continue
            if c.type in ('parallel', 'perpendicular') and len(c.indices) == 2:
                a, b = (find(i) for i in c.indices)
                if a == b or (a in fixed and b in fixed):
                    implied.add(k)
                    continue
                parent[a] = b
                if a in fixed:
                    fixed.add(b)
            elif c.type in ('horizontal', 'vertical') and c.indices:
                r = find(c.indices[0])
                if r in fixed:
                    implied.add(k)
                fixed.add(r)
        return implied

    def _direction_locks(self) -> list[tuple[int, set[int]]]:
        """Lines whose current direction to hold in apply_constraints_local:
        one per group of lines whose angles are tied together (parallel /
        perpendicular), skipping groups already fixed by horizontal /
        vertical or an include.  One lock per group, so a lock never
        duplicates a user constraint (SolveSpace rejects redundancy).
        Returns [(locked line, its group)]."""
        lines = [i for i, e in enumerate(self.entities) if isinstance(e, LineEntity)]
        parent = {i: i for i in range(len(self.entities))}

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        fixed = set()
        for c in self.constraints:
            if c.type in ('parallel', 'perpendicular') and len(c.indices) == 2:
                a, b = (find(i) for i in c.indices)
                parent[a] = b
        for c in self.constraints:
            if c.type in ('horizontal', 'vertical'):
                fixed.add(find(c.indices[0]))
        for i, e in enumerate(self.entities):
            if isinstance(e, ReferenceEntity):
                fixed.add(find(i))
        groups: dict[int, set[int]] = {}
        for i in lines:
            groups.setdefault(find(i), set()).add(i)
        locks = []
        for r, group in groups.items():
            if r in fixed:
                continue
            i = next((i for i in sorted(group) if np.linalg.norm(
                self.entities[i].p1 - self.entities[i].p0) > 1e-9), None)
            if i is not None:
                locks.append((i, group))
        return locks

    def _gap_sign(self, ref: int, mov: int) -> float:
        """libslvs' point-line distance is signed in a workplane (+ = right
        of ref's direction); keep mov on the side it is on now."""
        a, b = self.entities[ref].p0, self.entities[ref].p1
        p = self.entities[mov].p0
        cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        return -1.0 if cross > 0 else 1.0

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
        elif con.type in ('parallel', 'perpendicular', 'equal', 'gap'):
            mov = con.indices[1]
            if mov < len(self.entities) and isinstance(self.entities[mov], LineEntity):
                free.add(canon_map[(mov, 'p0')])
                free.add(canon_map[(mov, 'p1')])
        elif con.type in ('horizontal', 'vertical'):
            i = con.indices[0]
            if i < len(self.entities) and isinstance(self.entities[i], LineEntity):
                free.add(canon_map[(i, 'p1')])
        elif con.type in ('pt_distance', 'edge_gap', 'concentric'):
            mov = con.indices[1]
            for w in ('p0', 'p1', 'center'):
                if (mov, w) in canon_map:
                    free.add(canon_map[(mov, w)])
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

    def _constraint_holds(self, con, slvs, canon_map, slvs_pts, ref_geo,
                          arc_dist_ents=None) -> bool:
        """Whether a geometric constraint is satisfied by the solved points."""
        if con.type in ('equal', 'concentric') and all(
                0 <= i < len(self.entities) and
                isinstance(self.entities[i], ArcEntity) for i in con.indices):
            def at(i, w):
                return np.array(slvs.params(
                    slvs_pts[canon_map[(i, w)]].params)[:2])

            def radius(i):
                if arc_dist_ents and i in arc_dist_ents:
                    return float(slvs.params(arc_dist_ents[i].params)[0])
                return float(np.linalg.norm(at(i, 'p0') - at(i, 'center')))
            a, b = con.indices
            if con.type == 'concentric':
                return float(np.linalg.norm(at(a, 'center') - at(b, 'center'))) < 1e-6
            return abs(radius(a) - radius(b)) < 1e-6 * max(1.0, radius(a))
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
                          attachments: list | None = None,
                          free_points: set | None = None,
                          lock_lines: list | None = None,
                          write_back: bool = True) -> bool:
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
        free_points: hold every canonical point not in this set, and
              lock_lines: [(line, its group)] whose current directions to
              hold (both: see apply_constraints_local).  No fallbacks —
              fails fast so the caller can widen / relax.
        write_back: False = only report whether it solves.
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
            # Local attempt: a constraint among points that are all held is
            # already satisfied and can't change — and emitting it on top of
            # the holds is redundant, which SolveSpace rejects outright.
            held = None
            if free_points is not None:
                held = set(slvs_pts) - set(free_points)
                ent_cks: dict[int, set] = {}
                for (i, _w), ck in canon_map.items():
                    ent_cks.setdefault(i, set()).add(ck)
                held_ents = {i for i in range(len(self.entities))
                             if ent_cks.get(i, set()) <= held}
            circ_cks = self._add_snap_coincidences(
                slvs, wp, slvs_pts, slvs_lines, slvs_arcs, canon_map,
                self.constraints, implicit_lengths=implicit_lengths,
                pinned=frozenset(pins) | {k for k, *_ in attachments},
                held=held)
            no_drag = set(attached_cks)
            # Pinned points without a hard attachment: dragged() keeps them
            # at their start (the target) when it can.
            for key in pins:
                ck = canon_map.get(key)
                if ck is None or ck not in slvs_pts or ck in no_drag:
                    continue
                slvs.dragged(slvs_pts[ck], wp)
                no_drag.add(ck)

            if free_points is not None:
                # Held arcs keep their size.  A partial arc is fixed by its
                # ends + radius: holding the centre too is one equation too
                # many (the arc keeps its end on its circle by itself) and
                # SolveSpace rejects the system.  A held circle's centre
                # doesn't fix its radius, so that is held too.  One radius
                # hold per equal-radius group (see _radius_locks); sized
                # groups are held by their diameter constraint.
                centre_uses: dict = {}
                for i, ent in enumerate(self.entities):
                    if isinstance(ent, ArcEntity):
                        ck = canon_map[(i, 'center')]
                        centre_uses[ck] = centre_uses.get(ck, 0) + 1
                lock_group = {j: frozenset(g) for _i, g in self._radius_locks()
                              for j in g}
                radius_held, held_groups = set(), set()
                for i, ent in enumerate(self.entities):
                    if not isinstance(ent, ArcEntity):
                        continue
                    full = _is_full_circle(ent)
                    cks = [canon_map[(i, w)]
                           for w in (('center',) if full else ('center', 'p0', 'p1'))]
                    if any(ck in free_points or ck in circ_cks or ck in no_drag
                           for ck in cks):
                        continue
                    if not full:
                        if centre_uses[cks[0]] > 1:
                            continue        # shared centre: drag as before
                        no_drag.add(cks[0])
                        # (A semicircle's diameter is its chord: singular —
                        # leave the centre to min-motion there.)
                        if abs(ent.end_angle - ent.start_angle - np.pi) < 1e-3:
                            continue
                    g = lock_group.get(i)
                    if g is None or g in held_groups:
                        continue
                    slvs.diameter(slvs_arcs[i], 2.0 * float(ent.radius))
                    held_groups.add(g)
                    radius_held |= g
                for ck, sp in slvs_pts.items():
                    if ck not in free_points and ck not in circ_cks and ck not in no_drag:
                        slvs.dragged(sp, wp)
                for i, group in lock_lines or ():
                    if group & held_ents or i in radius_held:
                        continue            # a held line already fixes the group
                    if i in slvs_arcs:
                        # Radius hold: the arc moves or turns, keeps its size.
                        slvs.diameter(slvs_arcs[i],
                                      2.0 * float(self.entities[i].radius))
                        continue
                    if i not in slvs_lines:
                        continue
                    d = self.entities[i].p1 - self.entities[i].p0
                    if abs(d[1]) < 1e-9:
                        slvs.horizontal(slvs_lines[i], wp)
                    elif abs(d[0]) < 1e-9:
                        slvs.vertical(slvs_lines[i], wp)
                    else:
                        a = slvs.add_point_2d(0.0, 0.0, wp)
                        b = slvs.add_point_2d(float(d[0]), float(d[1]), wp)
                        slvs.dragged(a, wp); slvs.dragged(b, wp)
                        slvs.parallel(slvs_lines[i], slvs.add_line_2d(a, b, wp), wp)
            elif free_for_last and self.constraints:
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
                # SolveSpace return result=1, so dedupe.  A centre a point
                # distance measures from is what that distance moves.
                measured = {canon_map.get((i, w)) for c in self.constraints
                            if c.type == 'pt_distance'
                            for i, w in zip(c.indices, c.points or ())
                            if w == 'center'}
                measured |= {canon_map.get((c.indices[1], 'center'))
                             for c in self.constraints if c.type == 'edge_gap'}
                # Concentric circles: pin one centre per group.
                for c in self.constraints:
                    if c.type == 'concentric' and all(
                            (i, 'center') in canon_map for i in c.indices):
                        measured.add(canon_map[(c.indices[1], 'center')])
                for i, ent in enumerate(self.entities):
                    if not isinstance(ent, ArcEntity):
                        continue
                    ck = canon_map.get((i, 'center'))
                    if ck and ck in slvs_pts and ck not in no_drag \
                            and ck not in measured:
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

            cons = [c for k, c in enumerate(self.constraints) if k not in drop_con
                    and not (held is not None
                             and all(i in held_ents for i in c.indices)
                             # a held arc's size isn't held by its points
                             and c.type != 'diameter'
                             and not (c.type == 'equal' and all(
                                 isinstance(self.entities[i], ArcEntity)
                                 for i in c.indices)))]
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
            slvs, _, canon_map, slvs_pts, _, _, arc_dist_ents, ref_geo, \
                att_checks = state
            ok = all(chk(slvs) for k, chk in att_checks.items() if k in drop_att)
            for k in drop_con:
                ok = ok and self._constraint_holds(
                    self.constraints[k], slvs, canon_map, slvs_pts, ref_geo,
                    arc_dist_ents)
            return ok

        result, state = _attempt(implicit_lengths=True)
        if result is None:
            return False
        if result != 0 and free_points is not None:
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
        if not write_back:
            return True

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
        from OCP.collections import List_TopoDS_Shape as TopTools_ListOfShape
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
                    from OCP.collections import Array1_gp_Pnt as TColgp_Array1OfPnt
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
        from OCP.collections import (
            IndexedMap_TopoDS_Shape_TopTools_ShapeMapHasher as TopTools_IndexedMapOfShape)
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
                occ_face = TopoDS.Face(exp.Current())
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
                    uvs = _wire_uv_pts(TopoDS.Wire(wexp.Current()))
                    if len(uvs) >= 3:
                        wire_uvs.append(uvs)
                    wexp.Next()
                if not wire_uvs:
                    continue
                wire_uvs.sort(key=_uv_area, reverse=True)
                faces.append(Face(_drop_open_wires(occ_face)))
                regions.append((wire_uvs[0], wire_uvs[1:]))
                bounds.append(_face_entities(occ_face))
            except Exception as ex:
                print(f"[Sketch] build_faces: region extraction error — {ex}")

        return faces, regions, bounds


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _drop_open_wires(occ_face):
    """The splitter leaves a dangling sketch line that ends inside a region
    (e.g. a duplicate or overhanging include) on that face as an open wire
    holding one INTERNAL edge.  The face still checks valid, but extruding
    it gives an invalid solid (an internal face), and booleans against that
    tool return garbage: inside-out (negative-volume) and phantom solids
    that a cut then keeps as split bodies.  An open wire bounds nothing and
    region extraction already ignores it, so dropping it changes neither
    the face's area nor face numbering."""
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_WIRE
    from OCP.ShapeBuild import ShapeBuild_ReShape
    from OCP.BRep import BRep_Tool
    from OCP.TopoDS import TopoDS
    reshape = None
    wexp = TopExp_Explorer(occ_face, TopAbs_WIRE)
    while wexp.More():
        w = TopoDS.Wire(wexp.Current())
        if not BRep_Tool.IsClosed_s(w):
            reshape = reshape or ShapeBuild_ReShape()
            reshape.Remove(w)
        wexp.Next()
    return TopoDS.Face(reshape.Apply(occ_face)) if reshape else occ_face


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
