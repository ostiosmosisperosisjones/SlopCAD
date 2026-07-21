"""
viewer/viewport.py

Viewport — the main GL widget.  Thin coordinator; logic lives in:

  viewer/renderer.py       — draw_opaque, draw_overlays
  viewer/hover.py          — HoverState
  viewer/sketch_overlay.py — SketchOverlay
  viewer/sketch_pick.py    — SketchPickMixin  (sketch face rebuild/pick/draw)
  viewer/sketch_modal.py   — SketchModalMixin (enter/exit/commit sketch)
  viewer/vp_history.py     — HistoryMixin     (undo/redo/seek/delete/reorder)
  viewer/vp_extrude.py     — ExtrudeMixin     (extrude/cut panel and dispatch)
"""

from __future__ import annotations
import numpy as np
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtCore import Qt, pyqtSignal
from OpenGL.GL import *
from OpenGL.GLU import *

from viewer.camera import Camera, _quat_to_matrix
from viewer.mesh import Mesh
from viewer.renderer import draw_opaque, draw_overlays
from viewer.hover import HoverState
from viewer.view_cube import ViewCube
from viewer.sketch_pick import SketchPickMixin
from viewer.sketch_modal import SketchModalMixin
from viewer.vp_history import HistoryMixin
from viewer.vp_extrude import ExtrudeMixin
from viewer.vp_thicken import ThickenMixin
from viewer.vp_revolve import RevolveMixin
from viewer.vp_async import AsyncOpMixin
from viewer.vp_offset import VpOffsetMixin
from viewer.vp_pattern import VpPatternMixin
from viewer.vp_fillet import VpFilletMixin
from viewer.vp_fillet3d import Fillet3DMixin
from viewer.vp_offset_plane import OffsetPlaneMixin
from viewer.vp_loft import LoftMixin
from viewer.vp_chamfer import ChamferMixin
from viewer.vp_boolean import BooleanMixin
from cad.workspace import Workspace
from cad.history import History
from cad.selection import SelectionSet


class Viewport(AsyncOpMixin, SketchPickMixin, SketchModalMixin, HistoryMixin, ExtrudeMixin, ThickenMixin, RevolveMixin, VpOffsetMixin, VpPatternMixin, VpFilletMixin, Fillet3DMixin, OffsetPlaneMixin, LoftMixin, ChamferMixin, BooleanMixin, QOpenGLWidget):
    history_changed     = pyqtSignal()
    selection_changed   = pyqtSignal()
    sketch_mode_changed = pyqtSignal(bool)
    body_selected       = pyqtSignal(object)   # str | None

    def __init__(self, workspace: Workspace, history: History):
        super().__init__()
        # Request MSAA on this widget's own surface format. Setting only the
        # app-wide default format is unreliable on some GLX/Mesa drivers, so we
        # set it here too. Must happen before the GL context is created (i.e.
        # before first show), which the constructor guarantees.
        from cad.prefs import prefs as _prefs
        if _prefs.msaa_samples and _prefs.msaa_samples > 0:
            from PyQt6.QtGui import QSurfaceFormat
            fmt = self.format()
            fmt.setSamples(int(_prefs.msaa_samples))
            self.setFormat(fmt)
        self.workspace = workspace
        self.history   = history
        self.camera    = Camera()
        self._view_cube = ViewCube()

        self._meshes: dict[str, Mesh] = {}
        self._body_visible: dict[str, bool] = {}

        self._modelview  = None
        self._projection = None
        self._viewport   = None
        self._dim_labels: list = []   # dimension label descriptors for QPainter pass
        self._dragging_label: dict | None = None  # label being dragged (stable ref)
        self._drag_start_screen: tuple | None = None
        # Stable drag state — set on press, used across move events
        self._drag_constraint   = None   # the actual SketchConstraint object
        self._drag_perp_world   = None   # perp world direction np.ndarray
        self._drag_line_mid     = None   # world pos of line midpoint (no offset)
        # Diameter-specific drag state
        self._drag_is_diameter      = False
        self._drag_chord_world      = None   # chord direction world np.ndarray
        self._drag_arc_center_world = None   # arc center in world coords
        self._drag_arc_radius       = None   # arc radius in mm (sketch coords)

        from viewer.line_hud import LineHUD
        self._line_hud = LineHUD(self)
        self._line_hud.draw_requested.connect(self._on_hud_draw)

        self.selection = SelectionSet()
        self.hover     = HoverState()

        self._hovered_vertex: tuple[str | None, int | None] = (None, None)
        self._hovered_edge:   tuple[str | None, int | None] = (None, None)

        from viewer.history_commands import CommandStack
        self._cmd_stack = CommandStack()

        self._sketch = None                              # SketchMode | None
        self._sketch_faces: dict[int, list] = {}        # history_idx → [Face]
        self._selected_sketch_entry: int | None = None  # history_idx
        self._selected_sketch_face: list[int] | None = None   # face indices within entry
        self._editing_sketch_history_idx: int | None = None  # set during re-entry
        self._include_armed = False                     # click-to-include mode

        self.camera_projection_changed = None
        self.request_extrude_distance  = None
        self._offset_panel         = None
        self._offset_preview       = []     # ghosted offset result entities
        self._pattern_panel        = None
        self._fillet_panel         = None
        self._offset_plane_panel        = None
        self._offset_plane_face_active  = False
        self._offset_plane_preview      = None
        self._editing_offset_plane_idx  = None
        self._hovered_plane_idx         = None
        self._selected_plane_idx        = None
        self._loft_panel               = None
        self._loft_profile_pick_active = False
        self._loft_body_pick_active    = False
        self._loft_preview_mesh        = None
        self._loft_preview_is_cut      = False
        self._editing_loft_idx         = None
        self._chamfer_panel            = None
        self._chamfer_body_id          = None
        self._chamfer_face_indices     = []
        self._chamfer_edge_indices     = []
        self._chamfer_preview_mesh     = None
        self._chamfer_preview_token    = None
        self._chamfer_computing        = False
        self._chamfer_pending          = None
        self._chamfer_pick_face        = False
        self._chamfer_pick_edge        = False
        self._editing_chamfer_idx      = None
        self._boolean_panel            = None
        self._boolean_body_ids             = []
        self._boolean_tool_ids             = []
        self._boolean_operation            = None
        self._boolean_keep_inputs          = False
        self._boolean_preview_mesh         = None
        self._boolean_pick_active          = False
        self._boolean_pick_target          = False
        self._boolean_pick_tool            = False
        self._editing_boolean_idx          = None
        self._extrude_panel        = None
        self._extrude_pick_active  = False
        self._extrude_vtx_active   = False
        self._extrude_body_active  = False
        self._extrude_face_active  = False
        self._extrude_sketch_idx   = None
        self._extrude_preview_mesh = None   # list of build123d solids | None
        self._extrude_preview_buf  = None   # (solids, (tri_verts, edge_verts)) cache
        self._extrude_preview_dist = 0.0
        self._extrude_arrow_origin = None
        self._extrude_arrow_dir    = None
        self._drag_arrow_active      = False  # True while dragging any op arrow
        self._drag_arrow_axis_origin = None  # world point on drag axis
        self._drag_arrow_op          = None  # 'extrude' | 'thicken' | 'revolve' | 'fillet3d' | 'chamfer'
        self._drag_arrow_grab_offset = 0.0   # s_at_click - panel_value; subtracted during drag
        self._revolve_preview_mesh   = None
        self._revolve_arrow_origin   = None
        self._revolve_arrow_dir      = None
        self._revolve_axis_point     = None  # stored for drag projection
        self._revolve_axis_dir       = None

        self._thicken_panel               = None
        self._thicken_body_id             = None
        self._thicken_preview_mesh        = None
        self._thicken_preview_dist        = 0.0
        self._thicken_arrow_origin        = None
        self._thicken_arrow_dir           = None
        self._thicken_cage_wires          = None
        self._thicken_cage_key            = None
        self._thicken_last_preview_params = None
        self._thicken_mesh_gen            = 0
        self._thicken_mesh_inflight       = False
        self._thicken_mesh_pending        = None
        self._editing_thicken_idx         = None

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)

        from PyQt6.QtWidgets import QApplication
        QApplication.instance().installEventFilter(self)

    def eventFilter(self, obj, e):
        from PyQt6.QtCore import QEvent
        from PyQt6.QtWidgets import QLineEdit, QTextEdit
        if e.type() == QEvent.Type.KeyPress:
            from PyQt6.QtWidgets import QApplication
            if QApplication.activeModalWidget() is not None:
                return False
            focused = self.__class__._focused_widget()
            # Don't intercept Tab while the line HUD is visible
            if (e.key() == Qt.Key.Key_Tab and
                    getattr(self, '_line_hud', None) is not None and
                    self._line_hud.isVisible()):
                return False
            # Only intercept if the key matches a bound action for the current mode
            from cad.prefs import prefs, KEYBINDS_3D, KEYBINDS_SKETCH
            active_binds = KEYBINDS_SKETCH if self._sketch is not None else KEYBINDS_3D
            if not any(prefs.matches(action, e) for action in active_binds):
                return False
            # Don't steal bound keys from actual text fields (HUD fields, search bars)
            if isinstance(focused, (QLineEdit, QTextEdit)):
                hud_visible = (getattr(self, '_line_hud', None) is not None and
                               self._line_hud.isVisible())
                if hud_visible and focused in (self._line_hud._len_field,
                                               self._line_hud._ang_field):
                    return False
            self.keyPressEvent(e)
            return True
        return False

    @staticmethod
    def _focused_widget():
        from PyQt6.QtWidgets import QApplication
        return QApplication.focusWidget()

    # ------------------------------------------------------------------
    # Backward-compat shims
    # ------------------------------------------------------------------

    @property
    def picked_body_id(self) -> str | None:
        sf = self.selection.single_face
        return sf.body_id if sf else None

    @property
    def picked_face(self) -> int | None:
        sf = self.selection.single_face
        return sf.face_idx if sf else None

    @property
    def picked_face_obj(self):
        sf = self.selection.single_face
        if sf is None:
            return None
        mesh = self._meshes.get(sf.body_id)
        return mesh.occt_faces[sf.face_idx] if mesh else None

    # ------------------------------------------------------------------
    # Mesh management
    # ------------------------------------------------------------------

    def build_meshes(self):
        self.makeCurrent()
        gl_ready = self.context() is not None and self.context().isValid()
        for body_id, shape in self.workspace.all_current_shapes():
            mesh = Mesh(shape)
            if gl_ready:
                mesh.upload()
            self._meshes[body_id] = mesh
            body = self.workspace.bodies.get(body_id)
            self._body_visible.setdefault(
                body_id, body.visible if body is not None else True)
        self.doneCurrent()

    def set_body_visible(self, body_id: str, visible: bool):
        self._body_visible[body_id] = visible
        body = self.workspace.bodies.get(body_id)
        if body is not None:
            body.visible = visible
        self.update()

    def set_world_plane_visible(self, axis: str, visible: bool):
        self.workspace.world_plane_visible[axis] = visible
        self.update()

    def set_active_body(self, body_id: str | None):
        self.selection.clear()
        if body_id is not None:
            self.workspace.set_active_body(body_id)
        self.selection_changed.emit()
        self.update()

    def _visible_meshes(self) -> dict:
        meshes = {bid: m for bid, m in self._meshes.items()
                  if self._body_visible.get(bid, True)}
        # Fillet3D preview: swap source body mesh with preview mesh
        preview = getattr(self, '_fillet3d_preview_mesh', None)
        body_id = getattr(self, '_fillet3d_body_id', None)
        if preview is not None and body_id is not None and body_id in meshes:
            meshes = dict(meshes)
            meshes[body_id] = preview
        # Chamfer preview: same pattern
        ch_preview = getattr(self, '_chamfer_preview_mesh', None)
        ch_body    = getattr(self, '_chamfer_body_id', None)
        if ch_preview is not None and ch_body is not None and ch_body in meshes:
            meshes = dict(meshes)
            meshes[ch_body] = ch_preview
        # Boolean preview: swap first body mesh with preview mesh
        bool_preview = getattr(self, '_boolean_preview_mesh', None)
        bool_body_ids = getattr(self, '_boolean_body_ids', [])
        bool_tool_ids = getattr(self, '_boolean_tool_ids', [])
        bool_op = getattr(self, '_boolean_operation', None)
        if bool_preview is not None:
            if bool_op == "subtract" and bool_body_ids:
                # Subtract: preview replaces target body
                target = bool_body_ids[0]
                if target in meshes:
                    meshes = dict(meshes)
                    meshes[target] = bool_preview
            elif bool_body_ids:
                # Union/intersect: preview replaces the first body
                first = bool_body_ids[0]
                if first in meshes:
                    meshes = dict(meshes)
                    meshes[first] = bool_preview
        return meshes

    def fit_camera_to_scene(self):
        # Fit what the user can see; fall back to all meshes if everything
        # is hidden (visibility is cosmetic — meshes exist for every body).
        meshes = self._visible_meshes() or self._meshes
        if not meshes:
            # Empty scene — set a default scale suitable for mm-unit CAD work
            self.camera.fit_scene([-100, -100, -100], [100, 100, 100])
            return
        all_min = np.array([ np.inf,  np.inf,  np.inf])
        all_max = np.array([-np.inf, -np.inf, -np.inf])
        for mesh in meshes.values():
            all_min = np.minimum(all_min, mesh.bbox_min)
            all_max = np.maximum(all_max, mesh.bbox_max)
        self.camera.fit_scene(all_min, all_max)

    def _rebuild_body_mesh(self, body_id: str):
        shape = self.workspace.current_shape(body_id)
        _body = self.workspace.bodies.get(body_id)
        self._body_visible.setdefault(
            body_id, _body.visible if _body is not None else True)
        if shape is not None:
            try:
                mesh = Mesh(shape)   # tessellate off-GL (pure CPU)
            except Exception as ex:
                print(f"[Mesh] Could not tessellate body {body_id}: {ex}")
                mesh = None
        else:
            mesh = None
        self.makeCurrent()
        # Always drop the old mesh — whether we're replacing it or hiding
        # the body because shape is now None.  Keeping a stale mesh when
        # workspace.current_shape() returns None would let an errored body
        # remain visible in the viewport (which violates the parametric
        # invariant: an op's failure makes its downstream geometry vanish).
        old = self._meshes.pop(body_id, None)
        if old is not None:
            for buf in (old.vbo_verts, old.vbo_normals, old.vbo_edges, old.ebo):
                if buf is not None:
                    glDeleteBuffers(1, [buf])
        if mesh is not None:
            mesh.upload()
            self._meshes[body_id] = mesh
        self.selection.clear_for_body(body_id)
        if self._hovered_vertex[0] == body_id:
            self._hovered_vertex = (None, None)
        if self._hovered_edge[0] == body_id:
            self._hovered_edge = (None, None)
        self.doneCurrent()
        self._rebuild_sketch_faces()
        self.selection_changed.emit()
        self.update()

    def _rebuild_all_meshes(self):
        # Collect shapes first — tessellation is pure CPU and can run in parallel
        bodies = list(self.workspace.all_current_shapes())
        new_meshes = _tessellate_parallel(bodies)

        self.makeCurrent()
        for body_id, old in list(self._meshes.items()):
            for buf in (old.vbo_verts, old.vbo_normals, old.vbo_edges, old.ebo):
                if buf is not None:
                    glDeleteBuffers(1, [buf])
        self._meshes.clear()
        for body_id, mesh in new_meshes.items():
            mesh.upload()
            self._meshes[body_id] = mesh
        self.selection.clear()
        self._hovered_vertex = (None, None)
        self._hovered_edge   = (None, None)
        self.hover.clear()
        self.doneCurrent()
        self._rebuild_sketch_faces()
        self.selection_changed.emit()
        self.update()

    # ------------------------------------------------------------------
    # GL lifecycle
    # ------------------------------------------------------------------

    def initializeGL(self):
        from cad.prefs import prefs
        r, g, b = prefs.background_color
        glClearColor(r, g, b, 1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_POSITION, [1, 2, 3, 0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE,  [0.8, 0.8, 0.8, 1])
        glLightfv(GL_LIGHT0, GL_AMBIENT,  [0.2, 0.2, 0.2, 1])
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

        # Antialiasing. MSAA (multisample rasterisation) antialiases BOTH
        # polygon silhouettes and lines cleanly, with no blending required. We
        # deliberately do NOT use GL_LINE_SMOOTH: coverage-AA lines need alpha
        # blending, and when the wireframe is depth-tested against a surface the
        # partial-coverage fringe fragments z-fight and deposit dark speckles
        # along the edge — the "frizzle" that got worse zoomed out (denser
        # edges) and cleared up zoomed in. MSAA handles line AA without that.
        if prefs.msaa_samples and prefs.msaa_samples > 0:
            glEnable(GL_MULTISAMPLE)

        for mesh in self._meshes.values():
            if mesh.vbo_verts is None:
                mesh.upload()

    def _px_per_mm(self) -> float:
        """On-screen device pixels per world-mm at the sketch/view plane.

        Drives zoom-adaptive sketch curve tessellation. In ortho the view's
        vertical extent is 2·ortho_scale world-mm mapped to the framebuffer
        height (device px). In perspective we approximate at the camera target
        depth using the 45° vertical FOV. Returns 0 if it can't be computed, so
        callers fall back to the fixed segment count.
        """
        try:
            h_px = max(1, self.height()) * self.devicePixelRatio()
            if self.camera.ortho:
                view_h_mm = 2.0 * self.camera.ortho_scale
            else:
                # Vertical world extent visible at the target plane:
                # 2·distance·tan(fovy/2), fovy = 45°.
                import math
                view_h_mm = 2.0 * self.camera.distance * math.tan(math.radians(45) / 2.0)
            if view_h_mm <= 1e-9:
                return 0.0
            return float(h_px / view_h_mm)
        except Exception:
            return 0.0

    def _set_projection(self, w, h):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        aspect = w / h
        if self.camera.ortho:
            s = self.camera.ortho_scale
            glOrtho(-s * aspect, s * aspect, -s, s, -50000, 50000)
        else:
            # Fit near/far tightly around the actual scene depth. The eye sits
            # `distance` from the target and the scene spans ~`_scene_extent`
            # around it, so useful depth lives in
            # [distance − extent, distance + extent]. A fixed far floor (was
            # 50000 mm) blows the near/far ratio to ~10⁵ and wrecks 24-bit depth
            # precision — which shows up as z-fighting "frizzle" on thin curved
            # features like small-radius fillets. Keep the ratio bounded.
            dist   = self.camera.distance
            extent = max(dist * 0.05, getattr(self.camera, "_scene_extent", dist))
            far  = dist + extent * 1.5
            near = max(dist - extent * 1.5, far / 5000.0)  # cap far/near ≤ 5000
            near = max(near, 1e-3)
            gluPerspective(45, aspect, near, far)
        glMatrixMode(GL_MODELVIEW)

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        self._set_projection(w, h)
        self._position_extrude_panel()
        self._position_offset_panel()
        self._position_fillet_panel()

    def paintGL(self):
        from cad.profiler import profiler
        profiler.frame_begin()

        self._set_projection(self.width(), self.height())
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        c = self.camera
        eye, up = c.get_eye(), c.get_up()
        gluLookAt(eye[0], eye[1], eye[2],
                  c.target[0], c.target[1], c.target[2],
                  up[0], up[1], up[2])

        with profiler.section("visible_meshes"):
            visible = self._visible_meshes()
        with profiler.section("draw_opaque"):
            draw_opaque(visible, self.workspace, self.selection)

        # World planes drawn after opaque geometry so they never occlude it
        if any(self.workspace.world_plane_visible.values()):
            from viewer.renderer import draw_world_planes
            import numpy as np
            if visible:
                all_mins = np.vstack([m.bbox_min for m in visible.values()])
                all_maxs = np.vstack([m.bbox_max for m in visible.values()])
                scene_r = float(np.linalg.norm(
                    all_maxs.max(axis=0) - all_mins.min(axis=0))) * 0.75
            else:
                scene_r = 100.0
            draw_world_planes(self.workspace.world_plane_visible,
                              scene_radius=max(20.0, scene_r))

        # Offset plane datums + in-progress panel preview
        from viewer.plane_overlay import draw_offset_planes
        draw_offset_planes(self)

        self._modelview  = glGetDoublev(GL_MODELVIEW_MATRIX)
        self._projection = glGetDoublev(GL_PROJECTION_MATRIX)
        self._viewport   = glGetIntegerv(GL_VIEWPORT)
        # Hover picks against LIVE meshes only — preview meshes (chamfer,
        # fillet3d, etc.) have different edge indices, so feeding them to
        # hover would cause the user's next pick to resolve to the wrong
        # edge once their preview is on screen.
        hover_meshes = {bid: m for bid, m in self._meshes.items()
                        if self._body_visible.get(bid, True)}
        with profiler.section("hover.rebuild"):
            self.hover.rebuild(hover_meshes, self.workspace,
                               self._modelview, self._projection,
                               self._viewport, self.devicePixelRatio(),
                               camera_eye=self.camera.get_eye(),
                               history=self.history,
                               active_sketch=self._sketch,
                               editing_sketch_idx=self._editing_sketch_history_idx)

        with profiler.section("sketch_faces"):
            self._draw_sketch_faces()
        with profiler.section("previews"):
            self._draw_extrude_preview()
            self._draw_thicken_preview()
            self._draw_revolve_preview()
            self._draw_fillet3d_preview()
            self._draw_chamfer_preview()
            self._draw_boolean_preview()
            self._draw_loft_preview()

        # Selection and hover highlights are indexed against LIVE meshes
        # (because selection was captured before any preview swap). Drawing
        # them against `visible` — which may contain a preview mesh with
        # different edge ordering — would highlight the wrong polylines.
        with profiler.section("draw_overlays"):
            self._dim_labels = draw_overlays(hover_meshes, self.selection,
                          self._hovered_vertex, self._hovered_edge,
                          sketch=self._sketch,
                          camera_distance=self.camera.distance,
                          history=self.history,
                          editing_sketch_idx=self._editing_sketch_history_idx,
                          in_sketch=self._sketch is not None,
                          px_per_mm=self._px_per_mm()) or []

            self._draw_sketch_vertex_overlays()

            R = _quat_to_matrix(self.camera.rotation)
            self._view_cube.draw(R, self.width(), self.height(),
                                 self.devicePixelRatio())

        profiler.frame_end()

    def _draw_sketch_vertex_overlays(self):
        """Draw selected and hovered sketch vertex points (not in real mesh)."""
        from viewer.hover import parse_sketch_vtx_key
        from cad.prefs import prefs
        from OpenGL.GL import (glDisable, glEnable, glBegin, glEnd, glVertex3f,
                                glColor3f, glPointSize, GL_POINTS,
                                GL_DEPTH_TEST, GL_LIGHTING, GL_BLEND,
                                GL_POINT_SMOOTH, GL_SRC_ALPHA,
                                GL_ONE_MINUS_SRC_ALPHA, glBlendFunc)
        has_sel = any(parse_sketch_vtx_key(v.body_id) is not None
                      for v in self.selection.vertices)
        hov_body, hov_idx = self._hovered_vertex
        has_hov = hov_body is not None and parse_sketch_vtx_key(hov_body) is not None

        if not has_sel and not has_hov:
            return

        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_POINT_SMOOTH)

        if has_sel:
            glPointSize(10.0)
            glColor3f(*prefs.vertex_selected_color)
            glBegin(GL_POINTS)
            for v in self.selection.vertices:
                if parse_sketch_vtx_key(v.body_id) is None:
                    continue
                p = self.hover.vertex_world_pos(v.body_id, v.vertex_idx)
                if p is not None:
                    glVertex3f(float(p[0]), float(p[1]), float(p[2]))
            glEnd()

        if has_hov:
            p = self.hover.vertex_world_pos(hov_body, hov_idx)
            if p is not None:
                glPointSize(12.0)
                glColor3f(*prefs.vertex_hovered_color)
                glBegin(GL_POINTS)
                glVertex3f(float(p[0]), float(p[1]), float(p[2]))
                glEnd()

        glPointSize(1.0)
        glEnable(GL_DEPTH_TEST)

    def paintEvent(self, event):
        """GL content first, then QPainter for dimension text labels."""
        super().paintEvent(event)
        if (not self._dim_labels or self._modelview is None):
            self._draw_profiler_hud()
            return

        from PyQt6.QtGui import QPainter, QFont, QColor, QPen, QBrush, QFontMetrics
        from PyQt6.QtCore import Qt, QRect, QPoint
        from OpenGL.GLU import gluProject

        dpr = self.devicePixelRatio()
        h   = self.height()

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        font = QFont("monospace", 9)
        font.setBold(True)
        painter.setFont(font)
        fm = QFontMetrics(font)

        # Phase 1: project every label to its screen rect. Stash the resolved
        # rect on the label dict so hit-testing (_hit_dim_label) uses the exact
        # same position the pill is drawn at, even after de-collision below.
        drawable = []
        for lbl in self._dim_labels:
            lbl['screen_rect'] = None   # reset; may be re-set below
            wx, wy, wz = lbl['world']
            try:
                sx, sy, _ = gluProject(wx, wy, wz,
                                       self._modelview,
                                       self._projection,
                                       self._viewport)
            except Exception:
                continue
            # gluProject gives OpenGL y (bottom=0); flip to Qt y (top=0)
            sx = int(sx / dpr)
            sy = int((self._viewport[3] - sy) / dpr)

            text = lbl['text']
            tw   = fm.horizontalAdvance(text) + 10
            th   = fm.height() + 6
            drawable.append({'lbl': lbl, 'cx': sx, 'cy': sy, 'tw': tw, 'th': th})

        # Phase 2: de-collide overlapping pills. Constraint markers (the little
        # ||, ⊥, =, — glyphs) frequently share a line's midpoint, so several
        # land on the same pixel and stack into an unreadable blob. Nudge any
        # pill that overlaps an already-placed one straight down until it clears.
        # Dimension pills (with leader lines) are placed first and never moved,
        # so their glyphs stay anchored to their dimension line; free-floating
        # constraint glyphs then nudge around them.
        placed = []  # list of (x0, y0, x1, y1) already-committed rects
        drawable.sort(key=lambda d: bool(d['lbl'].get('parallel')))
        for d in drawable:
            x0 = d['cx'] - d['tw'] // 2
            y0 = d['cy'] - d['th'] // 2
            if d['lbl'].get('parallel'):
                # Push down in whole-row steps until this glyph clears every
                # already-placed pill.
                for _ in range(len(placed) + 1):
                    r = (x0, y0, x0 + d['tw'], y0 + d['th'])
                    hit = next((p for p in placed
                                if r[0] < p[2] and r[2] > p[0]
                                and r[1] < p[3] and r[3] > p[1]), None)
                    if hit is None:
                        break
                    y0 = hit[3] + 2   # drop just below the pill we collided with
            placed.append((x0, y0, x0 + d['tw'], y0 + d['th']))
            d['x0'], d['y0'] = x0, y0

        # Phase 3: draw.
        for d in drawable:
            lbl = d['lbl']
            rect = QRect(d['x0'], d['y0'], d['tw'], d['th'])
            lbl['screen_rect'] = (d['x0'], d['y0'], d['tw'], d['th'])

            dimmed = lbl.get('dimmed', False)
            bg_alpha  = 60  if dimmed else 200
            txt_alpha = 80  if dimmed else 255

            # Background pill
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor(20, 60, 120, bg_alpha)))
            painter.drawRoundedRect(rect, 4, 4)

            # Text
            painter.setPen(QPen(QColor(140, 200, 255, txt_alpha)))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, lbl['text'])

        painter.end()
        self._draw_profiler_hud()

    def _toggle_profiler(self):
        """F8 handler: flip the profiler and drive continuous repaints while on.

        A free-running 0 ms timer forces back-to-back frames so the FPS / frame
        readout reflects real steady-state cost (otherwise the viewport only
        repaints on input and FPS would look artificially idle)."""
        from cad.profiler import profiler
        from PyQt6.QtCore import QTimer
        on = profiler.toggle()
        if getattr(self, "_profiler_timer", None) is None:
            self._profiler_timer = QTimer(self)
            self._profiler_timer.setInterval(0)
            self._profiler_timer.timeout.connect(self.update)
        if on:
            self._profiler_timer.start()
        else:
            self._profiler_timer.stop()
        self.update()

    def _draw_profiler_hud(self):
        """Overlay live profiler stats in the top-left corner (F8 to toggle)."""
        from cad.profiler import profiler
        if not profiler.enabled:
            return
        from PyQt6.QtGui import QPainter, QFont, QColor, QPen, QBrush
        from PyQt6.QtCore import Qt, QRect

        lines = profiler.hud_lines()
        painter = QPainter(self)
        font = QFont("monospace", 9)
        painter.setFont(font)
        from PyQt6.QtGui import QFontMetrics
        fm = QFontMetrics(font)
        line_h = fm.height()
        w = max(fm.horizontalAdvance(s) for s in lines) + 16
        h = line_h * len(lines) + 12

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(0, 0, 0, 180)))
        painter.drawRoundedRect(QRect(8, 8, w, h), 6, 6)

        y = 8 + line_h
        for i, s in enumerate(lines):
            # Header line in amber, section rows in light grey.
            painter.setPen(QPen(QColor(255, 200, 80) if i == 0
                                else QColor(210, 210, 210)))
            painter.drawText(16, y, s)
            y += line_h
        painter.end()

    # ------------------------------------------------------------------
    # Ray casting
    # ------------------------------------------------------------------

    def get_ray(self, x, y):
        if self._modelview is None:
            return None, None
        dpr  = self.devicePixelRatio()
        x_px = x * dpr
        y_px = self._viewport[3] - y * dpr
        near = np.array(gluUnProject(x_px, y_px, 0.0,
                        self._modelview, self._projection, self._viewport))
        far  = np.array(gluUnProject(x_px, y_px, 1.0,
                        self._modelview, self._projection, self._viewport))
        d    = far - near
        n    = np.linalg.norm(d)
        return (near, d / n) if n > 1e-10 else (None, None)

    def _edge_pick_mode_active(self) -> bool:
        return bool(
            getattr(self, '_extrude_pick_active', False) or
            getattr(self, '_revolve_axis_active', False) or
            getattr(self, '_fillet3d_pick_edge', False) or
            getattr(self, '_chamfer_pick_edge', False))

    def _face_pick_mode_active(self) -> bool:
        thicken = getattr(self, '_thicken_panel', None)
        return bool(
            getattr(self, '_extrude_face_active', False) or
            getattr(self, '_revolve_face_active', False) or
            getattr(self, '_fillet3d_pick_face', False) or
            getattr(self, '_chamfer_pick_face', False) or
            getattr(self, '_offset_plane_face_active', False) or
            (thicken is not None and getattr(thicken, '_picking_face', False)))

    def _route_active_pick(self, e, hov_ebody, hov_eidx) -> bool:
        """When an "Add Edge/Face" pick mode is active, route the click to the
        owning op before passive hover/selection can swallow it. Returns True
        if the click was consumed.

        Priority: an active edge-pick consumes a hovered edge first; an active
        face-pick then tries sketch-face, then body-face (ray pick). Each
        route self-guards on its own mode flag, so only the active op consumes.
        """
        edge_active = self._edge_pick_mode_active()
        face_active = self._face_pick_mode_active()
        if not edge_active and not face_active:
            return False

        # Edge picks: honour the hovered body edge (skip sketch edges — those
        # have their own dedicated routing in the main handler).
        if edge_active and hov_ebody is not None:
            from viewer.hover import parse_sketch_key
            if parse_sketch_key(hov_ebody) is None:
                if self.route_edge_pick_for_revolve(hov_eidx, hov_ebody):
                    return True
                if self.route_edge_pick_for_extrude(hov_eidx, hov_ebody):
                    return True
                if self.route_edge_pick_for_fillet3d(hov_eidx, hov_ebody):
                    return True
                if self.route_edge_pick_for_chamfer(hov_eidx, hov_ebody):
                    return True

        if face_active:
            origin, direction = self.get_ray(e.position().x(), e.position().y())
            if origin is None:
                return False
            # Sketch-face (loop) picks take precedence over body faces.
            if self._sketch_faces:
                hidx, fidx = self._pick_sketch_face(origin, direction)
                if hidx is not None:
                    if self.route_sketch_face_pick_for_extrude(hidx, fidx):
                        self.update(); return True
                    if self.route_sketch_face_pick_for_revolve(hidx, fidx):
                        self.update(); return True
            body_id, idx = self._pick_at(e.position())
            if idx is not None:
                if self.route_face_pick_for_fillet3d(body_id, idx):
                    return True
                if self.route_face_pick_for_chamfer(body_id, idx):
                    return True
                if self.route_face_pick_for_thicken(body_id, idx):
                    return True
                if self.route_face_pick_for_extrude(body_id, idx):
                    return True
                if self.route_face_pick_for_revolve(body_id, idx):
                    return True
                if self.route_face_pick_for_offset_plane(body_id, idx):
                    return True
        return False

    def _pick_at(self, pos):
        from cad.picker import pick_face
        origin, direction = self.get_ray(pos.x(), pos.y())
        if origin is None:
            return None, None
        best_t, best_body, best_face = np.inf, None, None
        for body_id, mesh in self._meshes.items():
            body = self.workspace.bodies.get(body_id)
            if body and not body.visible:
                continue
            result = pick_face(mesh, origin, direction, return_t=True)
            if result and result[1] < best_t:
                best_t, best_body, best_face = result[1], body_id, result[0]
        return best_body, best_face

    # ------------------------------------------------------------------
    # Drag-arrow interaction (shared by extrude, thicken, revolve)
    # ------------------------------------------------------------------

    def _arrow_scale(self, dist: float) -> float:
        scale = self.camera.distance * 0.10
        return max(scale, abs(dist) * 0.18) if dist != 0.0 else scale

    @staticmethod
    def _closest_point_on_axis(ray_o, ray_d, axis_o, axis_d):
        """
        Return s = signed distance along axis_d from axis_o at the closest
        approach between the ray and the axis line. Returns None if parallel.
        """
        import numpy as np
        w   = np.asarray(ray_o,  dtype=float) - np.asarray(axis_o, dtype=float)
        rd  = np.asarray(ray_d,  dtype=float)
        ad  = np.asarray(axis_d, dtype=float)
        a11 = float(np.dot(rd, rd))
        a12 = -float(np.dot(rd, ad))
        a21 = -a12
        a22 = -1.0
        b1  = -float(np.dot(rd, w))
        b2  = -float(np.dot(ad, w))
        det = a11 * a22 - a12 * a21
        if abs(det) < 1e-10:
            return None
        return (a11 * b2 - a21 * b1) / det

    def _hit_extrude_arrow(self, mx: int, my: int) -> bool:
        if getattr(self, '_extrude_panel', None) is None:
            return False
        arrow_origin = getattr(self, '_extrude_arrow_origin', None)
        arrow_dir    = getattr(self, '_extrude_arrow_dir',    None)
        if arrow_origin is None or arrow_dir is None:
            return False
        ray_o, ray_d = self.get_ray(mx, my)
        if ray_o is None:
            return False
        from viewer.drag_arrow import DragArrow
        dist = getattr(self, '_extrude_preview_dist', 0.0)
        if DragArrow().hit_test(ray_o, ray_d, arrow_origin, arrow_dir,
                                self._arrow_scale(dist)) is None:
            return False
        import numpy as np
        panel     = self._extrude_panel
        start_off = (panel._start_offset.mm_value() or 0.0) if panel else 0.0
        face_origin = panel._face_origin if panel else None
        if face_origin is not None:
            self._drag_arrow_axis_origin = (np.asarray(face_origin, dtype=float)
                                            + np.asarray(arrow_dir, dtype=float) * start_off)
        else:
            self._drag_arrow_axis_origin = np.asarray(arrow_origin, dtype=float)
        s_click = self._closest_point_on_axis(ray_o, ray_d,
                                              self._drag_arrow_axis_origin, arrow_dir)
        current = (panel._spinbox.mm_value() or 0.0) if panel else 0.0
        self._drag_arrow_grab_offset = (s_click - current) if s_click is not None else 0.0
        self._drag_arrow_active = True
        self._drag_arrow_op     = 'extrude'
        return True

    def _hit_thicken_arrow(self, mx: int, my: int) -> bool:
        if getattr(self, '_thicken_panel', None) is None:
            return False
        arrow_origin = getattr(self, '_thicken_arrow_origin', None)
        arrow_dir    = getattr(self, '_thicken_arrow_dir',    None)
        if arrow_origin is None or arrow_dir is None:
            return False
        ray_o, ray_d = self.get_ray(mx, my)
        if ray_o is None:
            return False
        from viewer.drag_arrow import DragArrow
        dist = getattr(self, '_thicken_preview_dist', 0.0)
        if DragArrow().hit_test(ray_o, ray_d, arrow_origin, arrow_dir,
                                self._arrow_scale(dist)) is None:
            return False
        import numpy as np
        # axis_origin = face centroid (zero-thickness base)
        self._drag_arrow_axis_origin = (np.asarray(arrow_origin, dtype=float)
                                        - np.asarray(arrow_dir, dtype=float) * abs(dist))
        s_click = self._closest_point_on_axis(ray_o, ray_d,
                                              self._drag_arrow_axis_origin, arrow_dir)
        panel = getattr(self, '_thicken_panel', None)
        current = (panel._spinbox.mm_value() or 0.0) if panel else 0.0
        self._drag_arrow_grab_offset = (s_click - current) if s_click is not None else 0.0
        self._drag_arrow_active = True
        self._drag_arrow_op     = 'thicken'
        # Invalidate any inflight mesh compute so stale results don't land during drag.
        self._thicken_mesh_gen = getattr(self, '_thicken_mesh_gen', 0) + 1
        self._thicken_preview_mesh = None
        return True

    def _hit_revolve_arrow(self, mx: int, my: int) -> bool:
        if getattr(self, '_revolve_panel', None) is None:
            return False
        arrow_origin = getattr(self, '_revolve_arrow_origin', None)
        arrow_dir    = getattr(self, '_revolve_arrow_dir',    None)
        if arrow_origin is None or arrow_dir is None:
            return False
        ray_o, ray_d = self.get_ray(mx, my)
        if ray_o is None:
            return False
        from viewer.drag_arrow import DragArrow
        angle = getattr(self, '_revolve_preview_angle', 0.0)
        scale = self._arrow_scale(angle)  # angle used as proxy for size feel
        if DragArrow().hit_test(ray_o, ray_d, arrow_origin, arrow_dir, scale) is None:
            return False
        import numpy as np
        # For revolve the drag is angular — store the arrow tip as drag origin.
        self._drag_arrow_axis_origin = np.asarray(arrow_origin, dtype=float)
        self._drag_arrow_grab_offset = 0.0   # revolve uses incremental deltas, no offset needed
        self._drag_arrow_active      = True
        self._drag_arrow_op          = 'revolve'
        self._revolve_preview_mesh   = None  # drop mesh so cage shows during drag
        return True

    def _hit_fillet3d_arrow(self, mx: int, my: int) -> bool:
        if getattr(self, '_fillet3d_panel', None) is None:
            return False
        arrow_origin = getattr(self, '_fillet3d_arrow_origin', None)
        arrow_dir    = getattr(self, '_fillet3d_arrow_dir',    None)
        if arrow_origin is None or arrow_dir is None:
            return False
        ray_o, ray_d = self.get_ray(mx, my)
        if ray_o is None:
            return False
        from viewer.drag_arrow import DragArrow
        panel  = getattr(self, '_fillet3d_panel', None)
        radius = (panel._spinbox.mm_value() or 1.0) if panel else 1.0
        scale  = self._arrow_scale(radius)
        if DragArrow().hit_test(ray_o, ray_d, arrow_origin, arrow_dir, scale) is None:
            return False
        import numpy as np
        # Use the base (face centroid) as axis origin so _closest_point_on_axis
        # returns the radius directly (distance along normal from base).
        base = getattr(self, '_fillet3d_arrow_base', None)
        self._drag_arrow_axis_origin = (np.asarray(base, dtype=float)
                                        if base is not None
                                        else np.asarray(arrow_origin, dtype=float))
        s_click = self._closest_point_on_axis(ray_o, ray_d,
                                              self._drag_arrow_axis_origin, arrow_dir)
        current = radius  # already read above
        self._drag_arrow_grab_offset = (s_click - current) if s_click is not None else 0.0
        self._drag_arrow_active = True
        self._drag_arrow_op     = 'fillet3d'
        return True

    def _hit_chamfer_arrow(self, mx: int, my: int) -> bool:
        if getattr(self, '_chamfer_panel', None) is None:
            return False
        arrow_origin = getattr(self, '_chamfer_arrow_origin', None)
        arrow_dir    = getattr(self, '_chamfer_arrow_dir',    None)
        if arrow_origin is None or arrow_dir is None:
            return False
        ray_o, ray_d = self.get_ray(mx, my)
        if ray_o is None:
            return False
        from viewer.drag_arrow import DragArrow
        panel = getattr(self, '_chamfer_panel', None)
        dist  = (panel._distance_spin.mm_value() or 1.0) if panel else 1.0
        scale = self._arrow_scale(dist)
        if DragArrow().hit_test(ray_o, ray_d, arrow_origin, arrow_dir, scale) is None:
            return False
        import numpy as np
        base = getattr(self, '_chamfer_arrow_base', None)
        self._drag_arrow_axis_origin = (np.asarray(base, dtype=float)
                                        if base is not None
                                        else np.asarray(arrow_origin, dtype=float))
        s_click = self._closest_point_on_axis(ray_o, ray_d,
                                              self._drag_arrow_axis_origin, arrow_dir)
        current = dist
        self._drag_arrow_grab_offset = (s_click - current) if s_click is not None else 0.0
        self._drag_arrow_active = True
        self._drag_arrow_op     = 'chamfer'
        return True

    def _hit_any_arrow(self, mx: int, my: int) -> bool:
        return (self._hit_extrude_arrow(mx, my) or
                self._hit_thicken_arrow(mx, my) or
                self._hit_revolve_arrow(mx, my) or
                self._hit_fillet3d_arrow(mx, my) or
                self._hit_chamfer_arrow(mx, my))

    def _update_arrow_drag(self, mx: int, my: int):
        import numpy as np
        op          = getattr(self, '_drag_arrow_op', None)
        axis_origin = getattr(self, '_drag_arrow_axis_origin', None)
        if op is None or axis_origin is None:
            return
        ray_o, ray_d = self.get_ray(mx, my)
        if ray_o is None:
            return

        grab_off = getattr(self, '_drag_arrow_grab_offset', 0.0)

        if op == 'extrude':
            arrow_dir = getattr(self, '_extrude_arrow_dir', None)
            if arrow_dir is None:
                return
            s = self._closest_point_on_axis(ray_o, ray_d, axis_origin, arrow_dir)
            if s is None:
                return
            s = max(0.001, min(10000.0, s - grab_off))
            panel = getattr(self, '_extrude_panel', None)
            if panel is None:
                return
            panel._spinbox.set_mm(s)
            panel._emit_preview()

        elif op == 'thicken':
            arrow_dir = getattr(self, '_thicken_arrow_dir', None)
            if arrow_dir is None:
                return
            s = self._closest_point_on_axis(ray_o, ray_d, axis_origin, arrow_dir)
            if s is None:
                return
            s = max(0.001, min(10000.0, s - grab_off))
            panel = getattr(self, '_thicken_panel', None)
            if panel is None:
                return
            panel._spinbox.set_mm(s)
            panel._fire_preview()

        elif op == 'revolve':
            # Project mouse onto the tangent plane at the arrow tip to get angle delta.
            # axis_origin = arrow tip = rotated face centroid.
            # _revolve_axis_point / _revolve_axis_dir define the rotation axis.
            # arrow_dir = tangent direction at the tip.
            arrow_dir  = getattr(self, '_revolve_arrow_dir',   None)
            axis_pt    = getattr(self, '_revolve_axis_point',  None)
            axis_dir   = getattr(self, '_revolve_axis_dir',    None)
            if arrow_dir is None or axis_pt is None or axis_dir is None:
                return
            # Radial vector from axis to arrow tip
            tip = np.asarray(axis_origin, dtype=float)
            ap  = np.asarray(axis_pt,     dtype=float)
            ad  = np.asarray(axis_dir,    dtype=float)
            radial = tip - ap - np.dot(tip - ap, ad) * ad
            r = np.linalg.norm(radial)
            if r < 1e-10:
                return
            radial /= r
            # Project mouse ray onto the plane spanned by tangent and radial at tip.
            # s along the tangent = arc length, angle = s / r (radians).
            tangent = np.asarray(arrow_dir, dtype=float)
            s_tan = self._closest_point_on_axis(ray_o, ray_d, tip, tangent)
            if s_tan is None:
                return
            delta_angle_rad = s_tan / r
            # Current angle (in degrees) + delta
            panel = getattr(self, '_revolve_panel', None)
            if panel is None:
                return
            cur = panel._angle_spinbox.mm_value() or 0.0
            new_angle = max(0.1, min(360.0, cur + float(np.degrees(delta_angle_rad))))
            # Reset drag origin to current tip so incremental deltas don't accumulate
            import math
            new_angle_rad = math.radians(new_angle)
            base_pt = getattr(self, '_revolve_face_centroid', None)
            if base_pt is not None:
                c   = math.cos(new_angle_rad)
                s_r = math.sin(new_angle_rad)
                bp  = np.asarray(base_pt, dtype=float)
                bp_proj = bp - np.dot(bp - ap, ad) * ad
                r_vec   = bp_proj - ap
                r_perp  = r_vec - np.dot(r_vec, ad) * ad
                rn      = np.linalg.norm(r_perp)
                if rn > 1e-10:
                    r_perp /= rn
                    tan_vec = np.cross(ad, r_perp)
                    new_tip = ap + np.dot(bp - ap, ad) * ad + rn * (c * r_perp + s_r * tan_vec)
                    self._drag_arrow_axis_origin = new_tip
            panel._angle_spinbox.set_mm(new_angle)
            panel._fire_preview()

        elif op == 'fillet3d':
            arrow_dir = getattr(self, '_fillet3d_arrow_dir', None)
            if arrow_dir is None:
                return
            s = self._closest_point_on_axis(ray_o, ray_d, axis_origin, arrow_dir)
            if s is None:
                return
            s = max(0.001, min(10000.0, s - grab_off))
            panel = getattr(self, '_fillet3d_panel', None)
            if panel is None:
                return
            panel._spinbox.set_mm(s)
            self._reposition_fillet3d_arrow()
            self.update()
            panel._emit_preview()

        elif op == 'chamfer':
            arrow_dir = getattr(self, '_chamfer_arrow_dir', None)
            if arrow_dir is None:
                return
            s = self._closest_point_on_axis(ray_o, ray_d, axis_origin, arrow_dir)
            if s is None:
                return
            s = max(0.001, min(10000.0, s - grab_off))
            panel = getattr(self, '_chamfer_panel', None)
            if panel is None:
                return
            panel._distance_spin.set_mm(s)
            self._reposition_chamfer_arrow()
            self.update()
            panel._emit_preview()

    # ------------------------------------------------------------------
    # Keyboard
    # ------------------------------------------------------------------

    def _toggle_construction(self):
        """Flip the construction flag on the selected active-sketch entities.

        Construction geometry stays drawn and snappable but is excluded from
        profile/face building.  Mixed selections are normalised: if any picked
        entity is not yet construction, all become construction; otherwise all
        revert to normal.
        """
        from cad.sketch import LineEntity, ArcEntity
        if self._sketch is None:
            return
        idxs = sorted({se.entity_idx for se in self.selection.sketch_edges
                       if se.history_idx == -1})
        ents = [self._sketch.entities[i] for i in idxs
                if i < len(self._sketch.entities)
                and isinstance(self._sketch.entities[i], (LineEntity, ArcEntity))]
        if not ents:
            print("[Sketch] Select sketch lines or arcs first to toggle "
                  "construction.")
            return
        # If any is normal, make all construction; else clear all.
        make_construction = any(not e.construction for e in ents)
        self._sketch.push_undo_snapshot()
        for e in ents:
            e.construction = make_construction
        verb = "construction" if make_construction else "normal"
        print(f"[Sketch] Set {len(ents)} "
              f"{'entity' if len(ents) == 1 else 'entities'} to {verb}.")
        self._rebuild_sketch_faces()
        self.sketch_mode_changed.emit(True)
        self.update()

    def _activate_mirror(self):
        """Stash the selected active-sketch entities and activate Mirror so the
        next click picks the axis line."""
        from cad.sketch import SketchTool, LineEntity, ArcEntity
        if self._sketch is None:
            return
        idxs = sorted({se.entity_idx for se in self.selection.sketch_edges
                       if se.history_idx == -1
                       and se.entity_idx < len(self._sketch.entities)
                       and isinstance(self._sketch.entities[se.entity_idx],
                                      (LineEntity, ArcEntity))})
        if not idxs:
            print("[Sketch] Select lines or arcs to mirror first, then press "
                  "mirror and click a line as the axis.")
            return
        self._sketch.pending_selection = idxs
        self._sketch.set_tool(SketchTool.MIRROR)
        self.selection.clear()
        self.selection_changed.emit()
        print(f"[Sketch] Mirror: click a line to use as the axis "
              f"({len(idxs)} {'entity' if len(idxs) == 1 else 'entities'} "
              f"selected).")

    def event(self, e):
        from PyQt6.QtCore import QEvent
        from cad.sketch import SketchTool
        if (e.type() == QEvent.Type.KeyPress and
                e.key() == Qt.Key.Key_Tab and
                self._sketch is not None and
                self._sketch.tool == SketchTool.GEOMETRIC):
            self.keyPressEvent(e)
            return True
        return super().event(e)

    # ------------------------------------------------------------------
    # Include (project references into the active sketch)
    # ------------------------------------------------------------------

    def _activate_include(self):
        """Include the current selection into the active sketch.  With nothing
        usable selected, arm click-to-include mode instead: each subsequent
        click on a body edge, sketch line, or vertex includes it immediately.
        Esc disarms."""
        from cad.sketch import SketchTool
        if self._sketch is None:
            return
        if self._apply_include_selection():
            return
        only_active_sketch_lines = (
            bool(self.selection.sketch_edges)
            and not self.selection.edges and not self.selection.vertices
            and all(se.history_idx == -1 for se in self.selection.sketch_edges))
        if only_active_sketch_lines:
            print("[Sketch] Selected line is already part of this sketch — "
                  "pick edges from bodies or other sketches.")
            return
        # Nothing usable selected — arm click-to-include.
        if self._sketch.tool != SketchTool.NONE:
            self._sketch.cancel_tool()
            self.sketch_mode_changed.emit(True)
        self._include_armed = True
        print("[Sketch] Include armed — click body edges, sketch lines, or "
              "vertices to include them (Esc to finish).")
        self.update()

    def _apply_include_selection(self) -> bool:
        """Project the current selection into the active sketch.
        Returns True if anything was included."""
        from cad.sketch_tools.include import IncludeTool
        self._sketch.push_undo_snapshot()
        n = IncludeTool.apply_with_history(
            self._sketch, self.selection, self._meshes, self.history)
        if not n:
            self._sketch._entity_snapshots.pop()
            return False
        print(f"[Sketch] Included {n} reference "
              f"{'entity' if n == 1 else 'entities'}")
        self.selection.clear()
        self.selection_changed.emit()
        self.update()
        return True

    def _apply_armed_include(self):
        """Armed click-to-include: include whatever the click just selected."""
        if self._sketch is None:
            self._include_armed = False
            return
        self._apply_include_selection()

    def keyPressEvent(self, e):
        from cad.prefs import prefs
        from cad.sketch import SketchTool
        from PyQt6.QtWidgets import QApplication

        if QApplication.activeModalWidget() is not None:
            return

        # Profiler: F8 toggles the live HUD, F9 dumps a full table to stdout.
        # Works in any mode (sketch or not), checked before everything else.
        if e.key() == Qt.Key.Key_F8:
            self._toggle_profiler()
            return
        if e.key() == Qt.Key.Key_F9:
            from cad.profiler import profiler
            profiler.dump()
            return

        if e.key() == Qt.Key.Key_Escape:
            if self._sketch is not None:
                if getattr(self, '_include_armed', False):
                    self._include_armed = False
                    print("[Sketch] Include mode ended.")
                    self.update()
                    return
                if getattr(self, '_pattern_panel', None) is not None:
                    # Pattern is mid-parameter — close panel (also cancels tool).
                    self._close_pattern_panel()
                    self.sketch_mode_changed.emit(True)
                    self.update()
                    return
                if self._sketch.snap.declared_type is not None:
                    self._sketch.snap.consume_declared()
                    self.sketch_mode_changed.emit(True)
                elif self._sketch.tool != SketchTool.NONE:
                    self._sketch.cancel_tool()
                    self.sketch_mode_changed.emit(True)
                else:
                    self._complete_sketch()
            else:
                self._selected_sketch_entry = None
                self._selected_sketch_face  = None
                self.selection.clear()
                self.body_selected.emit(None)
                self.selection_changed.emit()
            self.update()
            return

        if self._sketch is not None:
            from cad.sketch_tools.snap import SnapType
            tool_active = self._sketch.tool != SketchTool.NONE
            _NO_SNAP_TOOLS = {SketchTool.DIMENSION, SketchTool.GEOMETRIC}
            snap_active = tool_active and self._sketch.tool not in _NO_SNAP_TOOLS

            # Snap declarations only apply to drawing tools
            if snap_active and prefs.matches("snap_endpoint", e):
                self._sketch.snap.declare(SnapType.ENDPOINT)
                self.sketch_mode_changed.emit(True); self.update(); return
            if snap_active and prefs.matches("snap_midpoint", e):
                self._sketch.snap.declare(SnapType.MIDPOINT)
                self.sketch_mode_changed.emit(True); self.update(); return
            if snap_active and prefs.matches("snap_center", e):
                self._sketch.snap.declare(SnapType.CENTER)
                self.sketch_mode_changed.emit(True); self.update(); return
            if snap_active and prefs.matches("snap_nearest", e):
                self._sketch.snap.declare(SnapType.NEAREST)
                self.sketch_mode_changed.emit(True); self.update(); return
            if snap_active and prefs.matches("snap_tangent", e):
                self._sketch.snap.declare(SnapType.TANGENT)
                self.sketch_mode_changed.emit(True); self.update(); return
            if snap_active and prefs.matches("snap_intersection", e):
                self._sketch.snap.declare(SnapType.INTERSECTION)
                self.sketch_mode_changed.emit(True); self.update(); return

            # H/V lines work from any state
            if prefs.matches("sketch_hline", e):
                self._close_offset_panel()
                self._activate_constrained_line('H'); return
            if prefs.matches("sketch_vline", e):
                self._close_offset_panel()
                self._activate_constrained_line('V'); return

            # Tab cycles geometric constraint tool mode before anything else.
            if (e.key() == Qt.Key.Key_Tab and
                    self._sketch.tool == SketchTool.GEOMETRIC):
                from cad.sketch_tools.geometric import GeometricConstraintTool
                tool = self._sketch._active_tool
                if isinstance(tool, GeometricConstraintTool):
                    tool.cycle_mode()
                    self.sketch_mode_changed.emit(True)
                    self.update()
                return

            # Hide HUD on any tool switch (LINE tool will re-show it on move)
            self._line_hud.hide()

            # A tool key while the pattern panel is open abandons the pattern.
            if getattr(self, '_pattern_panel', None) is not None:
                self._close_pattern_panel()

            # Tool switches + actions
            if prefs.matches("sketch_line", e):
                self._close_offset_panel()
                self._sketch.set_tool(SketchTool.LINE)
            elif prefs.matches("sketch_arc", e):
                self._close_offset_panel()
                self._sketch.set_tool(SketchTool.ARC3)
            elif prefs.matches("sketch_square", e):
                self._close_offset_panel()
                self._sketch.set_tool(SketchTool.SQUARE)
            elif prefs.matches("sketch_spline", e):
                self._close_offset_panel()
                self._sketch.set_tool(SketchTool.SPLINE)
            elif prefs.matches("sketch_circle", e):
                self._close_offset_panel()
                self._sketch.set_tool(SketchTool.CIRCLE)
            elif prefs.matches("sketch_fillet", e):
                self._close_offset_panel()
                self._close_fillet_panel()
                self._sketch.set_tool(SketchTool.FILLET)
            elif prefs.matches("sketch_point", e):
                self._close_offset_panel()
                self._sketch.set_tool(SketchTool.POINT)
            elif prefs.matches("sketch_trim", e):
                self._close_offset_panel()
                self._sketch.set_tool(SketchTool.TRIM)
            elif prefs.matches("sketch_divide", e):
                self._close_offset_panel()
                self._sketch.set_tool(SketchTool.DIVIDE)
            elif prefs.matches("sketch_offset", e):
                # Select-first: if lines/arcs are selected, offset them all and
                # open the panel; otherwise fall back to click-to-select.
                if not self._activate_offset_selection():
                    self._sketch.set_tool(SketchTool.OFFSET)
                else:
                    return
            elif prefs.matches("sketch_include", e):
                self._activate_include()
                return
            elif prefs.matches("sketch_import_image", e):
                from cad.sketch_tools.image_import import ImageImportTool
                self._sketch.push_undo_snapshot()
                n = ImageImportTool.apply(self._sketch, self)
                if n:
                    print(f"[Sketch] Traced {n} line "
                          f"{'segment' if n == 1 else 'segments'} from image")
                    self.update()
                else:
                    self._sketch._entity_snapshots.pop()
                return
            elif prefs.matches("sketch_construction", e):
                self._toggle_construction()
                return
            elif prefs.matches("sketch_mirror", e):
                self._activate_mirror()
                return
            elif prefs.matches("sketch_commit", e):
                # Give the active tool first crack at Enter (e.g. spline commits
                # its accumulated points).  If it consumes it, the sketch stays
                # open; otherwise Enter commits the whole sketch.
                tool = self._sketch._active_tool
                if tool is not None and tool.finish(self._sketch):
                    self._rebuild_sketch_faces()
                    self.sketch_mode_changed.emit(True)
                    self.update()
                    return
                self._complete_sketch()
                return
            elif prefs.matches("sketch_projection_toggle", e):
                self.toggle_projection(); return
            else:
                super().keyPressEvent(e); return

            self.sketch_mode_changed.emit(True)
            self.update()
            return

        if e.key() == Qt.Key.Key_Escape:
            if getattr(self, '_extrude_pick_active', False):
                panel = getattr(self, '_extrude_panel', None)
                if panel:
                    panel._end_pick_edge()
                return
            if getattr(self, '_fillet_panel', None) is not None:
                self._close_fillet_panel()
                return
            if getattr(self, '_fillet3d_panel', None) is not None:
                self._close_fillet3d_panel()
                return
            if getattr(self, '_offset_panel', None) is not None:
                self._close_offset_panel()
                return
            if getattr(self, '_extrude_panel', None) is not None:
                self._close_extrude_panel()
                return
            if getattr(self, '_thicken_panel', None) is not None:
                self._close_thicken_panel()
                return
            if getattr(self, '_revolve_panel', None) is not None:
                self._close_revolve_panel()
                return
        if prefs.matches("undo", e):
            self._do_undo()
        elif prefs.matches("redo", e):
            self._do_redo()
        elif prefs.matches("projection_toggle", e):
            self.toggle_projection()
        elif prefs.matches("extrude", e):
            self._try_extrude()
        elif prefs.matches("thicken", e):
            self._try_thicken()
        elif prefs.matches("fillet", e):
            self._try_fillet()
        elif prefs.matches("revolve", e):
            self._try_revolve()
        else:
            super().keyPressEvent(e)

    def _activate_constrained_line(self, constrain: str):
        """Activate LINE tool with H or V constraint."""
        from cad.sketch import SketchTool
        from cad.sketch_tools.line import LineTool
        self._sketch.tool = SketchTool.LINE
        self._sketch._active_tool = LineTool(constrain=constrain)
        self.sketch_mode_changed.emit(True)
        self.update()

    def toggle_projection(self):
        self.camera.toggle_ortho()
        if self.camera_projection_changed:
            self.camera_projection_changed(self.camera.ortho)
        self.update()

    # ------------------------------------------------------------------
    # Mouse
    # ------------------------------------------------------------------

    def mouseDoubleClickEvent(self, e):
        if e.button() != Qt.MouseButton.LeftButton:
            return
        # Double-click on the view cube resets camera target to origin.
        mx, my = int(e.position().x()), int(e.position().y())
        if self._view_cube.is_over_cube(mx, my, self.width(), self.height(),
                                        self.devicePixelRatio()):
            self.camera.target = np.zeros(3)
            self.update()
            return
        if self._sketch is not None:
            return
        # Try solid face first — opens a fresh sketch
        body_id, idx = self._pick_at(e.position())
        if idx is not None:
            self._enter_sketch(body_id, idx)
            return
        origin, direction = self.get_ray(e.position().x(), e.position().y())
        if origin is None:
            return
        # Try committed sketch face — re-opens for editing
        if self._sketch_faces:
            hidx, _ = self._pick_sketch_face(origin, direction)
            if hidx is not None:
                self._reenter_sketch(hidx)
                return
        # Try offset planes
        from viewer.plane_overlay import pick_offset_plane
        plane_idx = pick_offset_plane(self, origin, direction)
        if plane_idx is not None:
            self._enter_sketch_on_offset_plane(plane_idx)
            return
        # Try visible world planes
        axis = self._pick_world_plane(origin, direction)
        if axis is not None:
            self._enter_sketch_on_plane(axis)

    def _pick_world_plane(self, ray_origin, ray_dir) -> str | None:
        """
        Return the axis ('XY','XZ','YZ') of the closest visible world plane
        hit by the ray, or None.  Only considers planes whose visibility is on.
        """
        import numpy as np
        # Plane normals and the component index of the plane equation (val=0)
        planes = {
            "XY": (np.array([0., 0., 1.]), 2),
            "XZ": (np.array([0., 1., 0.]), 1),
            "YZ": (np.array([1., 0., 0.]), 0),
        }
        best_t, best_axis = np.inf, None
        for axis, (normal, _) in planes.items():
            if not self.workspace.world_plane_visible.get(axis, False):
                continue
            denom = float(np.dot(normal, ray_dir))
            if abs(denom) < 1e-8:
                continue   # ray parallel to plane
            t = -float(np.dot(normal, ray_origin)) / denom
            if t < 0:
                continue   # plane behind camera
            if t < best_t:
                best_t = t
                best_axis = axis
        return best_axis

    def _delete_constraint_label(self, lbl: dict):
        """Shift+click — remove the constraint from sketch or committed entry."""
        ci = lbl['constraint_idx']

        if self._sketch is not None and ci < len(self._sketch.constraints):
            self._sketch.push_undo_snapshot()
            del self._sketch.constraints[ci]
            from cad.sketch import SketchEntry, LineEntity, ArcEntity
            tmp = SketchEntry.from_sketch_mode(self._sketch)
            if tmp.apply_last_constraint() if tmp.constraints else True:
                for i, ent in enumerate(tmp.entities):
                    if isinstance(ent, LineEntity):
                        self._sketch.entities[i].p0 = ent.p0.copy()
                        self._sketch.entities[i].p1 = ent.p1.copy()
                    elif isinstance(ent, ArcEntity):
                        self._sketch.entities[i].center      = ent.center.copy()
                        self._sketch.entities[i].radius      = ent.radius
                        self._sketch.entities[i].start_angle = ent.start_angle
                        self._sketch.entities[i].end_angle   = ent.end_angle
            self._rebuild_sketch_faces()
            self.sketch_mode_changed.emit(True)
            self.update()
            return

        for entry in self.history.entries:
            if entry.operation != 'sketch':
                continue
            se = entry.params.get('sketch_entry')
            if se is None or ci >= len(se.constraints):
                continue
            del se.constraints[ci]
            se.apply_last_constraint()
            self._rebuild_sketch_faces()
            self.history_changed.emit()
            self.update()
            return

    def _edit_dimension_label(self, lbl: dict):
        """Re-open the dimension value editor for a label that was clicked."""
        if lbl.get('parallel'):
            return  # geometric labels are not editable via click
        if self._sketch is None:
            return  # sketch exited before mouse-release arrived
        from PyQt6.QtWidgets import QInputDialog
        from cad.sketch import SketchConstraint
        from cad.prefs import prefs
        from cad.units import from_mm, to_mm
        unit = prefs.default_unit

        ci = lbl['constraint_idx']
        ei = lbl['entity_idx']
        is_diameter = lbl.get('is_diameter', False)
        con_type    = 'diameter' if is_diameter else 'distance'
        edit_title  = "Edit Diameter" if is_diameter else "Edit Length"
        edit_label  = f"Diameter ({unit}):" if is_diameter else f"Length ({unit}):"

        def _ask(current_mm):
            return QInputDialog.getDouble(
                self, edit_title, edit_label,
                from_mm(current_mm, unit), 0.0, 1_000_000, prefs.display_decimals)

        def _make_con(old_con, new_val_mm):
            return SketchConstraint(con_type, (ei,), new_val_mm,
                                    label_offset=old_con.label_offset,
                                    label_angle=getattr(old_con, 'label_angle', None))

        # Active sketch.
        if self._sketch is not None and ci < len(self._sketch.constraints):
            con = self._sketch.constraints[ci]
            if con.type == con_type and con.indices[0] == ei:
                val_disp, ok = _ask(con.value)
                if ok:
                    self._sketch.push_undo_snapshot()
                    self._sketch.constraints[ci] = _make_con(con, to_mm(val_disp, unit))
                    from cad.sketch import SketchEntry, LineEntity, ArcEntity
                    tmp = SketchEntry.from_sketch_mode(self._sketch)
                    if tmp.apply_last_constraint():
                        for i, ent in enumerate(tmp.entities):
                            if isinstance(ent, LineEntity):
                                self._sketch.entities[i].p0 = ent.p0.copy()
                                self._sketch.entities[i].p1 = ent.p1.copy()
                            elif isinstance(ent, ArcEntity):
                                self._sketch.entities[i].center      = ent.center.copy()
                                self._sketch.entities[i].radius      = ent.radius
                                self._sketch.entities[i].start_angle = ent.start_angle
                                self._sketch.entities[i].end_angle   = ent.end_angle
                    self._rebuild_sketch_faces()
                    self.sketch_mode_changed.emit(True)
                    self.update()
                return

        # Committed sketch.
        for entry in self.history.entries:
            if entry.operation != 'sketch':
                continue
            se = entry.params.get('sketch_entry')
            if se is None or ci >= len(se.constraints):
                continue
            con = se.constraints[ci]
            if con.type != con_type or con.indices[0] != ei:
                continue
            val_disp, ok = _ask(con.value)
            if ok:
                se.constraints[ci] = _make_con(con, to_mm(val_disp, unit))
                se.apply_last_constraint()
                self._rebuild_sketch_faces()
                self.history_changed.emit()
                self.update()
            return

    def _hit_dim_label(self, mx: int, my: int) -> dict | None:
        """Return the dimension label dict under screen pos (mx,my), or None."""
        if not self._dim_labels or self._modelview is None:
            return None
        from OpenGL.GLU import gluProject
        from PyQt6.QtGui import QFontMetrics, QFont
        dpr = self.devicePixelRatio()
        font = QFont("monospace", 9)
        font.setBold(True)
        fm = QFontMetrics(font)
        for lbl in self._dim_labels:
            # Prefer the exact rect the pill was last drawn at (post de-collision
            # nudge), so clicks land where the user sees the pill.
            sr = lbl.get('screen_rect')
            if sr is not None:
                x0, y0, tw, th = sr
                if x0 <= mx <= x0 + tw and y0 <= my <= y0 + th:
                    return lbl
                continue
            wx, wy, wz = lbl['world']
            try:
                sx, sy, _ = gluProject(wx, wy, wz,
                                       self._modelview, self._projection, self._viewport)
            except Exception:
                continue
            sx = int(sx / dpr); sy = int((self._viewport[3] - sy) / dpr)
            tw = fm.horizontalAdvance(lbl['text']) + 10
            th = fm.height() + 6
            if abs(mx - sx) <= tw // 2 and abs(my - sy) <= th // 2:
                return lbl
        return None

    def mousePressEvent(self, e):
        self.setFocus()
        if e.button() == Qt.MouseButton.LeftButton:
            mx, my = int(e.position().x()), int(e.position().y())

            # Drag-arrow hit test — takes priority when any op panel is open.
            if self._hit_any_arrow(mx, my):
                return

            # Shift+click a constraint label to delete it.
            shift = bool(e.modifiers() & Qt.KeyboardModifier.ShiftModifier)
            lbl = self._hit_dim_label(mx, my)
            if lbl is not None and shift:
                # Only allow deleting constraints when inside sketch mode.
                if self._sketch is None:
                    return
                self._delete_constraint_label(lbl)
                return
            if lbl is not None:
                # Only allow dragging constraints when inside sketch mode.
                if self._sketch is None:
                    return
                constraints = lbl.get('constraints')
                ci = lbl['constraint_idx']
                con = constraints[ci] if constraints and ci < len(constraints) else None
                if con is not None:
                    perp_w = np.array(lbl['perp_world'], dtype=np.float64)
                    pn = float(np.linalg.norm(perp_w))
                    if pn > 1e-9:
                        perp_w /= pn
                        cur_offset = con.label_offset or 0.0
                        self._drag_constraint  = con
                        self._drag_perp_world  = perp_w
                        self._drag_line_mid    = lbl['world'] - perp_w * cur_offset
                        self._dragging_label   = lbl
                        self._drag_start_screen = (mx, my)
                        if lbl.get('is_diameter'):
                            self._drag_is_diameter      = True
                            cw = np.array(lbl['chord_world'], dtype=np.float64)
                            cn = float(np.linalg.norm(cw))
                            self._drag_chord_world      = cw / cn if cn > 1e-9 else cw
                            self._drag_arc_center_world = np.array(
                                lbl['arc_center_world'], dtype=np.float64)
                            self._drag_arc_radius       = float(lbl['arc_radius'])
                        else:
                            self._drag_is_diameter = False
                        return

            R = _quat_to_matrix(self.camera.rotation)
            mx, my = int(e.position().x()), int(e.position().y())
            cube_hit = self._view_cube.handle_mouse_press(
                mx, my, self.width(), self.height(), R, self.devicePixelRatio())
            if cube_hit is not None:
                normal, is_corner = cube_hit
                self.camera.snap_to_normal(*normal,
                                           origin=tuple(self.camera.target))
                self.update()
                return

            if self._sketch is not None:
                from cad.sketch import SketchTool
                if self._sketch.tool != SketchTool.NONE:
                    origin, direction = self.get_ray(e.position().x(),
                                                     e.position().y())
                    if origin is not None:
                        shift = bool(e.modifiers() &
                                     Qt.KeyboardModifier.ShiftModifier)
                        self._sketch.snap.set_angle_snap(shift)
                        self._sketch.snap.snap_radius_mm = self._snap_radius_mm()
                        consumed = self._sketch.handle_click(
                            origin, direction, self.camera.distance)
                        self.sketch_mode_changed.emit(True)
                        # Open panels when tools transition to SELECTED
                        from cad.sketch import SketchTool
                        from cad.sketch_tools.offset import OffsetTool
                        from cad.sketch_tools.fillet import FilletTool
                        # Mirror is one-shot: return to NONE once it commits.
                        if (self._sketch.tool == SketchTool.MIRROR and consumed):
                            self._sketch.cancel_tool()
                            self._rebuild_sketch_faces()
                        # Pattern: once the reference is picked, open the panel.
                        from cad.sketch_tools.pattern import PatternTool
                        if (self._sketch.tool in (SketchTool.PATTERN_LINEAR,
                                                  SketchTool.PATTERN_CIRCULAR)
                                and consumed
                                and isinstance(self._sketch._active_tool, PatternTool)
                                and self._sketch._active_tool._state ==
                                    PatternTool.STATE_PARAMS
                                and getattr(self, '_pattern_panel', None) is None):
                            self._show_pattern_panel()
                        if (self._sketch.tool == SketchTool.OFFSET and
                                isinstance(self._sketch._active_tool, OffsetTool) and
                                self._sketch._active_tool._state == OffsetTool.STATE_SELECTED and
                                getattr(self, '_offset_panel', None) is None):
                            self._show_offset_panel()
                        elif (self._sketch.tool == SketchTool.FILLET and
                                isinstance(self._sketch._active_tool, FilletTool) and
                                self._sketch._active_tool._state == FilletTool.STATE_SELECTED and
                                getattr(self, '_fillet_panel', None) is None):
                            self._show_fillet_panel()
                        self.update()
                    return

            additive = bool(e.modifiers() & Qt.KeyboardModifier.ShiftModifier)
            hov_body,  hov_idx  = self._hovered_vertex
            hov_ebody, hov_eidx = self._hovered_edge

            # While an op panel's "Add Face/Edge" pick mode is active, the pick
            # must win over passive hover/selection — otherwise a body edge or
            # vertex under the cursor (common while a preview overlaps the
            # target, e.g. on re-entry) swallows the click and nothing gets
            # added. _route_active_pick handles sketch-face, body-face, and
            # body-edge picks for every op uniformly.
            if self._route_active_pick(e, hov_ebody, hov_eidx):
                return

            if hov_body is not None:
                from viewer.hover import parse_sketch_vtx_key
                is_sketch_vtx = parse_sketch_vtx_key(hov_body) is not None
                # Route to extrude panel pick-vertex mode if active (only real verts)
                if not is_sketch_vtx and self.route_vertex_pick_for_extrude(hov_body, hov_idx):
                    return
                self.selection.select_vertex(hov_body, hov_idx,
                                             additive=additive)
                if is_sketch_vtx:
                    p = self.hover.vertex_world_pos(hov_body, hov_idx)
                    if p is not None:
                        print(f"Picked sketch vertex {hov_idx} | "
                              f"pos=({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})")
                else:
                    body = self.workspace.bodies.get(hov_body)
                    mesh = self._meshes.get(hov_body)
                    if body is None or mesh is None or hov_idx >= len(mesh.topo_verts):
                        return
                    self.workspace.set_active_body(hov_body)
                    self.body_selected.emit(hov_body)
                    p = mesh.topo_verts[hov_idx]
                    print(f"Picked vertex {hov_idx} | {body.name} | "
                          f"pos=({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})")

            elif hov_ebody is not None:
                from viewer.hover import parse_sketch_key
                sk = parse_sketch_key(hov_ebody)
                if sk is not None:
                    history_idx, entity_idx = sk
                    # Route to revolve axis-pick if active
                    if history_idx != -1 and self.route_sketch_edge_pick_for_revolve(
                            history_idx, entity_idx):
                        return
                    self.selection.select_sketch_edge(
                        history_idx, entity_idx, additive=additive)
                    if history_idx == -1:
                        print(f"Picked active sketch line entity {entity_idx}")
                    else:
                        print(f"Picked sketch edge | entry {history_idx} "
                              f"entity {entity_idx}")
                else:
                    # Route to revolve axis-pick (body edge) if active
                    if self.route_edge_pick_for_revolve(hov_eidx, hov_ebody):
                        return
                    # Route to extrude panel pick-edge mode if active
                    if self.route_edge_pick_for_extrude(hov_eidx, hov_ebody):
                        return
                    # Route to 3D fillet edge-pick if active
                    if self.route_edge_pick_for_fillet3d(hov_eidx, hov_ebody):
                        return
                    # Route to chamfer edge-pick if active
                    if self.route_edge_pick_for_chamfer(hov_eidx, hov_ebody):
                        return
                    body = self.workspace.bodies.get(hov_ebody)
                    if body is None:
                        # Hover cache pointed at a body that has since been
                        # removed (e.g. by a split/replace from a downstream op).
                        return
                    self.selection.select_edge(hov_ebody, hov_eidx,
                                               additive=additive)
                    self.workspace.set_active_body(hov_ebody)
                    self.body_selected.emit(hov_ebody)
                    print(f"Picked edge {hov_eidx} | {body.name}")

            else:
                origin, direction = self.get_ray(e.position().x(),
                                                 e.position().y())
                if origin is not None and self._sketch_faces:
                    hidx, fidx = self._pick_sketch_face(origin, direction)
                    if hidx is not None:
                        if self.route_sketch_face_pick_for_extrude(hidx, fidx):
                            self.update()
                            return
                        if self.route_sketch_face_pick_for_offset_plane(hidx):
                            self.update()
                            return
                        if self.route_sketch_face_pick_for_loft(hidx, fidx):
                            self.update()
                            return
                        if self.route_sketch_face_pick_for_revolve(hidx, fidx):
                            self.update()
                            return
                        self._selected_sketch_entry = hidx
                        self._selected_sketch_face  = [fidx]
                        self.selection.clear()
                        self.body_selected.emit(None)
                        self.selection_changed.emit()
                        self.update()
                        face_count = len(self._sketch_faces.get(hidx, []))
                        print(f"[Sketch] Selected sketch face {fidx} "
                              f"(entry {hidx}, {face_count} face(s)) — press E to extrude")
                        return

                body_id, idx = self._pick_at(e.position())
                if idx is not None and self.route_face_pick_for_fillet3d(body_id, idx):
                    return
                if idx is not None and self.route_face_pick_for_chamfer(body_id, idx):
                    return
                if idx is not None and self.route_face_pick_for_thicken(body_id, idx):
                    return
                if idx is not None and self.route_face_pick_for_extrude(body_id, idx):
                    return
                if idx is not None and self.route_body_pick_for_extrude(body_id):
                    return
                if idx is not None and self.route_face_pick_for_revolve(body_id, idx):
                    return
                if idx is not None and self.route_body_pick_for_revolve(body_id):
                    return
                if idx is not None and self.route_face_pick_for_offset_plane(body_id, idx):
                    return
                if idx is not None and self.route_body_pick_for_loft(body_id):
                    return
                if idx is not None and self.route_body_pick_for_boolean(body_id):
                    return
                if idx is not None and body_id in self.workspace.bodies:
                    self._selected_sketch_entry = None
                    self._selected_sketch_face  = None
                    self._selected_plane_idx    = None
                    self.selection.select_face(body_id, idx, additive=additive)
                    self.workspace.set_active_body(body_id)
                    self.body_selected.emit(body_id)
                    mesh = self._meshes[body_id]
                    try:
                        from build123d import Plane
                        plane = Plane(mesh.occt_faces[idx])
                        print(f"Picked face {idx} | "
                              f"{self.workspace.bodies[body_id].name} | "
                              f"planar | normal={plane.z_dir}")
                    except Exception:
                        print(f"Picked face {idx} | "
                              f"{self.workspace.bodies[body_id].name} | "
                              f"non-planar")
                else:
                    # Try offset-plane pick before clearing selection
                    from viewer.plane_overlay import pick_offset_plane
                    ro, rd = self.get_ray(e.position().x(), e.position().y())
                    plane_idx = (pick_offset_plane(self, ro, rd)
                                 if ro is not None else None)
                    if plane_idx is not None:
                        self._selected_plane_idx    = plane_idx
                        self._selected_sketch_entry = None
                        self._selected_sketch_face  = None
                        self.selection.clear()
                        self.body_selected.emit(None)
                        entry = self.history.entries[plane_idx]
                        print(f"[OffsetPlane] Selected {entry.label}")
                    elif not additive:
                        self._selected_plane_idx    = None
                        self._selected_sketch_entry = None
                        self._selected_sketch_face  = None
                        self.selection.clear()
                        self.body_selected.emit(None)

            # Armed click-to-include: consume whatever this click selected.
            if getattr(self, '_include_armed', False):
                self._apply_armed_include()

            self.selection_changed.emit()
            self.update()

        elif e.button() == Qt.MouseButton.RightButton:
            mx, my = int(e.position().x()), int(e.position().y())
            if self._view_cube.is_over_cube(mx, my, self.width(), self.height(),
                                            self.devicePixelRatio()):
                return
            body_id, idx = self._pick_at(e.position())
            pivot = None
            if idx is not None and body_id in self._meshes:
                mesh  = self._meshes[body_id]
                start, count = mesh.get_face_triangle_range(idx)
                pivot = mesh.verts[mesh.tris[start:start+count]].mean(
                    axis=(0, 1)).tolist()
            self.camera.begin_orbit(e.position(), self.width(), self.height(),
                                    pivot=pivot)

        elif e.button() == Qt.MouseButton.MiddleButton:
            self.camera.begin_pan(e.position())

    def mouseMoveEvent(self, e):
        buttons = e.buttons()

        # Extrude arrow drag.
        if (buttons & Qt.MouseButton.LeftButton) and self._drag_arrow_active:
            self._update_arrow_drag(int(e.position().x()), int(e.position().y()))
            return

        # Dimension label drag.
        if (buttons & Qt.MouseButton.LeftButton) and self._drag_constraint is not None:
            self._update_dim_drag(int(e.position().x()), int(e.position().y()))
            return

        if buttons & Qt.MouseButton.RightButton:
            self.camera.orbit(e.position()); self.update(); return
        if buttons & Qt.MouseButton.MiddleButton:
            self.camera.pan(e.position());   self.update(); return

        x, y = e.position().x(), e.position().y()

        R = _quat_to_matrix(self.camera.rotation)
        cube_changed = self._view_cube.handle_mouse_move(
            int(x), int(y), self.width(), self.height(), R,
            self.devicePixelRatio())
        if cube_changed:
            self.update()

        new_v = self.hover.vertex_at(x, y)
        new_e = (None, None) if new_v[0] is not None \
                else self.hover.edge_at(x, y)
        # While picking edges for fillet/chamfer, sketch edges are not valid
        # targets — drop them from hover so they don't highlight or click.
        if new_e[0] is not None and (
                getattr(self, '_fillet3d_pick_edge', False) or
                getattr(self, '_chamfer_pick_edge', False)):
            from viewer.hover import parse_sketch_key
            if parse_sketch_key(new_e[0]) is not None:
                new_e = (None, None)
        hover_changed = (new_v != self._hovered_vertex or
                         new_e != self._hovered_edge)
        self._hovered_vertex = new_v
        self._hovered_edge   = new_e

        # Plane hover — only when no body-surface element is under the cursor,
        # and we're not in active sketch mode.
        new_plane_idx = None
        if (new_v[0] is None and new_e[0] is None and
                self._sketch is None):
            from viewer.plane_overlay import pick_offset_plane
            ro, rd = self.get_ray(x, y)
            if ro is not None:
                # Also need to lose to a face hit so picking-a-face-behind-a-plane
                # still works in pick modes.  Cheap check: ask the GPU pick.
                face_body, face_idx = self._pick_at(e.position())
                if face_idx is None:
                    new_plane_idx = pick_offset_plane(self, ro, rd)
        if new_plane_idx != self._hovered_plane_idx:
            self._hovered_plane_idx = new_plane_idx
            hover_changed = True

        sketch_changed = False
        if self._sketch is not None:
            from cad.sketch import SketchTool
            if self._sketch.tool != SketchTool.NONE:
                origin, direction = self.get_ray(x, y)
                if origin is not None:
                    shift = bool(e.modifiers() &
                                 Qt.KeyboardModifier.ShiftModifier)
                    self._sketch.snap.set_angle_snap(shift)
                    self._sketch.snap.snap_radius_mm = self._snap_radius_mm()
                    self._sketch.handle_mouse_move(
                        origin, direction, self.camera.distance)
                    sketch_changed = True
            self._update_line_hud(int(x), int(y))

        if hover_changed or sketch_changed:
            self.update()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton and self._drag_arrow_active:
            was_revolve  = (self._drag_arrow_op == 'revolve')
            was_thicken  = (self._drag_arrow_op == 'thicken')
            self._drag_arrow_active      = False
            self._drag_arrow_axis_origin = None
            if was_revolve:
                params = getattr(self, '_revolve_last_preview_params', None)
                if params is not None:
                    self._revolve_preview_mesh = None  # drop stale mesh, show cage until new one lands
                    self._revolve_compute_mesh(*params)
            elif was_thicken:
                params = getattr(self, '_thicken_last_preview_params', None)
                if params is not None:
                    self._thicken_preview_mesh = None  # show cage until mesh lands
                    self._thicken_compute_mesh(*params)
            return
        if e.button() == Qt.MouseButton.LeftButton and self._dragging_label is not None:
            mx, my = int(e.position().x()), int(e.position().y())
            dx = mx - self._drag_start_screen[0]
            dy = my - self._drag_start_screen[1]
            if dx * dx + dy * dy < 16:
                self._edit_dimension_label(self._dragging_label)
            self._dragging_label        = None
            self._drag_start_screen     = None
            self._drag_constraint       = None
            self._drag_perp_world       = None
            self._drag_line_mid         = None
            self._drag_is_diameter      = False
            self._drag_chord_world      = None
            self._drag_arc_center_world = None
            self._drag_arc_radius       = None
            return
        if e.button() == Qt.MouseButton.RightButton:
            self.camera.end_orbit()
        elif e.button() == Qt.MouseButton.MiddleButton:
            self.camera.end_pan()

    def _update_line_hud(self, mx: int, my: int):
        from cad.sketch import SketchTool
        from cad.sketch_tools.line import LineTool
        from cad.prefs import prefs
        import math

        if (self._sketch is None or
                self._sketch.tool != SketchTool.LINE or
                not isinstance(self._sketch._active_tool, LineTool)):
            self._line_hud.hide()
            return

        tool  = self._sketch._active_tool
        start = tool.line_start
        cur   = tool.cursor_2d

        if start is None or cur is None:
            self._line_hud.hide()
            return

        dx = float(cur[0] - start[0])
        dy = float(cur[1] - start[1])
        length_mm = math.sqrt(dx*dx + dy*dy)
        angle_deg = math.degrees(math.atan2(dy, dx))

        self._line_hud.update_live(length_mm, angle_deg,
                                   prefs.default_unit, prefs.display_decimals)
        self._line_hud.move_to_cursor(mx, my, self.width(), self.height())
        self._line_hud.show()
        self._line_hud.raise_()

    def _on_hud_draw(self):
        """Enter pressed in HUD — compute endpoint from fields and place it."""
        import math
        from cad.sketch_tools.line import LineTool
        from cad.sketch_tools.snap import SnapResult, SnapType
        if self._sketch is None:
            return
        tool = self._sketch._active_tool
        if not isinstance(tool, LineTool) or tool.line_start is None:
            return

        length_mm = self._line_hud.read_length_mm()
        angle_deg = self._line_hud.read_angle_deg()

        if length_mm is not None and angle_deg is not None:
            rad = math.radians(angle_deg)
            pt = np.array([
                tool.line_start[0] + length_mm * math.cos(rad),
                tool.line_start[1] + length_mm * math.sin(rad),
            ], dtype=np.float64)
        elif length_mm is not None and tool.cursor_2d is not None:
            # Lock length, keep cursor direction
            dx = float(tool.cursor_2d[0] - tool.line_start[0])
            dy = float(tool.cursor_2d[1] - tool.line_start[1])
            cur_len = math.sqrt(dx*dx + dy*dy) or 1.0
            pt = np.array([
                tool.line_start[0] + length_mm * dx / cur_len,
                tool.line_start[1] + length_mm * dy / cur_len,
            ], dtype=np.float64)
        elif angle_deg is not None and tool.cursor_2d is not None:
            # Lock angle, keep cursor distance
            dx = float(tool.cursor_2d[0] - tool.line_start[0])
            dy = float(tool.cursor_2d[1] - tool.line_start[1])
            cur_len = math.sqrt(dx*dx + dy*dy)
            rad = math.radians(angle_deg)
            pt = np.array([
                tool.line_start[0] + cur_len * math.cos(rad),
                tool.line_start[1] + cur_len * math.sin(rad),
            ], dtype=np.float64)
        elif tool.cursor_2d is not None:
            pt = tool.cursor_2d.copy()
        else:
            return

        snap = SnapResult(point=pt, type=SnapType.FREE, cursor_raw=pt)
        tool.handle_click(snap, self._sketch)
        self._sketch.snap.consume_declared()
        self._rebuild_sketch_faces()
        self.update()

    def _update_dim_drag(self, mx: int, my: int):
        """Update label_offset (and label_angle for diameter) on the dragged constraint."""
        if self._drag_constraint is None or self._modelview is None:
            return
        from OpenGL.GLU import gluProject
        dpr = self.devicePixelRatio()

        dx = mx - self._drag_start_screen[0]
        dy = my - self._drag_start_screen[1]

        def _project(pt):
            return gluProject(pt[0], pt[1], pt[2],
                              self._modelview, self._projection, self._viewport)

        if self._drag_is_diameter:
            # 2D drag: decompose mouse movement into angular + radial components.
            # Angular: project delta onto screen-space chord direction.
            # Radial:  project delta onto screen-space perp (chord_perp) direction.
            lm  = self._drag_arc_center_world
            cw  = self._drag_chord_world    # chord direction world unit vector
            pw  = self._drag_perp_world     # perp to chord, world unit vector
            try:
                sx0, sy0, _ = _project(lm)
                sxc, syc, _ = _project(lm + cw)
                sxp, syp, _ = _project(lm + pw)
            except Exception:
                return
            # Screen vectors (Qt coords: flip Y).
            chord_sx = (sxc - sx0) / dpr;  chord_sy = -(syc - sy0) / dpr
            perp_sx  = (sxp - sx0) / dpr;  perp_sy  = -(syp - sy0) / dpr
            chord_len = float(np.sqrt(chord_sx**2 + chord_sy**2))
            perp_len  = float(np.sqrt(perp_sx**2  + perp_sy**2))
            if chord_len < 1e-3 or perp_len < 1e-3:
                return

            # Project mouse delta onto each screen-space axis.
            chord_dot = (dx * chord_sx + dy * chord_sy) / chord_len
            perp_dot  = (dx * perp_sx  + dy * perp_sy)  / perp_len

            # chord_dot in pixels → angle change: moving 1 radius in screen
            # space = 90° rotation, so dangle = chord_dot / (r_pixels) * (pi/2).
            r_mm      = self._drag_arc_radius
            r_pixels  = chord_len * r_mm   # pixels per mm * radius mm
            if r_pixels > 1e-3:
                cur_angle = self._drag_constraint.label_angle or 0.0
                new_angle = cur_angle - (chord_dot / r_pixels) * (np.pi / 2.0)
                self._drag_constraint.label_angle = float(new_angle)

            # perp_dot in pixels → radial label offset change (mm).
            cur_off = self._drag_constraint.label_offset or 0.0
            new_off = cur_off + perp_dot / perp_len
            self._drag_constraint.label_offset = float(new_off)

            # Update drag start so next event is a delta, not absolute.
            self._drag_start_screen = (mx, my)

        else:
            # Linear dimension: project onto single perp direction.
            lm = self._drag_line_mid
            pw = self._drag_perp_world
            try:
                sx0, sy0, _ = _project(lm)
                sx1, sy1, _ = _project(lm + pw)
            except Exception:
                return
            spx = (sx1 - sx0) / dpr
            spy = -(sy1 - sy0) / dpr
            screen_perp_len = float(np.sqrt(spx*spx + spy*spy))
            if screen_perp_len < 1e-3:
                return
            dot = (dx * spx + dy * spy) / screen_perp_len
            self._drag_constraint.label_offset = dot / screen_perp_len

        self.update()

    def wheelEvent(self, e):
        pos = e.position()
        self.camera.scroll_at(
            e.angleDelta().y(),
            pos.x(), pos.y(),
            self.width(), self.height(),
        )
        self.update()


# ---------------------------------------------------------------------------
# Parallel tessellation helper
# ---------------------------------------------------------------------------

def _tessellate_one(args):
    """Tessellate a single body — runs in a worker thread (no GL calls)."""
    body_id, shape = args
    try:
        return body_id, Mesh(shape)
    except Exception as ex:
        print(f"[Mesh] Could not tessellate body {body_id}: {ex}")
        return body_id, None


def _tessellate_parallel(bodies: list) -> dict:
    """
    Tessellate all (body_id, shape) pairs in parallel using a thread pool.
    Returns {body_id: Mesh} for bodies that succeeded.
    OCCT tessellation is CPU-bound and thread-safe (no GL state involved).
    """
    from concurrent.futures import ThreadPoolExecutor
    results = {}
    if not bodies:
        return results
    # Use min(len(bodies), cpu_count) threads — no point spinning more than needed
    import os
    workers = min(len(bodies), os.cpu_count() or 4)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for body_id, mesh in pool.map(_tessellate_one, bodies):
            if mesh is not None:
                results[body_id] = mesh
    return results
