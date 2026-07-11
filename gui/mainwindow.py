"""
gui/mainwindow.py
"""

import os
from PyQt6.QtWidgets import (
    QMainWindow, QFileDialog, QSplitter, QWidget
)
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtCore import Qt, QEvent

from gui.toolbar import OpsToolbar, SketchToolbar

from viewer.viewport import Viewport
from viewer.mesh import Mesh
from cad.importer import load_step
from cad.history import History
from cad.workspace import Workspace
from cad.prefs import prefs
from gui.sidebar import Sidebar


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SlopCAD")
        self.resize(1280, 768)
        self._viewport: Viewport | None = None
        self._sidebar:  Sidebar  | None = None
        self._build_menu()
        self._build_toolbar()
        self._build_statusbar()
        self.new_workspace()

    def changeEvent(self, event):
        # When the main window is activated, raise all child Qt.Tool panels
        # so they don't get stuck behind it on WMs that don't propagate
        # owner-window z-order automatically.
        if event.type() == QEvent.Type.ActivationChange and self.isActiveWindow():
            for child in self.findChildren(QWidget):
                if child.isWindow() and child.isVisible():
                    child.raise_()
        super().changeEvent(event)

    def _build_toolbar(self):
        self._ops_toolbar = OpsToolbar(self)
        self._ops_toolbar.extrude_requested.connect(self._toolbar_extrude)
        self._ops_toolbar.thicken_requested.connect(self._toolbar_thicken)
        self._ops_toolbar.fillet_requested.connect(self._toolbar_fillet)
        self._ops_toolbar.sketch_requested.connect(self._toolbar_sketch)
        self._ops_toolbar.revolve_requested.connect(self._toolbar_revolve)
        self._ops_toolbar.offset_plane_requested.connect(self._toolbar_offset_plane)
        self._ops_toolbar.loft_requested.connect(self._toolbar_loft)
        self._ops_toolbar.chamfer_requested.connect(self._toolbar_chamfer)
        self._ops_toolbar.boolean_requested.connect(self._toolbar_boolean)
        self.addToolBar(self._ops_toolbar)

        self._sketch_toolbar = SketchToolbar(self)
        self._sketch_toolbar.tool_line_requested.connect(
            lambda: self._sketch_set_tool("LINE"))
        self._sketch_toolbar.tool_arc_requested.connect(
            lambda: self._sketch_set_tool("ARC3"))
        self._sketch_toolbar.tool_square_requested.connect(
            lambda: self._sketch_set_tool("SQUARE"))
        self._sketch_toolbar.tool_spline_requested.connect(
            lambda: self._sketch_set_tool("SPLINE"))
        self._sketch_toolbar.tool_circle_requested.connect(
            self._sketch_set_circle)
        self._sketch_toolbar.tool_trim_requested.connect(
            lambda: self._sketch_set_tool("TRIM"))
        self._sketch_toolbar.tool_divide_requested.connect(
            lambda: self._sketch_set_tool("DIVIDE"))
        self._sketch_toolbar.tool_fillet_requested.connect(
            lambda: self._sketch_set_tool("FILLET"))
        self._sketch_toolbar.tool_offset_requested.connect(
            self._sketch_offset)
        self._sketch_toolbar.tool_include_requested.connect(
            self._sketch_include)
        self._sketch_toolbar.tool_mirror_requested.connect(
            self._sketch_mirror)
        self._sketch_toolbar.tool_pattern_requested.connect(
            self._sketch_pattern)
        self._sketch_toolbar.tool_construction_requested.connect(
            self._sketch_construction)
        self._sketch_toolbar.tool_import_image_requested.connect(
            self._sketch_import_image)
        self._sketch_toolbar.tool_constraint_requested.connect(
            self._sketch_set_constraint)
        self._sketch_toolbar.commit_requested.connect(
            lambda: self._viewport and self._viewport._complete_sketch())
        self.addToolBar(self._sketch_toolbar)
        self._sketch_toolbar.setVisible(False)

        # Keep a ref for backwards compat
        self._toolbar = self._ops_toolbar

    def refresh_toolbars(self):
        """Rebuild both toolbars so icon size and styles pick up the new scale."""
        sketch_visible = self._sketch_toolbar.isVisible()
        enabled = self._ops_toolbar._btn_extrude.isEnabled()
        self.removeToolBar(self._ops_toolbar)
        self.removeToolBar(self._sketch_toolbar)
        self._ops_toolbar.deleteLater()
        self._sketch_toolbar.deleteLater()
        self._build_toolbar()
        self._ops_toolbar.set_enabled(enabled)
        self._sketch_toolbar.setVisible(sketch_visible)
        self._ops_toolbar.setVisible(not sketch_visible)

    def _toolbar_sketch(self):
        if not self._viewport:
            return
        vp = self._viewport
        # Selected offset plane takes precedence over a selected face.
        if getattr(vp, '_selected_plane_idx', None) is not None:
            vp._enter_sketch_on_offset_plane(vp._selected_plane_idx)
            return
        if vp.selection.face_count == 0:
            self.statusBar().showMessage(
                "Select a face or offset plane to sketch on.", 3000)
            return
        sf = vp.selection.single_face or vp.selection.faces[0]
        vp._enter_sketch(sf.body_id, sf.face_idx)

    def _sketch_set_tool(self, tool_name: str):
        if not self._viewport or not self._viewport._sketch:
            return
        from cad.sketch import SketchTool
        self._viewport._sketch.set_tool(SketchTool[tool_name])
        self._viewport.sketch_mode_changed.emit(True)
        self._viewport.update()

    def _sketch_set_circle(self, mode: str):
        if not self._viewport or not self._viewport._sketch:
            return
        from cad.sketch import SketchTool
        from cad.sketch_tools.circle import CircleTool
        self._viewport._sketch.set_tool(SketchTool.CIRCLE)
        tool = self._viewport._sketch._active_tool
        if isinstance(tool, CircleTool):
            tool.mode = mode
        self._viewport.sketch_mode_changed.emit(True)
        self._viewport.update()

    def _sketch_set_constraint(self, mode: str):
        if not self._viewport or not self._viewport._sketch:
            return
        from cad.sketch import SketchTool
        vp = self._viewport
        if mode in ('distance', 'diameter'):
            vp._sketch.set_tool(SketchTool.DIMENSION)
            vp._sketch._dimension_callback = vp._on_dimension_requested
        else:
            from cad.sketch_tools.geometric import GeometricConstraintTool
            vp._sketch.set_tool(SketchTool.GEOMETRIC)
            tool = vp._sketch._active_tool
            if isinstance(tool, GeometricConstraintTool):
                tool._mode = mode
        self._viewport.sketch_mode_changed.emit(True)
        self._viewport.update()

    def _sketch_include(self):
        if not self._viewport or not self._viewport._sketch:
            return
        from cad.sketch_tools.include import IncludeTool
        vp = self._viewport
        vp._sketch.push_undo_snapshot()
        n = IncludeTool.apply_with_history(
            vp._sketch, vp.selection, vp._meshes, vp.history)
        if not n:
            vp._sketch._entity_snapshots.pop()
        vp.update()

    def _sketch_import_image(self):
        """Toolbar Image — open the trace modal, append fitted geometry."""
        if not self._viewport or not self._viewport._sketch:
            return
        from cad.sketch_tools.image_import import ImageImportTool
        vp = self._viewport
        vp._sketch.push_undo_snapshot()
        n = ImageImportTool.apply(vp._sketch, vp)
        if n:
            print(f"[Sketch] Traced {n} line "
                  f"{'segment' if n == 1 else 'segments'} from image")
        else:
            vp._sketch._entity_snapshots.pop()
        vp.update()

    def _sketch_mirror(self):
        """Toolbar Mirror — same path as the M keybind (uses current selection)."""
        if self._viewport and self._viewport._sketch:
            self._viewport._activate_mirror()

    def _sketch_pattern(self, mode: str):
        """Toolbar Pattern (linear/circular) — uses the current selection."""
        if self._viewport and self._viewport._sketch:
            self._viewport._activate_pattern(mode)

    def _sketch_offset(self):
        """Toolbar Offset — select-first if a selection exists, else click-select."""
        vp = self._viewport
        if not vp or not vp._sketch:
            return
        from cad.sketch import SketchTool
        if not vp._activate_offset_selection():
            vp._sketch.set_tool(SketchTool.OFFSET)
            vp.sketch_mode_changed.emit(True)
            vp.update()

    def _sketch_construction(self):
        """Toolbar Construction — same path as the G keybind."""
        if self._viewport and self._viewport._sketch:
            self._viewport._toggle_construction()

    def _on_sketch_mode_changed(self, in_sketch: bool):
        self._ops_toolbar.setVisible(not in_sketch)
        self._sketch_toolbar.setVisible(in_sketch)
        if in_sketch and self._viewport and self._viewport._sketch:
            tool_name = self._viewport._sketch.tool.name
            self._sketch_toolbar.set_active_tool(tool_name)
        self._update_sketch_label(in_sketch)

    def _toolbar_extrude(self):
        if not self._viewport:
            return
        vp = self._viewport
        vp._try_extrude()

    def _toolbar_thicken(self):
        if not self._viewport:
            return
        self._viewport._try_thicken()

    def _toolbar_fillet(self):
        if not self._viewport:
            return
        self._viewport._try_fillet()

    def _toolbar_revolve(self):
        if not self._viewport:
            return
        self._viewport._try_revolve()

    def _toolbar_offset_plane(self):
        if not self._viewport:
            return
        self._viewport._show_offset_plane_panel()

    def _toolbar_loft(self):
        if not self._viewport:
            return
        self._viewport._show_loft_panel()

    def _toolbar_chamfer(self):
        if not self._viewport:
            return
        self._viewport._try_chamfer()

    def _toolbar_boolean(self):
        if not self._viewport:
            return
        self._viewport._try_boolean()

    def _build_statusbar(self):
        sb = self.statusBar()
        sb.setStyleSheet(prefs.scale_stylesheet("""
            QStatusBar {
                background: #161616;
                color: #555;
                font-size: 11px;
                border-top: 1px solid #2a2a2a;
            }
            QStatusBar::item { border: none; }
        """))
        from PyQt6.QtWidgets import QLabel
        from PyQt6.QtCore import Qt
        # The sketch hint lives in a dedicated wrapping banner above the status
        # bar (added to the central column in the workspace builder) — a
        # word-wrapping QLabel directly in a QStatusBar triggers a
        # heightForWidth resize loop, so keep it out of the status bar.
        # Create it here (before any signal can target it); style/place later.
        self._sketch_label = QLabel("")
        self._sketch_label.setTextFormat(Qt.TextFormat.RichText)
        self._sketch_label.setWordWrap(True)
        self._sketch_label.setStyleSheet(prefs.scale_stylesheet("""
            QLabel {
                background: #161616;
                color: #4fc3f7;
                font-weight: bold;
                font-size: 11px;
                border-top: 1px solid #2a2a2a;
                padding: 4px 8px;
            }
        """))
        self._sketch_label.setVisible(False)

        self._meas_label = QLabel("")
        self._meas_label.setStyleSheet(
            "color: #4fc3f7; padding-right: 12px; font-weight: bold;")
        sb.addPermanentWidget(self._meas_label)

        self._sel_label = QLabel("")
        self._sel_label.setStyleSheet("color: #888; padding-right: 8px;")
        sb.addPermanentWidget(self._sel_label)

    def _update_sketch_label(self, in_sketch: bool):
        if not in_sketch:
            self._sketch_label.clear()
            self._sketch_label.setVisible(False)
            return
        sketch = self._viewport._sketch if self._viewport else None
        if sketch is None:
            self._sketch_label.clear()
            self._sketch_label.setVisible(False)
            return
        # In sketch mode with an active sketch: the banner is shown and its
        # text is set by the branches below.
        self._sketch_label.setVisible(True)

        from cad.sketch import SketchTool
        from cad.sketch_tools.line import LineTool
        from cad.sketch_tools.snap import SnapType

        # Snap keys with optional highlight on the active one
        _SNAP_KEYS = [
            ("E", "endpt",  SnapType.ENDPOINT),
            ("M", "mid",    SnapType.MIDPOINT),
            ("C", "ctr",    SnapType.CENTER),
            ("N", "near",   SnapType.NEAREST),
            ("T", "tan",    SnapType.TANGENT),
            ("I", "isect",  SnapType.INTERSECTION),
        ]
        _SNAP_NAMES = {
            SnapType.ENDPOINT:     "ENDPOINT",
            SnapType.MIDPOINT:     "MIDPOINT",
            SnapType.CENTER:       "CENTER",
            SnapType.NEAREST:      "NEAREST",
            SnapType.TANGENT:      "TANGENT",
            SnapType.INTERSECTION: "INTERSECTION",
        }

        def _snap_hint_html(active_type=None):
            parts = []
            for key, label, stype in _SNAP_KEYS:
                entry = f"{key}={label}"
                if stype == active_type:
                    entry = (f'<span style="background:#1a3a5a;color:#7dd3fc;'
                             f'border-radius:2px;padding:0 3px;">'
                             f'{entry}</span>')
                parts.append(entry)
            return "snap: " + "  ".join(parts)

        tool   = sketch.tool
        active = sketch._active_tool
        declared = sketch.snap.declared_type

        if tool == SketchTool.NONE:
            tools_hint = (
                "L=line  A=arc  C=circle  T=trim  D=divide  "
                "F=fillet  O=offset  P=point  H/V=h/v line  "
                "B=spline  G=construction  M=mirror  Return=commit"
            )
            con_html = ''
            if getattr(sketch, 'constraints', None):
                from cad.sketch import SketchEntry
                tmp = SketchEntry.from_sketch_mode(sketch)
                status, dof = tmp.compute_constraint_status()
                if status == 'over':
                    con_html = ('  <span style="background:#5a1a1a;color:#ff8080;'
                                'border-radius:2px;padding:0 4px;">OVER-CONSTRAINED</span>')
                elif status == 'fully':
                    con_html = ('  <span style="background:#1a3a1a;color:#80ff80;'
                                'border-radius:2px;padding:0 4px;">FULLY CONSTRAINED</span>')
                elif status == 'under':
                    con_html = (f'  <span style="background:#2a2a1a;color:#ffd080;'
                                f'border-radius:2px;padding:0 4px;">'
                                f'UNDER ({dof} dof)</span>')
            self._sketch_label.setText(
                f"✏  SKETCH  —  {tools_hint}  |  {_snap_hint_html()}{con_html}"
            )

        else:
            tool_hints = {
                SketchTool.LINE:   "LINE — click to draw",
                SketchTool.ARC3:   "ARC — click 3 points",
                SketchTool.CIRCLE: "CIRCLE — click to draw",
                SketchTool.TRIM:   "TRIM — click segment to remove",
                SketchTool.DIVIDE: "DIVIDE — click entity to split",
                SketchTool.FILLET:    "FILLET — click corner",
                SketchTool.SQUARE:    "SQUARE — click two corners",
                SketchTool.SPLINE:    "SPLINE — click points; Enter or click start to finish",
                SketchTool.OFFSET:    "OFFSET — click entity or loop",
                SketchTool.POINT:     "POINT — click to place",
                SketchTool.DIMENSION: "DIMENSION — click a line",
                SketchTool.GEOMETRIC: "CONSTRAINTS — click a line  (Tab=cycle mode)",
                SketchTool.MIRROR:    "MIRROR — click a line to use as the axis",
                SketchTool.PATTERN_LINEAR:   "LINEAR PATTERN — click a line for the direction",
                SketchTool.PATTERN_CIRCULAR: "CIRCULAR PATTERN — click a point for the center",
            }

            if tool == SketchTool.LINE and isinstance(active, LineTool):
                if active._constrain == 'H':
                    tool_hints[SketchTool.LINE] = "HLINE — click to place"
                elif active._constrain == 'V':
                    tool_hints[SketchTool.LINE] = "VLINE — click to place"

            if tool == SketchTool.GEOMETRIC:
                from cad.sketch_tools.geometric import GeometricConstraintTool, MODE_LABELS
                if isinstance(active, GeometricConstraintTool):
                    mode_label = MODE_LABELS.get(active.mode, active.mode.upper())
                    step = ("click reference line"
                            if active.first_idx is None else "click second line")
                    tool_hints[SketchTool.GEOMETRIC] = f"{mode_label} — {step}"

            tool_str = tool_hints.get(tool, tool.name)
            self._sketch_label.setText(
                f"✏  {tool_str}  |  {_snap_hint_html(declared)}  |  ESC=cancel"
            )

    def _update_selection_label(self):
        if self._viewport is None:
            self._sel_label.setText("")
            self._meas_label.setText("")
            return
        text = self._viewport.selection.status_text()
        self._sel_label.setText(text)
        self._update_measurement_label()

    def _update_measurement_label(self):
        if self._viewport is None:
            self._meas_label.setText("")
            return
        vp = self._viewport
        verts = vp.selection.vertices
        if len(verts) != 2:
            self._meas_label.setText("")
            return

        from viewer.hover import parse_sketch_vtx_key
        from cad.units import format_value
        from cad.prefs import prefs
        import numpy as np

        positions = []
        for v in verts:
            if parse_sketch_vtx_key(v.body_id) is not None:
                p = vp.hover.vertex_world_pos(v.body_id, v.vertex_idx)
            else:
                mesh = vp._meshes.get(v.body_id)
                p = mesh.topo_verts[v.vertex_idx] if mesh is not None else None
            if p is None:
                self._meas_label.setText("")
                return
            positions.append(np.array(p, dtype=np.float64))

        dist_mm = float(np.linalg.norm(positions[1] - positions[0]))
        text = format_value(dist_mm, prefs.default_unit, prefs.display_decimals)
        self._meas_label.setText(f"dist: {text}")

    def _build_menu(self):
        from cad.prefs import prefs
        menubar = self.menuBar()

        file_menu = menubar.addMenu("File")
        new_act = QAction("New", self)
        new_act.setShortcut(QKeySequence.StandardKey.New)
        new_act.triggered.connect(self.new_workspace)
        file_menu.addAction(new_act)

        open_act  = QAction("Import STEP…", self)
        open_act.triggered.connect(self.open_step)
        file_menu.addAction(open_act)

        file_menu.addSeparator()

        save_act = QAction("Save Project…", self)
        save_act.setShortcut(QKeySequence.StandardKey.Save)
        save_act.triggered.connect(self.save_project)
        file_menu.addAction(save_act)

        load_act = QAction("Open Project…", self)
        load_act.setShortcut(QKeySequence.StandardKey.Open)
        load_act.triggered.connect(self.open_project)
        file_menu.addAction(load_act)

        view_menu = menubar.addMenu("View")
        self._proj_action = QAction("Perspective", self)
        self._proj_action.setCheckable(True)
        self._proj_action.setShortcut(prefs.key("projection_toggle"))
        self._proj_action.triggered.connect(self._toggle_projection)
        view_menu.addAction(self._proj_action)

        view_menu.addSeparator()
        prefs_act = QAction("Preferences…", self)
        prefs_act.triggered.connect(self._open_prefs)
        view_menu.addAction(prefs_act)

        fidelity_act = QAction("Visual Fidelity…", self)
        fidelity_act.triggered.connect(self._open_fidelity)
        view_menu.addAction(fidelity_act)

        ops_menu = menubar.addMenu("Operations")

        extrude_act = QAction(
            f"Extrude face…  ({prefs.key('extrude')})", self)
        extrude_act.triggered.connect(self._menu_extrude)
        ops_menu.addAction(extrude_act)

        ops_menu.addSeparator()

        undo_act = QAction("Undo", self)
        undo_act.setShortcut(QKeySequence.StandardKey.Undo)
        undo_act.triggered.connect(
            lambda: self._viewport and self._viewport.handle_undo())
        ops_menu.addAction(undo_act)

        redo_act = QAction("Redo", self)
        redo_act.setShortcut(QKeySequence.StandardKey.Redo)
        redo_act.triggered.connect(
            lambda: self._viewport and self._viewport.handle_redo())
        ops_menu.addAction(redo_act)

    def _open_prefs(self):
        from gui.prefs_dialog import PrefsDialog
        dlg = PrefsDialog(self)
        if dlg.exec() and self._viewport:
            from OpenGL.GL import glClearColor
            from cad.prefs import prefs
            r, g, b = prefs.background_color
            self._viewport.makeCurrent()
            glClearColor(r, g, b, 1.0)
            self._viewport.doneCurrent()
            self._viewport.update()
            # Refresh history panel so unit labels update immediately
            self._sidebar.history_panel.refresh()
            # Rebuild op tooltips so a changed keybinding shows right away
            self._ops_toolbar.refresh_tooltips()

    def _open_fidelity(self):
        if not self._viewport:
            return
        # Non-modal, single-instance: reuse if already open.
        dlg = getattr(self, "_fidelity_dlg", None)
        if dlg is not None and dlg.isVisible():
            dlg.raise_()
            dlg.activateWindow()
            return
        from gui.fidelity_dialog import FidelityDialog
        self._fidelity_dlg = FidelityDialog(self._viewport, self)
        self._fidelity_dlg.show()

    def _toggle_projection(self):
        if self._viewport:
            self._viewport.toggle_projection()
            is_ortho = self._viewport.camera.ortho
            self._proj_action.setChecked(is_ortho)
            self._proj_action.setText(
                "Orthographic" if is_ortho else "Perspective")

    def _sync_proj_menu(self, is_ortho: bool):
        self._proj_action.setChecked(is_ortho)
        self._proj_action.setText(
            "Orthographic" if is_ortho else "Perspective")

    def _menu_extrude(self):
        if not self._viewport:
            return
        if self._viewport.selection.face_count == 0:
            self.statusBar().showMessage("Select a face first.", 3000)
            return
        sf = self._viewport.selection.single_face or \
             self._viewport.selection.faces[0]
        self._show_extrude_dialog(sf.body_id, sf.face_idx)

    def _show_extrude_dialog(self, body_id: str, face_idx: int):
        from viewer.vp_extrude import _extrude_distance_dialog
        dist = _extrude_distance_dialog(self)
        if dist is not None:
            self._viewport.do_extrude(body_id, face_idx, dist)

    def save_project(self):
        if not self._viewport:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Project", "", "VC Project (*.vc)"
        )
        if not path:
            return
        if not path.endswith(".vc"):
            path += ".vc"
        try:
            from cad.serializer import save
            camera = self._viewport.camera
            data = save(self._viewport.workspace, self._viewport.history, camera)
            with open(path, "wb") as f:
                f.write(data)
            print(f"Project saved to {path}")
        except Exception as ex:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Save failed", str(ex))

    def open_project(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Project", "", "VC Project (*.vc)"
        )
        if not path:
            return
        try:
            from cad.serializer import load, replay_all
            with open(path, "rb") as f:
                data = f.read()
            workspace, history, camera_dict = load(data)
            print(f"Loaded project from {path}, replaying history…")
            warnings = replay_all(workspace, history)
            for w in warnings:
                print(f"  [warn] {w}")
            self._setup_viewport(workspace, history)
            if camera_dict and self._viewport:
                cam = self._viewport.camera
                import numpy as np
                cam.target      = np.array(camera_dict["target"])
                cam.distance    = camera_dict["distance"]
                cam.ortho_scale = camera_dict["ortho_scale"]
                cam.ortho       = camera_dict["ortho"]
                cam.rotation    = np.array(camera_dict["rotation"])
                self._sync_proj_menu(cam.ortho)
            elif self._viewport:
                self._viewport.fit_camera_to_scene()
            if warnings:
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.warning(self, "Load warnings",
                                    "\n".join(warnings))
            print("Done.")
        except Exception as ex:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Load failed", str(ex))
            raise

    def open_step(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open STEP File", "", "STEP Files (*.step *.stp)"
        )
        if path:
            self.load(path)

    def new_workspace(self):
        workspace = Workspace()
        history   = History()
        workspace.history  = history
        history._workspace = workspace
        self._setup_viewport(workspace, history)

    def load(self, path: str):
        print(f"Loading {path}…")
        compound = load_step(path)

        workspace = self._viewport.workspace if self._viewport else None
        history   = self._viewport.history   if self._viewport else None
        if workspace is None or history is None:
            self.new_workspace()
            workspace = self._viewport.workspace
            history   = self._viewport.history

        solids = list(compound.solids())
        if not solids:
            solids = [compound]

        basename = os.path.splitext(os.path.basename(path))[0]

        for i, solid in enumerate(solids):
            name = (f"{basename}" if len(solids) == 1
                    else f"{basename}  [{i+1}]")
            body = workspace.add_body(name, solid)
            history.push(
                label        = f"Import  {name}",
                operation    = "import",
                params       = {"path": path, "solid_index": i},
                body_id      = body.id,
                face_ref     = None,
                shape_before = None,
                shape_after  = solid,
            )
            print(f"  Body '{name}' — {len(list(solid.faces()))} faces")

        self._viewport.build_meshes()
        self._viewport.fit_camera_to_scene()
        self._sidebar.refresh()
        print("Done.")

    def _setup_viewport(self, workspace: Workspace, history: History):
        vp = Viewport(workspace, history)
        vp.build_meshes()
        vp.fit_camera_to_scene()
        vp.camera_projection_changed = self._sync_proj_menu
        self._sync_proj_menu(vp.camera.ortho)

        sidebar = Sidebar(workspace, history)
        sidebar.seek_requested.connect(vp.seek_history)
        sidebar.replay_requested.connect(vp.do_replay)
        sidebar.body_visibility_changed.connect(vp.set_body_visible)
        vp.history_changed.connect(sidebar.refresh)
        vp.body_selected.connect(sidebar.parts_panel.set_selected_body)
        vp.body_selected.connect(sidebar.history_panel.set_selected_body)
        sidebar.parts_panel.body_selected.connect(vp.set_active_body)
        sidebar.parts_panel.body_selected.connect(
            sidebar.history_panel.set_selected_body)
        sidebar.history_panel.sketch_vis_changed.connect(vp._rebuild_sketch_faces)
        sidebar.history_panel.sketch_vis_changed.connect(vp.update)
        sidebar.history_panel.reenter_sketch_requested.connect(vp._reenter_sketch)
        sidebar.history_panel.reopen_extrude_requested.connect(vp.reopen_extrude)
        sidebar.history_panel.reopen_thicken_requested.connect(vp.reopen_thicken)
        sidebar.history_panel.reopen_revolve_requested.connect(vp.reopen_revolve)
        sidebar.history_panel.reopen_fillet_requested.connect(vp.reopen_fillet)
        sidebar.history_panel.reopen_loft_requested.connect(vp.reopen_loft)
        sidebar.history_panel.reopen_offset_plane_requested.connect(vp.reopen_offset_plane)
        sidebar.history_panel.reopen_chamfer_requested.connect(vp.reopen_chamfer)
        sidebar.history_panel.reopen_boolean_requested.connect(vp.reopen_boolean)
        sidebar.history_panel.delete_requested.connect(vp.do_delete)
        sidebar.history_panel.reorder_requested.connect(vp.do_reorder)
        sidebar.plane_visibility_changed.connect(vp.set_world_plane_visible)
        sidebar.sketch_on_plane_requested.connect(vp._enter_sketch_on_plane)

        vp.selection_changed.connect(self._update_selection_label)
        vp.sketch_mode_changed.connect(self._on_sketch_mode_changed)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(sidebar)
        splitter.addWidget(vp)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([210, 1070])
        splitter.setHandleWidth(2)
        splitter.setStyleSheet("QSplitter::handle { background: #2a2a2a; }")

        self._viewport = vp
        self._sidebar  = sidebar

        # Central column: splitter on top, wrapping sketch-hint banner below.
        from PyQt6.QtWidgets import QWidget, QVBoxLayout
        central = QWidget()
        col = QVBoxLayout(central)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(0)
        col.addWidget(splitter, 1)
        col.addWidget(self._sketch_label)
        self.setCentralWidget(central)
        self._toolbar.set_enabled(True)

        # Program-wide progress indicator for heavy OCCT ops + checks. Handed to
        # the viewport so any op can report phases / offer cancel through it.
        from gui.progress import ProgressController
        self.progress = ProgressController(self)
        vp.progress = self.progress
