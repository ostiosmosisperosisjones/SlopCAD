"""
gui/parts_panel.py

PartsPanel — pinned world-planes section + per-body visibility list.

Signals
-------
body_selected(body_id: str | None)
    Emitted when the user clicks a body row. None if deselected.
body_visibility_changed(body_id: str, visible: bool)
    Emitted when a body eye toggle is clicked.
plane_visibility_changed(axis: str, visible: bool)
    Emitted when a world-plane eye toggle is clicked.
sketch_on_plane_requested(axis: str)
    Emitted when the user double-clicks a world-plane row.
"""

from __future__ import annotations
from cad.prefs import prefs
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QScrollArea, QFrame, QSizePolicy, QMenu
)
from PyQt6.QtCore import Qt, pyqtSignal
from cad.workspace import Workspace


_BG         = "#1e1e1e"
_BG_ROW     = "#2a2a2a"
_BG_SEL     = "#2f3f52"
_TEXT       = "#d4d4d4"
_TEXT_DIM   = "#555"
_BORDER_SEL = "#4a90d9"
_BORDER_ROW = "#333"
_BG_PLANES  = "#1a1e22"

_EYE_ON  = "👁"
_EYE_OFF = "◌"
_BODY_ICON = "⬡"

# Per-axis accent colours (match GL rendering)
_PLANE_COLOR = {"XY": "#4a7eb5", "XZ": "#4ab57e", "YZ": "#b54a4a"}
_PLANE_LABEL = {"XY": "XY  Plane", "XZ": "XZ  Plane", "YZ": "YZ  Plane"}


# ---------------------------------------------------------------------------
# World-plane row
# ---------------------------------------------------------------------------

class _PlaneRow(QFrame):
    visibility_toggled  = pyqtSignal(str, bool)   # axis, visible
    double_clicked      = pyqtSignal(str)          # axis

    def __init__(self, axis: str, visible: bool):
        super().__init__()
        self._axis    = axis
        self._visible = visible
        self._build()

    def _build(self):
        color = _PLANE_COLOR[self._axis]
        self.setStyleSheet(f"""
            QFrame {{
                background: {_BG_PLANES};
                border-left: 3px solid {color};
                border-radius: 2px;
                margin: 1px 4px 1px 12px;
            }}
        """)
        row = QHBoxLayout(self)
        row.setContentsMargins(6, 3, 6, 3)
        row.setSpacing(6)

        self._eye = QPushButton(_EYE_ON if self._visible else _EYE_OFF)
        self._eye.setFixedSize(20, 20)
        self._eye.setStyleSheet(prefs.scale_stylesheet("""
            QPushButton { background: transparent; border: none;
                          color: #666; font-size: 13px; padding: 0; }
            QPushButton:hover { color: #aaa; }
        """))
        self._eye.clicked.connect(self._toggle)
        row.addWidget(self._eye)

        lbl = QLabel(_PLANE_LABEL[self._axis])
        lbl.setStyleSheet(prefs.scale_stylesheet(
            f"color: {_PLANE_COLOR[self._axis]}; font-size: 11px; "
            "background: transparent; border: none;"))
        lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        row.addWidget(lbl)

        hint = QLabel("✏")
        hint.setStyleSheet(prefs.scale_stylesheet("color: #333; font-size: 10px; background: transparent; border: none;"))
        hint.setToolTip("Double-click to sketch on this plane")
        row.addWidget(hint)

    def _toggle(self):
        self._visible = not self._visible
        self._eye.setText(_EYE_ON if self._visible else _EYE_OFF)
        self.visibility_toggled.emit(self._axis, self._visible)

    def mouseDoubleClickEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self.double_clicked.emit(self._axis)
        super().mouseDoubleClickEvent(e)


# ---------------------------------------------------------------------------
# World-planes section (pinned at top)
# ---------------------------------------------------------------------------

class _WorldPlanesSection(QWidget):
    visibility_toggled    = pyqtSignal(str, bool)
    sketch_requested      = pyqtSignal(str)

    _AXES = ("XY", "XZ", "YZ")

    def __init__(self, workspace: Workspace):
        super().__init__()
        self._workspace = workspace
        self._collapsed = False
        self._rows: dict[str, _PlaneRow] = {}
        self._build()

    def _build(self):
        self.setStyleSheet(f"background: {_BG_PLANES};")
        self._root = QVBoxLayout(self)
        self._root.setContentsMargins(0, 0, 0, 4)
        self._root.setSpacing(0)

        # Header row
        hdr = QFrame()
        hdr.setStyleSheet(f"""
            QFrame {{
                background: #161e24;
                border-bottom: 1px solid #2a2a2a;
            }}
        """)
        hdr_row = QHBoxLayout(hdr)
        hdr_row.setContentsMargins(8, 5, 8, 5)
        hdr_row.setSpacing(4)

        self._toggle_btn = QPushButton("▾")
        self._toggle_btn.setFixedSize(16, 16)
        self._toggle_btn.setStyleSheet(prefs.scale_stylesheet("""
            QPushButton { background: transparent; border: none;
                          color: #666; font-size: 10px; padding: 0; }
            QPushButton:hover { color: #aaa; }
        """))
        self._toggle_btn.clicked.connect(self._toggle_collapse)
        hdr_row.addWidget(self._toggle_btn)

        # Parent visibility toggle — hides/shows all planes at once
        self._all_visible = True
        self._eye_all = QPushButton(_EYE_ON)
        self._eye_all.setFixedSize(20, 20)
        self._eye_all.setStyleSheet(prefs.scale_stylesheet("""
            QPushButton { background: transparent; border: none;
                          color: #666; font-size: 13px; padding: 0; }
            QPushButton:hover { color: #aaa; }
        """))
        self._eye_all.clicked.connect(self._toggle_all_visibility)
        hdr_row.addWidget(self._eye_all)

        title = QLabel("World Planes")
        title.setStyleSheet(prefs.scale_stylesheet(
            "color: #555; font-size: 11px; font-weight: bold; "
            "letter-spacing: 1px; background: transparent; border: none;"))
        hdr_row.addWidget(title)
        hdr_row.addStretch()
        self._root.addWidget(hdr)

        # Plane rows container
        self._rows_widget = QWidget()
        self._rows_widget.setStyleSheet(f"background: {_BG_PLANES};")
        rows_layout = QVBoxLayout(self._rows_widget)
        rows_layout.setContentsMargins(0, 2, 0, 2)
        rows_layout.setSpacing(0)

        for axis in self._AXES:
            visible = self._workspace.world_plane_visible.get(axis, False)
            row = _PlaneRow(axis, visible)
            row.visibility_toggled.connect(self._on_vis)
            row.double_clicked.connect(self.sketch_requested)
            rows_layout.addWidget(row)
            self._rows[axis] = row

        self._root.addWidget(self._rows_widget)

    def _toggle_collapse(self):
        self._collapsed = not self._collapsed
        self._rows_widget.setVisible(not self._collapsed)
        self._toggle_btn.setText("▸" if self._collapsed else "▾")

    def _toggle_all_visibility(self):
        # If any plane is visible, hide all. Otherwise show all.
        any_on = any(self._workspace.world_plane_visible.values())
        self._all_visible = not any_on
        self._eye_all.setText(_EYE_ON if self._all_visible else _EYE_OFF)
        for axis in self._AXES:
            self._workspace.world_plane_visible[axis] = self._all_visible
            self.visibility_toggled.emit(axis, self._all_visible)

    def _on_vis(self, axis: str, visible: bool):
        self._workspace.world_plane_visible[axis] = visible
        self.visibility_toggled.emit(axis, visible)
        # Keep parent eye in sync
        any_on = any(self._workspace.world_plane_visible.values())
        self._eye_all.setText(_EYE_ON if any_on else _EYE_OFF)


class _BodyRow(QFrame):
    clicked            = pyqtSignal(str)
    visibility_toggled = pyqtSignal(str, bool)
    export_requested   = pyqtSignal(str)   # body_id

    def __init__(self, body_id: str, name: str, visible: bool, selected: bool):
        super().__init__()
        self._body_id = body_id
        self._visible = visible
        self._selected = selected
        self._build(name)
        self._apply_style()

    def _build(self, name: str):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(6)

        # Visibility toggle
        self._eye_btn = QPushButton(_EYE_ON if self._visible else _EYE_OFF)
        self._eye_btn.setFixedSize(20, 20)
        self._eye_btn.setStyleSheet(prefs.scale_stylesheet("""
            QPushButton {
                background: transparent;
                border: none;
                color: #666;
                font-size: 13px;
                padding: 0;
            }
            QPushButton:hover { color: #aaa; }
        """))
        self._eye_btn.clicked.connect(self._toggle_visibility)
        layout.addWidget(self._eye_btn)

        # Body icon
        icon = QLabel(_BODY_ICON)
        icon.setStyleSheet(prefs.scale_stylesheet(
            "color: #4a90d9; font-size: 11px; background: transparent; border: none;"))
        layout.addWidget(icon)

        # Name
        self._name_lbl = QLabel(name)
        self._name_lbl.setStyleSheet(prefs.scale_stylesheet(
            f"color: {_TEXT if self._visible else _TEXT_DIM}; "
            f"font-size: 12px; background: transparent; border: none;"))
        self._name_lbl.setSizePolicy(QSizePolicy.Policy.Expanding,
                                     QSizePolicy.Policy.Preferred)
        layout.addWidget(self._name_lbl)

    def _apply_style(self):
        border = _BORDER_SEL if self._selected else _BORDER_ROW
        bg     = _BG_SEL     if self._selected else _BG_ROW
        self.setStyleSheet(f"""
            QFrame {{
                background: {bg};
                border-left: 3px solid {border};
                border-radius: 3px;
                margin: 1px 4px;
            }}
        """)

    def _toggle_visibility(self):
        self._visible = not self._visible
        self._eye_btn.setText(_EYE_ON if self._visible else _EYE_OFF)
        self._name_lbl.setStyleSheet(prefs.scale_stylesheet(
            f"color: {_TEXT if self._visible else _TEXT_DIM}; "
            f"font-size: 12px; background: transparent; border: none;"))
        self.visibility_toggled.emit(self._body_id, self._visible)

    def set_selected(self, selected: bool):
        self._selected = selected
        self._apply_style()

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self._body_id)
        elif e.button() == Qt.MouseButton.RightButton:
            self._show_context_menu(e.globalPosition().toPoint())
        super().mousePressEvent(e)

    def _show_context_menu(self, pos):
        menu = QMenu(self)
        export_act = menu.addAction("Export as…")
        action = menu.exec(pos)
        if action == export_act:
            self.export_requested.emit(self._body_id)


class PartsPanel(QWidget):
    body_selected            = pyqtSignal(object)   # str | None
    body_visibility_changed  = pyqtSignal(str, bool)
    plane_visibility_changed = pyqtSignal(str, bool) # axis, visible
    sketch_on_plane_requested = pyqtSignal(str)      # axis

    def __init__(self, workspace: Workspace, parent=None):
        super().__init__(parent)
        self._workspace   = workspace
        self._selected_id: str | None = None
        self._visible:     dict[str, bool] = {}
        self._rows:        dict[str, _BodyRow] = {}
        self._setup()
        self.refresh()

    def _on_export_requested(self, body_id: str):
        from gui.export_dialog import ExportDialog
        body = self._workspace.bodies.get(body_id)
        if body is None:
            return
        shape = self._workspace.current_shape(body_id)
        if shape is None:
            return
        dlg = ExportDialog(body.name, self)
        if dlg.exec() != ExportDialog.DialogCode.Accepted:
            return
        fmt, path = dlg.get_result()
        if not path:
            return
        try:
            if fmt == "STEP":
                from build123d import export_step, Compound
                from cad.importer import _heal
                export_step(Compound(_heal(shape.wrapped)), path)
            elif fmt == "STL":
                from build123d import export_stl
                export_stl(shape, path)
        except Exception as ex:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Export failed", str(ex))

    def _setup(self):
        self.setStyleSheet(f"background: {_BG};")
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        header = QLabel("  Parts")
        header.setFixedHeight(prefs.scaled_px(28))
        header.setStyleSheet(prefs.scale_stylesheet("""
            background: #161616;
            color: #555;
            font-size: 11px;
            font-weight: bold;
            letter-spacing: 1px;
            border-bottom: 1px solid #2a2a2a;
        """))
        root.addWidget(header)

        # Pinned world-planes section
        self._planes_section = _WorldPlanesSection(self._workspace)
        self._planes_section.visibility_toggled.connect(self.plane_visibility_changed)
        self._planes_section.sketch_requested.connect(self.sketch_on_plane_requested)
        root.addWidget(self._planes_section)

        # Divider
        div = QFrame()
        div.setFixedHeight(1)
        div.setStyleSheet("background: #2a2a2a;")
        root.addWidget(div)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setStyleSheet("border: none; background: transparent;")

        self._list_widget = QWidget()
        self._list_layout = QVBoxLayout(self._list_widget)
        self._list_layout.setContentsMargins(0, 4, 0, 4)
        self._list_layout.setSpacing(0)
        self._list_layout.addStretch()
        self._scroll.setWidget(self._list_widget)
        root.addWidget(self._scroll, stretch=1)

    def refresh(self):
        """Rebuild rows from current workspace bodies."""
        # Clear existing rows
        while self._list_layout.count() > 1:
            item = self._list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._rows.clear()

        for body_id, body in self._workspace.bodies.items():
            # Only show bodies that exist at the current history cursor
            if self._workspace.current_shape(body_id) is None:
                continue
            visible = self._visible.get(body_id, True)
            selected = (body_id == self._selected_id)
            row = _BodyRow(body_id, body.name, visible, selected)
            row.clicked.connect(self._on_row_clicked)
            row.visibility_toggled.connect(self._on_visibility_toggled)
            row.export_requested.connect(self._on_export_requested)
            self._rows[body_id] = row
            self._list_layout.insertWidget(
                self._list_layout.count() - 1, row)

    def _on_row_clicked(self, body_id: str):
        # Toggle deselect if clicking the already-selected body
        if self._selected_id == body_id:
            self._selected_id = None
            self.body_selected.emit(None)
        else:
            self._selected_id = body_id
            self.body_selected.emit(body_id)
        # Update selection highlight on all rows
        for bid, row in self._rows.items():
            row.set_selected(bid == self._selected_id)

    def _on_visibility_toggled(self, body_id: str, visible: bool):
        self._visible[body_id] = visible
        self.body_visibility_changed.emit(body_id, visible)

    def selected_body_id(self) -> str | None:
        return self._selected_id

    def set_selected_body(self, body_id: str | None):
        """Driven externally (e.g. from viewport click) — updates highlight
        without emitting body_selected to avoid signal loops."""
        self._selected_id = body_id
        for bid, row in self._rows.items():
            row.set_selected(bid == self._selected_id)

    def is_body_visible(self, body_id: str) -> bool:
        return self._visible.get(body_id, True)
