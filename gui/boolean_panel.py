"""
gui/boolean_panel.py

BooleanPanel — floating panel for boolean operations between bodies.

Union / Intersect : single "Bodies" list (pick multiple), operation radio buttons.
Subtract          : Target (single) + Tool(s) (pick multiple).

"Keep inputs" toggle prevents input bodies from being hidden after the operation.
"""

from __future__ import annotations
from cad.prefs import prefs

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame,
    QRadioButton, QButtonGroup, QCheckBox,
)
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QKeyEvent

from gui.selection_list import SelectionList


_PANEL_STYLE = """
QWidget#BooleanPanel {
    background: #1e1e1e;
    border: 1px solid #3a3a3a;
    border-radius: 6px;
}
QLabel { color: #d4d4d4; font-size: 12px; }
QLabel#title { color: #ffffff; font-size: 13px; font-weight: bold; }
QLabel#section {
    color: #888; font-size: 10px;
    text-transform: uppercase; letter-spacing: 1px;
}
QPushButton {
    background: #2a2a2a; color: #d4d4d4;
    border: 1px solid #444; border-radius: 3px;
    padding: 4px 10px; font-size: 12px;
}
QPushButton:hover  { background: #333; }
QPushButton:pressed { background: #222; }
QPushButton#ok {
    background: #1a4a7a; border-color: #4a90d9;
    color: #fff; padding: 5px 18px; font-weight: bold;
}
QPushButton#ok:hover { background: #1e5a8a; }
QPushButton#pick[active=true] {
    background: #1e2a3a; border-color: #4a90d9; color: #7ac4f7;
}
QPushButton#pick_tool[active=true] {
    background: #2a1e3a; border-color: #aa6aee; color: #cc8dff;
}
QRadioButton { color: #d4d4d4; font-size: 12px; spacing: 6px; }
QRadioButton::indicator {
    width: 13px; height: 13px; border-radius: 7px;
    border: 1px solid #555; background: #2a2a2a;
}
QRadioButton::indicator:checked {
    background: #4a90d9; border-color: #4a90d9;
}
QCheckBox { color: #d4d4d4; font-size: 12px; spacing: 6px; }
QCheckBox::indicator {
    width: 13px; height: 13px; border-radius: 3px;
    border: 1px solid #555; background: #2a2a2a;
}
QCheckBox::indicator:checked {
    background: #4a90d9; border-color: #4a90d9;
}
"""

_SEP_STYLE = "background: #333;"


class BooleanPanel(QWidget):
    confirmed = pyqtSignal(list, str, bool)
    # (body_ids, operation, keep_inputs)
    cancelled = pyqtSignal()
    preview_changed = pyqtSignal()
    picking_changed = pyqtSignal(bool)

    def __init__(self, workspace, parent=None):
        super().__init__(parent.window() if parent is not None else None)
        self.setWindowFlags(Qt.WindowType.Tool)
        self.setWindowTitle("Boolean")
        self._viewport = parent
        self.setObjectName("BooleanPanel")
        self.setStyleSheet(prefs.scale_stylesheet(_PANEL_STYLE))
        self.setMinimumWidth(prefs.scaled_px(280))
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        self._workspace = workspace
        self._picking = False

        self._build_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(10)
        root.setContentsMargins(14, 12, 14, 12)

        title = QLabel("Boolean")
        title.setObjectName("title")
        root.addWidget(title)

        root.addWidget(self._separator())

        # ── Operation ─────────────────────────────────────────────────
        root.addWidget(self._section_label("Operation"))
        self._op_group = QButtonGroup(self)
        self._op_union = QRadioButton("Union — merge bodies")
        self._op_subtract = QRadioButton("Subtract — cut tools from target")
        self._op_intersect = QRadioButton("Intersect — keep overlap")
        self._op_union.setChecked(True)
        self._op_group.addButton(self._op_union)
        self._op_group.addButton(self._op_subtract)
        self._op_group.addButton(self._op_intersect)
        self._op_union.toggled.connect(self._on_op_changed)
        self._op_subtract.toggled.connect(self._on_op_changed)
        self._op_intersect.toggled.connect(self._on_op_changed)
        root.addWidget(self._op_union)
        root.addWidget(self._op_subtract)
        root.addWidget(self._op_intersect)

        root.addWidget(self._separator())

        # ── Bodies / Target + Tools ───────────────────────────────────
        # Union/Intersect: single bodies list
        # Subtract: target (single) + tools (multi)
        self._target_label = QLabel("Bodies")
        self._target_label.setObjectName("section")

        target_header = QHBoxLayout()
        target_header.setSpacing(6)
        target_header.addWidget(self._target_label)
        target_header.addStretch()
        self._pick_btn = QPushButton("+ Pick")
        self._pick_btn.setObjectName("pick")
        self._pick_btn.setCheckable(True)
        self._pick_btn.clicked.connect(self._on_pick_toggle)
        target_header.addWidget(self._pick_btn)
        root.addLayout(target_header)

        self._target_list = SelectionList(empty_text="No bodies selected")
        self._target_list.entry_removed.connect(self._emit_preview)
        root.addWidget(self._target_list)

        # Subtract-specific: tools list (hidden for union/intersect)
        self._tool_label = QLabel("Tools")
        self._tool_label.setObjectName("section")
        self._tool_label.setVisible(False)

        tool_header = QHBoxLayout()
        tool_header.setSpacing(6)
        tool_header.addWidget(self._tool_label)
        tool_header.addStretch()
        self._pick_tool_btn = QPushButton("+ Pick")
        self._pick_tool_btn.setObjectName("pick_tool")
        self._pick_tool_btn.setCheckable(True)
        self._pick_tool_btn.setVisible(False)
        self._pick_tool_btn.clicked.connect(self._on_pick_tool_toggle)
        tool_header.addWidget(self._pick_tool_btn)
        root.addLayout(tool_header)

        self._tool_list = SelectionList(empty_text="No tool bodies")
        self._tool_list.entry_removed.connect(self._emit_preview)
        self._tool_list.setVisible(False)
        root.addWidget(self._tool_list)

        root.addWidget(self._separator())

        # ── Keep inputs ───────────────────────────────────────────────
        self._keep_chk = QCheckBox("Keep input bodies after operation")
        self._keep_chk.setToolTip(
            "When checked, all input bodies remain visible after the boolean.\n"
            "Unchecked: input bodies are consumed (hidden) by default.")
        self._keep_chk.toggled.connect(lambda _: self._emit_preview())
        root.addWidget(self._keep_chk)

        root.addWidget(self._separator())

        # ── Buttons ───────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.cancelled)
        btn_row.addWidget(cancel_btn)
        btn_row.addStretch()
        self._ok_btn = QPushButton("OK")
        self._ok_btn.setObjectName("ok")
        self._ok_btn.setDefault(True)
        self._ok_btn.clicked.connect(self._on_ok)
        btn_row.addWidget(self._ok_btn)
        root.addLayout(btn_row)

    def _separator(self):
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(_SEP_STYLE)
        sep.setFixedHeight(1)
        return sep

    def _section_label(self, text):
        lbl = QLabel(text.upper())
        lbl.setObjectName("section")
        return lbl

    # ------------------------------------------------------------------
    # Mode
    # ------------------------------------------------------------------

    def _get_operation(self) -> str:
        if self._op_subtract.isChecked():
            return "subtract"
        if self._op_intersect.isChecked():
            return "intersect"
        return "union"

    def _on_op_changed(self, _):
        op = self._get_operation()
        is_subtract = (op == "subtract")
        self._target_label.setText("Target" if is_subtract else "Bodies")
        self._tool_label.setVisible(is_subtract)
        self._tool_list.setVisible(is_subtract)
        self._pick_tool_btn.setVisible(is_subtract)
        self._emit_preview()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_operation(self, op: str):
        if op == "union":
            self._op_union.setChecked(True)
        elif op == "subtract":
            self._op_subtract.setChecked(True)
        elif op == "intersect":
            self._op_intersect.setChecked(True)

    def set_keep_inputs(self, keep: bool):
        self._keep_chk.setChecked(keep)

    def add_body_entry(self, body_id: str, label: str):
        self._target_list.add(body_id, label)

    def remove_body_entry(self, index: int):
        self._target_list.remove_at(index)

    def add_tool_entry(self, body_id: str, label: str):
        self._tool_list.add(body_id, label)

    def remove_tool_entry(self, index: int):
        self._tool_list.remove_at(index)

    def _has_selection(self) -> bool:
        op = self._get_operation()
        if op == "subtract":
            return len(self._target_list) > 0 and len(self._tool_list) > 0
        return len(self._target_list) >= 2

    def end_pick(self):
        self._picking = False
        self._pick_btn.setChecked(False)
        self._pick_btn.setProperty("active", False)
        self._pick_btn.style().unpolish(self._pick_btn)
        self._pick_btn.style().polish(self._pick_btn)
        self.picking_changed.emit(False)

    def end_pick_tool(self):
        self._picking = False
        self._pick_tool_btn.setChecked(False)
        self._pick_tool_btn.setProperty("active", False)
        self._pick_tool_btn.style().unpolish(self._pick_tool_btn)
        self._pick_tool_btn.style().polish(self._pick_tool_btn)
        self.picking_changed.emit(False)

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------

    def _emit_preview(self):
        self.preview_changed.emit()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_pick_toggle(self, checked: bool):
        if checked:
            self.end_pick_tool()
            self._picking = True
            self._pick_btn.setProperty("active", True)
            self._pick_btn.style().unpolish(self._pick_btn)
            self._pick_btn.style().polish(self._pick_btn)
            self.picking_changed.emit(True)
        else:
            self.end_pick()

    def _on_pick_tool_toggle(self, checked: bool):
        if checked:
            self.end_pick()
            self._picking = True
            self._pick_tool_btn.setProperty("active", True)
            self._pick_tool_btn.style().unpolish(self._pick_tool_btn)
            self._pick_tool_btn.style().polish(self._pick_tool_btn)
            self.picking_changed.emit(True)
        else:
            self.end_pick_tool()

    def _on_ok(self):
        if not self._has_selection():
            return
        op = self._get_operation()
        keep = self._keep_chk.isChecked()
        if op == "subtract":
            target_key = self._target_list.keys[0]
            tool_keys = self._tool_list.keys
            body_ids = [target_key] + tool_keys
        else:
            body_ids = self._target_list.keys
        self.confirmed.emit(body_ids, op, keep)

    # ------------------------------------------------------------------
    # Keyboard / window
    # ------------------------------------------------------------------

    def closeEvent(self, e):
        self.cancelled.emit()
        super().closeEvent(e)

    def keyPressEvent(self, e: QKeyEvent):
        if e.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            e.accept()
            self._on_ok()
        elif e.key() == Qt.Key.Key_Escape:
            if self._picking:
                self.end_pick()
                self.end_pick_tool()
            else:
                self.cancelled.emit()
        else:
            super().keyPressEvent(e)
