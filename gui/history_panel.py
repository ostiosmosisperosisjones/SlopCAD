"""
gui/history_panel.py

HistoryPanel — flat chronological operation list.
  - Click any entry to seek the history cursor there
  - Body-selection highlighting (via PartsPanel)
  - Hidden body entry dimming
  - Double-click past entry to edit parameters (parametric replay)
  - Sketch entries have a visibility eye toggle
  - Right-click context menu: Delete, Move Up, Move Down
  - Drag-and-drop reordering within the list
"""

from __future__ import annotations
from cad.prefs import prefs
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QSizePolicy, QDialog, QFormLayout,
    QDoubleSpinBox, QDialogButtonBox, QMessageBox,
    QListWidget, QListWidgetItem, QAbstractItemView, QMenu,
)
from gui.expr_spinbox import ExprSpinBox
from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtGui import QAction
from cad.history import History, HistoryEntry
from cad.workspace import Workspace
from cad.operations import EDIT_SCHEMA

_BG            = "#1e1e1e"
_BG_ENTRY      = "#2a2a2a"
_BG_EDITING    = "#2a2518"
_TEXT_EDITING  = "#d4aa60"
_BORDER_EDITING = "#c8882a"
_BG_CURRENT    = "#2f3f52"
_BG_FUTURE     = "#212121"
_BG_DIM        = "#1e1e1e"
_BG_HIDDEN     = "#191919"
_BG_DIVERGED   = "#252018"
_BG_ERROR      = "#2a1515"
_TEXT          = "#d4d4d4"
_TEXT_DIM      = "#404040"
_TEXT_HIDDEN   = "#2e2e2e"
_TEXT_FUTURE   = "#383838"
_TEXT_DIVERGED = "#7a6e3a"
_TEXT_ERROR    = "#c05050"
_BORDER_SEL    = "#4a90d9"
_BORDER_PAST   = "#404040"
_BORDER_DIM    = "#282828"
_BORDER_FUTURE = "#2a2a2a"
_BORDER_DIVERGED = "#6e5a20"
_BORDER_ERROR  = "#8a2020"

_OP_ACCENT = {
    "import":       "#505050",
    "extrude":      "#3a6e44",
    "cut":          "#6e3a3a",
    "sketch":       "#4a6e8a",
    "revolve_cut":  "#6e3a3a",
    "offset_plane": "#6acce0",
    "loft":         "#7a9e44",
    "loft_cut":     "#9e7a44",
    "chamfer":      "#4a8a82",
}


def _op_icon(op: str) -> str:
    return {
        "import":       "⬡",
        "extrude":      "▲",
        "cut":          "▼",
        "sketch":       "⬜",
        "revolve_cut":  "▼",
        "offset_plane": "▱",
        "loft":         "⌬",
        "loft_cut":     "⌬",
        "chamfer":      "◢",
    }.get(op, "•")


# ---------------------------------------------------------------------------
# Edit dialog
# ---------------------------------------------------------------------------

class _EditDialog(QDialog):
    def __init__(self, entry: HistoryEntry, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Edit — {entry.label}")
        self.setMinimumWidth(prefs.scaled_px(300))
        self.result_params = None

        schema = EDIT_SCHEMA.get(entry.operation, [])
        if not schema:
            self.result_params = entry.params.copy()
            return

        layout = QFormLayout(self)
        layout.setContentsMargins(16, 16, 16, 8)
        layout.setSpacing(10)

        self._spinboxes: dict[str, ExprSpinBox | QDoubleSpinBox] = {}

        signed_val_mm = float(entry.params.get("distance", 0))
        if entry.operation == "cut":
            signed_val_mm = -abs(signed_val_mm)

        for row_def in schema:
            key, label, typ, mn, mx, decimals = row_def[:6]
            kind = row_def[6] if len(row_def) > 6 else None

            if kind == "length":
                spin = ExprSpinBox(decimals=decimals)
                val_mm = signed_val_mm if key == "distance" else float(entry.params.get(key, 0))
                spin.set_mm(val_mm)
            else:
                spin = QDoubleSpinBox()
                spin.setDecimals(decimals)
                spin.setMinimum(-mx)
                spin.setMaximum(mx)
                spin.setValue(signed_val_mm if key == "distance"
                              else float(entry.params.get(key, 0)))

            self._spinboxes[key] = spin
            layout.addRow(label + "  (±):", spin)

        note = QLabel("Positive = add material   /   Negative = cut")
        note.setStyleSheet(prefs.scale_stylesheet("color: #555; font-size: 11px;"))
        layout.addRow(note)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def _on_accept(self):
        result = {}
        for k, s in self._spinboxes.items():
            if isinstance(s, ExprSpinBox):
                val = s.mm_value()
                if val is None:
                    return   # invalid expression — don't accept
                result[k] = val
            else:
                result[k] = s.value()
        self.result_params = result
        self.accept()


# ---------------------------------------------------------------------------
# Entry widget
# ---------------------------------------------------------------------------

class _EntryWidget(QFrame):
    sketch_vis_toggled = pyqtSignal(int)   # emits entry index
    double_clicked     = pyqtSignal(int)   # emits entry index

    def __init__(self, entry: HistoryEntry, index: int,
                 is_current: bool, is_future: bool,
                 body_name: str, is_selected_body: bool,
                 is_hidden_body: bool):
        super().__init__()
        self._index = index

        if is_hidden_body:
            bg, text_color, border = _BG_HIDDEN, _TEXT_HIDDEN, _BORDER_DIM
        elif entry.editing:
            bg, text_color, border = _BG_EDITING, _TEXT_EDITING, _BORDER_EDITING
        elif entry.error:
            bg, text_color, border = _BG_ERROR, _TEXT_ERROR, _BORDER_ERROR
        elif is_future and entry.diverged:
            bg, text_color, border = _BG_DIVERGED, _TEXT_DIVERGED, _BORDER_DIVERGED
        elif is_future:
            bg, text_color, border = _BG_FUTURE, _TEXT_FUTURE, _BORDER_FUTURE
        elif is_current:
            bg, text_color, border = _BG_CURRENT, _TEXT, _BORDER_SEL
        elif not is_selected_body:
            bg, text_color, border = _BG_DIM, _TEXT_DIM, _BORDER_DIM
        else:
            bg         = _BG_ENTRY
            text_color = _TEXT
            border     = _OP_ACCENT.get(entry.operation, _BORDER_PAST)

        self.setStyleSheet(f"""
            QFrame {{
                background: {bg};
                border-left: 3px solid {border};
                border-radius: 3px;
                margin: 1px 4px;
            }}
        """)

        # Show ✎ for any op the panel can reopen — EDIT_SCHEMA covers the
        # generic-dialog path; the hard-coded reopen branches below cover
        # the rest (thicken, revolve, loft, …).
        _REOPENABLE = {"extrude", "cut", "thicken", "revolve", "revolve_cut",
                       "fillet", "chamfer", "loft", "loft_cut", "offset_plane"}
        editable = ((entry.operation in EDIT_SCHEMA
                     or entry.operation in _REOPENABLE)
                    and not is_future)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 4, 10, 4)
        outer.setSpacing(1)

        if entry.operation not in ("import", "offset_plane"):
            body_lbl = QLabel(body_name)
            body_lbl.setStyleSheet(prefs.scale_stylesheet(
                "color: #333; font-size: 10px; "
                "border: none; background: transparent;"))
            outer.addWidget(body_lbl)

        row = QHBoxLayout()
        row.setSpacing(6)
        row.setContentsMargins(0, 0, 0, 0)

        tag = QLabel(_op_icon(entry.operation))
        tag.setFixedWidth(16)
        tag.setStyleSheet(prefs.scale_stylesheet(
            f"color: {border}; font-size: 11px; "
            "border: none; background: transparent;"))
        row.addWidget(tag)

        from cad.units import format_op_label
        lbl_text = (format_op_label(entry.operation, entry.params)
                    if entry.operation in ("extrude", "cut",
                                            "revolve", "revolve_cut",
                                            "loft", "loft_cut",
                                            "chamfer")
                    else entry.label)
        if entry.editing:
            lbl_text = "✎ " + lbl_text
        elif entry.error:
            lbl_text = "⚠ " + lbl_text
        lbl = QLabel(lbl_text)
        lbl.setStyleSheet(prefs.scale_stylesheet(
            f"color: {text_color}; font-size: 12px; "
            "border: none; background: transparent;"))
        lbl.setSizePolicy(QSizePolicy.Policy.Expanding,
                          QSizePolicy.Policy.Preferred)
        if entry.error and entry.error_msg:
            lbl.setToolTip(entry.error_msg)
        row.addWidget(lbl)

        if editable:
            hint = QLabel("✎")
            hint.setStyleSheet(prefs.scale_stylesheet(
                "color: #303030; font-size: 10px; "
                "border: none; background: transparent;"))
            hint.setToolTip("Double-click to edit")
            row.addWidget(hint)

        # Sketch entries get a clickable visibility eye
        if entry.operation == "sketch":
            se = entry.params.get("sketch_entry")
            is_visible = (se is None or se.visible)
            eye_btn = QPushButton("◉" if is_visible else "○")
            eye_btn.setFixedWidth(18)
            eye_btn.setToolTip("Toggle sketch visibility")
            eye_btn.setStyleSheet(prefs.scale_stylesheet("""
                QPushButton {
                    color: #4a6e8a;
                    font-size: 11px;
                    background: transparent;
                    border: none;
                    padding: 0;
                }
                QPushButton:hover { color: #7ab3d4; }
            """))
            eye_btn.clicked.connect(lambda _: self.sketch_vis_toggled.emit(self._index))
            row.addWidget(eye_btn)

        outer.addLayout(row)

    def mouseDoubleClickEvent(self, event):
        self.double_clicked.emit(self._index)
        event.accept()


# ---------------------------------------------------------------------------
# Drag-and-drop QListWidget
# ---------------------------------------------------------------------------

class _HistoryList(QListWidget):
    reorder_requested = pyqtSignal(int, int)  # src, dst
    delete_requested  = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)
        self.setStyleSheet("""
            QListWidget {
                background: transparent;
                border: none;
                outline: none;
            }
            QListWidget::item {
                background: transparent;
                border: none;
                padding: 0;
            }
            QListWidget::item:selected {
                background: transparent;
                border: none;
            }
        """)
        self._drag_src: int | None = None

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            item = self.itemAt(e.pos())
            self._drag_src_history_idx = (
                item.data(Qt.ItemDataRole.UserRole) if item else None)
        super().mousePressEvent(e)

    def dropEvent(self, e):
        # Snapshot history-index order before Qt moves the item
        order_before = [self.item(i).data(Qt.ItemDataRole.UserRole)
                        for i in range(self.count())]
        super().dropEvent(e)
        order_after = [self.item(i).data(Qt.ItemDataRole.UserRole)
                       for i in range(self.count())]
        src_hidx = self._drag_src_history_idx
        self._drag_src_history_idx = None
        if src_hidx is None or order_before == order_after:
            return
        try:
            src_pos = order_before.index(src_hidx)
            dst_pos = order_after.index(src_hidx)
        except ValueError:
            return
        if src_pos != dst_pos:
            # Pass history indices of src item and the item now occupying src's
            # old slot — history.reorder maps these back to positions itself.
            # Use order_before[dst_pos] as the "swap target" history index.
            self.reorder_requested.emit(src_hidx, order_before[dst_pos])

    def _show_context_menu(self, pos):
        item = self.itemAt(pos)
        if item is None:
            return
        hidx  = item.data(Qt.ItemDataRole.UserRole)  # real history index
        vrow  = self.row(item)
        n     = self.count()

        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background: #1e1e1e;
                color: #d4d4d4;
                border: 1px solid #3a3a3a;
            }
            QMenu::item:selected { background: #2f3f52; }
        """)

        del_act = QAction("Delete", self)
        del_act.triggered.connect(lambda: self.delete_requested.emit(hidx))
        menu.addAction(del_act)

        menu.addSeparator()

        # For Move Up/Down we need the history index of the neighbour
        up_act = QAction("Move Up", self)
        up_act.setEnabled(vrow > 0)
        if vrow > 0:
            nbr_up = self.item(vrow - 1).data(Qt.ItemDataRole.UserRole)
            up_act.triggered.connect(lambda: self.reorder_requested.emit(hidx, nbr_up))
        menu.addAction(up_act)

        dn_act = QAction("Move Down", self)
        dn_act.setEnabled(vrow < n - 1)
        if vrow < n - 1:
            nbr_dn = self.item(vrow + 1).data(Qt.ItemDataRole.UserRole)
            dn_act.triggered.connect(lambda: self.reorder_requested.emit(hidx, nbr_dn))
        menu.addAction(dn_act)

        menu.exec(self.viewport().mapToGlobal(pos))


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------

class HistoryPanel(QWidget):
    seek_requested           = pyqtSignal(int)
    replay_requested         = pyqtSignal(int)
    sketch_vis_changed       = pyqtSignal()
    reenter_sketch_requested = pyqtSignal(int)
    reopen_extrude_requested      = pyqtSignal(int)
    reopen_thicken_requested      = pyqtSignal(int)
    reopen_revolve_requested      = pyqtSignal(int)
    reopen_fillet_requested       = pyqtSignal(int)
    reopen_loft_requested         = pyqtSignal(int)
    reopen_offset_plane_requested = pyqtSignal(int)
    reopen_chamfer_requested      = pyqtSignal(int)
    reopen_boolean_requested      = pyqtSignal(int)
    delete_requested         = pyqtSignal(int)
    reorder_requested        = pyqtSignal(int, int)

    def __init__(self, workspace: Workspace, history: History, parent=None):
        super().__init__(parent)
        self._workspace      = workspace
        self._history        = history
        self._selected_body: str | None = None
        self._hidden_bodies: set[str]   = set()
        self._setup()
        self.refresh()

    def set_selected_body(self, body_id: str | None):
        self._selected_body = body_id
        self.refresh()

    def set_body_hidden(self, body_id: str, hidden: bool):
        if hidden:
            self._hidden_bodies.add(body_id)
        else:
            self._hidden_bodies.discard(body_id)
        self.refresh()

    def _setup(self):
        self.setStyleSheet(f"background: {_BG};")

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        header = QLabel("  History")
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

        self._list = _HistoryList()
        self._list.reorder_requested.connect(self.reorder_requested)
        self._list.delete_requested.connect(self.delete_requested)
        self._list.itemClicked.connect(self._on_item_clicked)
        self._list.itemDoubleClicked.connect(self._on_item_double_clicked)
        root.addWidget(self._list, stretch=1)

        btn_style = """
            QPushButton {
                background: #222; color: #666; border: none;
                border-top: 1px solid #2a2a2a;
                font-size: 12px; padding: 6px 0;
            }
            QPushButton:hover   { background: #2a2a2a; color: #d4d4d4; }
            QPushButton:pressed { background: #1a1a1a; }
            QPushButton:disabled { color: #2e2e2e; }
        """
        btn_row = QHBoxLayout()
        btn_row.setSpacing(0)
        btn_row.setContentsMargins(0, 0, 0, 0)

        self._undo_btn = QPushButton("↩ Undo")
        self._undo_btn.setStyleSheet(
            btn_style + "QPushButton { border-right: 1px solid #2a2a2a; }")
        self._undo_btn.clicked.connect(
            lambda: self.seek_requested.emit(self._history.cursor - 1))
        btn_row.addWidget(self._undo_btn)

        self._redo_btn = QPushButton("Redo ↪")
        self._redo_btn.setStyleSheet(btn_style)
        self._redo_btn.clicked.connect(
            lambda: self.seek_requested.emit(self._history.cursor + 1))
        btn_row.addWidget(self._redo_btn)

        bc = QWidget()
        bc.setLayout(btn_row)
        root.addWidget(bc)

    def refresh(self):
        if self._history is None:
            return
        self._list.clear()

        entries = self._history.entries
        cursor  = self._history.cursor

        for i, entry in enumerate(entries):
            # Hide all split-body import entries — they are tracked internally
            # but not exposed in the history panel.
            if (entry.operation == "import"
                    and ("split_from" in entry.params
                         or "source_entry_id" in entry.params)):
                continue

            body      = self._workspace.bodies.get(entry.body_id)
            body_name = body.name if body else "?"

            is_selected_body = (
                self._selected_body is None
                or entry.body_id == self._selected_body
            )
            is_hidden_body = entry.body_id in self._hidden_bodies

            w = _EntryWidget(
                entry, i,
                is_current       = (i == cursor),
                is_future        = (i > cursor),
                body_name        = body_name,
                is_selected_body = is_selected_body,
                is_hidden_body   = is_hidden_body,
            )
            w.sketch_vis_toggled.connect(self._on_sketch_vis_toggled)
            w.double_clicked.connect(self._on_item_double_clicked_idx)

            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, i)  # store real history index
            item.setSizeHint(QSize(0, w.sizeHint().height() + 2))
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsDragEnabled)
            self._list.addItem(item)
            self._list.setItemWidget(item, w)

        self._undo_btn.setEnabled(self._history.can_undo)
        self._redo_btn.setEnabled(self._history.can_redo)
        # Scroll to bottom (most recent entry)
        self._list.scrollToBottom()

    def _item_index(self, item: QListWidgetItem) -> int:
        """Return the real history index stored on a list item."""
        return item.data(Qt.ItemDataRole.UserRole)

    def _on_item_clicked(self, item: QListWidgetItem):
        self.seek_requested.emit(self._item_index(item))

    def _on_item_double_clicked(self, item: QListWidgetItem):
        self._on_item_double_clicked_idx(self._item_index(item))

    def _on_item_double_clicked_idx(self, index: int):
        entries = self._history.entries
        if index >= len(entries):
            return
        entry = entries[index]

        # Sketch entry: seek to it, then re-open for editing
        if entry.operation == "sketch":
            se = entry.params.get("sketch_entry")
            if se is None:
                return
            self.seek_requested.emit(index)
            self.reenter_sketch_requested.emit(index)
            return

        # Extrude/cut: de-compute and reopen the ExtrudePanel with preserved values
        if entry.operation in ("extrude", "cut"):
            if index > self._history.cursor:
                QMessageBox.information(
                    self, "Can't edit future entry",
                    "Single-click to seek here first, then double-click to edit.")
                return
            self.reopen_extrude_requested.emit(index)
            return

        # Thicken: reopen ThickenPanel
        if entry.operation == "thicken":
            if index > self._history.cursor:
                QMessageBox.information(
                    self, "Can't edit future entry",
                    "Single-click to seek here first, then double-click to edit.")
                return
            self.reopen_thicken_requested.emit(index)
            return

        # Revolve / revolve-cut: reopen RevolvePanel
        if entry.operation in ("revolve", "revolve_cut"):
            if index > self._history.cursor:
                QMessageBox.information(
                    self, "Can't edit future entry",
                    "Single-click to seek here first, then double-click to edit.")
                return
            self.reopen_revolve_requested.emit(index)
            return

        # Fillet: reopen Fillet3DPanel
        if entry.operation == "fillet":
            if index > self._history.cursor:
                QMessageBox.information(
                    self, "Can't edit future entry",
                    "Single-click to seek here first, then double-click to edit.")
                return
            self.reopen_fillet_requested.emit(index)
            return

        # Loft / loft-cut: reopen LoftPanel
        if entry.operation in ("loft", "loft_cut"):
            if index > self._history.cursor:
                QMessageBox.information(
                    self, "Can't edit future entry",
                    "Single-click to seek here first, then double-click to edit.")
                return
            self.reopen_loft_requested.emit(index)
            return

        # Offset plane: reopen OffsetPlanePanel
        if entry.operation == "offset_plane":
            if index > self._history.cursor:
                QMessageBox.information(
                    self, "Can't edit future entry",
                    "Single-click to seek here first, then double-click to edit.")
                return
            self.reopen_offset_plane_requested.emit(index)
            return

        # Chamfer: reopen ChamferPanel
        if entry.operation == "chamfer":
            if index > self._history.cursor:
                QMessageBox.information(
                    self, "Can't edit future entry",
                    "Single-click to seek here first, then double-click to edit.")
                return
            self.reopen_chamfer_requested.emit(index)
            return

        # Boolean: reopen BooleanPanel
        if entry.operation in ("union", "subtract", "intersect"):
            if index > self._history.cursor:
                QMessageBox.information(
                    self, "Can't edit future entry",
                    "Single-click to seek here first, then double-click to edit.")
                return
            self.reopen_boolean_requested.emit(index)
            return

        if entry.operation not in EDIT_SCHEMA:
            return
        if index > self._history.cursor:
            QMessageBox.information(
                self, "Can't edit future entry",
                "Single-click to seek here first, then double-click to edit.")
            return

        dlg = _EditDialog(entry, parent=self)
        if dlg.exec() != QDialog.DialogCode.Accepted or dlg.result_params is None:
            return

        dist = dlg.result_params.get("distance", 0)
        if dist == 0:
            return

        self._history.update_entry_op(index, {}, new_distance=dist)
        self.replay_requested.emit(index)

    def _on_sketch_vis_toggled(self, index: int):
        entries = self._history.entries
        if index >= len(entries):
            return
        entry = entries[index]
        se = entry.params.get("sketch_entry")
        if se is None:
            return
        se.visible = not se.visible
        self.refresh()
        self.sketch_vis_changed.emit()
