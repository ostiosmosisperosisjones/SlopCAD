"""
gui/selection_list.py

SelectionList — reusable widget that shows a removable list of labelled
selection entries (faces, edges, bodies, …) with per-entry error state.

Usage
-----
    sl = SelectionList(empty_text="No faces selected", parent=self)
    sl.entry_removed.connect(self._on_entry_removed)

    # add / remove
    sl.add(key, label)          # key is any hashable — (body_id, face_idx), int, …
    sl.remove_at(index)         # emits entry_removed(index)
    sl.clear()

    # error feedback (call from main thread)
    sl.set_error(index, "BRepOffset_C0Geometry: …")
    sl.clear_errors()

    # query
    sl.has_valid_entries        # True iff at least one entry and all are valid
    sl.keys                     # list of keys in order
    len(sl)                     # number of entries
"""

from __future__ import annotations
from cad.prefs import prefs

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSizePolicy,
)
from PyQt6.QtCore import pyqtSignal


_LABEL_OK  = "color: #d4d4d4; font-size: 11px; font-family: monospace;"
_LABEL_ERR = "color: #cc4444; font-size: 11px; font-family: monospace;"
_RM_STYLE  = (
    "QPushButton { background: #2a2a2a; color: #666; border: none; "
    "font-size: 10px; padding: 1px; }"
    "QPushButton:hover { color: #cc4444; }"
)


class SelectionList(QWidget):
    """Scrollable list of labelled entries with ✕ remove buttons and error state.

    When reorderable=True, each row gets ▲/▼ buttons to swap with its neighbor
    and the label is prefixed with #N to make the order visible.
    """

    entry_removed  = pyqtSignal(int)        # index that was removed
    order_changed  = pyqtSignal()           # fires after a successful move

    def __init__(self, empty_text: str = "Nothing selected", parent=None,
                 reorderable: bool = False):
        super().__init__(parent)

        # (key, label, valid, tooltip)
        self._entries: list[tuple] = []
        self._reorderable = reorderable

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(2)

        self._empty_label = QLabel(empty_text)
        self._empty_label.setStyleSheet(prefs.scale_stylesheet(
            "color: #555; font-size: 11px; font-family: monospace;"))
        self._layout.addWidget(self._empty_label)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def has_valid_entries(self) -> bool:
        return bool(self._entries) and all(e[2] for e in self._entries)

    @property
    def keys(self) -> list:
        return [e[0] for e in self._entries]

    def __len__(self) -> int:
        return len(self._entries)

    def add(self, key, label: str, *, valid: bool = True, tooltip: str = "") -> bool:
        """Add an entry. Returns False (no-op) if key already present."""
        if any(e[0] == key for e in self._entries):
            return False
        self._entries.append((key, label, valid, tooltip))
        self._rebuild()
        return True

    def remove_at(self, index: int):
        if 0 <= index < len(self._entries):
            self._entries.pop(index)
            self._rebuild()
            self.entry_removed.emit(index)

    def clear(self):
        self._entries.clear()
        self._rebuild()

    def set_error(self, index: int, message: str):
        """Mark entry red with a hover tooltip. Safe to call from main thread only."""
        if 0 <= index < len(self._entries):
            key, label, _valid, _tip = self._entries[index]
            self._entries[index] = (key, label, False, message)
            self._rebuild()

    def clear_errors(self):
        """Reset all entries to valid state."""
        changed = False
        for i, (key, label, valid, tip) in enumerate(self._entries):
            if not valid:
                self._entries[i] = (key, label, True, "")
                changed = True
        if changed:
            self._rebuild()

    def set_label(self, index: int, label: str):
        if 0 <= index < len(self._entries):
            key, _label, valid, tip = self._entries[index]
            self._entries[index] = (key, label, valid, tip)
            self._rebuild()

    def move(self, src: int, dst: int):
        """Move entry at src to dst, clamping dst. Fires order_changed if changed."""
        if not (0 <= src < len(self._entries)):
            return
        dst = max(0, min(dst, len(self._entries) - 1))
        if src == dst:
            return
        item = self._entries.pop(src)
        self._entries.insert(dst, item)
        self._rebuild()
        self.order_changed.emit()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _rebuild(self):
        while self._layout.count():
            item = self._layout.takeAt(0)
            w = item.widget()
            if w and w is not self._empty_label:
                w.deleteLater()

        if not self._entries:
            self._layout.addWidget(self._empty_label)
            self._empty_label.show()
            return

        self._empty_label.hide()
        n = len(self._entries)
        for i, (key, label, valid, tooltip) in enumerate(self._entries):
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(4)

            display_label = f"#{i + 1}  {label}" if self._reorderable else label
            lbl = QLabel(display_label)
            lbl.setStyleSheet(_LABEL_OK if valid else _LABEL_ERR)
            lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            if tooltip:
                lbl.setToolTip(tooltip)
            row_layout.addWidget(lbl)

            if self._reorderable:
                up = QPushButton("▲")
                up.setFixedWidth(prefs.scaled_px(22))
                up.setStyleSheet(prefs.scale_stylesheet(_RM_STYLE))
                up.setEnabled(i > 0)
                up.clicked.connect(lambda _, idx=i: self.move(idx, idx - 1))
                row_layout.addWidget(up)

                dn = QPushButton("▼")
                dn.setFixedWidth(prefs.scaled_px(22))
                dn.setStyleSheet(prefs.scale_stylesheet(_RM_STYLE))
                dn.setEnabled(i < n - 1)
                dn.clicked.connect(lambda _, idx=i: self.move(idx, idx + 1))
                row_layout.addWidget(dn)

            rm = QPushButton("✕")
            rm.setFixedWidth(prefs.scaled_px(22))
            rm.setStyleSheet(prefs.scale_stylesheet(_RM_STYLE))
            rm.clicked.connect(lambda _, idx=i: self.remove_at(idx))
            row_layout.addWidget(rm)

            self._layout.addWidget(row)
