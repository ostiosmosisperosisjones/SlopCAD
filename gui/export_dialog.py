"""
gui/export_dialog.py

Export dialog — format picker + save-as trigger.
"""

from __future__ import annotations
from cad.prefs import prefs
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QButtonGroup, QRadioButton, QFileDialog, QFrame
)
from PyQt6.QtCore import Qt


_FORMATS = [
    ("STEP",  "*.step", True,  "ISO 10303 — universal CAD exchange"),
    ("STL",   "*.stl",  True,  "Mesh triangles — 3D printing, simulation"),
    ("3MF",   "*.3mf",  False, "Coming soon"),
]


class ExportDialog(QDialog):
    """
    Returns (format_key, path) via .get_result() after exec(), or (None, None)
    if cancelled.
    """

    def __init__(self, body_name: str, parent=None):
        super().__init__(parent)
        self._body_name = body_name
        self._result = (None, None)
        self.setWindowTitle("Export Body")
        self.setModal(True)
        self.setMinimumWidth(prefs.scaled_px(340))
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        root.setSpacing(12)
        root.setContentsMargins(16, 16, 16, 16)

        title = QLabel(f"Export  <b>{self._body_name}</b>")
        title.setStyleSheet(prefs.scale_stylesheet("font-size: 13px; color: #d4d4d4;"))
        root.addWidget(title)

        sep = QFrame()
        sep.setFixedHeight(1)
        sep.setStyleSheet("background: #333;")
        root.addWidget(sep)

        fmt_label = QLabel("Format")
        fmt_label.setStyleSheet(prefs.scale_stylesheet("color: #888; font-size: 11px;"))
        root.addWidget(fmt_label)

        self._btn_group = QButtonGroup(self)
        self._radio: dict[str, QRadioButton] = {}

        for key, _ext, enabled, hint in _FORMATS:
            rb = QRadioButton(key)
            rb.setEnabled(enabled)
            rb.setToolTip(hint)
            rb.setStyleSheet(prefs.scale_stylesheet("color: #d4d4d4; font-size: 12px;"))
            if not enabled:
                rb.setStyleSheet(prefs.scale_stylesheet("color: #555; font-size: 12px;"))
            self._btn_group.addButton(rb)
            self._radio[key] = rb
            root.addWidget(rb)

        self._radio["STEP"].setChecked(True)

        root.addSpacing(4)

        btns = QHBoxLayout()
        btns.setSpacing(8)
        cancel = QPushButton("Cancel")
        cancel.setStyleSheet("color: #888;")
        cancel.clicked.connect(self.reject)
        btns.addWidget(cancel)

        btns.addStretch()

        export_btn = QPushButton("Save As…")
        export_btn.setDefault(True)
        export_btn.clicked.connect(self._on_save)
        btns.addWidget(export_btn)

        root.addLayout(btns)

    def _selected_format(self) -> tuple[str, str] | None:
        for key, ext, enabled, _ in _FORMATS:
            if enabled and self._radio[key].isChecked():
                return key, ext
        return None

    def _on_save(self):
        fmt = self._selected_format()
        if fmt is None:
            return
        key, ext = fmt
        filter_str = f"{key} Files ({ext})"
        default_name = f"{self._body_name}{ext.lstrip('*')}"
        path, _ = QFileDialog.getSaveFileName(
            self, f"Export as {key}", default_name, filter_str
        )
        if path:
            self._result = (key, path)
            self.accept()

    def get_result(self) -> tuple[str | None, str | None]:
        return self._result
