"""
gui/value_dialog.py

ask_value() — modal "enter a length" dialog built on ExprSpinBox, so sketch
dimensions accept expressions, unit suffixes and parameter names
("width/2", "1in + 3").  Replaces QInputDialog.getDouble, which only took a
bare number.
"""

from __future__ import annotations
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QDialogButtonBox


def ask_value(parent, title: str, label: str, current_mm: float,
              expr: str | None = None, expr_unit: str | None = None,
              unit: str | None = None, max_value: float | None = None):
    """Returns (value_mm, binding | None) on OK, or None on cancel.
    unit="deg" asks for an angle; max_value is an exclusive upper bound.
    binding is {"expr", "unit"} when the value references a parameter."""
    from gui.expr_spinbox import ExprSpinBox

    dlg = QDialog(parent)
    dlg.setWindowTitle(title)
    lay = QVBoxLayout(dlg)
    lay.addWidget(QLabel(label))
    box = ExprSpinBox(unit=unit) if unit else ExprSpinBox()
    box.set_binding({"expr": expr, "unit": expr_unit or box.unit} if expr else None,
                    current_mm)
    lay.addWidget(box)
    hint = QLabel("Number, expression or parameter name — e.g. width/2")
    hint.setStyleSheet("color: #777; font-size: 10px;")
    lay.addWidget(hint)
    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                               | QDialogButtonBox.StandardButton.Cancel)
    lay.addWidget(buttons)

    def _accept():
        box._on_commit()
        v = box.mm_value()
        if v is not None and v > 0 and (max_value is None or v < max_value):
            dlg.accept()
    buttons.accepted.connect(_accept)
    buttons.rejected.connect(dlg.reject)

    box.setFocus()
    if dlg.exec() != QDialog.DialogCode.Accepted:
        return None
    return box.mm_value(), box.binding()
