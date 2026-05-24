"""
gui/expr_spinbox.py

ExprSpinBox — a QLineEdit that accepts math expressions with unit suffixes.

Usage:
    box = ExprSpinBox(unit="mm", decimals=3)
    box.set_mm(10.0)       # set value in mm
    box.mm_value()         # -> float in mm, or None if invalid
    box.value_changed      # signal(float) emitted in mm on valid commit

The widget reads prefs.default_unit at construction time for the active unit,
but callers can pass unit= to override.

Display:  "10.000 mm"  (formatted in active unit)
Editing:  user types "1in + 5"  (parsed, converted to mm on commit)
Invalid:  red border, tooltip shows the error, value unchanged
"""

from __future__ import annotations
from cad.prefs import prefs
from PyQt6.QtWidgets import QLineEdit
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QPalette


class ExprSpinBox(QLineEdit):
    value_changed = pyqtSignal(float)   # emits mm value on valid commit

    _STYLE_NORMAL = """
        QLineEdit {
            background: #2a2a2a;
            color: #d4d4d4;
            border: 1px solid #444;
            border-radius: 3px;
            padding: 2px 6px;
            font-size: 12px;
            font-family: monospace;
        }
        QLineEdit:focus {
            border: 1px solid #4a90d9;
        }
    """
    _STYLE_ERROR = """
        QLineEdit {
            background: #2a1515;
            color: #c05050;
            border: 1px solid #8a2020;
            border-radius: 3px;
            padding: 2px 6px;
            font-size: 12px;
            font-family: monospace;
        }
        QLineEdit:focus {
            border: 1px solid #c05050;
        }
    """

    def __init__(self, unit: str | None = None, decimals: int | None = None,
                 parent=None):
        super().__init__(parent)
        from cad.prefs import prefs
        self._unit     = unit or prefs.default_unit
        self._decimals = decimals if decimals is not None else prefs.display_decimals
        self._mm_value: float = 0.0
        self._valid    = True

        self.setStyleSheet(prefs.scale_stylesheet(self._STYLE_NORMAL))
        self.setMinimumWidth(prefs.scaled_px(60))
        self.setAlignment(Qt.AlignmentFlag.AlignRight)

        self.editingFinished.connect(self._on_commit)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_mm(self, value_mm: float):
        """Set the displayed value from a mm float."""
        self._mm_value = value_mm
        self._valid    = True
        self._show_formatted()

    def mm_value(self) -> float | None:
        """Return current mm value, or None if in an error state."""
        return self._mm_value if self._valid else None

    @property
    def unit(self) -> str:
        return self._unit

    @unit.setter
    def unit(self, u: str):
        self._unit = u
        if self._valid:
            self._show_formatted()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _show_formatted(self):
        from cad.units import format_value
        self.setText(format_value(self._mm_value, self._unit, self._decimals))
        self.setStyleSheet(prefs.scale_stylesheet(self._STYLE_NORMAL))
        self.setToolTip("")

    def _on_commit(self):
        text = self.text().strip()
        if not text:
            self._show_formatted()   # revert to last valid
            return
        try:
            from cad.expression import parse_expr
            result = parse_expr(text, self._unit)
            self._mm_value = result
            self._valid    = True
            self._show_formatted()
            self.value_changed.emit(self._mm_value)
        except ValueError as ex:
            self._valid = False
            self.setStyleSheet(prefs.scale_stylesheet(self._STYLE_ERROR))
            self.setToolTip(str(ex))

    def focusInEvent(self, e):
        """On focus, show raw number in active unit for easy editing."""
        from cad.units import from_mm
        raw = from_mm(self._mm_value, self._unit)
        self.setText(f"{raw:.{self._decimals}f}")
        self.selectAll()
        super().focusInEvent(e)

    def focusOutEvent(self, e):
        self._on_commit()
        super().focusOutEvent(e)
