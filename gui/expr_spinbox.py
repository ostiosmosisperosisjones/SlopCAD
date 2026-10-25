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

Parameters: an expression that names a document parameter ("width/2") is
kept as a *binding* — binding() returns it for the op's params["exprs"],
the field shows "ƒ 20.000 mm" with a tinted background, and focusing it
shows the expression again.  ExprSpinBox.param_source (set by mainwindow)
supplies the current parameter values.
"""

from __future__ import annotations
from cad.prefs import prefs
from PyQt6.QtWidgets import QLineEdit
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QPalette


def collect_bindings(**fields) -> dict | None:
    """{param_key: binding} for every bound ExprSpinBox among *fields*
    (key=widget or None) — the value for an op's extra_params["exprs"].
    None when nothing is bound."""
    out = {k: b for k, w in fields.items()
           if w is not None and (b := w.binding())}
    return out or None


def exprs_extra(**fields) -> dict | None:
    """collect_bindings wrapped as an extra_params dict for Op.commit."""
    b = collect_bindings(**fields)
    return {"exprs": b} if b else None


class ExprSpinBox(QLineEdit):
    value_changed = pyqtSignal(float)   # emits mm value on valid commit

    # () -> {name: Quantity}; set by mainwindow to the open document's params.
    param_source = None

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
    _STYLE_BOUND = """
        QLineEdit {
            background: #1f2a33;
            color: #9cc4e4;
            border: 1px solid #35506a;
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
        self._expr: str | None = None       # parameter expression, if bound
        self._expr_unit: str   = self._unit
        self._shown: str | None = None      # text we displayed for _mm_value

        self.setStyleSheet(prefs.scale_stylesheet(self._STYLE_NORMAL))
        self.setMinimumWidth(prefs.scaled_px(60))
        self.setAlignment(Qt.AlignmentFlag.AlignRight)

        self.editingFinished.connect(self._on_commit)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_mm(self, value_mm: float):
        """Set the displayed value from a mm float (clears any binding)."""
        self._mm_value = value_mm
        self._valid    = True
        self._expr     = None
        self._show_formatted()

    def set_binding(self, binding: dict | None, value_mm: float):
        """Restore a field from an entry: value plus its params["exprs"]
        binding (or None)."""
        self.set_mm(value_mm)
        if binding:
            self._expr      = binding.get("expr")
            self._expr_unit = binding.get("unit", self._unit)
            self._show_formatted()

    def binding(self) -> dict | None:
        """The parameter binding for this field, or None if it holds a
        plain value."""
        if not self._expr or not self._valid:
            return None
        from cad.params import make_binding
        return make_binding(self._expr, self._expr_unit)

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
        txt = format_value(self._mm_value, self._unit, self._decimals)
        if self._expr:
            self._show_text(f"ƒ {txt}")
            self.setStyleSheet(prefs.scale_stylesheet(self._STYLE_BOUND))
            self.setToolTip(f"= {self._expr}")
        else:
            self._show_text(txt)
            self.setStyleSheet(prefs.scale_stylesheet(self._STYLE_NORMAL))
            self.setToolTip("")

    def _show_text(self, text: str):
        self._shown = text
        self.setText(text)

    def _on_commit(self):
        text = self.text().strip()
        if not text:
            self._show_formatted()   # revert to last valid
            return
        if self._expr and (text == self._expr or text.startswith("ƒ ")):
            self._show_formatted()  # focus left without editing the binding
            return
        if self._valid and text == self._shown:
            # Unedited: keep the exact value, not its rounded display
            # (4.2mm shown as "0.165 in" must not commit as 4.191mm).
            self._show_formatted()
            return
        try:
            from cad.expression import parse_expr, expr_refs
            src = ExprSpinBox.param_source
            names = src() if src is not None else {}
            result = parse_expr(text, self._unit, names)
            self._mm_value = result
            self._valid    = True
            if expr_refs(text) & set(names):
                self._expr, self._expr_unit = text, self._unit
            else:
                self._expr = None
            self._show_formatted()
            self.value_changed.emit(self._mm_value)
        except ValueError as ex:
            self._valid = False
            self.setStyleSheet(prefs.scale_stylesheet(self._STYLE_ERROR))
            self.setToolTip(str(ex))

    def focusInEvent(self, e):
        """On focus, show raw number in active unit for easy editing."""
        from cad.units import from_mm
        if self._expr:
            self.setText(self._expr)
            self.setStyleSheet(prefs.scale_stylesheet(self._STYLE_NORMAL))
        else:
            raw = from_mm(self._mm_value, self._unit)
            self._show_text(f"{raw:.{self._decimals}f}")
        self.selectAll()
        super().focusInEvent(e)

    def focusOutEvent(self, e):
        self._on_commit()
        super().focusOutEvent(e)
