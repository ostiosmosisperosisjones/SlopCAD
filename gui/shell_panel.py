"""
gui/shell_panel.py

ShellPanel — the thicken panel's face list + thickness, relabelled: the
picked faces are left open, and the mode picks which side of the skin the
walls grow on.
"""

from __future__ import annotations

from gui.thicken_panel import ThickenPanel
from PyQt6.QtWidgets import QLabel


class ShellPanel(ThickenPanel):
    # thicken_requested(float) / preview_changed(float) carry the wall
    # thickness (> 0); read is_outside() for the side.

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Shell")
        for lbl in self.findChildren(QLabel):
            if lbl.objectName() == "title":
                lbl.setText("Shell")
            elif lbl.objectName() == "section" and lbl.text() == "FACES":
                lbl.setText("OPEN FACES")
        self._pick_face_btn.setText("+ Add Open Face")
        self._face_list._empty_label.setText("None — closed hollow")
        self._radio_thicken.setText("Inside")
        self._radio_cut.setText("Outside")
        self._spinbox.set_mm(2.0)

    def is_outside(self) -> bool:
        return self._mode_group.checkedId() == 1

    def _signed_value(self) -> float | None:
        val = self._spinbox.mm_value()
        return None if val is None else abs(val)

    def _on_ok(self):
        # No open faces is fine: a closed hollow.
        val = self._signed_value()
        if val:
            self.thicken_requested.emit(val)

    def set_shell(self, thickness_mm: float, outside: bool):
        (self._radio_cut if outside else self._radio_thicken).setChecked(True)
        self._spinbox.set_mm(abs(thickness_mm))
