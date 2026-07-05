"""
gui/fidelity_dialog.py

FidelityDialog — a non-modal live-tuning panel for visual quality knobs.

Each knob maps to a pref (see cad.prefs, the "Visual fidelity" block). Dragging
a slider updates the pref and immediately re-tessellates committed body meshes
and repaints, so you can watch the FPS / frame-time readout react and find the
sweet spot between crispness and cost.

The FPS readout is driven by the existing cad.profiler: opening this dialog
enables the profiler (and the viewport's continuous-repaint timer) so frame
timing is live; closing it leaves the profiler as it was found.

MSAA sample count is applied at GL-context creation, so changing it here only
takes effect on the next launch — the row is labelled accordingly.
"""

from __future__ import annotations
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QSlider,
    QCheckBox, QComboBox, QPushButton, QFrame, QSizePolicy,
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont
from cad.prefs import prefs
from cad.profiler import profiler


class _Knob:
    """A float pref exposed as an integer slider with a value label.

    The slider works in integer steps; `to_val`/`from_val` convert between the
    slider position and the real pref value so we can express sub-unit ranges.
    """
    def __init__(self, attr, lo, hi, steps, fmt, to_val, from_val):
        self.attr = attr
        self.lo, self.hi, self.steps = lo, hi, steps
        self.fmt = fmt
        self.to_val = to_val        # slider int -> pref value
        self.from_val = from_val    # pref value -> slider int


class FidelityDialog(QDialog):
    def __init__(self, viewport, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Visual Fidelity")
        self.setModal(False)
        self._vp = viewport

        # Enable the profiler so the FPS readout is live; remember prior state
        # so we can restore it on close.
        self._profiler_was_enabled = profiler.enabled
        if not profiler.enabled:
            profiler.toggle()          # also resets stats
        # Drive continuous repaints so frame timing keeps flowing even when the
        # scene is otherwise idle.
        if hasattr(viewport, "_toggle_profiler"):
            # viewport owns a repaint timer keyed to profiler.enabled; make sure
            # it's running without flipping the profiler back off.
            self._ensure_repaint_timer()

        root = QVBoxLayout(self)

        # ---- FPS / frame-time readout -----------------------------------
        self._readout = QLabel("FPS —   frame — ms")
        f = QFont("monospace")
        f.setPointSize(11)
        self._readout.setFont(f)
        self._readout.setStyleSheet("color:#ffd050; padding:4px 2px;")
        root.addWidget(self._readout)

        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        root.addWidget(line)

        grid = QGridLayout()
        grid.setColumnStretch(1, 1)
        root.addLayout(grid)
        self._rows = []   # (knob, slider, value_label)
        r = 0

        # ---- Curved-surface smoothness (angular tolerance) --------------
        # Lower radians = smoother normals. Present as "smoothness" where the
        # slider maxes out at the finest tolerance.
        r = self._add_float_knob(
            grid, r, "Surface smoothness",
            _Knob("mesh_angular_tol", lo=0.03, hi=0.5, steps=47,
                  fmt=lambda v: f"{v:.3f} rad  (~{v*57.3:.0f}°/facet)",
                  to_val=lambda i: 0.5 - i * (0.5 - 0.03) / 47.0,
                  from_val=lambda v: round((0.5 - v) / ((0.5 - 0.03) / 47.0))))

        # ---- Chordal deviation scale ------------------------------------
        r = self._add_float_knob(
            grid, r, "Surface tightness",
            _Knob("mesh_deviation_scale", lo=0.0001, hi=0.004, steps=39,
                  fmt=lambda v: f"{v*100:.3f}% of size",
                  to_val=lambda i: 0.004 - i * (0.004 - 0.0001) / 39.0,
                  from_val=lambda v: round((0.004 - v) / ((0.004 - 0.0001) / 39.0))))

        # ---- Sketch curve segments --------------------------------------
        r = self._add_float_knob(
            grid, r, "Sketch curve detail",
            _Knob("sketch_curve_segments", lo=16, hi=256, steps=240,
                  fmt=lambda v: f"{int(round(v))} segments",
                  to_val=lambda i: 16 + i,
                  from_val=lambda v: int(round(v)) - 16))

        # ---- Line smoothing (bool) --------------------------------------
        self._line_smooth = QCheckBox("Line smoothing (crisp edges)")
        self._line_smooth.setChecked(bool(prefs.line_smoothing))
        self._line_smooth.toggled.connect(self._on_line_smooth)
        grid.addWidget(self._line_smooth, r, 0, 1, 3)
        r += 1

        # ---- MSAA (restart-scoped) --------------------------------------
        msaa_row = QHBoxLayout()
        msaa_row.addWidget(QLabel("MSAA (next launch):"))
        self._msaa = QComboBox()
        self._msaa_values = [0, 2, 4, 8, 16]
        for v in self._msaa_values:
            self._msaa.addItem("Off" if v == 0 else f"{v}×", v)
        cur = int(prefs.msaa_samples or 0)
        self._msaa.setCurrentIndex(self._msaa_values.index(cur)
                                   if cur in self._msaa_values else 2)
        self._msaa.currentIndexChanged.connect(self._on_msaa)
        msaa_row.addWidget(self._msaa)
        msaa_row.addStretch(1)
        wrap = QWidgetRow(msaa_row)
        grid.addWidget(wrap, r, 0, 1, 3)
        r += 1

        # ---- Buttons ----------------------------------------------------
        btns = QHBoxLayout()
        reset = QPushButton("Reset defaults")
        reset.clicked.connect(self._reset)
        close = QPushButton("Close")
        close.clicked.connect(self.close)
        btns.addWidget(reset)
        btns.addStretch(1)
        btns.addWidget(close)
        root.addLayout(btns)

        # Poll the profiler for the live readout.
        self._timer = QTimer(self)
        self._timer.setInterval(200)
        self._timer.timeout.connect(self._refresh_readout)
        self._timer.start()

        self.setMinimumWidth(360)

    # ------------------------------------------------------------------
    def _ensure_repaint_timer(self):
        """Start the viewport's profiler repaint timer without toggling the
        profiler off (its handler flips profiler.enabled)."""
        vp = self._vp
        t = getattr(vp, "_profiler_timer", None)
        if t is None:
            t = QTimer(vp)
            t.setInterval(0)
            t.timeout.connect(vp.update)
            vp._profiler_timer = t
        if not t.isActive():
            t.start()

    def _add_float_knob(self, grid, row, label, knob):
        grid.addWidget(QLabel(label), row, 0)
        s = QSlider(Qt.Orientation.Horizontal)
        s.setMinimum(0)
        s.setMaximum(knob.steps)
        cur = getattr(prefs, knob.attr)
        s.setValue(max(0, min(knob.steps, knob.from_val(cur))))
        s.valueChanged.connect(lambda _=None, k=knob: self._on_slider(k))
        grid.addWidget(s, row, 1)
        vlabel = QLabel(knob.fmt(cur))
        vlabel.setMinimumWidth(150)
        grid.addWidget(vlabel, row, 2)
        self._rows.append((knob, s, vlabel))
        return row + 1

    # ------------------------------------------------------------------
    def _on_slider(self, knob):
        for k, s, vlabel in self._rows:
            if k is knob:
                val = k.to_val(s.value())
                if k.attr == "sketch_curve_segments":
                    val = int(round(val))
                setattr(prefs, k.attr, val)
                vlabel.setText(k.fmt(val))
                break
        # Sketch-detail changes only need a repaint (hover.rebuild re-runs every
        # frame and re-tessellates sketch edges); mesh knobs need a rebuild.
        if knob.attr == "sketch_curve_segments":
            self._vp.update()
        else:
            self._rebuild_meshes()

    def _on_line_smooth(self, checked):
        prefs.line_smoothing = bool(checked)
        vp = self._vp
        vp.makeCurrent()
        from OpenGL.GL import (glEnable, glDisable, glHint, GL_LINE_SMOOTH,
                               GL_LINE_SMOOTH_HINT, GL_NICEST)
        if checked:
            glEnable(GL_LINE_SMOOTH)
            glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        else:
            glDisable(GL_LINE_SMOOTH)
        vp.doneCurrent()
        vp.update()

    def _on_msaa(self, _idx):
        prefs.msaa_samples = int(self._msaa.currentData())

    def _rebuild_meshes(self):
        # Re-tessellate committed bodies at the new tolerances. The tessellator
        # cache keys on deviation+angular_tolerance, so this returns fresh
        # geometry rather than stale cached triangles.
        self._vp.build_meshes()
        self._vp.update()

    def _refresh_readout(self):
        fps = profiler.fps
        ms = profiler.avg_frame_ms
        mx = profiler.max_frame_ms
        tris = 0
        for m in getattr(self._vp, "_meshes", {}).values():
            tris += getattr(m, "tri_count", 0) // 3
        self._readout.setText(
            f"FPS {fps:5.1f}   frame {ms:5.1f}ms (max {mx:5.1f})   "
            f"{tris:,} tris")

    def _reset(self):
        d = type(prefs)()
        for attr in ("mesh_angular_tol", "mesh_deviation_scale",
                     "mesh_deviation_floor", "sketch_curve_segments",
                     "line_smoothing", "msaa_samples"):
            setattr(prefs, attr, getattr(d, attr))
        # Refresh widgets from the restored values.
        for knob, s, vlabel in self._rows:
            cur = getattr(prefs, knob.attr)
            s.blockSignals(True)
            s.setValue(max(0, min(knob.steps, knob.from_val(cur))))
            s.blockSignals(False)
            vlabel.setText(knob.fmt(cur))
        self._line_smooth.setChecked(bool(prefs.line_smoothing))
        cur = int(prefs.msaa_samples or 0)
        self._msaa.setCurrentIndex(self._msaa_values.index(cur)
                                   if cur in self._msaa_values else 2)
        self._on_line_smooth(prefs.line_smoothing)
        self._rebuild_meshes()

    # ------------------------------------------------------------------
    def closeEvent(self, e):
        self._timer.stop()
        prefs.save()   # persist the tuned values
        # Restore profiler to how we found it.
        if not self._profiler_was_enabled and profiler.enabled:
            if hasattr(self._vp, "_toggle_profiler"):
                self._vp._toggle_profiler()   # flips off + stops repaint timer
            else:
                profiler.enabled = False
        self._vp.update()
        super().closeEvent(e)


# Small helper: wrap a layout in a QWidget so it can drop into a QGridLayout cell.
from PyQt6.QtWidgets import QWidget


class QWidgetRow(QWidget):
    def __init__(self, layout):
        super().__init__()
        self.setLayout(layout)
