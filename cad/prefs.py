"""
cad/prefs.py

User preferences — colors, display settings, keybindings.

Stored as JSON in the platform config directory:
  Linux/Mac : ~/.config/cadapp/prefs.json
  Windows   : %APPDATA%/cadapp/prefs.json

Usage
-----
    from cad.prefs import prefs          # the singleton
    prefs.body_color                     # read
    prefs.body_color = (0.5, 0.6, 0.7)  # write (in-memory only)
    prefs.save()                         # persist to disk

All color values are (R, G, B) tuples with components in [0.0, 1.0].
Keybind values are key strings like "L", "Ctrl+Z", "Return", "Escape".
"""

from __future__ import annotations
import json
import os
import re
from dataclasses import dataclass, field, asdict
from typing import Tuple


Color3 = Tuple[float, float, float]


def _p(r, g, b) -> Color3:
    return (float(r), float(g), float(b))


# ---------------------------------------------------------------------------
# Keybind metadata — consumed by prefs_dialog.py to build the UI
# ---------------------------------------------------------------------------

# Default key string for each action
KEYBIND_DEFAULTS: dict[str, str] = {
    # 3D mode
    "extrude":           "E",
    "thicken":           "T",
    "fillet":            "F",
    "revolve":           "R",
    "undo":              "Ctrl+Z",
    "redo":              "Ctrl+Y",
    "projection_toggle": "",
    # Sketch mode — tools
    "sketch_line":       "L",
    "sketch_arc":        "A",
    "sketch_circle":     "C",
    "sketch_trim":       "T",
    "sketch_divide":     "D",
    "sketch_point":      "P",
    "sketch_offset":     "O",
    "sketch_fillet":     "F",
    "sketch_include":    "I",
    "sketch_dimension":  "",
    "sketch_geometric":  "",
    "sketch_square":     "S",
    "sketch_spline":     "B",
    "sketch_construction": "G",
    "sketch_mirror":     "M",
    "sketch_import_image": "",
    "sketch_commit":     "Return",
    # Sketch mode — snap declarations (take priority over tool switches when tool active)
    "snap_endpoint":     "E",
    "snap_midpoint":     "M",
    "snap_center":       "C",
    "snap_nearest":      "N",
    "snap_tangent":      "T",
    "snap_intersection": "I",
    # Sketch mode — constrained line shortcuts
    "sketch_hline":      "H",
    "sketch_vline":      "V",
    # Sketch mode — navigation
    "sketch_projection_toggle": "",
}

# Actions that are only active in 3D mode (no sketch open)
KEYBINDS_3D: frozenset[str] = frozenset({
    "extrude", "thicken", "fillet", "revolve", "undo", "redo", "projection_toggle",
})

# Actions that are only active in sketch mode
KEYBINDS_SKETCH: frozenset[str] = frozenset({
    "sketch_line", "sketch_arc", "sketch_circle", "sketch_trim",
    "sketch_divide", "sketch_point", "sketch_offset", "sketch_fillet",
    "sketch_include", "sketch_dimension", "sketch_geometric", "sketch_square",
    "sketch_spline", "sketch_construction", "sketch_mirror",
    "sketch_import_image",
    "sketch_commit", "snap_endpoint", "snap_midpoint", "snap_center",
    "snap_nearest", "snap_tangent", "snap_intersection",
    "sketch_hline", "sketch_vline", "sketch_projection_toggle",
})

# Human-readable labels: action → (section, label)
KEYBIND_LABELS: dict[str, tuple[str, str]] = {
    "extrude":                  ("3D Mode",    "Extrude selected face"),
    "thicken":                  ("3D Mode",    "Thicken active body"),
    "fillet":                   ("3D Mode",    "Fillet selected edges"),
    "revolve":                  ("3D Mode",    "Revolve a sketch or face"),
    "undo":                     ("3D Mode",    "Undo"),
    "redo":                     ("3D Mode",    "Redo"),
    "projection_toggle":        ("3D Mode",    "Toggle ortho/perspective"),
    "sketch_line":              ("Sketch",     "Line tool"),
    "sketch_arc":               ("Sketch",     "Arc tool (3-point)"),
    "sketch_circle":            ("Sketch",     "Circle tool"),
    "sketch_trim":              ("Sketch",     "Trim tool"),
    "sketch_divide":            ("Sketch",     "Divide tool"),
    "sketch_point":             ("Sketch",     "Point tool"),
    "sketch_offset":            ("Sketch",     "Offset tool"),
    "sketch_fillet":            ("Sketch",     "Fillet tool"),
    "sketch_include":           ("Sketch",     "Include geometry"),
    "sketch_dimension":         ("Sketch",     "Dimension tool (length constraint)"),
    "sketch_geometric":         ("Sketch",     "Geometric constraint tool"),
    "sketch_square":            ("Sketch",     "Square tool (2-point)"),
    "sketch_spline":            ("Sketch",     "Spline tool (fit through points)"),
    "sketch_construction":      ("Sketch",     "Toggle selected geometry construction/normal"),
    "sketch_import_image":      ("Sketch",     "Trace an image into sketch geometry"),
    "sketch_mirror":            ("Sketch",     "Mirror selected geometry across a clicked line"),
    "snap_endpoint":            ("Sketch",     "Force snap: endpoint"),
    "snap_midpoint":            ("Sketch",     "Force snap: midpoint"),
    "snap_center":              ("Sketch",     "Force snap: center"),
    "snap_nearest":             ("Sketch",     "Force snap: nearest"),
    "snap_tangent":             ("Sketch",     "Force snap: tangent to arc/circle"),
    "snap_intersection":        ("Sketch",     "Force snap: intersection"),
    "sketch_hline":             ("Sketch",     "Horizontal line"),
    "sketch_vline":             ("Sketch",     "Vertical line"),
    "sketch_commit":            ("Sketch",     "Commit sketch"),
    "sketch_projection_toggle": ("Sketch",     "Toggle ortho/perspective"),
}


# ---------------------------------------------------------------------------
# Prefs dataclass
# ---------------------------------------------------------------------------

@dataclass
class Prefs:
    # ------------------------------------------------------------------
    # Viewport background
    # ------------------------------------------------------------------
    background_color:       Color3 = field(default_factory=lambda: _p(0.15, 0.15, 0.15))

    # ------------------------------------------------------------------
    # Body colors
    # ------------------------------------------------------------------
    body_color:             Color3 = field(default_factory=lambda: _p(0.55, 0.65, 0.75))
    body_color_active:      Color3 = field(default_factory=lambda: _p(0.65, 0.75, 0.85))

    # ------------------------------------------------------------------
    # Edge wireframe
    # ------------------------------------------------------------------
    edge_color:             Color3 = field(default_factory=lambda: _p(0.20, 0.20, 0.22))
    edge_width:             float  = 1.2
    show_edges:             bool   = True

    # ------------------------------------------------------------------
    # Selection highlights
    # ------------------------------------------------------------------
    face_selected_color:    Color3 = field(default_factory=lambda: _p(0.95, 0.60, 0.10))
    edge_hovered_color:     Color3 = field(default_factory=lambda: _p(0.90, 0.90, 0.30))
    edge_selected_color:    Color3 = field(default_factory=lambda: _p(1.00, 1.00, 0.40))
    vertex_hovered_color:   Color3 = field(default_factory=lambda: _p(1.00, 1.00, 1.00))
    vertex_selected_color:  Color3 = field(default_factory=lambda: _p(1.00, 0.85, 0.20))

    # ------------------------------------------------------------------
    # Sketch overlay
    # ------------------------------------------------------------------
    sketch_line_color:      Color3 = field(default_factory=lambda: _p(0.05, 0.05, 0.05))
    sketch_reference_color: Color3 = field(default_factory=lambda: _p(0.45, 0.55, 0.75))
    sketch_preview_color:   Color3 = field(default_factory=lambda: _p(0.95, 0.75, 0.10))
    sketch_cursor_color:    Color3 = field(default_factory=lambda: _p(0.10, 0.10, 0.10))
    sketch_axis_x_color:    Color3 = field(default_factory=lambda: _p(0.85, 0.20, 0.20))
    sketch_axis_y_color:    Color3 = field(default_factory=lambda: _p(0.20, 0.75, 0.30))
    sketch_grid_major_color:Color3 = field(default_factory=lambda: _p(0.35, 0.45, 0.60))
    sketch_grid_minor_color:Color3 = field(default_factory=lambda: _p(0.22, 0.30, 0.42))
    sketch_line_width:      float  = 2.5
    sketch_reference_width: float  = 1.8

    # ------------------------------------------------------------------
    # 3D operation preview
    # ------------------------------------------------------------------
    op_preview_color:       Color3 = field(default_factory=lambda: _p(0.55, 0.25, 0.85))
    op_preview_opacity:     float  = 0.45

    # ------------------------------------------------------------------
    # Visual fidelity  — tunable quality knobs (live-adjustable via the
    # Fidelity dialog / F8 HUD). These affect COMMITTED display geometry only;
    # operation previews stay at their own coarse tolerances for speed.
    # ------------------------------------------------------------------
    # Multisample antialiasing sample count on the GL context. 0 disables MSAA.
    # Applied at startup (context creation) — changing it needs a restart.
    msaa_samples:           int   = 4
    # Enable GL_LINE_SMOOTH for crisper wireframe/silhouette edges.
    line_smoothing:         bool  = True
    # Committed body mesh: max angle (radians) between adjacent facet normals on
    # curved surfaces. Lower = smoother shading (fewer visible facets), more
    # triangles. 0.3 ≈ 17°, 0.15 ≈ 8.6°.
    mesh_angular_tol:       float = 0.15
    # Chordal deviation as a fraction of the part's bbox diagonal, with an
    # absolute floor in mm. Lower = curves hug the true surface more tightly.
    mesh_deviation_scale:   float = 0.0005
    mesh_deviation_floor:   float = 0.02
    # Sketch curve display tessellation — segments per arc/spline for rendering.
    # Higher = smoother big circles/arcs on screen.
    sketch_curve_segments:  int   = 96

    # ------------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------------
    camera_rotate_speed:    float  = 1.0
    camera_pan_speed:       float  = 1.0
    camera_mode:            str    = 'trackball'
    camera_invert_yaw:      bool   = False
    camera_invert_pitch:    bool   = False

    # ------------------------------------------------------------------
    # Units
    # ------------------------------------------------------------------
    default_unit:     str = "mm"
    display_decimals: int = 3

    # ------------------------------------------------------------------
    # UI scale offset  — integer steps added to every base font size
    # 0 = default; positive = larger; negative = smaller
    # ------------------------------------------------------------------
    ui_scale_offset: int = 0

    # ------------------------------------------------------------------
    # Keybindings  — action_name → key string
    # ------------------------------------------------------------------
    keybinds: dict = field(
        default_factory=lambda: dict(KEYBIND_DEFAULTS))

    # ------------------------------------------------------------------
    # Keybind helpers
    # ------------------------------------------------------------------

    def key(self, action: str) -> str:
        """Return the current key string for an action, falling back to default."""
        return self.keybinds.get(action, KEYBIND_DEFAULTS.get(action, ""))

    def scaled_px(self, base: int) -> int:
        return max(1, base + self.ui_scale_offset)

    def scale_stylesheet(self, css: str) -> str:
        """Rewrite all font-size: Npx values by applying ui_scale_offset."""
        if self.ui_scale_offset == 0:
            return css
        def _sub(m):
            return f"font-size: {self.scaled_px(int(m.group(1)))}px"
        return re.sub(r'font-size:\s*(\d+)px', _sub, css)

    def matches(self, action: str, qt_key_event) -> bool:
        """
        Return True if a QKeyEvent matches the bound key for action.

        Handles plain keys ("L", "E", "Return") and modified keys ("Ctrl+Z").
        """
        from PyQt6.QtCore import Qt
        from PyQt6.QtGui import QKeySequence

        bound = self.key(action)
        if not bound:
            return False

        seq     = QKeySequence(bound)
        if seq.isEmpty():
            return False

        # Reconstruct a QKeySequence from the event for comparison
        mods = qt_key_event.modifiers()
        k    = qt_key_event.key()

        # Strip modifier-only keypresses
        if k in (Qt.Key.Key_Control, Qt.Key.Key_Shift,
                 Qt.Key.Key_Alt, Qt.Key.Key_Meta):
            return False

        event_seq = QKeySequence(int(mods.value) | k)
        return seq == event_seq

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @staticmethod
    def _config_path() -> str:
        if os.name == 'nt':
            base = os.environ.get('APPDATA', os.path.expanduser('~'))
        else:
            base = os.environ.get('XDG_CONFIG_HOME',
                                  os.path.join(os.path.expanduser('~'), '.config'))
        return os.path.join(base, 'cadapp', 'prefs.json')

    def save(self):
        path = self._config_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        d = asdict(self)
        with open(path, 'w') as f:
            json.dump(d, f, indent=2)

    def load(self):
        path = self._config_path()
        if not os.path.exists(path):
            return
        try:
            with open(path) as f:
                data = json.load(f)
            for key, val in data.items():
                if not hasattr(self, key):
                    continue
                if isinstance(val, list):
                    val = tuple(val)
                # Merge keybinds: start from defaults, then apply saved values.
                # Explicitly saved "" means intentionally unbound — preserve it.
                if key == 'keybinds' and isinstance(val, dict):
                    merged = dict(KEYBIND_DEFAULTS)
                    merged.update(val)   # saved "" overwrites default correctly
                    val = merged
                setattr(self, key, val)
        except Exception as e:
            print(f"[Prefs] Could not load preferences: {e}")

    def reset(self):
        """Reset all preferences to defaults."""
        defaults = Prefs()
        for key in asdict(self):
            setattr(self, key, getattr(defaults, key))

    def reset_keybinds(self):
        """Reset only keybindings to defaults."""
        self.keybinds = dict(KEYBIND_DEFAULTS)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

prefs = Prefs()
prefs.load()
