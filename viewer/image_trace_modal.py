"""
viewer/image_trace_modal.py

ImageTraceModal — the "load a photo, tweak dials, preview the lines, accept"
dialog (the OMAX-layout workflow).

Structure
---------
Left:  a preview pane showing the loaded image with the current traced
       polylines drawn over it (updates live as dials move).
Right: the dials — threshold / blur / invert / Canny toggle / edge simplify
       (epsilon) / min-area, plus a units-aware scale field (real-world image
       width) built on the app's ExprSpinBox so it respects prefs.default_unit.

The pipeline itself lives in cad/image_trace.py (Qt-free, unit-tested).  This
module only: loads a file into a numpy array, calls trace(), renders the
overlay, and returns the accepted polylines (UV mm) to ImageImportTool.

run() -> list[list[np.ndarray]] | None   (None when cancelled)
"""

from __future__ import annotations
import numpy as np
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QSlider,
    QCheckBox, QPushButton, QFrame, QFileDialog, QWidget, QComboBox,
    QScrollArea,
)
from PyQt6.QtCore import Qt, QRect
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor

from cad.image_trace import TraceParams


class _PreviewPane(QLabel):
    """Draws the source image scaled-to-fit with traced polylines overlaid.

    Polylines arrive in mm (UV, V-up); we map them back to image pixels for
    display using the same scale_mm the trace used, then into widget space.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(420, 420)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background: #1a1a1a; border: 1px solid #444;")
        self._qimg: QImage | None = None
        self._segments: list = []            # fitted LineSeg/ArcSeg/PolySeg (px)
        self._img_wh: tuple[int, int] = (1, 1)
        # View transform: zoom factor + pan (in image px of the top-left corner
        # shown).  1.0 zoom = fit-to-widget; pan (0,0) = centred.
        self._zoom: float = 1.0
        self._pan = np.array([0.0, 0.0])     # image-px offset added to view
        self._drag_last = None
        self.setMouseTracking(True)

    def set_image(self, qimg: QImage, wh, reset_view=True):
        self._qimg = qimg
        self._img_wh = wh
        if reset_view:
            self._zoom = 1.0
            self._pan = np.array([0.0, 0.0])
        self.update()

    def set_segments(self, segments):
        """Fitted primitives in PIXEL space (image coords, y-down)."""
        self._segments = segments
        self.update()

    # --- view transform ------------------------------------------------

    def _base_scale(self):
        """Fit-to-widget scale (before zoom)."""
        iw, ih = self._img_wh
        a = self.rect()
        return min(a.width() / max(iw, 1), a.height() / max(ih, 1))

    def _transform(self):
        """Return (scale, ox, oy) mapping image px → widget px:
        widget = image*scale + (ox, oy)."""
        iw, ih = self._img_wh
        s = self._base_scale() * self._zoom
        a = self.rect()
        ox = a.x() + (a.width() - iw * s) / 2 - self._pan[0] * s
        oy = a.y() + (a.height() - ih * s) / 2 - self._pan[1] * s
        return s, ox, oy

    def wheelEvent(self, e):
        if self._qimg is None:
            return
        # Zoom anchored at the cursor so the point under it stays put.
        s, ox, oy = self._transform()
        cx, cy = e.position().x(), e.position().y()
        img_x = (cx - ox) / s
        img_y = (cy - oy) / s
        factor = 1.0015 ** e.angleDelta().y()
        self._zoom = float(np.clip(self._zoom * factor, 1.0, 40.0))
        # Re-anchor: solve pan so (img_x,img_y) maps back under the cursor.
        s2 = self._base_scale() * self._zoom
        iw, ih = self._img_wh
        a = self.rect()
        self._pan[0] = (a.x() + (a.width() - iw * s2) / 2 - (cx - img_x * s2)) / s2
        self._pan[1] = (a.y() + (a.height() - ih * s2) / 2 - (cy - img_y * s2)) / s2
        self.update()

    def mousePressEvent(self, e):
        self._drag_last = np.array([e.position().x(), e.position().y()])

    def mouseMoveEvent(self, e):
        if self._drag_last is None:
            return
        cur = np.array([e.position().x(), e.position().y()])
        s = self._base_scale() * self._zoom
        self._pan -= (cur - self._drag_last) / s
        self._drag_last = cur
        self.update()

    def mouseReleaseEvent(self, e):
        self._drag_last = None

    def reset_view(self):
        self._zoom = 1.0
        self._pan = np.array([0.0, 0.0])
        self.update()

    def paintEvent(self, ev):
        super().paintEvent(ev)
        if self._qimg is None:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)

        s, ox, oy = self._transform()
        iw, ih = self._img_wh
        # Draw the (processed) backdrop image under the transform.
        target = QRect(int(ox), int(oy), int(iw * s), int(ih * s))
        p.drawImage(target, self._qimg)

        # Overlay fitted primitives in image px mapped to widget px.
        from cad.trace.fit import LineSeg, ArcSeg, PolySeg

        def W(pt):
            return (ox + float(pt[0]) * s, oy + float(pt[1]) * s)

        line_pen = QPen(QColor(80, 200, 255), 1.6); line_pen.setCosmetic(True)
        arc_pen = QPen(QColor(120, 230, 130), 1.8); arc_pen.setCosmetic(True)
        poly_pen = QPen(QColor(200, 150, 90), 1.2); poly_pen.setCosmetic(True)

        for seg in self._segments:
            if isinstance(seg, LineSeg):
                p.setPen(line_pen)
                x0, y0 = W(seg.p0); x1, y1 = W(seg.p1)
                p.drawLine(int(x0), int(y0), int(x1), int(y1))
            elif isinstance(seg, ArcSeg):
                p.setPen(arc_pen)
                self._draw_arc(p, seg, W)
            elif isinstance(seg, PolySeg):
                p.setPen(poly_pen)
                pts = seg.points
                for a, b in zip(pts[:-1], pts[1:]):
                    x0, y0 = W(a); x1, y1 = W(b)
                    p.drawLine(int(x0), int(y0), int(x1), int(y1))
        p.end()

    def _draw_arc(self, p, seg, W):
        """Tessellate an ArcSeg (image-space) into a polyline and draw it."""
        c = seg.center
        a0 = np.arctan2(seg.p0[1] - c[1], seg.p0[0] - c[0])
        a1 = np.arctan2(seg.p1[1] - c[1], seg.p1[0] - c[0])
        # Sweep in the fitter's direction (image ccw flag).
        if seg.ccw:
            while a1 <= a0:
                a1 += 2 * np.pi
        else:
            while a1 >= a0:
                a1 -= 2 * np.pi
        angs = np.linspace(a0, a1, 24)
        prev = None
        for a in angs:
            pt = (c[0] + seg.radius * np.cos(a), c[1] + seg.radius * np.sin(a))
            wx, wy = W(pt)
            if prev is not None:
                p.drawLine(int(prev[0]), int(prev[1]), int(wx), int(wy))
            prev = (wx, wy)


class ImageTraceModal(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Trace Image → Sketch")
        self.setModal(True)
        self._image: np.ndarray | None = None      # RGB uint8 (Pillow order)
        self._segments: list = []                   # fitted primitives (px)
        self._result: list | None = None            # accepted entities
        self._params = TraceParams()

        self.resize(1000, 720)
        root = QHBoxLayout(self)
        self._preview = _PreviewPane()
        root.addWidget(self._preview, stretch=1)

        # Controls live in a scroll area — there are many live dials and they
        # must not push the dialog taller than the screen.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(320)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        panel_host = QWidget()
        scroll.setWidget(panel_host)
        panel = QVBoxLayout(panel_host)
        root.addWidget(scroll)

        row0 = QHBoxLayout()
        load_btn = QPushButton("Load Image…")
        load_btn.clicked.connect(self._on_load)
        row0.addWidget(load_btn)
        view_btn = QPushButton("Reset view")
        view_btn.clicked.connect(lambda: self._preview.reset_view())
        row0.addWidget(view_btn)
        panel.addLayout(row0)
        hint = QLabel("Scroll to zoom · drag to pan")
        hint.setStyleSheet("color:#777; font-size:10px;")
        panel.addWidget(hint)

        from cad.trace.fit import FitParams
        fp = FitParams()
        grid = QGridLayout()
        panel.addLayout(grid)
        r = 0

        # Every fit knob is a live slider — there is no universal pixel tuning
        # (a 4K screenshot, a phone photo and a downloaded JPEG differ hugely),
        # so the user dials to their image with live visual feedback.  Each
        # slider carries a `scale` so integer positions map to real values.
        self._sliders: dict = {}

        def group(title):
            nonlocal r
            lbl = QLabel(title)
            lbl.setStyleSheet("color:#6cf; font-weight:bold; margin-top:4px;")
            grid.addWidget(lbl, r, 0, 1, 3); r += 1

        def slider(key, label, lo, hi, val, scale=1.0, suffix=""):
            nonlocal r
            self._add_fslider(grid, r, key, label, lo, hi, val, scale, suffix)
            r += 1

        # --- Colour / tone / noise preprocessing (the OMAX "colour levers") ---
        # Applied BEFORE thresholding; the preview backdrop shows the resulting
        # processed gray so these are tuned by eye.
        from cad.trace.mask import CHANNEL_MODES, PreprocParams
        pp = PreprocParams()
        group("Colour / tone")
        grid.addWidget(QLabel("Channel"), r, 0)
        self._channel = QComboBox()
        self._channel.addItems(CHANNEL_MODES)
        self._channel.setCurrentText(pp.channel)
        self._channel.currentTextChanged.connect(self._retrace)
        grid.addWidget(self._channel, r, 1, 1, 2); r += 1
        slider("black_pt", "Black point", 0, 255, pp.black_point)
        slider("white_pt", "White point", 0, 255, pp.white_point)
        slider("gamma",    "Gamma",       20, 400, int(pp.gamma * 100),
               scale=0.01)
        slider("median",   "Denoise",     0, 8, pp.median, suffix="px")

        group("Binarize")
        slider("threshold", "Threshold", 0, 255, self._params.threshold)
        slider("blur",      "Blur",      0, 12,  self._params.blur, suffix="px")
        slider("min_area",  "Min area",  0, 2000, int(self._params.min_area_px),
               suffix="px²")

        group("Noise / resample")
        slider("resample", "Resample", 5, 50, int(fp.resample_step * 10),
               scale=0.1, suffix="px")
        slider("smooth",   "Smoothing", 1, 15, fp.smooth_window, suffix="pt")

        group("Corners")
        slider("corner", "Corner angle", 5, 90,
               int(round(np.rad2deg(fp.corner_threshold))), suffix="°")

        group("Line vs arc")
        slider("tol",       "Fit tolerance", 2, 60,
               int(fp.fit_tolerance * 10), scale=0.1, suffix="px")
        slider("arc_margin", "Arc must beat line", 0, 30,
               int(fp.arc_line_margin * 10), scale=0.1, suffix="px")
        slider("rc_ratio",  "Max radius/chord", 20, 300,
               int(fp.max_radius_chord_ratio * 10), scale=0.1)
        slider("min_sweep", "Min arc angle", 5, 90,
               int(round(np.rad2deg(fp.min_arc_sweep))), suffix="°")

        group("Arc merge")
        slider("merge_center", "Merge centre", 0, 200,
               int(fp.merge_center_tol * 10), scale=0.1, suffix="px")
        slider("merge_radius", "Merge radius", 0, 200,
               int(fp.merge_radius_tol * 10), scale=0.1, suffix="px")

        # --- invert toggle ---
        self._invert = QCheckBox("Invert")
        self._invert.toggled.connect(self._retrace)
        grid.addWidget(self._invert, r, 0, 1, 3); r += 1

        # --- scale (units-aware) ---
        grid.addWidget(QLabel("Image width"), r, 0)
        from gui.expr_spinbox import ExprSpinBox
        self._scale = ExprSpinBox()
        self._scale.set_mm(self._params.scale_mm)
        self._scale.value_changed.connect(lambda _mm: self._retrace())
        grid.addWidget(self._scale, r, 1, 1, 2); r += 1

        self._count_lbl = QLabel("—")
        self._count_lbl.setStyleSheet("color: #888;")
        panel.addWidget(self._count_lbl)

        reset = QPushButton("Reset dials")
        reset.clicked.connect(self._reset_dials)
        panel.addWidget(reset)

        panel.addStretch(1)

        btn_row = QHBoxLayout()
        cancel = QPushButton("Cancel"); cancel.clicked.connect(self.reject)
        accept = QPushButton("Add to Sketch"); accept.clicked.connect(self._on_accept)
        accept.setDefault(True)
        btn_row.addWidget(cancel); btn_row.addWidget(accept)
        panel.addLayout(btn_row)

    # ------------------------------------------------------------------
    # Widget helpers
    # ------------------------------------------------------------------

    def _add_fslider(self, grid, row, key, label, lo, hi, val, scale, suffix):
        """A labelled slider with a live value readout, registered in
        self._sliders[key] as (slider, scale, suffix)."""
        grid.addWidget(QLabel(label), row, 0)
        s = QSlider(Qt.Orientation.Horizontal)
        s.setRange(lo, hi)
        s.setValue(int(val))
        val_lbl = QLabel()
        val_lbl.setStyleSheet("color:#aaa; font-family:monospace;")
        val_lbl.setMinimumWidth(52)

        def update_readout(v):
            real = v * scale
            txt = f"{real:.1f}" if scale != 1.0 else f"{int(real)}"
            val_lbl.setText(f"{txt}{suffix}")
        update_readout(s.value())
        s.valueChanged.connect(update_readout)
        s.valueChanged.connect(self._retrace)

        grid.addWidget(s, row, 1)
        grid.addWidget(val_lbl, row, 2)
        self._sliders[key] = (s, scale, val_lbl, suffix)

    def _sval(self, key: str) -> float:
        s, scale, _lbl, _sfx = self._sliders[key]
        return s.value() * scale

    def _reset_dials(self):
        """Restore all fit + preprocessing sliders to their defaults."""
        from cad.trace.fit import FitParams
        from cad.trace.mask import PreprocParams
        fp = FitParams()
        pp = PreprocParams()
        defaults = {
            # preprocessing
            "black_pt": pp.black_point, "white_pt": pp.white_point,
            "gamma": pp.gamma / 0.01, "median": pp.median,
            # fit
            "resample": fp.resample_step / 0.1, "smooth": fp.smooth_window,
            "corner": np.rad2deg(fp.corner_threshold),
            "tol": fp.fit_tolerance / 0.1,
            "arc_margin": fp.arc_line_margin / 0.1,
            "rc_ratio": fp.max_radius_chord_ratio / 0.1,
            "min_sweep": np.rad2deg(fp.min_arc_sweep),
            "merge_center": fp.merge_center_tol / 0.1,
            "merge_radius": fp.merge_radius_tol / 0.1,
        }
        for k, v in defaults.items():
            if k in self._sliders:
                self._sliders[k][0].setValue(int(round(v)))
        self._channel.setCurrentText(pp.channel)
        self._retrace()

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def _read_params(self) -> TraceParams:
        scale = self._scale.mm_value()
        return TraceParams(
            threshold   = int(self._sval("threshold")),
            blur        = int(self._sval("blur")),
            min_area_px = float(self._sval("min_area")),
            invert      = self._invert.isChecked(),
            use_canny   = False,
            scale_mm    = scale if scale is not None else 100.0,
        )

    def _read_fit_params(self):
        from cad.trace.fit import FitParams
        return FitParams(
            corner_threshold = np.deg2rad(self._sval("corner")),
            fit_tolerance    = self._sval("tol"),
            min_arc_sweep    = np.deg2rad(self._sval("min_sweep")),
            resample_step    = self._sval("resample"),
            smooth_window    = int(self._sval("smooth")),
            max_radius_chord_ratio = self._sval("rc_ratio"),
            arc_line_margin  = self._sval("arc_margin"),
            merge_center_tol = self._sval("merge_center"),
            merge_radius_tol = self._sval("merge_radius"),
        )

    def _read_preproc(self):
        from cad.trace.mask import PreprocParams, CHANNEL_MODES
        return PreprocParams(
            channel     = self._channel.currentText(),
            black_point = int(self._sval("black_pt")),
            white_point = int(self._sval("white_pt")),
            gamma       = self._sval("gamma"),
            blur        = int(self._sval("blur")),
            median      = int(self._sval("median")),
        )

    @staticmethod
    def _gray_to_qimage(gray: np.ndarray) -> QImage:
        """uint8 HxW gray → grayscale QImage (copied; buffer must outlive it)."""
        h, w = gray.shape
        buf = np.ascontiguousarray(gray)
        return QImage(buf.data, w, h, w,
                      QImage.Format.Format_Grayscale8).copy()

    def _retrace(self):
        if self._image is None:
            return
        params = self._read_params()
        fit_params = self._read_fit_params()
        preproc = self._read_preproc()
        try:
            from cad.image_trace import trace_segments_with_gray
            segs, gray = trace_segments_with_gray(
                self._image, params, fit_params, preproc)
        except Exception as ex:                # bad params shouldn't crash the modal
            self._count_lbl.setText(f"trace error: {ex}")
            return
        self._segments = segs
        # Backdrop shows the PROCESSED gray — exactly what the tracer saw — so
        # colour/tone/noise levers are tunable by eye.  Preserve zoom/pan.
        h, w = gray.shape
        self._preview.set_image(self._gray_to_qimage(gray), (w, h),
                                reset_view=False)
        self._preview.set_segments(segs)
        from cad.trace.fit import LineSeg, ArcSeg, PolySeg
        nl = sum(isinstance(x, LineSeg) for x in segs)
        na = sum(isinstance(x, ArcSeg) for x in segs)
        npl = sum(isinstance(x, PolySeg) for x in segs)
        extra = f" · {npl} polyline" if npl else ""
        self._count_lbl.setText(f"{nl} lines · {na} arcs{extra}")

    def _on_load(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)")
        if not path:
            return
        from PIL import Image
        try:
            with Image.open(path) as im:
                rgb = np.asarray(im.convert("RGB"))
        except Exception as ex:
            self._count_lbl.setText(f"could not read image: {ex}")
            return
        self._image = rgb                        # RGB uint8 — tracer input
        self._preview.reset_view()               # fresh image → fit-to-view
        # _retrace sets the backdrop to the PROCESSED gray (what the tracer sees).
        self._retrace()

    def _on_accept(self):
        if not self._segments or self._image is None:
            self.reject()
            return
        # Convert fitted pixel-space segments → sketch entities (UV mm) now, at
        # accept time, using the image dimensions for the px→mm/y-flip mapping.
        from cad.image_trace import segments_to_entities
        h, w = self._image.shape[:2]
        params = self._read_params()
        self._result = segments_to_entities(self._segments, w, h, params)
        if not self._result:
            self.reject()
            return
        self.accept()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(self):
        """Show modally.  Returns fitted sketch entities (LineEntity/ArcEntity,
        UV mm) or None if cancelled / nothing traced."""
        if self.exec() == QDialog.DialogCode.Accepted:
            return self._result
        return None
