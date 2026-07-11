"""
cad/image_trace.py

Headless raster-to-vector tracing pipeline.  Turns a photo/logo/scan into a
set of UV polylines (world millimetres) that the sketch layer can emit as
LineEntity chains — solver-visible, indistinguishable from drawn geometry.

No Qt, no OpenGL: pure numpy + Pillow + scipy so it can be unit-tested headless.
(OpenCV was dropped in Milestone A of Image Trace V2 — see cad/trace/ and
IMAGE_TRACE_V2_PLAN.md.)

Pipeline
--------
    grayscale -> optional blur -> threshold -> border-following (cad.trace)
        -> Douglas-Peucker -> scale px->mm

Everything downstream of trace() speaks raw millimetres; unit *display* is a
modal concern (see viewer/image_trace_modal.py), never handled here.  This
mirrors the app-wide rule in cad/units.py: stored values are always mm.

v1 emits polylines only.  v2 (Milestone B) will segment each contour and fit
LineEntity / ArcEntity primitives so the solver gets real center/radius handles.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class TraceParams:
    """User-tweakable dials for the trace.  All lengths that reach geometry are
    in millimetres; pixel-domain knobs (blur, epsilon_px) are in pixels."""
    # --- binarization ---
    threshold:   int   = 128     # 0..255; ignored when use_canny
    invert:      bool  = False   # swap fg/bg (dark-on-light vs light-on-dark)
    use_canny:   bool  = False   # edge detection instead of fill threshold
    canny_lo:    int   = 50
    canny_hi:    int   = 150
    blur:        int   = 0        # gaussian kernel radius in px (0 = off)
    # --- contour filtering / simplification ---
    min_area_px: float = 25.0     # drop contours smaller than this (px^2)
    epsilon_px:  float = 2.0      # Douglas-Peucker tolerance in px
    # --- scaling ---
    scale_mm:    float = 100.0    # real-world width of the *image* in mm
    # --- placement (UV offset of the image's top-left, in mm) ---
    origin_u:    float = 0.0
    origin_v:    float = 0.0
    # --- front-end selection (V3) ---
    #   "fill"   : boundary-follow (cad.trace.contours) — silhouettes/filled
    #   "stroke" : skeleton centerline (cad.trace.skeleton+walk) — line art
    trace_mode:  str   = "fill"
    prune_len:   int   = 8        # stroke mode: skeleton spur prune length (px)
    # Stroke mode must skeletonize the *stroke* (thin, minority region), not the
    # background it sits on.  When the thresholded foreground fills most of the
    # image it is almost certainly the background, so auto-flip which side we
    # skeletonize.  This makes stroke mode "just work" for dark-on-light art
    # regardless of the invert toggle.  Turn off to force the literal mask.
    auto_stroke_side: bool = True


def _to_gray(image: np.ndarray) -> np.ndarray:
    """Accept RGB, RGBA, or already-gray uint8 image -> single-channel uint8.

    NOTE: channel order is RGB (Pillow), unlike the old cv2 path's BGR.  Callers
    decode via cad.trace.mask.load_gray (Pillow) which yields RGB.
    """
    from cad.trace.mask import to_gray
    return to_gray(image)


def trace(image: np.ndarray, params: TraceParams) -> list[list[np.ndarray]]:
    """
    Trace *image* into a list of UV polylines.

    image  : HxW (gray) or HxWx3/4 (RGB/RGBA) uint8 numpy array.
    params : TraceParams.

    Returns a list of polylines; each polyline is a list of np.float64 (u, v)
    points in millimetres.  Closed contours repeat their first point at the end
    so downstream emission produces a welded LineEntity loop.
    """
    from cad.trace.mask import build_mask
    from cad.trace.contours import trace_contours
    from cad.trace.simplify import approx_poly, polygon_area

    gray = _to_gray(image)
    h, w = gray.shape[:2]

    # Canny (edge) mode was dropped in V2 Milestone A; fall back to the filled-
    # region threshold path so any stale param can't crash.  See the plan doc.
    mask = build_mask(gray, params.threshold, params.blur, params.invert)

    contours = trace_contours(mask, min_area=params.min_area_px)

    # px -> mm: image spans scale_mm across its *width*; preserve aspect.
    mm_per_px = params.scale_mm / float(w) if w else 1.0

    polylines: list[list[np.ndarray]] = []
    for c in contours:
        if polygon_area(c.points) < params.min_area_px:
            continue
        approx = approx_poly(c.points, params.epsilon_px, closed=True)
        if len(approx) < 2:
            continue
        poly: list[np.ndarray] = []
        for px, py in approx:
            # Image y grows downward; flip so sketch V grows upward.
            u = params.origin_u + float(px) * mm_per_px
            v = params.origin_v + float(h - py) * mm_per_px
            poly.append(np.array([u, v], dtype=np.float64))
        # Close the loop so endpoints weld (border contours are closed).
        if np.linalg.norm(poly[0] - poly[-1]) > 1e-9:
            poly.append(poly[0].copy())
        polylines.append(poly)

    return polylines


def polylines_to_line_entities(polylines: list[list[np.ndarray]],
                               weld_tol: float = 1e-3):
    """
    Convert traced polylines into a flat list of LineEntity segments.

    Consecutive points become segments; coincident endpoints (within
    _build_solver_system's tol) auto-weld in the solver, so a closed contour
    becomes a connected loop with no extra constraint bookkeeping.

    Degenerate segments (shorter than weld_tol) are dropped so we never emit a
    zero-length line the solver would choke on.
    """
    from cad.sketch import LineEntity
    ents: list = []
    for poly in polylines:
        for a, b in zip(poly[:-1], poly[1:]):
            if np.linalg.norm(b - a) < weld_tol:
                continue
            ents.append(LineEntity((float(a[0]), float(a[1])),
                                   (float(b[0]), float(b[1]))))
    return ents


# ---------------------------------------------------------------------------
# V2 (Milestone B): fitted line/arc emission
# ---------------------------------------------------------------------------

def trace_segments(image: np.ndarray, params: TraceParams,
                   fit_params=None, preproc=None) -> list:
    """
    Trace *image* into fitted primitives (LineSeg / ArcSeg / PolySeg) in PIXEL
    space.  This is the V2 path; convert to entities with segments_to_entities.

    preproc : optional PreprocParams — the colour/tone/noise front end.  When
              given it owns channel-collapse/levels/gamma/median/blur; only
              threshold+invert come from `params`.  When None, the legacy
              gray+blur path is used (default luminance).
    """
    segs, _gray = trace_segments_with_gray(image, params, fit_params, preproc)
    return segs


def trace_segments_with_gray(image: np.ndarray, params: TraceParams,
                             fit_params=None, preproc=None):
    """Like trace_segments but also returns the processed gray the mask was
    formed from, so the preview can show exactly what the tracer saw.

    The front end is selected by params.trace_mode:
      "fill"   → boundary follow (contours.py) — the V2 default.
      "stroke" → skeleton centerline (skeleton.py + walk.py) — V3, for line art.
    Both feed the SAME fit.py back end, so segments_to_entities is unchanged.

    Returns (segments, processed_gray_uint8)."""
    from cad.trace.mask import threshold_mask, preprocess_gray

    if preproc is not None:
        gray = preprocess_gray(image, preproc)
        mask = threshold_mask(gray, params.threshold, params.invert)
    else:
        gray = _to_gray(image)
        from cad.trace.mask import gaussian_blur
        gray = gaussian_blur(gray, params.blur)
        mask = threshold_mask(gray, params.threshold, params.invert)

    if params.trace_mode == "stroke":
        mask = _stroke_side(mask, params)
        return _fit_stroke(mask, params, fit_params), gray

    from cad.trace.contours import trace_contours
    from cad.trace.fit import fit_contours
    contours = trace_contours(mask, min_area=params.min_area_px)
    return fit_contours(contours, fit_params), gray


def _stroke_side(mask, params: TraceParams):
    """Pick the mask side to skeletonize in stroke mode.

    The stroke is the thin, minority region; if the current foreground fills
    most of the image it's the background, so flip.  Returns the (possibly
    inverted) mask.  A no-op when auto_stroke_side is off."""
    if not params.auto_stroke_side:
        return mask
    frac = float(mask.mean())            # fraction of pixels that are foreground
    return ~mask if frac > 0.5 else mask


def stroke_mask(image: np.ndarray, params: TraceParams, preproc=None):
    """The boolean mask stroke mode actually skeletonizes (after auto side-pick).

    Exposed so the modal preview can show exactly what the tracer saw, matching
    the fill-mode convention where the backdrop reflects the tracer's input."""
    from cad.trace.mask import threshold_mask, preprocess_gray
    if preproc is not None:
        gray = preprocess_gray(image, preproc)
    else:
        gray = _to_gray(image)
        from cad.trace.mask import gaussian_blur
        gray = gaussian_blur(gray, params.blur)
    mask = threshold_mask(gray, params.threshold, params.invert)
    return _stroke_side(mask, params)


def _fit_stroke(mask, params: TraceParams, fit_params=None) -> list:
    """V3 stroke front end: mask → skeleton → prune → walk → fit each branch.

    Reuses fit.py exactly like the fill path: closed branches go through
    fit_loop, open branches through _fit_run.  Emits the CENTERLINE primitives
    (offset-to-boundary is a separate step; see cad.trace.offset).  `mask` is
    assumed already side-picked (see _stroke_side)."""
    from cad.trace.skeleton import skeletonize_mask, prune_spurs
    from cad.trace.walk import walk_skeleton
    from cad.trace.fit import fit_loop, _fit_run, FitParams

    skel, _dist = skeletonize_mask(mask)
    skel = prune_spurs(skel, params.prune_len)
    polys = walk_skeleton(skel)

    P = fit_params or FitParams()
    segs: list = []
    for pl in polys:
        pts = pl.points
        if len(pts) < 2:
            continue
        if pl.closed and len(pts) >= 8:
            segs.extend(fit_loop(pts, P))
        else:
            segs.extend(_fit_run(np.asarray(pts, dtype=np.float64), P))
    return segs


def segments_to_entities(segments: list, image_w: int, image_h: int,
                         params: TraceParams, weld_tol: float = 1e-3):
    """
    Convert fitted pixel-space Segments to sketch entities in UV mm.

    LineSeg → LineEntity.  ArcSeg → ArcEntity (start_angle < end_angle, CCW).
    PolySeg → LineEntity chain (the V1 floor).

    The image→UV mapping flips y (v grows up), which REVERSES winding: an arc
    sweeping CCW in image coords sweeps CW in UV.  We map every point into the
    final UV frame first, then compute arc angles there, so the emitted arc
    passes through the same start/end points and the ArcEntity CCW contract
    (start_angle < end_angle) holds in UV.
    """
    from cad.sketch import LineEntity, ArcEntity
    from cad.trace.fit import LineSeg, ArcSeg, PolySeg

    mm_per_px = params.scale_mm / float(image_w) if image_w else 1.0
    ou, ov = params.origin_u, params.origin_v

    def to_uv(pt):
        x, y = float(pt[0]), float(pt[1])
        return np.array([ou + x * mm_per_px,
                         ov + (image_h - y) * mm_per_px])

    ents: list = []
    for s in segments:
        if isinstance(s, LineSeg):
            a, b = to_uv(s.p0), to_uv(s.p1)
            if np.linalg.norm(b - a) >= weld_tol:
                ents.append(LineEntity((a[0], a[1]), (b[0], b[1])))
        elif isinstance(s, ArcSeg):
            ents.extend(_arc_seg_to_entities(s, to_uv, mm_per_px, weld_tol))
        elif isinstance(s, PolySeg):
            uvpts = [to_uv(p) for p in s.points]
            for p, q in zip(uvpts[:-1], uvpts[1:]):
                if np.linalg.norm(q - p) >= weld_tol:
                    ents.append(LineEntity((p[0], p[1]), (q[0], q[1])))
    return ents


def _arc_seg_to_entities(s, to_uv, mm_per_px, weld_tol):
    """One ArcSeg → [ArcEntity] (or [] if degenerate).

    Full circles (p0 == p1) emit start=0, end=2π.  Partial arcs compute
    start/end angles in the UV frame and normalise so start < end while the arc
    still passes through the UV images of the original start/end points."""
    from cad.sketch import ArcEntity
    TWO_PI = 2.0 * np.pi

    center = to_uv(s.center)
    radius = float(s.radius) * mm_per_px
    if radius < weld_tol:
        return []

    p0 = to_uv(s.p0)
    p1 = to_uv(s.p1)
    is_full = np.linalg.norm(p1 - p0) < weld_tol

    if is_full:
        return [ArcEntity((center[0], center[1]), radius, 0.0, TWO_PI)]

    a0 = float(np.arctan2(p0[1] - center[1], p0[0] - center[0]))
    a1 = float(np.arctan2(p1[1] - center[1], p1[0] - center[0]))

    # ArcEntity always sweeps CCW from start_angle to end_angle (start < end),
    # with .p0 at start_angle and .p1 at end_angle.  We want .p0 == UV p0 and
    # .p1 == UV p1 so the entity's endpoints match the fitter's traversal
    # (needed for the profile to weld into a connected loop).
    #
    # In UV the sweep is the opposite winding of image-space s.ccw (the y-flip
    # reverses handedness).  A UV-CCW traversal p0→p1 means start=a0, end=a1
    # (a1 lifted above a0); a UV-CW traversal means the CCW arc from p0 to p1 is
    # the *major* complement, which is NOT what we want — instead we anchor .p0
    # at UV p0 and sweep CCW to p1 only when that is the minor arc.  Concretely:
    # pick start=a0 and unwrap end=a1 above it; if s is UV-clockwise the fitted
    # (minor) arc runs p1→p0 CCW, so swap the anchor to keep the minor arc while
    # still matching endpoints — .p0 lands on p1, .p1 on p0, which welds equally
    # well since both endpoints are shared corners.
    uv_ccw = not s.ccw
    if uv_ccw:
        start, end = a0, a1
    else:
        start, end = a1, a0
    while end <= start:
        end += TWO_PI
    if end - start >= TWO_PI - 1e-9:
        return [ArcEntity((center[0], center[1]), radius, 0.0, TWO_PI)]
    return [ArcEntity((center[0], center[1]), radius, start, end)]
