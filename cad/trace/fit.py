"""
cad/trace/fit.py

Stage 3 (Milestone B): curvature-segmented primitive fitting.  Turns an ordered
closed pixel loop (from cad.trace.contours) into a small set of geometric
primitives — straight lines and circular arcs — with a polyline fallback for
runs that fit neither.

All work is in PIXEL space (see IMAGE_TRACE_V2_PLAN.md: fit in px, calibrate
with a ruler, scale at emit).  The output is a list of `Segment`s that
image_trace.py converts to LineEntity / ArcEntity at emit time.

Algorithm per loop
------------------
1. Resample the loop to roughly unit-spaced points (kills pixel-staircase noise
   without losing shape) and lightly smooth.
2. Compute a wrapped tangent-angle profile; its derivative is curvature.
3. Split at CORNERS — points where the tangent turns sharply over a short span
   (governed by `corner_threshold` radians).
4. For each run between corners, try in order:
     a. straight line (total-least-squares) — accept if max deviation <= tol;
     b. circular arc (Taubin algebraic fit) — accept if max radial deviation
        <= tol AND the run actually curves (not a near-straight sliver);
     c. otherwise emit the run as a short polyline (the V1 floor).
5. Arc orientation (CW/CCW, minor/major) is derived from the run's point order,
   which the border-follower kept consistent — so emitted arcs wind correctly.

Tolerances are in pixels.  `fit_tolerance` is the max point deviation to accept
a primitive; `corner_threshold` (radians) is the tangent jump that marks a
corner.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


# ---------------------------------------------------------------------------
# Output representation — neutral, pixel-space.  image_trace.py maps to entities.
# ---------------------------------------------------------------------------

@dataclass
class LineSeg:
    p0: np.ndarray            # (x, y) px
    p1: np.ndarray


@dataclass
class ArcSeg:
    center: np.ndarray        # (x, y) px
    radius: float             # px
    p0: np.ndarray            # start point (x, y) px  — on the arc
    p1: np.ndarray            # end point   (x, y) px  — on the arc
    ccw: bool                 # sweep direction from p0 to p1 in image coords


@dataclass
class PolySeg:
    points: np.ndarray        # (N,2) px — fallback chain


Segment = object  # LineSeg | ArcSeg | PolySeg


@dataclass
class FitParams:
    corner_threshold: float = np.deg2rad(38.0)   # tangent jump = corner (rad)
    fit_tolerance:    float = 1.6                 # max px deviation to accept
    min_arc_sweep:    float = np.deg2rad(25.0)    # below this, prefer a line
    resample_step:    float = 1.5                 # px between resampled points
    smooth_window:    int   = 5                   # tangent smoothing (points)
    # Arc-vs-line discrimination — stop fitting arcs to noisy straight edges.
    # The physically-correct discriminator is radius/chord: a near-straight run
    # fits a huge-radius circle, so cap it.  arc_line_margin is an ADDITIONAL
    # optional bias (0 = off) the user can raise to force more lines.
    max_radius_chord_ratio: float = 10.0  # r > ratio*chord ⇒ treat as a line
    arc_line_margin:        float = 0.0   # extra px the arc must beat line by
    # Post-pass: merge consecutive arcs that are really one arc.
    merge_center_tol:  float = 4.0        # px: centres this close ⇒ same circle
    merge_radius_tol:  float = 3.0        # px: radii this close ⇒ same circle


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _resample_closed(points: np.ndarray, step: float) -> np.ndarray:
    """Resample a closed loop to ~`step`-spaced points by arc length."""
    p = np.asarray(points, dtype=np.float64)
    if len(p) < 4:
        return p
    closed = np.vstack([p, p[0]])
    seg = np.diff(closed, axis=0)
    seglen = np.hypot(seg[:, 0], seg[:, 1])
    s = np.concatenate([[0.0], np.cumsum(seglen)])
    total = s[-1]
    if total < step:
        return p
    n = max(int(total / step), 8)
    targets = np.linspace(0, total, n, endpoint=False)
    xs = np.interp(targets, s, closed[:, 0])
    ys = np.interp(targets, s, closed[:, 1])
    return np.column_stack([xs, ys])


def _tangent_angles(points: np.ndarray, window: int) -> np.ndarray:
    """Wrapped, smoothed tangent angle at each point of a closed loop."""
    n = len(points)
    nxt = np.roll(points, -1, axis=0)
    prv = np.roll(points, 1, axis=0)
    d = nxt - prv                        # central difference (wraps)
    ang = np.arctan2(d[:, 1], d[:, 0])
    if window > 1:
        # Smooth the *unwrapped* angle so averaging doesn't cross the ±pi seam.
        un = np.unwrap(ang)
        k = np.ones(window) / window
        un = np.convolve(np.concatenate([un[-window:], un, un[:window]]), k,
                         mode="same")[window:window + n]
        ang = un
    return ang


def _corner_turn(points: np.ndarray, span: int) -> np.ndarray:
    """Turning angle at each point measured between the chord arriving from
    `span` points back and the chord leaving `span` points ahead.

    This is a *local, staircase-robust* corner detector: a smooth arc spreads
    its turn over many points so each point's turn is small; a true corner
    concentrates the turn.  Unlike a smoothed tangent it does not blur the sharp
    corners we must keep (fixes squares fitting spurious arcs)."""
    n = len(points)
    fwd = np.roll(points, -span, axis=0) - points
    bwd = points - np.roll(points, span, axis=0)
    af = np.arctan2(fwd[:, 1], fwd[:, 0])
    ab = np.arctan2(bwd[:, 1], bwd[:, 0])
    turn = np.abs((af - ab + np.pi) % (2 * np.pi) - np.pi)
    return turn


def _find_corners(points: np.ndarray, threshold: float,
                  span: int = 3) -> list[int]:
    """Indices of local turning-angle maxima exceeding `threshold`.

    A corner is a point whose local turn is the largest in its neighbourhood
    (non-maximum suppression), so a single corner yields a single index even
    when the turn is spread over a few samples."""
    n = len(points)
    turn = _corner_turn(points, span)
    corners = []
    w = span                              # suppression radius
    for i in range(n):
        if turn[i] <= threshold:
            continue
        # Local maximum within ±w (wrapping)?
        window = turn[(np.arange(i - w, i + w + 1)) % n]
        if turn[i] >= window.max() - 1e-9:
            corners.append(i)
    # Deduplicate near-adjacent survivors, keeping the strongest.
    if not corners:
        return []
    merged = [corners[0]]
    for c in corners[1:]:
        if c - merged[-1] <= w:
            if turn[c] > turn[merged[-1]]:
                merged[-1] = c
        else:
            merged.append(c)
    # Also merge wrap-around adjacency (last near first).
    if len(merged) > 1 and (merged[0] + n - merged[-1]) <= w:
        if turn[merged[0]] >= turn[merged[-1]]:
            merged.pop()
        else:
            merged.pop(0)
    return merged


def _fit_line_dev(run: np.ndarray) -> float:
    """Max perpendicular deviation of `run` from its total-least-squares line."""
    c = run.mean(axis=0)
    u, s, vt = np.linalg.svd(run - c)
    normal = vt[1]                       # direction of least variance
    return float(np.max(np.abs((run - c) @ normal)))


def _fit_circle_taubin(run: np.ndarray):
    """Taubin algebraic circle fit.  Returns (cx, cy, r) or None."""
    x = run[:, 0]; y = run[:, 1]
    n = len(run)
    if n < 3:
        return None
    xm, ym = x.mean(), y.mean()
    u = x - xm; v = y - ym
    z = u * u + v * v
    zm = z.mean()
    Suu = (u * u).mean(); Svv = (v * v).mean(); Suv = (u * v).mean()
    Suz = (u * z).mean(); Svz = (v * z).mean()
    # Solve the 2x2 system from the Taubin/Kasa normal equations.
    A = np.array([[Suu, Suv], [Suv, Svv]])
    b = 0.5 * np.array([Suz, Svz])
    try:
        uc, vc = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None
    cx, cy = uc + xm, vc + ym
    r = float(np.sqrt(uc * uc + vc * vc + zm))
    if not np.isfinite(r) or r <= 0:
        return None
    return cx, cy, r


def _circle_dev(run: np.ndarray, cx: float, cy: float, r: float) -> float:
    d = np.hypot(run[:, 0] - cx, run[:, 1] - cy)
    return float(np.max(np.abs(d - r)))


def _arc_is_ccw(run: np.ndarray, cx: float, cy: float) -> bool:
    """Signed angular sweep of the run about (cx,cy); True if net CCW."""
    a = np.arctan2(run[:, 1] - cy, run[:, 0] - cx)
    da = np.diff(np.unwrap(a))
    return float(np.sum(da)) > 0


def _arc_sweep(run: np.ndarray, cx: float, cy: float) -> float:
    a = np.arctan2(run[:, 1] - cy, run[:, 0] - cx)
    return abs(float(np.sum(np.diff(np.unwrap(a)))))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit_loop(points: np.ndarray, params: FitParams | None = None) -> list:
    """
    Fit a single closed pixel loop into a list of Segments (LineSeg / ArcSeg /
    PolySeg), in pixel space.  Point order is preserved so arc winding is
    correct downstream.
    """
    P = params or FitParams()
    pts = _resample_closed(np.asarray(points, dtype=np.float64), P.resample_step)
    n = len(pts)
    if n < 4:
        return [PolySeg(points=pts)]

    # Whole-loop circle check first: a loop that is one circle should be a
    # single full-circle arc, not chopped by spurious corners.
    full = _fit_circle_taubin(pts)
    if full is not None:
        cx, cy, r = full
        if _circle_dev(pts, cx, cy, r) <= P.fit_tolerance:
            ccw = _arc_is_ccw(pts, cx, cy)
            return [ArcSeg(center=np.array([cx, cy]), radius=r,
                           p0=pts[0].copy(), p1=pts[0].copy(), ccw=ccw)]

    # Corner span: a few resampled points; scales with resample density.
    span = max(2, int(round(3.0 / P.resample_step * 1.5)))
    corners = _find_corners(pts, P.corner_threshold, span=span)
    if not corners:
        # No corners and not a full circle: one open run over the whole loop.
        corners = [0]

    segs: list = []
    m = len(corners)
    for i in range(m):
        a = corners[i]
        b = corners[(i + 1) % m]
        # Extract the run a..b inclusive, wrapping.
        if b > a:
            run = pts[a:b + 1]
        else:
            run = np.vstack([pts[a:], pts[:b + 1]])
        if len(run) < 2:
            continue
        segs.extend(_fit_run(run, P))

    segs = _merge_adjacent_arcs(segs, P)
    _weld_segment_endpoints(segs)
    return segs


def _merge_adjacent_arcs(segs: list, P: FitParams) -> list:
    """Collapse consecutive ArcSegs that belong to the same underlying circle
    (close centres and radii, same winding) into a single arc.

    A single rounded corner sometimes gets split into 2-4 slightly-different
    arcs by spurious corner detections; this stitches them back so the output
    reads as one clean fillet instead of a fan of near-duplicates."""
    if len(segs) < 2:
        return segs
    out: list = [segs[0]]
    for s in segs[1:]:
        prev = out[-1]
        if (isinstance(prev, ArcSeg) and isinstance(s, ArcSeg)
                and prev.ccw == s.ccw
                and np.linalg.norm(prev.center - s.center) <= P.merge_center_tol
                and abs(prev.radius - s.radius) <= P.merge_radius_tol):
            # Merge: keep prev's centre/radius (averaged), extend end to s.p1.
            c = 0.5 * (prev.center + s.center)
            r = 0.5 * (prev.radius + s.radius)
            out[-1] = ArcSeg(center=c, radius=r,
                             p0=prev.p0.copy(), p1=s.p1.copy(), ccw=prev.ccw)
        else:
            out.append(s)
    # Wrap-around: first and last may also be one arc across the seam.
    if len(out) >= 2:
        a, b = out[-1], out[0]
        if (isinstance(a, ArcSeg) and isinstance(b, ArcSeg)
                and a.ccw == b.ccw
                and np.linalg.norm(a.center - b.center) <= P.merge_center_tol
                and abs(a.radius - b.radius) <= P.merge_radius_tol):
            c = 0.5 * (a.center + b.center)
            r = 0.5 * (a.radius + b.radius)
            out[0] = ArcSeg(center=c, radius=r,
                            p0=a.p0.copy(), p1=b.p1.copy(), ccw=a.ccw)
            out.pop()
    return out


def _seg_p0(s):
    return s.points[0] if isinstance(s, PolySeg) else s.p0


def _seg_p1(s):
    return s.points[-1] if isinstance(s, PolySeg) else s.p1


def _project_to_circle(pt, center, radius):
    """Radially project pt onto the circle so it lies exactly `radius` from
    center — makes an ArcEntity's angle-derived endpoint land here exactly."""
    d = pt - center
    n = np.hypot(d[0], d[1])
    if n < 1e-9:
        return pt.copy()
    return center + d / n * radius


def _circle_circle_intersect(c0, r0, c1, r1, near):
    """Return the intersection point of two circles nearest `near`, or None if
    they don't intersect.  Standard two-circle intersection."""
    d = c1 - c0
    dist = float(np.hypot(d[0], d[1]))
    if dist < 1e-9 or dist > r0 + r1 or dist < abs(r0 - r1):
        return None
    a = (r0 * r0 - r1 * r1 + dist * dist) / (2 * dist)
    h2 = r0 * r0 - a * a
    if h2 < 0:
        return None
    h = np.sqrt(h2)
    mid = c0 + a * d / dist
    perp = np.array([-d[1], d[0]]) / dist
    p_a = mid + h * perp
    p_b = mid - h * perp
    return p_a if np.linalg.norm(p_a - near) <= np.linalg.norm(p_b - near) else p_b


def _weld_segment_endpoints(segs: list) -> None:
    """Force consecutive segments in a closed loop to share an EXACT endpoint.

    Runs share a corner in pixel space, but an ArcEntity reconstructs its
    endpoints from center+radius+angle, so an arc endpoint that isn't exactly
    `radius` from center drifts off its neighbour's endpoint by up to ~1px,
    breaking the welded profile.  Here we compute one shared vertex per corner
    and, for arcs, project it onto the arc's circle so the reconstructed
    endpoint coincides.  Lines get the raw shared vertex."""
    m = len(segs)
    if m < 2:
        return
    for i in range(m):
        s = segs[i]
        nxt = segs[(i + 1) % m]
        v = 0.5 * (_seg_p1(s) + _seg_p0(nxt))     # shared corner

        # The shared vertex must be IDENTICAL on both sides.  An arc constrains
        # its endpoint to its own circle (its entity reconstructs from angle),
        # so if either neighbour is an arc, the vertex is that arc's radial
        # projection of v; if both are arcs, project onto each independently and
        # average the two projections, then re-project so both land on it as
        # closely as possible.  A line simply adopts whatever the vertex is.
        s_arc = isinstance(s, ArcSeg)
        n_arc = isinstance(nxt, ArcSeg)
        if s_arc and n_arc:
            # Two arcs meet exactly only at a circle-circle intersection; use
            # the intersection nearest v so BOTH reconstructed endpoints land
            # on it.  Falls back to the averaged projection if the circles
            # don't intersect (near-tangent numerical case).
            inter = _circle_circle_intersect(s.center, s.radius,
                                             nxt.center, nxt.radius, near=v)
            if inter is not None:
                v = inter
            else:
                v = 0.5 * (_project_to_circle(v, s.center, s.radius) +
                           _project_to_circle(v, nxt.center, nxt.radius))
        elif s_arc:
            v = _project_to_circle(v, s.center, s.radius)
        elif n_arc:
            v = _project_to_circle(v, nxt.center, nxt.radius)

        if isinstance(s, PolySeg):
            s.points[-1] = v.copy()
        else:
            s.p1 = v.copy()
        if isinstance(nxt, PolySeg):
            nxt.points[0] = v.copy()
        else:
            nxt.p0 = v.copy()


def _run_length(run: np.ndarray) -> float:
    """Total polyline arc length of a run (robust vs endpoint chord, which can
    be ~0 for a closed or hairpin run)."""
    return float(np.sum(np.hypot(*np.diff(run, axis=0).T)))


def _fit_run(run: np.ndarray, P: FitParams, depth: int = 0) -> list:
    """Fit one run → primitives.  Tries line, then arc; if neither fits, splits
    the run and recurses (this is what decomposes a line→fillet→line run into
    its pieces, since fillet transitions are tangential, not sharp corners)."""
    if len(run) < 2:
        return []
    if len(run) == 2:
        return [LineSeg(p0=run[0].copy(), p1=run[1].copy())]

    line_dev = _fit_line_dev(run)
    if line_dev <= P.fit_tolerance:
        return [LineSeg(p0=run[0].copy(), p1=run[-1].copy())]

    circ = _fit_circle_taubin(run)
    arc_ok = False
    if circ is not None:
        cx, cy, r = circ
        dev = _circle_dev(run, cx, cy, r)
        sweep = _arc_sweep(run, cx, cy)
        length = _run_length(run)
        # A genuine arc: fits within tolerance, sweeps a real angle, and its
        # radius doesn't dwarf its arc LENGTH (a near-straight run fits a huge
        # circle — that's a line).  arc_line_margin is an optional extra bias.
        big_radius = r > P.max_radius_chord_ratio * max(length, 1e-6)
        beats_line = (line_dev - dev) >= P.arc_line_margin
        arc_ok = (dev <= P.fit_tolerance and sweep >= P.min_arc_sweep
                  and not big_radius and beats_line)
        if arc_ok:
            return [ArcSeg(center=np.array([cx, cy]), radius=r,
                           p0=run[0].copy(), p1=run[-1].copy(),
                           ccw=_arc_is_ccw(run, cx, cy))]

    # Neither line nor arc: split and recurse (bounded depth) so compound runs
    # (line + fillet + line) decompose into their tangent pieces.
    if len(run) >= 6 and depth < 12:
        mid = len(run) // 2
        left = _fit_run(run[:mid + 1], P, depth + 1)
        right = _fit_run(run[mid:], P, depth + 1)
        combined = left + right
        # Accept the split if it produced no polyline fallback (clean pieces).
        if combined and not any(isinstance(s, PolySeg) for s in combined):
            return combined
        # Otherwise keep whichever is cleaner: prefer the split if it has fewer
        # polyline points than emitting the whole run as one polyline.
        if combined:
            return combined
    return [PolySeg(points=run.copy())]


def fit_contours(contours, params: FitParams | None = None) -> list:
    """Fit every Contour (from cad.trace.contours) → flat list of Segments."""
    out: list = []
    for c in contours:
        out.extend(fit_loop(c.points, params))
    return out
