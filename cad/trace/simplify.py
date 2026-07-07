"""
cad/trace/simplify.py

Douglas-Peucker polyline simplification (replaces cv2.approxPolyDP) plus the
shoelace area helper (replaces cv2.contourArea).  Pure numpy.
"""

from __future__ import annotations
import numpy as np


def _dp(points: np.ndarray, epsilon: float) -> np.ndarray:
    """Recursive Douglas-Peucker on an open (N,2) point array.  Returns the
    kept subset (indices preserved order)."""
    if len(points) < 3:
        return points
    start, end = points[0], points[-1]
    line = end - start
    line_len = np.hypot(*line)
    if line_len < 1e-12:
        # Degenerate segment: distance is to the start point.
        dists = np.hypot(points[:, 0] - start[0], points[:, 1] - start[1])
    else:
        # Perpendicular distance of each point to the start-end line.
        nx, ny = -line[1] / line_len, line[0] / line_len
        dists = np.abs((points[:, 0] - start[0]) * nx +
                       (points[:, 1] - start[1]) * ny)
    idx = int(np.argmax(dists))
    if dists[idx] > epsilon:
        left = _dp(points[:idx + 1], epsilon)
        right = _dp(points[idx:], epsilon)
        return np.vstack([left[:-1], right])
    return np.vstack([start, end])


def approx_poly(points: np.ndarray, epsilon: float,
                closed: bool = True) -> np.ndarray:
    """
    Douglas-Peucker approximation of a polyline, matching the intent of
    cv2.approxPolyDP.

    points  : (N,2) array.
    epsilon : max distance (px) a kept vertex may deviate.
    closed  : treat the polyline as a closed loop.
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 3 or epsilon <= 0:
        return pts
    if closed:
        # Anchor the split at the two farthest-apart points so a closed loop
        # simplifies stably, then DP each half.
        # Simple, robust choice: rotate so index 0 is one extreme point.
        d0 = np.hypot(pts[:, 0] - pts[0, 0], pts[:, 1] - pts[0, 1])
        far = int(np.argmax(d0))
        a = _dp(np.vstack([pts[:far + 1]]), epsilon)
        b = _dp(np.vstack([pts[far:], pts[:1]]), epsilon)
        merged = np.vstack([a[:-1], b[:-1]])
        return merged
    return _dp(pts, epsilon)


def polygon_area(points: np.ndarray) -> float:
    """|Shoelace area| of a polygon (px^2).  Replaces cv2.contourArea."""
    p = np.asarray(points, dtype=np.float64)
    if len(p) < 3:
        return 0.0
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y)))
