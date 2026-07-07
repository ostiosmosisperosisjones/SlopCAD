"""
cad/trace/contours.py

Stage 2 of the V2 tracer: the cv2.findContours replacement.

Extracts ordered, closed boundary loops from a boolean foreground mask, with
outer/hole hierarchy, using scipy.ndimage.label for connected components +
Moore-neighbour border following for the boundary walk.

Parity bar (decided): geometric equivalence with cv2.findContours, not
point-for-point.  Same loops, correct winding, holes separated, boundary within
~1px.  Downstream Douglas-Peucker + primitive fitting make exact vertex parity
pointless.

Output
------
trace_contours(mask) -> list[Contour]
    Contour.points : (N,2) int array of (x, y) pixel coords, closed loop
                     (first != last; caller closes if needed), CCW for outer
                     boundaries in image coords (y-down).
    Contour.is_hole: True if this loop bounds a background region inside a
                     foreground component (an inner hole), else False.

Winding convention: outer boundaries are traced counter-clockwise in *image*
coordinates (x-right, y-down).  Holes come out clockwise (opposite), which is
the natural orientation for an inner boundary.  Downstream fitting reads this
winding to orient arcs; the mm emit step flips y (v grows up), which also flips
the effective winding to standard math CCW — handled at emit, not here.
"""

from __future__ import annotations
import numpy as np


# 8-connected Moore neighbourhood, clockwise starting from "east".
# (dx, dy) in image coords (x right, y down).
_MOORE = [(1, 0), (1, 1), (0, 1), (-1, 1),
          (-1, 0), (-1, -1), (0, -1), (1, -1)]


class Contour:
    __slots__ = ("points", "is_hole")

    def __init__(self, points: np.ndarray, is_hole: bool):
        self.points = points          # (N,2) int (x,y)
        self.is_hole = is_hole

    def area(self) -> float:
        """Signed shoelace area (image coords). |area| is the enclosed pixels;
        sign encodes winding (negative = CCW in y-down image space)."""
        p = self.points
        if len(p) < 3:
            return 0.0
        x = p[:, 0].astype(np.float64)
        y = p[:, 1].astype(np.float64)
        return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def _pad(mask: np.ndarray) -> np.ndarray:
    """1-px False border so boundary walks never index out of range and edge-
    touching shapes still have a background neighbour to hug."""
    return np.pad(mask, 1, mode="constant", constant_values=False)


def _trace_border(region: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    """
    Moore-neighbour border following of a single connected True region in a
    padded boolean array, beginning at `start` (the first foreground pixel in
    raster order, which is guaranteed on the boundary).

    Returns an (N,2) int array of (x, y) boundary pixel coords in the padded
    frame.  Uses Jacob's stopping criterion (revisit start from same incoming
    direction) so thin/one-pixel structures terminate correctly.
    """
    h, w = region.shape
    sx, sy = start
    boundary = [(sx, sy)]

    # Backtrack direction: we entered `start` from the west (came from -x),
    # so begin searching the neighbourhood from just past that.
    # Standard Moore tracing: from current pixel, start looking at the neighbour
    # clockwise-adjacent to where we came from.
    prev_dir = 4  # index into _MOORE pointing "west" (came from east side)
    cx, cy = sx, sy
    max_steps = h * w * 8      # hard cap: cannot loop longer than this
    steps = 0

    while steps < max_steps:
        steps += 1
        found = False
        # Search the 8 neighbours clockwise starting just after the backtrack.
        start_search = (prev_dir + 1) % 8
        for k in range(8):
            d = (start_search + k) % 8
            dx, dy = _MOORE[d]
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < w and 0 <= ny < h and region[ny, nx]:
                # Move here; the backtrack direction is opposite of our travel.
                cx, cy = nx, ny
                prev_dir = (d + 4) % 8
                found = True
                break
        if not found:
            # Isolated pixel — no foreground neighbour.
            break
        if (cx, cy) == (sx, sy):
            # Returned to start: loop closed.
            break
        boundary.append((cx, cy))

    return np.array(boundary, dtype=np.int64)


def trace_contours(mask: np.ndarray, min_area: float = 0.0) -> list[Contour]:
    """
    Extract outer + hole boundary loops from a boolean foreground mask.

    min_area drops loops whose |shoelace area| is below the threshold (px^2).
    Returns Contours in the padded-then-unpadded original coordinate frame.
    """
    from scipy.ndimage import label, binary_fill_holes

    padded = _pad(mask)

    # --- outer boundaries: one per foreground connected component ---
    fg_labels, n_fg = label(padded)          # 4-connectivity default
    contours: list[Contour] = []

    for lab in range(1, n_fg + 1):
        comp = fg_labels == lab
        ys, xs = np.nonzero(comp)
        if len(xs) == 0:
            continue
        # First pixel in raster order (min y, then min x) is on the boundary.
        i0 = np.lexsort((xs, ys))[0]
        start = (int(xs[i0]), int(ys[i0]))
        pts = _trace_border(comp, start)
        c = Contour(pts - 1, is_hole=False)   # undo the 1-px pad
        if abs(c.area()) >= min_area:
            contours.append(c)

        # --- holes of this component: background regions fully enclosed ---
        filled = binary_fill_holes(comp)
        holes = filled & ~comp                # pixels that were holes
        if not holes.any():
            continue
        hole_labels, n_h = label(holes)
        for hl in range(1, n_h + 1):
            hole = hole_labels == hl
            hys, hxs = np.nonzero(hole)
            if len(hxs) == 0:
                continue
            j0 = np.lexsort((hxs, hys))[0]
            hstart = (int(hxs[j0]), int(hys[j0]))
            hpts = _trace_border(hole, hstart)
            # _trace_border walks the hole region's own outer boundary, giving
            # it the same winding as a foreground outer contour.  An inner hole
            # boundary must wind oppositely (so shoelace sign distinguishes
            # outer vs hole, and downstream fitting reads a consistent inside/
            # outside).  Reverse the point order to flip the winding.
            hc = Contour((hpts - 1)[::-1].copy(), is_hole=True)
            if abs(hc.area()) >= min_area:
                contours.append(hc)

    return contours
