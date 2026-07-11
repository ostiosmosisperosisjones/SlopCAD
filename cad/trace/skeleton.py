"""
cad/trace/skeleton.py

Stage 2b of the V3 (inside-out / stroke) tracer: reduce a stroke mask to its
one-pixel-wide centerline (skeleton) plus the local half-thickness at each
skeleton pixel, then prune short spurs.

This is the STROKE-mode counterpart to contours.py (which does boundary
following for FILL mode).  Where contours.py walks *around* a shape, skeleton.py
collapses a stroke to its *middle*.  Both feed the same walk→fit back end.

Why skeleton, not boundary, for strokes
----------------------------------------
A boundary follower traces *around* a thin stroke, yielding a skinny closed loop
(two nearly-coincident sides) instead of one centerline.  The Medial Axis
Transform (Blum 1967) gives the centerline directly, and the distance transform
value at each skeleton pixel is the local half-thickness — everything needed to
later offset the fitted centerline back to an exact boundary (see offset.py).

Output
------
skeletonize_mask(mask) -> (skel, dist)
    skel : bool HxW, True on the one-px centerline.
    dist : float HxW, Euclidean distance to the nearest background pixel; on the
           skeleton this is the local half-thickness (medial radius).

prune_spurs(skel, max_len) -> bool HxW
    skel with short dead-end branches (< max_len px, endpoint→junction) removed.
    Every boundary bump on a stroke spawns a spur; pruning is what makes the
    skeleton read as clean primitives rather than a hairy tree.

Topology vs thickness — why two operators
------------------------------------------
`skimage.medial_axis` returns a distance transform but its axis FORKS at the
flat ends of a rectangular stroke (the medial axis genuinely branches toward the
two corners), so a straight stroke comes out with 4 endpoints + 2 junctions —
noisy topology.  `skimage.skeletonize` (Zhang-Suen thinning) gives the clean
2-endpoint centerline we want but no distances.  So: take TOPOLOGY from
`skeletonize`, take the half-thickness from a separate Euclidean distance
transform of the mask (`scipy.ndimage.distance_transform_edt`) sampled on the
skeleton.  Decoupling them is both cleaner and correct.

Dependency: scikit-image (skeletonize).  Robust thinning is not worth
hand-rolling; spur behaviour and 1-px-width guarantees are subtle.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy import ndimage as ndi


@dataclass
class SkeletonParams:
    """Stroke-mode skeleton dials (exposed as sliders, per the V2 philosophy)."""
    prune_len: int = 8        # px: remove dead-end spurs shorter than this


# 8-neighbour offsets (dr, dc) in row/col image coords.
_NB = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))

# Convolution kernel counting 8-connected True neighbours.
_DEG_K = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)


def skeletonize_mask(mask: np.ndarray):
    """Skeleton + distance transform of a boolean stroke mask.

    Returns (skel bool HxW, dist float HxW).  Topology comes from Zhang-Suen
    thinning (clean centerline, no rectangle-end forking); `dist` is the
    Euclidean distance to background, so sampled on the skeleton it is the local
    half-thickness (used later by offset.py).
    """
    from skimage.morphology import skeletonize
    m = mask.astype(bool)
    skel = skeletonize(m).astype(bool)
    dist = ndi.distance_transform_edt(m).astype(np.float64)
    return skel, dist


def neighbour_count(skel: np.ndarray) -> np.ndarray:
    """8-connected skeleton-neighbour count at each skeleton pixel (0 off-skel).

    A pixel's degree in the skeleton graph: 1 = endpoint, 2 = simple path,
    >=3 = junction.
    """
    conv = ndi.convolve(skel.astype(np.uint8), _DEG_K, mode="constant")
    return conv * skel


def _nbrs(skel: np.ndarray, r: int, c: int):
    """List of on-skeleton 8-neighbours of (r, c)."""
    h, w = skel.shape
    out = []
    for dr, dc in _NB:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w and skel[rr, cc]:
            out.append((rr, cc))
    return out


def prune_spurs(skel: np.ndarray, max_len: int) -> np.ndarray:
    """Iteratively delete dead-end branches shorter than `max_len` pixels.

    From each endpoint (degree 1) we walk the simple path until it hits a
    junction (degree >= 3) or dead-ends; if that branch is <= max_len px it is a
    spur and gets removed.  Repeats until stable, since removing one spur can
    expose another.  A non-positive max_len is a no-op.
    """
    if max_len <= 0:
        return skel.copy()
    skel = skel.copy()
    changed = True
    while changed:
        changed = False
        # One convolution per pass; index this snapshot instead of recomputing
        # neighbour_count inside the walk (that made prune O(passes·endpoints·
        # steps) full-image convolutions — the dominant retrace cost).  `deg`
        # goes mildly stale as we delete pixels this pass, but the outer
        # `while changed` loop re-evaluates on a fresh convolution, so nothing
        # is missed.
        deg = neighbour_count(skel)
        endpoints = list(zip(*np.where(deg == 1)))
        for (r, c) in endpoints:
            if not skel[r, c]:
                continue                      # already removed this pass
            path = [(r, c)]
            prev, cur = None, (r, c)
            while True:
                nb = [p for p in _nbrs(skel, *cur) if p != prev]
                if len(nb) != 1:
                    # 0 = dead end; >1 = we are adjacent to a junction pixel.
                    break
                nxt = nb[0]
                if deg[nxt] >= 3:
                    break                     # stop AT the junction, keep it
                path.append(nxt)
                prev, cur = cur, nxt
                if len(path) > max_len:
                    break                     # long enough — not a spur
            if len(path) <= max_len:
                for p in path:
                    skel[p] = False           # delete the spur (junction kept)
                changed = True
    return skel
