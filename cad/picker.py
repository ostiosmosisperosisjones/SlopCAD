"""
cad/picker.py

Vectorized Möller–Trumbore ray-triangle intersection.
Returns face index, or (face_index, t) when return_t=True.
"""

import numpy as np


def _ray_hits_aabb(origin, direction, bmin, bmax) -> bool:
    """Slab test: does the ray intersect the axis-aligned box [bmin, bmax]?

    Cheap (~6 divides) reject so pick_face can skip the full Möller–Trumbore
    triangle sweep for bodies the cursor ray misses entirely — the common case
    with many bodies on screen, where the ray hits only one or two."""
    # 1/dir with inf where dir==0 (ray parallel to that slab) handled by the
    # min/max logic below.
    with np.errstate(divide='ignore'):
        inv = 1.0 / direction
    t0 = (bmin - origin) * inv
    t1 = (bmax - origin) * inv
    tmin = np.minimum(t0, t1).max()
    tmax = np.maximum(t0, t1).min()
    return tmax >= max(tmin, 0.0)


def pick_face(mesh, ray_origin, ray_dir, return_t: bool = False):
    origin    = np.array(ray_origin, dtype=np.float64)
    direction = np.array(ray_dir,    dtype=np.float64)

    # Fast AABB reject before the per-triangle sweep.
    bmin = getattr(mesh, 'bbox_min', None)
    bmax = getattr(mesh, 'bbox_max', None)
    if bmin is not None and bmax is not None:
        if not _ray_hits_aabb(origin, direction,
                              np.asarray(bmin, dtype=np.float64),
                              np.asarray(bmax, dtype=np.float64)):
            return None

    tris  = mesh.tris

    # v0/e1/e2 depend only on geometry, not the ray — cache them on the mesh so
    # repeated picks (every mouse-move re-picks all bodies) skip the fancy-index
    # gather + subtraction that otherwise dominated each call.
    cache = getattr(mesh, '_pick_cache', None)
    if cache is None:
        verts = mesh.verts.astype(np.float64)
        v0 = verts[tris[:, 0]]
        e1 = verts[tris[:, 1]] - v0
        e2 = verts[tris[:, 2]] - v0
        cache = (v0, e1, e2)
        try:
            mesh._pick_cache = cache
        except Exception:
            pass
    v0, e1, e2 = cache

    h     = np.cross(direction[np.newaxis, :], e2)
    a     = np.einsum('ij,ij->i', e1, h)
    eps   = 1e-9
    valid = np.abs(a) > eps
    f     = np.where(valid, 1.0 / np.where(valid, a, 1.0), 0.0)

    s = origin[np.newaxis, :] - v0
    u = f * np.einsum('ij,ij->i', s, h)
    valid &= (u >= 0.0) & (u <= 1.0)

    q = np.cross(s, e1)
    v = f * (q @ direction)
    valid &= (v >= 0.0) & (u + v <= 1.0)

    t = f * np.einsum('ij,ij->i', e2, q)
    valid &= (t > eps)

    if not np.any(valid):
        return None

    t_vals      = np.where(valid, t, np.inf)
    closest_tri = int(np.argmin(t_vals))
    min_t       = float(t_vals[closest_tri])

    tpf      = mesh.triangles_per_face
    cumsum   = np.cumsum(tpf)
    face_idx = int(np.searchsorted(cumsum, closest_tri, side='right'))

    if return_t:
        return face_idx, min_t
    return face_idx
