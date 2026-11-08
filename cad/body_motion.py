"""
cad/body_motion.py

Where a body has been moved since a given entry.

Provenance (cad.face_origin) predicts a face or edge in the frame of the op
that made it: a FaceOrigin re-reads its sketch line, a rim re-reads its
circle.  A TransformOp after that op moves the body rigidly, so the
prediction has to be moved the same way before it is looked for.

    motion(history, body_id, before_index, since_index)
        -> (body id as of since_index, gp_Trsf | None)

walks back from before_index over the TransformOps on the body (following
a copy to its source body) down to, not including, since_index, and
composes them.  None means "not moved".

    in_frame(history, shape, at, since, find)

runs find() on the shape moved back into that frame and maps what it
returns (a face (idx, face) pair, or a TopoDS_Edge) to the real shape.
at = (body_id, before_index): where `shape` came from.  at=None means the
caller didn't say, and the shape is used as is.
"""

from __future__ import annotations

import numpy as np


def motion(history, body_id, before_index, since_index):
    b, T = body_id, None
    entries = history._entries
    for i in range(min(before_index, len(entries)) - 1, since_index, -1):
        e = entries[i]
        op = e.op
        if type(op).__name__ != "TransformOp" or e.error or e.body_id != b:
            continue
        Ti = op.trsf(history, i)
        if Ti is not None:
            T = Ti if T is None else T.Multiplied(Ti)
        if op.copy:
            b = op.source_body_id
    return b, T


def made_since(history, at, since_index):
    """motion() for at = (body_id, before_index), or (None, None)."""
    if at is None or since_index is None:
        return (at[0] if at else None), None
    return motion(history, at[0], at[1], since_index)


def pt(T, p):
    if T is None:
        return np.asarray(p, dtype=float)
    from OCP.gp import gp_Pnt
    q = gp_Pnt(*map(float, p)).Transformed(T)
    return np.array([q.X(), q.Y(), q.Z()])


def vec(T, v):
    if T is None:
        return np.asarray(v, dtype=float)
    from OCP.gp import gp_Vec
    q = gp_Vec(*map(float, v)).Transformed(T)
    return np.array([q.X(), q.Y(), q.Z()])


def moved(occ, T):
    """occ moved by T (a located copy: same TShape, so IsPartner maps back)."""
    if T is None:
        return occ
    from OCP.TopLoc import TopLoc_Location
    return occ.Moved(TopLoc_Location(T))


def _b3d(shape, occ):
    try:
        return type(shape)(occ)
    except Exception:
        from build123d import Compound
        return Compound(occ)


def in_frame(history, shape, at, since, find):
    _, T = made_since(history, at, since)
    if T is None:
        return find(shape)
    back = _b3d(shape, moved(shape.wrapped, T.Inverted()))
    res = find(back)
    if res is None:
        return None
    if isinstance(res, tuple):                   # (face index, face)
        idx, face = res
        if face is None:
            return None, None
        for k, f in enumerate(shape.faces()):
            if f.wrapped.IsPartner(face.wrapped):
                return k, f
        return None, None
    for e in shape.edges():                      # TopoDS_Edge
        if e.wrapped.IsPartner(res):
            return e.wrapped
    return None
