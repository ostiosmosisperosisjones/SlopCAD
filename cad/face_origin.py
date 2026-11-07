"""
cad/face_origin.py

Face provenance for sketch planes: "the side wall that sketch line <uid>
was extruded into".

FaceRef finds a face by its remembered normal/area/centroid, so it loses a
side wall that an upstream edit ROTATED (change one side of a polygon and
its neighbours swing) or resized past its tolerances.  A FaceOrigin instead
records a point on the face and the face normal in the frame of the sketch
line that made it: fraction along the line, mm across it, mm up the sketch
normal.  Resolving re-evaluates that frame from the line's CURRENT position
(the sketch has already replayed), so the probe moves, turns and stretches
with the wall, and the face is whichever planar face of the body passes
through the probe with that normal.

Only straight lines (and straight includes) consumed by a sketch-driven ExtrudeOp on the face's body get an
origin; everything else keeps relying on FaceRef alone.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from cad.face_ref import _occ_face_normal, _occ_face_props


_ON_PLANE   = 1e-4     # mm — the wall plane holds the extruded line
_NORMAL_DOT = 0.999    # resolved normal must match the line-frame prediction (~2.5°)
_ONE_FACE_DOT = 0.98   # one-face edge direction (~11°): the unnamed face may turn
_ONE_FACE_LOOSE = 0.8  # ...or ~37° when the edge is clearly the nearest one


@dataclass
class FaceOrigin:
    sketch_id: str       # entry_id of the sketch holding the line
    uid:       str       # uid of the LineEntity (or straight include)
    probe:     tuple     # (s fraction along line, w mm across, h mm along sketch normal)
    normal:    tuple     # face normal as (t, w, n) components
    # End cap: entry_id of the extrude whose far face this is.  The probe's
    # h is then re-read from that extrude's current depth, so the cap is
    # followed through depth edits too.
    cap_of:    str | None = None
    # Slanted wall: entry_id of the extrude/cut that swept the line along a
    # direction other than the sketch normal.  The frame's third axis is then
    # that op's CURRENT direction (h is mm along it) and its second the
    # wall's normal, so the wall turns when the direction does (a direction
    # picked from an edge follows the edge).
    swept_by:  str | None = None

    def to_dict(self) -> dict:
        d = {"sketch_id": self.sketch_id, "uid": self.uid,
             "probe": list(self.probe), "normal": list(self.normal)}
        if self.cap_of:
            d["cap_of"] = self.cap_of
        if self.swept_by:
            d["swept_by"] = self.swept_by
        return d

    @classmethod
    def from_dict(cls, d: dict | None) -> "FaceOrigin | None":
        if not d:
            return None
        return cls(d["sketch_id"], d["uid"], tuple(d["probe"]), tuple(d["normal"]),
                   d.get("cap_of"), d.get("swept_by"))

    def frame(self, history):
        """The line's current frame (A, L, t, w, n, dh) — dh lifts a cap's
        local heights to the extrude's current depth — or None if the line
        is gone."""
        idx = history.id_to_index(self.sketch_id)
        if idx is None:
            return None
        se = history._entries[idx].params.get("sketch_entry")
        line = _line_by_uid(se, self.uid)
        frame = _line_frame(se, line) if line is not None else None
        if frame is None:
            return None
        if self.swept_by:
            ci = history.id_to_index(self.swept_by)
            op = history._entries[ci].op if ci is not None else None
            return _swept_frame(frame, _extrude_dir(op, se)) if op is not None else None
        dh = 0.0
        if self.cap_of:
            ci = history.id_to_index(self.cap_of)
            op = history._entries[ci].op if ci is not None else None
            if op is None or not getattr(op, "along_normal", 0):
                return None
            dh = _cap_height(op) - self.probe[2]
        return (*frame, dh)

    @staticmethod
    def local(frame, p=None, v=None):
        """World point/vector → line-frame (s, w, h) / (t, w, n) components.
        A swept frame's axes aren't orthogonal (n runs along the sweep), so
        this solves for the components rather than projecting."""
        A, L, t, w, n, dh = frame
        M = np.column_stack([t, w, n])
        if p is not None:
            c = np.linalg.solve(M, np.asarray(p, dtype=float) - A)
            return (float(c[0]) / L, float(c[1]), float(c[2]) - dh)
        c = np.linalg.solve(M, np.asarray(v, dtype=float))
        return (float(c[0]), float(c[1]), float(c[2]))

    @staticmethod
    def world(frame, p=None, v=None):
        """Inverse of local()."""
        A, L, t, w, n, dh = frame
        if p is not None:
            return A + p[0] * L * t + p[1] * w + (p[2] + dh) * n
        return v[0] * t + v[1] * w + v[2] * n

    def find_in(self, history, shape) -> tuple[int, object] | tuple[None, None]:
        """The planar face of `shape` this origin names now, or (None, None)
        if the line is gone or no face passes through the probe."""
        frame = self.frame(history)
        if frame is None:
            return None, None
        P = self.world(frame, p=self.probe)
        N = self.world(frame, v=self.normal)
        N /= np.linalg.norm(N)
        reach = max(1.0, 0.5 * frame[1])
        n = frame[4]
        best = (None, None, np.inf)
        for fi, face in enumerate(shape.faces()):
            fn = _occ_face_normal(face.wrapped)
            if fn is None or float(np.dot(fn, N)) < _NORMAL_DOT:
                continue
            c, _ = _occ_face_props(face.wrapped)
            if abs(float(np.dot(P - c, fn))) > 1e-3:
                continue                       # not the wall's plane
            d = _dist_to_face(P, face.wrapped)
            if d is None or d >= best[2]:
                continue
            # Only the fraction along the line scales with the part: heights
            # and offsets are the mm it was built at, so a part that shrank
            # leaves the probe off its face.  A wall must still lie over the
            # line, at any height.  A cap or base is pinned by its plane.
            if abs(float(np.dot(fn, n))) < _NORMAL_DOT:
                u = n - float(np.dot(n, fn)) * fn
                level = P + float(np.dot(c - P, u)) / float(np.dot(u, u)) * u
                over = _dist_to_face(level, face.wrapped)
                if over is None or over >= reach:
                    continue
            best = (fi, face, d)
        return best[0], best[1]


def _swept_frame(frame, d):
    """A line frame (A, L, t, w, n) re-based on sweep direction d: (A, L, t,
    wall normal, d, 0), or None when d runs along the line."""
    if frame is None or d is None:
        return None
    A, L, t, _w, _n = frame[:5]
    d = np.asarray(d, dtype=float)
    d = d / np.linalg.norm(d)
    w = np.cross(t, d)
    if float(np.linalg.norm(w)) < 1e-6:
        return None
    return (A, L, t, w / np.linalg.norm(w), d, 0.0)


def origin_for_face(history, body_id: str, occ_face,
                    before_index: int) -> FaceOrigin | None:
    """Which extruded (or cut) sketch line made this planar face, if any.
    The most recent sketch-driven extrude/cut on the body wins (a later wall
    can be coplanar with an earlier one)."""
    N = _occ_face_normal(occ_face)
    if N is None:
        return None
    c, _ = _occ_face_props(occ_face)
    P = _point_on_face(c, occ_face)
    if P is None:
        return None
    for e in reversed(history._entries[:max(0, before_index)]):
        op = e.op
        sketch_id = _sketch_id_of(op)
        if sketch_id is None or e.error:
            continue
        if body_id not in (e.body_id, getattr(op, "merge_body_id", None),
                           getattr(op, "cut_body_id", None)):
            continue
        sk_idx = history.id_to_index(sketch_id)
        if sk_idx is None:
            continue
        sk_rec = history._entries[sk_idx]
        se = sk_rec.params.get("sketch_entry")
        if se is None:
            continue
        d = _extrude_dir(op, se)
        n_sk = np.asarray(se.plane_normal, dtype=float)
        end_cap = bool(op.along_normal) and abs(abs(float(np.dot(N, n_sk))) - 1.0) < 1e-9
        for ent in se.entities:
            frame = _line_frame(se, ent)
            if frame is None or not getattr(ent, "uid", None):
                continue
            A, L, t, w, n = frame
            if abs(float(np.dot(N, t))) > 1e-6:
                continue                       # wall must contain the line direction
            base = A + op.start_offset * d
            cap = None
            if min(abs(float(np.dot(A - P, N))), abs(float(np.dot(base - P, N)))) > _ON_PLANE:
                if not (end_cap and abs(float(np.dot(A + _cap_height(op) * n - P, N))) <= _ON_PLANE):
                    continue                   # line isn't on this wall's plane
                cap = e.entry_id               # the extrude's far face
            # A wall swept at a slant: measure it in the sweep's own frame,
            # where the face lies over the line (s in 0..1) and turns with it.
            swept = (cap is None and abs(float(np.dot(N, d))) < 1e-6
                     and abs(float(np.dot(d, n))) < 1 - 1e-9)
            fr = _swept_frame(frame, d) if swept else (*frame, 0.0)
            if fr is None:
                continue
            s, pw, ph = FaceOrigin.local(fr, p=P)
            if not (-1e-6 <= s <= 1 + 1e-6):
                continue                       # the face isn't over this line
            return FaceOrigin(
                sketch_id = sk_rec.entry_id,
                uid       = ent.uid,
                probe     = (round(s, 9), round(pw, 9), round(ph, 9)),
                normal    = tuple(round(x, 9) for x in FaceOrigin.local(fr, v=N)),
                cap_of    = cap,
                swept_by  = e.entry_id if swept else None,
            )
    return None


# ---------------------------------------------------------------------------
# chamfer faces: "the flat a chamfer cut on this edge"
# ---------------------------------------------------------------------------

@dataclass
class ChamferFaceOrigin:
    """A chamfer's planar face, named by the chamfer entry and which of its
    edge refs made it.  Resolved by re-finding that edge on the chamfer's
    input (its EdgeRef follows edits through EdgeOrigin) and taking the
    face that lies between the edge's two faces, nearest the edge.  Only
    for sketch planes — EdgeOrigin frames need a sketch line."""
    chamfer_of: str      # entry_id of the ChamferOp
    edge:       int      # index into that op's edge_refs

    def to_dict(self) -> dict:
        return {"chamfer_of": self.chamfer_of, "edge": self.edge}

    def find_in(self, history, shape) -> tuple[int, object] | tuple[None, None]:
        ci = history.id_to_index(self.chamfer_of)
        op = history._entries[ci].op if ci is not None else None
        refs = getattr(op, "edge_refs", None) or []
        if self.edge >= len(refs):
            return None, None
        src = history._shape_for_body_at(op.source_body_id, ci)
        if src is None:
            return None, None
        occ = refs[self.edge].resolve(history, src)
        geo = _chamfer_geometry(src.wrapped, occ) if occ is not None else None
        if geo is None:
            return None, None
        reach = 2.0 * op.distance + 0.5
        best = (None, None, reach)
        for fi, face in enumerate(shape.faces()):
            if not _between(face.wrapped, shape.wrapped, *geo):
                continue
            d = _dist_to_face(geo[0], face.wrapped)
            if d is not None and d < best[2]:
                best = (fi, face, d)
        return best[0], best[1]


# ---------------------------------------------------------------------------
# revolve caps: "the far (or start) face of revolve X's profile k"
# ---------------------------------------------------------------------------

_CAP_ON_PLANE = 1e-3   # mm — the body face lies in the rebuilt cap's plane
_CAP_NORMAL   = 0.999  # |cos| between body face and rebuilt cap normals


@dataclass
class RevolveCapOrigin:
    """A planar end face of a revolve (or revolve cut), named by the revolve
    entry, which of its profile faces made it, and which end: 0 = the
    profile's own plane, 1 = the profile swung through the angle.  Resolving
    rebuilds that cap from the revolve's CURRENT profile, axis and angle, so
    the far cap is followed when the angle (or the profile) is edited —
    its plane rotates, which FaceRef's normal match can't follow."""
    revolve_of: str      # entry_id of the RevolveOp / CrossBodyRevolveCutOp
    profile:    int      # index into the op's resolved profile faces
    end:        int      # 0 = start cap, 1 = far cap

    def to_dict(self) -> dict:
        return {"revolve_of": self.revolve_of, "profile": self.profile,
                "end": self.end}

    def cap(self, history):
        """The rebuilt cap as a build123d Face, or None."""
        caps = _revolve_caps(history, self.revolve_of)
        if caps is None or self.profile >= len(caps):
            return None
        return caps[self.profile][self.end]

    def find_in(self, history, shape) -> tuple[int, object] | tuple[None, None]:
        cap = self.cap(history)
        if cap is None:
            return None, None
        return _coplanar_nearest(shape, cap)


def _revolve_caps(history, entry_id):
    """[(start cap, far cap), ...] per profile face of a revolve entry, as of
    its own replay (sketch profiles reproject, the axis follows its sketch).
    None for a full turn (no caps) or when the entry can't be resolved."""
    from build123d import Axis
    ci = history.id_to_index(entry_id)
    if ci is None:
        return None
    e = history._entries[ci]
    op = e.op
    if type(op).__name__ not in ("RevolveOp", "CrossBodyRevolveCutOp") or e.error:
        return None
    if abs(float(op.angle_deg)) >= 360.0 - 1e-9 or op.axis_point is None:
        return None
    try:
        from cad.profile_source import faces_of
        from cad.op_revolve_thicken import _revolve_axis
        faces = faces_of(op.profiles, history, ci, "RevolveCap")
        pt, d = _revolve_axis(op, history)
        axis = Axis(tuple(map(float, pt)), tuple(map(float, d)))
        return [(f, f.rotate(axis, float(op.angle_deg))) for f in faces]
    except Exception:
        return None


def _coplanar_nearest(shape, cap):
    """The planar face of `shape` in `cap`'s plane nearest its centre."""
    N = _occ_face_normal(cap.wrapped)
    if N is None:
        return None, None
    C, area = _occ_face_props(cap.wrapped)
    best = (None, None, max(1.0, float(np.sqrt(abs(area)))))
    for fi, face in enumerate(shape.faces()):
        fn = _occ_face_normal(face.wrapped)
        if fn is None or abs(float(np.dot(fn, N))) < _CAP_NORMAL:
            continue
        c, _ = _occ_face_props(face.wrapped)
        if abs(float(np.dot(C - c, fn))) > _CAP_ON_PLANE:
            continue
        d = _dist_to_face(C, face.wrapped)
        if d is not None and d < best[2]:
            best = (fi, face, d)
    return best[0], best[1]


def _revolves_on(history, body_id, before_index):
    """Revolve entries that wrote body_id, most recent first."""
    for e in reversed(history._entries[:max(0, before_index)]):
        op = e.op
        if type(op).__name__ not in ("RevolveOp", "CrossBodyRevolveCutOp") or e.error:
            continue
        if body_id in (e.body_id, getattr(op, "merge_body_id", None),
                       getattr(op, "cut_body_id", None)) or \
                body_id in (e.params.get("child_body_ids") or []):
            yield e


def revolve_origin_for_face(history, body_id: str, occ_face,
                            before_index: int) -> RevolveCapOrigin | None:
    """Which revolve cap this planar face is, if any."""
    N = _occ_face_normal(occ_face)
    if N is None:
        return None
    c, _ = _occ_face_props(occ_face)
    P = _point_on_face(c, occ_face)
    if P is None:
        return None
    for e in _revolves_on(history, body_id, before_index):
        for k, ends in enumerate(_revolve_caps(history, e.entry_id) or []):
            for end in (1, 0):
                cap = ends[end]
                cn = _occ_face_normal(cap.wrapped)
                cc, _ = _occ_face_props(cap.wrapped)
                if cn is None or abs(float(np.dot(cn, N))) < 1 - 1e-6 or \
                        abs(float(np.dot(cc - P, N))) > _CAP_ON_PLANE:
                    continue
                d = _dist_to_face(P, cap.wrapped)
                if d is not None and d < _CAP_ON_PLANE:
                    return RevolveCapOrigin(e.entry_id, k, end)
    return None


def revolve_origin_from_ref(history, body_id: str, face_ref,
                            before_index: int) -> RevolveCapOrigin | None:
    """Older saves: a FaceRef taken off a revolve's far cap, from before
    caps had an origin.  The cap swings with the angle, so match by what a
    rotation about the axis keeps: height along the axis, distance from it,
    the normal's axial part and the area.  The far cap of any angle fits."""
    if face_ref is None or not hasattr(face_ref, "centroid_perp"):
        return None
    n = np.asarray(face_ref.normal, dtype=float)
    c = np.asarray(face_ref.centroid_perp, dtype=float) + face_ref.centroid_along * n
    for e in _revolves_on(history, body_id, before_index):
        caps = _revolve_caps(history, e.entry_id)
        if not caps:
            continue
        from cad.op_revolve_thicken import _revolve_axis
        try:
            p, a = (np.asarray(v, dtype=float) for v in _revolve_axis(e.op, history))
        except Exception:
            continue
        a = a / np.linalg.norm(a)

        def inv(cen, nor):
            r = cen - p
            h = float(r @ a)
            return h, float(np.linalg.norm(r - h * a)), abs(float(nor @ a))

        h, rad, na = inv(c, n)
        for k, (start, _far) in enumerate(caps):
            sc, sa = _occ_face_props(start.wrapped)
            sn = _occ_face_normal(start.wrapped)
            if sn is None:
                continue
            h0, rad0, na0 = inv(np.asarray(sc, dtype=float), sn)
            tol = 1e-3 * max(1.0, rad0)
            if abs(h - h0) < tol and abs(rad - rad0) < tol and abs(na - na0) < 1e-4 \
                    and abs(face_ref.area - sa) < 1e-3 * max(1.0, sa):
                return RevolveCapOrigin(e.entry_id, k, 1)
    return None


def face_origin_from_dict(d: dict | None):
    """FaceOrigin, ChamferFaceOrigin or RevolveCapOrigin, by shape of the dict."""
    if d and "chamfer_of" in d:
        return ChamferFaceOrigin(d["chamfer_of"], int(d["edge"]))
    if d and "revolve_of" in d:
        return RevolveCapOrigin(d["revolve_of"], int(d["profile"]), int(d["end"]))
    return FaceOrigin.from_dict(d)


def chamfer_origin_for_face(history, body_id: str, occ_face,
                            before_index: int) -> ChamferFaceOrigin | None:
    """Which chamfer (by edge ref) made this planar face, if any."""
    for e in reversed(history._entries[:max(0, before_index)]):
        op = e.op
        if type(op).__name__ != "ChamferOp" or e.error or op.source_body_id != body_id:
            continue
        ci = history.id_to_index(e.entry_id)
        src = history._shape_for_body_at(body_id, ci)
        if src is None:
            continue
        for k, ref in enumerate(op.edge_refs or []):
            occ = ref.resolve(history, src)
            geo = _chamfer_geometry(src.wrapped, occ) if occ is not None else None
            shape = history._shape_for_body_at(body_id, before_index)
            if geo is not None and shape is not None and \
                    _between(occ_face, shape.wrapped, *geo) and \
                    (_dist_to_face(geo[0], occ_face) or 1e9) < 2.0 * op.distance + 0.5:
                return ChamferFaceOrigin(e.entry_id, k)
    return None


def _chamfer_geometry(src_occ, occ_edge):
    """(edge midpoint, tangent, n1, n2) of a straight edge between two planar
    faces, or None."""
    from build123d import Edge
    faces = _adjacent_faces(src_occ, occ_edge)
    if len(faces) != 2:
        return None
    ns = [_outward_normal(f, src_occ) for f in faces]
    if any(n is None for n in ns):
        return None
    e = Edge(occ_edge)
    return (np.array(tuple(e.position_at(0.5))), np.array(tuple(e.tangent_at(0.5))), *ns)


def _outward_normal(occ_face, solid_occ):
    """Normal pointing out of solid_occ.  _occ_face_normal is the raw surface
    normal, and the face's orientation flag depends on how it was reached
    (an ancestor map and an explorer disagree), so probe: a point just off
    the face along the normal must be outside the solid."""
    from OCP.BRepClass3d import BRepClass3d_SolidClassifier
    from OCP.TopAbs import TopAbs_IN
    from OCP.gp import gp_Pnt
    n = _occ_face_normal(occ_face)
    if n is None:
        return None
    c, _ = _occ_face_props(occ_face)
    p = _point_on_face(c, occ_face)
    if p is None:
        return None
    q = p + 1e-3 * n
    cls = BRepClass3d_SolidClassifier(solid_occ, gp_Pnt(*map(float, q)), 1e-6)
    return -n if cls.State() == TopAbs_IN else n


def _between(occ_face, solid_occ, mid, tan, n1, n2) -> bool:
    """Planar face along the edge whose normal lies strictly inside the arc
    from n1 to n2 (N = a·n1 + b·n2, a, b > 0).  Not "dot > 0 with both": a
    45° chamfer on a 45° corner is perpendicular to one neighbour."""
    N = _outward_normal(occ_face, solid_occ)
    if N is None or abs(float(N @ tan)) > 1e-3:
        return False
    (a, b), *_ = np.linalg.lstsq(np.column_stack([n1, n2]), N, rcond=None)
    return a > 0.02 and b > 0.02


# ---------------------------------------------------------------------------
# edges: "the edge between these two faces"
# ---------------------------------------------------------------------------

def _adjacent_faces(shape_occ, occ_edge) -> list:
    from OCP.TopExp import TopExp
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE
    from OCP.TopoDS import TopoDS
    from OCP.collections import (
        IndexedDataMap_TopoDS_Shape_List_TopoDS_Shape_TopTools_ShapeMapHasher
        as TopTools_IndexedDataMapOfShapeListOfShape)
    m = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndAncestors_s(shape_occ, TopAbs_EDGE, TopAbs_FACE, m)
    for i in range(1, m.Extent() + 1):
        if m.FindKey(i).IsSame(occ_edge):
            out = []
            for f in m.FindFromIndex(i):
                f = TopoDS.Face(f)
                if not any(f.IsSame(g) for g in out):
                    out.append(f)
            return out
    return []


@dataclass
class EdgeOrigin:
    """An edge named through the faces it bounds: with two provenance faces,
    their shared edge; with one (the other is e.g. a fillet surface), the
    edge of that face matching the edge's midpoint and direction recorded
    in the face's line frame."""
    faces:  list         # 1–2 FaceOrigin
    mid:    tuple        # edge midpoint in the anchor's frame (s, w, h)
    tan:    tuple        # edge direction in the anchor's frame (t, w, n)
    # Frame the edge is predicted in: the sketch line that made it (on a cap
    # every line's frame is valid for the FACE, but an edit turns each line
    # differently).  None → faces[0].
    anchor: FaceOrigin | None = None

    def to_dict(self) -> dict:
        d = {"faces": [f.to_dict() for f in self.faces],
             "mid": list(self.mid), "tan": list(self.tan)}
        if self.anchor is not None:
            d["anchor"] = self.anchor.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict | None) -> "EdgeOrigin | None":
        if not d:
            return None
        return cls([face_origin_from_dict(f) for f in d["faces"]],
                   tuple(d["mid"]), tuple(d["tan"]), FaceOrigin.from_dict(d.get("anchor")))

    def faces_apart(self, history, shape) -> bool:
        """Both named faces still exist but no longer share an edge: the
        edge was removed upstream rather than lost by the resolver."""
        if len(self.faces) < 2:
            return False
        a, b = (o.find_in(history, shape)[1] for o in self.faces)
        if a is None or b is None or a.wrapped.IsSame(b.wrapped):
            return False
        other = list(b.edges())
        return not any(e.wrapped.IsSame(o.wrapped) for e in a.edges() for o in other)

    def find_in(self, history, shape):
        """The TopoDS_Edge this names now, or None."""
        frame = (self.anchor or self.faces[0]).frame(history)
        if frame is None:
            return None
        mid = FaceOrigin.world(frame, p=self.mid)
        tan = FaceOrigin.world(frame, v=self.tan)
        found = [o.find_in(history, shape)[1] for o in self.faces]
        if found[0] is None:
            return None
        cands = list(found[0].edges())
        if len(found) == 2 and found[1] is not None and not found[0].wrapped.IsSame(found[1].wrapped):
            other = list(found[1].edges())
            shared = [e for e in cands if any(e.wrapped.IsSame(o.wrapped) for o in other)]
            if shared:
                return min(shared, key=lambda e: _edge_mid_dist(e, mid)).wrapped
            # Both faces are there but no longer meet: something upstream now
            # runs between them (a fillet carried over a shallow crease, seed
            # 9).  The edge is gone; guessing one face's edge picks a smooth one.
            return None
        # One face: its edge running the predicted way, nearest the predicted
        # midpoint — only when unambiguous.  The direction is loose: the face
        # we can't name (a chamfer, a neighbour) may turn under an edit and
        # swing the edge within this face (5.7°, walker seed 14).  A distance
        # tie is broken by an exact direction match: two short edges in a
        # shallow V, 10° apart, sit 0.09mm apart (seed 13).
        scored = sorted(((_edge_mid_dist(e, mid), e) for e in cands
                         if _edge_dir_ok(e, tan, _ONE_FACE_DOT)), key=lambda t: t[0])
        if not scored:
            # The unnamed face turned further (21°, seed 10): accept the
            # nearest edge only when no other edge of the face comes close.
            byd = sorted(((_edge_mid_dist(e, mid), e) for e in cands), key=lambda t: t[0])
            if (byd and (len(byd) == 1 or byd[0][0] < 0.5 * byd[1][0])
                    and _edge_dir_ok(byd[0][1], tan, _ONE_FACE_LOOSE)):
                return byd[0][1].wrapped
            return None
        near = [e for d, e in scored if d < 2 * scored[0][0] + 0.1]
        if len(near) > 1:
            near = [e for e in near if _edge_dir_ok(e, tan)]
        return near[0].wrapped if len(near) == 1 else None


def edge_origin(history, body_id: str, shape_occ, occ_edge,
                before_index: int):
    """Provenance for an edge of body_id, or None if neither it nor its
    faces have any."""
    from build123d import Edge
    rim = rim_origin(history, body_id, occ_edge, before_index)
    if rim is not None:
        return rim
    faces = _adjacent_faces(shape_occ, occ_edge)
    origins = [o for o in (origin_for_face(history, body_id, f, before_index)
                           for f in faces) if o is not None]
    if not origins:
        return revolve_edge_origin(history, body_id, shape_occ, occ_edge, before_index)
    if len(origins) == 1 and len(faces) == 2:
        # The other face is a chamfer's: name it too, so the edge is the two
        # faces' shared edge.  One face alone predicts the edge where it was,
        # running the way it ran, and a chamfer angle edit moves and turns it.
        other = next((f for f in faces
                      if origin_for_face(history, body_id, f, before_index) is None), None)
        cf = (chamfer_origin_for_face(history, body_id, other, before_index)
              if other is not None else None)
        if cf is not None:
            origins.append(cf)
    e = Edge(occ_edge)
    mid, tan = np.array(tuple(e.position_at(0.5))), np.array(tuple(e.tangent_at(0.5)))
    anchor = _edge_anchor(history, origins[0], mid, tan)
    frame = (anchor or origins[0]).frame(history)
    if frame is None:
        return None
    return EdgeOrigin(
        faces  = origins[:2],
        mid    = tuple(round(x, 9) for x in FaceOrigin.local(frame, p=mid)),
        tan    = tuple(round(x, 9) for x in FaceOrigin.local(frame, v=tan)),
        anchor = anchor,
    )


# ---------------------------------------------------------------------------
# rims: a circular edge, named by the circle that made it
# ---------------------------------------------------------------------------
#
# A rim has no straight line to hang a FaceOrigin on, and "the edge near where
# it used to be" fails as soon as the hole changes size: the rim moves further
# than the matcher allows, or a neighbouring rim is now nearer.  But a rim's
# whole geometry can be worked out: from the sketch circle and the extrude
# that made it, or from the rim a chamfer / fillet was cut into.  Finding it
# is then an exact lookup, so it doesn't matter how far it moved, and a rim
# that isn't there is reported as missing rather than swapped for another
# (`authoritative`).

_RIM_TOL = 1e-3      # mm — a rim sits exactly where its circle says


def _edge_circle(occ_edge):
    """(centre, canonical unit axis, radius) of a circular edge, else None."""
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.GeomAbs import GeomAbs_Circle
    try:
        a = BRepAdaptor_Curve(occ_edge)
        if a.GetType() != GeomAbs_Circle:
            return None
        c = a.Circle()
    except Exception:
        return None
    p, d = c.Location(), c.Axis().Direction()
    return (np.array([p.X(), p.Y(), p.Z()]),
            _canonical_axis(np.array([d.X(), d.Y(), d.Z()])), float(c.Radius()))


def _canonical_axis(n):
    n = np.asarray(n, dtype=float)
    for c in n:
        if abs(c) > 1e-6:
            return -n if c < 0 else n
    return n


def _rim_angle(C, n, p) -> float:
    """Angle of point p around the axis (C, n), in a frame fixed by n: the
    part of the circle an arc rim covers, when a rim has been split."""
    u, v = _axis_frame(n)
    r = np.asarray(p, dtype=float) - C
    return float(np.arctan2(r @ v, r @ u))


def _axis_frame(n):
    k = np.eye(3)[int(np.argmin(np.abs(n)))]
    u = np.cross(n, k)
    u /= np.linalg.norm(u)
    return u, np.cross(n, u)


def _is_full_circle_edge(occ_edge) -> bool:
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    a = BRepAdaptor_Curve(occ_edge)
    return abs(a.LastParameter() - a.FirstParameter() - 2.0 * np.pi) < 1e-6


def _find_rim(shape, C, n, R, ang, full):
    """The circular edge of shape on circle (C, n, R), or None.  An arc rim
    that has been split: the piece nearest angle ang.  A full rim has no
    such piece — something now cuts through it, so it's gone."""
    hits = []
    for e in shape.edges():
        c = _edge_circle(e.wrapped)
        if c is not None and abs(float(c[1] @ n)) > _NORMAL_DOT and \
                abs(c[2] - R) < _RIM_TOL and np.linalg.norm(c[0] - C) < _RIM_TOL:
            hits.append(e)
    if full:
        hits = [e for e in hits if _is_full_circle_edge(e.wrapped)]
    if len(hits) <= 1:
        return hits[0].wrapped if hits else None
    u, v = _axis_frame(n)
    mid = C + R * (np.cos(ang) * u + np.sin(ang) * v)
    return min(hits, key=lambda e: _edge_mid_dist(e, mid)).wrapped


class _Rim:
    """A rim origin: circle(history) → (C, n, R) now.  Replay trusts the
    answer, so EdgeRef has no reason to go hunting when it finds no edge."""
    authoritative = True

    def faces_apart(self, history, shape) -> bool:
        return False

    def find_in(self, history, shape):
        circ = self.circle(history)
        if circ is None:
            return None
        return _find_rim(shape, *circ, self.ang, self.full)


@dataclass
class RimOrigin(_Rim):
    """An end of the wall a sketch arc / circle was extruded into."""
    sketch_id:  str
    uid:        str      # the ArcEntity
    extrude_of: str      # entry_id of the extrude / cut
    end:        int      # 0 = the sketch-plane end, 1 = the far end
    ang:        float
    full:       bool

    def to_dict(self) -> dict:
        return {"rim_of": self.sketch_id, "uid": self.uid, "extrude_of": self.extrude_of,
                "end": self.end, "ang": self.ang, "full": self.full}

    @classmethod
    def from_dict(cls, d: dict) -> "RimOrigin":
        return cls(d["rim_of"], d["uid"], d["extrude_of"], int(d["end"]),
                   float(d["ang"]), bool(d["full"]))

    def circle(self, history):
        si, ei = (history.id_to_index(i) for i in (self.sketch_id, self.extrude_of))
        if si is None or ei is None:
            return None
        se = history._entries[si].params.get("sketch_entry")
        op = history._entries[ei].op
        ent = next((e for e in (se.entities if se else ())
                    if getattr(e, "uid", None) == self.uid), None)
        if ent is None or not hasattr(ent, "radius"):
            return None
        return _rim_circle(se, ent, op, self.end)


def _rim_circle(se, arc, op, end):
    """(C, n, R) of one end of the wall op extruded arc into, or None when
    the extrude runs at a slant (the rim isn't that circle)."""
    n = np.asarray(se.plane_normal, dtype=float)
    d = _extrude_dir(op, se)
    if abs(float(d @ n)) < 1.0 - 1e-9:
        return None
    u, v = np.asarray(arc.center, dtype=float)[:2]
    c = (np.asarray(se.plane_origin, dtype=float) + u * np.asarray(se.plane_x_axis, dtype=float)
         + v * np.asarray(se.plane_y_axis, dtype=float))
    return c + (op.start_offset + end * abs(op.distance)) * d, _canonical_axis(n), float(arc.radius)


@dataclass
class BlendRimOrigin(_Rim):
    """A boundary circle of a chamfer / fillet on a rim: that rim offset by
    (dr, dh) in units of the blend's size, so it follows both the rim and
    the size."""
    blend_of: str        # entry_id of the ChamferOp / FaceFilletOp
    edge:     int        # index into that op's edge_refs
    dr:       float
    dh:       float
    ang:      float
    full:     bool

    def to_dict(self) -> dict:
        return {"blend_of": self.blend_of, "edge": self.edge, "dr": self.dr,
                "dh": self.dh, "ang": self.ang, "full": self.full}

    @classmethod
    def from_dict(cls, d: dict) -> "BlendRimOrigin":
        return cls(d["blend_of"], int(d["edge"]), float(d["dr"]), float(d["dh"]),
                   float(d["ang"]), bool(d["full"]))

    def circle(self, history):
        ci = history.id_to_index(self.blend_of)
        op = history._entries[ci].op if ci is not None else None
        refs = getattr(op, "edge_refs", None) or []
        size = _blend_size(op) if self.edge < len(refs) else None
        src = history._shape_for_body_at(op.source_body_id, ci) if size else None
        occ = refs[self.edge].resolve(history, src) if src is not None else None
        base = _edge_circle(occ) if occ is not None else None
        if base is None:
            return None
        C, n, R = base
        return C + self.dh * size * n, n, R + self.dr * size


def _blend_size(op) -> float | None:
    """A chamfer's distance or an (even) fillet's radius."""
    if type(op).__name__ == "ChamferOp":
        return float(op.distance)
    if type(op).__name__ == "FaceFilletOp" and not op.edge_radii:
        return float(op.radius)
    return None


def rim_origin(history, body_id: str, occ_edge, before_index: int):
    """Which circle made this circular edge of body_id, or None.  The most
    recent op that could have wins, as in origin_for_face."""
    from build123d import Edge
    circ = _edge_circle(occ_edge)
    if circ is None:
        return None
    C, n, R = circ
    ang = _rim_angle(C, n, tuple(Edge(occ_edge).position_at(0.5)))
    full = _is_full_circle_edge(occ_edge)
    for e in reversed(history._entries[:max(0, before_index)]):
        if e.error:
            continue
        found = (_blend_rim(history, e, body_id, circ, ang, full)
                 or _extruded_rim(history, e, body_id, circ, ang, full))
        if found is not None:
            return found
    return None


def _blend_rim(history, e, body_id, circ, ang, full):
    op = e.op
    size = _blend_size(op)
    if not size or size <= 0 or op.source_body_id != body_id:
        return None
    C, n, R = circ
    ci = history.id_to_index(e.entry_id)
    src = history._shape_for_body_at(body_id, ci)
    if src is None:
        return None
    for k, ref in enumerate(op.edge_refs or []):
        occ = ref.resolve(history, src)
        base = _edge_circle(occ) if occ is not None else None
        if base is None or abs(float(base[1] @ n)) < _NORMAL_DOT:
            continue
        delta = C - base[0]
        dh = float(delta @ n)
        dr = R - base[2]
        reach = max(abs(dr), abs(dh))
        # Coaxial, displaced, and within reach of this blend.
        if np.linalg.norm(delta - dh * n) < _RIM_TOL and _RIM_TOL < reach <= 3.0 * size:
            return BlendRimOrigin(e.entry_id, k, round(dr / size, 9),
                                  round(dh / size, 9), ang, full)
    return None


def _extruded_rim(history, e, body_id, circ, ang, full):
    op = e.op
    sketch_id = _sketch_id_of(op)
    if sketch_id is None or body_id not in (
            e.body_id, getattr(op, "merge_body_id", None), getattr(op, "cut_body_id", None)):
        return None
    si = history.id_to_index(sketch_id)
    se = history._entries[si].params.get("sketch_entry") if si is not None else None
    C, n, R = circ
    for ent in (se.entities if se else ()):
        if not hasattr(ent, "radius") or not getattr(ent, "uid", None) or \
                abs(float(ent.radius) - R) > _RIM_TOL:
            continue
        for end in (1, 0):
            rim = _rim_circle(se, ent, op, end)
            if rim is not None and np.linalg.norm(rim[0] - C) < _RIM_TOL and \
                    abs(float(rim[1] @ n)) > _NORMAL_DOT:
                return RimOrigin(sketch_id, ent.uid, e.entry_id, end, ang, full)
    return None


def upgrade_to_rim(history, body_id, occ_edge, ref, origin, before_index):
    """origin, or the rim origin of occ_edge if it's a rim saved before they
    existed.  Only while the edge is still where ref says, which means this
    is the edge that was picked and not a neighbour a fuzzy match landed on."""
    if isinstance(origin, _Rim) or occ_edge is None or _edge_circle(occ_edge) is None:
        return origin
    from cad.edge_ref import EdgeRef
    now = EdgeRef.from_occ_edge(occ_edge)
    if now is None or abs(now.length - ref.length) > 0.5 or \
            np.linalg.norm(np.subtract(now.midpoint, ref.midpoint)) > 0.1:
        return origin
    return rim_origin(history, body_id, occ_edge, before_index) or origin


# ---------------------------------------------------------------------------
# revolve edges: an edge of a revolve cap, or the arc a profile vertex swept
# ---------------------------------------------------------------------------

def _rotate(v, d, deg, about=None):
    """Rotate point (about a point on the axis) or vector v about unit d."""
    v = np.asarray(v, dtype=float)
    o = np.zeros(3) if about is None else np.asarray(about, dtype=float)
    r = v - o
    t = np.radians(deg)
    r = (r * np.cos(t) + np.cross(d, r) * np.sin(t)
         + d * float(d @ r) * (1 - np.cos(t)))
    return r + o


def _revolve_axis_now(history, entry_id):
    """(point, unit dir, angle) of a revolve entry as of its replay, or None."""
    ci = history.id_to_index(entry_id)
    if ci is None:
        return None
    op = history._entries[ci].op
    if type(op).__name__ not in ("RevolveOp", "CrossBodyRevolveCutOp") or op.axis_point is None:
        return None
    from cad.op_revolve_thicken import _revolve_axis
    try:
        pt, d = (np.asarray(v, dtype=float) for v in _revolve_axis(op, history))
    except Exception:
        return None
    return pt, d / np.linalg.norm(d), float(op.angle_deg)


@dataclass
class RevolveEdgeOrigin:
    """An edge a revolve made, named through the revolve so it follows angle
    edits (a far-cap edge swings with the cap; a swept arc grows).
      end 0/1 : an edge of that cap (see RevolveCapOrigin); p / t are its
                midpoint / direction turned back onto the profile's plane.
      end None: the arc profile vertex p swept, from the start cap to the
                far cap (t unused)."""
    revolve_of: str
    profile:    int
    end:        int | None
    p:          tuple
    t:          tuple = (0.0, 0.0, 0.0)

    def to_dict(self) -> dict:
        return {"revolve_of": self.revolve_of, "profile": self.profile,
                "end": self.end, "p": list(self.p), "t": list(self.t)}

    @classmethod
    def from_dict(cls, d: dict) -> "RevolveEdgeOrigin":
        return cls(d["revolve_of"], int(d.get("profile", 0)), d.get("end"),
                   tuple(d["p"]), tuple(d.get("t", (0.0, 0.0, 0.0))))

    def faces_apart(self, history, shape) -> bool:
        return False

    def find_in(self, history, shape):
        """The TopoDS_Edge this names now, or None."""
        ax = _revolve_axis_now(history, self.revolve_of)
        if ax is None:
            return None
        pt, d, angle = ax
        if self.end is None:
            return _swept_arc(shape, np.asarray(self.p, dtype=float),
                              _rotate(self.p, d, angle, pt))
        _, face = RevolveCapOrigin(self.revolve_of, self.profile, self.end
                                   ).find_in(history, shape)
        if face is None:
            return None
        turn = angle * self.end
        mid = _rotate(self.p, d, turn, pt)
        tan = _rotate(self.t, d, turn)
        scored = sorted(((_edge_mid_dist(e, mid), e) for e in face.edges()
                         if _edge_dir_ok(e, tan, _ONE_FACE_DOT)), key=lambda x: x[0])
        if not scored or scored[0][0] > max(1e-3, 1e-4 * float(np.linalg.norm(mid - pt))):
            return None
        return scored[0][1].wrapped


def _swept_arc(shape, a, b, tol=1e-4):
    """The edge of shape running between points a and b (either way)."""
    scale = max(1.0, float(np.linalg.norm(a)), float(np.linalg.norm(b)))
    for e in shape.edges():
        ends = _edge_ends(e)
        if ends is None:
            continue
        p0, p1 = ends
        if ((np.linalg.norm(p0 - a) < tol * scale and np.linalg.norm(p1 - b) < tol * scale)
                or (np.linalg.norm(p0 - b) < tol * scale and np.linalg.norm(p1 - a) < tol * scale)):
            return e.wrapped
    return None


def _edge_ends(e):
    try:
        return (np.array(tuple(e.position_at(0.0))), np.array(tuple(e.position_at(1.0))))
    except Exception:
        return None


def revolve_edge_origin(history, body_id: str, shape_occ, occ_edge,
                        before_index: int) -> RevolveEdgeOrigin | None:
    """Provenance for an edge a revolve on body_id made, or None."""
    from build123d import Edge
    e = Edge(occ_edge)
    mid = np.array(tuple(e.position_at(0.5)))
    tan = np.array(tuple(e.tangent_at(0.5)))
    # On a cap: prefer the far cap (the start cap doesn't move with the angle).
    caps = [o for o in (revolve_origin_for_face(history, body_id, f, before_index)
                        for f in _adjacent_faces(shape_occ, occ_edge)) if o is not None]
    caps.sort(key=lambda o: -o.end)
    for o in caps:
        ax = _revolve_axis_now(history, o.revolve_of)
        if ax is None:
            continue
        pt, d, angle = ax
        turn = -angle * o.end
        return RevolveEdgeOrigin(
            o.revolve_of, o.profile, o.end,
            tuple(round(float(x), 9) for x in _rotate(mid, d, turn, pt)),
            tuple(round(float(x), 9) for x in _rotate(tan, d, turn)))
    # A swept arc: one end is the other turned through the angle.
    ends = _edge_ends(e)
    if ends is None:
        return None
    for rev in _revolves_on(history, body_id, before_index):
        ax = _revolve_axis_now(history, rev.entry_id)
        if ax is None or abs(ax[2]) >= 360.0 - 1e-9:
            continue
        pt, d, angle = ax
        scale = max(1.0, *(float(np.linalg.norm(x - pt)) for x in ends))
        for a, b in (ends, ends[::-1]):
            if np.linalg.norm(_rotate(a, d, angle, pt) - b) < 1e-6 * scale:
                return RevolveEdgeOrigin(rev.entry_id, 0, None,
                                         tuple(round(float(x), 9) for x in a))
    return None


def edge_origin_from_dict(d: dict | None):
    """EdgeOrigin or RevolveEdgeOrigin, by shape of the dict."""
    if not d:
        return None
    if "revolve_of" in d:
        return RevolveEdgeOrigin.from_dict(d)
    if "rim_of" in d:
        return RimOrigin.from_dict(d)
    if "blend_of" in d:
        return BlendRimOrigin.from_dict(d)
    return EdgeOrigin.from_dict(d)


def _edge_anchor(history, face_origin, mid, tan) -> FaceOrigin | None:
    """For an edge on a cap (face normal along the sketch normal), the
    sketch line running parallel to it, nearest — the line whose extrude
    made the edge.  None when the face is a wall (its own line is the
    anchor) or no line is parallel."""
    if abs(abs(face_origin.normal[2]) - 1.0) > 1e-6:
        return None
    idx = history.id_to_index(face_origin.sketch_id)
    se = history._entries[idx].params.get("sketch_entry") if idx is not None else None
    if se is None:
        return None
    tan = tan / np.linalg.norm(tan)
    best = None
    for ent in se.entities:
        frame = _line_frame(se, ent)
        if frame is None or not getattr(ent, "uid", None):
            continue
        A, L, t, w, n = frame
        if abs(float(t @ tan)) < _NORMAL_DOT:
            continue
        r = mid - A
        s = float(r @ t) / L
        d = abs(float(r @ w)) + max(0.0, -s, s - 1.0) * L   # off the line's span costs too
        if best is None or d < best[0]:
            best = (d, ent.uid)
    if best is None or best[1] == face_origin.uid:
        return None
    return FaceOrigin(face_origin.sketch_id, best[1], (0.0, 0.0, face_origin.probe[2]),
                      (0.0, 0.0, 1.0), face_origin.cap_of)


def _edge_mid_dist(e, p) -> float:
    return float(np.linalg.norm(np.array(tuple(e.position_at(0.5))) - p))


def _edge_dir_ok(e, tan, dot=_NORMAL_DOT) -> bool:
    try:
        d = np.array(tuple(e.tangent_at(0.5)))
    except Exception:
        return False
    return abs(float(d @ tan)) / (np.linalg.norm(d) * np.linalg.norm(tan)) > dot


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _line_by_uid(se, uid):
    if se is None:
        return None
    for e in se.entities:
        if getattr(e, "uid", None) == uid and _segment(e) is not None:
            return e
    return None


def _segment(ent):
    """(p0, p1) uv of a straight line or a straight (2-point) include —
    the closing side of a profile drawn onto an included edge makes a wall
    too."""
    from cad.sketch import LineEntity, ReferenceEntity
    if isinstance(ent, LineEntity):
        return ent.p0, ent.p1
    if isinstance(ent, ReferenceEntity) and len(ent.points) == 2:
        return ent.points[0], ent.points[1]
    return None


def _line_frame(se, line):
    """(A world, length, t along, w across, n sketch normal) or None."""
    seg = _segment(line)
    if seg is None:
        return None
    p0, p1 = (np.asarray(p, dtype=float) for p in seg)
    o = np.asarray(se.plane_origin, dtype=float)
    x = np.asarray(se.plane_x_axis, dtype=float)
    y = np.asarray(se.plane_y_axis, dtype=float)
    n = np.asarray(se.plane_normal, dtype=float)
    A = o + p0[0] * x + p0[1] * y
    B = o + p1[0] * x + p1[1] * y
    L = float(np.linalg.norm(B - A))
    if L < 1e-9:
        return None
    t = (B - A) / L
    return A, L, t, np.cross(n, t), n


def _sketch_id_of(op):
    """The driving sketch of a sketch extrude / sketch-driven cut, else None."""
    from cad.op_types import ExtrudeOp, CrossBodyCutOp
    if isinstance(op, (ExtrudeOp, CrossBodyCutOp)):
        ids = op.from_sketch_ids
        return ids[0] if ids else None
    return None


def _cap_height(op) -> float:
    """Signed offset of a normal-following extrude's far face (a cut's
    pocket floor) along its sketch normal."""
    # Cuts are exact now; files saved with the old 0.01mm overshoot still
    # resolve, since a cap probe is read relative to this height.
    return op.along_normal * (op.start_offset + abs(op.distance))


def _extrude_dir(op, se):
    from cad.op_types import CrossBodyCutOp
    if op.along_normal and op.direction is not None:
        return op.along_normal * np.asarray(se.plane_normal, dtype=float)
    if op.direction is not None:
        d = np.asarray(op.direction, dtype=float)
        if np.linalg.norm(d) > 1e-12:
            return d / np.linalg.norm(d)
    n = np.asarray(se.plane_normal, dtype=float)
    if isinstance(op, CrossBodyCutOp):
        return -n                      # default: into the body (CrossBodyCutOp.execute)
    return n if op.distance >= 0 else -n


def _point_on_face(p, occ_face):
    """Point of the face nearest p (inside the trimmed face, unlike a
    concave face's centroid)."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
    from OCP.BRepExtrema import BRepExtrema_DistShapeShape
    from OCP.gp import gp_Pnt
    try:
        v = BRepBuilderAPI_MakeVertex(gp_Pnt(*map(float, p))).Vertex()
        dss = BRepExtrema_DistShapeShape(v, occ_face)
        if not dss.IsDone() or dss.NbSolution() < 1:
            return None
        q = dss.PointOnShape2(1)
        return np.array([q.X(), q.Y(), q.Z()])
    except Exception:
        return None


def _dist_to_face(p, occ_face):
    q = _point_on_face(p, occ_face)
    return None if q is None else float(np.linalg.norm(q - p))
