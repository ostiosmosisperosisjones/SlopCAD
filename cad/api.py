"""
cad/api.py

Scripting facade: build and edit SlopCAD models from Python (or an LLM
driving Python), then open the .vc in the GUI.

    from cad.api import Doc
    doc = Doc.new()
    doc.param("w", "40mm")
    s = doc.sketch("XY").rect("w", 20)
    box = doc.extrude(s, "w/4")
    top = doc.sketch(box, ">Z").circle((0, 0), 6)     # UV centred on the face
    doc.extrude(top, 3, cut=True)
    doc.fillet(box, edges=doc.edges(box, face="<Z"), r=1)
    doc.save("box.vc")
    doc.set_param("w", "60mm")       # whole history replays

Every call pushes a normal history entry (same Op params the GUI writes)
and replays from it, so geometry comes from the exact code path a reload
uses — no new geometry code here.  Sizes accept numbers (mm) or parameter
expressions ("w/2"); expressions are stored as bindings and follow
set_param().  Errors raise ApiError with the replay message.

Run a script headless:  python -m cad.api script.py [out.vc]
(the script gets `doc`, a fresh Doc, as a global; out.vc saves it).
"""

from __future__ import annotations

import numpy as np

_AXES = {"X": 0, "Y": 1, "Z": 2}


class ApiError(RuntimeError):
    pass


class Body:
    """Handle to a workspace body."""
    def __init__(self, doc: "Doc", body_id: str):
        self.doc, self.id = doc, body_id

    @property
    def shape(self):
        return self.doc.workspace.current_shape(self.id)

    @property
    def volume(self) -> float:
        s = self.shape
        return float(s.volume) if s is not None else 0.0

    def bbox(self) -> tuple:
        bb = self.shape.bounding_box()
        return ((bb.min.X, bb.min.Y, bb.min.Z), (bb.max.X, bb.max.Y, bb.max.Z))

    def __repr__(self):
        name = self.doc.workspace.bodies[self.id].name
        return f"<Body {name} vol={self.volume:.3f}>"


class Sketch:
    """A sketch being drawn.  Coordinates are plane UV in mm — world
    coordinates on a world plane, centred on the face for a face sketch.
    Sizes may be parameter expressions.  Pushed to history when first consumed (extrude)
    or by Doc.commit_sketch()."""

    def __init__(self, doc: "Doc", se):
        self.doc, self.se = doc, se
        self.entry = None           # HistoryEntry once pushed

    def _val(self, v) -> tuple[float, str | None]:
        if isinstance(v, str):
            return self.doc.eval(v), v
        return float(v), None

    def _con(self, ctype, indices, value=0.0, expr=None):
        from cad.sketch import SketchConstraint
        self.se.constraints.append(SketchConstraint(ctype, indices, value, expr=expr))

    def _add(self, ent) -> int:
        self.se.entities.append(ent)
        return len(self.se.entities) - 1

    def line(self, p0, p1) -> "Sketch":
        from cad.sketch import LineEntity
        self._add(LineEntity(tuple(map(float, p0)), tuple(map(float, p1))))
        return self

    def polyline(self, pts, close: bool = True) -> "Sketch":
        pts = list(pts)
        for a, b in zip(pts, pts[1:] + ([pts[0]] if close else [])):
            self.line(a, b)
        return self

    def rect(self, w, h, center=(0.0, 0.0)) -> "Sketch":
        """Axis-aligned rectangle centred on *center*.  w/h become dimension
        constraints (bound when expressions).  Centred because that's how
        the solver resizes it: no fix constraint exists, so a changed
        dimension grows the shape about its middle."""
        wv, we = self._val(w)
        hv, he = self._val(h)
        cx, cy = map(float, center)
        x, y = cx - wv / 2, cy - hv / 2
        i = len(self.se.entities)
        self.polyline([(x, y), (x + wv, y), (x + wv, y + hv), (x, y + hv)])
        for k, t in enumerate(("horizontal", "vertical", "horizontal", "vertical")):
            self._con(t, (i + k,))
        self._con("distance", (i,),     wv, we)
        self._con("distance", (i + 1,), hv, he)
        return self

    def circle(self, center, d) -> "Sketch":
        """Circle by diameter (constraint; bound when an expression)."""
        from cad.sketch import ArcEntity
        dv, de = self._val(d)
        i = self._add(ArcEntity(tuple(map(float, center)), dv / 2, 0.0, 2 * np.pi))
        self._con("diameter", (i,), dv, de)
        return self

    def constrain(self, ctype: str, *indices, value=0.0) -> "Sketch":
        """Raw constraint: 'parallel', 'perpendicular', 'equal', 'distance',
        ... on entity indices (in drawing order)."""
        v, e = self._val(value)
        self._con(ctype, tuple(indices), v, e)
        return self


class _Signal:
    def emit(self, *a, **k):
        pass


class _SketchFaces:
    """viewport._sketch_faces built on demand: {history index: [(face,
    outer_uvs, hole_uvs), ...]} as viewer/sketch_pick.py builds it."""

    def __init__(self, history):
        self._h = history

    def get(self, i, default=None):
        es = self._h.entries
        se = es[i].params.get("sketch_entry") if 0 <= i < len(es) else None
        if se is None:
            return default
        faces, regions = se.build_faces()
        return [(f, r[0], r[1]) for f, r in zip(faces, regions)] or default


class _Mesh:
    def __init__(self, shape):
        self.occt_faces = list(shape.faces())


class _Meshes:
    def __init__(self, workspace):
        self._ws = workspace

    def get(self, body_id, default=None):
        shape = self._ws.current_shape(body_id) if body_id else None
        return _Mesh(shape) if shape is not None else default


class _CommitHost:
    """Just enough viewport for an op's commit() — the GUI's own commit
    path, run headless (no meshes, nothing selected)."""

    def __init__(self, doc: "Doc"):
        self.workspace = doc.workspace
        self.history = doc.history
        self.history_changed = _Signal()
        self._pending_op_error = None
        self._selected_sketch_entry = None
        self._selected_sketch_face = None
        self._sketch_faces = _SketchFaces(doc.history)
        self._meshes = _Meshes(doc.workspace)

    def _post_push_cascade(self, body_id=None):
        if self.history.is_mid_history:
            ok, err, _ = self.history.replay_from(self.history.cursor)
            if not ok:
                print(f"[Cascade] Replay after insert: {err}")
            self.history.seek(len(self.history.entries) - 1)

    def _rebuild_bodies(self, body_ids):
        pass

    def _rebuild_body_mesh(self, body_id):
        pass

    def _rebuild_all_meshes(self):
        pass

    def update(self):
        pass


class Doc:
    def __init__(self, workspace, history):
        self.workspace, self.history = workspace, history

    # ------------------------------------------------------------------ io
    @classmethod
    def new(cls) -> "Doc":
        from cad.workspace import Workspace
        from cad.history import History
        ws, hist = Workspace(), History()
        ws.history, hist._workspace = hist, ws
        return cls(ws, hist)

    @classmethod
    def open(cls, path: str) -> "Doc":
        from cad import serializer
        with open(path, "rb") as f:
            ws, hist, _ = serializer.load(f.read())
        warnings = serializer.replay_all(ws, hist)
        doc = cls(ws, hist)
        doc.warnings = warnings
        return doc

    def save(self, path: str) -> None:
        from cad import serializer
        with open(path, "wb") as f:
            f.write(serializer.save(self.workspace, self.history))

    def export_step(self, path: str, body: Body | None = None) -> None:
        from build123d import export_step, Compound
        from cad.importer import _heal
        shapes = ([body.shape] if body is not None
                  else [s for _, s in self.workspace.all_current_shapes()])
        export_step(Compound([Compound(_heal(s.wrapped)) for s in shapes if s is not None]),
                    path)

    def export_stl(self, path: str, body: Body) -> None:
        from build123d import export_stl
        export_stl(body.shape, path)

    # ---------------------------------------------------------- parameters
    def param(self, name: str, expr: str, kind: str = "length",
              unit: str = "mm") -> None:
        """Define or change a parameter; replays everything that uses it."""
        from cad.params import Param, ParamError
        try:
            ok, err, _ = self.history.set_param(Param(name, str(expr), kind, unit))
        except ParamError as ex:
            raise ApiError(str(ex)) from ex
        if not ok:
            raise ApiError(err)

    set_param = param

    def eval(self, expr: str, unit: str = "mm") -> float:
        from cad.params import eval_binding
        try:
            return eval_binding({"expr": expr, "unit": unit},
                                self.history.param_values())
        except ValueError as ex:
            raise ApiError(f"{expr!r}: {ex}") from ex

    # -------------------------------------------------------------- query
    @property
    def bodies(self) -> list[Body]:
        return [Body(self, bid) for bid in self.workspace.bodies]

    def summary(self) -> str:
        lines = [f"params: " + ", ".join(
            f"{p['name']}={p['expr']}" for p in self.history.params.to_list())]
        for i, e in enumerate(self.history.entries):
            flag = f"  ERROR: {e.error_msg}" if e.error else ""
            lines.append(f"{i:3d} {e.label}{flag}")
        lines += [repr(b) for b in self.bodies]
        return "\n".join(lines)

    @staticmethod
    def _pick(items, sel, key):
        """sel: int index, or '>Z' / '<X' (max/min centroid along axis)."""
        if isinstance(sel, int):
            return sel
        if not (isinstance(sel, str) and len(sel) == 2 and sel[0] in "<>"
                and sel[1].upper() in _AXES):
            raise ApiError(f"bad selector {sel!r}: use an index or '>Z', '<X', ...")
        ax = _AXES[sel[1].upper()]
        vals = [key(it)[ax] for it in items]
        return int(np.argmax(vals) if sel[0] == ">" else np.argmin(vals))

    def face_index(self, body: Body, sel) -> int:
        """Index of a planar face: int, or '>Z'-style among faces whose
        normal is along that axis."""
        from build123d import Plane
        faces = list(body.shape.faces())
        if isinstance(sel, int):
            return sel
        ax = _AXES[sel[1].upper()]
        cands = []
        for i, f in enumerate(faces):
            try:
                n = Plane(f).z_dir
            except Exception:
                continue                     # not planar
            if abs((n.X, n.Y, n.Z)[ax]) > 0.999:
                cands.append(i)
        if not cands:
            raise ApiError(f"no planar face normal to {sel[1]}")
        return cands[self._pick([faces[i] for i in cands], sel,
                                lambda f: tuple(f.center()))]

    def edges(self, body: Body, face=None, parallel: str | None = None) -> list:
        """Edges of *body*: all, of one face (selector), and/or parallel to
        an axis.  Returns build123d Edges for fillet()/chamfer()."""
        shape = body.shape
        edges = (list(shape.faces()[self.face_index(body, face)].edges())
                 if face is not None else list(shape.edges()))
        if parallel:
            ax = _AXES[parallel.upper()]
            out = []
            for e in edges:
                d = np.subtract(tuple(e.position_at(1)), tuple(e.position_at(0)))
                n = np.linalg.norm(d)
                if e.geom_type.name == "LINE" and n > 1e-9 and abs(d[ax] / n) > 0.999:
                    out.append(e)
            edges = out
        return edges

    # ------------------------------------------------------------ history
    def _commit(self, entry):
        """Replay from *entry*; raise if it (or anything after it) failed."""
        i = self.history.id_to_index(entry.entry_id)
        ok, err, _ = self.history.replay_from(i)
        if not ok:
            raise ApiError(err)
        return entry

    @staticmethod
    def _bind(params: dict, **fields) -> dict:
        """Numbers go in params; expression strings become bindings
        (replay evaluates them before the op runs)."""
        for k, v in fields.items():
            if isinstance(v, str):
                params.setdefault("exprs", {})[k] = {"expr": v, "unit": "mm"}
                params[k] = 0.0          # filled by _apply_bindings on replay
            else:
                params[k] = float(v)
        return params

    def sketch(self, on="XY", face=None) -> Sketch:
        """New sketch on a world plane ('XY'|'XZ'|'YZ') or on a planar face
        of a body (doc.sketch(body, '>Z'))."""
        from build123d import Plane
        from cad.sketch import SketchEntry
        from cad.plane_ref import WorldPlaneSource, FacePlaneSource
        from cad.face_ref import FaceRef
        if isinstance(on, Body):
            fi = self.face_index(on, face if face is not None else ">Z")
            f = list(on.shape.faces())[fi]
            p = Plane(f)
            body_id = on.id
            src = FacePlaneSource.for_face(self.history, on.id, f, self.history.cursor + 1)
        else:
            p = {"XY": Plane.XY, "XZ": Plane.XZ, "YZ": Plane.YZ}[on.upper()]
            body_id, src, fi = "", WorldPlaneSource(on.upper()), -1
        v = lambda d: np.array([d.X, d.Y, d.Z], dtype=float)
        se = SketchEntry(plane_origin=v(p.origin), plane_x_axis=v(p.x_dir),
                         plane_y_axis=v(p.y_dir), plane_normal=v(p.z_dir),
                         entities=[], body_id=body_id, face_idx=fi,
                         plane_source=src)
        return Sketch(self, se)

    def commit_sketch(self, sk: Sketch):
        if sk.entry is None:
            if sk.se.constraints and not sk.se.apply_all_constraints():
                raise ApiError("sketch constraints don't solve")
            body_id = sk.se.body_id
            before = self.workspace.current_shape(body_id) if body_id else None
            sk.entry = self.history.push(
                label="sketch", operation="sketch",
                params={"sketch_entry": sk.se}, body_id=body_id,
                face_ref=None, shape_before=before, shape_after=before)
            self._commit(sk.entry)
        return sk.entry

    def extrude(self, sk: Sketch, distance, cut: bool = False,
                into: Body | None = None, regions: list[int] | None = None) -> Body:
        """Extrude the sketch's closed regions (all, or *regions* indices),
        committed exactly as the GUI's extrude panel commits it
        (cad.extrude_commit, through the op's own commit path):

          face sketch, no into   "Merge with" the sketch's body (ExtrudeOp)
          world sketch, no into  "New Body" (ExtrudeOp force_new_body)
          into=B                 "Merge with" B
          cut=True, into=B       Cut with B picked (CrossBodyCutOp)
          cut=True, no into      Cut with no target: one CrossBodyCutOp per
                                 body the tool hits (the sketch's body, if
                                 any, is the one returned)

        The direction is the sketch normal, reversed for a cut (the panel
        flips it when Cut is picked).  *distance* is a number or a parameter
        expression; a negative number extrudes the other way (the panel's
        ⇅, a positive distance along -normal)."""
        from cad.extrude_commit import extrude_op, cut_all_intersecting, NEW_BODY
        from cad.params import make_binding
        from cad.profile_source import SketchProfile
        se = sk.se
        sk_entry = self.commit_sketch(sk)
        faces, _ = se.build_faces()
        if not faces:
            raise ApiError("sketch has no closed regions")
        dist = self.eval(distance) if isinstance(distance, str) else float(distance)
        flip = not cut and not isinstance(distance, str) and dist < 0
        dist = abs(dist) if flip else dist       # the panel's ⇅
        if dist <= 0:
            raise ApiError(f"extrude distance must be positive, got {dist:g}")
        extra = ({"exprs": {"distance": make_binding(distance, "mm")}}
                 if isinstance(distance, str) else {})
        n = np.cross(np.asarray(se.plane_x_axis, float), np.asarray(se.plane_y_axis, float))
        n = n / np.linalg.norm(n)
        direction = -n if cut != flip else n
        profiles = [SketchProfile(sketch_id=sk_entry.entry_id,
                                  face_indices=None if regions is None else list(regions))]
        host = _CommitHost(self)
        before = {e.entry_id for e in self.history.entries}
        if cut and into is None:
            if not cut_all_intersecting(host, -dist, direction, profiles, extra):
                raise ApiError("cut: the tool doesn't cut any body")
        else:
            merge = into.id if into is not None else (se.body_id or NEW_BODY)
            op = extrude_op(profiles, -dist if cut else dist, direction, merge, extra)
            op.commit(host, extra or None)
        new = [e for e in self.history.entries if e.entry_id not in before]
        if not new:
            raise ApiError("extrude pushed no entry")
        bad = [e for e in new if e.error]
        if bad:
            raise ApiError(bad[0].error_msg or f"{bad[0].label} failed")
        self._commit(new[0])
        main = new[0]
        if cut:
            target = into.id if into is not None else (se.body_id or main.body_id)
        elif merge == NEW_BODY:
            target = (main.params.get("child_body_ids") or [main.body_id])[0]
        else:
            target = merge
        return Body(self, target)

    def _edge_op(self, operation, body: Body, edges, faces, **fields):
        from cad.edge_ref import EdgeRef
        shape = body.shape
        refs = [EdgeRef.for_edge(self.history, body.id, e.wrapped, shape.wrapped,
                                 self.history.cursor + 1) for e in (edges or [])]
        if any(r is None for r in refs):
            raise ApiError(f"{operation}: could not fingerprint an edge")
        face_idx = [self.face_index(body, f) for f in (faces or [])]
        if not refs and not face_idx:
            raise ApiError(f"{operation}: give edges= or faces=")
        from cad.op_types import Op
        op = Op.from_params(operation, {"source_body_id": body.id,
                                        "face_indices": face_idx})
        op.edge_refs = refs
        op.edge_indices = list(range(len(refs)))   # aligned with refs
        from cad.face_ref import bake_face_picks
        op.face_refs = bake_face_picks(self.history, body.id, shape, face_idx,
                                       self.history.cursor + 1)
        params = self._bind(op.to_params(), **fields)
        entry = self.history.push(
            label=operation, operation=operation, params=params,
            body_id=body.id, face_ref=None,
            shape_before=shape, shape_after=None)
        self._commit(entry)
        return body

    def fillet(self, body: Body, edges=None, faces=None, r=1.0) -> Body:
        return self._edge_op("fillet", body, edges, faces, radius=r)

    def chamfer(self, body: Body, edges=None, faces=None, d=1.0) -> Body:
        return self._edge_op("chamfer", body, edges, faces, distance=d)

    def _push_body_op(self, op, operation: str, cut, into, fields=None) -> Body:
        """Push a SketchLoftOp-family op: a new body (cut/into None), or cut
        from / fused into a body.  *fields* bind as in _bind."""
        if into is not None:
            op.merge_body_id = into.id
        if cut is not None:
            op.cut_body_id = cut.id
            operation += "_cut"
        params = self._bind(op.to_params(), **(fields or {}))
        target = cut or into
        if target is None:
            # New body: same shape the GUI's finalize leaves behind.
            body = self.workspace.add_body(self.workspace.next_part_name(), None)
            params["child_body_ids"] = [body.id]
            entry = self.history.push(
                label=operation, operation=operation, params=params,
                body_id=body.id, face_ref=None, shape_before=None, shape_after=None)
            body.created_at_entry_id = entry.entry_id
            self._commit(entry)
            return Body(self, body.id)
        entry = self.history.push(
            label=operation, operation=operation, params=params,
            body_id=target.id, face_ref=None,
            shape_before=target.shape, shape_after=None)
        self._commit(entry)
        return target

    def _profile(self, spec, region: int | None = None):
        """A ProfileSource from an API profile spec: a Sketch (its only
        region, or *region*), (sketch, region), or (body, face) with face an
        index or a '>Z'-style planar selector."""
        from cad.profile_source import SketchProfile, BodyFaceProfile, make_face_ref
        if isinstance(spec, tuple) and len(spec) == 2:
            spec, sel = spec
            if isinstance(spec, Sketch):
                region = sel
            else:
                fi = self.face_index(spec, sel)
                face = list(spec.shape.faces())[fi]
                return BodyFaceProfile(body_id=spec.id, face_idx=fi,
                                       face_ref=make_face_ref(face))
        if isinstance(spec, Sketch):
            entry = self.commit_sketch(spec)
            return SketchProfile(sketch_id=entry.entry_id,
                                 face_indices=None if region is None else [int(region)])
        raise ApiError(f"bad profile {spec!r}: use a Sketch, (sketch, region) "
                       f"or (body, face)")

    def loft(self, profiles: list, ruled: bool = False, continuity: str = "C1",
             cut: Body | None = None, into: Body | None = None) -> Body:
        """Loft through *profiles* in order — each a Sketch (its only region),
        (sketch, region) or (body, face).  New body, or fused into *into* /
        cut from *cut*."""
        from cad.op_loft import SketchLoftOp
        op = SketchLoftOp(from_profiles=[self._profile(p) for p in profiles],
                          ruled=ruled, continuity=continuity)
        op._bake_profiles(self.history, self.history.cursor + 1)
        return self._push_body_op(op, "loft", cut, into)

    def sweep(self, profile, path: Sketch, region: int | None = None,
              edges: list[int] | None = None, cut: Body | None = None,
              into: Body | None = None, frenet: bool = True) -> Body:
        """Sweep one profile — a Sketch's closed region (the only one, or
        *region*) or a (body, face) — along *path*'s drawn entities (all, or
        the *edges* indices), or along a helix entry (doc.helix).  New body,
        or fused into *into* / cut from *cut*."""
        from cad.op_sweep import SketchSweepOp, HELIX_PATH
        prof = self._profile(profile, region)
        if isinstance(path, Sketch):
            path_entry = self.commit_sketch(path)
            ents = path.se.entities
            idx = list(range(len(ents))) if edges is None else list(edges)
            uids = [ents[i].uid for i in idx]
        else:
            path_entry, uids = path, [HELIX_PATH]
        op = SketchSweepOp(from_profiles=[prof],
                           path_sketch_id=path_entry.entry_id,
                           path_uids=uids, frenet=frenet)
        op._bake_profiles(self.history, self.history.cursor + 1)
        return self._push_body_op(op, "sweep", cut, into)

    def coil(self, on, circle: int = 0, height=20.0, turns=5.0,
             size=1.0, section: str = "circle", left_hand: bool = False,
             flip: bool = False, cut: Body | None = None,
             into: Body | None = None, face: int | None = None) -> Body:
        """Helical coil around a Sketch's *circle* (entity index: its centre
        + the sketch normal are the axis, its diameter the coil's) or a
        cylindrical face of a Body (*face* index; the first cylindrical one
        by default).  height / turns / size (section diameter or side) take
        expressions."""
        from cad.op_coil import CoilOp
        if isinstance(on, Body):
            from cad.op_helix import cylinder_axis
            from cad.face_ref import AnyFaceRef
            faces = list(on.shape.faces())
            if face is None:
                face = next((i for i, f in enumerate(faces)
                             if cylinder_axis(f.wrapped) is not None), None)
                if face is None:
                    raise ApiError("body has no cylindrical face")
            op = CoilOp(body_id=on.id,
                        face_ref=AnyFaceRef.for_face(self.history, on.id, faces[face].wrapped,
                                                     self.history.cursor + 1),
                        section=section, left_hand=left_hand, flip=flip)
        else:
            entry = self.commit_sketch(on)
            op = CoilOp(circle_sketch_id=entry.entry_id,
                        circle_uid=on.se.entities[circle].uid, section=section,
                        left_hand=left_hand, flip=flip)
        return self._push_body_op(op, "coil", cut, into,
                                  dict(height=height, turns=turns, section_size=size))

    def helix(self, on, face: int | None = None, circle: int = 0, height=0.0,
              turns=5.0, pitch=0.0, left_hand: bool = False,
              flip: bool = False):
        """Helix curve around a cylindrical face of a Body (*face* index; the
        first cylindrical one by default) or a Sketch's *circle*.  height 0
        = the face's full length; pitch > 0 drives (else turns).  height /
        pitch take expressions.  Returns the history entry — sweep along
        it with doc.sweep(profile, path=entry)."""
        from cad.op_helix import HelixOp, cylinder_axis
        from cad.face_ref import AnyFaceRef
        if isinstance(on, Body):
            faces = list(on.shape.faces())
            if face is None:
                face = next((i for i, f in enumerate(faces)
                             if cylinder_axis(f.wrapped) is not None), None)
                if face is None:
                    raise ApiError("body has no cylindrical face")
            op = HelixOp(body_id=on.id,
                         face_ref=AnyFaceRef.for_face(self.history, on.id, faces[face].wrapped,
                                                      self.history.cursor + 1))
        else:
            entry = self.commit_sketch(on)
            op = HelixOp(circle_sketch_id=entry.entry_id,
                         circle_uid=on.se.entities[circle].uid)
        op.turns, op.left_hand, op.flip = float(turns), left_hand, flip
        params = self._bind(op.to_params(), height=height, pitch=pitch)
        entry = self.history.push(label="helix", operation="helix", params=params,
                                  body_id=None, face_ref=None,
                                  shape_before=None, shape_after=None)
        return self._commit(entry)

    def _transform(self, body: Body, copy: bool, fields: dict, **kw) -> Body:
        from cad.op_transform import TransformOp
        op = TransformOp(source_body_id=body.id, copy=copy, **kw)
        params = self._bind(op.to_params(), **fields)
        if "angle_deg" in params.get("exprs", {}):
            params["exprs"]["angle_deg"]["unit"] = "deg"
        if not copy:
            entry = self.history.push(
                label="transform", operation="transform", params=params,
                body_id=body.id, face_ref=None, shape_before=body.shape,
                shape_after=None)
            self._commit(entry)
            return body
        new = self.workspace.add_body(f"{self.workspace.bodies[body.id].name} copy", None)
        params["child_body_ids"] = [new.id]
        entry = self.history.push(
            label="transform", operation="transform", params=params,
            body_id=new.id, face_ref=None, shape_before=None, shape_after=None)
        new.created_at_entry_id = entry.entry_id
        self._commit(entry)
        return Body(self, new.id)

    def move(self, body: Body, dx=0.0, dy=0.0, dz=0.0, copy: bool = False) -> Body:
        """Translate *body* (or, copy=True, a new copy of it).  Distances
        take expressions (a bound one must evaluate >= 0)."""
        return self._transform(body, copy, dict(dx=dx, dy=dy, dz=dz))

    def rotate(self, body: Body, angle, axis="Z", edge=None, flip: bool = False,
               copy: bool = False) -> Body:
        """Rotate *body* by *angle* degrees about a world axis ('X'|'Y'|'Z',
        through the origin) or one of its straight / circular *edge*s."""
        ref = None
        if edge is not None:
            from cad.edge_ref import EdgeRef
            ref = EdgeRef.for_edge(self.history, body.id, edge.wrapped,
                                   body.shape.wrapped, self.history.cursor + 1)
            axis = "edge"
        return self._transform(body, copy, dict(angle_deg=angle), mode="rotate",
                               axis=axis, axis_edge_ref=ref, flip_axis=flip)

    def shell(self, body: Body, open=(), t=1.0, outside: bool = False) -> Body:
        """Hollow *body* into walls *t* thick; *open* faces (selectors) are
        removed.  No open faces = a closed hollow."""
        from cad.op_shell import ShellOp
        from cad.face_ref import AnyFaceRef
        shape = body.shape
        idx = [self.face_index(body, f) for f in open]
        faces = list(shape.faces())
        op = ShellOp(source_body_id=body.id, face_indices=idx, thickness=1.0,
                     outside=outside,
                     face_refs=[AnyFaceRef.for_face(self.history, body.id, faces[i].wrapped,
                                                    len(self.history._entries))
                                for i in idx])
        params = self._bind(op.to_params(), thickness=t)
        entry = self.history.push(
            label="shell", operation="shell", params=params,
            body_id=body.id, face_ref=None, shape_before=shape, shape_after=None)
        self._commit(entry)
        return body


def _main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    doc = Doc.new()
    src = open(argv[1]).read()
    exec(compile(src, argv[1], "exec"), {"doc": doc, "Doc": Doc, "__name__": "__script__"})
    if len(argv) > 2:
        doc.save(argv[2])
    print(doc.summary())
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_main(sys.argv))
