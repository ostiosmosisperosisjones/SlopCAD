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
        """Extrude the sketch's closed regions (all, or *regions* indices).
        Sketch on a body face → adds to (or cuts, cut=True) that body.
        World-plane sketch → new body, or fused into / cut from *into*."""
        from cad.op_types import SketchExtrudeOp
        se = sk.se
        sk_entry = self.commit_sketch(sk)
        faces, regions_uv = se.build_faces()
        if not faces:
            raise ApiError("sketch has no closed regions")
        idx = list(range(len(faces))) if regions is None else list(regions)
        bounds = se.build_face_boundaries()
        op = SketchExtrudeOp(
            from_sketch_id=sk_entry.entry_id, distance=1.0, face_indices=idx,
            face_centroids=[np.array(regions_uv[i][0]).mean(axis=0).tolist() for i in idx],
            face_boundaries=[sorted(bounds[i]) for i in idx])
        if into is not None and not se.body_id and not cut:
            op.merge_body_id = into.id
        params = self._bind(op.to_params(), distance=distance)   # sign = operation
        target = se.body_id or (into.id if into is not None else None)
        if target is None:
            if cut:
                raise ApiError("cut from a world-plane sketch needs into=body")
            target = self.workspace.add_body(self.workspace.next_part_name(), None).id
            sk_entry.body_id = se.body_id = target
        entry = self.history.push(
            label="cut" if cut else "extrude",
            operation="cut" if cut else "extrude", params=params,
            body_id=target, face_ref=None,
            shape_before=self.workspace.current_shape(target), shape_after=None)
        self._commit(entry)
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

    def sweep(self, profile: Sketch, path: Sketch, region: int | None = None,
              edges: list[int] | None = None, cut: Body | None = None,
              into: Body | None = None, frenet: bool = True) -> Body:
        """Sweep one closed region of *profile* (the only one, or *region*)
        along *path*'s drawn entities (all, or the *edges* indices), or
        along a helix entry (doc.helix).  New body, or fused into *into* /
        cut from *cut*."""
        from cad.op_sweep import SketchSweepOp, HELIX_PATH
        sk_entry = self.commit_sketch(profile)
        if isinstance(path, Sketch):
            path_entry = self.commit_sketch(path)
            ents = path.se.entities
            idx = list(range(len(ents))) if edges is None else list(edges)
            uids = [ents[i].uid for i in idx]
        else:
            path_entry, uids = path, [HELIX_PATH]
        op = SketchSweepOp(from_profiles=[(sk_entry.entry_id, region)],
                           path_sketch_id=path_entry.entry_id,
                           path_uids=uids, frenet=frenet)
        return self._push_body_op(op, "sweep", cut, into)

    def coil(self, sk: Sketch, circle: int = 0, height=20.0, turns=5.0,
             size=1.0, section: str = "circle", left_hand: bool = False,
             flip: bool = False, cut: Body | None = None,
             into: Body | None = None) -> Body:
        """Helical coil around the sketch's *circle* (entity index): its
        centre + the sketch normal are the axis, its diameter the coil's.
        height / turns / size (section diameter or side) take expressions."""
        from cad.op_coil import CoilOp
        entry = self.commit_sketch(sk)
        op = CoilOp(circle_sketch_id=entry.entry_id,
                    circle_uid=sk.se.entities[circle].uid, section=section,
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
                         face_ref=AnyFaceRef.from_occ_face(faces[face].wrapped))
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
                     face_refs=[AnyFaceRef.from_occ_face(faces[i].wrapped) for i in idx])
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
