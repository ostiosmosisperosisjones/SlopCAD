"""
cad/operations/sweep.py

Sweep a planar profile face along a path of edges (BRepOffsetAPI_MakePipe),
optionally cutting it from / fusing it into a target.  Runs in a kernel
worker.
"""

from __future__ import annotations

from cad.kernel_pool import offload


def chain_wire(edges: list, near=None):
    """A TopoDS_Wire through *edges* in path order, whatever order and
    direction they were picked in, starting at the free end nearest *near*
    (the profile) when given.  Raises if they don't form one chain."""
    import numpy as np
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeWire
    from OCP.BRep import BRep_Tool
    from OCP.TopExp import TopExp

    def ends(e):
        a, b = TopExp.FirstVertex_s(e), TopExp.LastVertex_s(e)
        p, q = BRep_Tool.Pnt_s(a), BRep_Tool.Pnt_s(b)
        return np.array([p.X(), p.Y(), p.Z()]), np.array([q.X(), q.Y(), q.Z()])

    if not edges:
        raise RuntimeError("Sweep path has no edges")
    left = [getattr(e, "wrapped", e) for e in edges]
    tol = 1e-3
    def free(p, e):
        """No other edge ends at p."""
        return not any(min(np.linalg.norm(p - q) for q in ends(o)) < tol
                       for o in left if o is not e)
    # Open path: start at an edge with a free end and walk away from it.
    # Closed loop: start anywhere.
    start, tip = left[0], ends(left[0])[1]
    best = None
    for e in left:
        for a, b in (ends(e), ends(e)[::-1]):
            if free(a, e):
                d = 0.0 if near is None else float(np.linalg.norm(a - near))
                if best is None or d < best:
                    best, start, tip = d, e, b
    order = [start]
    left.remove(start)
    while left:
        nxt = next((e for e in left
                    if min(np.linalg.norm(tip - q) for q in ends(e)) < tol), None)
        if nxt is None:
            raise RuntimeError("Sweep path edges don't connect into one chain")
        p, q = ends(nxt)
        tip = q if np.linalg.norm(tip - p) < tol else p
        order.append(nxt)
        left.remove(nxt)
    mk = BRepBuilderAPI_MakeWire()
    for e in order:
        mk.Add(e)
        if not mk.IsDone():
            raise RuntimeError("Sweep path edges don't connect into one chain")
    return mk.Wire()


def _pipe(wire, profile_wire, frenet: bool, fixed_ax2, binormal=None):
    """Solid of one closed profile wire swept along *wire*.  PipeShell, not
    MakePipe: MakePipe drops everything past a sharp path corner.
    *binormal* (a direction) overrides the mode: the profile keeps its angle
    to that direction — a helix's axis, so a thread profile doesn't roll."""
    from OCP.BRepOffsetAPI import BRepOffsetAPI_MakePipeShell
    from OCP.BRepBuilderAPI import BRepBuilderAPI_TransitionMode
    from OCP.gp import gp_Dir
    mk = BRepOffsetAPI_MakePipeShell(wire)
    if binormal is not None:
        mk.SetMode(gp_Dir(*binormal))
    elif frenet:
        mk.SetMode(False)                            # corrected Frenet
    else:
        mk.SetMode(fixed_ax2)                        # profile keeps its orientation
    mk.SetTransitionMode(BRepBuilderAPI_TransitionMode.BRepBuilderAPI_RightCorner)
    mk.Add(profile_wire)
    mk.Build()
    if not mk.IsDone() or not mk.MakeSolid():
        raise RuntimeError("Sweep failed — OCCT couldn't build the pipe")
    return mk.Shape()


def helix_from(edge, point):
    """The helix *edge* restarted at its point nearest *point* (the
    profile), running on toward whichever end leaves more of it — so a
    profile sketched anywhere along the helix (not just at its start, which
    can be the far end of the face) sweeps from where it is."""
    from OCP.BRep import BRep_Tool
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
    from OCP.GCPnts import GCPnts_AbscissaPoint
    from OCP.GeomAdaptor import GeomAdaptor_Curve
    from OCP.GeomAPI import GeomAPI_ProjectPointOnCurve
    from OCP.gp import gp_Pnt
    e = getattr(edge, "wrapped", edge)
    u0, u1 = BRep_Tool.Range_s(e)
    crv = BRep_Tool.Curve_s(e, 0.0, 1.0)
    if not e.Location().IsIdentity():        # build123d places the helix by Location
        crv = crv.Transformed(e.Location().Transformation())
    pr = GeomAPI_ProjectPointOnCurve(gp_Pnt(*map(float, point)), crv, u0, u1)
    if pr.NbPoints() == 0:
        return e
    u = pr.LowerDistanceParameter()
    ad = GeomAdaptor_Curve(crv)
    before = GCPnts_AbscissaPoint.Length_s(ad, u0, u)
    after = GCPnts_AbscissaPoint.Length_s(ad, u, u1)
    if after >= before:
        return e if u - u0 < 1e-9 else BRepBuilderAPI_MakeEdge(crv, u, u1).Edge()
    from OCP.TopoDS import TopoDS
    return TopoDS.Edge(BRepBuilderAPI_MakeEdge(crv, u0, u).Edge().Reversed())


def sweep_solid(face, path_edges: list, frenet: bool = True, binormal=None):
    """The profile *face* (holes included) swept along the path; a
    build123d Solid."""
    import numpy as np
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCP.gp import gp_Ax2, gp_Pnt, gp_Dir
    from build123d import Face, Compound
    f = Face(getattr(face, "wrapped", face))
    c, n = f.center(), f.normal_at()
    if binormal is not None and len(path_edges) == 1:     # a helix
        path_edges = [helix_from(path_edges[0], (c.X, c.Y, c.Z))]
    wire = chain_wire(path_edges, near=np.array([c.X, c.Y, c.Z]))
    ax2 = gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(n.X, n.Y, n.Z))
    shape = _pipe(wire, f.outer_wire().wrapped, frenet, ax2, binormal)
    for hole in f.inner_wires():
        cut = BRepAlgoAPI_Cut(shape, _pipe(wire, hole.wrapped, frenet, ax2, binormal))
        if not cut.IsDone():
            raise RuntimeError("Sweep failed cutting a profile hole")
        shape = cut.Shape()
    if not BRepCheck_Analyzer(shape).IsValid():
        raise RuntimeError("Sweep made an invalid solid — the path may turn "
                           "tighter than the profile allows")
    solids = Compound(shape).solids()
    if not solids:
        raise RuntimeError("Sweep made no solid (is the profile closed?)")
    if sum(s.volume for s in solids) < 1e-9:
        raise RuntimeError("Sweep made an empty solid — the profile lies "
                           "along the path; sketch it across the path")
    return solids[0] if len(solids) == 1 else Compound(children=list(solids))


@offload()
def sweep_task(faces, path_edges, frenet=True, target=None, mode=None,
               binormal=None):
    """The swept solid, or (mode "cut" / "fuse") target with it cut / fused."""
    from cad.solid_ref import as_compound
    from cad.operations.boolean import bool_op
    part = sweep_solid(faces[0], path_edges, frenet, binormal)
    if mode is None:
        return part
    out = bool_op(mode, [target], [part])      # already in the worker: runs inline
    if out is None:
        raise RuntimeError(f"Boolean {mode} failed")
    return as_compound(out)
