"""
cad/sketch_tools/include.py

IncludeTool — project selected faces / edges / vertices onto the sketch
plane as ReferenceEntity instances.

Every include carries a parametric EdgeSource so that the replay system can
re-project the geometry when upstream operations change.  A face comes in as
one include per boundary edge: those are what constraints and regions name,
and each follows its own edge.

Unlike drawing tools, Include is a one-shot action (triggered by pressing I),
not a persistent mode.  It is not in TOOLS and does not subclass BaseTool.
Call IncludeTool.apply_with_history(sketch, selection, meshes, history) from
the viewport.
"""

from __future__ import annotations
import numpy as np


def _is_circle_edge_on(occ_edge, sketch_plane_normal: np.ndarray,
                        tol: float = 1e-3) -> bool:
    """True if occ_edge is a circle whose normal is perpendicular to the sketch plane."""
    try:
        from OCP.BRepAdaptor import BRepAdaptor_Curve as _BAC
        from OCP.GeomAbs import GeomAbs_Circle
        adp = _BAC(occ_edge)
        if adp.GetType() != GeomAbs_Circle:
            return False
        ax = adp.Circle().Axis().Direction()
        circ_normal = np.array([ax.X(), ax.Y(), ax.Z()], dtype=np.float64)
        return abs(float(np.dot(circ_normal, sketch_plane_normal))) < tol
    except Exception:
        return False


def _collapse_if_degenerate(uv_pts: list, tol: float = 1e-3) -> list:
    """
    If all UV points are collinear (e.g. a circle projected edge-on), return
    just the two extreme endpoints.  Otherwise return uv_pts unchanged.
    """
    if len(uv_pts) < 3:
        return uv_pts
    pts = np.array(uv_pts, dtype=np.float64)
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    # PCA: largest singular value direction
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    axis = vt[0]
    projections = centered @ axis
    spread_along = projections.max() - projections.min()
    spread_perp  = np.linalg.norm(centered - np.outer(projections, axis), axis=1).max()
    if spread_along < 1e-9:              # a point: an edge seen end-on
        return [uv_pts[0], uv_pts[-1]]
    if spread_perp / spread_along > tol:
        return uv_pts  # genuinely 2-D shape, keep all points
    # Degenerate: return the two extreme points
    i_min = int(np.argmin(projections))
    i_max = int(np.argmax(projections))
    return [uv_pts[i_min], uv_pts[i_max]]


def project_body_edge(world_pts, occ_edge, project, normal,
                      collapse: bool = True) -> tuple:
    """(uv points, occ_edges) for an included body edge.  The Include tool
    and replay both go through here, so a re-projected include keeps the
    shape it was included with: straight edges collapse to their two ends
    (replay passes collapse=False for includes saved uncollapsed — their
    point count decides the sketch's regions), and the occ edge is dropped
    for an edge-on circle.
    With an occ edge the points are sampled from it exactly — float32 mesh
    points sit ~1e-7 off and drawn geometry snapped to them would no
    longer meet the include after replay."""
    if occ_edge is not None:
        from cad.edge_ref import sample_occ_edge
        world_pts = sample_occ_edge(occ_edge)
    uv_pts = [project(np.asarray(p, dtype=np.float64)) for p in world_pts]
    if collapse:
        uv_pts = _collapse_if_degenerate(uv_pts)
    if occ_edge is not None and _is_circle_edge_on(occ_edge, normal):
        occ_edge = None
    return uv_pts, ([occ_edge] if occ_edge is not None else None)


def _body_edge_source(mesh, body_id, occ_edge, history):
    """The parametric source for an edge of *mesh*, or None."""
    from cad.edge_ref import EdgeRef, BodyEdgeSource
    edge_ref = EdgeRef.from_occ_edge(occ_edge)
    if edge_ref is None:
        return None
    origin = None
    if history is not None:
        from cad.face_origin import edge_origin
        try:
            origin = edge_origin(history, body_id, mesh.shape.wrapped, occ_edge,
                                 history.cursor + 1)
        except Exception as ex:
            print(f"[Include] edge provenance skipped: {ex}")
    return BodyEdgeSource(body_id, edge_ref, origin)


def _face_edge_indices(mesh, face_idx: int) -> list:
    """topo_edges indices of a face's boundary, without seams (twice in the
    one face) or a sphere's degenerate poles: neither bounds anything."""
    from OCP.BRep import BRep_Tool
    from OCP.TopAbs import TopAbs_EDGE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS
    edges = []
    exp = TopExp_Explorer(mesh.occt_faces[face_idx].wrapped, TopAbs_EDGE)
    while exp.More():
        edges.append(TopoDS.Edge(exp.Current()))
        exp.Next()
    out = []
    for e in edges:
        if BRep_Tool.Degenerated_s(e) or sum(e.IsSame(o) for o in edges) > 1:
            continue
        out += [k for k, occ in enumerate(mesh.topo_edges_occ) if occ.IsSame(e)]
    return out


def _vertex_edges(mesh, world_pt, tol: float = 1e-3) -> list:
    """Indices of up to two edges of *mesh* ending at world_pt."""
    out = []
    for k, pts in enumerate(mesh.topo_edges[:len(mesh.topo_edges_occ)]):
        if len(pts) >= 2 and min(np.linalg.norm(pts[0] - world_pt),
                                 np.linalg.norm(pts[-1] - world_pt)) < tol:
            out.append(k)
            if len(out) == 2:
                break
    return out


def _covered(sketch, uv_pts, tol: float = 1e-6) -> bool:
    """An edge with no extent on this plane (it runs along the normal), or
    one an include already covers (a wall's two rims project to one line)."""
    from cad.sketch import ReferenceEntity
    pts = np.asarray(uv_pts, dtype=np.float64)
    if float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))) < tol:
        return True
    for ent in sketch.entities:
        if not isinstance(ent, ReferenceEntity) or len(ent.points) != len(pts):
            continue
        other = np.asarray(ent.points, dtype=np.float64)
        if np.abs(pts - other).max() < tol or \
                np.abs(pts - other[::-1]).max() < tol:
            return True
    return False


def _same_circle(sketch, ref, tol: float = 1e-6) -> bool:
    """Whether another include already draws ref's circle.  Two rims of a
    cylinder start their sampling at different seams, so _covered misses it."""
    from cad.sketch import ReferenceEntity
    if ref.circle is None or \
            np.linalg.norm(ref.points[0] - ref.points[-1]) > tol:
        return False                     # an arc: only whole circles coincide
    return any(isinstance(ent, ReferenceEntity) and ent.circle
               and np.linalg.norm(ent.points[0] - ent.points[-1]) < tol
               and abs(ent.circle[1] - ref.circle[1]) < tol
               and np.linalg.norm(ent.circle[0] - ref.circle[0]) < tol
               for ent in sketch.entities)


def _plane_view(plane):
    """plane_origin / _x_axis / ... view of a SketchPlane, the names
    _ref_circle reads off a SketchEntry."""
    from types import SimpleNamespace
    return SimpleNamespace(plane_origin=plane.origin, plane_x_axis=plane.x_axis,
                           plane_y_axis=plane.y_axis, plane_normal=plane.normal)


class IncludeTool:

    @staticmethod
    def apply_with_history(sketch, selection, meshes: dict, history) -> int:
        """
        Project all selected faces, edges, vertices, and committed sketch
        line entities onto sketch.plane as ReferenceEntity instances appended
        to sketch.entities.

        Each included edge is stored with a stable EdgeSource so parametric
        replay can re-derive its world geometry and UV projection.

        Parameters
        ----------
        sketch    : SketchMode
        selection : SelectionSet
        meshes    : dict[body_id, Mesh]
        history   : History | None

        Returns the number of reference entities added.
        """
        from cad.sketch import ReferenceEntity, _ref_circle
        from cad.edge_ref import BodyVertexSource, SketchEdgeSource

        added = 0
        view = _plane_view(sketch.plane)

        def _new(points, **kw):
            ref = ReferenceEntity(points, **kw)
            ref.circle = _ref_circle(view, ref) if len(ref.points) >= 2 else None
            return ref

        def _add(ref):
            nonlocal added
            sketch.entities.append(ref)
            added += 1

        def _add_body_edge(mesh, body_id, edge_idx, whole_face):
            occ_edge = source = None
            if edge_idx < len(mesh.topo_edges_occ):
                occ_edge = mesh.topo_edges_occ[edge_idx]
                source = _body_edge_source(mesh, body_id, occ_edge, history)
            uv_pts, occ_edges = project_body_edge(
                mesh.topo_edges[edge_idx], occ_edge,
                sketch.plane.project_point, sketch.plane.normal)
            if len(uv_pts) < 2 or (whole_face and _covered(sketch, uv_pts)):
                return
            ref = _new(uv_pts, source_type='edge',
                       occ_edges=occ_edges, source=source)
            if not (whole_face and _same_circle(sketch, ref)):
                _add(ref)

        # ------------------------------------------------------------------
        # Selected body faces: each boundary edge, as if picked one by one
        # ------------------------------------------------------------------
        for fs in getattr(selection, 'faces', ()):
            mesh = meshes.get(fs.body_id)
            if mesh is None or fs.face_idx >= len(mesh.occt_faces):
                continue
            for edge_idx in _face_edge_indices(mesh, fs.face_idx):
                _add_body_edge(mesh, fs.body_id, edge_idx, whole_face=True)

        # ------------------------------------------------------------------
        # Selected mesh edges
        # ------------------------------------------------------------------
        for es in selection.edges:
            mesh = meshes.get(es.body_id)
            if mesh is None or es.edge_idx >= len(mesh.topo_edges):
                continue

            _add_body_edge(mesh, es.body_id, es.edge_idx, whole_face=False)

        # ------------------------------------------------------------------
        # Selected sketch edges (committed LineEntity from a history entry)
        # ------------------------------------------------------------------
        for se in selection.sketch_edges:
            if history is None:
                continue
            entries = history.entries
            if se.history_idx < 0 or se.history_idx >= len(entries):
                continue
            entry = entries[se.history_idx]
            sketch_entry = entry.params.get("sketch_entry")
            if sketch_entry is None:
                continue
            if se.entity_idx >= len(sketch_entry.entities):
                continue

            from cad.sketch import LineEntity, ArcEntity

            def _uv_to_world(uv, se):
                return (se.plane_origin
                        + float(uv[0]) * se.plane_x_axis
                        + float(uv[1]) * se.plane_y_axis)

            ent = sketch_entry.entities[se.entity_idx]
            occ_edge = None
            if isinstance(ent, LineEntity):
                p0_world = _uv_to_world(ent.p0, sketch_entry)
                p1_world = _uv_to_world(ent.p1, sketch_entry)
                uv_pts = [sketch.plane.project_point(p0_world),
                          sketch.plane.project_point(p1_world)]
            elif isinstance(ent, ArcEntity):
                import math
                from OCP.gp import gp_Pnt, gp_Dir, gp_Ax2
                from OCP.Geom import Geom_Circle
                from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
                c3d = _uv_to_world(ent.center, sketch_entry)
                ax2 = gp_Ax2(
                    gp_Pnt(*c3d.tolist()),
                    gp_Dir(float(sketch_entry.plane_normal[0]),
                           float(sketch_entry.plane_normal[1]),
                           float(sketch_entry.plane_normal[2])),
                    gp_Dir(float(sketch_entry.plane_x_axis[0]),
                           float(sketch_entry.plane_x_axis[1]),
                           float(sketch_entry.plane_x_axis[2])),
                )
                geom_circ = Geom_Circle(ax2, ent.radius)
                span = ent.end_angle - ent.start_angle
                if abs(span - 2 * math.pi) < 1e-9 or abs(span) < 1e-9:
                    occ_edge = BRepBuilderAPI_MakeEdge(geom_circ).Edge()
                else:
                    occ_edge = BRepBuilderAPI_MakeEdge(
                        geom_circ, ent.start_angle, ent.end_angle).Edge()
                p0_world = _uv_to_world(ent.p0, sketch_entry)
                p1_world = _uv_to_world(ent.p1, sketch_entry)
                uv_pts = [sketch.plane.project_point(p0_world),
                          sketch.plane.project_point(p1_world)]
            else:
                continue

            src_entry_id = history.index_to_id(se.history_idx)
            if src_entry_id is None:
                continue
            source = SketchEdgeSource(src_entry_id, se.entity_idx, ent.uid)
            _add(_new(uv_pts, source_type='sketch_edge',
                      occ_edges=[occ_edge] if occ_edge is not None else None,
                      source=source))

        # ------------------------------------------------------------------
        # Selected vertices
        # ------------------------------------------------------------------
        for vs in selection.vertices:
            mesh = meshes.get(vs.body_id)
            if mesh is None or vs.vertex_idx >= len(mesh.topo_verts):
                continue
            world_pt = mesh.topo_verts[vs.vertex_idx]
            uv_pt    = sketch.plane.project_point(world_pt)
            sources = [_body_edge_source(mesh, vs.body_id,
                                         mesh.topo_edges_occ[k], history)
                       for k in _vertex_edges(mesh, world_pt)]
            sources = [s for s in sources if s is not None]
            _add(_new([uv_pt], source_type='vertex',
                      source=BodyVertexSource(sources, world_pt) if sources else None))

        return added
