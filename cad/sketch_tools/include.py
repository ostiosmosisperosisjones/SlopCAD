"""
cad/sketch_tools/include.py

IncludeTool — project selected edges / vertices onto the sketch plane
as ReferenceEntity instances.

Every included edge now carries a parametric EdgeSource so that the
replay system can re-project the geometry when upstream operations change.

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
    if spread_along < 1e-9:
        return uv_pts
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


class IncludeTool:

    @staticmethod
    def apply_with_history(sketch, selection, meshes: dict, history) -> int:
        """
        Project all selected edges, vertices, and committed sketch line
        entities onto sketch.plane as ReferenceEntity instances appended
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
        from cad.sketch import ReferenceEntity
        from cad.edge_ref import EdgeRef, BodyEdgeSource, SketchEdgeSource

        added = 0

        # ------------------------------------------------------------------
        # Selected mesh edges
        # ------------------------------------------------------------------
        for es in selection.edges:
            mesh = meshes.get(es.body_id)
            if mesh is None or es.edge_idx >= len(mesh.topo_edges):
                continue

            occ_edge = None
            source   = None
            if es.edge_idx < len(mesh.topo_edges_occ):
                occ_edge = mesh.topo_edges_occ[es.edge_idx]
                edge_ref = EdgeRef.from_occ_edge(occ_edge)
                if edge_ref is not None:
                    origin = None
                    if history is not None:
                        from cad.face_origin import edge_origin
                        try:
                            origin = edge_origin(
                                history, es.body_id, mesh.shape.wrapped, occ_edge,
                                history.cursor + 1)
                        except Exception as ex:
                            print(f"[Include] edge provenance skipped: {ex}")
                    source = BodyEdgeSource(es.body_id, edge_ref, origin)

            uv_pts, occ_edges = project_body_edge(
                mesh.topo_edges[es.edge_idx], occ_edge,
                sketch.plane.project_point, sketch.plane.normal)
            if len(uv_pts) < 2:
                continue

            sketch.entities.append(
                ReferenceEntity(
                    uv_pts,
                    source_type = 'edge',
                    occ_edges   = occ_edges,
                    source      = source,
                )
            )
            added += 1

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
            source = SketchEdgeSource(src_entry_id, se.entity_idx)
            sketch.entities.append(
                ReferenceEntity(uv_pts, source_type='sketch_edge',
                                occ_edges=[occ_edge] if occ_edge is not None else None,
                                source=source))
            added += 1

        # ------------------------------------------------------------------
        # Selected vertices — point references, no parametric source yet
        # ------------------------------------------------------------------
        for vs in selection.vertices:
            mesh = meshes.get(vs.body_id)
            if mesh is None or vs.vertex_idx >= len(mesh.topo_verts):
                continue
            world_pt = mesh.topo_verts[vs.vertex_idx]
            uv_pt    = sketch.plane.project_point(world_pt)
            sketch.entities.append(
                ReferenceEntity([uv_pt], source_type='vertex'))
            added += 1

        return added
