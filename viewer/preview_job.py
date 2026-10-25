"""
viewer/preview_job.py

PreviewJob — the one background-compute pattern every op preview needs:

  - at most ONE worker in flight per job, for the job's whole lifetime
    (keep one job per op on the viewport, don't recreate it per panel open,
    or a reopened panel can start a second worker beside a stale one —
    concurrent OCCT on the same shape)
  - latest request wins: submits while busy replace the pending one
  - cancel() drops the pending request and discards the in-flight result
  - the result is delivered on the Qt main thread
  - on_idle (optional) fires whenever nothing is in flight or pending,
    including after a discarded result — for busy indicators

Usage:
    job = PreviewJob(on_done)          # on_done(result, error) — main thread
    job.submit(lambda: heavy(shape))   # runs in a worker thread
    job.cancel()                       # panel closed

Also `preview_mesh_data()`: triangles + edge polylines of a shape for the
immediate-mode previews (revolve, thicken).
"""

from __future__ import annotations

import threading
from typing import Callable

from PyQt6.QtCore import QObject, pyqtSignal

from cad.kernel_pool import offload


class PreviewJob(QObject):
    _finished = pyqtSignal(int, object, object)     # gen, result, error

    def __init__(self, on_done: Callable, name: str = "preview",
                 on_idle: Callable | None = None):
        super().__init__()
        self._on_done  = on_done
        self._on_idle  = on_idle
        self._name     = name
        self._gen      = 0
        self._inflight = False
        self._pending: Callable | None = None
        # Emitted from the worker; this QObject lives on the main thread, so
        # the default (auto) connection queues the slot onto it.
        self._finished.connect(self._deliver)

    @property
    def busy(self) -> bool:
        return self._inflight

    def submit(self, compute: Callable) -> None:
        self._gen += 1
        self._pending = compute
        if not self._inflight:
            self._start()

    def cancel(self) -> None:
        self._gen += 1
        self._pending = None

    def _start(self) -> None:
        compute, self._pending = self._pending, None
        self._inflight = True
        threading.Thread(target=self._run, args=(self._gen, compute),
                         daemon=True).start()

    def _run(self, gen: int, compute: Callable) -> None:
        try:
            result, error = compute(), None
        except Exception as ex:
            result, error = None, ex
            print(f"[{self._name}] {ex}")
        self._finished.emit(gen, result, error)

    def _deliver(self, gen: int, result, error) -> None:
        self._inflight = False
        if gen == self._gen:
            self._on_done(result, error)
        if self._pending is not None:
            self._start()
        elif self._on_idle is not None:
            self._on_idle()


def job_for(owner, attr: str, on_done: Callable, name: str,
            on_idle: Callable | None = None) -> PreviewJob:
    """The PreviewJob stored on owner.attr, created on first use.  One per
    op for the viewport's lifetime (see module docstring)."""
    job = getattr(owner, attr, None)
    if job is None:
        job = PreviewJob(on_done, name, on_idle)
        setattr(owner, attr, job)
    return job


# The revolve / thicken / loft preview builds run whole in a kernel worker: OCCT
# plus the triangle loops below hold the GIL, and on a PreviewJob thread that
# starves the UI thread (GIL convoy).

@offload()
def revolve_preview_meshes(faces, axis_pt, axis_d, angle):
    """Packed preview (see pack_preview) of every face revolved."""
    from cad.operations.revolve import _do_revolve_solid
    meshes = [preview_mesh_data(
                  _do_revolve_solid(f, axis_pt, axis_d, angle).wrapped, 1.0)
              for f in faces]
    return pack_preview([m[0] for m in meshes], [m[1] for m in meshes])


@offload()
def thicken_preview_meshes(face_occs, thickness: float):
    """(packed preview, [error str | None]) — one error slot per face."""
    from cad.operations.thicken import thicken_face_preview
    tris_list, edges_list, errors = [], [], []
    for fo in face_occs:
        try:
            tris, edges = preview_mesh_data(
                thicken_face_preview(fo, thickness).wrapped, 0.15)
            tris_list.append(tris)
            edges_list.append(edges)
            errors.append(None)
        except Exception as ex:
            errors.append(str(ex))
    return pack_preview(tris_list, edges_list), errors


@offload()
def loft_preview_meshes(faces, ruled: bool, continuity: str):
    """Packed preview of the loft through `faces`."""
    from cad.op_loft import build_loft_solid
    tris, edges = preview_mesh_data(
        build_loft_solid(faces, ruled, continuity).wrapped, 0.15, 24)
    return pack_preview([tris], [edges])


def pack_preview(tris_list, edges_list):
    """(tri verts, line-segment verts) as float32 Nx3 arrays, ready for
    draw_preview; None when there is nothing to draw."""
    import numpy as np
    tris = [t for t in tris_list if t]
    if not tris:
        return None
    tri_arr = np.ascontiguousarray(np.concatenate(
        [np.asarray(t, dtype=np.float32) for t in tris]))
    segs = []
    for edges in edges_list:
        for pts in edges:
            a = np.asarray(pts, dtype=np.float32)
            if len(a) >= 2:
                segs.append(np.stack([a[:-1], a[1:]], axis=1).reshape(-1, 3))
    line_arr = (np.ascontiguousarray(np.concatenate(segs)) if segs
                else np.zeros((0, 3), np.float32))
    return tri_arr, line_arr


def draw_preview(packed, fill_color, edge_color):
    """Draw a pack_preview result (blend/depth state is the caller's)."""
    from OpenGL.GL import (glColor4f, glLineWidth, glEnableClientState,
                           glDisableClientState, glVertexPointer, glDrawArrays,
                           GL_VERTEX_ARRAY, GL_FLOAT, GL_TRIANGLES, GL_LINES)
    tri_arr, line_arr = packed
    glEnableClientState(GL_VERTEX_ARRAY)
    try:
        glColor4f(*fill_color)
        glVertexPointer(3, GL_FLOAT, 0, tri_arr)
        glDrawArrays(GL_TRIANGLES, 0, len(tri_arr))
        if len(line_arr):
            glColor4f(*edge_color)
            glLineWidth(1.4)
            glVertexPointer(3, GL_FLOAT, 0, line_arr)
            glDrawArrays(GL_LINES, 0, len(line_arr))
            glLineWidth(1.0)
    finally:
        glDisableClientState(GL_VERTEX_ARRAY)


def preview_mesh_data(wrapped, deflection: float, edge_samples: int = 16):
    """(tris, edges) for a TopoDS_Shape: tris is a flat list of xyz triples
    (3 per triangle), edges a list of sampled polylines."""
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.BRep import BRep_Tool
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_FACE, TopAbs_EDGE
    from OCP.TopoDS import TopoDS
    from OCP.TopLoc import TopLoc_Location
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.GCPnts import GCPnts_UniformAbscissa

    BRepMesh_IncrementalMesh(wrapped, deflection)
    identity = TopLoc_Location()
    tris = []
    exp = TopExp_Explorer(wrapped, TopAbs_FACE)
    while exp.More():
        tri = BRep_Tool.Triangulation_s(TopoDS.Face(exp.Current()), identity)
        if tri is not None:
            for j in range(1, tri.NbTriangles() + 1):
                for ni in tri.Triangle(j).Get():
                    p = tri.Node(ni)
                    tris.append((p.X(), p.Y(), p.Z()))
        exp.Next()

    edges = []
    exp = TopExp_Explorer(wrapped, TopAbs_EDGE)
    while exp.More():
        try:
            adp  = BRepAdaptor_Curve(TopoDS.Edge(exp.Current()))
            disc = GCPnts_UniformAbscissa()
            disc.Initialize(adp, edge_samples)
            if disc.IsDone() and disc.NbPoints() >= 2:
                pts = []
                for pi in range(1, disc.NbPoints() + 1):
                    p = adp.Value(disc.Parameter(pi))
                    pts.append((p.X(), p.Y(), p.Z()))
                edges.append(pts)
        except Exception:
            pass
        exp.Next()
    return tris, edges
